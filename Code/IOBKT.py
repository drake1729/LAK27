import os
N_CHAINS = min(2, os.cpu_count() or 2)
print(f"CPU cores detected: {os.cpu_count()}. Using N_CHAINS={N_CHAINS}")

import numpyro
numpyro.set_host_device_count(N_CHAINS)   # MUST precede any jax use

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import pickle

print("JAX devices:", jax.devices())

# ── Configuration ─────────────────────────────────────────────
FILE_PATH  = r"C:\Users\vivek\jupyter_dellvivek\analysis_final\chisquare_ready_data_withvignette_withepisode_LT21CSonly_new_wtGE_FILLED.xlsx"
OUTPUT_DIR = r"C:\Users\vivek\JupyterProjects\bayesian"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPISODE_COL = "Episode"
SKILL_COLS  = ["Problem Solving", "Creativity",
               "Collaborative Learning", "Critical Thinking"]
EP_COLS = ["Behaviourism", "Cognitivism", "Constructivism"]
KD_COLS = ["Factual", "Conceptual", "Procedural", "Metacognitive"]
CP_COLS = ["Remembering", "Understanding", "Applying",
           "Analyzing", "Evaluating", "Creating"]
feat_cols = EP_COLS + KD_COLS + CP_COLS
MAX_TMAX = 921
EPS = 1e-7

EPISODE_SUBJECT = {
    1:'Science', 2:'Science', 3:'Science', 4:'Science', 5:'Science',
    6:'Science', 7:'Science', 8:'Science', 9:'Science', 10:'Science',
    11:'Science', 12:'Science', 13:'Science', 14:'Science', 15:'Science',
    16:'Science', 17:'Science', 18:'Science', 19:'Science', 20:'Science',
    21:'Science', 22:'Science', 23:'Science', 24:'Science', 25:'Science',
    26:'Science', 27:'Science', 28:'Science', 29:'Science', 30:'Science',
    49:'Science',
    31:'Math', 32:'Math', 33:'Math', 34:'Math', 35:'Math',
    36:'Math', 37:'Math', 38:'Math', 39:'Math', 40:'Math',
    41:'Math', 42:'Math', 43:'Math', 44:'Math', 45:'Math',
    46:'Math', 47:'Math', 48:'Math',
}
EPISODE_GRADE = {
    1:6, 2:6, 3:6, 4:7, 5:7, 6:7, 7:9, 8:9, 9:9, 10:10,
    11:10, 12:10, 13:10, 14:11, 15:11, 16:11, 17:11, 18:11,
    19:11, 20:11, 21:11, 22:11, 23:12, 24:12, 25:12, 26:12,
    27:12, 28:12, 29:12, 30:12, 31:6, 32:6, 33:6, 34:7, 35:7,
    36:7, 37:8, 38:8, 39:8, 40:9, 41:9, 42:9, 43:10, 44:10,
    45:11, 46:11, 47:11, 48:12, 49:7,
}

# ── Load data ────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_excel(FILE_PATH)
df = df[df[EPISODE_COL] != 49].copy()          # E=48 corpus
df["any_skill"] = (df[SKILL_COLS].sum(axis=1) > 0).astype(int)
df = df.sort_values(EPISODE_COL).reset_index(drop=True)

episodes = sorted(df[EPISODE_COL].unique())
E = len(episodes)
lengths = df.groupby(EPISODE_COL).size()
Tmax = min(int(lengths.max()), MAX_TMAX)
assert E == 48, f"Expected E=48, got E={E}"
print(f"  E={E}, Tmax={Tmax}")

# cells: {(subject, grade): index} -- SAVED below so downstream
# scripts reuse this exact mapping instead of rebuilding their own.
cells    = {}
cell_idx = np.zeros(E, dtype=np.int32)
for i, e in enumerate(episodes):
    key = (EPISODE_SUBJECT.get(e, "NA"), EPISODE_GRADE.get(e, -1))
    if key not in cells:
        cells[key] = len(cells)
    cell_idx[i] = cells[key]
n_cells = len(cells)
print(f"  n_cells={n_cells}: {cells}")

obs  = np.zeros((E, Tmax), dtype=np.float32)
mask = np.zeros((E, Tmax), dtype=np.float32)
for i, e in enumerate(episodes):
    d = df[df[EPISODE_COL] == e].iloc[:Tmax]
    T = len(d)
    obs[i, :T]  = d["any_skill"].values.astype(np.float32)
    mask[i, :T] = 1.0

obs_jax      = jnp.array(obs)
mask_jax     = jnp.array(mask)
cell_idx_jax = jnp.array(cell_idx)

def build_Z_full():
    """Full 13-feature + intercept design matrix."""
    P = len(feat_cols) + 1
    Z = np.zeros((E, Tmax, P), dtype=np.float32)
    for i, e in enumerate(episodes):
        d = df[df[EPISODE_COL] == e].iloc[:Tmax]
        T = len(d)
        Z[i, :T, :len(feat_cols)] = d[feat_cols].values.astype(np.float32)
        Z[i, :T, -1] = 1.0
    return jnp.array(Z), P

Z_full, P = build_Z_full()

# ── Forward algorithm ───────────────────────────────────────
def forward_single(obs_e, mask_e, pon_e, poff_e, pi0, pS, pG):
    o0   = obs_e[0]
    e0_0 = jnp.where(o0 == 1., pG,      1. - pG)
    e1_0 = jnp.where(o0 == 1., 1. - pS, pS)
    a0   = (1. - pi0) * e0_0
    a1   = pi0        * e1_0
    c0   = a0 + a1 + EPS
    a0   = a0 / c0
    a1   = a1 / c0
    ll0  = jnp.log(c0) * mask_e[0]

    def step(carry, t):
        a0, a1, ll = carry
        o_t    = obs_e[t]
        m_t    = mask_e[t]
        pon_t  = pon_e[t]
        poff_t = poff_e[t]
        pred0  = a0 * (1. - pon_t) + a1 * poff_t
        pred1  = a0 * pon_t        + a1 * (1. - poff_t)
        e0     = jnp.where(o_t == 1., pG,      1. - pG)
        e1     = jnp.where(o_t == 1., 1. - pS, pS)
        b0     = pred0 * e0
        b1     = pred1 * e1
        c      = b0 + b1 + EPS
        return (b0/c, b1/c, ll + jnp.log(c) * m_t), None

    (_, _, ll_final), _ = lax.scan(
        step, (a0, a1, ll0), jnp.arange(1, Tmax)
    )
    return ll_final

forward_all = vmap(forward_single, in_axes=(0, 0, 0, 0, None, None, None))

def iobkt_model(Z, obs, mask, cell_idx, n_cells, Tmax):
    b_on  = numpyro.sample("b_on",  dist.Normal(jnp.zeros(P), jnp.ones(P)))
    b_off = numpyro.sample("b_off", dist.Normal(jnp.zeros(P), jnp.ones(P)))
    sigma_cell = numpyro.sample("sigma_cell", dist.HalfNormal(1.0))
    u_cell = numpyro.sample("u_cell",
                 dist.Normal(jnp.zeros(n_cells),
                             sigma_cell * jnp.ones(n_cells)))
    pi0 = numpyro.sample("pi0", dist.Beta(2.0,  3.0))
    pS  = numpyro.sample("pS",  dist.Beta(2.0, 30.0))
    pG  = numpyro.sample("pG",  dist.Beta(1.5, 30.0))

    u_e     = u_cell[cell_idx]
    eta_on  = jnp.einsum("etp,p->et", Z, b_on)  + u_e[:, None]
    eta_off = jnp.einsum("etp,p->et", Z, b_off) + u_e[:, None]
    p_on    = jax.nn.sigmoid(eta_on)
    p_off   = jax.nn.sigmoid(eta_off)

    ll_per_episode = forward_all(obs, mask, p_on, p_off, pi0, pS, pG)
    numpyro.factor("seq_loglik", jnp.sum(ll_per_episode))


# ── SWAP-based label-switching correction (verified correct) ────
def correct_label_switching_perchain(idata):
    """
    Checks pS PER CHAIN and applies the SWAP correction
    (b_on <-> b_off) independently to each chain that needs it.

    Verified via log-likelihood invariance test: swap reproduces
    a normal chain's log-likelihood to within Monte Carlo noise
    (-9.4 diff); negation does not (-16,156 diff).

    u_cell / sigma_cell are NOT swapped: u_e is added identically
    to both eta_on and eta_off, so swapping which equation is
    "on" vs. "off" doesn't change what u_cell means -- it stays
    a shared per-cell offset regardless of labeling.
    """
    post = idata.posterior
    n_chains = post["b_on"].shape[0]

    b_on_raw  = np.array(post["b_on"])    # (chain, draw, P)
    b_off_raw = np.array(post["b_off"])
    pS_raw    = np.array(post["pS"])      # (chain, draw)
    pG_raw    = np.array(post["pG"])
    pi0_raw   = np.array(post["pi0"])

    pS_per_chain = pS_raw.mean(axis=1)
    flip_mask = pS_per_chain > 0.5

    b_on_fixed  = b_on_raw.copy()
    b_off_fixed = b_off_raw.copy()
    pS_fixed    = pS_raw.copy()
    pG_fixed    = pG_raw.copy()
    pi0_fixed   = pi0_raw.copy()

    for c in range(n_chains):
        if flip_mask[c]:
            b_on_fixed[c]  = b_off_raw[c].copy()
            b_off_fixed[c] = b_on_raw[c].copy()
            pS_fixed[c]    = 1.0 - pG_raw[c]
            pG_fixed[c]    = 1.0 - pS_raw[c]
            pi0_fixed[c]   = 1.0 - pi0_raw[c]

    corrected_posterior = {
        "b_on": b_on_fixed, "b_off": b_off_fixed,
        "pS": pS_fixed, "pG": pG_fixed, "pi0": pi0_fixed,
    }
    for extra_key in ["sigma_cell", "u_cell"]:
        if extra_key in post.data_vars:
            corrected_posterior[extra_key] = np.array(post[extra_key])

    idata_corrected = az.from_dict(posterior=corrected_posterior)

    return {
        "b_on_flat":  b_on_fixed.reshape(-1, b_on_fixed.shape[-1]),
        "b_off_flat": b_off_fixed.reshape(-1, b_off_fixed.shape[-1]),
        "pS_flat":    pS_fixed.flatten(),
        "pG_flat":    pG_fixed.flatten(),
        "pi0_flat":   pi0_fixed.flatten(),
        "per_chain_pS": pS_per_chain,
        "flip_mask": flip_mask,
        "idata_corrected": idata_corrected,
    }


def run_with_diagnostics(name, Z, seed=42):
    print(f"\n{'='*70}")
    print(f"RUNNING: {name}  (P={P}, N_CHAINS={N_CHAINS})")
    print(f"{'='*70}")

    kernel = NUTS(iobkt_model, target_accept_prob=0.95, max_tree_depth=10)
    mcmc = MCMC(kernel, num_warmup=1500, num_samples=2000,
                num_chains=N_CHAINS, chain_method='sequential',
                progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed),
             Z, obs_jax, mask_jax, cell_idx_jax, n_cells, Tmax)

    idata = az.from_numpyro(mcmc)

    result = correct_label_switching_perchain(idata)
    print(f"\n  Per-chain pS means: {result['per_chain_pS']}")
    print(f"  Chains flipped: {result['flip_mask']}")

    rhat = az.rhat(result["idata_corrected"])
    ess  = az.ess(result["idata_corrected"])
    b_on_rhat_max  = float(np.array(rhat["b_on"]).max())
    b_off_rhat_max = float(np.array(rhat["b_off"]).max())
    b_on_ess_min   = float(np.array(ess["b_on"]).min())
    b_off_ess_min  = float(np.array(ess["b_off"]).min())

    print(f"\n  r-hat (max): b_on={b_on_rhat_max:.4f}, b_off={b_off_rhat_max:.4f}")
    print(f"  ESS (min):   b_on={b_on_ess_min:.0f}, b_off={b_off_ess_min:.0f}")

    converged = (b_on_rhat_max < 1.01) and (b_off_rhat_max < 1.01)
    print(f"  CONVERGED (<1.01): {'YES' if converged else 'NO -- DO NOT TRUST THESE NUMBERS'}")

    b_on_mean  = result["b_on_flat"].mean(axis=0)
    b_on_sd    = result["b_on_flat"].std(axis=0)
    b_off_mean = result["b_off_flat"].mean(axis=0)
    b_off_sd   = result["b_off_flat"].std(axis=0)

    # ── u_cell / sigma_cell: flatten (chain, draw, n_cells) -> (draws, n_cells) ──
    u_cell_flat = np.array(
        result["idata_corrected"].posterior["u_cell"]
    ).reshape(-1, n_cells)
    sigma_cell_flat = np.array(
        result["idata_corrected"].posterior["sigma_cell"]
    ).reshape(-1)

    rhat_u_cell = float(np.array(rhat["u_cell"]).max()) if "u_cell" in rhat else float("nan")
    print(f"  r-hat (max): u_cell={rhat_u_cell:.4f}")

    out_path = os.path.join(OUTPUT_DIR, f"{name}_2chain_corrected.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "b_on": result["b_on_flat"], "b_off": result["b_off_flat"],
            "pS": result["pS_flat"], "pG": result["pG_flat"],
            "pi0": result["pi0_flat"],
            "u_cell": u_cell_flat,            # (n_draws, n_cells)
            "sigma_cell": sigma_cell_flat,    # (n_draws,)
            "cells": cells,                   # {(subject, grade): index}
                                               # -- reuse THIS mapping downstream,
                                               # don't rebuild it independently
            "feat_cols": feat_cols,           # so downstream scripts don't
                                               # have to hardcode column order
            "rhat_b_on": b_on_rhat_max, "rhat_b_off": b_off_rhat_max,
            "rhat_u_cell": rhat_u_cell,
            "converged": converged,
        }, f)
    print(f"  Saved -> {out_path}")

    return {
        "name": name, "b_on_mean": b_on_mean, "b_on_sd": b_on_sd,
        "b_off_mean": b_off_mean, "b_off_sd": b_off_sd,
        "rhat_b_on": b_on_rhat_max, "rhat_b_off": b_off_rhat_max,
        "converged": converged,
    }


# ============================================================
# RUN: multivariate baseline only, E=48
# ============================================================
result_baseline = run_with_diagnostics("baseline_multivariate", Z_full)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Converged (<1.01 r-hat on b_on AND b_off): {result_baseline['converged']}")
for nm, m_on, m_off in zip(feat_cols, result_baseline["b_on_mean"],
                            result_baseline["b_off_mean"]):
    print(f"  {nm:<22} b_on={m_on:+.3f}  b_off={m_off:+.3f}")
print(f"  {'Intercept':<22} b_on={result_baseline['b_on_mean'][-1]:+.3f}  "
      f"b_off={result_baseline['b_off_mean'][-1]:+.3f}")

if not result_baseline["converged"]:
    print("\n⚠ NOT CONVERGED — do not report these coefficients or feed this "
          "trace into knowledge_change_curves.py until r-hat < 1.01.")

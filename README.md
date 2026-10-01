# What Moves the Classroom? A Bayesian Latent-State Model of Pedagogy and 21st-Century Skills in Whole-Classroom Discourse

This repository accompanies the paper of the same title. It contains the code, extended results, documentation images and expert-validation materials needed to inspect and reproduce the analyses reported therein.

> [!IMPORTANT]
> **Data release and ethics statement**
>
> -**Data availability:** If the paper is published, all data accompanying this work will be made publicly available.
> - **Ethics:** The ethical norms of the institute have been followed in the collection and use of this data.

## Overview

Bayesian Knowledge Tracing (BKT) models the evolving mastery of an individual learner. The present work extends this line of research to whole-classroom discourse, where a teacher's pedagogical choices shape the learning conditions of all students simultaneously. We treat classroom talk as evidence of a latent, time-varying productive-discourse regime, defined as a state in which 21st-century skills (21CS) are active at the classroom level, and we refer to this approach as discourse-level pedagogical state tracking. The framework comprises three models that span a tractability-expressiveness spectrum.

| Model | Specification | Research question |
|---|---|---|
| BLFM (Bayesian Latent Factor Model) | Exchangeable; episode-level feature means predict the episode 21CS rate | RQ1a |
| PST (Probabilistic Skill Tracing) | Two-state hidden Markov model (HMM) with fixed transitions and no content features | RQ1b |
| IO-BKT (Input-Output BKT) | HMM whose ignition and dropout transitions depend on pedagogical covariates | RQ1c |

The repository additionally provides the baselines (frequentist BKT and DKT-LSTM), the coefficient-reliability pipeline, the teacher-to-student lag analysis, and the materials for the expert validation study with ten practising teachers (RQ2).

## Repository Structure

The layout below is the intended organisation of the repository; file names should be adjusted to match the committed contents.

```
LAK27/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md                  data dictionary and availability statement
│   ├── annotated_lines.csv        line-level annotations
│   └── episode_metadata.csv       episode identifier, subject, grade
├── src/
│   ├── preprocess.py              construction of z_t, O_t, and Subject x Grade cells
│   ├── blfm.py                    BLFM
│   ├── pst.py                     PST
│   ├── iobkt.py                   IO-BKT
│   ├── baselines/
│   │   ├── bkt_pybkt.py           frequentist BKT (aggregate)
│   │   └── dkt_lstm.py            DKT-LSTM, leave-one-episode-out CV
│   ├── reliability.py             coefficient-reliability pipeline
│   ├── metrics.py                 one-step-ahead accuracy, AUC, F1
│   └── lag_analysis.py            teacher-to-student lag-1 analysis
├── scripts/
│   ├── run_all.sh                 end-to-end reproduction
│   ├── make_tables.py
│   ├── generate_corrected_coefficient_table.py
│   └── make_figures.py
├── results/
│   ├── posteriors/
│   ├── tables/
│   └── figures/
├── expert_validation/
│   ├── clips/                     de-identified transcripts of the three clips
│   ├── cards/                     true and foil recommendation notes
│   ├── case_mapping.csv           the eight recommendation types
│   ├── instruments/               questionnaires and rating items
│   ├── responses/                 de-identified teacher responses
│   ├── coding/                    blinded match-coding rubric and scores
│   └── analysis_rq2.py
├── paper/
│   └── main.tex
└── LICENSE
```

## Installation

```bash
git clone https://github.com/drake1729/LAK27.git
cd LAK27
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The principal dependencies are NumPyro and JAX (a GPU build is recommended for PST and IO-BKT), ArviZ, NumPy, pandas, scikit-learn, statsmodels, SciPy, pyBKT, PyTorch, and Matplotlib. In our experiments, BLFM ran in under five minutes on a CPU, PST required approximately fifteen minutes on a T4 GPU, and IO-BKT was compiled to GPU through `jax.lax.scan`.

## Data

The corpus consists of 48 classroom episodes (Grades 6 to 12; Science and Mathematics; approximately 23 hours) comprising 19,141 transcribed lines, each annotated by trained raters. Each row of the annotation file corresponds to one utterance and contains the fields listed below.

| Field | Description |
|---|---|
| `episode_id`, `line_idx` | Episode identifier and position of the line within the episode |
| `speaker` | Teacher (Speaker1), student (Speaker2), or joint |
| `subject`, `grade` | Determine the Subject x Grade cell (13 cells) used for the random effect |
| `EP_Beh`, `EP_Cog`, `EP_Con` | Educational practice: behaviourism, cognitivism, constructivism (binary) |
| `KD_Fact`, `KD_Conc`, `KD_Proc`, `KD_Meta` | Knowledge dimension: factual, conceptual, procedural, metacognitive (binary) |
| `CP_Rem`, `CP_Und`, `CP_App`, `CP_Ana`, `CP_Eva`, `CP_Cre` | Cognitive process: remembering through creating (binary) |
| `O` | Outcome; equals 1 if problem solving, creativity, collaborative learning, or critical thinking is active on the line |

The covariate vector is z_t = [EP || KD || CP || 1], a 14-dimensional binary vector that includes the intercept.

**Data availability.** If the paper is published, all data will be made public. The ethical norms of the institute have been followed.

## Usage

### Reproducing all results

```bash
bash scripts/run_all.sh
```

### Fitting the models

All models are fitted with the No-U-Turn Sampler in NumPyro (500 warmup iterations and 1,000 draws for each of two chains; target acceptance 0.90).

```bash
python src/preprocess.py --input data/annotated_lines.csv --out data/processed/
python src/blfm.py  --data data/processed/ --out results/posteriors/blfm
python src/iobkt.py --data data/processed/ --out results/posteriors/iobkt
python src/pst.py   --data data/processed/ --out results/posteriors/pst
```

IO-BKT is fitted on full-length sequences (T_max = 921) without truncation. In PST, the slip and guess probabilities are fixed at the IO-BKT posterior means (p_S = 0.015, p_G = 0.009) to avoid label switching, so IO-BKT should be run first if these values are to be re-derived rather than taken from the paper.

### Baselines and model comparison

```bash
python src/baselines/bkt_pybkt.py
python src/baselines/dkt_lstm.py --hidden 64 --dropout 0.2 --epochs 30
python scripts/make_tables.py --table comparison
```

Following Scarlatos et al. (2025), predictions are one-step-ahead, P(O_t = 1 | O_{1:t-1}), and are evaluated at the line level with the first line of each episode excluded. It should be noted that IO-BKT and BLFM are evaluated in-sample (posterior predictive), whereas DKT-LSTM is evaluated by leave-one-episode-out cross-validation (48 folds); the comparison is therefore not strictly symmetric.

### Coefficient-reliability pipeline

```bash
python src/reliability.py --posterior results/posteriors/iobkt
python scripts/generate_corrected_coefficient_table.py
```

A coefficient is admitted to a teacher-facing recommendation only if it satisfies four criteria: (1) correction of HMM label switching at the chain level, in which relabelling swaps, rather than negates, the vectors β_on and β_off; (2) convergence, defined as R-hat below 1.01 and bulk effective sample size above 400; (3) a 95% posterior credible interval that excludes zero; and (4) stability under collinearity and prior-sensitivity checks, using variance inflation factors, agreement in sign between multivariate and univariate fits, and a refit under a weakly informative Student-t prior (ν = 7, scale 2.5). The script produces the coefficient table reported in the paper.

### Lag analysis and figures

```bash
python src/lag_analysis.py
python scripts/make_figures.py
```

The lag analysis pairs each student turn with the immediately preceding teacher move (n = 4,120 pairs) and estimates the probability of student 21CS conditional on that move. Figures are written to `results/figures/`.

## Expert validation (RQ2)

The `expert_validation/` directory contains the materials for the study with ten secondary Science and Mathematics teachers. Three de-identified clips of four to five minutes were selected using the filtered probability of the productive state under IO-BKT. For each clip, three true notes were derived from coefficients that passed the reliability pipeline and were grounded in a verbatim line of the segment. Each was paired with a foil note that used the same segment and vocabulary but stated the opposite direction of a credible coefficient. `case_mapping.csv` records the eight recommendation types, which follow from whether a feature is present or absent and from the role of its coefficient.

Each teacher completed four steps per clip: independent advice, forced choice among the true note, the foil, and a "neither" option, a reveal, and ratings of understandability and likelihood of use on five-point scales. A second coder, blind to card identity, scored each teacher's independent advice against both notes on a three-point rubric (2 = same feature or clear synonym; 1 = same general direction; 0 = unrelated or contradictory). The analysis is reproduced with

```bash
python expert_validation/analysis_rq2.py
```

The teacher is the unit of analysis (N = 10), with each teacher's nine trials aggregated to avoid pseudoreplication. The script reports one-sided Wilcoxon signed-rank tests, an exact sign test, a permutation test, matched-pairs rank-biserial correlations, and Holm-adjusted p-values.

## Summary of Findings

PST indicates a largely stable regime with infrequent switching (p_L = 0.073, p_F = 0.021). In line-level prediction, IO-BKT (AUC = 0.873) outperformed BLFM (0.772) and DKT-LSTM (0.672). Applying was the strongest ignition driver (β_on = +2.34) and the strongest protective factor against dropout (β_off = −3.42), whereas Remembering suppressed ignition (β_on = −2.06). Constructivist teacher moves were followed by student 21CS in 93.5% of immediate next turns (n = 46). In the expert study, teachers selected the true note in 74.4% of forced-choice trials, and ratings of understandability and likelihood of use exceeded the scale midpoint.

## Limitations

The data originate from a single school over three months and are dominated by behaviourist and cognitivist instruction, which limits generalisability. The observational design supports associative, not causal, interpretation, and the coefficients describe association with 21CS activation rather than the general quality of a teaching move. Analysing, Evaluating, and Creating occur rarely in the corpus, so their coefficients remain close to the prior; this reflects corpus coverage and should not be read as evidence of ineffectiveness. Finally, the expert study (ten teachers, three clips) supports alignment, validity, and actionability, but not instructional or student-outcome effects.

## Citation

```bibtex
@inproceedings{smith2027moves,
  title     = {What Moves the Classroom? A Bayesian Latent-State Model of Pedagogy and 21st-Century Skills in Whole-Classroom Discourse},
  author    = {Smith, John and Smith, James},
  booktitle = {Proceedings of the International Conference on Learning Analytics \& Knowledge (LAK27)},
  year      = {2027}
}
```

## License and Contact

This repository is distributed under the license given in `LICENSE`. Correspondence should be addressed to the corresponding author (somebody@gmail.com) or raised as a GitHub issue.

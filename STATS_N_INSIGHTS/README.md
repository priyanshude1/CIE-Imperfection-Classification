# Stats and Insights

This directory contains diagnostic reports and statistical summaries generated during the preprocessing and model selection phases of the CIE Project pipeline. These files are reference artifacts — they are not consumed by training or inference, but provide transparency into data characteristics and preprocessing decisions.

---

## Contents

| File | Description |
|---|---|
| `EquivalentStress_scale_used_by_regime.csv` | Records the equivalent stress scaling factor applied per loading regime during event construction |
| `candidate_counts_tauRel5_tauAbs0.csv` | Event candidate counts with relative threshold τ=5%, absolute threshold=0 |
| `candidate_counts_tauRel10_tauAbs0.csv` | Event candidate counts with relative threshold τ=10%, absolute threshold=0 |
| `candidate_counts_tauRel10_tauAbs10.csv` | Event candidate counts with both relative (τ=10%) and absolute (τ=10) thresholds |
| `candidate_counts_equiv_only_tauRel5_tauAbs0.csv` | Candidate counts using equivalent stress only (no displacement), τ=5% |
| `column_rankings.csv` | Feature importance / ranking of input columns based on variance or correlation analysis |
| `scaler_build_report.csv` | Summary of the scaler fitting process — per-feature statistics (mean, std, min, max) |

---

## How These Were Generated

- **Candidate count files** — produced by `equistress_candidates.py` during threshold sensitivity analysis. Different τ values were tested to understand how event density varies with threshold choice.
- **Equivalent stress scale** — recorded by `event_maker.py` as events are constructed, capturing the per-regime normalisation applied before thresholding.
- **Column rankings** — produced by `column_scaler_report.py` to guide feature selection decisions.
- **Scaler build report** — produced by `04_fit_score_normalizer.py` as a diagnostic output alongside the fitted normalizer JSON.

---

## Purpose

These reports informed key pipeline decisions:

- Which equivalent stress threshold (τ) to use for event extraction
- Whether displacement residuals add signal beyond stress alone
- Which features carry the most discriminative information
- Whether per-feature scaling was applied consistently across simulations

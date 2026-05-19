# CIE Project – Group 7
## Imperfection Classification using Set Transformer Architecture

A complete end-to-end machine learning pipeline for classifying structural imperfection types from Finite Element (FE) simulation data. The model learns permutation-invariant patterns in stress/displacement residuals using a **Set Transformer** neural network, avoiding any dependence on fixed spatial grids or node ordering.

---

## Problem Statement

Given FE simulations across different imperfection types and loading scenarios, the objective is to:

- Extract physically meaningful **residual events** from stress and displacement fields
- Represent each simulation as an **unordered set of events**
- Classify the **imperfection type** using a permutation-invariant classifier

---

## Repository Structure

```
.
├── README.md                          ← You are here
│
├── 01_residual_scores.py              ← Compute scalar residual scores
├── 02_label_maker.py                  ← Assign class labels to simulations
├── 03_train_test_val_split.py         ← Split dataset across seeds
├── 04_fit_score_normalizer.py         ← Fit normalization statistics
├── 05_train_model_modular.py          ← Train Set Transformer / Deep Sets (plug-in)
│
├── candidate_graph.py                 ← Similarity graph over candidates
├── equistress_candidates.py           ← Candidate ranking by equivalent stress
├── event_maker.py                     ← Event set construction
├── column_scaler_report.py            ← Feature scaling diagnostics
│
├── dataset_events_manifest.csv        ← Central dataset index
├── events_equivstress_index.csv       ← Event file index
├── label_map_events.json              ← Class label mapping
├── node_coordinates.txt               ← FE node spatial coordinates
│
├── DEPLOYMENT_TRAINING/               ← Final model trained on full dataset
├── FINAL_INFERENCE_PIPELINE/          ← Inference on unseen simulation data
└── STATS_N_INSIGHTS/                  ← Preprocessing statistics and diagnostics
```

---

## Pipeline Overview

```
Raw FE Simulation Results
  → 01  Residual Score Computation       (stress & displacement fields)
  → 02  Label Assignment
  → 03  Train / Val / Test Split
  → 04  Feature Normalization (fit)
  → 05  Set Transformer Training
        ↓
     Evaluation (confusion matrix, classification report, loss curves)
        ↓
  DEPLOYMENT_TRAINING  →  Retrain on full data (no held-out split)
        ↓
  FINAL_INFERENCE_PIPELINE  →  Predict on unseen cases
```

---

## Models

| Model | Description |
|---|---|
| **Set Transformer** | ISAB attention blocks + PMA pooling + MLP head |
| **Deep Sets** | Permutation-invariant baseline with element-wise MLP |

Both are plug-in architectures accepted by `05_train_model_modular.py`.

---

## Experiments

Training was run across **3 random seeds (1, 30, 42)** with hyperparameter tuning. Key outputs per seed:

- `*_metrics_seed*.json` — accuracy, F1, per-class scores
- `*_confusion_matrix_seed*.csv` — raw confusion matrix
- `*_train_history.csv` — loss/accuracy per epoch
- `score_normalizer_seed*.json` — fitted normalization statistics

---

## Key Design Principles

- **Permutation invariance** — set-based representation with no assumed node ordering
- **Physics-guided events** — equivalent stress thresholds drive event construction
- **No data leakage** — normalization fitted on training split only
- **Reproducibility** — fixed seeds, deterministic manifests, fully saved artifacts

---

## Subfolders

| Folder | Purpose |
|---|---|
| [`DEPLOYMENT_TRAINING/`](DEPLOYMENT_TRAINING/) | Trains the final model on the complete dataset for deployment |
| [`FINAL_INFERENCE_PIPELINE/`](FINAL_INFERENCE_PIPELINE/) | Runs the frozen trained model on completely unseen simulation data |
| [`STATS_N_INSIGHTS/`](STATS_N_INSIGHTS/) | Diagnostic reports from preprocessing and feature analysis |

---

## Status

- Full experimentation pipeline complete (3 seeds, 2 architectures)
- Best model selected: Set Transformer (seed 1)
- Deployment model trained on full dataset
- Inference pipeline validated on blind test cases

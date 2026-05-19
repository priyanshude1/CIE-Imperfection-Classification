# Structural Health Monitoring via FE Data
## Detecting & Localizing Bridge Imperfections — CIE Project, Group 7

**Authors:** Priyanshu De (465483), Ved Aiyar (465207), Tejas Khandekar (463188), Nandini Jain (465269), Abhishek Chavan (463185)  
**Programme:** M.Sc. Computer-Aided Conception and Production in Mechanical Engineering

---

## Overview

This project develops a data-driven framework for detecting and localizing structural imperfections in a bridge using Finite Element (FE) simulation data. The subject structure is an equivalent three-dimensional FE model of the **Hohenzollernbrücke**, analysed under multiple load cases including train traffic (2000 N and 5000 N), pedestrian loads, and seasonal thermal effects (28°C and 5°C).

The core insight driving the approach is that **stress residuals** — computed by subtracting perfect-configuration responses from imperfect ones under identical loading — are orders of magnitude more sensitive to localized defects than displacement residuals (which remain at the order of 10⁻⁷). This motivates a stress-first feature strategy throughout the pipeline.

A **two-stage learning framework** is adopted:
1. A **voting-based regime classifier** that distinguishes perfect from imperfect structural states
2. A **Set Transformer** that localizes the specific imperfection class from an event-based representation of stress hotspots

The final model achieves **91.67% test accuracy** and a **macro F1-score of 0.9111** on a 6-class imperfection classification problem, with high-confidence predictions on fully unseen simulation cases.

---

## Problem Definition

Given FE simulation outputs for multiple imperfection types and loading scenarios, the objectives are to:

- Determine whether a structure is **perfect or imperfect** (Stage 1)
- If imperfect, **localize the defect region** by classifying among 6 imperfection classes (Stage 2)

Each simulation produces nodal-level, time-dependent stress and deformation responses across all Cartesian directions. The number of stress events varies per simulation and spatial ordering carries no physical meaning — making set-based learning a natural fit.

---

## End-to-End Pipeline

```
Raw FE Simulation Results (stress + displacement per node, per timestep)
        │
        ▼
[00] Data Cleaning
        Remove invalid nodes (roof region), merge stress/displacement components
        │
        ▼
[01] Residual Score Computation
        Subtract perfect-baseline responses from imperfect cases (identical nodes + timesteps)
        Stress residuals selected as primary features — displacement residuals ~10⁻⁷, negligible
        │
        ▼
[02] Event Construction (Equivalent Stress-based)
        Extract spatio-temporal hotspot events exceeding adaptive thresholds
        Each event: (x, y, z, t, stress_score) — 5 features per event
        Each simulation → unordered set of variable cardinality
        │
        ▼
[03] Dataset Manifest
        Build central index linking event files to imperfection class labels
        │
        ▼
[04] Feature Normalization
        Fit and store regime-aware scalers on training data only
        Frozen at inference time — never refit on test/unseen data
        │
        ▼
[05] Set Transformer Training
        Permutation-invariant classifier over event sets
        ISAB → PMA → MLP head (~16k parameters)
        │
        ▼
Predicted Imperfection Class + Confidence Score
```

---

## Model Architecture

Each simulation is represented as a set of hotspot events with shape **[B, K, 5]**, where K is the variable number of events per case.

| Layer | Input Shape | Output Shape | Parameters | Description |
|---|---|---|---|---|
| Input Events | (B, K, 5) | (B, K, 5) | 0 | Hotspot events: (x, y, z, t, score) |
| Linear Embedding | (B, K, 5) | (B, K, 32) | 192 | Projects raw features to 32-dim latent space |
| ISAB ×1 | (B, K, 32) | (B, K, 32) | ~9k | Induced Set Attention Block — models event-to-event relationships |
| PMA (1 seed) | (B, K, 32) | (B, 1, 32) | ~4.4k | Pooling by Multihead Attention — aggregates to fixed-length representation |
| Squeeze | (B, 1, 32) | (B, 32) | 0 | Remove seed dimension |
| FC Layer 1 | (B, 32) | (B, 64) | 2,112 | Dense classifier layer |
| FC Layer 2 | (B, 64) | (B, 6) | 390 | Class logits |
| **TOTAL** | — | — | **~16k** | Lightweight permutation-invariant classifier |

### Why Set Transformer over DeepSets?

A DeepSets baseline was also evaluated. While it achieved high test accuracy on the small test split, its validation performance was notably lower (Val F1 = 0.822, Val Acc = 0.833) with higher loss variance — signs of overfitting due to its pooling-based aggregation, which cannot model interactions between events. The Set Transformer's attention mechanism captures spatial clustering and temporal co-occurrence of stress hotspots, yielding more stable convergence and better generalization.

---

## Results

### Cross-Validation Performance

| Metric | Set Transformer | DeepSets (baseline) |
|---|---|---|
| Test Accuracy | **91.67%** (11/12) | ~100% (overfit on small split) |
| Macro F1-Score | **0.9111** | — |
| Macro Precision | **0.9444** | — |
| Macro Recall | **0.9167** | — |
| Val F1 | stable | 0.822 |
| Val Accuracy | stable | 0.833 |

### Confusion Matrix Summary
- Predominantly diagonal — most classes cleanly separated across all 6 classes
- Single observed confusion: one Class 4 sample misclassified as Class 2, suggesting shared stress-hotspot patterns between these two imperfection types
- All remaining classes classified perfectly

### Inference on Unseen Cases (Blind Test)

| Case | Train Size | Season | Confidence |
|---|---|---|---|
| Case 5 | Small | Winter | 0.9252 |
| Case 6 | Big | Summer | 0.9900 |
| Case 7 | Big | Summer | 0.8266 |

The model generalizes robustly across varying operating regimes without any retraining.

---

## Repository Structure

```
.
├── README.md                                      ← You are here
│
├── 01_residual_scores.py                          ← Compute stress/displacement residuals
├── 02_label_maker.py                              ← Assign imperfection class labels
├── 03_train_test_val_split.py                     ← Stratified splitting (seeds 1, 30, 42)
├── 04_fit_score_normalizer.py                     ← Fit and store normalization scalers
├── 05_train_model_modular.py                      ← Modular training (plug-in architectures)
│
├── equistress_candidates.py                       ← Candidate event ranking by equivalent stress
├── candidate_graph.py                             ← Similarity graph over candidate events
├── event_maker.py                                 ← Event extraction utilities
├── column_scaler_report.py                        ← Feature scaling diagnostics
│
├── dataset_events_manifest.csv                    ← Central dataset index
├── events_equivstress_index.csv                   ← Event file index
├── label_map_events.json                          ← Class label mapping
├── node_coordinates.txt                           ← FE node spatial coordinates
│
├── score_normalizer_seed{1,30,42}.json            ← Fitted normalizers per seed
├── best_config_settransformer{,30}.json           ← Best hyperparameter configs
├── settransformer_metrics_seed{1,30,42}.json      ← Evaluation metrics per seed
├── deepsets_metrics_seed{1,30}.json               ← DeepSets baseline metrics
│
├── DEPLOYMENT_TRAINING/                           ← Final model trained on full dataset
├── FINAL_INFERENCE_PIPELINE/                      ← Inference on unseen simulation data
└── STATS_N_INSIGHTS/                              ← Preprocessing diagnostics and analysis
```

---

## Reproducing Results

### 1. Experimentation (train/val/test splits)

Run scripts in order from the root directory:

```bash
python 01_residual_scores.py
python 02_label_maker.py
python 03_train_test_val_split.py
python 04_fit_score_normalizer.py
python 05_train_model_modular.py
```

Results (metrics, confusion matrices, training history) are saved with seed suffixes (`_seed1`, `_seed30`, `_seed42`) for full reproducibility.

### 2. Deployment Training (full dataset, no held-out split)

```bash
cd DEPLOYMENT_TRAINING
python 00_clean_data.py
# ... run 01 through 05 in order
python 05_train_model_modular.py
```

Produces `final_full_settransformer_seed1.pt` — the frozen model used in inference.

### 3. Inference on New Data

```bash
cd FINAL_INFERENCE_PIPELINE
# Place new residual CSVs here, verify node_coordinates.txt, then:
python 0X_inference_pipeline.py
```

Outputs predicted imperfection classes and intermediate event files for each input case.

---

## Subfolders

| Folder | Purpose |
|---|---|
| [`DEPLOYMENT_TRAINING/`](DEPLOYMENT_TRAINING/) | Retrains on the complete dataset (no split) to produce the final deployment model |
| [`FINAL_INFERENCE_PIPELINE/`](FINAL_INFERENCE_PIPELINE/) | Runs the frozen model on completely unseen simulation data — no learning occurs here |
| [`STATS_N_INSIGHTS/`](STATS_N_INSIGHTS/) | Candidate counts, column rankings, scaler reports, and stress scale diagnostics |

---

## Key Design Principles

- **Permutation invariance** — event ordering carries no physical meaning; the Set Transformer handles sets natively via ISAB attention
- **Physics-guided features** — equivalent stress hotspots encode localized structural anomalies rather than raw nodal values
- **Zero data leakage** — normalization scalers are fit only on training data and frozen before any test or inference step
- **Reproducibility** — fixed random seeds, deterministic manifests, fully saved training artifacts across all runs

---

## References

- Zaheer et al. (2017). *Deep Sets*. Advances in Neural Information Processing Systems (NeurIPS).
- Lee et al. (2019). *Set Transformer: A Framework for Attention-Based Permutation-Invariant Neural Networks*. ICML.

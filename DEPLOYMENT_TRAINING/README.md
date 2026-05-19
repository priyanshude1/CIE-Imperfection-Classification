# Deployment Training

This directory contains the **final training run** for the CIE Project model. Unlike the experimental runs in the root directory (which used train/val/test splits for model selection), this pipeline trains the chosen Set Transformer architecture on the **complete available dataset** — no held-out split — to maximise the information available to the deployed model.

---

## Purpose

The root directory was used to compare architectures and tune hyperparameters across multiple seeds. Once the best configuration was identified (Set Transformer, seed 1), this directory performs a single definitive training run on all data, producing the model checkpoint used in the inference pipeline.

---

## Pipeline

```
Raw FE Simulation Results
  → 00  Data Cleaning & Validation
  → 01  Residual Score Computation
  → 02  Event Set Construction (equivalent stress–based)
  → 03  Dataset Manifest Assembly
  → 04  Feature Normalization (fit on full data)
  → 05  Set Transformer Training (full dataset)
        ↓
  final_full_settransformer_seed1.pt   ← deployed model checkpoint
```

---

## Scripts

| Script | Description |
|---|---|
| `00_clean_data.py` | Validates and cleans raw displacement and stress residual files |
| `01_residual_scores.py` | Computes scalar residual scores from stress/displacement fields |
| `02_event_maker.py` | Constructs unordered event sets using equivalent stress thresholds and node geometry |
| `03_build_dataset_manifest.py` | Assembles the central index linking event files to class labels |
| `04_fit_score_normalizer.py` | Fits normalization statistics on the full dataset and saves them |
| `05_train_model_modular.py` | Trains the Set Transformer on the complete dataset |
| `equistress_candidates.py` | Candidate ranking by equivalent stress magnitude |
| `candidate_graph.py` | Similarity graph construction over event candidates |
| `column_scaler_report.py` | Diagnostics for per-feature scaling |
| `plot_training_loss.py` | Plots training loss curves |
| `plot_confusion_matrix.py` | Plots confusion matrix from saved predictions |

---

## Outputs

| File | Description |
|---|---|
| `final_full_settransformer_seed1.pt` | Trained model weights — used directly by the inference pipeline |
| `final_full_settransformer_metrics_seed1.json` | Training-set accuracy, F1, per-class scores |
| `final_full_settransformer_history_seed1.csv` | Loss and accuracy per epoch |
| `final_full_settransformer_cm_train_seed1.csv` | Confusion matrix on training data |
| `score_normalizer_full.json` | Normalization statistics fit on full dataset |
| `score_normalizer_diagnostics_full.csv` | Per-feature scaling diagnostics |
| `scaler_build_report.csv` | Summary report of the scaler fitting process |
| `confusion_matrix_plot.png` | Visual confusion matrix |
| `training_loss_plot.png` | Visual training loss curve |

---

## Important Notes

- The normalizer (`score_normalizer_full.json`) fitted here is the one used in inference — it must not be refit at inference time.
- The model checkpoint (`final_full_settransformer_seed1.pt`) is copied to `FINAL_INFERENCE_PIPELINE/` for deployment.
- Training on the full dataset means no validation loss is available — model selection was done in the root directory experiments.

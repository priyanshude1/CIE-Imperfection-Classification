# Final Inference Pipeline

This directory contains the **deployment-ready inference pipeline** for the CIE Project. It applies the frozen Set Transformer model to completely unseen FE simulation data, producing imperfection class predictions without any retraining.

---

## Purpose

After the model is trained in `DEPLOYMENT_TRAINING/`, this pipeline:

1. Takes new residual CSV files from previously unseen simulations
2. Reproduces the exact preprocessing used at training time (event extraction + normalization)
3. Runs a forward pass through the frozen Set Transformer
4. Outputs predicted imperfection classes and optional confidence scores

---

## Inference Flow

```
New FE Simulation Residuals (residual_eqstress_*.csv)
  → Event Extraction using equivalent stress thresholds
  → Feature normalization using frozen training-time scalers
  → Set padding and batching
  → Set Transformer forward pass
  → Predicted imperfection class (+ confidence scores)
```

---

## Directory Contents

| File / Folder | Description |
|---|---|
| `0X_inference_pipeline.py` | Main end-to-end inference script |
| `final_full_settransformer_seed1.pt` | Frozen model checkpoint from deployment training |
| `score_normalizer_full.json` | Frozen normalization statistics (must not be refit) |
| `stress_score_scaler.json` | Frozen stress score scaler |
| `node_coordinates.txt` | FE node spatial coordinates (must match training geometry) |
| `residual_eqstress__*.csv` | Unseen simulation residuals used for blind testing |
| `extracted_events_output_case*.csv` | Intermediate event sets generated during inference |
| `test7_events.csv` | Event set for test case 7 |

---

## Test Cases

Six blind test cases were run through this pipeline:

| File | Case |
|---|---|
| `extracted_events_output_case5.csv` | Case 5 |
| `extracted_events_output_case6.csv` | Case 6 |
| `extracted_events_output_case7.csv` | Case 7 |
| `extracted_events_output_case12.1.csv` | Case 12.1 |
| `extracted_events_output_case12.2.csv` | Case 12.2 |
| `extracted_events_output_case12.3.csv` | Case 12.3 |

---

## Usage

1. Place new residual CSVs in this directory (matching the schema of `residual_eqstress__*.csv`)
2. Verify `node_coordinates.txt` matches the training geometry
3. Run:
   ```bash
   python 0X_inference_pipeline.py
   ```
4. Inspect predicted classes and the `extracted_events_output_*.csv` intermediate files

---

## Consistency Guarantees

This pipeline enforces strict consistency with training:

- Identical feature ordering and column schema
- Identical normalization statistics (scalers loaded from JSON, never refit)
- Permutation-invariant event handling
- Zero information leakage from test data

---

## Important Notes

- This pipeline performs **no learning** — all weights and scalers are frozen.
- Any change to the feature schema or node geometry requires retraining from scratch.
- The `extracted_events_output_*.csv` files are useful for debugging and interpretability analysis.

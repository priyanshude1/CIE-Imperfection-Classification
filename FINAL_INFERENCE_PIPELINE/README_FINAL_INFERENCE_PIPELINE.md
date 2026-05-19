
CIE Project – Final Inference Pipeline
## Imperfection Classification on Unseen Simulation Data

This directory contains the **final inference pipeline** for the CIE Project.  
It uses a **previously trained Set Transformer model** to classify **completely unseen simulation data**, without any retraining.

The pipeline mirrors the training-time preprocessing (event construction + normalization) to ensure **distributional consistency** between training and inference.

---

## 1. Purpose of This Pipeline

The goal of this pipeline is to:

- take **unknown FE simulation residual data**
- construct **event-based set representations**
- apply the **trained Set Transformer**
- output **predicted imperfection classes** (and confidence scores, if enabled)

This directory is intended for **deployment, validation, and blind testing** scenarios.

---

## 2. High-Level Inference Flow

Raw Simulation Residuals  
→ Event Extraction (Equivalent Stress–based)  
→ Feature Normalization (frozen scalers)  
→ Set Transformer Forward Pass  
→ Predicted Imperfection Class

---

## 3. Directory Structure

```
FINAL_INFERENCE_PIPELINE/
│
├── 0X_inference_pipeline.py
│
├── extracted_events_output_case5.csv
├── extracted_events_output_case6.csv
├── extracted_events_output_case7.csv
├── extracted_events_output_case12.1.csv
├── extracted_events_output_case12.2.csv
├── extracted_events_output_case12.3.csv
│
├── residual_eqstress_G2_H_S.csv
├── residual_eqstress_G2_L_S_G12.csv
├── residual_eqstress_G2_L_W.csv
├── residual_eqstress_G3_H_S.csv
├── residual_eqstress_G3_H_S_G12.csv
├── residual_eqstress_G3_L_W_G12.csv
│
├── final_full_settransformer_seed1.pt
│
├── score_normalizer_full.json
├── stress_score_scaler.json
│
├── node_coordinates.txt
├── test7_events.csv
│
└── README.md
```

---

## 4. Core Inference Script

### `0X_inference_pipeline.py`

**Purpose**  
End-to-end inference script that reproduces the training-time data handling and applies the trained model.

**Inputs**
- Raw residual CSV files (stress-based)
- `node_coordinates.txt`
- Frozen normalization files
- Trained model checkpoint

**Key Operations**
- Residual parsing and validation
- Event extraction using equivalent stress logic
- Feature normalization using **training-fitted scalers**
- Set padding / batching
- Forward pass through Set Transformer
- Class prediction

**Outputs**
- Predicted class labels per input case
- Optional confidence / probability scores

---

## 5. Input Data Description

### Residual Files (`residual_eqstress_*.csv`)
Contain equivalent stress residuals for **previously unseen simulations**.

These files must:
- match the schema used during training
- contain physically consistent residual quantities

---

### Node Geometry

#### `node_coordinates.txt`
Provides FE node spatial information required during event construction.

This file **must match** the coordinate system used during training.

---

## 6. Event Extraction Outputs

### `extracted_events_output_*.csv`
Intermediate artifacts containing the generated **event sets** for each inference case.

Each row corresponds to a single event with normalized feature values.

These files are useful for:
- debugging
- interpretability
- downstream analysis

---

## 7. Model & Normalization Artifacts

### Trained Model
- `final_full_settransformer_seed1.pt`  
  Frozen Set Transformer trained during the main CIE training phase.

---

### Normalizers
- `score_normalizer_full.json`
- `stress_score_scaler.json`

These files store **training-time normalization statistics** and must never be refit during inference.

---

## 8. Consistency Guarantees

This pipeline enforces:

- identical feature ordering
- identical normalization statistics
- permutation-invariant event handling
- zero information leakage from test data

This ensures that inference results are **directly comparable** to training metrics.

---

## 9. Typical Usage

1. Place new residual CSVs in this directory
2. Verify `node_coordinates.txt`
3. Run:
   ```bash
   python 0X_inference_pipeline.py
   ```
4. Inspect predicted classes and extracted event files

---

## 10. Status

✔ Model frozen  
✔ Inference-only pipeline  
✔ Ready for blind testing and deployment  

---

## Notes

- This pipeline does **not** perform any learning.
- Any change in feature schema or scaling requires retraining.
- Designed to be lightweight and reproducible.

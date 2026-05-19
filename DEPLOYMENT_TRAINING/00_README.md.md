

# &nbsp; 			CIE Project  Group 7

# &nbsp;Imperfection Classification using Set Transformer Architecture



This repository contains the complete end-to-end pipeline developed in the CIE Project to classify imperfection types from finite-element simulation data using a Set Transformer–based neural network.



The core contribution of this work is a set-based representation of simulation residuals, allowing the model to learn permutation-invariant patterns in stress/displacement events without relying on fixed spatial grids or ordering.



---



#### 1\. Problem Definition



Given multiple FE simulations corresponding to different imperfection types and loading scenarios, the objective is to:



\- extract **physically meaningful residual events**

\- represent each simulation as an **unordered set of events**

\- classify the **imperfection type** using a **Set Transformer**



This avoids explicit dependence on global coordinates and enables robust generalization across simulation variants.



---



#### 2\. High-Level Pipeline



Raw Simulation Results  

→ Residual Computation (Stress / Displacement)  

→ Event Construction (Equivalent Stress–based)  

→ Dataset Manifest + Label Mapping  

→ Feature Normalization  

→ Set Transformer Training  

→ Evaluation \& Candidate Analysis



---



#### 3\. Directory Structure



```

.

├── residuals\_disp/

├── residuals\_stress/

├── events\_equivstress/

├── events\_equivstress.zip

│

├── 00\_clean\_data.py

├── 01\_residual\_scores.py

├── 02\_event\_maker.py

├── 03\_build\_dataset\_manifest.py

├── 04\_fit\_score\_normalizer.py

├── 05\_train\_model\_modular.py

│

├── equistress\_candidates.py

├── candidate\_graph.py

│

├── plot\_training\_loss.py

├── plot\_confusion\_matrix.py

│

├── column\_scaler\_report.py

│

├── dataset\_events\_manifest.csv

├── events\_equivstress\_index.csv

├── label\_map\_events.json

├── node\_coordinates.txt

│

├── final\_full\_settransformer\_seed1.pt

├── final\_full\_settransformer\_metrics\_seed1.json

├── final\_full\_settransformer\_history\_seed1.csv

├── final\_full\_settransformer\_cm\_train\_seed1.csv

└── confusion\_matrix\_plot.png

```



---



#### 4\. Script-by-Script Description



##### 00\_clean\_data.py

Initial preprocessing and validation of simulation outputs.



Inputs

\- Raw displacement and stress residual files



Outputs

\- Clean residual datasets



---



##### 01\_residual\_scores.py

Computes scalar residual scores from displacement and stress fields.



---



##### 02\_event\_maker.py

Constructs unordered event sets from residual scores using equivalent stress and geometry.



---



##### 03\_build\_dataset\_manifest.py

Builds the central dataset index linking event files and labels.



---



##### 04\_fit\_score\_normalizer.py

Fits and stores normalization statistics for event features.



---



##### 05\_train\_model\_modular.py

Trains the Set Transformer classifier using ISAB blocks, PMA pooling, and an MLP head.



---



##### 5\. Analysis \& Visualization



\- plot\_training\_loss.py – loss curves

\- plot\_confusion\_matrix.py – confusion matrix

\- column\_scaler\_report.py – feature scaling diagnostics



---

##### 

##### 6\. Candidate \& Similarity Analysis



\- equistress\_candidates.py – candidate ranking

\- candidate\_graph.py – similarity graph construction



---



##### 7\. Reproducibility



\- Fixed random seed

\- Deterministic dataset manifest

\- Fully saved training artifacts



---



##### 8\. Key Design Principles



\- Permutation invariance

\- Physics-guided event construction

\- No reliance on global ordering

\- Scalable to variable set sizes



---



##### 9\. Status



✔ End-to-end pipeline complete  

✔ Model trained and evaluated  

✔ Ready for extension and experimentation




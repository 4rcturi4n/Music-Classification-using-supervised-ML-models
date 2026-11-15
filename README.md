# 🎵 Music Genre Classification with Supervised ML

Classifying songs into **10 music genres** from hand-crafted audio features, comparing several classical machine learning models.

## Overview

1,000 audio clips (100 per genre, GTZAN-style dataset: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock) were turned into numerical features with **librosa** and classified with supervised models tuned by grid search.

## Pipeline

1. **Feature extraction (librosa)** – for every clip:
   - Mel spectrogram
   - 13 MFCCs
   - Chromagram
   - Tempo
   - Zero-crossing rate
   - Spectral centroid, bandwidth and roll-off
2. **Feature engineering** – mean and variance of each feature over time → one feature vector per song, standardized with `StandardScaler`.
3. **Models** (hyperparameters tuned with `GridSearchCV` / validation curves):
   - Random Forest
   - Support Vector Classifier (RBF kernel)
   - AdaBoost (SVC weak learner)
   - k-Nearest Neighbors
   - XGBoost (with feature-importance analysis)
4. **Evaluation** – test accuracy, confusion matrices per model, comparison to a random-guess baseline.

## Results

| | Accuracy |
|---|---|
| Random guess (10 classes) | ~0.10–0.12 |
| Best model on the test set | **~0.71** |

*Scaling is now fitted on the training set only, and XGBoost early stopping uses a validation split instead of the test set. Re-running the notebook may change these numbers slightly.*

Confusion matrices show which genres are most often mixed up (e.g. rock ↔ metal / country), and XGBoost feature importance shows which audio features matter most.

## Tech stack

Python · librosa · scikit-learn · XGBoost · pandas · NumPy · Matplotlib · Seaborn

## How to run

```bash
pip install librosa scikit-learn xgboost pandas numpy matplotlib seaborn yellowbrick
jupyter notebook machine_learning_project_whole.ipynb
```

Set `data_folder` in the notebook to the folder containing the `genres/<genre>/*.au` files.

## Author

Michaela Chrisafaki – Physicist, Machine Learning

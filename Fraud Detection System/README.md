# Fraud Detection System – Credit Card Transactions

## 1. Project Overview

This project implements an end-to-end machine learning system to detect fraudulent credit card transactions using a highly imbalanced dataset.

The objective is to:

* Detect fraudulent transactions effectively in rare-event data (~0.17% fraud rate).
* Optimize performance using Precision-Recall metrics.
* Apply threshold tuning to simulate real-world business trade-offs.
* Package the solution into a reusable prediction pipeline.

The final system supports inference on new CSV files using a command-line interface.

---

## 2. Dataset

**Source:** Kaggle – Credit Card Fraud Detection

**Dataset Characteristics:**

* 284,807 transactions
* 492 fraud cases (~0.17%)
* Highly imbalanced classification problem

**Features:**

* V1–V28: PCA-transformed numerical features
* Time: Seconds elapsed between transactions
* Amount: Transaction amount
* Class: Target variable (1 = fraud, 0 = legitimate)

Due to the severe class imbalance, model evaluation focuses on:

* Precision
* Recall
* F1-score
* PR-AUC (Primary evaluation metric)

Accuracy is not used as a main metric.

---

## 3. Methodology

### 3.1 Data Preparation

* Stratified train/test split to preserve class distribution.
* Leakage-safe preprocessing (no test data used during training).
* Class imbalance handled using class weights.

---

### 3.2 Feature Engineering

The following additional features were created:

* `log_amount`: Log-transformed transaction amount.
* `hour`: Hour extracted from Time.
* `hour_sin`, `hour_cos`: Cyclical encoding of hour.

Feature engineering is integrated directly into the final pipeline.

---

### 3.3 Models Evaluated

1. Logistic Regression (baseline)
2. Logistic Regression with class weights
3. Random Forest
4. HistGradientBoosting
5. Tuned Random Forest (RandomizedSearchCV)

The tuned Random Forest was selected as the final model.

---

## 4. Final Model Performance (Test Set)

Tuned Random Forest with feature engineering:

| Metric    | Value |
| --------- | ----- |
| Precision | 0.95  |
| Recall    | 0.79  |
| F1-score  | 0.86  |
| PR-AUC    | 0.86  |

---

## 5. Threshold Tuning

Default classification threshold: 0.50
Tuned threshold: 0.01

Lowering the threshold increases recall (detecting more fraud) but increases false positives. This demonstrates how business requirements can influence deployment decisions.

---

## 6. Project Structure

```
Fraud Detection System/
│
├── data/
│   └── raw/
│       └── creditcard.csv
│
├── models/
│   ├── fraud_pipeline_rf_v1.joblib
│   └── fraud_threshold_v1.joblib
│
├── reports/
│
├── src/
│   ├── feature_engineering.py
│   ├── predict.py
│   └── __init__.py
│
└── notebooks/
```

---

## 7. Running Inference

### Step 1: Navigate to Project Root

```
cd Fraud Detection System
```

### Step 2: Run Prediction Script

```
python -m src.predict --input data/raw/creditcard.csv --output reports/predictions_full.csv
```

---

## 8. Output

The script:

* Loads the saved pipeline
* Applies feature engineering automatically
* Generates fraud probabilities
* Applies the tuned threshold
* Saves results to the specified output file

The output CSV includes:

* `fraud_probability`
* `fraud_prediction`

Example console output:

```
Saved predictions to: reports/predictions_full.csv
Threshold used: 0.01
Predicted fraud count: 1843 / 284807
```

---

## 9. Error Analysis Summary

Key observations from error analysis:

* Some high-confidence false positives involve low-amount transactions.
* Missed fraud cases (false negatives) often resemble legitimate patterns in PCA-transformed space.
* Lower thresholds significantly increase fraud detection but increase flagged legitimate transactions.

These findings highlight the importance of threshold selection in fraud detection systems.

---

## 10. Technologies Used

* Python
* pandas
* numpy
* scikit-learn
* joblib

---

## 11. Key Engineering Practices

* Proper train/test separation
* Leakage-safe preprocessing
* Class imbalance handling
* Lightweight hyperparameter tuning
* Threshold tuning for business alignment
* Packaged and reusable prediction pipeline
* Command-line inference support

---

## 12. Future Improvements

* Model explainability (e.g., SHAP analysis)
* Cost-sensitive optimization
* Model monitoring framework
* API deployment
* Containerization

---

## Author

Saiful Islam Shihab
La Trobe University

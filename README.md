
# Credit card Fraud Detection using Machine Learning

This project demonstrates the use of supervised machine learning to detect fraudulent transactions in financial data. It focuses on addressing the real-world challenge of class imbalance using multiple resampling techniques and evaluating model performance using a robust pipeline.

---

##  Overview

Financial fraud is rare but costly, making accurate detection critical. In this project, I apply different balancing techniques combined with classification models to distinguish between genuine and fraudulent transactions.

---

##  Objectives

- Explore and visualize transaction data
- Address severe class imbalance using:
  - SMOTE (oversampling)
  - NearMiss (undersampling)
  - SMOTE-ENN (combined technique)
- Evaluate models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

---

## 🧪 Methodology

### 1. Preprocessing
- Standard scaling with `StandardScaler`
- Class balancing with `SMOTE`, `NearMiss`, and `SMOTE-ENN`
- Dimensionality reduction using `LinearDiscriminantAnalysis`

### 2. Modeling
- `RandomForestClassifier` trained within a `Pipeline`
- Hyperparameter tuning using `GridSearchCV`
- Cross-validation via `RepeatedStratifiedKFold`
- Evaluation using `cross_val_score` and classification metrics

---

##  Requirements

To install all required packages:

```bash
pip install -r requirements.txt
```

---

##  Project Structure

```
fraud-detection-ml/
│
├── fraud_detection_ml_project.ipynb   # Main notebook
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── README.md                          # Project overview and documentation
```

---

##  Dataset

The dataset used is a publicly available anonymized set of European credit card transactions from 2013. It is highly imbalanced, with fraudulent transactions making up less than 0.2% of all data.

- **File name**: `data/creditcard.csv`
- **Rows**: 284,807
- **Fraudulent cases**: 492
- **Legit transactions**: 284,315
- **Features**:
  - `Time`, `Amount`
  - `V1` to `V28`: Principal components from PCA (sensitive features obscured)
  - `Class`: Binary target (0 = Non-Fraud, 1 = Fraud)

>  [Download from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

##  License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

##  Connect

Built by [Elizabeth Osanyinro](https://www.linkedin.com/in/elizabethosanyinro/)  
GitHub: [@ElizabethOsanyinro](https://github.com/ElizabethOsanyinro)

---


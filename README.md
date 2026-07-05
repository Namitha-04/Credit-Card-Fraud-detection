# 💳 Credit Card Fraud Detection

A complete end-to-end machine learning project to detect fraudulent credit card transactions using supervised and unsupervised techniques — with novel additions including threshold tuning, cost-sensitive analysis, and fraud pattern profiling.

---

## 📌 Project Overview

Credit card fraud is a severe global problem costing billions annually. This project builds a fraud detection system on a real-world dataset of 284,807 transactions — where only 0.18% are fraudulent — making it one of the most challenging class imbalance problems in data science.

---

## 📂 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud cases:** 492 (0.18%)
- **Features:** 30 (V1–V28 PCA-transformed, Time, Amount)
- **Target:** Class (0 = Legitimate, 1 = Fraud)

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualisation |
| Scikit-learn | ML models and evaluation |
| Imbalanced-learn | SMOTE oversampling |
| XGBoost | Gradient boosting model |
| SHAP | Model explainability |

---

## 🚀 Project Pipeline

```
1. Data Loading & Exploration
      ↓
2. Visualising Class Imbalance
      ↓
3. Preprocessing (Scaling + Train/Test Split)
      ↓
4. Handling Imbalance with SMOTE
      ↓
5. Training 3 Models
      ↓
6. Evaluation with Proper Metrics
      ↓
7. Novel Additions (Threshold Tuning + Cost Analysis + Fraud Profiling)
```

---

## 🤖 Models Used

### Logistic Regression (Baseline)
Simple linear model used as a starting point to benchmark all other models against.

### Random Forest
An ensemble of 100 decision trees that vote together — more powerful than a single tree and resistant to overfitting.

### XGBoost
A sequential boosting algorithm where each tree corrects the mistakes of the previous one — the industry standard for tabular fraud detection.

---

## ⚖️ Handling Class Imbalance

With only 0.18% fraud in the dataset, standard models would simply predict everything as legitimate and achieve 99.8% accuracy while catching zero fraud.

Two strategies were used:
- **SMOTE** (Synthetic Minority Oversampling) for Logistic Regression and Random Forest
- **scale_pos_weight** parameter for XGBoost

---

## 📊 Model Results

| Model | Fraud Precision | Fraud Recall | F1 Score |
|---|---|---|---|
| Logistic Regression | 0.11 | 0.92 | 0.20 |
| Random Forest | 0.91 | 0.81 | 0.85 |
| XGBoost | 0.76 | 0.89 | 0.82 |

> **Note:** Accuracy is not reported — it is a misleading metric for imbalanced datasets. PR-AUC and Recall are the primary evaluation metrics.

---

## 🔬 Novel Contributions

### 1. Threshold Tuning
Rather than using the default 0.5 decision threshold, the optimal threshold was found using the Precision-Recall curve:

```
Default threshold (0.5)  → Precision: 0.76, Recall: 0.89
Tuned threshold (0.9999) → Precision: 0.99, Recall: 0.77
```

This allows the model to be adjusted based on business priorities — catching more fraud vs minimising false alarms.

### 2. Cost-Sensitive Analysis
Models were evaluated not just on scores but on real business cost:

```
Cost of missing fraud     = $50,000 per transaction
Cost of false alarm       = $1,000 per investigation
```

This quantifies how much money each model saves the bank compared to having no fraud detection at all — translating ML metrics into business value.

### 3. Fraud Pattern Profiling
After detection, fraudulent transactions were analysed to understand their characteristics:

**When does fraud happen?**
- Peaks at 1am–2am (late night when cardholders are asleep)
- Significant spike at 11am
- Another cluster around 5pm–6pm

**How much are fraudulent transactions?**
- Heavily concentrated below $50
- Fraudsters use small amounts to test stolen cards before larger purchases
- This is a known pattern called card testing fraud

**Which features are strongest fraud signals?**
- V3 and V14 show the largest difference between fraud and legitimate transactions
- These likely correspond to location, device, or behavioural patterns in the original data

---

## 📈 Key Takeaways

```
1. Never use accuracy for imbalanced datasets — use PR-AUC and Recall
2. SMOTE must only be applied to training data, never test data
3. Threshold tuning can dramatically improve business-relevant metrics
4. Understanding fraud patterns is as valuable as detecting it
5. XGBoost with scale_pos_weight is the strongest performer overall
```

---

## 🗂️ Repository Structure

```
├── fraud_detection.ipynb   # Main Jupyter notebook
├── README.md               # This file
└── requirements.txt        # Dependencies
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place creditcard.csv in the folder

# 4. Open the notebook
jupyter notebook fraud_detection.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
```

---

## 👤 Author

Made by Namitha G  
Feel free to connect on [LinkedIn](http://www.linkedin.com/in/namithag)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

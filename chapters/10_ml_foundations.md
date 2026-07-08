# Chapter 10 — Machine Learning Foundations

[← Previous: EDA & Data Cleaning](09_eda_cleaning.md) | [Back to Index](../README.md) | [Next: ML Algorithms →](11_ml_algorithms.md)

---

## 10.1 What is Machine Learning?

**Machine Learning (ML)** is a subset of AI where systems learn patterns from data and improve performance on a task without being explicitly programmed with fixed rules.

### AI vs ML vs DL vs NLP

| Term | Definition |
|------|-----------|
| **AI (Artificial Intelligence)** | The broad field of building systems that mimic intelligent human behavior |
| **ML (Machine Learning)** | Subset of AI — systems that learn patterns from data |
| **DL (Deep Learning)** | Subset of ML — uses multi-layered neural networks to learn hierarchical representations |
| **NLP (Natural Language Processing)** | Field (using ML/DL techniques) focused on understanding/generating human language |

```
AI ⊃ ML ⊃ DL
NLP uses techniques from ML and DL, applied specifically to text/language
```

---

## 10.2 Types of Machine Learning

### Supervised Learning
Learn from **labeled** data (input → known output).
- **Classification**: predict a discrete category (spam/not spam, survived/died)
- **Regression**: predict a continuous value (house price, temperature)

### Unsupervised Learning
Learn from **unlabeled** data — find hidden structure.
- **Clustering**: group similar data points (K-Means)
- **Dimensionality Reduction**: compress features while retaining information (PCA)

### Reinforcement Learning
An agent learns by interacting with an environment, receiving rewards/penalties for actions, aiming to maximize cumulative reward over time.

🔑 **Interview Q: Classification vs Regression — how do you know which one you need?**
If the target variable is categorical/discrete (a finite set of classes) → classification. If the target is continuous/numeric → regression. E.g., predicting "will this customer churn?" (yes/no) is classification; predicting "how much will this customer spend?" is regression.

---

## 10.3 The ML Workflow

```
1. Define the problem & collect data
2. EDA & data cleaning
3. Feature engineering & selection
4. Train-test split
5. Choose & train a model
6. Evaluate on test data
7. Tune hyperparameters
8. Deploy / iterate
```

---

## 10.4 Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **`test_size`**: fraction of data reserved for testing (commonly 0.2–0.3)
- **`random_state`**: seed for reproducibility — same split every run
- **`stratify=y`**: preserves the class distribution proportion in both train and test sets — critical for imbalanced classification problems

🔑 **Interview Q: Why do we need a train-test split at all?**
To evaluate how well a model **generalizes** to unseen data. Training and evaluating on the same data gives a falsely optimistic performance estimate, since the model may simply memorize the training set rather than learn generalizable patterns.

### Data Leakage
Data leakage occurs when information from outside the training set (often from the test set, or from the future) improperly influences the model — causing overly optimistic performance during evaluation that fails to hold up in production.

⚠️ **Common mistake:** Scaling/normalizing the **entire dataset** before splitting. Always fit scalers/encoders on the training set only, then transform the test set with those same fitted parameters.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)     # fit AND transform on train
X_test_scaled = scaler.transform(X_test)             # only transform on test (no fit!)
```

### Validation Set & Cross-Validation
A **validation set** is a third split (besides train/test) used for tuning hyperparameters without touching the final test set.

**K-Fold Cross-Validation**: split data into `k` folds; train on `k-1` folds, validate on the remaining fold; repeat `k` times, rotating the validation fold; average the results for a more robust performance estimate.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)     # 5-fold cross-validation
print(scores.mean())
```

---

## 10.5 Bias-Variance Tradeoff

- **Bias**: error from overly simplistic assumptions — the model **underfits**, missing real patterns.
- **Variance**: error from excessive sensitivity to training data fluctuations — the model **overfits**, capturing noise as if it were signal.

| | Bias | Variance |
|---|------|----------|
| **High** | Underfitting (too simple) | Overfitting (too complex) |
| Symptom | Poor performance on both train & test | Great on train, poor on test |
| Fix | More complex model, more features | Regularization, more data, simpler model |

```
Total Error = Bias² + Variance + Irreducible Error
```

🔑 **Interview Q: How do you detect overfitting?**
Training accuracy is very high, but test/validation accuracy is significantly lower — the model has memorized training data patterns (including noise) that don't generalize.

### Overfitting vs Underfitting — Fixes

**Overfitting fixes:**
- Get more training data
- Simplify the model (fewer features, shallower tree, fewer layers)
- Regularization (L1/L2)
- Cross-validation
- Early stopping
- Dropout (neural networks)

**Underfitting fixes:**
- More complex model
- More/better features
- Reduce regularization
- Train longer

---

## 10.6 Feature Scaling

Many algorithms (KNN, SVM, gradient descent-based models, PCA) are sensitive to feature scale.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

StandardScaler()   # (x - mean) / std → mean 0, std 1  ("standardization")
MinMaxScaler()      # (x - min) / (max - min) → scales to [0, 1]  ("normalization")
```

💡 Tree-based models (Decision Tree, Random Forest, XGBoost) are **scale-invariant** and don't require feature scaling.

---

## ✅ Chapter 10 Summary

- AI ⊃ ML ⊃ DL; NLP applies ML/DL to language
- Supervised (classification/regression) vs Unsupervised (clustering/dim. reduction) vs Reinforcement Learning
- Train-test split evaluates generalization; `random_state` for reproducibility, `stratify` for balanced classes
- Fit scalers/encoders on train data only — never leak test data into training
- Bias-Variance tradeoff: underfitting (high bias) vs overfitting (high variance)

---

[← Previous: EDA & Data Cleaning](09_eda_cleaning.md) | [Back to Index](../README.md) | [Next: ML Algorithms →](11_ml_algorithms.md)

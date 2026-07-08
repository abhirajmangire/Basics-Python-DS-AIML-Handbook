# Chapter 12 — Model Evaluation

[← Previous: ML Algorithms](11_ml_algorithms.md) | [Back to Index](../README.md) | [Next: Deep Learning →](13_deep_learning.md)

---

## 12.1 Confusion Matrix

The foundation for nearly every classification metric.

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

- **TP**: correctly predicted positive
- **TN**: correctly predicted negative
- **FP** (Type I error): predicted positive, actually negative — "false alarm"
- **FN** (Type II error): predicted negative, actually positive — "missed detection"

---

## 12.2 Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

⚠️ **Common mistake:** Using accuracy alone on **imbalanced datasets**. If 95% of emails are "not spam," a model that always predicts "not spam" gets 95% accuracy while being completely useless.

---

## 12.3 Precision

**"Of everything I predicted positive, how many were actually positive?"**
```
Precision = TP / (TP + FP)
```
```python
from sklearn.metrics import precision_score
precision_score(y_test, y_pred)
```
💡 Prioritize precision when **false positives are costly** (e.g., spam filter wrongly flagging an important email).

---

## 12.4 Recall (Sensitivity)

**"Of all actual positives, how many did I correctly catch?"**
```
Recall = TP / (TP + FN)
```
```python
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)
```
💡 Prioritize recall when **false negatives are costly** (e.g., missing a cancer diagnosis, missing fraud).

---

## 12.5 F1 Score

The **harmonic mean** of precision and recall — balances both when you need a single metric.
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
```python
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)
```

🔑 **Interview Q: Why harmonic mean instead of a simple average for F1?**
The harmonic mean punishes extreme imbalance between precision and recall more heavily than a simple arithmetic average — a model with precision=1.0 and recall=0.0 would get an arithmetic average of 0.5 (misleadingly good), but an F1 (harmonic mean) of 0, correctly reflecting that the model is useless.

---

## 12.6 Precision-Recall Tradeoff

Raising the classification threshold (e.g., from 0.5 to 0.8) typically **increases precision** but **decreases recall**, and vice versa — they trade off against each other. The right balance depends on the business cost of FP vs FN.

```python
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
```

---

## 12.7 ROC Curve & AUC

**ROC (Receiver Operating Characteristic) curve** plots **True Positive Rate (Recall)** vs **False Positive Rate** at every possible classification threshold.

```
FPR = FP / (FP + TN)
```

**AUC (Area Under the Curve)** summarizes the ROC curve into a single number:
- AUC = 1.0 → perfect classifier
- AUC = 0.5 → no better than random guessing
- AUC < 0.5 → worse than random (predictions are inverted)

```python
from sklearn.metrics import roc_auc_score, roc_curve
roc_auc_score(y_test, y_probs)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
```

💡 **AUC vs F1**: AUC evaluates ranking quality across ALL thresholds (threshold-independent); F1 is threshold-specific. AUC is great for comparing models overall; F1 is better once you've committed to a specific decision threshold.

---

## 12.8 Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** (Mean Absolute Error) | `mean(\|y - ŷ\|)` | Same units as target, robust to outliers |
| **MSE** (Mean Squared Error) | `mean((y - ŷ)²)` | Penalizes large errors heavily (squared) |
| **RMSE** (Root MSE) | `sqrt(MSE)` | Same units as target, still penalizes large errors |
| **R² (R-squared)** | `1 - (SS_res / SS_tot)` | Proportion of variance explained by the model (0 to 1, higher is better) |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
```

🔑 **Interview Q: MAE vs RMSE — when would you prefer one over the other?**
MAE treats all errors linearly and is robust to outliers. RMSE squares errors before averaging, so it penalizes large errors much more heavily — prefer RMSE when large errors are especially undesirable/costly; prefer MAE when you want a metric that's easy to interpret and resistant to a few extreme outliers.

---

## 12.9 Cross-Validation (Revisited)

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
print(f"Mean F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

💡 Use `StratifiedKFold` (not plain `KFold`) for classification tasks to preserve class proportions in every fold — especially critical for imbalanced datasets.

---

## 12.10 Handling Class Imbalance

- **Resampling**: oversample the minority class (e.g., **SMOTE**) or undersample the majority class
- **Class weights**: penalize misclassifying the minority class more heavily (`class_weight="balanced"` in scikit-learn)
- **Threshold tuning**: adjust the decision threshold instead of the default 0.5
- **Use appropriate metrics**: F1, AUC, Precision-Recall curve instead of raw accuracy

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight="balanced")
```

---

## ✅ Chapter 12 Summary

- Confusion Matrix (TP/TN/FP/FN) underlies all classification metrics
- Precision = trustworthiness of positive predictions; Recall = coverage of actual positives; F1 = harmonic mean of both
- ROC-AUC evaluates ranking quality across all thresholds; use for comparing models overall
- MAE/RMSE/R² for regression — RMSE penalizes large errors more than MAE
- Never rely on accuracy alone with imbalanced classes — use stratified CV + F1/AUC + class weighting

---

[← Previous: ML Algorithms](11_ml_algorithms.md) | [Back to Index](../README.md) | [Next: Deep Learning →](13_deep_learning.md)

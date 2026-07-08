# Chapter 16 — Cheat Sheets

[← Previous: Interview Questions](15_interview_questions.md) | [Back to Index](../README.md)

---

## 🐍 Python Quick Reference

```python
# Data types
int, float, str, bool, list, tuple, set, dict, NoneType

# String methods
s.lower(), s.upper(), s.strip(), s.split(), s.replace(), s.find(), s[::-1]

# List operations
lst.append(x), lst.remove(x), lst.pop(), lst.sort(), [x for x in lst if cond]

# Dict operations
d.get(k, default), d.keys(), d.values(), d.items(), {k:v for k,v in d.items()}

# Function essentials
def f(a, b=1, *args, **kwargs): ...
lambda x: x + 1

# Exception handling
try: ... except ExceptionType as e: ... finally: ...

# OOP
class Child(Parent):
    def __init__(self):
        super().__init__()
```

---

## 🔢 NumPy Quick Reference

```python
np.array([1,2,3])            np.zeros((3,3))            np.ones((2,4))
np.arange(0,10,2)             np.linspace(0,1,5)          np.random.rand(3,3)
a.shape  a.ndim  a.dtype       a.reshape(r,c)              a.flatten()
a + b (broadcasting)            a @ b (matrix mult)          np.dot(a,b)
np.mean(a) np.std(a) np.sum(a)  np.min(a) np.max(a)          np.sort(a)
a[a > 5]  (boolean indexing)      a[1:3, 0:2]  (slicing)
```

---

## 🐼 Pandas Quick Reference

```python
pd.read_csv("file.csv")             df.head()  df.tail()  df.shape  df.info()  df.describe()
df["col"]  df[["c1","c2"]]            df.loc[label]  df.iloc[position]
df[df["col"] > x]                       df.isnull().sum()  df.dropna()  df.fillna(val)
df.drop_duplicates()                       df.groupby("col").mean()
pd.merge(df1, df2, on="key", how="inner")     pd.concat([df1, df2])
df.sort_values("col")                            df["col"].value_counts()
df.corr()                                          df.apply(func)  df["col"].map(dict)
```

| how (merge) | Result |
|---|---|
| inner | matching keys only |
| left | all left + matched right |
| right | all right + matched left |
| outer | all rows, NaN where unmatched |

---

## 📊 Chart Selection Cheat Sheet

| Data | Best Chart |
|------|-----------|
| 1 numeric variable | Histogram, Box Plot |
| 1 categorical variable | Bar Chart |
| 2 numeric variables | Scatter Plot |
| Numeric + Categorical | Box Plot, Violin Plot |
| Time series | Line Plot |
| Correlation (many vars) | Heatmap |
| Proportions | Bar Chart (Pie only if ≤5 categories) |

---

## 📈 Statistics Cheat Sheet

```
Mean = Σx / n                       Median = middle value (sorted)
Variance = Σ(x-mean)² / (n-1)         Std Dev = √Variance
IQR = Q3 - Q1                            Outlier bounds = Q1 - 1.5×IQR, Q3 + 1.5×IQR
Z-score = (x - mean) / std                  |Z| > 3 → likely outlier

Bayes: P(A|B) = P(B|A) × P(A) / P(B)

p-value < 0.05 → reject null hypothesis (statistically significant)
Type I error = false positive (reject true H0)
Type II error = false negative (fail to reject false H0)
```

---

## 🤖 ML Algorithm Selection Cheat Sheet

| Problem | Try First |
|---------|-----------|
| Simple continuous prediction | Linear Regression |
| Binary classification, need interpretability | Logistic Regression |
| Non-linear, interpretable | Decision Tree |
| Non-linear, best accuracy, tabular data | Random Forest / XGBoost |
| High-dimensional, text | Naive Bayes / SVM |
| Small dataset, non-linear boundary | SVM (RBF kernel) |
| No assumption on data distribution | KNN |
| Unlabeled data, find groups | K-Means |
| Images | CNN |
| Sequential/text data | RNN/LSTM or Transformers |

---

## 📐 Model Evaluation Cheat Sheet

```
Accuracy  = (TP+TN) / (TP+TN+FP+FN)        — misleading on imbalanced data
Precision = TP / (TP+FP)                     — trustworthiness of positive predictions
Recall    = TP / (TP+FN)                       — coverage of actual positives
F1        = 2 × (P×R) / (P+R)                    — harmonic mean of P & R
AUC-ROC   = ranking quality across thresholds      — 0.5 random, 1.0 perfect

Regression:
MAE  = mean(|y - ŷ|)              — robust to outliers
RMSE = √(mean((y-ŷ)²))              — penalizes large errors more
R²   = variance explained by model     — 0 to 1, higher better
```

---

## 🧠 Deep Learning Cheat Sheet

```
Neuron: output = activation(Σ(w·x) + b)

Activations: Sigmoid (0-1, output layer binary) | Softmax (multi-class output)
             ReLU (hidden layers, avoids vanishing gradient) | Tanh (-1 to 1)

Loss: Binary Cross-Entropy (binary) | Categorical Cross-Entropy (multi-class) | MSE (regression)

Optimizer: Adam (adaptive, most common default)

Overfitting fixes: Dropout | Early Stopping | L1/L2 Regularization | Data Augmentation

CNN → images (Conv + Pooling layers)
RNN/LSTM → sequences (text, time series)
Transformer → self-attention, parallel processing, state-of-the-art NLP
```

---

## 📝 NLP Pipeline Cheat Sheet

```
Raw Text
  → Lowercase
  → Remove punctuation/numbers/URLs (regex)
  → Tokenize
  → Remove stopwords
  → Stem (fast) or Lemmatize (accurate)
  → Vectorize: BoW (counts) or TF-IDF (weighted) or Word2Vec (embeddings)
  → Feed into model: Naive Bayes (fast baseline) or Neural Net/Transformer (best accuracy)
```

---

## ⚠️ Common Errors Quick Fix Reference

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | `pip install <package>`, check active environment |
| `IndentationError` | Use consistent 4-space indentation, no tab/space mixing |
| `KeyError` | Check exact column/key spelling with `df.columns` |
| `ValueError: could not convert` | Check data type before casting (`int()`, `float()`) |
| `SettingWithCopyWarning` | Use `df.loc[condition, "col"] = value` |
| Shape mismatch on merge | Ensure join key columns have matching dtypes |
| `IndexError` | Check list/array length before indexing |
| `AttributeError` | Verify the object type — method may not exist on it |

---

## 🎯 Final Revision Checklist

- [ ] Can explain every function used in your projects (not just run the code)
- [ ] Can explain why you chose median over mean for imputation (or vice versa)
- [ ] Can explain your model's evaluation metric and why it's appropriate for the problem
- [ ] Can explain overfitting vs underfitting with your own project as an example
- [ ] Can walk through your entire pipeline from raw data to final prediction, out loud, without notes

---

[← Previous: Interview Questions](15_interview_questions.md) | [Back to Index](../README.md)

---

### 🏁 End of Handbook

*You've reached the end of the Python & Data Science Complete Handbook. Revisit chapters as needed, use the interview bank for active recall, and keep the cheat sheets handy for quick pre-interview revision.*

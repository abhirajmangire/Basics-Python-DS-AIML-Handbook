# Chapter 9 — EDA & Data Cleaning

[← Previous: Statistics](08_statistics.md) | [Back to Index](../README.md) | [Next: ML Foundations →](10_ml_foundations.md)

---

## 9.1 What is EDA & Why It Matters

**Exploratory Data Analysis (EDA)** is the process of investigating a dataset to understand its structure, spot patterns, detect anomalies, and test assumptions — usually with summary statistics and visualizations — before applying any modeling.

**Why it's important:**
- Reveals data quality issues (missing values, duplicates, wrong types) early
- Uncovers relationships and patterns that inform feature engineering
- Prevents "garbage in, garbage out" — bad data quietly produces bad models
- Builds intuition about the problem before jumping to algorithms

---

## 9.2 Types of EDA

### Univariate Analysis (one variable at a time)
```python
df["Age"].describe()
sns.histplot(df["Age"])
sns.boxplot(x=df["Age"])
df["Gender"].value_counts()
```

### Bivariate Analysis (relationship between two variables)
```python
sns.scatterplot(x="Age", y="Income", data=df)
df.groupby("Gender")["Income"].mean()
pd.crosstab(df["Gender"], df["Survived"])
```

### Multivariate Analysis (relationships among 3+ variables)
```python
sns.pairplot(df, hue="Survived")
sns.heatmap(df.corr(numeric_only=True), annot=True)
```

---

## 9.3 The Standard EDA Workflow

1. **Load & inspect** — `df.head()`, `df.info()`, `df.shape`, `df.describe()`
2. **Check missing values** — `df.isnull().sum()`
3. **Check duplicates** — `df.duplicated().sum()`
4. **Understand data types** — are numeric columns actually numeric? Are categorical columns encoded as strings?
5. **Univariate analysis** — distribution of each variable
6. **Bivariate/multivariate analysis** — relationships with the target variable
7. **Detect outliers** — box plots, IQR method, z-scores
8. **Handle missing data & outliers**
9. **Feature engineering** — create new, more useful variables
10. **Encode categorical variables** — prepare for modeling

---

## 9.4 Missing Values — Deep Dive

### Types of Missing Data
| Type | Meaning |
|------|---------|
| **MCAR** (Missing Completely at Random) | Missingness unrelated to any variable |
| **MAR** (Missing at Random) | Missingness related to *other observed* variables |
| **MNAR** (Missing Not at Random) | Missingness related to the *missing value itself* (e.g., high earners refusing to disclose income) |

```python
df.isnull().sum()
df.isnull().mean() * 100      # percentage missing per column
sns.heatmap(df.isnull(), cbar=False)    # visualize missingness pattern
```

**Strategies:**
- Drop rows/columns (only if missingness is small % or column is mostly empty)
- Impute with mean/median/mode
- Impute with a model-based prediction (KNN imputer, regression imputation)
- Flag with a separate "was_missing" indicator column (preserves signal from MNAR)

---

## 9.5 Outlier Detection

### IQR Method
```python
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["Age"] < lower_bound) | (df["Age"] > upper_bound)]
```

### Z-Score Method
```python
from scipy import stats
z_scores = np.abs(stats.zscore(df["Age"]))
outliers = df[z_scores > 3]     # more than 3 std deviations from the mean
```

💡 **What to do with outliers**: investigate first (are they data entry errors or genuine extreme values?). Options: remove, cap (winsorize), transform (log transform), or leave as-is if genuinely valid and important to the analysis.

---

## 9.6 Duplicate & Inconsistent Data

```python
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df["City"] = df["City"].str.strip().str.lower().str.title()   # standardize text
df["City"].replace({"NY": "New York", "ny": "New York"}, inplace=True)   # fix inconsistent categories
```

⚠️ **Common mistake:** Treating "New York", "new york", and "NY " as three different categories because of inconsistent casing/whitespace — always normalize text columns before analysis.

---

## 9.7 Feature Engineering

Creating new features from existing data to improve model performance.

```python
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1          # combine related columns
df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.')      # extract title from name (regex)
df["AgeGroup"] = pd.cut(df["Age"], bins=[0,12,18,60,100],
                           labels=["Child","Teen","Adult","Senior"])   # binning
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)              # boolean → int flag
df["LogIncome"] = np.log1p(df["Income"])                             # log transform (reduces skew)
```

🔑 **Interview Q: Why apply a log transform to skewed features?**
Log transforms compress large values and stretch small ones, reducing right-skewness and making the distribution more symmetric/normal — which helps algorithms (especially linear models) that assume normally distributed or homoscedastic features.

---

## 9.8 EDA Case Study Pattern (Titanic-style)

```python
df = pd.read_csv("titanic.csv")
df.info()
df.isnull().sum()                                    # Age, Cabin, Embarked have missing values

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.drop(columns=["Cabin"], inplace=True)                # too many missing → drop

sns.countplot(x="Survived", hue="Sex", data=df)           # survival by gender
sns.boxplot(x="Survived", y="Age", data=df)                 # age distribution by survival
sns.heatmap(df.corr(numeric_only=True), annot=True)           # correlation with target
```

---

## ✅ Chapter 9 Summary

- EDA = univariate → bivariate → multivariate analysis, before any modeling
- Missing data types: MCAR, MAR, MNAR — strategy depends on which applies
- Outliers detected via IQR (1.5× rule) or Z-score (>3 std devs)
- Always standardize text/categorical inconsistencies before analysis
- Feature engineering (binning, combining columns, log transforms) often improves model performance more than algorithm choice

---

[← Previous: Statistics](08_statistics.md) | [Back to Index](../README.md) | [Next: ML Foundations →](10_ml_foundations.md)

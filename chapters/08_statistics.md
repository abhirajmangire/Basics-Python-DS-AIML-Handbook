# Chapter 8 — Statistics & Probability for Data Science

[← Previous: Visualization](07_visualization.md) | [Back to Index](../README.md) | [Next: EDA & Data Cleaning →](09_eda_cleaning.md)

---

## 8.1 Descriptive vs Inferential Statistics

- **Descriptive statistics**: summarizing/describing the data you have (mean, median, spread, charts).
- **Inferential statistics**: using a sample to draw conclusions or make predictions about a larger population (hypothesis tests, confidence intervals).

🔑 **Interview Q: Give an example distinguishing descriptive vs inferential stats.**
Reporting the average age of survey respondents is descriptive. Using that sample average to estimate the average age of the entire country's population, with a margin of error, is inferential.

---

## 8.2 Measures of Central Tendency

| Measure | Definition | When to Use |
|---------|-----------|-------------|
| **Mean** | Sum of values ÷ count | Normally distributed data, no extreme outliers |
| **Median** | Middle value when sorted | Skewed data, or data with outliers |
| **Mode** | Most frequently occurring value | Categorical data, or finding the most common value |

```python
df["Age"].mean()
df["Age"].median()
df["Age"].mode()[0]
```

⚠️ **Common mistake:** Using mean on heavily skewed data (e.g., income) — a few billionaires can drag the mean far above what's "typical." Median is more robust here.

---

## 8.3 Measures of Dispersion (Spread)

| Measure | Meaning |
|---------|---------|
| **Range** | max − min |
| **Variance (σ²)** | Average squared deviation from the mean |
| **Standard Deviation (σ)** | Square root of variance — same units as the data |
| **IQR (Interquartile Range)** | Q3 − Q1, spread of the middle 50% of data |

```python
df["Age"].var()
df["Age"].std()
Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1
```

🔑 **Interview Q: Population vs Sample standard deviation — why divide by `n-1` for samples?**
Sample variance divides by `n-1` (**Bessel's correction**) rather than `n` because using the sample mean (instead of the true population mean) to compute deviations slightly underestimates variance. Dividing by `n-1` corrects this bias, giving an unbiased estimator of population variance.

---

## 8.4 Probability Basics

```
P(A) = favorable outcomes / total outcomes
P(A or B) = P(A) + P(B) - P(A and B)          # addition rule
P(A and B) = P(A) × P(B)                        # multiplication rule, IF A and B independent
P(A|B) = P(A and B) / P(B)                        # conditional probability
```

### Bayes' Theorem
```
P(A|B) = [ P(B|A) × P(A) ] / P(B)
```
- **P(A)** = Prior probability (belief before evidence)
- **P(A|B)** = Posterior probability (belief after evidence)
- **P(B|A)** = Likelihood (probability of evidence given hypothesis)
- **P(B)** = Evidence / marginal probability

💡 This is the theoretical basis for the **Naive Bayes** classifier covered in Chapter 11.

---

## 8.5 Probability Distributions

### Normal (Gaussian) Distribution
Symmetric, bell-shaped, defined by mean (μ) and standard deviation (σ). ~68% of data falls within 1σ, ~95% within 2σ, ~99.7% within 3σ (**Empirical Rule**).

```python
import numpy as np
data = np.random.normal(loc=0, scale=1, size=1000)
```

### Binomial Distribution
Models the number of successes in `n` independent yes/no trials, each with success probability `p`.
```python
np.random.binomial(n=10, p=0.5, size=1000)
```

### Poisson Distribution
Models the number of events occurring in a fixed interval of time/space, given a known average rate (λ).
```python
np.random.poisson(lam=3, size=1000)
```

---

## 8.6 Skewness & Kurtosis

- **Skewness**: measures asymmetry of a distribution.
  - Right/positive skew: long tail on the right (e.g., income)
  - Left/negative skew: long tail on the left
  - Zero skew: symmetric (like normal distribution)
- **Kurtosis**: measures "tailedness" — how much data is in the tails vs the center.

```python
df["Age"].skew()
df["Age"].kurt()
```

---

## 8.7 Correlation vs Causation

**Correlation** measures the strength and direction of a linear relationship between two variables (ranges -1 to +1). **Causation** means one variable directly causes a change in another.

🔑 **Interview Q: "Correlation does not imply causation" — give an example.**
Ice cream sales and drowning incidents are positively correlated — but ice cream doesn't cause drowning. Both increase during summer (a confounding variable: hot weather).

---

## 8.8 Hypothesis Testing

- **Null Hypothesis (H₀)**: the default assumption (no effect/no difference)
- **Alternative Hypothesis (H₁)**: what you're trying to prove (there IS an effect/difference)
- **p-value**: probability of observing the data (or more extreme) if H₀ were true
- **Significance level (α)**: typically 0.05 — the threshold below which we reject H₀

```python
from scipy import stats
t_stat, p_value = stats.ttest_ind(group1, group2)   # independent two-sample t-test

if p_value < 0.05:
    print("Reject null hypothesis — statistically significant difference")
else:
    print("Fail to reject null hypothesis")
```

🔑 **Interview Q: What is a Type I vs Type II error?**
- **Type I error (α)**: rejecting H₀ when it's actually true (false positive)
- **Type II error (β)**: failing to reject H₀ when it's actually false (false negative)

### Common Statistical Tests

| Test | Use Case |
|------|----------|
| **t-test** | Compare means of two groups |
| **ANOVA** | Compare means across 3+ groups |
| **Chi-Square Test** | Test relationship between two categorical variables |
| **Correlation Test** | Test significance of a correlation coefficient |

---

## ✅ Chapter 8 Summary

- Descriptive stats summarize known data; inferential stats generalize from a sample to a population
- Mean is sensitive to outliers; median is robust
- Standard deviation measures spread in the data's original units
- Bayes' Theorem underlies Naive Bayes classification
- Hypothesis testing uses p-values against a significance threshold (commonly 0.05) to decide statistical significance
- Correlation ≠ causation — always consider confounding variables

---

[← Previous: Visualization](07_visualization.md) | [Back to Index](../README.md) | [Next: EDA & Data Cleaning →](09_eda_cleaning.md)

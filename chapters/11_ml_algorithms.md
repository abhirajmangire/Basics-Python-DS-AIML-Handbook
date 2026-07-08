# Chapter 11 — Machine Learning Algorithms Deep-Dive

[← Previous: ML Foundations](10_ml_foundations.md) | [Back to Index](../README.md) | [Next: Model Evaluation →](12_model_evaluation.md)

---

## 11.1 Linear Regression

Predicts a continuous target as a **linear combination** of input features.

```
y = b0 + b1*x1 + b2*x2 + ... + bn*xn
```

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
model.predict(X_test)
model.coef_, model.intercept_
```

- Trained by minimizing **Mean Squared Error (MSE)** using **Ordinary Least Squares** (or gradient descent for large data).
- **Assumptions**: linearity, independence of errors, homoscedasticity (constant variance of errors), normally distributed residuals, no multicollinearity.

🔑 **Interview Q: What is multicollinearity and why is it a problem?**
Multicollinearity occurs when independent variables are highly correlated with each other. It inflates the variance of coefficient estimates, making them unstable and hard to interpret (though it doesn't necessarily hurt pure prediction accuracy). Detected via **VIF (Variance Inflation Factor)**.

### Regularization: Ridge & Lasso
```python
from sklearn.linear_model import Ridge, Lasso
Ridge(alpha=1.0)     # L2 penalty — shrinks coefficients, keeps all features
Lasso(alpha=1.0)      # L1 penalty — can shrink coefficients to exactly 0 (feature selection)
```

---

## 11.2 Logistic Regression

Despite the name, this is a **classification** algorithm — predicts probability of belonging to a class using the **sigmoid function**.

```
sigmoid(z) = 1 / (1 + e^(-z))          # squashes any real number into (0, 1)
```

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)                # class labels
model.predict_proba(X_test)            # class probabilities
```

- Uses **log loss (cross-entropy)** as its cost function, not MSE.
- Decision boundary is linear; a threshold (default 0.5) converts probability → class.

🔑 **Interview Q: Why not use Linear Regression for classification?**
Linear Regression outputs unbounded continuous values, which don't correspond meaningfully to probabilities, and it's very sensitive to outliers shifting the decision threshold. Logistic Regression's sigmoid function bounds outputs to (0,1), directly modeling class probability.

---

## 11.3 Decision Tree

A tree-like model that splits data based on feature values, forming a hierarchy of decisions leading to a prediction at each leaf.

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, criterion="gini")
model.fit(X_train, y_train)
```

### Key Concepts
- **Root Node**: topmost node, represents the entire dataset before any split
- **Leaf Node**: terminal node, gives the final prediction
- **Splitting**: dividing a node into two+ sub-nodes based on a feature threshold
- **Entropy**: measure of impurity/disorder in a node
  ```
  Entropy = - Σ p(i) * log2(p(i))
  ```
  Entropy = 0 means perfectly pure (all one class); Entropy = 1 means maximum impurity (50/50 split for binary).
- **Information Gain**: reduction in entropy after a split — the tree picks the split that **maximizes** information gain.
  ```
  Information Gain = Entropy(parent) - WeightedAverage(Entropy(children))
  ```
- **Gini Index**: alternative impurity measure (faster to compute than entropy, used by default in scikit-learn's CART implementation)
  ```
  Gini = 1 - Σ p(i)²
  ```

🔑 **Interview Q: Gini vs Entropy — does it matter which you use?**
Both measure impurity and usually produce similar trees. Gini is computationally cheaper (no logarithm) and is scikit-learn's default. Entropy is more sensitive to changes in class probabilities in some edge cases. In practice, the choice rarely changes results dramatically.

### Overfitting in Decision Trees
Decision trees easily overfit by growing deep enough to perfectly memorize training data.

**Pruning** — limiting tree growth to improve generalization:
- **Pre-pruning**: set constraints upfront — `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Post-pruning**: grow the full tree, then remove branches that don't improve validation performance (cost-complexity pruning, `ccp_alpha`)

**Advantages**: easy to interpret/visualize, handles both numeric & categorical data, no feature scaling needed.
**Disadvantages**: prone to overfitting, unstable (small data changes can produce very different trees), biased toward features with more levels.

---

## 11.4 Random Forest

An **ensemble** of many Decision Trees, each trained on a random subset of data (bagging) and a random subset of features per split, with the final prediction being a majority vote (classification) or average (regression).

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)
model.feature_importances_          # relative importance of each feature
```

💡 **Why it beats a single Decision Tree**: averaging across many de-correlated trees reduces variance dramatically, making it far more robust to overfitting while retaining most of the interpretability benefits (via feature importance).

🔑 **Interview Q: What is "bagging"?**
**B**ootstrap **Agg**regat**ing** — training multiple models on different bootstrap samples (random samples **with replacement**) of the training data, then combining their predictions (vote/average) to reduce variance.

---

## 11.5 Support Vector Machine (SVM)

Finds the **hyperplane** that best separates classes by **maximizing the margin** (distance) between the hyperplane and the nearest data points from each class (**support vectors**).

```python
from sklearn.svm import SVC
model = SVC(kernel="rbf", C=1.0)
model.fit(X_train, y_train)
```

- **Kernel trick**: transforms data into a higher-dimensional space to make it linearly separable, without explicitly computing that transformation (`linear`, `poly`, `rbf`, `sigmoid` kernels).
- **`C` parameter**: controls the tradeoff between maximizing the margin and minimizing classification error — small `C` = wider margin, more tolerance for misclassification (more regularization); large `C` = narrower margin, less tolerance.

---

## 11.6 K-Nearest Neighbors (KNN)

A **lazy, instance-based** algorithm — makes no assumptions about data distribution and does no explicit "training." To classify a new point, it finds the `k` closest training points (by distance metric, usually Euclidean) and takes a majority vote (classification) or average (regression).

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

⚠️ **Common mistakes**: 
- KNN is **very sensitive to feature scale** — always scale features first.
- Choosing `k` too small → sensitive to noise (overfitting); too large → oversmoothed, ignores local structure (underfitting). `k` is usually chosen via cross-validation, and odd values avoid ties in binary classification.

---

## 11.7 Naive Bayes

A probabilistic classifier based on **Bayes' Theorem**, with the "naive" assumption that all features are **conditionally independent** given the class.

```
P(class | features) ∝ P(class) × ∏ P(feature_i | class)
```

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
```

| Variant | Use Case |
|---------|----------|
| **GaussianNB** | Continuous features assumed normally distributed |
| **MultinomialNB** | Discrete counts — **most common for text classification** (word counts, TF-IDF) |
| **BernoulliNB** | Binary/boolean features (word present/absent) |

🔑 **Interview Q: Why is Multinomial Naive Bayes preferred for text classification?**
Text data is naturally represented as word frequency counts (Bag of Words / TF-IDF vectors). MultinomialNB models the distribution of these counts directly, and despite the strong independence assumption being unrealistic for language, it works remarkably well in practice for text due to the high dimensionality diluting the effect of correlated features, and it's extremely fast to train even on large vocabularies.

---

## 11.8 K-Means Clustering (Unsupervised)

Partitions data into `k` clusters by iteratively assigning points to the nearest centroid and recomputing centroids.

```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
model.labels_               # cluster assignment for each point
model.cluster_centers_        # coordinates of each centroid
```

**Algorithm steps:**
1. Randomly initialize `k` centroids
2. Assign each point to its nearest centroid
3. Recompute each centroid as the mean of its assigned points
4. Repeat steps 2-3 until centroids stop moving (convergence)

**Choosing `k` — The Elbow Method:**
```python
inertias = [KMeans(n_clusters=k).fit(X).inertia_ for k in range(1, 10)]
# plot k vs inertia — pick the "elbow" where the rate of improvement sharply slows
```

---

## 11.9 Ensemble Boosting (Gradient Boosting, XGBoost)

Unlike bagging (parallel, independent trees), **boosting** builds trees **sequentially**, where each new tree focuses on correcting the errors (residuals) of the previous ones.

```python
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)
```

🔑 **Interview Q: Bagging vs Boosting?**
**Bagging** (e.g., Random Forest) trains models **independently/in parallel** on bootstrapped samples to reduce **variance**. **Boosting** (e.g., XGBoost, AdaBoost, Gradient Boosting) trains models **sequentially**, each new model correcting the previous model's errors, primarily reducing **bias** while also lowering variance — often achieving higher accuracy but with greater risk of overfitting and longer training time.

---

## 11.10 Algorithm Cheat Table

| Algorithm | Type | Key Idea | Sensitive to Scaling? |
|-----------|------|----------|----------------------|
| Linear Regression | Regression | Best-fit line minimizing MSE | Yes |
| Logistic Regression | Classification | Sigmoid → probability | Yes |
| Decision Tree | Both | Recursive splits by impurity | No |
| Random Forest | Both | Bagged ensemble of trees | No |
| SVM | Both | Max-margin hyperplane | Yes |
| KNN | Both | Vote among k nearest neighbors | Yes (very) |
| Naive Bayes | Classification | Bayes' theorem + independence assumption | No |
| K-Means | Clustering | Iterative centroid assignment | Yes |
| XGBoost/Gradient Boosting | Both | Sequential error-correcting trees | No |

---

## ✅ Chapter 11 Summary

- Linear Regression (continuous target) vs Logistic Regression (probability via sigmoid, for classification)
- Decision Trees split on Entropy/Gini-based Information Gain; prone to overfitting without pruning
- Random Forest = bagged ensemble of trees → reduces variance
- SVM maximizes margin between classes; kernel trick handles non-linear separability
- KNN is instance-based, scale-sensitive, requires choosing `k` carefully
- Naive Bayes applies Bayes' Theorem with a (naive) feature-independence assumption — great for text
- K-Means clusters via iterative centroid updates; elbow method picks `k`
- Boosting (sequential, bias-reducing) vs Bagging (parallel, variance-reducing)

---

[← Previous: ML Foundations](10_ml_foundations.md) | [Back to Index](../README.md) | [Next: Model Evaluation →](12_model_evaluation.md)

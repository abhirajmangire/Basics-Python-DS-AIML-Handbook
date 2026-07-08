# Chapter 15 — Interview Question Bank

[← Previous: NLP](14_nlp.md) | [Back to Index](../README.md) | [Next: Cheat Sheets →](16_cheatsheets.md)

---

*Questions are grouped by topic and level. Use these for internship interviews, college viva, and placement prep. Try answering before checking the answer.*

---

## 🟢 Python — Beginner

1. **What is Python and what type of language is it?** → High-level, interpreted, dynamically-typed, general-purpose language.
2. **Difference between list and tuple?** → List is mutable, tuple is immutable.
3. **What is indentation used for in Python?** → Defines code blocks instead of braces.
4. **What is a variable?** → A name bound to a value/object in memory.
5. **Difference between `==` and `is`?** → `==` compares values, `is` compares object identity.
6. **What are Python's built-in data types?** → int, float, str, bool, list, tuple, set, dict, complex, NoneType.
7. **What is type casting?** → Explicit conversion between data types (`int()`, `str()`, `float()`).
8. **What is the difference between `append()` and `extend()`?** → `append()` adds one element (even a list, as a single nested item); `extend()` unpacks and adds each element individually.
9. **What does `len()` do?** → Returns the number of items in a sequence/collection.
10. **What is a dictionary?** → A mutable collection of key-value pairs.
11. **What is slicing?** → Extracting a sub-sequence via `[start:stop:step]`.
12. **Difference between `range()` and `list`?** → `range()` is a lazy sequence generator; a list stores all elements in memory.
13. **What is the purpose of `self` in a class method?** → Refers to the current instance, giving access to its attributes/methods.
14. **What is PEP 8?** → Python's official style guide for writing readable code.
15. **How do you handle exceptions in Python?** → `try/except/else/finally` blocks.

## 🟡 Python — Intermediate

16. **What are `*args` and `**kwargs`?** → Collect variable numbers of positional/keyword arguments respectively.
17. **What is a lambda function?** → An anonymous, single-expression function.
18. **Difference between shallow copy and deep copy?** → Shallow copy copies the outer object but shares nested objects; deep copy fully duplicates all nested objects independently.
19. **What is list comprehension? Give an example.** → Concise syntax to build lists: `[x**2 for x in range(10)]`.
20. **What is the difference between `is` and `==` for strings?** → `==` checks value equality; `is` checks whether they're the same object in memory (small strings may be interned, causing surprising `is` results).
21. **What are Python decorators?** → Functions that wrap other functions to extend behavior without modifying their code.
22. **Explain the LEGB rule.** → Variable scope resolution order: Local → Enclosing → Global → Built-in.
23. **What is a generator? How is it different from a normal function?** → Uses `yield` to lazily produce values one at a time, pausing/resuming state, instead of returning all values at once.
24. **What does the `with` statement do?** → Manages context (like file handles) automatically via `__enter__`/`__exit__`, ensuring cleanup even if exceptions occur.
25. **What is the difference between `.py` file and a module?** → A module IS a `.py` file when imported; the terms are essentially synonymous once imported.
26. **How do you remove duplicates from a list?** → `list(set(my_list))` (loses order) or a dict-based approach to preserve order.
27. **What is string immutability, and why does it matter?** → Strings cannot be changed after creation; any modification creates a new string object — impacts performance in loops with heavy concatenation (use `"".join()` instead).
28. **Explain `try/except/else/finally`.** → `try`: code that might fail; `except`: handles specific errors; `else`: runs if no exception occurred; `finally`: always runs, used for cleanup.

## 🔴 Python — Advanced

29. **What is the GIL and how does it affect multithreading?** → Global Interpreter Lock allows only one thread to execute Python bytecode at a time, limiting true parallelism for CPU-bound tasks in threads (use multiprocessing instead).
30. **Explain closures with an example.** → A function that captures and remembers variables from its enclosing scope even after that scope exits.
31. **What is a metaclass?** → A "class of a class" — defines how classes themselves are constructed (advanced OOP concept, `type` is the default metaclass).
32. **Difference between `@staticmethod` and `@classmethod`?** → Staticmethod takes no implicit argument; classmethod takes `cls`, often used for alternate constructors.
33. **What is name mangling?** → Double-underscore prefixed attributes (`__x`) are internally renamed (`_ClassName__x`) to avoid subclass naming collisions.
34. **How does Python manage memory?** → Reference counting plus a cyclic garbage collector to handle reference cycles.
35. **What's the difference between multiprocessing and multithreading?** → Multiprocessing runs separate processes with independent memory (true parallelism, bypasses GIL); multithreading shares memory within one process (limited by GIL for CPU-bound work, useful for I/O-bound work).

---

## 🟢 Data Science / Pandas / NumPy

36. **Why is NumPy faster than Python lists?** → Contiguous memory, homogeneous types, vectorized C-level operations.
37. **What is broadcasting in NumPy?** → Performing operations on arrays of different shapes by implicitly expanding the smaller one.
38. **Difference between `.loc` and `.iloc`?** → `.loc` is label-based (inclusive slicing), `.iloc` is position-based (exclusive slicing).
39. **How do you handle missing data in pandas?** → `isnull()`, `dropna()`, `fillna()` with mean/median/mode or forward/backward fill.
40. **What does `groupby()` do?** → Implements split-apply-combine: groups rows by a key, applies an aggregation, combines results.
41. **Difference between `merge()` and `concat()`?** → `merge()` does SQL-style joins on keys; `concat()` stacks DataFrames along rows or columns.
42. **What is the difference between `Series` and `DataFrame`?** → Series is 1D labeled array; DataFrame is a 2D table of Series sharing an index.
43. **How would you find and remove duplicate rows?** → `df.duplicated()` to detect, `df.drop_duplicates()` to remove.
44. **What does `value_counts()` do?** → Returns frequency counts of unique values in a Series.
45. **Explain mean vs median imputation.** → Mean for roughly symmetric distributions without outliers; median for skewed data or data with outliers (more robust).
46. **What is `SettingWithCopyWarning` and how do you avoid it?** → Warns about ambiguous chained assignment on a DataFrame slice; avoid by using `.loc[]` explicitly for assignment.

## 🟡 Statistics

47. **Mean vs Median vs Mode — when to use each?** → Mean for symmetric data; median for skewed/outlier-prone data; mode for categorical data.
48. **What is standard deviation?** → Square root of variance; measures spread in the same units as the original data.
49. **Explain the Central Limit Theorem.** → The sampling distribution of the sample mean approaches a normal distribution as sample size increases, regardless of the population's original distribution.
50. **What is a p-value?** → Probability of observing data at least as extreme as what was observed, assuming the null hypothesis is true.
51. **Type I vs Type II error?** → Type I: false positive (rejecting a true null); Type II: false negative (failing to reject a false null).
52. **What is correlation? Does it imply causation?** → Measures linear relationship strength (-1 to +1); does NOT imply causation (confounding variables may exist).
53. **Explain Bayes' Theorem.** → `P(A|B) = P(B|A) × P(A) / P(B)` — updates prior belief P(A) into posterior belief P(A|B) given evidence B.

## 🟡 EDA

54. **What is EDA and why is it important?** → Exploratory analysis to understand data structure/quality/patterns before modeling — prevents "garbage in, garbage out."
55. **Difference between univariate, bivariate, and multivariate analysis?** → One variable, two variables, three+ variables respectively.
56. **How do you detect outliers?** → IQR method (1.5×IQR beyond Q1/Q3) or Z-score method (>3 standard deviations).
57. **What is feature engineering? Give an example.** → Creating new, more predictive features from raw data, e.g., combining SibSp+Parch into "FamilySize."

---

## 🔴 Machine Learning

58. **Supervised vs Unsupervised Learning?** → Supervised uses labeled data (classification/regression); unsupervised finds structure in unlabeled data (clustering, dimensionality reduction).
59. **What is overfitting? How do you prevent it?** → Model memorizes training data/noise, performs poorly on new data; prevent via more data, regularization, simpler models, cross-validation, early stopping.
60. **Explain bias-variance tradeoff.** → Bias = error from oversimplification (underfitting); Variance = error from over-sensitivity to training data (overfitting); total error = bias² + variance + irreducible error.
61. **Why do we split data into train/test sets?** → To evaluate generalization to unseen data rather than memorization.
62. **What is cross-validation?** → Splitting data into k folds, training/validating across rotations, to get a robust performance estimate.
63. **Explain entropy and information gain in Decision Trees.** → Entropy measures node impurity; Information Gain measures the entropy reduction achieved by a split — trees pick splits maximizing this gain.
64. **Gini Index vs Entropy?** → Both measure impurity; Gini is computationally cheaper (default in scikit-learn), Entropy is log-based.
65. **What is pruning in Decision Trees?** → Limiting tree growth (pre-pruning via max_depth, etc., or post-pruning via cost-complexity) to reduce overfitting.
66. **Explain Random Forest and why it outperforms a single Decision Tree.** → Ensemble of trees trained on bootstrap samples with random feature subsets; averaging reduces variance/overfitting.
67. **Bagging vs Boosting?** → Bagging trains models in parallel on bootstrap samples (reduces variance); Boosting trains sequentially, correcting prior errors (reduces bias).
68. **What is the kernel trick in SVM?** → Implicitly maps data to higher dimensions to find a linear separator, without explicitly computing the transformation.
69. **How does KNN work? What are its weaknesses?** → Classifies based on majority vote among k nearest neighbors; weaknesses: computationally expensive at prediction time, sensitive to feature scale and irrelevant features.
70. **Why use Naive Bayes for text classification?** → Fast, works well with high-dimensional sparse data (word counts/TF-IDF), despite the unrealistic independence assumption.
71. **Explain the Elbow Method for K-Means.** → Plot inertia vs number of clusters `k`; pick the `k` where the improvement rate sharply diminishes ("elbow" point).
72. **What is data leakage?** → When information from outside the training set (e.g., test data, future data) improperly influences training, causing over-optimistic evaluation.
73. **Why must you fit scalers only on training data?** → To prevent information from the test set leaking into the training process, which would bias evaluation.

## 🔴 Model Evaluation

74. **Precision vs Recall — give a real-world example of prioritizing each.** → Precision matters for spam filters (avoid flagging real emails); Recall matters for cancer/fraud detection (avoid missing true positives).
75. **Why not rely on accuracy for imbalanced datasets?** → A trivial "always predict majority class" model can achieve high accuracy while being useless.
76. **What is ROC-AUC?** → Plots True Positive Rate vs False Positive Rate across thresholds; AUC summarizes overall ranking quality (0.5 = random, 1.0 = perfect).
77. **F1 Score — why harmonic mean?** → Punishes imbalance between precision and recall more than a simple average would, giving a more honest single combined metric.
78. **MAE vs RMSE?** → MAE treats errors linearly (robust to outliers); RMSE squares errors, penalizing large errors more heavily.

## 🔴 Deep Learning & NLP

79. **Why is ReLU preferred over Sigmoid in hidden layers?** → Avoids vanishing gradients since its gradient doesn't saturate for positive inputs, enabling faster/deeper training.
80. **What is backpropagation?** → Algorithm that computes gradients of the loss with respect to every weight via the chain rule, propagating error backward to update weights.
81. **What is dropout and why is it used?** → Randomly deactivates neurons during training to prevent overfitting/co-adaptation.
82. **CNN vs RNN — when do you use each?** → CNNs for spatial/grid data (images); RNNs/LSTMs for sequential data (text, time series).
83. **How do LSTMs solve the vanishing gradient problem in RNNs?** → Gating mechanisms (input/forget/output gates) and a separate cell state let information persist over long sequences.
84. **Stemming vs Lemmatization?** → Stemming is fast, rule-based, may produce non-words; Lemmatization uses vocabulary/POS tagging to produce real dictionary words.
85. **BoW vs TF-IDF?** → BoW uses raw counts; TF-IDF down-weights corpus-common words and up-weights document-distinctive words.
86. **What is the attention mechanism in Transformers?** → Lets the model weigh the relevance of every word to every other word simultaneously, capturing long-range context better than sequential RNN processing.

---

## 💼 HR / Behavioral Questions (Internship & Placement)

87. Tell me about yourself and your interest in Data Science.
88. Walk me through a project from your resume/GitHub — what was the goal, your approach, and the outcome?
89. What was the most challenging bug/problem you faced in this project, and how did you solve it?
90. Why did you choose this dataset/algorithm for this project?
91. How do you stay updated with new developments in ML/AI?
92. Describe a time you had to learn a new tool/library quickly.
93. What would you improve if you revisited this project today?
94. Where do you see yourself in the Data Science field in 5 years?

---

## 🎓 Viva / Internship Presentation Questions

95. Explain your Titanic EDA project end-to-end — what did you find in the data?
96. Why did you fill missing Age values with median instead of mean?
97. Why did you drop the Cabin column?
98. Explain the Decision Tree you built — what was its depth, and why did you choose that?
99. What accuracy did your model achieve, and how do you know if that's actually good?
100. Explain your text preprocessing steps for the sentiment analysis project, in order.
101. Why did you choose TF-IDF/Naive Bayes for the sentiment analysis task specifically?
102. What would happen if you didn't remove stopwords from your text data?
103. What's the difference between your training accuracy and test accuracy — does it suggest overfitting?

---

[← Previous: NLP](14_nlp.md) | [Back to Index](../README.md) | [Next: Cheat Sheets →](16_cheatsheets.md)

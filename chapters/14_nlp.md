# Chapter 14 — NLP Complete

[← Previous: Deep Learning](13_deep_learning.md) | [Back to Index](../README.md) | [Next: Interview Questions →](15_interview_questions.md)

---

## 14.1 What is NLP?

**Natural Language Processing (NLP)** is the field focused on enabling computers to understand, interpret, and generate human language — combining linguistics with ML/DL techniques. Applications: sentiment analysis, machine translation, chatbots, text summarization, spam detection.

---

## 14.2 Text Preprocessing Pipeline

Raw text is messy and must be cleaned/standardized before feeding into models.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "The children were running quickly through the Park!!"

text = text.lower()                                    # lowercase
text = re.sub(r'[^a-zA-Z\s]', '', text)                    # remove punctuation/numbers via regex
tokens = text.split()                                         # tokenization (simple whitespace split)
```

### Tokenization
Splitting text into smaller units (words, subwords, or sentences).
```python
from nltk.tokenize import word_tokenize, sent_tokenize
word_tokenize("Data Science is amazing!")     # ['Data', 'Science', 'is', 'amazing', '!']
```

### Stopword Removal
Removing common words that carry little semantic meaning ("the", "is", "and", "a").
```python
stop_words = set(stopwords.words("english"))
filtered = [w for w in tokens if w not in stop_words]
```
💡 **Why remove stopwords?** They appear extremely frequently across all documents/classes, adding noise rather than discriminative signal for most classification tasks (though they matter for tasks like authorship style or grammar analysis).

### Stemming vs Lemmatization

| | Stemming | Lemmatization |
|---|----------|---------------|
| **Approach** | Crude rule-based chopping of word endings | Dictionary-based, considers part-of-speech |
| **Speed** | Faster | Slower |
| **Accuracy** | Can produce non-words ("studies" → "studi") | Produces real dictionary words ("studies" → "study") |
| **Example** | `PorterStemmer().stem("running")` → "run" | `WordNetLemmatizer().lemmatize("running", pos="v")` → "run" |

```python
stemmer = PorterStemmer()
stemmer.stem("studies")            # "studi"  (not a real word!)

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("studies")     # "study"  (real word, more accurate)
```

🔑 **Interview Q: When would you choose stemming over lemmatization?**
Stemming is faster and sufficient when speed matters more than precision (e.g., search engine indexing where exact word forms matter less). Lemmatization is preferred when interpretability and grammatical correctness matter, such as detailed text analysis or when word meaning must be preserved precisely.

### Regex in Text Cleaning
```python
re.sub(r'http\S+', '', text)          # remove URLs
re.sub(r'@\w+', '', text)               # remove mentions (@username)
re.sub(r'#\w+', '', text)                 # remove hashtags
re.sub(r'\d+', '', text)                    # remove digits
```

---

## 14.3 Text Representation (Feature Extraction)

Machine learning models need numeric input — text must be converted into numeric vectors.

### Bag of Words (BoW)
Represents text as a vector of word **counts**, ignoring grammar and word order.
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(["I love data science", "I love NLP"])
vectorizer.get_feature_names_out()   # vocabulary
X.toarray()                            # count matrix
```

### TF-IDF (Term Frequency - Inverse Document Frequency)
Weighs words by how important/distinctive they are to a document **relative to the whole corpus** — common words across all documents get down-weighted.

```
TF(t, d) = (count of term t in document d) / (total terms in d)
IDF(t) = log( total documents / documents containing t )
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus)
```

- **Why stop words are removed**: they'd otherwise get inflated importance from sheer frequency, despite carrying no discriminative meaning; removing them focuses the vector representation on genuinely informative words.
- **Internal working**: words that appear frequently in one document but rarely across the whole corpus get **high TF-IDF scores** (highly distinctive); words appearing everywhere (like "the") get **low scores** because IDF shrinks toward zero as document frequency approaches the total document count.

🔑 **Interview Q: BoW vs TF-IDF — what's the key difference?**
BoW weighs every word purely by raw frequency within a document, treating common and rare words equally important if they occur the same number of times. TF-IDF additionally down-weights words that are common **across the entire corpus** (like "the," "is") and up-weights words that are distinctive to a specific document, generally producing more informative features for classification.

### N-grams
Sequences of `n` consecutive words/tokens, capturing some local word order/context that BoW/unigrams miss.
```python
CountVectorizer(ngram_range=(1, 2))    # unigrams + bigrams: "data", "science", "data science"
```

### Word Embeddings (Word2Vec, GloVe)
Dense vector representations where semantically similar words are located close together in vector space (unlike BoW/TF-IDF's sparse, meaning-agnostic vectors).
```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
model.wv.most_similar("king")     # finds semantically similar words
```

💡 Famous property: `vector("king") - vector("man") + vector("woman") ≈ vector("queen")` — embeddings capture semantic relationships as geometric relationships.

---

## 14.4 Naive Bayes for Text Classification (Recap)

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])
pipeline.fit(X_train_text, y_train)
pipeline.predict(X_test_text)
```

See Chapter 11.7 for the full Naive Bayes / Bayes' Theorem explanation.

---

## 14.5 Sentiment Analysis Workflow (Complete Pattern)

```python
# 1. Load & clean data
df["clean_text"] = df["review"].apply(clean_text_function)

# 2. Feature extraction
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 4. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluate
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))
```

---

## 14.6 Transformers — Brief Introduction

Modern NLP (BERT, GPT, etc.) is built on the **Transformer architecture**, which uses a mechanism called **self-attention** to weigh the relevance of every word in a sequence to every other word simultaneously — unlike RNNs, which process sequentially and struggle with long-range dependencies.

**Key ideas:**
- **Self-attention**: each word's representation is updated based on a weighted combination of all other words in the sentence, allowing the model to capture context regardless of distance.
- **Pre-training + fine-tuning**: models like BERT are pre-trained on massive text corpora (learning general language patterns), then **fine-tuned** on smaller task-specific datasets (sentiment analysis, Q&A, etc.).
- Transformers process sequences **in parallel** (not step-by-step like RNNs), making them far more efficient to train at scale.

---

## ✅ Chapter 14 Summary

- Preprocessing pipeline: lowercase → clean (regex) → tokenize → remove stopwords → stem/lemmatize
- Stemming = fast, crude; Lemmatization = slower, produces real dictionary words
- BoW = raw word counts; TF-IDF = frequency weighted by corpus-wide distinctiveness
- N-grams capture local word order; Word Embeddings (Word2Vec) capture semantic meaning geometrically
- MultinomialNB + TF-IDF is a strong, fast baseline for text classification
- Transformers (BERT/GPT) use self-attention for parallelized, context-aware language understanding

---

[← Previous: Deep Learning](13_deep_learning.md) | [Back to Index](../README.md) | [Next: Interview Questions →](15_interview_questions.md)

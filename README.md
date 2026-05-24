# 🧠 Sentiment Analysis — Tweet Classification

A complete end-to-end NLP pipeline for classifying tweets into **Positive**, **Negative**, and **Neutral** sentiments. Covers data cleaning, exploratory data analysis, text preprocessing, and multi-model comparison using classical ML algorithms.

---

## 📁 Dataset

The project uses `sentiment_analysis.csv`, a tweet dataset containing:

- `text` — raw tweet content
- `sentiment` — label: `positive`, `negative`, or `neutral`
- `Year`, `Month`, `Day` — temporal metadata
- `Time of Tweet` — time-of-day category
- `Platform` — platform the tweet was posted from

---

## 🔄 Pipeline Overview

```
Raw CSV
   ↓
Data Cleaning (dedup, nulls)
   ↓
EDA (distributions, trends, correlations)
   ↓
Text Preprocessing (lowercase → tokenize → remove stopwords → stem)
   ↓
Feature Engineering (CountVectorizer + TF-IDF)
   ↓
Model Training & Evaluation (8 models compared)
```

---

## ✨ Key Steps

### 1. Data Cleaning
- Removed duplicate rows
- Checked and handled null values

### 2. Exploratory Data Analysis (EDA)
- Sentiment distribution (pie chart + count plot)
- Temporal trends — sentiment over year, month, and day
- Platform and time-of-day breakdowns
- Text statistics: character count, word count, sentence count per sentiment class
- Correlation heatmap

### 3. Text Preprocessing
Each tweet goes through the following steps:
- Lowercasing
- Tokenization (NLTK)
- Removal of special characters and non-alphanumeric tokens
- Stopword and punctuation removal
- Porter Stemming

### 4. Visualization (Post-preprocessing)
- Word clouds for each sentiment class
- Top-30 most frequent words per sentiment (positive, negative, neutral)

### 5. Model Building
Features vectorized using **CountVectorizer** (and **TF-IDF** for XGBoost). Models trained and evaluated on an 80/20 train-test split:

| Model | Vectorizer |
|---|---|
| Gaussian Naive Bayes | CountVectorizer/TF-IDF |
| Multinomial Naive Bayes | CountVectorizer/TF-IDF |
| Bernoulli Naive Bayes | CountVectorizer/TF-IDF |
| Logistic Regression | CountVectorizer/TF-IDF |
| Support Vector Machine (Linear) | CountVectorizer/TF-IDF |
| Random Forest | CountVectorizer/TF-IDF |
| Gradient Boosting | CountVectorizer/TF-IDF |
| XGBoost | CountVectorizer/TF-IDF |

Each model is evaluated on **Accuracy**, **Precision**, **Recall**, and **F1 Score** (weighted average), along with a **Confusion Matrix**.

---

## 📦 Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud xgboost
```

Also download the required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 🚀 How to Run

1. Clone the repository and navigate to the project folder.
2. Place `sentiment_analysis.csv` in the same directory as the notebook.
3. Install dependencies (see above).
4. Open and run `sentiment_analysis.ipynb` cell by cell.

---

## 🗂️ Project Structure

```
sentiment-analysis/
│
├── sentiment_analysis.ipynb   ← Main notebook
├── sentiment_analysis.csv     ← Dataset (not included, add your own)
└── README.md
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualization |
| `nltk` | Tokenization, stopwords, stemming |
| `wordcloud` | Word cloud generation |
| `scikit-learn` | Feature extraction, model training, evaluation |
| `xgboost` | Gradient boosted tree classifier |

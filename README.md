# 🧠 Sentiment Analysis using Machine Learning

## 📌 Overview

This project performs **Sentiment Analysis** on text data using multiple Machine Learning algorithms. The notebook includes a complete NLP pipeline from raw data to model evaluation.

### What's Covered
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Text Preprocessing
- Feature Engineering
- Data Visualization
- Model Building
- Model Evaluation
- Performance Comparison

### Goal
Classify text into three sentiment categories:
- ✅ **Positive** Sentiment
- ❌ **Negative** Sentiment
- ➖ **Neutral** Sentiment

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **NLP** | NLTK |
| **Machine Learning** | Scikit-learn, XGBoost |

---

## 🔄 Project Workflow

### 1. Data Loading
```python
import pandas as pd
df = pd.read_csv('sentiment_analysis.csv')
```

### 2. Data Cleaning
- Checked dataset information
- Checked null values
- Checked and removed duplicate records

```python
df.drop_duplicates(keep='first', inplace=True)
```

### 3. Exploratory Data Analysis (EDA)

**Visualizations Included:**
- Sentiment distribution pie chart
- Count plot of sentiments
- Year-wise sentiment analysis
- Monthly sentiment trends
- Daily sentiment trends
- Sentiment distribution by platform
- Time-based sentiment analysis

**Text Statistics Created:**
```python
df['num_characters']   # Number of characters
df['num_words']        # Number of words
df['num_sentences']    # Number of sentences
```

A **Correlation Heatmap** was also generated to visualize feature relationships.

### 4. Text Preprocessing
Steps applied to clean raw text:
1. Lowercasing
2. Tokenization
3. Removing special characters
4. Removing stopwords
5. Stemming

Transformed text is stored in the `transformed_text` column.

### 5. WordCloud Visualization
Word clouds were generated separately for each sentiment class to identify the most frequent words:
- ☁️ Positive tweets
- ☁️ Negative tweets
- ☁️ Neutral tweets

### 6. Feature Extraction
Text data was converted into numerical format using two techniques:

**Count Vectorizer**
```python
from sklearn.feature_extraction.text import CountVectorizer
```

**TF-IDF Vectorization** *(used for XGBoost model)*

### 7. Train-Test Split
```python
from sklearn.model_selection import train_test_split
# 80% Training | 20% Testing
```

---

## 🤖 Machine Learning Models Used

| Model | Description |
|---|---|
| Gaussian Naive Bayes | Probabilistic classifier |
| Multinomial Naive Bayes | Best suited for text frequency data |
| Bernoulli Naive Bayes | Works with binary features |
| Logistic Regression | Linear classification algorithm |
| Support Vector Machine (SVM) | Margin-based classifier |
| Random Forest | Ensemble tree-based model |
| Gradient Boosting | Boosting ensemble method |
| XGBoost | Advanced gradient boosting algorithm |

---

## 📊 Model Evaluation

**Metrics Used:**
```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
```

### Performance Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Gaussian Naive Bayes | 0.430 | 0.461 | 0.430 | 0.413 |
| Multinomial Naive Bayes | 0.532 | 0.550 | 0.532 | 0.533 |
| Bernoulli Naive Bayes | 0.506 | 0.637 | 0.506 | 0.437 |
| Logistic Regression | **0.595** | **0.613** | **0.595** | **0.583** |
| Support Vector Machine | 0.570 | 0.581 | 0.570 | 0.563 |
| Random Forest | 0.582 | 0.618 | 0.582 | 0.550 |
| Gradient Boosting | 0.544 | 0.564 | 0.544 | 0.510 |
| XGBoost | 0.481 | 0.469 | 0.481 | 0.469 |

### 🏆 Best Performing Model: Logistic Regression

**Reasons:**
- ✔️ Highest Accuracy
- ✔️ Highest F1 Score
- ✔️ Balanced Precision and Recall
- ✔️ Performs well on sparse text data
- ✔️ Efficient for NLP classification tasks

---

## 📁 Folder Structure

```
project-folder/
│
├── sentiment_analysis.ipynb
├── sentiment_analysis.csv
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### Clone Repository
```bash
git clone <your-repository-link>
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements
```
numpy
pandas
matplotlib
seaborn
nltk
wordcloud
scikit-learn
xgboost
```

---

## ▶️ How to Run

```bash
jupyter notebook
```
Then open `sentiment_analysis.ipynb`.

---

## 🚀 Future Improvements

- Hyperparameter tuning
- Deep Learning models (LSTM, GRU)
- Transformer models (BERT)
- Larger dataset
- Real-time sentiment prediction web app
- Model deployment using Streamlit or Flask

---

## 📚 Learning Outcomes

Through this project you can learn:
- Natural Language Processing basics
- Text preprocessing techniques
- Feature extraction methods
- Machine Learning for NLP
- Model evaluation techniques
- Data visualization

---

## ✅ Conclusion

This project demonstrates a complete NLP pipeline for sentiment analysis using traditional Machine Learning techniques. Multiple models were compared, and **Logistic Regression** achieved the best overall performance on the dataset.

The notebook provides a strong foundation for building more advanced NLP applications.

---

## 👤 Author

**Gaurav Sharma**

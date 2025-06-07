## Sentiment Analysis Using Machine Learning

This project focuses on building a sentiment classification model using tweets from the Sentiment140 dataset. The main goal is to accurately classify tweets as positive or negative, which has applications in social media monitoring, brand reputation tracking, and understanding public opinion.

---

### Dataset

**Source**: Kaggle (Sentiment140)
**Size**: 1.6 million tweets
**Format**: CSV

**Columns:**

* `target`: Sentiment label (0 = Negative, 4 = Positive)
* `ids`: Tweet ID
* `date`: Timestamp of the tweet
* `flag`: Query used (usually “NO\_QUERY”)
* `user`: Twitter username
* `text`: The tweet content

---

### Approach

The notebook walks through the following steps:

#### Data Loading

* The dataset is loaded using Pandas with proper encoding (`ISO-8859-1`).

#### Exploratory Data Analysis (EDA)

* Word clouds and frequency analysis are used to understand the data.

#### Text Preprocessing

* Lowercasing
* Removing URLs, mentions, punctuation, and stopwords
* Tokenization and lemmatization

#### Feature Extraction

* TF-IDF Vectorization

#### Model Training

**Models used:**

* Logistic Regression
* Linear Support Vector Classifier (SVC)
* Bernoulli Naive Bayes

#### Evaluation

* Confusion matrix
* Classification report (Precision, Recall, F1-score)

---

### Results

The trained models were evaluated using accuracy metrics and confusion matrices to compare their performance in classifying tweet sentiment.

---

### Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* NLTK
* WordCloud
* Google Colab

---

### Getting Started

To run this project locally:

1. Clone the repository
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud
```

3. Open the `Sentiment_Analysis.ipynb` in Jupyter or Google Colab.
4. Run all cells.

---

### Notes

* Ensure that the Sentiment140 dataset is downloaded and placed in the correct directory.
* Encoding is critical; use `ISO-8859-1` to avoid decoding errors.

---

### 📚 References

* Go, A., Bhayani, R., & Huang, L. (2009). Twitter Sentiment Classification using Distant Supervision.
* Dataset: [https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

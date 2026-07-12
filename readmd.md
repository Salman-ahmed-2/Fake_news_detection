# Fake News Detection

A machine learning project that classifies news articles as **real** or **fake** using natural language processing and classical ML models.

## Overview

This project builds a text classification pipeline that analyzes news article content (headline + body) and predicts whether the article is legitimate or fabricated. It covers the full ML workflow: data cleaning, feature engineering, model training, evaluation, and a simple interface for testing new articles.

## Features

- Text preprocessing (lowercasing, punctuation removal, stopword removal, stemming/lemmatization)
- TF-IDF vectorization of article text
- Multiple model comparison (Logistic Regression, Naive Bayes, Passive-Aggressive Classifier, Random Forest)
- Evaluation via accuracy, precision, recall, F1-score, and confusion matrix
- Simple CLI/script to classify a new article as Real or Fake

## Tech Stack

- **Language:** Python 3
- **ML/NLP:** scikit-learn, NLTK / spaCy, pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Dataset:** [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (or specify your actual source)

## Project Structure

```
fake-news-detection/
├── data/
│   ├── raw/                # original dataset files
│   └── processed/          # cleaned/preprocessed data
├── notebooks/
│   └── eda.ipynb           # exploratory data analysis
├── src/
│   ├── preprocess.py       # text cleaning and preprocessing
│   ├── train.py            # model training script
│   ├── evaluate.py         # evaluation and metrics
│   └── predict.py          # classify a new article
├── models/
│   └── model.pkl           # saved trained model
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<salman_ahmed_2>/Fake_news_detection.git
cd fake-news-detection
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage




```bash
python main.py 
```

## How It Works

1. **Data loading** — Load labeled real/fake news articles into a pandas DataFrame.
2. **Preprocessing** — Clean text (remove punctuation, stopwords, lowercase, tokenize).
3. **Feature extraction** — Convert text to numerical features using TF-IDF.
4. **Model training** — Train and compare classifiers on the training split.
5. **Evaluation** — Measure performance on a held-out test set using accuracy, precision, recall, and F1-score.
6. **Prediction** — Feed new, unseen article text through the trained pipeline to get a Real/Fake label with a confidence score.

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | -- | -- | -- | -- |
| Naive Bayes | -- | -- | -- | -- |
| Passive-Aggressive Classifier | -- | -- | -- | -- |
| Random Forest | -- | -- | -- | -- |

*(Fill in with your actual metrics after training.)*

## Limitations

- Performance depends heavily on the dataset's source and time period; may not generalize to news topics or writing styles outside the training data.
- Detects linguistic/stylistic patterns of fake news, not factual accuracy — it is not a substitute for verified fact-checking.

## Future Improvements

- Incorporate deep learning models (LSTM, BERT) for better contextual understanding
- Add a web interface (Flask/Streamlit) for interactive use
- Expand dataset with more diverse and recent sources

## License

This project is licensed under the MIT License.

## Author

Built by [Your Name] as part of an ML/software engineering portfolio project.
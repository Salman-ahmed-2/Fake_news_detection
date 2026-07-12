# Fake News Detection

A machine learning project that classifies news articles as **Fake** or **Not Fake** using text vectorization (TF-IDF) and four classifiers: Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest.

## 1. Requirements

- Python 3.8+
- Datasets: `Fake.csv` and `True.csv` (must be in the same folder as `main.py`)

## 2. Setup

Create and activate a virtual environment, then install dependencies:

```bash
# install python 
sudo apt update && sudo apt install python3

# create venv
python -m venv venv

# activate it

# macOS/Linux:
source venv/bin/activate

# install packages
pip install numpy pandas nltk scikit-learn
```

The script downloads NLTK's stopwords list automatically on first run (`nltk.download('stopwords')`), so make sure you have an internet connection the first time.

## 3. Dataset columns

Your CSVs need either a `text` or a `content` column with the article body. The script auto-renames `text` → `content` if needed. If neither column exists, it will raise an error — check your CSV headers first:

```bash
python -c "import pandas as pd; print(pd.read_csv('Fake.csv').columns.tolist())"
```

## 4. Run

```bash
python main.py
```

The script will:
1. Load and label the two datasets (`0` = fake, `1` = true).
2. Clean/merge them and drop `title`, `subject`, `date`.
3. Clean text (remove non-letters, lowercase, remove stopwords, stem words).
4. Split into train/test sets (80/20).
5. Vectorize text with TF-IDF.
6. Train Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest.
7. Prompt you for a headline/article to classify:

```
Enter some news text here
```

It will then print each model's prediction, e.g.:

```
LR Prediction: Fake News
DT Prediction: Fake News
GB Prediction: Not a Fake News
RF Prediction: Fake News
```

**Note:** stemming + TF-IDF fitting on the whole dataset can take a minute or two depending on dataset size — this is expected, not a hang.

<!-- ## 5. Known issues to fix later

- The two `for i in range(...)` loops meant to drop the last rows used for manual testing don't actually iterate (the start/stop/step values don't produce a valid range), so those rows currently aren't dropped from the training data. Worth revisiting so manual-test rows aren't also used for training.
- Model accuracy isn't printed to the console — add `print(lr.score(...))` etc. (or log them) if you want to see performance per model. -->

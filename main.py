import numpy as np
import pandas as pd
import re
import sys
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Prefer NLTK stopwords when available, otherwise fall back to scikit-learn's list.
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords as _nltk_stopwords
    stop_words = set(_nltk_stopwords.words('english'))
except Exception:
    stop_words = set(ENGLISH_STOP_WORDS)


data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

data_fake['class'] = 0
data_true['class'] = 1

fake_rows = data_fake.shape[0]
true_rows = data_true.shape[0]



true_manual_testinhg = data_true.tail(10)
fake_manual_testing = data_fake.tail(10)
for i in range(21416, 21406,-1):
    data_true.drop([i],axis=0, inplace=True)
for i in range(23480, 23470,-1):
    data_fake.drop([i],axis=0, inplace=True)


data_merge = pd.concat([data_fake, data_true], axis=0)
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

if 'content' not in data.columns:
    if 'text' in data.columns:
        data = data.rename(columns={'text': 'content'})
    else:
        raise ValueError("Expected a 'content' or 'text' column in the dataset.")


data = data.sample(frac=1)

data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)


port_stem = PorterStemmer()
def stemming(content):
    content = '' if pd.isna(content) else str(content)
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in ENGLISH_STOP_WORDS]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


data['content'] = data['content'].apply(stemming)

x = data['content']
y = data['class']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

lr = LogisticRegression()
lr.fit(xv_train, y_train)


pred_lr = lr.predict(xv_test)
lr.score(xv_test, y_test)


dt = DecisionTreeClassifier()
dt.fit(xv_train, y_train)

pred_dt = dt.predict(xv_test)
dt.score(xv_test, y_test)

gb = GradientBoostingClassifier()
gb.fit(xv_train, y_train)

pred_gb = gb.predict(xv_test)
gb.score(xv_test, y_test)

rf = RandomForestClassifier()
rf.fit(xv_train, y_train)

pred_rf = rf.predict(xv_test)
rf.score(xv_test, y_test)

def  output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not a Fake News"
    
def manual_testing(news):
    testing_news = {"content":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['content'] = new_def_test['content'].apply(stemming)
    new_x_test = new_def_test['content']
    new_xv_test = vectorizer.transform(new_x_test)
    pred_lr = lr.predict(new_xv_test)
    pred_dt = dt.predict(new_xv_test)
    pred_gb = gb.predict(new_xv_test)
    pred_rf = rf.predict(new_xv_test)

    return print(f"LR Prediction: {output_lable(pred_lr[0])}\nDT Prediction: {output_lable(pred_dt[0])}\nGB Prediction: {output_lable(pred_gb[0])}\nRF Prediction: {output_lable(pred_rf[0])}")


print("Enter news text:")
if sys.stdin is not None and not sys.stdin.isatty():
    news_head = sys.stdin.read().strip()
    if news_head:
        manual_testing(news_head)
else:
    news_head = input('Enter news text to classify: ')
    manual_testing(news_head)
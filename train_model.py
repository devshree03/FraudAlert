import pandas as pd
import re
import pickle
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from tqdm import tqdm
from sklearn.metrics import classification_report

# One-time NLTK download
nltk.download('stopwords')

# ---------- Text preprocessing ----------
ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def text_processing(text):
    if not isinstance(text, str):
        return ""
    review = re.sub(r'[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(w) for w in review if w not in STOPWORDS]
    return ' '.join(review)

def keyword_counter(text):
    fake_keywords = [
        'hiring', 'freelance', 'home', 'work from home', 'online',
        'easy money', 'no experience', 'fast cash', 'income', 'earning',
        'whatsapp', 'registration fee', 'telegram', 'dm', 'immediate joiner'
    ]
    t = (text or "").lower()
    return sum(t.count(k) for k in fake_keywords)

def yn(v):
    return 1 if str(v).strip().lower() in {"y", "yes", "true", "1"} else 0

print("Loading dataset...")
df = pd.read_csv('DataSet.csv')

# Build text fields
tqdm.pandas()
df['text'] = (
    df['title'].fillna('') + ' ' + df['location'].fillna('') + ' ' +
    df['company_profile'].fillna('') + ' ' + df['description'].fillna('') + ' ' +
    df['requirements'].fillna('') + ' ' + df['benefits'].fillna('') + ' ' +
    df['industry'].fillna('') + ' ' + df['function'].fillna('')
)
df['processed_text'] = df['text'].progress_apply(text_processing)

# Numerical features
num_cols = ['telecommuting', 'has_company_logo', 'has_questions']
df_num = df[num_cols].copy()

for col in num_cols:
    df_num[col] = df_num[col].apply(yn)

df_num['profile_length'] = df['company_profile'].fillna('').apply(lambda s: min(len(s), 3000))
df_num['keyword_count'] = df['text'].apply(keyword_counter)

# Target
#y = df['fraudulent'].astype(int)
# Target
raw_y = df['fraudulent']

# Normalize various truthy/falsey encodings to {0,1}
true_set = {"1","true","t","yes","y","fake","fraud","fraudulent"}
false_set = {"0","false","f","no","n","real","legit","legitimate"}

def to01(v):
    s = str(v).strip().lower()
    if s in true_set:
        return 1
    if s in false_set:
        return 0
    # Fallback: try numeric cast else treat nonzero-ish as 1
    try:
        return 1 if float(s) != 0.0 else 0
    except:
        return 0

y = raw_y.apply(to01).astype(int)


# TF-IDF
print("Vectorizing text...")
tfidf_v = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2), sublinear_tf=True)
X_text = tfidf_v.fit_transform(df['processed_text'])

# Combine with numeric (as sparse csr)
num_mat = sparse.csr_matrix(df_num.astype(np.float32).values)
X = sparse.hstack([X_text, num_mat], format='csr')

# Split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a balanced linear model (works well on sparse text)
print("Training model...")
clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear")
clf.fit(X_train, y_train)

# Evaluate
print("Evaluating...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

# Save artifacts
print("Saving artifacts...")
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('tfidf_v.pkl', 'wb') as f:
    pickle.dump(tfidf_v, f)

print("Done. Run `flask run` or `python app.py` to serve the app.")

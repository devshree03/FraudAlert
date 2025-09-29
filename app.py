from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy import sparse

# Ensure NLTK resources are available (run once)
nltk.download('stopwords')

app = Flask(__name__)

# ---------- Text preprocessing ----------
ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def text_processing(text):
    if not isinstance(text, str):
        text = ""
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

# ---------- Load artifacts ----------
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_v.pkl', 'rb') as f:
        vect = pickle.load(f)
except FileNotFoundError:
    model = None
    vect = None
    print("Artifacts missing: run `python train_model.py` to train and save model/vectorizer.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vect is None:
        return jsonify({'prediction': 'Error: Model not loaded'})

    # ----- Collect form inputs -----
    title = request.form.get('title', '')
    location = request.form.get('location', '')
    company_profile = request.form.get('company_profile', '')
    description = request.form.get('description', '')
    requirements = request.form.get('requirements', '')
    benefits = request.form.get('benefits', '')
    industry = request.form.get('industry', '')
    function = request.form.get('function', '')

    telecommuting = yn(request.form.get('telecommuting', 'No'))
    has_company_logo = yn(request.form.get('has_company_logo', 'No'))
    has_questions = yn(request.form.get('has_questions', 'No'))

    # Derived features
    combined_text = f"{title} {location} {company_profile} {description} {requirements} {benefits} {industry} {function}"
    profile_length = min(len(company_profile or ""), 3000)  # cap extreme lengths
    keyword_count = keyword_counter(combined_text)

    # ----- Vectorize text -----
    processed_text = text_processing(combined_text)
    X_text = vect.transform([processed_text])  # (1, V)

    # ----- Build numeric features as sparse csr -----
    num = np.array([[telecommuting, has_company_logo, has_questions, profile_length, keyword_count]], dtype=np.float32)
    X_num = sparse.csr_matrix(num)  # (1, 5)

    # ----- Combine -----
    X = sparse.hstack([X_text, X_num], format='csr')

    # ----- Predict -----
    pred = int(model.predict(X)[0])  # 1 = FAKE, 0 = REAL
    result = "FAKE" if pred == 1 else "REAL"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

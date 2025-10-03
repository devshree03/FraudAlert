from flask import Flask, render_template, request, jsonify
import pickle, re, nltk, numpy as np, tempfile, os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy import sparse
from werkzeug.utils import secure_filename
from pypdf import PdfReader
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

nltk.download('stopwords')

app = Flask(__name__)
ALLOWED_EXTS = {'.pdf', '.docx'}

# --------- Text preprocessing ----------
ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def text_processing(text):
    if not isinstance(text, str):
        text = ""
    review = re.sub(r'[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(w) for w in review if w not in STOPWORDS]
    return ' '.join(review)

RISK_PHRASES = [
    r'\bregistration fee\b', r'\bactivation fee\b', r'\brefundable fee\b',
    r'\bno interview\b', r'\bjoin (immediately|now)\b', r'\bearn (per day|daily|hourly)\b',
    r'\bdaily payout\b', r'\bpay now\b', r'\bwhatsapp\b', r'\btelegram\b',
    r'\bdirect message\b', r'\bdm\b', r'\bwork from (whatsapp|telegram)\b'
]
RISK_REGEX = re.compile("(" + "|".join(RISK_PHRASES) + ")", flags=re.IGNORECASE)
def keyword_counter(text: str) -> int:
    return len(RISK_REGEX.findall(text or ""))

def yn(v):
    return 1 if str(v).strip().lower() in {"y","yes","true","1"} else 0

# --------- Simple field extraction ----------
FIELD_PATTERNS = {
    "title": [re.compile(r'(?i)^title\s*:?\s*(.+)$')],
    "location": [re.compile(r'(?i)^(location|work location|city)\s*:?\s*(.+)$')],
    "department": [re.compile(r'(?i)^(department|function|team)\s*:?\s*(.+)$')],
    "salary_range": [
        re.compile(r'(?i)^(salary|ctc|compensation|package|pay)\s*:?\s*(.+)$'),
        re.compile(r'(?i)[₹$]?\s?[\d,]+(\.\d+)?\s*(lpa|per\s*(month|annum|hour))(\s*[-to]\s*[₹$]?\s?[\d,]+(\.\d+)?\s*(lpa|per\s*(month|annum|hour))*)?')
    ],
    "employment_type": [re.compile(r'(?i)^(employment\s*type|contract\s*type)\s*:?\s*(.+)$')]
}
SECTION_HEADERS = {
    "requirements": re.compile(r'(?i)^(requirements?|qualifications?|skills?)\b', re.M),
    "benefits": re.compile(r'(?i)^(benefits?|perks?)\b', re.M),
    "description": re.compile(r'(?i)^(job\s*description|about\s*the\s*role|responsibilities?)\b', re.M),
}
def split_lines(text): return [ln.strip() for ln in (text or "").splitlines() if ln.strip()][:4000]
def guess_title(lines):
    for ln in lines[:10]:
        if 5 <= len(ln) <= 120:
            words = ln.split()
            if words and sum(w[:1].isupper() for w in words)/len(words) >= 0.5:
                return ln
    return lines[0] if lines else ""
def extract_section(text, header_rx):
    m = header_rx.search(text)
    if not m: return ""
    start = m.end(); after = text[start:]
    next_hdr = re.search(r'\n[A-Z][A-Za-z ]{2,30}\n', after)
    end = next_hdr.start() if next_hdr else len(after)
    return after[:end].strip()
def extract_fields_from_text(text):
    lines = split_lines(text); joined = "\n".join(lines)
    fields = {"title":"","location":"","department":"","salary_range":"","employment_type":"",
              "company_profile":"","description":"","requirements":"","benefits":"","industry":"","function":""}
    for name, patterns in FIELD_PATTERNS.items():
        if fields.get(name): continue
        for ln in lines[:200]:
            for rx in patterns:
                m = rx.search(ln)
                if m:
                    fields[name] = (m.group(2) if m.lastindex and m.lastindex>=2 else m.group(1)).strip()
                    break
            if fields[name]: break
    if not fields["title"]: fields["title"] = guess_title(lines)
    for key, rx in SECTION_HEADERS.items():
        sec = extract_section(joined, rx)
        if sec and not fields[key]: fields[key] = sec
    if not fields["company_profile"]:
        paras = [p.strip() for p in joined.split("\n\n") if p.strip()]
        if len(paras)>=2: fields["company_profile"] = paras[0][:500]
    return fields

# --------- Load model/vectorizer ----------
try:
    with open('model.pkl','rb') as f: model = pickle.load(f)
    with open('tfidf_v.pkl','rb') as f: vect = pickle.load(f)
except FileNotFoundError:
    model = None; vect = None
    print("Artifacts missing: run `python train_model.py` first.")

@app.route('/')
def home(): return render_template('home.html')

def _predict_from_combined(combined_text, tele=0, logo=0, qs=0, company_profile_len=0):
    processed = text_processing(combined_text)
    X_text = vect.transform([processed])
    profile_length = min(company_profile_len, 3000)
    keyword_count = keyword_counter(combined_text)
    X_num = sparse.csr_matrix(np.array([[tele,logo,qs,profile_length,keyword_count]], dtype=np.float32))
    X = sparse.hstack([X_text, X_num], format='csr')
    pred = int(model.predict(X)[0])
    label = "FAKE" if pred==1 else "REAL"
    proba = float(model.predict_proba(X)[0,1]) if hasattr(model,"predict_proba") else None
    reasons = []
    if hasattr(model,"coef_"):
        terms = vect.get_feature_names_out(); coefs = model.coef_[0]
        present_idx = X_text.nonzero()[1]
        scored = sorted([(terms[i],coefs[i]) for i in present_idx], key=lambda x:x[1], reverse=True)
        reasons = [w for w,c in scored[:5] if c>0]
        percent = None if proba is None else round(proba*100, 1)
    return label, proba, reasons, percent


# ----- Page routes (render result.html) -----
@app.route('/predict_page', methods=['POST'])
def predict_page():
    if model is None or vect is None:
        return render_template('result.html', prediction="Error", prob_fake=None, reasons=[], extracted={"error":"Model not loaded"})
    g = request.form.get
    title, location = g('title',''), g('location','')
    company_profile, description = g('company_profile',''), g('description','')
    requirements, benefits = g('requirements',''), g('benefits','')
    industry, function = g('industry',''), g('function','')
    tele, logo, qs = yn(g('telecommuting','No')), yn(g('has_company_logo','No')), yn(g('has_questions','No'))
    combined = f"{title} {location} {company_profile} {description} {requirements} {benefits} {industry} {function}"
    label, proba, reasons, percent = _predict_from_combined(combined, tele, logo, qs, len(company_profile or ""))
    return render_template('result.html', prediction=label, prob_fake=proba, percent=percent, reasons=reasons, extracted=None)


@app.route('/analyze_text_page', methods=['POST'])
def analyze_text_page():
    if model is None or vect is None:
        return render_template('result.html', prediction="Error", prob_fake=None, reasons=[], extracted={"error":"Model not loaded"})
    raw_text = (request.form.get('text') or '').strip()
    fields = extract_fields_from_text(raw_text)
    combined = " ".join([fields.get(k,'') for k in ['title','location','company_profile','description','requirements','benefits','industry','function']])
    label, proba, reasons, percent = _predict_from_combined(combined, 0,0,0, len(fields.get('company_profile','')))
    return render_template('result.html', prediction=label, prob_fake=proba, percent=percent, reasons=reasons, extracted=fields)

@app.route('/analyze_file_page', methods=['POST'])
def analyze_file_page():
    if model is None or vect is None:
        return render_template('result.html', prediction="Error", prob_fake=None, reasons=[], extracted={"error":"Model not loaded"})
    f = request.files.get('file')
    if not f:
        return render_template('result.html', prediction="Error", prob_fake=None, reasons=[], extracted={"error":"No file uploaded"})
    ext = os.path.splitext(f.filename or "upload")[1].lower()
    if ext not in ALLOWED_EXTS or (ext=='.docx' and not HAS_DOCX):
        return render_template('result.html', prediction="Error", prob_fake=None, reasons=[], extracted={"error":"Unsupported file type"})
    with tempfile.TemporaryDirectory() as tmpd:
        path = os.path.join(tmpd, f.filename)
        f.save(path)
        text = ""
        if ext=='.pdf':
            try:
                reader = PdfReader(path)
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages).strip()
            except Exception:
                text = ""
        else:
            try:
                doc = Document(path)
                text = "\n".join(p.text for p in doc.paragraphs).strip()
            except Exception:
                text = ""
    if not text:
        return render_template('result.html', prediction="Error", prob_fake=None, reasons=[], extracted={"error":"Could not extract text"})
    fields = extract_fields_from_text(text)
    combined = " ".join([fields.get(k,'') for k in ['title','location','company_profile','description','requirements','benefits','industry','function']])
    label, proba, reasons, percent = _predict_from_combined(combined, 0,0,0, len(fields.get('company_profile','')))
    return render_template('result.html', prediction=label, prob_fake=proba, percent=percent, reasons=reasons, extracted=fields)

# Optional JSON APIs (keep if needed)
@app.route('/predict', methods=['POST'])
def predict_api():
    return jsonify({"error":"Use /predict_page for page result"})

@app.route('/analyze_text', methods=['POST'])
def analyze_text_api():
    return jsonify({"error":"Use /analyze_text_page for page result"})

@app.route('/analyze_file', methods=['POST'])
def analyze_file_api():
    return jsonify({"error":"Use /analyze_file_page for page result"})

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/how-it-works')
def how(): return render_template('how.html')

@app.route('/contact')
def contact(): return render_template('contact.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    name = request.form.get('name','').strip()
    email = request.form.get('email','').strip()
    phone = request.form.get('phone','').strip()
    msg = request.form.get('message','').strip()

    # TODO: replace with email or DB write
    print(f"[Feedback] {name} <{email}> {phone}: {msg}")

    # Reuse result page as a simple confirmation
    return render_template(
        'result.html',
        prediction="Thanks!",
        prob_fake=None,
        percent=None,
        reasons=["Feedback received"],
        extracted={"name": name, "email": email, "phone": phone}
    )

if __name__ == '__main__':
    app.run(debug=True)

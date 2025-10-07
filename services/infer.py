import re, numpy as np, math
from scipy import sparse

# --- Risk phrases ---
RISK_PHRASES = [
    r'\bregistration fee\b', r'\bactivation fee\b', r'\brefundable fee\b',
    r'\bno interview\b', r'\bjoin (immediately|now)\b', r'\bearn (per day|daily|hourly)\b',
    r'\bdaily payout\b', r'\bpay now\b', r'\bwhatsapp\b', r'\btelegram\b',
    r'\bdirect message\b', r'\bdm\b', r'\bwork from (whatsapp|telegram)\b'
]
RISK_REGEX = re.compile("(" + "|".join(RISK_PHRASES) + ")", flags=re.IGNORECASE)

def keyword_count(text: str) -> int:
    return len(RISK_REGEX.findall(text or ""))

def yn(v) -> int:
    return 1 if str(v).strip().lower() in {"y","yes","true","1"} else 0

# --- Lightweight field extraction ---
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

def split_lines(text):
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()][:4000]

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

# --- Probability helpers ---
def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def safe_proba(model, X) -> float:
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(X)[0,1])
        return min(max(p, 0.0), 1.0)
    if hasattr(model, "decision_function"):
        raw = float(model.decision_function(X)[0])
        return _sigmoid(raw)
    pred = int(model.predict(X)[0])
    return 0.75 if pred == 1 else 0.25

def calibrate_percent(p: float) -> float:
    T = 1.25
    odds = p / max(1 - p, 1e-6)
    pT = 1.0 / (1.0 + (1.0 / odds) ** (1.0 / T))
    p_adj = min(max(pT, 0.15), 0.90)
    return round(p_adj * 100.0, 1)

# --- Structure (REAL-leaning) signals ---
STRUCTURE_RX = [
    re.compile(r'(?i)\b(responsibilities|about the role|about us|who you are)\b'),
    re.compile(r'(?i)\b(qualifications|requirements|skills)\b'),
    re.compile(r'(?i)\b(benefits|perks)\b'),
]
EMAIL_RX = re.compile(r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b')
URL_RX   = re.compile(r'(?i)\bhttps?://[^\s)]+')

def structure_score(text: str) -> int:
    score = 0
    for rx in STRUCTURE_RX:
        if rx.search(text or ""):
            score += 1
    if EMAIL_RX.search(text or ""):
        score += 1
    if URL_RX.search(text or ""):
        score += 1
    return min(score, 5)

def bullet_score(text: str) -> int:
    return sum(marker in (text or "") for marker in ["\n- ", "\n•", "\n*", "\n– "])

def length_bucket(text: str) -> int:
    L = len(text or "")
    if L >= 3500: return 3
    if L >= 2000: return 2
    if L >= 1000: return 1
    return 0

# --- Main predictor ---
def predict(model, vect, cleaned_text: str, original_text: str,
            tele=0, logo=0, qs=0, company_profile_len=0):
    # Text features
    X_text = vect.transform([cleaned_text])

    # Numeric features (keep original 5 for model compatibility)
    profile_length = min(company_profile_len, 3000)
    kcount = keyword_count(original_text)
    X_num = sparse.csr_matrix(np.array([[tele, logo, qs, profile_length, kcount]], dtype=np.float32))

    # Stack and predict
    X = sparse.hstack([X_text, X_num], format='csr')
    pred = int(model.predict(X)[0])
    label = "FAKE" if pred==1 else "REAL"

    # Probability -> calibrated percent
    p = safe_proba(model, X)
    if p != p:  # NaN guard
        p = 0.5
    percent = calibrate_percent(p)
    proba = p

    # Reasons from linear model weights (if available)
    reasons = []
    if hasattr(model, "coef_"):
        try:
            terms = vect.get_feature_names_out()
            coefs = model.coef_[0]
            present_idx = X_text.nonzero()[1]
            scored = sorted([(terms[i],coefs[i]) for i in present_idx], key=lambda x:x[1], reverse=True)
            reasons = [w for w,c in scored[:5] if c>0]
        except Exception:
            reasons = []

       # REAL-friendly nudges (display-time only)
    sstruct = structure_score(original_text)
    if sstruct >= 4:
        percent = max(10.0, percent - 25.0)
    elif sstruct == 3 and 15 <= percent <= 85:
        percent = max(10.0, percent - 15.0)
    if sstruct >= 3:
        reasons = (reasons or []) + ["Structured JD detected", "Contact/URL present"]

    lb = length_bucket(original_text)
    bs = bullet_score(original_text)
    if (lb >= 2 and bs >= 2):
        percent = max(10.0, percent - 15.0)
        reasons = (reasons or []) + ["Detailed length detected", "Bullet points present"]

    # Optional cap for clearly structured, detailed JDs (keeps display friendly)
    if sstruct >= 4 and (lb >= 2 and bs >= 2):
        percent = min(percent, 60.0)  # cap fake% at 60 for very structured posts

    # Convert to percent(REAL) for display
    percent_fake = percent
    percent_real = max(0.0, 100.0 - percent_fake)

    return label, proba, percent_real, reasons

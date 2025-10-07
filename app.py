from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pypdf import PdfReader

# Services / utils
from utils.text import clean_text
from services.infer import extract_fields_from_text, predict, yn

# Stdlib / libs
import os, time, uuid, logging, tempfile, pickle, nltk
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional DOCX support
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# NLTK data (only needed once; harmless if already present)
nltk.download('stopwords')

# -----------------------------------------------------------------------------
# App and configuration
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MODEL_NAME  = os.getenv("MODEL_NAME",  "LogReg")
ALLOWED_EXTS = {".pdf", ".docx"}

# Logging
from utils.logging import init_logging
logger = init_logging("fraudalert")

@app.context_processor
def inject_globals():
    return {"APP_VERSION": APP_VERSION, "MODEL_NAME": MODEL_NAME}

@app.errorhandler(413)
def too_large(e):
    return jsonify(error=f"File too large. Max {os.getenv('MAX_UPLOAD_MB','10')} MB."), 413

@app.errorhandler(Exception)
def handle_ex(e):
    rid = uuid.uuid4().hex[:8]
    logging.exception("rid=%s route=unhandled %s", rid, e)
    return render_template(
        "result.html",
        prediction="Error",
        prob_fake=None,
        percent=None,
        reasons=["Something went wrong"],
        extracted={"error": "Unexpected error"},
    ), 500

# -----------------------------------------------------------------------------
# Load artifacts
# -----------------------------------------------------------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_v.pkl", "rb") as f:
        vect = pickle.load(f)
except FileNotFoundError:
    model = None
    vect = None
    print("Artifacts missing: run `python train_model.py` first.")

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", sub="Analyze text, files, or forms")

@app.route("/predict_page", methods=["POST"])
def predict_page():
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()

    if model is None or vect is None:
        logging.error("rid=%s route=predict_page error=model_not_loaded", rid)
        return render_template(
            "result.html",
            prediction="Error",
            prob_fake=None,
            percent=None,
            reasons=[],
            extracted={"error": "Model not loaded"},
        )

    g = request.form.get
    title, location = g("title", ""), g("location", "")
    company_profile, description = g("company_profile", ""), g("description", "")
    requirements, benefits = g("requirements", ""), g("benefits", "")
    industry, function = g("industry", ""), g("function", "")
    tele, logo, qs = yn(g("telecommuting", "No")), yn(g("has_company_logo", "No")), yn(g("has_questions", "No"))

    combined = f"{title} {location} {company_profile} {description} {requirements} {benefits} {industry} {function}"
    processed = clean_text(combined)
    label, proba, percent, reasons = predict(
        model, vect, processed, combined, tele, logo, qs, len(company_profile or "")
    )

    latency = int((time.time() - t0) * 1000)
    logging.info(
        "rid=%s route=predict_page label=%s percent=%s latency_ms=%d ip=%s",
        rid, label, percent, latency, request.remote_addr
    )
    return render_template("result.html", prediction=label, prob_fake=proba, percent=percent, reasons=reasons, extracted=None)

@app.route("/analyze_text_page", methods=["POST"])
def analyze_text_page():
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()

    if model is None or vect is None:
        logging.error("rid=%s route=analyze_text_page error=model_not_loaded", rid)
        return render_template(
            "result.html",
            prediction="Error",
            prob_fake=None,
            percent=None,
            reasons=[],
            extracted={"error": "Model not loaded"},
        )

    raw_text = (request.form.get("text") or "").strip()
    fields = extract_fields_from_text(raw_text)
    combined = " ".join([fields.get(k, "") for k in [
        "title", "location", "company_profile", "description", "requirements", "benefits", "industry", "function"
    ]])
    
    processed = clean_text(combined)
    label, proba, percent, reasons = predict(model, vect, processed, combined, 0, 0, 0, len(fields.get("company_profile", "")))

    latency = int((time.time() - t0) * 1000)
    logging.info(
        "rid=%s route=analyze_text_page label=%s percent=%s latency_ms=%d ip=%s",
        rid, label, percent, latency, request.remote_addr
    )
    return render_template("result.html", prediction=label, prob_fake=proba, percent=percent, reasons=reasons, extracted=fields)

'''@app.route('/echo_upload', methods=['POST'])
def echo_upload():
    info = {
        "content_type": request.content_type,
        "files_keys": list(request.files.keys()),
        "form_keys": list(request.form.keys()),
        "has_file_key": "file" in request.files,
    }
    f = request.files.get('file')
    if f and f.filename:
        info["file_name"] = f.filename
        # Read a few bytes to prove content exists
        chunk = f.read(1024)
        info["first_kb_len"] = len(chunk or b"")
        f.seek(0)
    return jsonify(info)'''


@app.route('/analyze_file_page', methods=['POST'])
def analyze_file_page():
    rid = uuid.uuid4().hex[:8]; t0 = time.time()
    try:
        if model is None or vect is None:
            logging.error("rid=%s route=analyze_file_page error=model_not_loaded", rid)
            return render_template('result.html', prediction="Error", prob_fake=None, percent=None,
                                   reasons=["Model not loaded"], extracted={"error":"Model not loaded"})

        # Accept file (already proven by echo_upload)
        f = request.files.get('file')
        if (not f or not f.filename) and request.files:
            first_key = next(iter(request.files.keys()))
            f = request.files.get(first_key)
            logging.warning("rid=%s using_first_file key=%s", rid, first_key)
        if not f or not f.filename:
            logging.warning("rid=%s missing_file_after_echo", rid)
            return render_template('result.html', prediction="Error", prob_fake=None, percent=None,
                                   reasons=["No file uploaded"], extracted={"error":"No file"})

        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXTS or (ext=='.docx' and not HAS_DOCX):
            return render_template('result.html', prediction="Error", prob_fake=None, percent=None,
                                   reasons=["Unsupported file type"], extracted={"error":"Unsupported type"})

        # Extract text robustly
        text = ""
        with tempfile.TemporaryDirectory() as tmpd:
            safe_name = secure_filename(f.filename) or ("upload" + ext)
            path = os.path.join(tmpd, safe_name)
            f.save(path)

            if ext == '.pdf':
                try:
                    reader = PdfReader(path)
                    # Try standard extraction
                    pages = [p.extract_text() or "" for p in reader.pages]
                    text = "\n".join(pages).strip()
                    # Fallbacks for stubborn PDFs
                    if not text:
                        logging.warning("rid=%s pdf_text_empty_firstpass", rid)
                        pages2 = []
                        for p in reader.pages:
                            t = p.extract_text() or ""
                            t = t.replace("\x00","")  # strip nulls
                            pages2.append(t)
                        text = "\n".join(pages2).strip()
                    if not text:
                        # As a last resort, take first N pages only to avoid huge empties
                        pages3 = [p.extract_text() or "" for p in reader.pages[:3]]
                        text = "\n".join(pages3).strip()
                except Exception as e:
                    logging.exception("rid=%s pdf_extract_error %s", rid, e)
                    text = ""
            else:
                try:
                    doc = Document(path)
                    text = "\n".join(p.text for p in doc.paragraphs).strip()
                except Exception as e:
                    logging.exception("rid=%s docx_extract_error %s", rid, e)
                    text = ""

        if not text:
            return render_template('result.html', prediction="Error", prob_fake=None, percent=None,
                                   reasons=["Could not extract text"], extracted={"error":"Empty text from file"})

        # Build fields and predict
        fields = extract_fields_from_text(text)
        combined = " ".join([fields.get(k,'') for k in [
            'title','location','company_profile','description','requirements','benefits','industry','function'
        ]])
        processed = clean_text(combined)
        label, proba, percent, reasons = predict(
            model, vect, processed, combined, 0, 0, 0, len(fields.get('company_profile',''))
        )

        latency = int((time.time() - t0) * 1000)
        logging.info("rid=%s route=analyze_file_page ext=%s label=%s percent=%s latency_ms=%d",
                     rid, ext, label, percent, latency)

        return render_template('result.html', prediction=label, prob_fake=proba,
                               percent=percent, reasons=reasons, extracted=fields)
    except Exception as e:
        logging.exception("rid=%s route=analyze_file_page unhandled %s", rid, e)
        return render_template('result.html', prediction="Error", prob_fake=None, percent=None,
                               reasons=["Something went wrong"], extracted={"error":"Unexpected error"}), 500

# Optional JSON placeholders (non-UI)
@app.route("/predict", methods=["POST"])
def predict_api():
    return jsonify({"error": "Use /predict_page for page result"})

@app.route("/analyze_text", methods=["POST"])
def analyze_text_api():
    return jsonify({"error": "Use /analyze_text_page for page result"})

@app.route("/analyze_file", methods=["POST"])
def analyze_file_api():
    return jsonify({"error": "Use /analyze_file_page for page result"})

@app.route("/about")
def about():
    return render_template("about.html", sub="Our Mission & Vision")

@app.route("/how-it-works")
def how():
    return render_template("how.html", sub="How FraudAlert evaluates risk")

@app.route("/contact")
def contact():
    return render_template("contact.html", sub="Get in touch")

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        name  = (request.form.get("name", "")  or "").strip()[:80]
        email = (request.form.get("email", "") or "").strip()[:120]
        phone = (request.form.get("phone", "") or "").strip()[:30]
        msg   = (request.form.get("message", "") or "").strip()[:4000]

        logging.info("feedback name=%s email=%s phone=%s len=%d", name, email, phone, len(msg))

        return render_template(
            "result.html",
            prediction="Success",
            prob_fake=None,
            percent=None,
            reasons=["Thank you! Feedback received."],
            extracted={"name": name, "email": email, "phone": phone},
        )
    except Exception as e:
        logging.exception("feedback_error %s", e)
        return render_template(
            "result.html",
            prediction="Error",
            prob_fake=None,
            percent=None,
            reasons=["Something went wrong"],
            extracted={"error": "Unexpected error"},
        ), 500

if __name__ == "__main__":
    app.run(debug=True)

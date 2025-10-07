import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    review = re.sub(r'[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(w) for w in review if w not in STOPWORDS]
    return ' '.join(review)

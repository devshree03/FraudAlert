import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('DataSet.csv')

# 2. Data Cleaning and Preprocessing
print("Preprocessing data...")
# Combine relevant text columns into a single feature
df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + df['industry'] + ' ' + df['function']

# Handle missing values by replacing them with empty strings
df.fillna('', inplace=True)

# Text Pre-processing function
ps = PorterStemmer()
def text_processing(text):
    if not isinstance(text, str):
        text = ""
    # Remove non-alphabetic characters and convert to lowercase
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    # Remove stopwords and apply stemming
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

# Apply the processing function to our combined text column
df['processed_text'] = df['text'].apply(text_processing)

# 3. Split the data into training and testing sets
print("Splitting data...")
X = df['processed_text']
y = df['fraudulent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorize the text data
# Vectorization turns our text into numbers that the model can understand
print("Vectorizing text data...")
tfidf_v = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vectorized = tfidf_v.fit_transform(X_train)
X_test_vectorized = tfidf_v.transform(X_test)

# 5. Train the machine learning model
print("Training the model...")
model = PassiveAggressiveClassifier(max_iter=50, random_state=42, C=0.5)
model.fit(X_train_vectorized, y_train)

# 6. Save the trained model and vectorizer
# We save these so we don't have to re-train the model every time we run the app
print("Saving the model and vectorizer...")
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_v.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_v, vectorizer_file)

print("Model and vectorizer saved successfully. You can now run 'python train_model.py'")
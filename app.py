from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create the Flask application object
app = Flask(__name__)

# Pre-processing function for text data
# This must be exactly the same as the one used for training
ps = PorterStemmer()
def text_processing(text):
    if not isinstance(text, str):
        text = ""
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

# Load the trained model and vectorizer
# We use a try-except block to handle cases where files might not exist
try:
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf_v = pickle.load(open('tfidf_v.pkl', 'rb'))
except FileNotFoundError:
    model = None
    tfidf_v = None
    print("Model or vectorizer file not found. Please train the model first by running 'python train_model.py'")

# Define the main route for our application
# The '@app.route('/')' decorator links the URL '/' to the home() function
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route that handles form submissions
# The methods=['POST'] argument means this route only accepts POST requests
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the model files were loaded successfully
    if model is None or tfidf_v is None:
        return jsonify({'prediction': 'Error: Model not loaded'})

    # Get the job description text from the form
    data = request.form['job_description']

    # Preprocess the input using the same function as the training script
    processed_text = text_processing(data)

    # Vectorize the preprocessed text
    vectorized_text = tfidf_v.transform([processed_text])

    # Make a prediction using the loaded model
    prediction = model.predict(vectorized_text)[0]

    # Interpret the prediction result
    result = "FAKE" if prediction == 1 else "REAL"

    # Return the result as a JSON object
    return jsonify({'prediction': result})

# Run the application when the script is executed
if __name__ == '__main__':
    app.run(debug=True)
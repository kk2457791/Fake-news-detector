"""
Fake News Detector - Flask Backend
This backend handles text analysis using a pre-trained ML model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import requests
from bs4 import BeautifulSoup
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load the model (mock code - you'll need to replace with your actual trained model)
def load_model():
    """
    In a real implementation, you would load your trained model here.
    For the hackathon demo, we're creating a placeholder.
    """
    # Check if model exists, if not, train a simple one
    if os.path.exists("model/fake_news_model.pkl"):
        # Load the model
        model = pickle.load(open("model/fake_news_model.pkl", "rb"))
        vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
        return model, vectorizer
    else:
        # For demo purposes, return None - we'll use a mock prediction
        return None, None

model, vectorizer = load_model()

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def extract_content_from_url(url):
    """Extract main text content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error extracting content from URL: {e}")
        return None

def analyze_content(content):
    """
    Analyze content using the pre-trained model
    For the hackathon demo, we're using mock predictions
    """
    # Preprocess the content
    processed_text = preprocess_text(content)
    
    if model and vectorizer:
        # Transform text using the vectorizer
        text_vectorized = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get the confidence score
        confidence = round(max(probabilities) * 100)
        
        # Map prediction to category
        if prediction == 1:
            category = "Fake"
        elif confidence < 65:  # If confidence is low
            category = "Needs Fact-Checking"
        else:
            category = "Real"
            
    else:
        # For the demo, make a mock prediction based on text features
        # This is just an example - replace with better heuristics or a real model
        
        # Check for clickbait-like features
        clickbait_words = ['shocking', 'unbelievable', 'amazing', 'won\'t believe', 
                          'secret', 'trick', 'conspiracy', 'hoax']
        
        # Count how many clickbait words appear in the text
        clickbait_count = sum(1 for word in clickbait_words if word in content.lower())
        
        # Apply simplistic rules for demo purposes
        if clickbait_count >= 2:
            category = "Fake"
            confidence = 75 + min(clickbait_count * 5, 15)  # 75-90% confidence
        elif len(content) < 100:  # Very short content is suspicious
            category = "Needs Fact-Checking"
            confidence = 70
        else:
            # Random but weighted selection for demo purposes
            import random
            r = random.random()
            if r < 0.5:
                category = "Real"
                confidence = 65 + random.randint(5, 30)
            elif r < 0.8:
                category = "Needs Fact-Checking"
                confidence = 55 + random.randint(5, 20)
            else:
                category = "Fake"
                confidence = 70 + random.randint(5, 25)
    
    return {
        "category": category,
        "confidence": confidence,
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze content"""
    try:
        data = request.json
        content_type = data.get('type', 'text')
        content = data.get('content', '')
        
        if not content:
            return jsonify({"error": "No content provided"}), 400
        
        # If URL is provided, extract content from it
        if content_type == 'url':
            extracted_content = extract_content_from_url(content)
            if not extracted_content:
                return jsonify({"error": "Failed to extract content from URL"}), 400
            content = extracted_content
        
        # Analyze the content
        result = analyze_content(content)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    # Make sure model directory exists
    os.makedirs("model", exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

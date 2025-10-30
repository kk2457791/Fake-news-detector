"""
Fake News Detection - Machine Learning Model Training

This script trains a fake news classification model using a real-world dataset.
It uses a LIAR dataset or Kaggle's fake news dataset and implements a pipeline
for text preprocessing, feature extraction, and model training.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import urllib.request
import zipfile

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Create necessary directories
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def download_fake_news_dataset():
    """Download Kaggle's fake news dataset"""
    print("Downloading Fake News dataset...")
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/news-fake-detection.zip"
    zip_path = "data/news-fake-detection.zip"
    
    try:
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Dataset downloaded successfully!")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/")
        print("Dataset extracted successfully!")
        
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_kaggle_fake_news_dataset():
    """Load Kaggle's fake news dataset"""
    try:
        # Try to load the dataset
        df_true = pd.read_csv('data/True.csv')
        df_fake = pd.read_csv('data/Fake.csv')
        
        # Add labels
        df_true['label'] = 0  # 0 for true news
        df_fake['label'] = 1  # 1 for fake news
        
        # Combine datasets
        df = pd.concat([df_true, df_fake], ignore_index=True)
        
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Dataset files not found. Attempting to download...")
        if download_fake_news_dataset():
            # Try loading again
            return load_kaggle_fake_news_dataset()
        else:
            # Create a small synthetic dataset if download fails
            return create_synthetic_dataset()

def create_synthetic_dataset():
    """Create a small synthetic dataset for demonstration"""
    print("Creating a synthetic dataset for demonstration...")
    
    fake_news = [
        "SHOCKING: Scientists discover conspiracy to hide alien life on Mars",
        "You won't believe what this celebrity did to lose 50 pounds in 1 week!",
        "Government officials hiding the truth about mind control experiments",
        "Secret cure for all diseases suppressed by big pharma",
        "Breaking: Famous actor arrested for bizarre ritual involving farm animals",
        "New study shows vaccines cause autism in children",
        "Doctor reveals FDA is hiding cancer cure from the public",
        "5G towers are actually mind control devices being tested on population",
        "Politician caught smuggling illegal substances across border",
        "New evidence shows moon landing was filmed in Hollywood studio"
    ]
    
    real_news = [
        "Study shows moderate exercise may improve cognitive function in older adults",
        "New species of frog discovered in Amazon rainforest, scientists report",
        "Tech company announces quarterly earnings above analyst expectations",
        "Local community raises funds for family affected by recent house fire",
        "Research indicates regular sleep schedule may help prevent certain health conditions",
        "New climate report suggests urgent action needed to reduce emissions",
        "Archeologists discover ancient artifacts at excavation site",
        "City council approves new infrastructure development plan",
        "Scientists identify potential new treatment for common illness",
        "New educational policy aims to improve literacy rates in underserved areas"
    ]
    
    # Create DataFrame
    data = {
        'title': fake_news + real_news,
        'text': fake_news + real_news,  # Using same content for title and text in synthetic data
        'label': [1] * len(fake_news) + [0] * len(real_news)  # 1 for fake, 0 for real
    }
    
    df = pd.DataFrame(data)
    return df

def prepare_data(df):
    """Prepare and preprocess the dataset"""
    print("Preparing and preprocessing data...")
    
    # Check if required columns exist
    required_columns = ['text', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        # If 'text' column is missing but 'title' and 'subject' exist, create text column
        if 'text' in missing_columns and 'title' in df.columns:
            if 'subject' in df.columns:
                df['text'] = df['title'] + " " + df['subject']
            else:
                df['text'] = df['title']
    
    # Handle missing values
    df = df.dropna(subset=['text', 'label'])
    
    # Balance the dataset if very imbalanced
    min_count = min(df['label'].value_counts())
    if min_count / len(df) < 0.3:  # If minority class is less than 30%
        print("Balancing dataset...")
        df_majority = df[df['label'] == df['label'].value_counts().idxmax()]
        df_minority = df[df['label'] == df['label'].value_counts().idxmin()]
        
        # Downsample majority class
        df_majority_downsampled = df_majority.sample(
            min(len(df_majority), 3 * len(df_minority)), 
            random_state=42
        )
        
        # Combine minority class with downsampled majority class
        df = pd.concat([df_majority_downsampled, df_minority])
    
    print(f"Class distribution after preprocessing: {df['label'].value_counts()}")
    
    # Preprocess text
    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Remove rows with empty processed text
    df = df[df['processed_text'].str.strip().astype(bool)]
    
    return df

def train_model(df):
    """Train and evaluate the fake news detection model"""
    # Prepare features and target
    X = df['processed_text']
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create a pipeline with TF-IDF and classifiers
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Train the model
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Try different classifiers
    best_model = try_different_classifiers(X_train, X_test, y_train, y_test)
    
    # Save the best model
    print("Saving the best model...")
    with open("model/fake_news_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    # Extract and save the vectorizer from the best model pipeline
vectorizer = best_model.named_steps['vectorizer']

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
    # Save a sample text for testing
    with open("model/sample_texts.pkl", "wb") as f:
        sample_texts = {
            'fake': X_test[y_test == 1].iloc[:5].tolist(),
            'real': X_test[y_test == 0].iloc[:5].tolist()
        }
        pickle.dump(sample_texts, f)
    
    return best_model

def try_different_classifiers(X_train, X_test, y_train, y_test):
    """Try different classifiers and find the best one"""
    print("\nTrying different classifiers...")
    
    # Create TF-IDF features
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # List of classifiers
    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000, C=1.0)),
        ('Multinomial Naive Bayes', MultinomialNB()),
        ('Random Forest', RandomForestClassifier(n_estimators=100))
    ]
    
    best_accuracy = 0
    best_model = None
    results = []
    
    for name, clf in classifiers:
        print(f"Training {name}...")
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {accuracy:.4f}")
        
        results.append((name, accuracy))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Create a pipeline with the best classifier
            best_model = Pipeline([
                ('vectorizer', tfidf),
                ('classifier', clf)
            ])
    
    print("\nClassifier Results:")
    for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name}: {acc:.4f}")
    
    return best_model

def plot_confusion_matrix(y_test, y_pred):
    """Plot and save the confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the confusion matrix
    os.makedirs("model/plots", exist_ok=True)
    plt.savefig("model/plots/confusion_matrix.png")
    print("Confusion matrix saved as model/plots/confusion_matrix.png")

def test_model(model):
    """Test the model with a few examples"""
    print("\nTesting the model with examples:")
    
    test_examples = [
        "Scientists discover new species in the Amazon rainforest",
        "Local community raises funds for family affected by recent fire",
        "SHOCKING: Government hiding alien technology from the public",
        "Miracle cure for all diseases discovered but suppressed by pharmaceutical companies",
        "New study shows correlation between exercise and improved mental health"
    ]
    
    for text in test_examples:
        processed = preprocess_text(text)
        prediction = model.predict([processed])[0]
        confidence = model.predict_proba([processed])[0].max() * 100
        
        result = "FAKE" if prediction == 1 else "REAL"
        print(f"Text: {text}")
        print(f"Prediction: {result} (Confidence: {confidence:.2f}%)\n")

def main():
    """Main function to execute the training pipeline"""
    print("=== Fake News Detection Model Training ===")
    
    # Load dataset
    df = load_kaggle_fake_news_dataset()
    
    # Prepare and preprocess data
    df = prepare_data(df)
    
    # Train the model
    model = train_model(df)
    
    # Test the model
    test_model(model)
    
    print("\nModel training complete! The model is saved in the 'model' directory.")
    print("You can now use this model with the Flask backend.")

if __name__ == "__main__":
    main()
    vectorizer = best_model.named_steps['vectorizer']
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

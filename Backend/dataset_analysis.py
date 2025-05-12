"""
Fake News Dataset Analysis and Feature Engineering

This script analyzes the fake news dataset characteristics and
implements feature engineering to improve model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import os
import traceback
from tqdm import tqdm

# Create necessary directories
os.makedirs("analysis", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Set up error handling
def safe_execution(func, *args, **kwargs):
    """Execute a function safely with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error in {func.__name__}: {str(e)}")
        traceback.print_exc()
        return None

# Download necessary NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")
    print("Continuing without NLTK data...")

def load_dataset():
    """Load the fake news dataset"""
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
        print("Dataset files not found. Please ensure data/True.csv and data/Fake.csv exist.")
        # Create a small sample dataset for testing
        print("Creating sample dataset for testing...")
        sample_data = {
            'title': ['Real News Title', 'Fake News Title!'],
            'text': ['This is some real news content that is factual and straightforward.', 
                    'YOU WON\'T BELIEVE what happened next! This SHOCKING revelation will blow your mind!'],
            'label': [0, 1]
        }
        return pd.DataFrame(sample_data)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def analyze_dataset(df):
    """Perform exploratory data analysis on the dataset"""
    if df is None:
        print("Cannot analyze dataset: No data available")
        return None
        
    print("\n=== Dataset Analysis ===")
    
    # Basic info
    print("\nDataset Information:")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    try:
        # Analyze text length
        df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='text_length', hue='label', bins=50, alpha=0.7)
        plt.title('Distribution of Text Length by Class')
        plt.xlabel('Number of Words')
        plt.ylabel('Count')
        plt.legend(['Real', 'Fake'])
        plt.savefig("analysis/text_length_distribution.png")
        
        # Analyze average text length by class
        avg_length = df.groupby('label')['text_length'].mean()
        print("\nAverage text length by class:")
        print(f"Real news: {avg_length.get(0, 'N/A'):.2f} words" if 0 in avg_length else "Real news: N/A")
        print(f"Fake news: {avg_length.get(1, 'N/A'):.2f} words" if 1 in avg_length else "Fake news: N/A")
        
        # Analyze common topics/subjects if available
        if 'subject' in df.columns:
            print("\nSubject distribution:")
            subject_counts = df.groupby(['subject', 'label']).size().unstack()
            print(subject_counts)
            
            plt.figure(figsize=(12, 6))
            subject_counts.plot(kind='bar', stacked=False)
            plt.title('Subject Distribution by Class')
            plt.xlabel('Subject')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("analysis/subject_distribution.png")
    except Exception as e:
        print(f"Error in dataset analysis: {str(e)}")
    
    return df

def extract_frequent_words(df):
    """Extract most frequent words for real and fake news"""
    if df is None:
        print("Cannot extract frequent words: No data available")
        return None, None
        
    print("\n=== Most Frequent Words Analysis ===")
    
    try:
        # Create a function to extract words
        def get_top_words(texts, n=20):
            try:
                vectorizer = CountVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 1)
                )
                X = vectorizer.fit_transform(texts)
                words = vectorizer.get_feature_names_out()
                counts = X.sum(axis=0).A1
                
                # Create a dictionary of words and counts
                word_counts = dict(zip(words, counts))
                top_words = {k: v for k, v in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:n]}
                
                return top_words
            except Exception as e:
                print(f"Error in get_top_words: {str(e)}")
                return {}
        
        # Get top words for each class
        real_news = df[df['label'] == 0]['text'].fillna('')
        fake_news = df[df['label'] == 1]['text'].fillna('')
        
        top_real_words = get_top_words(real_news)
        top_fake_words = get_top_words(fake_news)
        
        if top_real_words and top_fake_words:
            # Plot word frequency for real news
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.bar(list(top_real_words.keys())[:10], list(top_real_words.values())[:10])
            plt.title('Most Frequent Words in Real News')
            plt.xticks(rotation=90)
            
            # Plot word frequency for fake news
            plt.subplot(1, 2, 2)
            plt.bar(list(top_fake_words.keys())[:10], list(top_fake_words.values())[:10])
            plt.title('Most Frequent Words in Fake News')
            plt.xticks(rotation=90)
            
            plt.tight_layout()
            plt.savefig("analysis/word_frequency.png")
            
            # Try to create word clouds if WordCloud is available
            try:
                from wordcloud import WordCloud
                create_word_clouds(real_news, fake_news)
            except ImportError:
                print("WordCloud not available, skipping word cloud generation")
                
        return top_real_words, top_fake_words
    except Exception as e:
        print(f"Error in extract_frequent_words: {str(e)}")
        return {}, {}

def create_word_clouds(real_news, fake_news):
    """Create word clouds for real and fake news"""
    try:
        from wordcloud import WordCloud
        print("Creating word clouds...")
        
        # Function to combine all text
        def combine_texts(texts):
            return ' '.join(texts.tolist())
        
        # Create word clouds
        real_text = combine_texts(real_news)
        fake_text = combine_texts(fake_news)
        
        # Real news word cloud
        plt.figure(figsize=(10, 8))
        wc_real = WordCloud(
            background_color='white',
            max_words=200,
            width=800,
            height=600,
            stopwords=set(stopwords.words('english')) if 'stopwords' in dir(nltk.corpus) else set()
        ).generate(real_text)
        
        plt.imshow(wc_real, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Real News')
        plt.tight_layout()
        plt.savefig("analysis/wordcloud_real.png")
        
        # Fake news word cloud
        plt.figure(figsize=(10, 8))
        wc_fake = WordCloud(
            background_color='white',
            max_words=200,
            width=800,
            height=600,
            stopwords=set(stopwords.words('english')) if 'stopwords' in dir(nltk.corpus) else set()
        ).generate(fake_text)
        
        plt.imshow(wc_fake, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Fake News')
        plt.tight_layout()
        plt.savefig("analysis/wordcloud_fake.png")
    except Exception as e:
        print(f"Error creating word clouds: {str(e)}")

def engineer_features(df):
    """Add engineered features to improve model performance"""
    if df is None:
        print("Cannot engineer features: No data available")
        return None
        
    print("\n=== Feature Engineering ===")
    
    try:
        # Calculate readability metrics if textstat is available
        try:
            import textstat
            print("Calculating readability scores...")
            # Sample a subset for readability analysis (can be computationally intensive)
            sample_size = min(1000, len(df))
            df_sample = df.sample(sample_size, random_state=42) if len(df) > 1 else df
            
            # Calculate readability metrics
            df_sample['flesch_reading_ease'] = df_sample['text'].apply(
                lambda x: textstat.flesch_reading_ease(str(x)) if isinstance(x, str) else 0
            )
            df_sample['flesch_kincaid_grade'] = df_sample['text'].apply(
                lambda x: textstat.flesch_kincaid_grade(str(x)) if isinstance(x, str) else 0
            )
        except ImportError:
            print("textstat library not available, skipping readability analysis")
            df_sample = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1 else df
        
        # Analyze exclamation marks and question marks
        print("Analyzing punctuation patterns...")
        df_sample['exclamation_count'] = df_sample['text'].apply(
            lambda x: str(x).count('!') if isinstance(x, str) else 0
        )
        df_sample['question_count'] = df_sample['text'].apply(
            lambda x: str(x).count('?') if isinstance(x, str) else 0
        )
        
        # Calculate uppercase ratio
        print("Calculating uppercase ratio...")
        df_sample['uppercase_ratio'] = df_sample['text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if isinstance(x, str) and len(str(x)) > 0 else 0
        )
        
        # Plot features if we have both classes
        if len(df_sample['label'].unique()) > 1:
            plt.figure(figsize=(12, 8))
            
            # Plot based on available columns
            plot_idx = 1
            for feature in ['flesch_reading_ease', 'flesch_kincaid_grade', 'exclamation_count', 'uppercase_ratio']:
                if feature in df_sample.columns:
                    plt.subplot(2, 2, plot_idx)
                    sns.boxplot(x='label', y=feature, data=df_sample)
                    plt.title(f'{feature.replace("_", " ").title()} by Class')
                    plt.xlabel('Class (0=Real, 1=Fake)')
                    plot_idx += 1
            
            plt.tight_layout()
            plt.savefig("analysis/feature_engineering.png")
            
            # Calculate and print statistics
            stats_columns = [col for col in ['flesch_reading_ease', 'flesch_kincaid_grade', 
                         'exclamation_count', 'question_count', 'uppercase_ratio'] if col in df_sample.columns]
            
            if stats_columns:
                stats = df_sample.groupby('label')[stats_columns].mean()
                print("\nFeature statistics by class:")
                print(stats)
        
        return df_sample
    except Exception as e:
        print(f"Error in engineer_features: {str(e)}")
        return df

def analyze_clickbait_patterns(df):
    """Analyze clickbait patterns in headlines"""
    if df is None:
        print("Cannot analyze clickbait patterns: No data available")
        return None
        
    print("\n=== Clickbait Pattern Analysis ===")
    
    try:
        # Clickbait indicators
        clickbait_patterns = [
            r'\b(?:you won\'t believe|shocking|mind(-|\s)blowing)\b',
            r'\b(?:secret|trick|hack|miracle)\b',
            r'\b(?:amazing|incredible|unbelievable)\b',
            r'^\d+\s+',  # Starts with number (e.g., "10 ways to...")
            r'\?$',  # Ends with question mark
            r'\!$',  # Ends with exclamation point
            r'\b(?:this is why|here\'s why|the reason why)\b'
        ]
        
        # Get titles
        title_column = 'title' if 'title' in df.columns else 'text'
        
        # Function to check for clickbait patterns
        def check_clickbait(title):
            if not isinstance(title, str):
                return 0
                
            title = title.lower()
            count = 0
            for pattern in clickbait_patterns:
                if re.search(pattern, title):
                    count += 1
            return count
        
        # Count clickbait patterns
        df['clickbait_score'] = df[title_column].apply(check_clickbait)
        
        # Plot clickbait score distribution if we have both classes
        if len(df['label'].unique()) > 1:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='clickbait_score', hue='label', bins=range(0, 8), alpha=0.7)
            plt.title('Clickbait Score Distribution by Class')
            plt.xlabel('Clickbait Score')
            plt.ylabel('Count')
            plt.xticks(range(0, 8))
            plt.legend(['Real', 'Fake'])
            plt.savefig("analysis/clickbait_score.png")
        
        # Average clickbait score by class
        avg_clickbait = df.groupby('label')['clickbait_score'].mean()
        print("\nAverage clickbait score by class:")
        print(f"Real news: {avg_clickbait.get(0, 'N/A'):.2f}" if 0 in avg_clickbait else "Real news: N/A")
        print(f"Fake news: {avg_clickbait.get(1, 'N/A'):.2f}" if 1 in avg_clickbait else "Fake news: N/A")
        
        return df
    except Exception as e:
        print(f"Error in analyze_clickbait_patterns: {str(e)}")
        return df

def main():
    """Main function to run the analysis pipeline"""
    try:
        print("Starting Fake News Dataset Analysis...")
        
        # Load dataset
        df = safe_execution(load_dataset)
        if df is None or df.empty:
            print("Error: Unable to load dataset. Exiting...")
            return
            
        # Analyze dataset
        df = safe_execution(analyze_dataset, df)
        
        # Extract frequent words
        safe_execution(extract_frequent_words, df)
        
        # Engineer features
        df_with_features = safe_execution(engineer_features, df)
        
        # Analyze clickbait patterns
        safe_execution(analyze_clickbait_patterns, df)
        
        print("\nAnalysis completed successfully!")
        print("Check the 'analysis' directory for generated visualizations.")
        
    except Exception as e:
        print(f"An error occurred in the main function: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

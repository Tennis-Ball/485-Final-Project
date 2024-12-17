# data_preprocessing.py

import re
import logging
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Extend the default English stopwords with some financial-specific terms
# This list can be expanded based on domain knowledge or references.
financial_stopwords = {
    'factors', 'risk', 'company', 'securities', 'quarter', 'investment',
    'forward-looking', 'statements', 'factors', 'further', 'may', 'material',
    'results', 'financial', 'market', 'analysis', 'management', 'report', 'form',
    'sec', 'edgar', 'filer', 'commission', 'shares', 'stock', 'value', 'annual'
}

def remove_boilerplate(text):
    # Remove common boilerplate text patterns often found in SEC filings
    # For example: 
    # - Safe Harbor statements or repetitive disclaimers.
    # - We can identify these if they have characteristic phrases.
    # This is a heuristic approach and can be expanded as needed.
    
    # Example patterns that might appear frequently and add no value:
    boilerplate_patterns = [
        r'this quarterly report.*?exchange commission',  # common disclaimers
        r'forward-looking statements[^.]*\.',  # forward-looking disclaimers
        r'please see the section.*?for more information', 
        r'\bitem\s*\d+\.*',  # Remove item headings if any remain
        r'\bunited states securities and exchange commission\b'
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE|re.DOTALL)
    return text

def preprocess_text(text):
    # Remove boilerplate text first
    text = remove_boilerplate(text)
    
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = text.split()

    # Default English stopwords
    stop_words = set(stopwords.words('english'))
    # Combine with financial-specific stopwords
    all_stopwords = stop_words.union(financial_stopwords)

    # Remove stopwords
    words = [word for word in words if word not in all_stopwords]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join back into a string
    cleaned_text = ' '.join(words)
    return cleaned_text

def preprocess_filings(filings, min_length=100):
    # min_length: minimum number of tokens required to consider the document valid
    processed_filings = []
    for filing in filings:
        text = filing.get('Text', '')
        if not text.strip():
            logging.warning(f"No original text found for {filing.get('Ticker','Unknown')} on {filing.get('Filing Date','Unknown')}. Skipping.")
            continue

        cleaned_text = preprocess_text(text)
        tokens = cleaned_text.split()

        # Quality assurance checks
        if len(tokens) < min_length:
            logging.warning(f"Document for {filing['Ticker']} on {filing['Filing Date']} is too short after preprocessing ({len(tokens)} tokens). May contain insufficient meaningful text.")
        
        if not cleaned_text.strip():
            logging.warning(f"Document for {filing['Ticker']} on {filing['Filing Date']} is empty after preprocessing.")
            continue

        # Update the filing with cleaned text
        updated_filing = dict(filing)
        updated_filing['Cleaned Text'] = cleaned_text
        processed_filings.append(updated_filing)

    return processed_filings

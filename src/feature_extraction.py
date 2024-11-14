from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_features(filings, max_features=5000):
    documents = [filing['Cleaned Text'] for filing in filings]

    # Check the contents of documents
    print("Sample preprocessed documents:")
    for idx, doc in enumerate(documents[:5]):
        print(f"Document {idx+1} length: {len(doc)}")
        print(f"Content: {repr(doc[:500])}\n")  # Print first 500 characters

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(documents).toarray()
    return X, vectorizer

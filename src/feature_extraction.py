import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Try importing FinBERT and Transformers
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

from sklearn.decomposition import LatentDirichletAllocation

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Example financial sentiment lexicon
# In reality, this should be more comprehensive and sourced from a known financial sentiment dictionary.
FINANCIAL_SENTIMENT_LEXICON = {
    'positive': {'growth', 'profit', 'increase', 'opportunity', 'outperform', 'improvement'},
    'negative': {'loss', 'decline', 'risk', 'uncertain', 'downgrade', 'shortfall'}
}

RISK_KEYWORDS = {'risk', 'volatile', 'uncertain', 'exposure'}

def get_sentiment_scores(documents):
    """
    Compute simple sentiment scores based on a financial sentiment lexicon.
    Returns an array of shape (n_docs, 2): [positive_score, negative_score].
    """
    sentiment_features = []
    for doc in documents:
        tokens = doc.split()
        pos_count = sum(1 for t in tokens if t in FINANCIAL_SENTIMENT_LEXICON['positive'])
        neg_count = sum(1 for t in tokens if t in FINANCIAL_SENTIMENT_LEXICON['negative'])
        sentiment_features.append([pos_count, neg_count])
    return np.array(sentiment_features)

def get_numeric_features(documents):
    """
    Extract simple numeric features:
    - Document length (number of tokens)
    - Count of risk-related keywords
    """
    numeric_features = []
    for doc in documents:
        tokens = doc.split()
        length = len(tokens)
        risk_count = sum(1 for t in tokens if t in RISK_KEYWORDS)
        numeric_features.append([length, risk_count])
    return np.array(numeric_features)

def get_finbert_embeddings(documents, model_name='ProsusAI/finbert'):
    """
    Generate document-level embeddings using FinBERT.
    Takes the mean of the last hidden states for each document as its representation.
    """
    if not FINBERT_AVAILABLE:
        logging.warning("FinBERT or transformers not available. Skipping FinBERT embeddings.")
        return None

    logging.info("Loading FinBERT model for embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    for doc in documents:
        inputs = tokenizer(doc, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        all_embeddings.append(embeddings)

    return np.array(all_embeddings)

def fit_lda_topic_model(documents, n_topics=10, max_features=2000):
    """
    Fit an LDA model on the corpus to extract topic distributions.
    Returns the fitted LDA model and the feature matrix it was fit on.
    """
    logging.info("Fitting LDA topic model...")
    tf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tf)
    return lda, tf_vectorizer

def get_topic_features(documents, lda_model, tf_vectorizer):
    """
    Transform documents into topic distributions using a fitted LDA model.
    """
    tf = tf_vectorizer.transform(documents)
    topic_distributions = lda_model.transform(tf)
    return topic_distributions

def extract_features(
    filings, 
    use_finbert=True, 
    use_sentiment=True, 
    use_topic_modeling=True, 
    n_topics=10, 
    tfidf_max_features=5000
):
    """
    Extract combined features from the filings.
    Returns:
        X: Feature matrix combining TF-IDF, (optionally) FinBERT embeddings, sentiment, topic, and numeric features.
        vectorizer: The fitted TF-IDF vectorizer
        lda_model, lda_vectorizer: The fitted LDA model and vectorizer if topic modeling is used, else None.
    """
    documents = [filing['Cleaned Text'] for filing in filings]

    # Base TF-IDF features
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english')
    X_tfidf = vectorizer.fit_transform(documents).toarray()
    logging.info(f"TF-IDF feature shape: {X_tfidf.shape}")

    # FinBERT embeddings
    if use_finbert:
        finbert_embeddings = get_finbert_embeddings(documents)
    else:
        finbert_embeddings = None

    # Sentiment features
    if use_sentiment:
        sentiment_features = get_sentiment_scores(documents)
        logging.info(f"Sentiment feature shape: {sentiment_features.shape}")
    else:
        sentiment_features = None

    # Numeric features
    numeric_features = get_numeric_features(documents)
    logging.info(f"Numeric feature shape: {numeric_features.shape}")

    # Topic Modeling
    if use_topic_modeling:
        lda_model, lda_vectorizer = fit_lda_topic_model(documents, n_topics=n_topics)
        topic_features = get_topic_features(documents, lda_model, lda_vectorizer)
        logging.info(f"Topic feature shape: {topic_features.shape}")
    else:
        lda_model, lda_vectorizer = None, None
        topic_features = None

    # Combine all features
    feature_list = [X_tfidf, numeric_features]
    if finbert_embeddings is not None:
        feature_list.append(finbert_embeddings)
    if sentiment_features is not None:
        feature_list.append(sentiment_features)
    if topic_features is not None:
        feature_list.append(topic_features)

    X = np.concatenate(feature_list, axis=1)
    logging.info(f"Final combined feature shape: {X.shape}")

    return X, vectorizer, lda_model, lda_vectorizer

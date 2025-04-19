from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(corpus, max_features=500):
    """Compute TF-IDF features from cleaned text."""
    tfidf = TfidfVectorizer(max_features=max_features)
    return tfidf.fit_transform(corpus).toarray()

import pandas as pd
from src.preprocessing import apply_text_cleaning, normalize_metadata
from src.tfidf_vectorizer import compute_tfidf
from src.roberta_embedder import extract_embeddings
from src.feature_combiner import combine_features
from src.classifier import train_xgboost

def main():
    print("Loading data...")
    df = pd.read_csv("data/your_dataset.csv")

    # Update based on your actual metadata column names
    metadata_cols = ['metadata1', 'metadata2']

    print("Cleaning text...")
    df = apply_text_cleaning(df, text_column='text')

    print("Normalizing metadata...")
    df = normalize_metadata(df, metadata_cols)

    print("Generating TF-IDF features...")
    tfidf_features = compute_tfidf(df['clean_text'])

    print("Generating RoBERTa embeddings...")
    roberta_features = extract_embeddings(df['clean_text'])

    print("Combining features...")
    metadata_features = df[metadata_cols].values
    X = combine_features(tfidf_features, roberta_features, metadata_features)
    y = df['label'].map({'fake': 0, 'real': 1})

    print("Training XGBoost classifier...")
    train_xgboost(X, y)

if __name__ == "__main__":
    main()

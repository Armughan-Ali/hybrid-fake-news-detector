import re
import nltk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text: str) -> str:
    """Clean input text by removing URLs, special characters, numbers, and stopwords."""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in nltk.word_tokenize(text) if word not in stop_words])
    return text

def apply_text_cleaning(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Apply cleaning to an entire column of text data."""
    df['clean_text'] = df[text_column].apply(clean_text)
    return df

def normalize_metadata(df: pd.DataFrame, metadata_cols: list) -> pd.DataFrame:
    """Normalize metadata features using MinMaxScaler."""
    scaler = MinMaxScaler()
    df[metadata_cols] = scaler.fit_transform(df[metadata_cols])
    return df

import re
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

def custom_preprocessor(text):
    """
    Preprocess text by:
      - Lower-casing,
      - Removing punctuation,
      - Tokenizing with TreebankWordTokenizer (avoiding dependency on 'punkt'),
      - Lemmatizing tokens,
      - Removing extra whitespace.
    """
    text = text.lower()
    # Remove punctuation (replace with space)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

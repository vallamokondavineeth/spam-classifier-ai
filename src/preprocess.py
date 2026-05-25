print("Script started")
import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize stemmer
ps = PorterStemmer()

# Load dataset
df = pd.read_csv("data/spam.csv", encoding='latin-1')

# Keep useful columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Function for text preprocessing
def transform_text(text):

    # Lowercase
    text = text.lower()

    # Tokenization
    text = word_tokenize(text)

    y = []

    # Remove special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Apply preprocessing
df['transformed_message'] = df['message'].apply(transform_text)

# Display results
print("\nOriginal Message:\n")
print(df['message'][0])

print("\nProcessed Message:\n")
print(df['transformed_message'][0])

print("\nDataset Preview:\n")
print(df.head())
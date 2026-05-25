import pandas as pd
import string
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# INITIALIZATION
# -----------------------------------

ps = PorterStemmer()

# -----------------------------------
# LOAD DATASET
# -----------------------------------

print("\nLoading dataset...")

df = pd.read_csv("data/spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]

df.columns = ['label', 'message']

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert labels
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("Dataset loaded!")

# -----------------------------------
# TEXT PREPROCESSING
# -----------------------------------

def transform_text(text):

    # Lowercase
    text = text.lower()

    # Tokenization
    words = word_tokenize(text)

    filtered_words = []

    # Remove special characters
    for word in words:
        if word.isalnum():
            filtered_words.append(word)

    cleaned_words = []

    # Remove stopwords
    for word in filtered_words:
        if (
            word not in stopwords.words('english')
            and word not in string.punctuation
        ):
            cleaned_words.append(word)

    stemmed_words = []

    # Stemming
    for word in cleaned_words:
        stemmed_words.append(ps.stem(word))

    return " ".join(stemmed_words)

print("\nPreprocessing text...")

df['transformed_message'] = df['message'].apply(transform_text)

print("Preprocessing completed!")

# -----------------------------------
# TF-IDF VECTORIZATION
# -----------------------------------

print("\nCreating TF-IDF vectors...")

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = tfidf.fit_transform(df['transformed_message'])

y = df['label']

print("Vectorization completed!")

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------
# MODEL TRAINING
# -----------------------------------

print("\nTraining model...")

model = ComplementNB()

model.fit(X_train, y_train)

print("Training completed!")

# -----------------------------------
# PREDICTIONS
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# EVALUATION
# -----------------------------------

print("\nMODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------------
# SAVE MODEL
# -----------------------------------

print("\nSaving model and vectorizer...")

joblib.dump(
    model,
    "models/spam_classifier.pkl"
)

joblib.dump(
    tfidf,
    "models/vectorizer.pkl"
)

print("\nModel saved successfully!")
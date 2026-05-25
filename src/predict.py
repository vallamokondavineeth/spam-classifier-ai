import joblib
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

ps = PorterStemmer()

# -----------------------------------
# TEXT PREPROCESSING
# -----------------------------------

def transform_text(text):

    text = text.lower()

    words = word_tokenize(text)

    filtered_words = []

    for word in words:
        if word.isalnum():
            filtered_words.append(word)

    cleaned_words = []

    for word in filtered_words:
        if (
            word not in stopwords.words('english')
            and word not in string.punctuation
        ):
            cleaned_words.append(word)

    stemmed_words = []

    for word in cleaned_words:
        stemmed_words.append(ps.stem(word))

    return " ".join(stemmed_words)

# -----------------------------------
# USER INPUT
# -----------------------------------

message = input("\nEnter a message: ")

# -----------------------------------
# PREPROCESS
# -----------------------------------

transformed_message = transform_text(message)

# -----------------------------------
# VECTORIZE
# -----------------------------------

vector_input = vectorizer.transform([transformed_message])

# -----------------------------------
# PREDICT
# -----------------------------------

prediction = model.predict(vector_input)[0]

probability = model.predict_proba(vector_input)[0]

# -----------------------------------
# OUTPUT
# -----------------------------------

print("\nPrediction Result")
print("=" * 30)

if prediction == 1:
    print("SPAM MESSAGE")
    print(f"Confidence: {probability[1] * 100:.2f}%")
else:
    print("NOT SPAM")
    print(f"Confidence: {probability[0] * 100:.2f}%")
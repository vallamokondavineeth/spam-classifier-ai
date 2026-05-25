from flask import Flask, render_template, request
import joblib
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = joblib.load("../models/spam_classifier.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

ps = PorterStemmer()

# -----------------------------------
# FLASK APP
# -----------------------------------

app = Flask(__name__)

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
# HOME ROUTE
# -----------------------------------

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None

    if request.method == "POST":

        message = request.form["message"]

        transformed_message = transform_text(message)

        vector_input = vectorizer.transform(
            [transformed_message]
        )

        result = model.predict(vector_input)[0]

        if result == 1:
            prediction = "SPAM MESSAGE"
        else:
            prediction = "NOT SPAM"

    return render_template(
        "index.html",
        prediction=prediction
    )

# -----------------------------------
# RUN APP
# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)
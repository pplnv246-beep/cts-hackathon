import re
import joblib

from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "final_sentiment_model.pkl"
)

VECTORIZER_FILE = (
    BASE_DIR
    / "models"
    / "final_tfidf_vectorizer.pkl"
)


# ============================================================
# NLP SETUP
# ============================================================

stop_words = set(
    stopwords.words("english")
)

important_words = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "hardly",
    "barely",
    "nothing"
}

stop_words = stop_words - important_words

lemmatizer = WordNetLemmatizer()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    processed_words = []

    for word in words:

        if word not in stop_words:

            processed_words.append(
                lemmatizer.lemmatize(word)
            )

    return " ".join(
        processed_words
    )


# ============================================================
# LOAD FINAL MODEL
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - FINAL MODEL")
print("=" * 70)

print("\nLoading final model...")

model = joblib.load(
    MODEL_FILE
)

vectorizer = joblib.load(
    VECTORIZER_FILE
)

print("Final model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

def predict_sentiment(review):

    processed_review = preprocess_text(
        review
    )

    vector = vectorizer.transform(
        [processed_review]
    )

    prediction = model.predict(
        vector
    )[0]

    probabilities = model.predict_proba(
        vector
    )[0]

    confidence = probabilities.max()

    probability_dict = dict(
        zip(
            model.classes_,
            probabilities
        )
    )

    return (
        prediction,
        confidence,
        probability_dict
    )


# ============================================================
# TEST REVIEWS
# ============================================================

if __name__ == "__main__":

    test_reviews = [

        "The product is amazing and I really love it.",

        "The product is terrible and completely useless.",

        "The product is okay, nothing special.",

        "I am extremely disappointed with this product.",

        "The delivery was fast and the product quality is excellent."
    ]


    for review in test_reviews:

        sentiment, confidence, probabilities = (
            predict_sentiment(review)
        )

        print("\n" + "-" * 70)

        print("Review:")
        print(review)

        print(
            "\nPredicted Sentiment:",
            sentiment
        )

        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )

        print("\nClass Probabilities:")

        for label, probability in (
            probabilities.items()
        ):

            print(
                f"{label}: "
                f"{probability * 100:.2f}%"
            )


    print("\n" + "=" * 70)

    print("FINAL MODEL TESTING COMPLETED")

    print("=" * 70)
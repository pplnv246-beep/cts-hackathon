import os
import joblib

from backend.ml.concern_detection import detect_concerns


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_tfidf_vectorizer.pkl"
)


model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment(review: str):
    features = vectorizer.transform([review])

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    classes = model.classes_

    probability_dict = {
        str(class_name): float(probability)
        for class_name, probability in zip(classes, probabilities)
    }

    confidence = max(probabilities)

    concerns = detect_concerns(review)

    return {
        "sentiment": str(prediction),
        "confidence": float(confidence),
        "probabilities": probability_dict,
        "concerns": concerns
    }
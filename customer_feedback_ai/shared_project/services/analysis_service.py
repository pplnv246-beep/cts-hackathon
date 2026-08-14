import os
import pandas as pd
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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def find_review_column(df):

    possible_columns = [
        "Review",
        "Review Text",
        "review",
        "review_text",
        "ReviewText",
        "Feedback",
        "feedback",
        "Comment",
        "comment",
        "Customer Feedback",
        "customer_feedback"
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    # Case-insensitive matching
    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for column in possible_columns:
        key = column.lower()

        if key in normalized_columns:
            return normalized_columns[key]

    return None


def analyze_csv(file_path):

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    review_column = find_review_column(df)

    if review_column is None:
        raise ValueError(
            "No review column found. Expected columns such as "
            "'Review', 'Review Text', 'Feedback', or 'Comment'."
        )

    reviews = df[review_column].fillna("").astype(str)

    # Sentiment prediction for all reviews
    features = vectorizer.transform(reviews)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    confidence = probabilities.max(axis=1)

    df["Predicted_Sentiment"] = predictions
    df["Prediction_Confidence"] = confidence

    # Concern detection
    concerns = []

    for review in reviews:
        detected = detect_concerns(review)

        if detected:
            concerns.append(", ".join(detected))
        else:
            concerns.append("None")

    df["Detected_Concerns"] = concerns

    # Sentiment summary
    sentiment_counts = df["Predicted_Sentiment"].value_counts()

    total_reviews = len(df)

    positive = int(sentiment_counts.get("Positive", 0))
    negative = int(sentiment_counts.get("Negative", 0))
    neutral = int(sentiment_counts.get("Neutral", 0))

    # Save analyzed dataset
    output_file = os.path.join(
        OUTPUT_DIR,
        "uploaded_analyzed_reviews.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    return {
        "total_reviews": total_reviews,
        "review_column": review_column,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "output_file": output_file
    }
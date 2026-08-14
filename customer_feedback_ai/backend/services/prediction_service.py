import os
import joblib

from backend.ml.concern_detection import detect_concerns
from backend.ml.text_sentiment_analyzer import analyze_text_sentiment


# ============================================================
# CUSTOMER FEEDBACK AI - PREDICTION SERVICE
# ============================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# MODEL PATHS
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


vectorizer = joblib.load(
    VECTORIZER_PATH
)


# ============================================================
# PREDICT SENTIMENT
# ============================================================

def predict_sentiment(
    review: str
):

    # ========================================================
    # VALIDATE REVIEW
    # ========================================================

    if review is None:

        raise ValueError(
            "Review cannot be empty."
        )


    review = str(
        review
    ).strip()


    if not review:

        raise ValueError(
            "Review cannot be empty."
        )


    # ========================================================
    # MACHINE LEARNING PREDICTION
    # ========================================================

    features = vectorizer.transform(
        [review]
    )


    prediction = model.predict(
        features
    )[0]


    probabilities = model.predict_proba(
        features
    )[0]


    classes = model.classes_


    probability_dict = {

        str(class_name):
            float(probability)

        for class_name, probability
        in zip(
            classes,
            probabilities
        )

    }


    confidence = max(
        probabilities
    )


    # ========================================================
    # CONFIDENCE LEVEL
    # ========================================================

    if confidence >= 0.75:

        confidence_level = "High"

    elif confidence >= 0.50:

        confidence_level = "Moderate"

    else:

        confidence_level = "Low"


    # ========================================================
    # CUSTOMER CONCERNS
    # ========================================================

    concerns = detect_concerns(
        review
    )


    # ========================================================
    # TEXT SENTIMENT ANALYSIS
    # ========================================================

    text_analysis = analyze_text_sentiment(
        review
    )


    text_sentiment = (
        text_analysis[
            "text_sentiment"
        ]
    )


    negative_evidence = (
        text_analysis[
            "negative_evidence"
        ]
    )


    positive_evidence = (
        text_analysis[
            "positive_evidence"
        ]
    )


    evidence = (
        text_analysis[
            "evidence"
        ]
    )


    # ========================================================
    # DETECT POSSIBLE AMBIGUITY
    # ========================================================

    prediction_adjusted = False

    adjustment_reason = None


    # --------------------------------------------------------
    # Case 1:
    # ML and text strongly agree
    # --------------------------------------------------------

    if (
        prediction == text_sentiment
        and text_sentiment in [
            "Positive",
            "Negative"
        ]
    ):

        adjustment_reason = (
            "ML prediction and text evidence agree."
        )


    # --------------------------------------------------------
    # Case 2:
    # ML prediction differs from text sentiment
    # --------------------------------------------------------

    elif (
        text_sentiment in [
            "Positive",
            "Negative"
        ]
        and prediction != text_sentiment
    ):

        adjustment_reason = (
            "ML prediction differs from "
            "strong text sentiment evidence."
        )


    # --------------------------------------------------------
    # Case 3:
    # Text is mixed
    # --------------------------------------------------------

    elif text_sentiment == "Mixed":

        adjustment_reason = (
            "Review contains both positive "
            "and negative sentiment evidence."
        )


    # --------------------------------------------------------
    # Case 4:
    # No strong text evidence
    # --------------------------------------------------------

    else:

        adjustment_reason = (
            "No strong sentiment phrases "
            "were detected in the review."
        )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "sentiment":
            str(prediction),

        "confidence":
            float(confidence),

        "confidence_level":
            confidence_level,

        "probabilities":
            probability_dict,

        "concerns":
            concerns,

        "text_sentiment":
            text_sentiment,

        "negative_evidence":
            negative_evidence,

        "positive_evidence":
            positive_evidence,

        "evidence":
            evidence,

        "prediction_adjusted":
            prediction_adjusted,

        "adjustment_reason":
            adjustment_reason

    }
import os
import warnings

warnings.filterwarnings("ignore")

try:
    import threadpoolctl
    orig_find = getattr(threadpoolctl.ThreadpoolController, "_find_libraries_with_enum_process_module_ex", None)
    if orig_find:
        def safe_find(self):
            try:
                return orig_find(self)
            except Exception:
                return []
        threadpoolctl.ThreadpoolController._find_libraries_with_enum_process_module_ex = safe_find
except Exception:
    pass

import pandas as pd
import joblib

from backend.ml.concern_detection import detect_concerns
from backend.services.upload_service import load_csv
from backend.ml.ai_summary import generate_ai_summary


# ============================================================
# CUSTOMER FEEDBACK AI - ANALYSIS SERVICE
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
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

model = joblib.load(
    MODEL_PATH
)

if not hasattr(model, "multi_class"):
    model.multi_class = "auto"

vectorizer = joblib.load(
    VECTORIZER_PATH
)


# ============================================================
# FIND REVIEW COLUMN
# ============================================================

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

    # Exact match
    for column in possible_columns:

        if column in df.columns:
            return column

    # Case-insensitive match
    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for column in possible_columns:

        key = column.lower()

        if key in normalized_columns:
            return normalized_columns[key]

    return None


# ============================================================
# ANALYZE CSV
# ============================================================

def analyze_csv(file_path):

    print()
    print("=" * 70)
    print("CUSTOMER FEEDBACK AI - CSV ANALYSIS")
    print("=" * 70)

    print()
    print("Loading CSV...")

    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------

    df = load_csv(
        file_path
    )

    print(
        "CSV loaded successfully."
    )

    print(
        "Total rows:",
        len(df)
    )

    print()

    print(
        "Available columns:"
    )

    print(
        df.columns.tolist()
    )

    print()

    # ========================================================
    # FIND REVIEW COLUMN
    # ========================================================

    review_column = find_review_column(
        df
    )

    if review_column is None:

        raise ValueError(
            "No review column found. "
            "Expected columns such as "
            "'Review', 'Review Text', "
            "'Feedback', or 'Comment'."
        )

    print(
        "Review column:",
        review_column
    )

    # ========================================================
    # PREPARE REVIEWS
    # ========================================================

    reviews = (
        df[review_column]
        .fillna("")
        .astype(str)
    )

    # ========================================================
    # SENTIMENT PREDICTION
    # ========================================================

    print()
    print(
        "Running sentiment prediction..."
    )

    features = vectorizer.transform(
        reviews
    )

    predictions = model.predict(
        features
    )

    probabilities = model.predict_proba(
        features
    )

    confidence = probabilities.max(
        axis=1
    )

    # Add predictions
    df["Predicted_Sentiment"] = predictions

    df["Prediction_Confidence"] = (
        confidence.round(4)
    )

    print(
        "Sentiment prediction completed."
    )

    # ========================================================
    # CONCERN DETECTION
    # ========================================================

    print()
    print(
        "Detecting customer concerns..."
    )

    concerns = []

    total_reviews = len(reviews)

    for index, review in enumerate(
        reviews,
        start=1
    ):

        detected = detect_concerns(
            review
        )

        if detected:

            concerns.append(
                "; ".join(detected)
            )

        else:

            concerns.append(
                "None"
            )

        # Progress message
        if index % 1000 == 0:

            print(
                f"Processed {index}/{total_reviews} reviews..."
            )

    df["Detected_Concerns"] = concerns

    print(
        "Concern detection completed."
    )

    # ========================================================
    # SENTIMENT SUMMARY
    # ========================================================

    sentiment_counts = (
        df["Predicted_Sentiment"]
        .value_counts()
    )

    positive = int(
        sentiment_counts.get(
            "Positive",
            0
        )
    )

    negative = int(
        sentiment_counts.get(
            "Negative",
            0
        )
    )

    neutral = int(
        sentiment_counts.get(
            "Neutral",
            0
        )
    )

    # ========================================================
    # CONCERN SUMMARY
    # ========================================================

    concern_counts = {}

    for concerns_text in df[
        "Detected_Concerns"
    ]:

        if concerns_text == "None":
            continue

        concern_list = (
            concerns_text.split("; ")
        )

        for concern in concern_list:

            concern_counts[concern] = (
                concern_counts.get(
                    concern,
                    0
                ) + 1
            )

    # ========================================================
    # SORT CONCERNS
    # ========================================================

    concern_distribution = dict(
        sorted(
            concern_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # ========================================================
    # SAVE ANALYZED DATASET
    # ========================================================

    output_file = os.path.join(
        OUTPUT_DIR,
        "uploaded_analyzed_reviews.csv"
    )

    print()
    print(
        "Saving analyzed dataset..."
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        "Analyzed dataset saved."
    )

    print(
        output_file
    )

    # ========================================================
    # GENERATE CURRENT AI SUMMARY
    # ========================================================

    print()
    print(
        "Generating AI business summary..."
    )

    try:

        ai_summary = generate_ai_summary()

        print(
            "AI business summary generated successfully."
        )

        print(
            "Summary:",
            ai_summary.get(
                "summary",
                ""
            )
        )

    except Exception as error:

        print(
            "WARNING: AI summary generation failed."
        )

        print(
            "Reason:",
            error
        )

        # Do not stop the complete CSV analysis
        # if summary generation fails.

        ai_summary = None

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("CSV ANALYSIS COMPLETED")
    print("=" * 70)

    print()

    print(
        "Total reviews:",
        total_reviews
    )

    print(
        "Positive:",
        positive
    )

    print(
        "Negative:",
        negative
    )

    print(
        "Neutral:",
        neutral
    )

    print()

    print(
        "Top concerns:"
    )

    for concern, count in list(
        concern_distribution.items()
    )[:10]:

        print(
            f"{concern:<25} {count}"
        )

    print()

    print(
        "Output file:"
    )

    print(
        output_file
    )

    print()

    print(
        "AI summary:"
    )

    if ai_summary:

        print(
            ai_summary.get(
                "summary",
                "Generated successfully."
            )
        )

    else:

        print(
            "Not generated."
        )

    print()

    print("=" * 70)

    # ========================================================
    # RETURN API RESULT
    # ========================================================

    return {
        "total_reviews": total_reviews,
        "review_column": review_column,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "concern_distribution": concern_distribution,
        "output_file": output_file,
        "ai_summary": ai_summary
    }
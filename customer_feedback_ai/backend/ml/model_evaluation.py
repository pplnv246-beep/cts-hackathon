import os
import re
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CUSTOMER FEEDBACK AI - MODEL EVALUATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# PATHS
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

TEST_FILE = os.path.join(
    BASE_DIR,
    "data",
    "uploads",
    "test.csv"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "model"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# FIND COLUMNS
# ============================================================

def find_rating_column(df):

    possible_columns = [
        "Rating",
        "rating",
        "Stars",
        "stars"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    raise ValueError(
        "Rating column not found. "
        f"Available columns: {df.columns.tolist()}"
    )


def find_review_column(df):

    possible_columns = [
        "Review Text",
        "Review",
        "review",
        "review_text",
        "ReviewText",
        "Feedback",
        "feedback",
        "Comment",
        "comment"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    raise ValueError(
        "Review text column not found. "
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# CONVERT RATING TO SENTIMENT
# ============================================================

def rating_to_sentiment(rating):

    if pd.isna(rating):
        return None

    text = str(rating).strip().lower()

    match = re.search(
        r"([1-5])",
        text
    )

    if not match:
        return None

    stars = int(
        match.group(1)
    )

    if stars <= 2:

        return "Negative"

    elif stars == 3:

        return "Neutral"

    else:

        return "Positive"


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_model():

    print()
    print("=" * 70)
    print("CUSTOMER FEEDBACK AI - MODEL EVALUATION")
    print("=" * 70)
    print()

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(VECTORIZER_PATH):

        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_PATH}"
        )

    if not os.path.exists(TEST_FILE):

        raise FileNotFoundError(
            f"Test CSV not found: {TEST_FILE}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("Loading trained model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    print()

    # ========================================================
    # LOAD VECTORIZER
    # ========================================================

    print("Loading TF-IDF vectorizer...")

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    print(
        "Vectorizer loaded successfully."
    )

    print()

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print("Loading test dataset...")

    try:

        df = pd.read_csv(
            TEST_FILE,
            low_memory=False
        )

    except Exception:

        df = pd.read_csv(
            TEST_FILE,
            engine="python"
        )

    print(
        "Dataset loaded successfully."
    )

    print(
        "Total rows:",
        len(df)
    )

    print()

    # ========================================================
    # FIND COLUMNS
    # ========================================================

    rating_column = find_rating_column(
        df
    )

    review_column = find_review_column(
        df
    )

    print(
        "Rating column:",
        rating_column
    )

    print(
        "Review column:",
        review_column
    )

    print()

    # ========================================================
    # CREATE GROUND TRUTH
    # ========================================================

    df["Actual_Sentiment"] = (
        df[rating_column]
        .apply(
            rating_to_sentiment
        )
    )

    # Remove unrated reviews
    evaluation_df = df[
        df["Actual_Sentiment"].notna()
    ].copy()

    print(
        "Rated reviews:",
        len(evaluation_df)
    )

    print(
        "Excluded unrated reviews:",
        len(df) - len(evaluation_df)
    )

    print()

    # ========================================================
    # GROUND TRUTH DISTRIBUTION
    # ========================================================

    actual_counts = (
        evaluation_df[
            "Actual_Sentiment"
        ]
        .value_counts()
    )

    print(
        "GROUND TRUTH DISTRIBUTION"
    )

    print("-" * 40)

    print(
        "Negative:",
        int(
            actual_counts.get(
                "Negative",
                0
            )
        )
    )

    print(
        "Neutral:",
        int(
            actual_counts.get(
                "Neutral",
                0
            )
        )
    )

    print(
        "Positive:",
        int(
            actual_counts.get(
                "Positive",
                0
            )
        )
    )

    print()

    # ========================================================
    # PREPARE TEXT
    # ========================================================

    reviews = (
        evaluation_df[
            review_column
        ]
        .fillna("")
        .astype(str)
    )

    # ========================================================
    # TRANSFORM TEXT
    # ========================================================

    print(
        "Transforming reviews using TF-IDF..."
    )

    features = vectorizer.transform(
        reviews
    )

    print(
        "TF-IDF transformation completed."
    )

    print()

    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "Running model predictions..."
    )

    predictions = model.predict(
        features
    )

    evaluation_df[
        "Predicted_Sentiment"
    ] = predictions

    print(
        "Predictions completed."
    )

    print()

    # ========================================================
    # LABEL NORMALIZATION
    # ========================================================

    evaluation_df[
        "Predicted_Sentiment"
    ] = (
        evaluation_df[
            "Predicted_Sentiment"
        ]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # ========================================================
    # METRICS
    # ========================================================

    y_true = evaluation_df[
        "Actual_Sentiment"
    ]

    y_pred = evaluation_df[
        "Predicted_Sentiment"
    ]

    labels = [
        "Negative",
        "Neutral",
        "Positive"
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )

    # ========================================================
    # PRINT OVERALL METRICS
    # ========================================================

    print("=" * 70)

    print(
        "MODEL PERFORMANCE"
    )

    print("=" * 70)

    print()

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score : {f1 * 100:.2f}%"
    )

    print()

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("=" * 70)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 70)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        zero_division=0
    )

    print(
        report
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("=" * 70)

    print(
        "CONFUSION MATRIX"
    )

    print("=" * 70)

    print()

    print(
        "Rows    = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print()

    print(
        "                 Predicted"
    )

    print(
        "              Neg    Neu    Pos"
    )

    print(
        f"Actual Neg   {cm[0][0]:5d}  "
        f"{cm[0][1]:5d}  "
        f"{cm[0][2]:5d}"
    )

    print(
        f"Actual Neu   {cm[1][0]:5d}  "
        f"{cm[1][1]:5d}  "
        f"{cm[1][2]:5d}"
    )

    print(
        f"Actual Pos   {cm[2][0]:5d}  "
        f"{cm[2][1]:5d}  "
        f"{cm[2][2]:5d}"
    )

    print()

    # ========================================================
    # PREDICTED DISTRIBUTION
    # ========================================================

    predicted_counts = (
        y_pred.value_counts()
    )

    print("=" * 70)

    print(
        "PREDICTED DISTRIBUTION"
    )

    print("=" * 70)

    print()

    for label in labels:

        print(
            f"{label}: "
            f"{int(predicted_counts.get(label, 0))}"
        )

    print()

    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Negative",
            "Actual Neutral",
            "Actual Positive"
        ],
        columns=[
            "Predicted Negative",
            "Predicted Neutral",
            "Predicted Positive"
        ]
    )

    cm_file = os.path.join(
        REPORT_DIR,
        "confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_file
    )

    # ========================================================
    # SAVE EVALUATION REPORT
    # ========================================================

    report_file = os.path.join(
        REPORT_DIR,
        "model_evaluation.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "CUSTOMER FEEDBACK AI - MODEL EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Total dataset rows: {len(df)}\n"
        )

        file.write(
            f"Rated reviews evaluated: "
            f"{len(evaluation_df)}\n"
        )

        file.write(
            f"Unrated reviews excluded: "
            f"{len(df) - len(evaluation_df)}\n\n"
        )

        file.write(
            "GROUND TRUTH RULE\n"
        )

        file.write(
            "1-2 stars = Negative\n"
        )

        file.write(
            "3 stars   = Neutral\n"
        )

        file.write(
            "4-5 stars = Positive\n\n"
        )

        file.write(
            f"Accuracy : {accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Precision: {precision * 100:.2f}%\n"
        )

        file.write(
            f"Recall   : {recall * 100:.2f}%\n"
        )

        file.write(
            f"F1 Score : {f1 * 100:.2f}%\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            report
        )

        file.write(
            "\nCONFUSION MATRIX\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            str(cm_df)
        )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_file = os.path.join(
        REPORT_DIR,
        "test_predictions.csv"
    )

    evaluation_df.to_csv(
        prediction_file,
        index=False
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("=" * 70)

    print(
        "EVALUATION COMPLETED"
    )

    print("=" * 70)

    print()

    print(
        "Evaluation report:"
    )

    print(
        report_file
    )

    print()

    print(
        "Confusion matrix:"
    )

    print(
        cm_file
    )

    print()

    print(
        "Prediction file:"
    )

    print(
        prediction_file
    )

    print()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    evaluate_model()
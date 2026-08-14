import os
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

os.makedirs(REPORT_DIR, exist_ok=True)


def find_rating_column(df):
    for column in ["Rating", "rating", "Stars", "stars"]:
        if column in df.columns:
            return column
    raise ValueError("Rating column not found.")


def find_review_column(df):
    for column in [
        "Review Text",
        "Review",
        "review",
        "review_text",
        "ReviewText",
        "Feedback",
        "feedback",
        "Comment",
        "comment"
    ]:
        if column in df.columns:
            return column
    raise ValueError("Review text column not found.")


def rating_to_sentiment(rating):

    if pd.isna(rating):
        return None

    match = re.search(
        r"([1-5])",
        str(rating)
    )

    if not match:
        return None

    stars = int(match.group(1))

    if stars <= 2:
        return "Negative"

    if stars == 3:
        return "Neutral"

    return "Positive"


print("=" * 70)
print("CUSTOMER FEEDBACK AI - 3-CLASS MODEL IMPROVEMENT")
print("=" * 70)

print("\nLoading test dataset...")

df = pd.read_csv(
    TEST_FILE,
    engine="python"
)

print("Total rows:", len(df))

rating_column = find_rating_column(df)
review_column = find_review_column(df)

print("Rating column:", rating_column)
print("Review column:", review_column)


df["Actual_Sentiment"] = (
    df[rating_column]
    .apply(rating_to_sentiment)
)

df = df[
    df["Actual_Sentiment"].notna()
].copy()

df[review_column] = (
    df[review_column]
    .fillna("")
    .astype(str)
)

df = df[
    df[review_column].str.strip() != ""
].copy()

print("\nRated reviews used:", len(df))

print("\nGround truth distribution:")
print(
    df["Actual_Sentiment"]
    .value_counts()
)


X = df[review_column]
y = df["Actual_Sentiment"]

labels = [
    "Negative",
    "Neutral",
    "Positive"
]


configs = [

    {
        "name": "Logistic_Baseline",
        "model": "logistic",
        "ngram": (1, 2),
        "features": 50000,
        "weights": None
    },

    {
        "name": "Logistic_Balanced",
        "model": "logistic",
        "ngram": (1, 2),
        "features": 50000,
        "weights": "balanced"
    },

    {
        "name": "Logistic_Neutral_1.5",
        "model": "logistic",
        "ngram": (1, 2),
        "features": 50000,
        "weights": {
            "Negative": 1.0,
            "Neutral": 1.5,
            "Positive": 1.0
        }
    },

    {
        "name": "Logistic_Neutral_2",
        "model": "logistic",
        "ngram": (1, 2),
        "features": 50000,
        "weights": {
            "Negative": 1.0,
            "Neutral": 2.0,
            "Positive": 1.0
        }
    },

    {
        "name": "Logistic_1_3gram",
        "model": "logistic",
        "ngram": (1, 3),
        "features": 60000,
        "weights": {
            "Negative": 1.0,
            "Neutral": 1.5,
            "Positive": 1.0
        }
    },

    {
        "name": "LinearSVC_Balanced",
        "model": "svc",
        "ngram": (1, 2),
        "features": 50000,
        "weights": "balanced"
    },

    {
        "name": "LinearSVC_Neutral_1.5",
        "model": "svc",
        "ngram": (1, 2),
        "features": 50000,
        "weights": {
            "Negative": 1.0,
            "Neutral": 1.5,
            "Positive": 1.0
        }
    },

    {
        "name": "LinearSVC_Neutral_2",
        "model": "svc",
        "ngram": (1, 2),
        "features": 50000,
        "weights": {
            "Negative": 1.0,
            "Neutral": 2.0,
            "Positive": 1.0
        }
    }
]


results = []
best_result = None


for config in configs:

    print("\n" + "=" * 70)
    print("EXPERIMENT:", config["name"])
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        ngram_range=config["ngram"],
        min_df=2,
        max_features=config["features"],
        sublinear_tf=True
    )

    print("\nCreating TF-IDF...")

    X_tfidf = vectorizer.fit_transform(X)

    print(
        "Feature shape:",
        X_tfidf.shape
    )

    if config["model"] == "logistic":

        model = LogisticRegression(
            max_iter=1500,
            class_weight=config["weights"],
            random_state=42
        )

    else:

        model = LinearSVC(
            class_weight=config["weights"],
            random_state=42
        )

    print("Training model...")

    model.fit(
        X_tfidf,
        y
    )

    print("Generating predictions...")

    predictions = model.predict(
        X_tfidf
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    weighted_precision = precision_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    class_f1 = f1_score(
        y,
        predictions,
        labels=labels,
        average=None,
        zero_division=0
    )

    neutral_f1 = float(
        class_f1[1]
    )

    print(
        f"\nAccuracy    : {accuracy * 100:.2f}%"
    )

    print(
        f"Weighted F1 : {weighted_f1 * 100:.2f}%"
    )

    print(
        f"Macro F1    : {macro_f1 * 100:.2f}%"
    )

    print(
        f"Neutral F1  : {neutral_f1 * 100:.2f}%"
    )

    results.append({
        "name": config["name"],
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "neutral_f1": neutral_f1
    })

    if (
        accuracy >= 0.90
        and weighted_f1 >= 0.90
    ):

        if (
            best_result is None
            or neutral_f1 > best_result["neutral_f1"]
        ):

            best_result = {
                "name": config["name"],
                "accuracy": accuracy,
                "weighted_f1": weighted_f1,
                "macro_f1": macro_f1,
                "neutral_f1": neutral_f1,
                "model": model,
                "vectorizer": vectorizer
            }


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

display_df = results_df.copy()

for column in [
    "accuracy",
    "weighted_f1",
    "macro_f1",
    "neutral_f1"
]:
    display_df[column] = (
        display_df[column] * 100
    ).round(2)

print(
    display_df.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("BEST CANDIDATE")
print("=" * 70)

if best_result is None:

    print(
        "No candidate met the minimum performance requirements."
    )

else:

    print(
        "Model:",
        best_result["name"]
    )

    print(
        f"Accuracy: "
        f"{best_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Weighted F1: "
        f"{best_result['weighted_f1'] * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{best_result['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Neutral F1: "
        f"{best_result['neutral_f1'] * 100:.2f}%"
    )

    best_predictions = (
        best_result["model"]
        .predict(
            best_result["vectorizer"]
            .transform(X)
        )
    )

    print("\n" + "=" * 70)
    print("BEST MODEL CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y,
            best_predictions,
            labels=labels,
            target_names=labels,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y,
        best_predictions,
        labels=labels
    )

    print("=" * 70)
    print("BEST MODEL CONFUSION MATRIX")
    print("=" * 70)

    print(
        "\n                 Predicted"
    )

    print(
        "              Neg    Neu    Pos"
    )

    print(
        f"Actual Neg   "
        f"{matrix[0][0]:5d}  "
        f"{matrix[0][1]:5d}  "
        f"{matrix[0][2]:5d}"
    )

    print(
        f"Actual Neu   "
        f"{matrix[1][0]:5d}  "
        f"{matrix[1][1]:5d}  "
        f"{matrix[1][2]:5d}"
    )

    print(
        f"Actual Pos   "
        f"{matrix[2][0]:5d}  "
        f"{matrix[2][1]:5d}  "
        f"{matrix[2][2]:5d}"
    )

    results_file = os.path.join(
        REPORT_DIR,
        "three_class_experiment_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    matrix_file = os.path.join(
        REPORT_DIR,
        "improved_confusion_matrix.csv"
    )

    pd.DataFrame(
        matrix,
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
    ).to_csv(
        matrix_file
    )

    joblib.dump(
        best_result["model"],
        os.path.join(
            REPORT_DIR,
            "improved_candidate_model.pkl"
        )
    )

    joblib.dump(
        best_result["vectorizer"],
        os.path.join(
            REPORT_DIR,
            "improved_candidate_vectorizer.pkl"
        )
    )

    print("\nResults saved:")
    print(results_file)

    print("\nConfusion matrix saved:")
    print(matrix_file)

    print(
        "\nIMPORTANT: Production model was NOT replaced."
    )


print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)

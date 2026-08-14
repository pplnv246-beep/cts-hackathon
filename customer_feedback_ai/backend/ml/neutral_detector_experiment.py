import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CUSTOMER FEEDBACK AI
# NEUTRAL DETECTOR EXPERIMENT
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - NEUTRAL DETECTOR EXPERIMENT")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nlp_ready_reviews.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_FILE
)

print(
    "Total reviews:",
    len(df)
)


# ============================================================
# FIND TEXT COLUMN
# ============================================================

if "Processed_Review" in df.columns:

    text_column = "Processed_Review"

elif "Cleaned_Review" in df.columns:

    text_column = "Cleaned_Review"

elif "Review" in df.columns:

    text_column = "Review"

else:

    raise ValueError(
        "No review text column found."
    )


print(
    "Text column:",
    text_column
)


# ============================================================
# CLEAN DATA
# ============================================================

df[text_column] = (
    df[text_column]
    .fillna("")
    .astype(str)
)


df = df[
    df[text_column].str.strip() != ""
].copy()


# ============================================================
# CREATE BINARY TARGET
# ============================================================

# Neutral = 1
# Negative / Positive = 0

df["Is_Neutral"] = (
    df["Sentiment"]
    == "Neutral"
).astype(int)


print("\n" + "=" * 70)
print("BINARY TARGET DISTRIBUTION")
print("=" * 70)

print(
    df["Is_Neutral"]
    .value_counts()
    .rename(
        index={
            0: "Non-Neutral",
            1: "Neutral"
        }
    )
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X = df[text_column]

y = df["Is_Neutral"]


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("CREATING TF-IDF FEATURES")
print("=" * 70)


vectorizer = TfidfVectorizer(

    ngram_range=(1, 3),

    min_df=2,

    max_df=0.95,

    max_features=40000,

    sublinear_tf=True

)


X_train_tfidf = (
    vectorizer.fit_transform(
        X_train
    )
)


X_test_tfidf = (
    vectorizer.transform(
        X_test
    )
)


print(
    "Training feature shape:",
    X_train_tfidf.shape
)

print(
    "Testing feature shape :",
    X_test_tfidf.shape
)


# ============================================================
# EXPERIMENT WITH CLASS WEIGHTS
# ============================================================

neutral_weights = [
    1,
    2,
    3,
    4,
    5,
    7,
    10
]


results = []

best_model = None

best_weight = None

best_macro_f1 = -1


# ============================================================
# TRAIN EXPERIMENTS
# ============================================================

for weight in neutral_weights:

    print("\n" + "=" * 70)

    print(
        "NEUTRAL DETECTOR - CLASS WEIGHT:",
        weight
    )

    print("=" * 70)


    class_weights = {

        0: 1.0,

        1: float(weight)

    }


    model = LogisticRegression(

        max_iter=1000,

        class_weight=class_weights,

        random_state=42

    )


    print("\nTraining model...")


    model.fit(
        X_train_tfidf,
        y_train
    )


    predictions = model.predict(
        X_test_tfidf
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    neutral_precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )


    neutral_recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )


    neutral_f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )


    print(
        f"\nAccuracy       : {accuracy:.4f}"
    )

    print(
        f"Macro Precision: {precision:.4f}"
    )

    print(
        f"Macro Recall   : {recall:.4f}"
    )

    print(
        f"Macro F1       : {macro_f1:.4f}"
    )

    print(
        f"Neutral Precision: "
        f"{neutral_precision:.4f}"
    )

    print(
        f"Neutral Recall   : "
        f"{neutral_recall:.4f}"
    )

    print(
        f"Neutral F1       : "
        f"{neutral_f1:.4f}"
    )


    results.append({

        "weight": weight,

        "accuracy": accuracy,

        "macro_precision": precision,

        "macro_recall": recall,

        "macro_f1": macro_f1,

        "neutral_precision":
            neutral_precision,

        "neutral_recall":
            neutral_recall,

        "neutral_f1":
            neutral_f1

    })


    # Select according to Neutral F1.
    #
    # This is intentional because Neutral
    # is our current weakness.

    if neutral_f1 > (
        max(
            [r["neutral_f1"] for r in results[:-1]],
            default=-1
        )
    ):

        best_model = model

        best_weight = weight


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("NEUTRAL DETECTOR EXPERIMENT RESULTS")
print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_result = results_df.loc[
    results_df["neutral_f1"].idxmax()
]


best_neutral_f1 = (
    best_result["neutral_f1"]
)


print("\n" + "=" * 70)
print("BEST NEUTRAL DETECTOR")
print("=" * 70)


print(
    "Best Neutral Weight:",
    int(
        best_result["weight"]
    )
)


print(
    "Best Neutral F1:",
    f"{best_neutral_f1:.4f}"
)


# ============================================================
# BEST MODEL REPORT
# ============================================================

best_predictions = best_model.predict(
    X_test_tfidf
)


print("\n" + "=" * 70)
print("BEST MODEL CLASSIFICATION REPORT")
print("=" * 70)


print(
    classification_report(
        y_test,
        best_predictions,
        target_names=[
            "Non-Neutral",
            "Neutral"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


matrix = confusion_matrix(
    y_test,
    best_predictions
)


print(matrix)


# ============================================================
# SAVE EXPERIMENTAL MODEL
# ============================================================

MODEL_PATH = (
    MODEL_DIR
    / "neutral_detector_experiment.pkl"
)


VECTORIZER_PATH = (
    MODEL_DIR
    / "neutral_detector_vectorizer.pkl"
)


RESULTS_PATH = (
    MODEL_DIR
    / "neutral_detector_results.pkl"
)


joblib.dump(
    best_model,
    MODEL_PATH
)


joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)


joblib.dump(
    results,
    RESULTS_PATH
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("NEUTRAL DETECTOR EXPERIMENT COMPLETED")
print("=" * 70)


print(
    "\nExperimental model saved:"
)

print(
    MODEL_PATH
)


print(
    "\nExperimental vectorizer saved:"
)

print(
    VECTORIZER_PATH
)


print(
    "\nExperimental results saved:"
)

print(
    RESULTS_PATH
)


print(
    "\nIMPORTANT:"
)

print(
    "The production sentiment model was NOT replaced."
)
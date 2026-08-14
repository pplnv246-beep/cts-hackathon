import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# CUSTOMER FEEDBACK AI - LINEAR SVC EXPERIMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nlp_ready_reviews.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - LINEAR SVC EXPERIMENT")
print("=" * 70)

print("\nLoading NLP-ready dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

df["Processed_Review"] = (
    df["Processed_Review"]
    .fillna("")
    .astype(str)
)

df = df[
    df["Processed_Review"].str.strip() != ""
].reset_index(drop=True)

print("Total reviews:", len(df))


# ============================================================
# INPUT / TARGET
# ============================================================

X_text = df["Processed_Review"]

y = df["Sentiment"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train_text))
print("Testing samples :", len(X_test_text))


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    max_features=40000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(
    X_train_text
)

X_test = vectorizer.transform(
    X_test_text
)

print(
    "Training feature shape:",
    X_train.shape
)


# ============================================================
# LINEAR SVC EXPERIMENTS
# ============================================================

C_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0
]


results = []

best_model = None
best_macro_f1 = -1
best_c = None


# ============================================================
# RUN EXPERIMENTS
# ============================================================

for c_value in C_VALUES:

    print("\n" + "=" * 70)

    print(
        "EXPERIMENT - LinearSVC C:",
        c_value
    )

    print("=" * 70)


    print("\nTraining LinearSVC...")

    model = LinearSVC(
        C=c_value,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )


    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )


    negative_f1 = report.get(
        "Negative",
        {}
    ).get(
        "f1-score",
        0
    )


    neutral_recall = report.get(
        "Neutral",
        {}
    ).get(
        "recall",
        0
    )


    neutral_f1 = report.get(
        "Neutral",
        {}
    ).get(
        "f1-score",
        0
    )


    positive_f1 = report.get(
        "Positive",
        {}
    ).get(
        "f1-score",
        0
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
        f"Negative F1    : {negative_f1:.4f}"
    )

    print(
        f"Neutral Recall : {neutral_recall:.4f}"
    )

    print(
        f"Neutral F1     : {neutral_f1:.4f}"
    )

    print(
        f"Positive F1    : {positive_f1:.4f}"
    )


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    results.append({

        "C": c_value,

        "accuracy": accuracy,

        "macro_precision": precision,

        "macro_recall": recall,

        "macro_f1": macro_f1,

        "negative_f1": negative_f1,

        "neutral_recall": neutral_recall,

        "neutral_f1": neutral_f1,

        "positive_f1": positive_f1
    })


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        best_model = model

        best_c = c_value


# ============================================================
# RESULTS TABLE
# ============================================================

print("\n" + "=" * 70)
print("LINEAR SVC EXPERIMENT RESULTS")
print("=" * 70)

print(
    f"\n{'C':<8}"
    f"{'Accuracy':<12}"
    f"{'Macro F1':<12}"
    f"{'Neutral Recall':<16}"
    f"{'Neutral F1':<12}"
)

print("-" * 70)


for result in results:

    print(
        f"{result['C']:<8}"
        f"{result['accuracy']:<12.4f}"
        f"{result['macro_f1']:<12.4f}"
        f"{result['neutral_recall']:<16.4f}"
        f"{result['neutral_f1']:<12.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("BEST LINEAR SVC MODEL")
print("=" * 70)

print(
    "\nBest C:",
    best_c
)

print(
    "Best Macro F1:",
    f"{best_macro_f1:.4f}"
)


# ============================================================
# BEST MODEL REPORT
# ============================================================

best_predictions = best_model.predict(
    X_test
)

print("\n" + "=" * 70)
print("BEST MODEL CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)


# ============================================================
# SAVE EXPERIMENTAL MODEL
# ============================================================

MODEL_FILE = (
    MODEL_DIR
    / "linear_svc_experiment_model.pkl"
)

VECTORIZER_FILE = (
    MODEL_DIR
    / "linear_svc_experiment_vectorizer.pkl"
)

RESULTS_FILE = (
    MODEL_DIR
    / "linear_svc_experiment_results.pkl"
)


joblib.dump(
    best_model,
    MODEL_FILE
)

joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)

joblib.dump(
    results,
    RESULTS_FILE
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("LINEAR SVC EXPERIMENT COMPLETED")
print("=" * 70)

print("\nExperimental model saved:")
print(MODEL_FILE)

print("\nExperimental vectorizer saved:")
print(VECTORIZER_FILE)

print("\nExperimental results saved:")
print(RESULTS_FILE)

print("\nIMPORTANT:")
print("The existing production model was NOT replaced.")
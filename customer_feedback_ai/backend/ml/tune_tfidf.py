import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# CUSTOMER FEEDBACK AI - SAFE TF-IDF TUNING
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
print("CUSTOMER FEEDBACK AI - SAFE TF-IDF TUNING")
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
# EXPERIMENTS
# ============================================================

experiments = [

    {
        "name": "Unigram",
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 20000
    },

    {
        "name": "Unigram + Bigram",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 20000
    },

    {
        "name": "Higher min_df",
        "ngram_range": (1, 2),
        "min_df": 5,
        "max_df": 0.95,
        "max_features": 20000
    },

    {
        "name": "Larger vocabulary",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 40000
    },

    {
        "name": "Unigram + Bigram + Trigram",
        "ngram_range": (1, 3),
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 40000
    }
]


# ============================================================
# RESULTS
# ============================================================

results = []

best_macro_f1 = -1
best_model = None
best_vectorizer = None
best_config = None


# ============================================================
# RUN EXPERIMENTS
# ============================================================

for experiment in experiments:

    print("\n" + "=" * 70)

    print(
        "EXPERIMENT:",
        experiment["name"]
    )

    print("=" * 70)


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(

        ngram_range=experiment["ngram_range"],

        min_df=experiment["min_df"],

        max_df=experiment["max_df"],

        max_features=experiment["max_features"],

        sublinear_tf=True
    )


    print("\nCreating TF-IDF features...")


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


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")


    model = LogisticRegression(

        max_iter=1500,

        class_weight="balanced",

        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTION
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


    neutral_f1 = report.get(
        "Neutral",
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
    # STORE
    # --------------------------------------------------------

    results.append({

        "experiment": experiment["name"],

        "accuracy": accuracy,

        "macro_precision": precision,

        "macro_recall": recall,

        "macro_f1": macro_f1,

        "negative_f1": negative_f1,

        "neutral_recall": neutral_recall,

        "neutral_f1": neutral_f1,

        "positive_f1": positive_f1,

        "features": X_train.shape[1]
    })


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        best_model = model

        best_vectorizer = vectorizer

        best_config = experiment


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAFE TF-IDF EXPERIMENT RESULTS")
print("=" * 70)


print(
    f"\n{'Experiment':<30}"
    f"{'Accuracy':<12}"
    f"{'Macro F1':<12}"
    f"{'Neutral Recall':<16}"
    f"{'Neutral F1':<12}"
)


print("-" * 90)


for result in results:

    print(
        f"{result['experiment']:<30}"
        f"{result['accuracy']:<12.4f}"
        f"{result['macro_f1']:<12.4f}"
        f"{result['neutral_recall']:<16.4f}"
        f"{result['neutral_f1']:<12.4f}"
    )


# ============================================================
# BEST CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("BEST EXPERIMENT")
print("=" * 70)

print(
    "\nExperiment:",
    best_config["name"]
)

print(
    "N-gram range:",
    best_config["ngram_range"]
)

print(
    "min_df:",
    best_config["min_df"]
)

print(
    "max_df:",
    best_config["max_df"]
)

print(
    "max_features:",
    best_config["max_features"]
)

print(
    "Best Macro F1:",
    f"{best_macro_f1:.4f}"
)


# ============================================================
# FINAL REPORT
# ============================================================

best_predictions = best_model.predict(
    best_vectorizer.transform(
        X_test_text
    )
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
# SAFE SAVE
# ============================================================

SAFE_MODEL_FILE = (
    MODEL_DIR
    / "tfidf_experiment_model.pkl"
)

SAFE_VECTORIZER_FILE = (
    MODEL_DIR
    / "tfidf_experiment_vectorizer.pkl"
)

SAFE_RESULTS_FILE = (
    MODEL_DIR
    / "tfidf_experiment_results.pkl"
)


joblib.dump(
    best_model,
    SAFE_MODEL_FILE
)

joblib.dump(
    best_vectorizer,
    SAFE_VECTORIZER_FILE
)

joblib.dump(
    results,
    SAFE_RESULTS_FILE
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("SAFE TF-IDF TUNING COMPLETED")
print("=" * 70)

print("\nExperimental model saved:")
print(SAFE_MODEL_FILE)

print("\nExperimental vectorizer saved:")
print(SAFE_VECTORIZER_FILE)

print("\nExperimental results saved:")
print(SAFE_RESULTS_FILE)

print("\nIMPORTANT:")
print("The existing final sentiment model was NOT replaced.")
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
# PATHS
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
print("CUSTOMER FEEDBACK AI - TF-IDF TUNING")
print("=" * 70)

print("\nLoading NLP-ready dataset...")

df = pd.read_csv(INPUT_FILE)

df["Processed_Review"] = (
    df["Processed_Review"]
    .fillna("")
    .astype(str)
)

df = df[
    df["Processed_Review"].str.strip() != ""
]

df = df.reset_index(drop=True)

print("Total reviews:", len(df))


# ============================================================
# INPUT AND TARGET
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
# TF-IDF EXPERIMENTS
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
    }
]


results = []

best_f1 = -1
best_config = None
best_model = None
best_vectorizer = None


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
    # CREATE VECTORIZER
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(

        ngram_range=experiment["ngram_range"],

        min_df=experiment["min_df"],

        max_df=experiment["max_df"],

        max_features=experiment["max_features"],

        sublinear_tf=True
    )


    # --------------------------------------------------------
    # FIT ONLY ON TRAINING DATA
    # --------------------------------------------------------

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

    print(
        "Testing feature shape:",
        X_test.shape
    )


    # --------------------------------------------------------
    # TRAIN LOGISTIC REGRESSION
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(

        max_iter=1000,

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


    # --------------------------------------------------------
    # CLASS-SPECIFIC METRICS
    # --------------------------------------------------------

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


    positive_f1 = report.get(
        "Positive",
        {}
    ).get(
        "f1-score",
        0
    )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

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
        f"Neutral F1     : {neutral_f1:.4f}"
    )

    print(
        f"Positive F1    : {positive_f1:.4f}"
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "experiment": experiment["name"],

        "accuracy": accuracy,

        "macro_precision": precision,

        "macro_recall": recall,

        "macro_f1": macro_f1,

        "negative_f1": negative_f1,

        "neutral_f1": neutral_f1,

        "positive_f1": positive_f1,

        "features": X_train.shape[1]
    })


    # --------------------------------------------------------
    # SELECT BEST CONFIGURATION
    # --------------------------------------------------------

    if macro_f1 > best_f1:

        best_f1 = macro_f1

        best_config = experiment

        best_model = model

        best_vectorizer = vectorizer


# ============================================================
# RESULTS TABLE
# ============================================================

print("\n" + "=" * 70)

print("TF-IDF EXPERIMENT RESULTS")

print("=" * 70)


print(
    f"\n{'Experiment':<25}"
    f"{'Accuracy':<12}"
    f"{'Macro F1':<12}"
    f"{'Neutral F1':<12}"
    f"{'Features':<12}"
)

print("-" * 75)


for result in results:

    print(
        f"{result['experiment']:<25}"
        f"{result['accuracy']:<12.4f}"
        f"{result['macro_f1']:<12.4f}"
        f"{result['neutral_f1']:<12.4f}"
        f"{result['features']:<12}"
    )


# ============================================================
# BEST CONFIGURATION
# ============================================================

print("\n" + "=" * 70)

print("BEST TF-IDF CONFIGURATION")

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
    f"{best_f1:.4f}"
)


# ============================================================
# FINAL CLASSIFICATION REPORT
# ============================================================

X_test_best = best_vectorizer.transform(
    X_test_text
)

best_predictions = best_model.predict(
    X_test_best
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
# SAVE BEST VECTORIZER
# ============================================================

BEST_VECTORIZER_FILE = (
    MODEL_DIR
    / "final_tfidf_vectorizer.pkl"
)

joblib.dump(
    best_vectorizer,
    BEST_VECTORIZER_FILE
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

BEST_MODEL_FILE = (
    MODEL_DIR
    / "final_sentiment_model.pkl"
)

joblib.dump(
    best_model,
    BEST_MODEL_FILE
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_FILE = (
    MODEL_DIR
    / "tfidf_tuning_results.pkl"
)

joblib.dump(
    results,
    RESULTS_FILE
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)

print("TF-IDF TUNING COMPLETED")

print("=" * 70)

print("\nFinal model saved:")
print(BEST_MODEL_FILE)

print("\nFinal vectorizer saved:")
print(BEST_VECTORIZER_FILE)

print("\nExperiment results saved:")
print(RESULTS_FILE)
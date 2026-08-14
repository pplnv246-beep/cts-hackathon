import joblib
import pandas as pd
import re

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
# HYBRID ML + TEXT EVIDENCE EXPERIMENT
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - HYBRID DECISION EXPERIMENT")
print("=" * 70)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nlp_ready_reviews.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

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
# CLEAN TEXT
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
# TRAIN / TEST SPLIT
# ============================================================

X = df[text_column]

y = df["Sentiment"]


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


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


# ============================================================
# TRAIN BASE MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING BASE MODEL")
print("=" * 70)


base_model = LogisticRegression(

    max_iter=1000,

    class_weight={
        "Negative": 1.0,
        "Neutral": 5.0,
        "Positive": 1.0
    },

    random_state=42

)


base_model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# BASE MODEL PREDICTION
# ============================================================

base_predictions = (
    base_model.predict(
        X_test_tfidf
    )
)


base_probabilities = (
    base_model.predict_proba(
        X_test_tfidf
    )
)


classes = (
    base_model.classes_
)


# ============================================================
# TEXT EVIDENCE
# ============================================================

negative_phrases = [

    "bad",
    "terrible",
    "worst",
    "awful",
    "horrible",
    "poor",
    "disappointed",
    "disappointing",
    "broken",
    "damaged",
    "late",
    "delayed",
    "delay",
    "slow",
    "expensive",
    "useless",
    "waste",
    "problem",
    "problems",
    "issue",
    "issues",
    "complaint",
    "complaints",
    "frustrating",
    "frustrated",
    "angry",
    "unhappy",
    "unusable",
    "missing",
    "wrong",
    "failed",
    "failure",
    "not good",
    "not happy",
    "not satisfied",
    "poor quality",
    "does not work",
    "did not work",
    "delivery was late",
    "arrived broken"
]


positive_phrases = [

    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "love",
    "loved",
    "like",
    "liked",
    "happy",
    "satisfied",
    "perfect",
    "wonderful",
    "fantastic",
    "best",
    "fast",
    "quick",
    "reliable",
    "worth it",
    "good quality",
    "works well",
    "very good"
]


def normalize(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def find_evidence(
    text,
    phrases
):

    found = []

    for phrase in phrases:

        if phrase in text:

            found.append(
                phrase
            )

    return found


# ============================================================
# HYBRID DECISION FUNCTION
# ============================================================

def hybrid_prediction(
    ml_prediction,
    probabilities,
    text
):

    text = normalize(text)


    negative_evidence = find_evidence(
        text,
        negative_phrases
    )


    positive_evidence = find_evidence(
        text,
        positive_phrases
    )


    negative_count = len(
        negative_evidence
    )

    positive_count = len(
        positive_evidence
    )


    # --------------------------------------------------------
    # GET ML PROBABILITIES
    # --------------------------------------------------------

    probability_dict = {
        str(label): float(probability)
        for label, probability
        in zip(
            classes,
            probabilities
        )
    }


    negative_probability = (
        probability_dict
        .get(
            "Negative",
            0
        )
    )


    neutral_probability = (
        probability_dict
        .get(
            "Neutral",
            0
        )
    )


    positive_probability = (
        probability_dict
        .get(
            "Positive",
            0
        )
    )


    # --------------------------------------------------------
    # STRONG NEGATIVE EVIDENCE
    # --------------------------------------------------------

    if (
        negative_count >= 2
        and negative_probability >= 0.25
        and negative_probability >= positive_probability
    ):

        return "Negative"


    # --------------------------------------------------------
    # STRONG POSITIVE EVIDENCE
    # --------------------------------------------------------

    if (
        positive_count >= 2
        and positive_probability >= 0.25
        and positive_probability >= negative_probability
    ):

        return "Positive"


    # --------------------------------------------------------
    # SINGLE STRONG NEGATIVE PHRASE
    # --------------------------------------------------------

    if (
        negative_count >= 1
        and negative_probability >= 0.45
        and neutral_probability >= negative_probability
    ):

        return "Negative"


    # --------------------------------------------------------
    # SINGLE STRONG POSITIVE PHRASE
    # --------------------------------------------------------

    if (
        positive_count >= 1
        and positive_probability >= 0.70
    ):

        return "Positive"


    # --------------------------------------------------------
    # OTHERWISE KEEP ML
    # --------------------------------------------------------

    return ml_prediction


# ============================================================
# CREATE HYBRID PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING HYBRID PREDICTIONS")
print("=" * 70)


hybrid_predictions = []


for text, prediction, probabilities in zip(
    X_test,
    base_predictions,
    base_probabilities
):

    final_prediction = hybrid_prediction(

        prediction,

        probabilities,

        text

    )


    hybrid_predictions.append(
        final_prediction
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    actual,
    predictions
):

    print("\n" + "=" * 70)

    print(
        name
    )

    print("=" * 70)


    accuracy = accuracy_score(
        actual,
        predictions
    )


    macro_precision = precision_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )


    macro_recall = recall_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )


    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )


    neutral_f1 = f1_score(
        actual,
        predictions,
        labels=["Neutral"],
        average="macro",
        zero_division=0
    )


    neutral_recall = recall_score(
        actual,
        predictions,
        labels=["Neutral"],
        average="macro",
        zero_division=0
    )


    print(
        f"\nAccuracy       : {accuracy:.4f}"
    )

    print(
        f"Macro Precision: {macro_precision:.4f}"
    )

    print(
        f"Macro Recall   : {macro_recall:.4f}"
    )

    print(
        f"Macro F1       : {macro_f1:.4f}"
    )

    print(
        f"Neutral Recall : {neutral_recall:.4f}"
    )

    print(
        f"Neutral F1     : {neutral_f1:.4f}"
    )


    return {

        "accuracy": accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "neutral_recall":
            neutral_recall,

        "neutral_f1":
            neutral_f1

    }


# ============================================================
# EVALUATE BASE MODEL
# ============================================================

base_results = evaluate_model(

    "BASE MODEL",

    y_test,

    base_predictions

)


# ============================================================
# EVALUATE HYBRID MODEL
# ============================================================

hybrid_results = evaluate_model(

    "HYBRID MODEL",

    y_test,

    hybrid_predictions

)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


comparison = pd.DataFrame({

    "Metric": [

        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Neutral Recall",
        "Neutral F1"

    ],

    "Base Model": [

        base_results["accuracy"],
        base_results["macro_precision"],
        base_results["macro_recall"],
        base_results["macro_f1"],
        base_results["neutral_recall"],
        base_results["neutral_f1"]

    ],

    "Hybrid Model": [

        hybrid_results["accuracy"],
        hybrid_results["macro_precision"],
        hybrid_results["macro_recall"],
        hybrid_results["macro_f1"],
        hybrid_results["neutral_recall"],
        hybrid_results["neutral_f1"]

    ]

})


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

print("\n" + "=" * 70)
print("BASE MODEL CLASSIFICATION REPORT")
print("=" * 70)


print(
    classification_report(
        y_test,
        base_predictions,
        zero_division=0
    )
)


print("\n" + "=" * 70)
print("HYBRID MODEL CLASSIFICATION REPORT")
print("=" * 70)


print(
    classification_report(
        y_test,
        hybrid_predictions,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 70)
print("BASE MODEL CONFUSION MATRIX")
print("=" * 70)


print(
    confusion_matrix(
        y_test,
        base_predictions,
        labels=[
            "Negative",
            "Neutral",
            "Positive"
        ]
    )
)


print("\n" + "=" * 70)
print("HYBRID MODEL CONFUSION MATRIX")
print("=" * 70)


print(
    confusion_matrix(
        y_test,
        hybrid_predictions,
        labels=[
            "Negative",
            "Neutral",
            "Positive"
        ]
    )
)


# ============================================================
# SAVE EXPERIMENT RESULTS
# ============================================================

OUTPUT_DIR = (
    BASE_DIR
    / "models"
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "hybrid_decision_results.pkl"
)


results = {

    "base_results":
        base_results,

    "hybrid_results":
        hybrid_results,

    "comparison":
        comparison,

    "base_model":
        base_model,

    "vectorizer":
        vectorizer

}


joblib.dump(
    results,
    OUTPUT_FILE
)


# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("HYBRID EXPERIMENT CONCLUSION")
print("=" * 70)


if (
    hybrid_results["macro_f1"]
    >
    base_results["macro_f1"]
    and
    hybrid_results["neutral_f1"]
    >
    base_results["neutral_f1"]
):

    print(
        "HYBRID MODEL IMPROVED BOTH "
        "MACRO F1 AND NEUTRAL F1."
    )

    print(
        "Candidate for further testing."
    )

else:

    print(
        "HYBRID MODEL DID NOT IMPROVE "
        "BOTH MACRO F1 AND NEUTRAL F1."
    )

    print(
        "Do NOT replace the production model."
    )


print("\n" + "=" * 70)
print("HYBRID DECISION EXPERIMENT COMPLETED")
print("=" * 70)

print(
    "\nResults saved:"
)

print(
    OUTPUT_FILE
)
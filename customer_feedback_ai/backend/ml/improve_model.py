import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
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

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "tfidf_features.pkl"
)

TARGET_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "target.pkl"
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
print("CUSTOMER FEEDBACK AI - MODEL IMPROVEMENT")
print("=" * 70)

print("\nLoading TF-IDF features...")

X = joblib.load(FEATURE_FILE)

y = joblib.load(TARGET_FILE)

print("Features shape:", X.shape)

print("Target size:", len(y))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print("\nTraining samples:", len(y_train))

print("Testing samples :", len(y_test))


# ============================================================
# EXPERIMENTS
# ============================================================

neutral_weights = [1, 2, 3, 4, 5]

results = []

best_model = None
best_weight = None
best_macro_f1 = -1


# ============================================================
# RUN EXPERIMENTS
# ============================================================

for neutral_weight in neutral_weights:

    print("\n" + "=" * 70)

    print(
        f"EXPERIMENT - NEUTRAL CLASS WEIGHT: "
        f"{neutral_weight}"
    )

    print("=" * 70)


    class_weights = {
        "Negative": 1.0,
        "Neutral": float(neutral_weight),
        "Positive": 1.0
    }


    model = LogisticRegression(

        max_iter=1000,

        class_weight=class_weights,

        random_state=42
    )


    print("\nTraining model...")

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
    # NEUTRAL F1
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    neutral_f1 = report.get(
        "Neutral",
        {}
    ).get(
        "f1-score",
        0
    )


    # --------------------------------------------------------
    # DISPLAY
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
        f"Neutral F1     : {neutral_f1:.4f}"
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "neutral_weight": neutral_weight,

        "accuracy": accuracy,

        "macro_precision": precision,

        "macro_recall": recall,

        "macro_f1": macro_f1,

        "neutral_f1": neutral_f1
    })


    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        best_model = model

        best_weight = neutral_weight


# ============================================================
# RESULTS SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("EXPERIMENT RESULTS")

print("=" * 70)


print(
    f"\n{'Weight':<10}"
    f"{'Accuracy':<15}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'Macro F1':<15}"
    f"{'Neutral F1':<15}"
)

print("-" * 85)


for result in results:

    print(
        f"{result['neutral_weight']:<10}"
        f"{result['accuracy']:<15.4f}"
        f"{result['macro_precision']:<15.4f}"
        f"{result['macro_recall']:<15.4f}"
        f"{result['macro_f1']:<15.4f}"
        f"{result['neutral_f1']:<15.4f}"
    )


# ============================================================
# BEST MODEL
# ============================================================

print("\n" + "=" * 70)

print("BEST MODEL")

print("=" * 70)

print(
    f"\nBest Neutral Weight: {best_weight}"
)

print(
    f"Best Macro F1: {best_macro_f1:.4f}"
)


# ============================================================
# FINAL CLASSIFICATION REPORT
# ============================================================

final_predictions = best_model.predict(
    X_test
)

print("\n" + "=" * 70)

print("BEST MODEL CLASSIFICATION REPORT")

print("=" * 70)

print(
    classification_report(
        y_test,
        final_predictions,
        zero_division=0
    )
)


# ============================================================
# SAVE IMPROVED MODEL
# ============================================================

IMPROVED_MODEL_FILE = (
    MODEL_DIR
    / "improved_sentiment_model.pkl"
)

joblib.dump(
    best_model,
    IMPROVED_MODEL_FILE
)

print("\nImproved model saved to:")

print(IMPROVED_MODEL_FILE)


# ============================================================
# SAVE EXPERIMENT RESULTS
# ============================================================

RESULTS_FILE = (
    MODEL_DIR
    / "model_improvement_results.pkl"
)

joblib.dump(
    results,
    RESULTS_FILE
)

print("\nExperiment results saved to:")

print(RESULTS_FILE)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)

print("MODEL IMPROVEMENT COMPLETED")

print("=" * 70)
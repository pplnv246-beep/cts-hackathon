import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = BASE_DIR / "data" / "processed" / "tfidf_features.pkl"
TARGET_FILE = BASE_DIR / "data" / "processed" / "target.pkl"
MODEL_FILE = BASE_DIR / "models" / "sentiment_model.pkl"

REPORT_DIR = BASE_DIR / "reports" / "model"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - MODEL EVALUATION")
print("=" * 70)

print("\nLoading TF-IDF features...")

X = joblib.load(FEATURE_FILE)
y = joblib.load(TARGET_FILE)

print("Features shape:", X.shape)
print("Target size:", len(y))


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# SAME TRAIN/TEST SPLIT
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
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

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

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# DISPLAY PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nAccuracy        : {accuracy:.4f}")
print(f"Accuracy        : {accuracy * 100:.2f}%")
print(f"Macro Precision : {precision:.4f}")
print(f"Macro Recall    : {recall:.4f}")
print(f"Macro F1-score  : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot()

plt.title("Customer Sentiment Confusion Matrix")

plt.tight_layout()

confusion_file = REPORT_DIR / "confusion_matrix.png"

plt.savefig(
    confusion_file,
    dpi=300
)

plt.close()

print("\nConfusion matrix saved:")
print(confusion_file)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "accuracy": accuracy,
    "macro_precision": precision,
    "macro_recall": recall,
    "macro_f1": f1
}

metrics_file = REPORT_DIR / "metrics.pkl"

joblib.dump(
    metrics,
    metrics_file
)

print("\nMetrics saved:")
print(metrics_file)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)
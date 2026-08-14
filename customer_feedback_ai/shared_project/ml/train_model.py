import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
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

MODEL_FILE = (
    MODEL_DIR
    / "sentiment_model.pkl"
)


# ============================================================
# LOAD TF-IDF FEATURES
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - MODEL TRAINING")
print("=" * 70)

print("\nLoading TF-IDF features...")

X = joblib.load(FEATURE_FILE)

y = joblib.load(TARGET_FILE)

print("Features shape:", X.shape)

print("Target size:", len(y))


# ============================================================
# CHECK TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(y.value_counts())

print("\nTarget percentage:")

print(
    y.value_counts(
        normalize=True
    ).mul(100).round(2)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("Training samples:", X_train.shape[0])

print("Testing samples :", X_test.shape[0])


# ============================================================
# CREATE MODEL
# ============================================================

print("\n" + "=" * 70)
print("CREATING LOGISTIC REGRESSION MODEL")
print("=" * 70)

model = LogisticRegression(

    max_iter=1000,

    class_weight="balanced",

    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy percentage: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# MODEL CLASSES
# ============================================================

print("\nModel classes:")

print(
    model.classes_
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\nModel saved successfully:")

print(MODEL_FILE)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)

print("\nNext step:")
print("Model evaluation and prediction testing.")
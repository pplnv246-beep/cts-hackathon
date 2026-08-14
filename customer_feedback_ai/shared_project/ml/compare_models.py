import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - MODEL COMPARISON")
print("=" * 70)

print("\nLoading TF-IDF features...")

X = joblib.load(FEATURE_FILE)
y = joblib.load(TARGET_FILE)

print("Features:", X.shape)
print("Target:", len(y))


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
# LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 - LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_precision = precision_score(
    y_test,
    logistic_pred,
    average="macro",
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_pred,
    average="macro",
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred,
    average="macro",
    zero_division=0
)

print(
    f"Accuracy       : {logistic_accuracy:.4f}"
)

print(
    f"Macro Precision: {logistic_precision:.4f}"
)

print(
    f"Macro Recall   : {logistic_recall:.4f}"
)

print(
    f"Macro F1       : {logistic_f1:.4f}"
)


# ============================================================
# LINEAR SVM
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 - LINEAR SVM")
print("=" * 70)

svm_model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

svm_model.fit(
    X_train,
    y_train
)

svm_pred = svm_model.predict(
    X_test
)


svm_accuracy = accuracy_score(
    y_test,
    svm_pred
)

svm_precision = precision_score(
    y_test,
    svm_pred,
    average="macro",
    zero_division=0
)

svm_recall = recall_score(
    y_test,
    svm_pred,
    average="macro",
    zero_division=0
)

svm_f1 = f1_score(
    y_test,
    svm_pred,
    average="macro",
    zero_division=0
)

print(
    f"Accuracy       : {svm_accuracy:.4f}"
)

print(
    f"Macro Precision: {svm_precision:.4f}"
)

print(
    f"Macro Recall   : {svm_recall:.4f}"
)

print(
    f"Macro F1       : {svm_f1:.4f}"
)


# ============================================================
# DETAILED SVM REPORT
# ============================================================

print("\n" + "=" * 70)
print("LINEAR SVM CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        svm_pred,
        zero_division=0
    )
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"\n{'Metric':<20}"
    f"{'Logistic Regression':<25}"
    f"{'Linear SVM':<20}"
)

print("-" * 65)

print(
    f"{'Accuracy':<20}"
    f"{logistic_accuracy:<25.4f}"
    f"{svm_accuracy:<20.4f}"
)

print(
    f"{'Macro Precision':<20}"
    f"{logistic_precision:<25.4f}"
    f"{svm_precision:<20.4f}"
)

print(
    f"{'Macro Recall':<20}"
    f"{logistic_recall:<25.4f}"
    f"{svm_recall:<20.4f}"
)

print(
    f"{'Macro F1':<20}"
    f"{logistic_f1:<25.4f}"
    f"{svm_f1:<20.4f}"
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

if svm_f1 > logistic_f1:

    best_model = svm_model
    best_name = "Linear SVM"
    best_f1 = svm_f1

else:

    best_model = logistic_model
    best_name = "Logistic Regression"
    best_f1 = logistic_f1


print("\n" + "=" * 70)

print(
    f"BEST MODEL: {best_name}"
)

print(
    f"Best Macro F1: {best_f1:.4f}"
)

print("=" * 70)
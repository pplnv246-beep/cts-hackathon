import joblib
from pathlib import Path


# ============================================================
# FINAL MODEL CANDIDATE TEST
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - FINAL MODEL CANDIDATE TEST")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "tfidf_experiment_model.pkl"
)

VECTORIZER_PATH = (
    BASE_DIR
    / "models"
    / "tfidf_experiment_vectorizer.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading candidate model...")

model = joblib.load(
    MODEL_PATH
)

vectorizer = joblib.load(
    VECTORIZER_PATH
)

print("Candidate model loaded successfully.")


# ============================================================
# TEST REVIEWS
# ============================================================

test_reviews = [

    "The product arrived broken and the delivery was late.",

    "I absolutely love this product. The quality is excellent.",

    "The package contains the item I ordered. It arrived yesterday.",

    "The product is good but the delivery was late.",

    "The product arrived on time and it is exactly as described.",

    "The service was okay and nothing special.",

    "I am very disappointed with this product.",

    "Excellent product and very fast delivery."

]


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE MODEL PREDICTIONS")
print("=" * 70)


for index, review in enumerate(
    test_reviews,
    start=1
):

    features = vectorizer.transform(
        [review]
    )

    prediction = model.predict(
        features
    )[0]

    probabilities = (
        model.predict_proba(
            features
        )[0]
    )

    classes = model.classes_

    print("\n" + "-" * 70)

    print(
        f"REVIEW {index}:"
    )

    print(
        review
    )

    print(
        "\nPredicted Sentiment:",
        prediction
    )

    print("\nProbabilities:")

    for class_name, probability in zip(
        classes,
        probabilities
    ):

        print(
            f"{class_name:<10}: "
            f"{probability * 100:.2f}%"
        )

    print(
        "\nConfidence:",
        f"{max(probabilities) * 100:.2f}%"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FINAL CANDIDATE TEST COMPLETED")
print("=" * 70)
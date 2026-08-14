import joblib
from pathlib import Path


# ============================================================
# TEST EXPERIMENTAL SENTIMENT MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "tfidf_experiment_model.pkl"
)

VECTORIZER_FILE = (
    BASE_DIR
    / "models"
    / "tfidf_experiment_vectorizer.pkl"
)


print("=" * 70)
print("EXPERIMENTAL MODEL TEST")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading experimental model...")

model = joblib.load(
    MODEL_FILE
)

vectorizer = joblib.load(
    VECTORIZER_FILE
)

print("Model loaded successfully.")


# ============================================================
# TEST REVIEWS
# ============================================================

reviews = [

    "The product arrived on time and it is the same item shown in the description.",

    "I received the product yesterday. It is as described and I have no particular complaints.",

    "The package contains the item I ordered. It arrived yesterday.",

    "The product arrived broken and delivery was extremely late. Customer service never responded.",

    "I absolutely love this product. The quality is excellent and delivery was very fast."

]


# ============================================================
# PREDICTIONS
# ============================================================

for index, review in enumerate(
    reviews,
    start=1
):

    print("\n" + "-" * 70)

    print(
        f"REVIEW {index}:"
    )

    print(review)


    X = vectorizer.transform(
        [review]
    )


    prediction = model.predict(
        X
    )[0]


    print(
        "\nPredicted Sentiment:",
        prediction
    )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            X
        )[0]


        print("\nProbabilities:")

        for label, probability in zip(
            model.classes_,
            probabilities
        ):

            print(
                f"{label:<10}: "
                f"{probability * 100:.2f}%"
            )


        confidence = max(
            probabilities
        )


        print(
            "\nConfidence:",
            f"{confidence * 100:.2f}%"
        )


print("\n" + "=" * 70)

print(
    "EXPERIMENTAL MODEL TEST COMPLETED"
)

print("=" * 70)
import joblib
from pathlib import Path


# ============================================================
# CUSTOMER FEEDBACK AI
# PRODUCTION vs CANDIDATE MODEL COMPARISON
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - FINAL MODEL COMPARISON")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PRODUCTION_MODEL = (
    BASE_DIR
    / "models"
    / "final_sentiment_model.pkl"
)

PRODUCTION_VECTORIZER = (
    BASE_DIR
    / "models"
    / "final_tfidf_vectorizer.pkl"
)

CANDIDATE_MODEL = (
    BASE_DIR
    / "models"
    / "tfidf_experiment_model.pkl"
)

CANDIDATE_VECTORIZER = (
    BASE_DIR
    / "models"
    / "tfidf_experiment_vectorizer.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading production model...")

production_model = joblib.load(
    PRODUCTION_MODEL
)

production_vectorizer = joblib.load(
    PRODUCTION_VECTORIZER
)

print("Production model loaded.")


print("\nLoading candidate model...")

candidate_model = joblib.load(
    CANDIDATE_MODEL
)

candidate_vectorizer = joblib.load(
    CANDIDATE_VECTORIZER
)

print("Candidate model loaded.")


# ============================================================
# TEST REVIEWS
# ============================================================

test_reviews = [

    (
        "The product arrived broken and the delivery was late.",
        "Negative"
    ),

    (
        "I absolutely love this product. The quality is excellent.",
        "Positive"
    ),

    (
        "The package contains the item I ordered. It arrived yesterday.",
        "Neutral"
    ),

    (
        "The product is good but the delivery was late.",
        "Negative"
    ),

    (
        "The product arrived on time and it is exactly as described.",
        "Positive"
    ),

    (
        "The service was okay and nothing special.",
        "Neutral"
    ),

    (
        "I am very disappointed with this product.",
        "Negative"
    ),

    (
        "Excellent product and very fast delivery.",
        "Positive"
    )

]


# ============================================================
# COMPARE
# ============================================================

print("\n" + "=" * 70)
print("MODEL PREDICTION COMPARISON")
print("=" * 70)


production_correct = 0

candidate_correct = 0


for index, (
    review,
    expected
) in enumerate(
    test_reviews,
    start=1
):

    production_features = (
        production_vectorizer.transform(
            [review]
        )
    )

    candidate_features = (
        candidate_vectorizer.transform(
            [review]
        )
    )


    production_prediction = (
        production_model
        .predict(
            production_features
        )[0]
    )


    candidate_prediction = (
        candidate_model
        .predict(
            candidate_features
        )[0]
    )


    production_probabilities = (
        production_model
        .predict_proba(
            production_features
        )[0]
    )


    candidate_probabilities = (
        candidate_model
        .predict_proba(
            candidate_features
        )[0]
    )


    production_confidence = (
        max(
            production_probabilities
        )
    )


    candidate_confidence = (
        max(
            candidate_probabilities
        )
    )


    if (
        production_prediction
        == expected
    ):

        production_correct += 1


    if (
        candidate_prediction
        == expected
    ):

        candidate_correct += 1


    print("\n" + "-" * 70)

    print(
        f"REVIEW {index}"
    )

    print(
        f"\nText: {review}"
    )

    print(
        f"Expected: {expected}"
    )

    print(
        "\nProduction Model:"
    )

    print(
        f"Prediction: {production_prediction}"
    )

    print(
        f"Confidence: "
        f"{production_confidence * 100:.2f}%"
    )


    print(
        "\nCandidate Model:"
    )

    print(
        f"Prediction: {candidate_prediction}"
    )

    print(
        f"Confidence: "
        f"{candidate_confidence * 100:.2f}%"
    )


# ============================================================
# FINAL SCORE
# ============================================================

print("\n" + "=" * 70)
print("SANITY TEST RESULTS")
print("=" * 70)


total_tests = len(
    test_reviews
)


production_accuracy = (
    production_correct
    / total_tests
    * 100
)


candidate_accuracy = (
    candidate_correct
    / total_tests
    * 100
)


print(
    f"\nProduction correct: "
    f"{production_correct}/{total_tests}"
)


print(
    f"Production sanity accuracy: "
    f"{production_accuracy:.2f}%"
)


print(
    f"\nCandidate correct: "
    f"{candidate_correct}/{total_tests}"
)


print(
    f"Candidate sanity accuracy: "
    f"{candidate_accuracy:.2f}%"
)


# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("FINAL CANDIDATE DECISION")
print("=" * 70)


if (
    candidate_correct
    >
    production_correct
):

    print(
        "Candidate performs better "
        "on the sanity tests."
    )

    print(
        "Further evaluation is recommended."
    )

elif (
    candidate_correct
    <
    production_correct
):

    print(
        "Production model performs better "
        "on the sanity tests."
    )

    print(
        "Keep the production model."
    )

else:

    print(
        "Both models have the same "
        "sanity-test performance."
    )

    print(
        "Use overall evaluation metrics "
        "and class-level performance "
        "for the final decision."
    )


print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON COMPLETED")
print("=" * 70)
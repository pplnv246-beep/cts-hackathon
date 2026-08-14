import pandas as pd
from pathlib import Path


# ============================================================
# CUSTOMER FEEDBACK AI
# RATING vs SENTIMENT ANALYSIS
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - RATING vs SENTIMENT ANALYSIS")
print("=" * 70)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nlp_ready_reviews.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total reviews:", len(df))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Rating",
    "Sentiment"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# CLEAN RATING
# ============================================================

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Rating",
        "Sentiment"
    ]
)

df["Rating"] = df["Rating"].astype(int)


# ============================================================
# RATING DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RATING DISTRIBUTION")
print("=" * 70)

rating_counts = (
    df["Rating"]
    .value_counts()
    .sort_index()
)

print(rating_counts)


# ============================================================
# SENTIMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT DISTRIBUTION")
print("=" * 70)

sentiment_counts = (
    df["Sentiment"]
    .value_counts()
)

print(sentiment_counts)


# ============================================================
# RATING vs SENTIMENT
# ============================================================

print("\n" + "=" * 70)
print("RATING vs SENTIMENT")
print("=" * 70)

cross_table = pd.crosstab(
    df["Rating"],
    df["Sentiment"]
)

print(
    cross_table.to_string()
)


# ============================================================
# RATING vs SENTIMENT PERCENTAGES
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT PERCENTAGE WITHIN EACH RATING")
print("=" * 70)

percentage_table = pd.crosstab(
    df["Rating"],
    df["Sentiment"],
    normalize="index"
) * 100

print(
    percentage_table.round(2).to_string()
)


# ============================================================
# EXPECTED SENTIMENT BASED ON RATING
# ============================================================

def expected_sentiment(rating):

    if rating <= 2:
        return "Negative"

    elif rating == 3:
        return "Neutral"

    elif rating >= 4:
        return "Positive"

    return "Unknown"


df["Expected_Sentiment"] = (
    df["Rating"]
    .apply(expected_sentiment)
)


# ============================================================
# RATING / LABEL AGREEMENT
# ============================================================

df["Rating_Label_Agree"] = (
    df["Sentiment"]
    == df["Expected_Sentiment"]
)


agreement_count = (
    df["Rating_Label_Agree"]
    .sum()
)

disagreement_count = (
    (~df["Rating_Label_Agree"])
    .sum()
)

agreement_percentage = (
    agreement_count
    / len(df)
    * 100
)

disagreement_percentage = (
    disagreement_count
    / len(df)
    * 100
)


print("\n" + "=" * 70)
print("RATING / SENTIMENT AGREEMENT")
print("=" * 70)

print(
    "Agreement:",
    agreement_count
)

print(
    "Disagreement:",
    disagreement_count
)

print(
    "Agreement percentage:",
    round(
        agreement_percentage,
        2
    ),
    "%"
)

print(
    "Disagreement percentage:",
    round(
        disagreement_percentage,
        2
    ),
    "%"
)


# ============================================================
# 3-STAR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("3-STAR REVIEW ANALYSIS")
print("=" * 70)

three_star_df = df[
    df["Rating"] == 3
].copy()

print(
    "Total 3-star reviews:",
    len(three_star_df)
)


three_star_sentiment = (
    three_star_df["Sentiment"]
    .value_counts()
)

print(
    "\nSentiment distribution among 3-star reviews:"
)

print(
    three_star_sentiment
)


# ============================================================
# 3-STAR SENTIMENT PERCENTAGES
# ============================================================

three_star_percentage = (
    three_star_df["Sentiment"]
    .value_counts(
        normalize=True
    )
    * 100
)

print(
    "\n3-star sentiment percentages:"
)

print(
    three_star_percentage.round(2)
)


# ============================================================
# 3-STAR NON-NEUTRAL REVIEWS
# ============================================================

three_star_non_neutral = three_star_df[
    three_star_df["Sentiment"] != "Neutral"
].copy()


print("\n" + "=" * 70)
print("3-STAR REVIEWS WITH NON-NEUTRAL LABELS")
print("=" * 70)

print(
    "Count:",
    len(three_star_non_neutral)
)


if len(three_star_non_neutral) > 0:

    print("\nExamples:")

    review_column = None

    possible_review_columns = [
        "Review",
        "Cleaned_Review",
        "Processed_Review"
    ]

    for column in possible_review_columns:

        if column in three_star_non_neutral.columns:

            review_column = column
            break


    if review_column:

        for index, row in (
            three_star_non_neutral
            .head(20)
            .iterrows()
        ):

            print(
                f"\nRating: {row['Rating']}"
            )

            print(
                f"Sentiment: {row['Sentiment']}"
            )

            print(
                f"Review: {row[review_column]}"
            )

    else:

        print(
            "No review text column found."
        )


# ============================================================
# EXPECTED LABEL RULE
# ============================================================

print("\n" + "=" * 70)
print("EXPECTED RATING-BASED SENTIMENT RULE")
print("=" * 70)

print(
    "1–2 stars → Negative"
)

print(
    "3 stars   → Neutral"
)

print(
    "4–5 stars → Positive"
)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("INITIAL CONCLUSION")
print("=" * 70)


three_star_neutral_count = (
    len(
        three_star_df[
            three_star_df["Sentiment"]
            == "Neutral"
        ]
    )
)


three_star_neutral_percentage = (
    three_star_neutral_count
    / len(three_star_df)
    * 100
    if len(three_star_df) > 0
    else 0
)


print(
    "3-star Neutral reviews:",
    three_star_neutral_count
)

print(
    "3-star Neutral percentage:",
    round(
        three_star_neutral_percentage,
        2
    ),
    "%"
)


if (
    three_star_neutral_percentage
    >= 95
):

    print(
        "\nObservation:"
    )

    print(
        "The dataset strongly associates "
        "3-star ratings with Neutral sentiment."
    )

elif (
    three_star_neutral_percentage
    >= 80
):

    print(
        "\nObservation:"
    )

    print(
        "Most 3-star ratings are Neutral, "
        "but there are notable exceptions."
    )

else:

    print(
        "\nObservation:"
    )

    print(
        "3-star ratings are not consistently "
        "labeled as Neutral."
    )


print("\n" + "=" * 70)
print(
    "RATING vs SENTIMENT ANALYSIS COMPLETED"
)
print("=" * 70)
import pandas as pd
import re
from pathlib import Path
from collections import Counter


# ============================================================
# CUSTOMER FEEDBACK AI
# DEEP NEUTRAL REVIEW ANALYSIS
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - DEEP NEUTRAL REVIEW ANALYSIS")
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
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Rating",
    "Sentiment"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# FIND REVIEW COLUMN
# ============================================================

review_column = None

possible_columns = [
    "Review",
    "Cleaned_Review",
    "Processed_Review"
]

for column in possible_columns:

    if column in df.columns:

        review_column = column
        break


if review_column is None:

    raise ValueError(
        "No review text column found."
    )


print(
    "Review column:",
    review_column
)


# ============================================================
# FILTER 3-STAR / NEUTRAL REVIEWS
# ============================================================

neutral_df = df[
    (df["Rating"] == 3) &
    (df["Sentiment"] == "Neutral")
].copy()


print("\nNeutral reviews:", len(neutral_df))


# ============================================================
# SENTIMENT WORD LISTS
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
    "disappointment",
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
    "never",
    "not good",
    "not happy",
    "not satisfied",
    "poor quality",
    "does not work",
    "did not work"
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
    "quality",
    "worth it",
    "good quality",
    "works well",
    "very good"
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    if pd.isna(text):

        return ""

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


# ============================================================
# PHRASE DETECTION
# ============================================================

def find_phrases(
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
# CLASSIFICATION
# ============================================================

analysis_results = []


for index, row in neutral_df.iterrows():

    review = normalize(
        row[review_column]
    )


    negative_found = find_phrases(
        review,
        negative_phrases
    )


    positive_found = find_phrases(
        review,
        positive_phrases
    )


    negative_count = len(
        negative_found
    )

    positive_count = len(
        positive_found
    )


    # --------------------------------------------------------
    # CLASSIFY TEXT
    # --------------------------------------------------------

    if (
        negative_count > 0
        and positive_count > 0
    ):

        category = "Mixed Sentiment"


    elif negative_count >= 2:

        category = "Strong Negative Language"


    elif negative_count == 1:

        category = "Negative Language"


    elif positive_count >= 2:

        category = "Strong Positive Language"


    elif positive_count == 1:

        category = "Positive Language"


    else:

        category = "Clearly Neutral"


    analysis_results.append({

        "index": index,

        "review": row[review_column],

        "negative_evidence":
            negative_found,

        "positive_evidence":
            positive_found,

        "negative_count":
            negative_count,

        "positive_count":
            positive_count,

        "category":
            category

    })


# ============================================================
# RESULT DATAFRAME
# ============================================================

analysis_df = pd.DataFrame(
    analysis_results
)


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("NEUTRAL TEXT CATEGORY DISTRIBUTION")
print("=" * 70)

category_counts = (
    analysis_df["category"]
    .value_counts()
)

print(
    category_counts
)


print("\nPercentages:")

category_percentages = (
    analysis_df["category"]
    .value_counts(
        normalize=True
    )
    * 100
)

print(
    category_percentages.round(2)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)


for category in category_counts.index:

    count = category_counts[
        category
    ]

    percentage = (
        count
        / len(analysis_df)
        * 100
    )

    print(
        f"{category:<30}"
        f"{count:>6}"
        f"  ({percentage:>6.2f}%)"
    )


# ============================================================
# EVIDENCE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("EVIDENCE STATISTICS")
print("=" * 70)


reviews_with_negative = (
    analysis_df[
        analysis_df["negative_count"] > 0
    ]
)


reviews_with_positive = (
    analysis_df[
        analysis_df["positive_count"] > 0
    ]
)


reviews_with_both = (
    analysis_df[
        (
            analysis_df["negative_count"] > 0
        )
        &
        (
            analysis_df["positive_count"] > 0
        )
    ]
)


print(
    "Reviews with negative evidence:",
    len(reviews_with_negative)
)


print(
    "Reviews with positive evidence:",
    len(reviews_with_positive)
)


print(
    "Reviews with both:",
    len(reviews_with_both)
)


print(
    "Reviews with no strong evidence:",
    len(
        analysis_df[
            (
                analysis_df["negative_count"]
                == 0
            )
            &
            (
                analysis_df["positive_count"]
                == 0
            )
        ]
    )
)


# ============================================================
# SHOW MIXED REVIEWS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE MIXED SENTIMENT REVIEWS")
print("=" * 70)


mixed_df = analysis_df[
    analysis_df["category"]
    == "Mixed Sentiment"
]


for number, (_, row) in enumerate(
    mixed_df.head(10).iterrows(),
    start=1
):

    print(
        f"\n{number}. {row['review']}"
    )

    print(
        "Negative evidence:",
        row["negative_evidence"]
    )

    print(
        "Positive evidence:",
        row["positive_evidence"]
    )


# ============================================================
# SHOW NEGATIVE REVIEWS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE NEGATIVE-LANGUAGE NEUTRAL REVIEWS")
print("=" * 70)


negative_df = analysis_df[
    analysis_df["category"].isin([
        "Negative Language",
        "Strong Negative Language"
    ])
]


for number, (_, row) in enumerate(
    negative_df.head(10).iterrows(),
    start=1
):

    print(
        f"\n{number}. {row['review']}"
    )

    print(
        "Negative evidence:",
        row["negative_evidence"]
    )


# ============================================================
# SHOW POSITIVE REVIEWS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE POSITIVE-LANGUAGE NEUTRAL REVIEWS")
print("=" * 70)


positive_df = analysis_df[
    analysis_df["category"].isin([
        "Positive Language",
        "Strong Positive Language"
    ])
]


for number, (_, row) in enumerate(
    positive_df.head(10).iterrows(),
    start=1
):

    print(
        f"\n{number}. {row['review']}"
    )

    print(
        "Positive evidence:",
        row["positive_evidence"]
    )


# ============================================================
# SHOW CLEARLY NEUTRAL REVIEWS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE CLEARLY NEUTRAL REVIEWS")
print("=" * 70)


clear_neutral_df = analysis_df[
    analysis_df["category"]
    == "Clearly Neutral"
]


for number, (_, row) in enumerate(
    clear_neutral_df.head(10).iterrows(),
    start=1
):

    print(
        f"\n{number}. {row['review']}"
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

OUTPUT_FILE = (
    BASE_DIR
    / "reports"
    / "model"
    / "neutral_deep_analysis.csv"
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


analysis_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("ANALYSIS FILE SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DEEP NEUTRAL ANALYSIS COMPLETED")
print("=" * 70)
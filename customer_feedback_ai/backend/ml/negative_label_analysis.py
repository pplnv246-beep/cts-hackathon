import re
import pandas as pd

from pathlib import Path


# ============================================================
# CUSTOMER FEEDBACK AI - NEGATIVE LABEL QUALITY ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nlp_ready_reviews.csv"
)


# ============================================================
# SENTIMENT PHRASES
# ============================================================

NEGATIVE_PHRASES = [

    "broken",
    "damaged",
    "defective",
    "faulty",

    "not working",
    "does not work",
    "doesn't work",
    "stopped working",

    "not good",
    "very poor",
    "poor quality",

    "terrible",
    "horrible",
    "worst",
    "useless",

    "late delivery",
    "delivery was late",
    "arrived late",
    "delivered late",

    "extremely late",
    "very late",

    "delayed delivery",
    "delivery delayed",

    "never arrived",
    "did not arrive",
    "didn't arrive",

    "never received",
    "not received",

    "never responded",
    "did not respond",
    "didn't respond",

    "no response",

    "bad experience",
    "poor experience",

    "disappointed",
    "very disappointed",

    "too expensive",
    "overpriced",

    "not happy",
    "unhappy",

    "complaint",
    "frustrating",
    "frustrated"
]


POSITIVE_PHRASES = [

    "excellent",
    "amazing",
    "fantastic",
    "perfect",

    "very good",
    "really good",
    "great product",
    "great quality",
    "excellent quality",

    "love this",
    "love the product",
    "absolutely love",

    "works perfectly",
    "works great",
    "working perfectly",

    "fast delivery",
    "very fast delivery",

    "arrived on time",
    "delivered on time",

    "highly recommend",
    "recommend this",

    "very satisfied",
    "completely satisfied",

    "happy with",
    "good quality"
]


# ============================================================
# PHRASE DETECTION
# ============================================================

def find_matches(
    text,
    phrases
):

    matches = []

    text = str(text).lower()

    for phrase in phrases:

        pattern = (
            r"\b"
            + re.escape(phrase)
            + r"\b"
        )

        if re.search(
            pattern,
            text
        ):

            matches.append(
                phrase
            )

    return matches


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - NEGATIVE LABEL QUALITY ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    "Total reviews:",
    len(df)
)


# ============================================================
# FILTER NEGATIVE REVIEWS
# ============================================================

negative_df = df[
    df["Sentiment"] == "Negative"
].copy()

print(
    "\nNegative reviews:",
    len(negative_df)
)


# ============================================================
# FIND TEXT COLUMN
# ============================================================

if "Cleaned_Review" in negative_df.columns:

    text_column = "Cleaned_Review"

elif "Processed_Review" in negative_df.columns:

    text_column = "Processed_Review"

else:

    raise ValueError(
        "Review text column not found."
    )


# ============================================================
# ANALYZE NEGATIVE REVIEWS
# ============================================================

negative_signal = 0
positive_signal = 0
mixed_signal = 0
no_signal = 0

examples = []


for _, row in negative_df.iterrows():

    text = str(
        row[text_column]
    )


    negative_matches = find_matches(
        text,
        NEGATIVE_PHRASES
    )


    positive_matches = find_matches(
        text,
        POSITIVE_PHRASES
    )


    # --------------------------------------------------------
    # Negative only
    # --------------------------------------------------------

    if (
        negative_matches
        and not positive_matches
    ):

        negative_signal += 1


    # --------------------------------------------------------
    # Positive only
    # --------------------------------------------------------

    elif (
        positive_matches
        and not negative_matches
    ):

        positive_signal += 1

        if len(examples) < 30:

            examples.append({

                "type": "Positive-looking",

                "text": text,

                "negative": negative_matches,

                "positive": positive_matches

            })


    # --------------------------------------------------------
    # Mixed
    # --------------------------------------------------------

    elif (
        negative_matches
        and positive_matches
    ):

        mixed_signal += 1

        if len(examples) < 30:

            examples.append({

                "type": "Mixed",

                "text": text,

                "negative": negative_matches,

                "positive": positive_matches

            })


    # --------------------------------------------------------
    # No signal
    # --------------------------------------------------------

    else:

        no_signal += 1


# ============================================================
# RESULTS
# ============================================================

total_negative = len(
    negative_df
)


print("\n" + "=" * 70)
print("NEGATIVE LABEL QUALITY RESULTS")
print("=" * 70)

print(
    "\nNegative-looking Negative:",
    negative_signal
)

print(
    "Positive-looking Negative:",
    positive_signal
)

print(
    "Mixed Negative:",
    mixed_signal
)

print(
    "No strong signal:",
    no_signal
)


# ============================================================
# PERCENTAGES
# ============================================================

if total_negative > 0:

    print("\nPercentages:")

    print(
        "Negative-looking:",
        round(
            negative_signal
            / total_negative
            * 100,
            2
        ),
        "%"
    )

    print(
        "Positive-looking:",
        round(
            positive_signal
            / total_negative
            * 100,
            2
        ),
        "%"
    )

    print(
        "Mixed:",
        round(
            mixed_signal
            / total_negative
            * 100,
            2
        ),
        "%"
    )

    print(
        "No strong signal:",
        round(
            no_signal
            / total_negative
            * 100,
            2
        ),
        "%"
    )


# ============================================================
# SAMPLE POSSIBLE LABEL ISSUES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE NEGATIVE LABEL QUALITY ISSUES")
print("=" * 70)


for index, example in enumerate(
    examples,
    start=1
):

    print(
        f"\n{index}. [{example['type']}]"
    )

    print(
        "Review:",
        example["text"][:500]
    )

    print(
        "Negative phrases:",
        example["negative"]
    )

    print(
        "Positive phrases:",
        example["positive"]
    )


# ============================================================
# RATING DISTRIBUTION
# ============================================================

if "Rating" in negative_df.columns:

    print("\n" + "=" * 70)
    print("NEGATIVE RATING DISTRIBUTION")
    print("=" * 70)

    print(
        negative_df["Rating"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("NEGATIVE LABEL QUALITY ANALYSIS COMPLETED")
print("=" * 70)
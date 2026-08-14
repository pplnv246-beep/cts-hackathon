import re
import pandas as pd

from pathlib import Path


# ============================================================
# CUSTOMER FEEDBACK AI - LABEL QUALITY ANALYSIS
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
print("CUSTOMER FEEDBACK AI - LABEL QUALITY ANALYSIS")
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
# CHECK SENTIMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT DISTRIBUTION")
print("=" * 70)

print(
    df["Sentiment"]
    .value_counts()
)


# ============================================================
# NEUTRAL REVIEWS
# ============================================================

neutral_df = df[
    df["Sentiment"] == "Neutral"
].copy()

print("\n" + "=" * 70)
print("NEUTRAL LABEL QUALITY")
print("=" * 70)

print(
    "\nNeutral reviews:",
    len(neutral_df)
)


# ============================================================
# FIND TEXT COLUMN
# ============================================================

if "Cleaned_Review" in neutral_df.columns:

    text_column = "Cleaned_Review"

elif "Processed_Review" in neutral_df.columns:

    text_column = "Processed_Review"

else:

    raise ValueError(
        "Review text column not found."
    )


# ============================================================
# ANALYZE NEUTRAL REVIEWS
# ============================================================

neutral_negative = 0
neutral_positive = 0
neutral_mixed = 0
neutral_no_signal = 0


neutral_examples = []


for _, row in neutral_df.iterrows():

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


    if (
        negative_matches
        and not positive_matches
    ):

        neutral_negative += 1

        if len(neutral_examples) < 20:

            neutral_examples.append({

                "type": "Negative-looking",

                "text": text,

                "negative": negative_matches,

                "positive": positive_matches

            })


    elif (
        positive_matches
        and not negative_matches
    ):

        neutral_positive += 1

        if len(neutral_examples) < 20:

            neutral_examples.append({

                "type": "Positive-looking",

                "text": text,

                "negative": negative_matches,

                "positive": positive_matches

            })


    elif (
        negative_matches
        and positive_matches
    ):

        neutral_mixed += 1

        if len(neutral_examples) < 20:

            neutral_examples.append({

                "type": "Mixed",

                "text": text,

                "negative": negative_matches,

                "positive": positive_matches

            })


    else:

        neutral_no_signal += 1


# ============================================================
# RESULTS
# ============================================================

print(
    "\nNegative-looking Neutral:",
    neutral_negative
)

print(
    "Positive-looking Neutral:",
    neutral_positive
)

print(
    "Mixed Neutral:",
    neutral_mixed
)

print(
    "No strong sentiment signal:",
    neutral_no_signal
)


# ============================================================
# PERCENTAGES
# ============================================================

total_neutral = len(
    neutral_df
)


if total_neutral > 0:

    print("\nPercentages:")

    print(
        "Negative-looking:",
        round(
            neutral_negative
            / total_neutral
            * 100,
            2
        ),
        "%"
    )

    print(
        "Positive-looking:",
        round(
            neutral_positive
            / total_neutral
            * 100,
            2
        ),
        "%"
    )

    print(
        "Mixed:",
        round(
            neutral_mixed
            / total_neutral
            * 100,
            2
        ),
        "%"
    )

    print(
        "No strong signal:",
        round(
            neutral_no_signal
            / total_neutral
            * 100,
            2
        ),
        "%"
    )


# ============================================================
# DISPLAY EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE LABEL QUALITY ISSUES")
print("=" * 70)


for index, example in enumerate(
    neutral_examples,
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
# RATING CHECK
# ============================================================

if "Rating" in neutral_df.columns:

    print("\n" + "=" * 70)
    print("NEUTRAL RATING CHECK")
    print("=" * 70)

    print(
        neutral_df["Rating"]
        .value_counts()
        .sort_index()
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("LABEL QUALITY ANALYSIS COMPLETED")
print("=" * 70)
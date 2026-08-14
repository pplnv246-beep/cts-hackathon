import re
import pandas as pd

from pathlib import Path


# ============================================================
# CUSTOMER FEEDBACK AI - LABEL CONFLICT ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nlp_ready_reviews.csv"
)


# ============================================================
# STRONG SENTIMENT PHRASES
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
# FIND PHRASES
# ============================================================

def find_matches(text, phrases):

    text = str(text).lower()

    matches = []

    for phrase in phrases:

        pattern = (
            r"\b"
            + re.escape(phrase)
            + r"\b"
        )

        if re.search(pattern, text):

            matches.append(phrase)

    return matches


# ============================================================
# TEXT COLUMN
# ============================================================

def get_text_column(df):

    possible_columns = [

        "Cleaned_Review",

        "Processed_Review",

        "Review",

        "review",

        "text",

        "Text"

    ]

    for column in possible_columns:

        if column in df.columns:

            return column

    raise ValueError(
        "Could not find review text column."
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - LABEL CONFLICT ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    "Total reviews:",
    len(df)
)


text_column = get_text_column(df)


# ============================================================
# CONFLICT COUNTERS
# ============================================================

neutral_negative = []
neutral_positive = []

negative_positive = []
positive_negative = []


# ============================================================
# ANALYZE REVIEWS
# ============================================================

for index, row in df.iterrows():

    sentiment = str(
        row["Sentiment"]
    )

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
    # Neutral but negative-looking
    # --------------------------------------------------------

    if (
        sentiment == "Neutral"
        and negative_matches
        and not positive_matches
    ):

        neutral_negative.append({

            "index": index,

            "rating": row.get(
                "Rating",
                None
            ),

            "text": text,

            "negative_phrases":
                negative_matches

        })


    # --------------------------------------------------------
    # Neutral but positive-looking
    # --------------------------------------------------------

    elif (
        sentiment == "Neutral"
        and positive_matches
        and not negative_matches
    ):

        neutral_positive.append({

            "index": index,

            "rating": row.get(
                "Rating",
                None
            ),

            "text": text,

            "positive_phrases":
                positive_matches

        })


    # --------------------------------------------------------
    # Negative but positive-looking
    # --------------------------------------------------------

    elif (
        sentiment == "Negative"
        and positive_matches
        and not negative_matches
    ):

        negative_positive.append({

            "index": index,

            "rating": row.get(
                "Rating",
                None
            ),

            "text": text,

            "positive_phrases":
                positive_matches

        })


    # --------------------------------------------------------
    # Positive but negative-looking
    # --------------------------------------------------------

    elif (
        sentiment == "Positive"
        and negative_matches
        and not positive_matches
    ):

        positive_negative.append({

            "index": index,

            "rating": row.get(
                "Rating",
                None
            ),

            "text": text,

            "negative_phrases":
                negative_matches

        })


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LABEL CONFLICT SUMMARY")
print("=" * 70)


print(
    "\nNeutral → Negative conflicts:",
    len(neutral_negative)
)

print(
    "Neutral → Positive conflicts:",
    len(neutral_positive)
)

print(
    "Negative → Positive conflicts:",
    len(negative_positive)
)

print(
    "Positive → Negative conflicts:",
    len(positive_negative)
)


# ============================================================
# PERCENTAGES
# ============================================================

neutral_count = (
    df["Sentiment"]
    .eq("Neutral")
    .sum()
)

negative_count = (
    df["Sentiment"]
    .eq("Negative")
    .sum()
)

positive_count = (
    df["Sentiment"]
    .eq("Positive")
    .sum()
)


print("\n" + "=" * 70)
print("CONFLICT PERCENTAGES")
print("=" * 70)


if neutral_count:

    print(
        "\nNeutral → Negative:",
        round(
            len(neutral_negative)
            / neutral_count
            * 100,
            2
        ),
        "%"
    )

    print(
        "Neutral → Positive:",
        round(
            len(neutral_positive)
            / neutral_count
            * 100,
            2
        ),
        "%"
    )


if negative_count:

    print(
        "\nNegative → Positive:",
        round(
            len(negative_positive)
            / negative_count
            * 100,
            2
        ),
        "%"
    )


if positive_count:

    print(
        "Positive → Negative:",
        round(
            len(positive_negative)
            / positive_count
            * 100,
            2
        ),
        "%"
    )


# ============================================================
# SAMPLE CONFLICTS
# ============================================================

def display_examples(
    title,
    examples,
    limit=10
):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


    if not examples:

        print("\nNo conflicts found.")

        return


    for number, item in enumerate(
        examples[:limit],
        start=1
    ):

        print(
            f"\n{number}."
        )

        print(
            "Rating:",
            item["rating"]
        )

        print(
            "Review:",
            item["text"][:500]
        )


        if "negative_phrases" in item:

            print(
                "Negative phrases:",
                item["negative_phrases"]
            )


        if "positive_phrases" in item:

            print(
                "Positive phrases:",
                item["positive_phrases"]
            )


# ============================================================
# DISPLAY
# ============================================================

display_examples(
    "NEUTRAL → NEGATIVE EXAMPLES",
    neutral_negative
)

display_examples(
    "NEUTRAL → POSITIVE EXAMPLES",
    neutral_positive
)

display_examples(
    "NEGATIVE → POSITIVE EXAMPLES",
    negative_positive
)

display_examples(
    "POSITIVE → NEGATIVE EXAMPLES",
    positive_negative
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("LABEL CONFLICT ANALYSIS COMPLETED")
print("=" * 70)
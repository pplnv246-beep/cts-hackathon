import re


# ============================================================
# CUSTOMER FEEDBACK AI - TEXT SENTIMENT ANALYZER
# ============================================================


# ============================================================
# STRONG NEGATIVE EXPRESSIONS
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


# ============================================================
# STRONG POSITIVE EXPRESSIONS
# ============================================================

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
# FIND MATCHING PHRASES
# ============================================================

def find_phrase_matches(
    text,
    phrases
):

    text = str(text).lower()

    matches = []

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
# ANALYZE TEXT SENTIMENT
# ============================================================

def analyze_text_sentiment(
    review
):

    if review is None:

        raise ValueError(
            "Review cannot be empty."
        )


    review = str(
        review
    ).strip()


    if not review:

        raise ValueError(
            "Review cannot be empty."
        )


    negative_matches = find_phrase_matches(
        review,
        NEGATIVE_PHRASES
    )


    positive_matches = find_phrase_matches(
        review,
        POSITIVE_PHRASES
    )


    negative_count = len(
        negative_matches
    )

    positive_count = len(
        positive_matches
    )


    # ========================================================
    # TEXT SENTIMENT DECISION
    # ========================================================

    if (
        negative_count > 0
        and negative_count > positive_count
    ):

        text_sentiment = "Negative"


    elif (
        positive_count > 0
        and positive_count > negative_count
    ):

        text_sentiment = "Positive"


    elif (
        negative_count > 0
        and positive_count > 0
    ):

        text_sentiment = "Mixed"


    else:

        text_sentiment = "Neutral"


    # ========================================================
    # EVIDENCE
    # ========================================================

    evidence = []

    evidence.extend(
        negative_matches
    )

    evidence.extend(
        positive_matches
    )


    # ========================================================
    # RESULT
    # ========================================================

    return {

        "text_sentiment":
            text_sentiment,

        "negative_evidence":
            negative_matches,

        "positive_evidence":
            positive_matches,

        "evidence":
            evidence

    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CUSTOMER FEEDBACK AI - TEXT SENTIMENT ANALYZER")
    print("=" * 70)


    test_reviews = [

        "The product arrived broken and the delivery was late.",

        "I absolutely love this product. The quality is excellent.",

        "The package contains the item I ordered. It arrived yesterday.",

        "The product is good but the delivery was late."

    ]


    for index, review in enumerate(
        test_reviews,
        start=1
    ):

        result = analyze_text_sentiment(
            review
        )


        print("\n" + "-" * 70)

        print(
            f"REVIEW {index}:"
        )

        print(
            review
        )

        print(
            "Text Sentiment:",
            result["text_sentiment"]
        )

        print(
            "Negative Evidence:",
            result["negative_evidence"]
        )

        print(
            "Positive Evidence:",
            result["positive_evidence"]
        )

        print(
            "Evidence:",
            result["evidence"]
        )


    print("\n" + "=" * 70)
    print("TEXT SENTIMENT ANALYSIS COMPLETED")
    print("=" * 70)
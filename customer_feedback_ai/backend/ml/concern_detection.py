import re
import os
import pandas as pd


# ============================================================
# CUSTOMER FEEDBACK AI - CONCERN DETECTION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "nlp_ready_reviews.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "concern_analyzed_reviews.csv"
)


# ============================================================
# CONCERN KEYWORDS
# ============================================================

CONCERN_KEYWORDS = {

    "Delivery": [
        "delivery",
        "delivered",
        "shipping",
        "shipment",
        "courier",
        "arrived",
        "late",
        "delay",
        "delayed",
        "dispatch"
    ],

    "Product Quality": [
        "quality",
        "broken",
        "defective",
        "damaged",
        "faulty",
        "poor quality",
        "stopped working",
        "not working",
        "durable",
        "material"
    ],

    "Customer Service": [
        "customer service",
        "support",
        "representative",
        "agent",
        "response",
        "assistance",
        "help desk"
    ],

    "Price": [
        "price",
        "expensive",
        "cost",
        "cheap",
        "value",
        "worth",
        "overpriced"
    ],

    "Refund": [
        "refund",
        "money back",
        "reimbursement",
        "return my money",
        "refunded"
    ],

    "Return": [
        "return",
        "returned",
        "replacement",
        "replace",
        "exchange"
    ],

    "Packaging": [
        "package",
        "packaging",
        "box",
        "packed",
        "packing"
    ],

    "Product Features": [
        "feature",
        "features",
        "function",
        "functionality",
        "design",
        "performance",
        "camera",
        "battery",
        "screen",
        "sound"
    ],

    "Payment": [
        "payment",
        "paid",
        "transaction",
        "billing",
        "charge",
        "charged"
    ]
}


# ============================================================
# DETECT CONCERNS
# ============================================================

def detect_concerns(review):

    if pd.isna(review):
        return []

    text = str(review).lower()

    text = re.sub(r"\s+", " ", text)

    detected = []

    for concern, keywords in CONCERN_KEYWORDS.items():

        for keyword in keywords:

            pattern = r"\b" + re.escape(keyword) + r"\b"

            if re.search(pattern, text):

                detected.append(concern)

                break

    return detected


# ============================================================
# FIND REVIEW COLUMN
# ============================================================

def find_review_column(df):

    possible_columns = [
        "Review",
        "review",
        "reviews",
        "Review Text",
        "review_text",
        "cleaned_text",
        "text",
        "Text",
        "feedback",
        "Feedback",
        "comment",
        "Comment"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    raise ValueError(
        "Could not find review/text column. "
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# FIND SENTIMENT COLUMN
# ============================================================

def find_sentiment_column(df):

    possible_columns = [
        "Sentiment",
        "sentiment",
        "label",
        "Label",
        "target",
        "Target"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


# ============================================================
# MAIN DATASET ANALYSIS
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CUSTOMER FEEDBACK AI - FULL DATASET CONCERN ANALYSIS")
    print("=" * 70)

    print()

    print("Loading NLP-ready dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print("Dataset loaded successfully.")

    print()

    print("Total reviews:", len(df))

    print()

    print("Available columns:")
    print(df.columns.tolist())

    print()

    # --------------------------------------------------------
    # Find columns
    # --------------------------------------------------------

    review_column = find_review_column(df)

    sentiment_column = find_sentiment_column(df)

    print("Review column:", review_column)

    if sentiment_column:

        print("Sentiment column:", sentiment_column)

    else:

        print("Sentiment column: Not found")

    print()

    # --------------------------------------------------------
    # Detect concerns
    # --------------------------------------------------------

    print("Detecting customer concerns...")

    df["Detected_Concerns"] = df[review_column].apply(
        detect_concerns
    )

    df["Concern_Count"] = df["Detected_Concerns"].apply(
        len
    )

    # --------------------------------------------------------
    # Convert list to readable text
    # --------------------------------------------------------

    df["Detected_Concerns"] = df["Detected_Concerns"].apply(
        lambda x: "; ".join(x) if x else "None"
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    print()

    print("=" * 70)
    print("CONCERN ANALYSIS RESULTS")
    print("=" * 70)

    print()

    print("Total Reviews:", len(df))

    reviews_with_concerns = (
        df["Concern_Count"] > 0
    ).sum()

    print(
        "Reviews with detected concerns:",
        reviews_with_concerns
    )

    print(
        "Reviews without detected concerns:",
        len(df) - reviews_with_concerns
    )

    print()

    # --------------------------------------------------------
    # Concern frequency
    # --------------------------------------------------------

    concern_counts = {}

    for concerns in df["Detected_Concerns"]:

        if concerns == "None":
            continue

        concern_list = concerns.split("; ")

        for concern in concern_list:

            concern_counts[concern] = (
                concern_counts.get(concern, 0) + 1
            )

    concern_df = pd.DataFrame(
        list(concern_counts.items()),
        columns=["Concern", "Count"]
    )

    if not concern_df.empty:

        concern_df = concern_df.sort_values(
            by="Count",
            ascending=False
        )

        concern_df["Percentage"] = (
            concern_df["Count"]
            / len(df)
            * 100
        ).round(2)

        print("Concern Distribution:")
        print()

        print(
            concern_df.to_string(index=False)
        )

    else:

        print("No concerns detected.")

    # ========================================================
    # SENTIMENT + CONCERN ANALYSIS
    # ========================================================

    if sentiment_column and not concern_df.empty:

        print()

        print("=" * 70)
        print("SENTIMENT VS CONCERN ANALYSIS")
        print("=" * 70)

        print()

        sentiment_concern_data = []

        for _, row in df.iterrows():

            concerns = row["Detected_Concerns"]

            if concerns == "None":
                continue

            for concern in concerns.split("; "):

                sentiment_concern_data.append(
                    {
                        "Sentiment": row[sentiment_column],
                        "Concern": concern
                    }
                )

        if sentiment_concern_data:

            sentiment_concern_df = pd.DataFrame(
                sentiment_concern_data
            )

            pivot = pd.crosstab(
                sentiment_concern_df["Concern"],
                sentiment_concern_df["Sentiment"]
            )

            print(
                pivot.to_string()
            )

    # ========================================================
    # TOP CONCERNS
    # ========================================================

    if not concern_df.empty:

        print()

        print("=" * 70)
        print("TOP CUSTOMER CONCERNS")
        print("=" * 70)

        print()

        for index, row in concern_df.head(10).iterrows():

            print(
                f"{row['Concern']:<25} "
                f"{row['Count']:>6} reviews "
                f"({row['Percentage']:.2f}%)"
            )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()

    print("=" * 70)
    print("CONCERN ANALYSIS COMPLETED")
    print("=" * 70)

    print()

    print("Output file saved:")

    print(OUTPUT_FILE)

    print()

    print("=" * 70)
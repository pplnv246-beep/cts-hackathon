import pandas as pd

from pathlib import Path


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

print("=" * 70)
print("CUSTOMER FEEDBACK AI - NEUTRAL CLASS ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nTotal reviews:", len(df))


# ============================================================
# FILTER NEUTRAL REVIEWS
# ============================================================

neutral_df = df[
    df["Sentiment"] == "Neutral"
].copy()

print(
    "Neutral reviews:",
    len(neutral_df)
)


# ============================================================
# DISPLAY SAMPLE REVIEWS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE NEUTRAL REVIEWS")
print("=" * 70)

for i, review in enumerate(
    neutral_df["Cleaned_Review"]
    .head(30),
    start=1
):

    print(f"\n{i}. {review}")


# ============================================================
# RATING DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("NEUTRAL RATING DISTRIBUTION")
print("=" * 70)

print(
    neutral_df["Rating"]
    .value_counts()
    .sort_index()
)


# ============================================================
# REVIEW LENGTH
# ============================================================

neutral_df["Word_Count"] = (
    neutral_df["Processed_Review"]
    .fillna("")
    .str.split()
    .str.len()
)

print("\n" + "=" * 70)
print("NEUTRAL REVIEW LENGTH")
print("=" * 70)

print(
    "Average words:",
    round(
        neutral_df["Word_Count"].mean(),
        2
    )
)

print(
    "Minimum words:",
    neutral_df["Word_Count"].min()
)

print(
    "Maximum words:",
    neutral_df["Word_Count"].max()
)


# ============================================================
# COMMON NEUTRAL WORDS
# ============================================================

print("\n" + "=" * 70)
print("COMMON WORDS IN NEUTRAL REVIEWS")
print("=" * 70)

words = (
    neutral_df["Processed_Review"]
    .fillna("")
    .str.split()
    .explode()
)

print(
    words
    .value_counts()
    .head(30)
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("NEUTRAL ANALYSIS COMPLETED")
print("=" * 70)
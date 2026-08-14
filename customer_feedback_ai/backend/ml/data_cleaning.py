import pandas as pd
import os
import re
import csv
import sys

csv.field_size_limit(2147483647)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "../data/raw/Amazon_Reviews.csv"
OUTPUT_FILE = "../data/processed/cleaned_reviews.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("CUSTOMER FEEDBACK AI - DATA CLEANING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    engine="python",
    on_bad_lines="warn"
)

print(f"Original rows    : {len(df)}")
print(f"Original columns : {len(df.columns)}")


# ============================================================
# 2. DISPLAY ORIGINAL COLUMNS
# ============================================================

print("\nOriginal columns:")

for column in df.columns:
    print("-", column)


# ============================================================
# 3. REMOVE COMPLETELY DUPLICATE ROWS
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print("\nDuplicate rows removed:", before_duplicates - after_duplicates)


# ============================================================
# 4. CLEAN RATING COLUMN
# ============================================================

print("\nCleaning Rating column...")

def extract_rating(value):
    if pd.isna(value):
        return None

    match = re.search(r"(\d+)", str(value))

    if match:
        rating = int(match.group(1))

        if 1 <= rating <= 5:
            return rating

    return None


df["Rating"] = df["Rating"].apply(extract_rating)


# ============================================================
# 5. REMOVE ROWS WITHOUT RATING
# ============================================================

before_rating = len(df)

df = df.dropna(subset=["Rating"])

after_rating = len(df)

print("Rows removed because of missing/invalid rating:",
      before_rating - after_rating)

df["Rating"] = df["Rating"].astype(int)


# ============================================================
# 6. HANDLE REVIEW TEXT
# ============================================================

print("\nCleaning review text...")

df["Review Title"] = df["Review Title"].fillna("").astype(str)

df["Review Text"] = df["Review Text"].fillna("").astype(str)


# ============================================================
# 7. COMBINE REVIEW TITLE + REVIEW TEXT
# ============================================================

df["Review"] = (
    df["Review Title"].str.strip()
    + " "
    + df["Review Text"].str.strip()
)

df["Review"] = df["Review"].str.strip()


# ============================================================
# 8. REMOVE REVIEWS WITHOUT TEXT
# ============================================================

before_text = len(df)

df = df[df["Review"].str.len() > 0]

after_text = len(df)

print("Rows removed because review text is empty:",
      before_text - after_text)


# ============================================================
# 9. CLEAN REVIEW TEXT
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["Cleaned_Review"] = df["Review"].apply(clean_text)


# ============================================================
# 10. REMOVE EMPTY CLEANED REVIEWS
# ============================================================

df = df[df["Cleaned_Review"].str.len() > 0]


# ============================================================
# 11. CREATE SENTIMENT LABEL
# ============================================================

print("\nCreating sentiment labels...")


def get_sentiment(rating):

    if rating <= 2:
        return "Negative"

    elif rating == 3:
        return "Neutral"

    else:
        return "Positive"


df["Sentiment"] = df["Rating"].apply(get_sentiment)


# ============================================================
# 12. REMOVE DUPLICATE REVIEWS
# ============================================================

before_review_duplicates = len(df)

df = df.drop_duplicates(subset=["Cleaned_Review"])

after_review_duplicates = len(df)

print(
    "Duplicate reviews removed:",
    before_review_duplicates - after_review_duplicates
)


# ============================================================
# 13. SELECT REQUIRED COLUMNS
# ============================================================

final_columns = [
    "Country",
    "Review Date",
    "Date of Experience",
    "Rating",
    "Review Title",
    "Review Text",
    "Review",
    "Cleaned_Review",
    "Sentiment"
]

df = df[final_columns]


# ============================================================
# 14. RESET INDEX
# ============================================================

df = df.reset_index(drop=True)


# ============================================================
# 15. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ============================================================
# 16. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 17. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)

print("\nFinal dataset shape:")
print(df.shape)

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())

print("\nSentiment percentage:")
print(
    df["Sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nRating distribution:")
print(df["Rating"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 cleaned records:")
print(df.head())

print("\nOutput file:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("READY FOR EDA AND NLP")
print("=" * 60)
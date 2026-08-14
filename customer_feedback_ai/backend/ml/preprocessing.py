import pandas as pd
import re
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "cleaned_reviews.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "nlp_ready_reviews.csv"


# ============================================================
# NLP SETUP
# ============================================================

stop_words = set(stopwords.words("english"))

# Preserve important negation words for sentiment analysis
important_words = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "hardly",
    "barely",
    "nothing"
}

stop_words = stop_words - important_words

lemmatizer = WordNetLemmatizer()


# ============================================================
# TEXT PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = text.split()

    # Stopword removal + lemmatization
    processed_words = []

    for word in words:

        if word not in stop_words:

            lemma = lemmatizer.lemmatize(word)

            processed_words.append(lemma)

    return " ".join(processed_words)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - NLP PREPROCESSING")
print("=" * 70)

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Records loaded:", len(df))


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {
    "Cleaned_Review",
    "Sentiment",
    "Rating"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# APPLY NLP PREPROCESSING
# ============================================================

print("\nApplying NLP preprocessing...")

df["Processed_Review"] = (
    df["Cleaned_Review"]
    .apply(preprocess_text)
)


# ============================================================
# REMOVE EMPTY PROCESSED REVIEWS
# ============================================================

before = len(df)

df = df[
    df["Processed_Review"].str.len() > 0
]

after = len(df)

print(
    "Empty processed reviews removed:",
    before - after
)


# ============================================================
# RESET INDEX
# ============================================================

df = df.reset_index(drop=True)


# ============================================================
# SAVE NLP-READY DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("BEFORE → AFTER EXAMPLES")
print("=" * 70)

examples = df[
    ["Cleaned_Review", "Processed_Review", "Rating", "Sentiment"]
].head(10)

print(examples.to_string(index=False))


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("NLP PREPROCESSING COMPLETED")
print("=" * 70)

print("\nFinal dataset shape:")
print(df.shape)

print("\nSentiment distribution:")
print(df["Sentiment"].value_counts())

print("\nOutput file:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
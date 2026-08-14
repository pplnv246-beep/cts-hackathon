import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "nlp_ready_reviews.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VECTORIZER_FILE = (
    MODEL_DIR
    / "tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - TF-IDF FEATURE ENGINEERING")
print("=" * 70)

print("\nLoading NLP-ready dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = {
    "Processed_Review",
    "Sentiment",
    "Rating"
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# REMOVE EMPTY REVIEWS
# ============================================================

df["Processed_Review"] = (
    df["Processed_Review"]
    .fillna("")
    .astype(str)
)

df = df[
    df["Processed_Review"].str.strip() != ""
]

df = df.reset_index(drop=True)


# ============================================================
# CREATE TF-IDF VECTORIZER
# ============================================================

print("\nCreating TF-IDF vectorizer...")


vectorizer = TfidfVectorizer(

    # Use single words and two-word combinations
    ngram_range=(1, 2),

    # Ignore extremely rare words
    min_df=2,

    # Ignore extremely common words
    max_df=0.95,

    # Limit vocabulary size
    max_features=20000,

    # Normalize each document
    sublinear_tf=True
)


# ============================================================
# TRANSFORM TEXT INTO NUMERICAL FEATURES
# ============================================================

print("\nTransforming reviews into numerical features...")

X = vectorizer.fit_transform(
    df["Processed_Review"]
)


# ============================================================
# TARGET VARIABLE
# ============================================================

y = df["Sentiment"]


# ============================================================
# DISPLAY FEATURE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF RESULTS")
print("=" * 70)

print("\nNumber of reviews:", X.shape[0])

print("Number of TF-IDF features:", X.shape[1])

print("Matrix shape:", X.shape)

print(
    "Matrix type:",
    type(X)
)


# ============================================================
# DISPLAY SAMPLE FEATURES
# ============================================================

feature_names = (
    vectorizer
    .get_feature_names_out()
)

print("\nFirst 30 TF-IDF features:")

print(
    feature_names[:30]
)


# ============================================================
# DISPLAY SAMPLE REVIEW VECTOR
# ============================================================

print("\nFirst review:")

print(
    df["Processed_Review"].iloc[0]
)

print("\nFirst review TF-IDF vector:")

print(
    X[0]
)


# ============================================================
# SENTIMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    y.value_counts()
)

print("\nTarget percentages:")

print(
    y.value_counts(
        normalize=True
    ).mul(100).round(2)
)


# ============================================================
# SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)

print("\nTF-IDF vectorizer saved to:")

print(VECTORIZER_FILE)


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "tfidf_features.pkl"
)

joblib.dump(
    X,
    FEATURE_FILE
)

print("\nTF-IDF feature matrix saved to:")

print(FEATURE_FILE)


# ============================================================
# SAVE TARGET
# ============================================================

TARGET_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "target.pkl"
)

joblib.dump(
    y,
    TARGET_FILE
)

print("\nTarget labels saved to:")

print(TARGET_FILE)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print("\nFiles created:")

print("- models/tfidf_vectorizer.pkl")
print("- data/processed/tfidf_features.pkl")
print("- data/processed/target.pkl")

print("\nReady for ML model training.")
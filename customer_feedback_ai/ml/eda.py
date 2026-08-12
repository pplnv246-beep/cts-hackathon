import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "cleaned_reviews.csv"
REPORT_DIR = BASE_DIR / "reports" / "eda"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CLEANED DATA
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for column in df.columns:
    print("-", column)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("2. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)


# ============================================================
# SENTIMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("3. SENTIMENT DISTRIBUTION")
print("=" * 70)

sentiment_counts = df["Sentiment"].value_counts()

print(sentiment_counts)

sentiment_percentage = (
    df["Sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nSentiment percentage:")
print(sentiment_percentage)


# ============================================================
# SENTIMENT BAR CHART
# ============================================================

plt.figure(figsize=(8, 5))

sentiment_counts.plot(kind="bar")

plt.title("Customer Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=0)

plt.tight_layout()

sentiment_chart = REPORT_DIR / "sentiment_distribution.png"

plt.savefig(sentiment_chart)

plt.close()

print("\nSaved:", sentiment_chart)


# ============================================================
# RATING DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. RATING DISTRIBUTION")
print("=" * 70)

rating_counts = df["Rating"].value_counts().sort_index()

print(rating_counts)


# ============================================================
# RATING BAR CHART
# ============================================================

plt.figure(figsize=(8, 5))

rating_counts.plot(kind="bar")

plt.title("Customer Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")

plt.xticks(rotation=0)

plt.tight_layout()

rating_chart = REPORT_DIR / "rating_distribution.png"

plt.savefig(rating_chart)

plt.close()

print("\nSaved:", rating_chart)


# ============================================================
# AVERAGE RATING
# ============================================================

print("\n" + "=" * 70)
print("5. AVERAGE RATING")
print("=" * 70)

average_rating = df["Rating"].mean()

print("Average rating:", round(average_rating, 2))


# ============================================================
# REVIEW LENGTH
# ============================================================

print("\n" + "=" * 70)
print("6. REVIEW LENGTH ANALYSIS")
print("=" * 70)

df["Review_Length"] = df["Cleaned_Review"].str.len()

df["Word_Count"] = (
    df["Cleaned_Review"]
    .str.split()
    .str.len()
)

print("Minimum characters:", df["Review_Length"].min())
print("Maximum characters:", df["Review_Length"].max())
print("Average characters:", round(df["Review_Length"].mean(), 2))

print("\nMinimum words:", df["Word_Count"].min())
print("Maximum words:", df["Word_Count"].max())
print("Average words:", round(df["Word_Count"].mean(), 2))


# ============================================================
# REVIEW LENGTH HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Word_Count"],
    bins=30
)

plt.title("Review Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Reviews")

plt.tight_layout()

length_chart = REPORT_DIR / "review_length_distribution.png"

plt.savefig(length_chart)

plt.close()

print("\nSaved:", length_chart)


# ============================================================
# COUNTRY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. TOP COUNTRIES")
print("=" * 70)

country_counts = (
    df["Country"]
    .dropna()
    .value_counts()
    .head(10)
)

print(country_counts)


# ============================================================
# COUNTRY CHART
# ============================================================

plt.figure(figsize=(10, 6))

country_counts.sort_values().plot(kind="barh")

plt.title("Top 10 Countries by Number of Reviews")
plt.xlabel("Number of Reviews")
plt.ylabel("Country")

plt.tight_layout()

country_chart = REPORT_DIR / "top_countries.png"

plt.savefig(country_chart)

plt.close()

print("\nSaved:", country_chart)


# ============================================================
# SENTIMENT BY RATING
# ============================================================

print("\n" + "=" * 70)
print("8. RATING → SENTIMENT RELATIONSHIP")
print("=" * 70)

rating_sentiment = pd.crosstab(
    df["Rating"],
    df["Sentiment"]
)

print(rating_sentiment)


# ============================================================
# SENTIMENT BY COUNTRY
# ============================================================

print("\n" + "=" * 70)
print("9. SENTIMENT BY COUNTRY")
print("=" * 70)

top_countries = (
    df["Country"]
    .dropna()
    .value_counts()
    .head(10)
    .index
)

country_sentiment = pd.crosstab(
    df[df["Country"].isin(top_countries)]["Country"],
    df[df["Country"].isin(top_countries)]["Sentiment"],
    normalize="index"
) * 100

country_sentiment = country_sentiment.round(2)

print(country_sentiment)


# ============================================================
# MOST COMMON WORDS
# ============================================================

print("\n" + "=" * 70)
print("10. MOST COMMON WORDS")
print("=" * 70)

all_words = (
    df["Cleaned_Review"]
    .fillna("")
    .str.split()
    .explode()
)

word_counts = all_words.value_counts().head(30)

print(word_counts)


# ============================================================
# COMMON WORDS CHART
# ============================================================

plt.figure(figsize=(10, 7))

word_counts.sort_values().plot(kind="barh")

plt.title("Top 30 Most Common Words")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.tight_layout()

word_chart = REPORT_DIR / "common_words.png"

plt.savefig(word_chart)

plt.close()

print("\nSaved:", word_chart)


# ============================================================
# NEGATIVE REVIEW ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("11. NEGATIVE REVIEW ANALYSIS")
print("=" * 70)

negative_reviews = df[
    df["Sentiment"] == "Negative"
]

print("Number of negative reviews:", len(negative_reviews))

if len(negative_reviews) > 0:

    negative_words = (
        negative_reviews["Cleaned_Review"]
        .fillna("")
        .str.split()
        .explode()
        .value_counts()
        .head(20)
    )

    print("\nMost common words in negative reviews:")
    print(negative_words)


# ============================================================
# POSITIVE REVIEW ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("12. POSITIVE REVIEW ANALYSIS")
print("=" * 70)

positive_reviews = df[
    df["Sentiment"] == "Positive"
]

print("Number of positive reviews:", len(positive_reviews))

if len(positive_reviews) > 0:

    positive_words = (
        positive_reviews["Cleaned_Review"]
        .fillna("")
        .str.split()
        .explode()
        .value_counts()
        .head(20)
    )

    print("\nMost common words in positive reviews:")
    print(positive_words)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("13. FINAL DATA QUALITY CHECK")
print("=" * 70)

print("Total records:", len(df))

print(
    "Duplicate reviews:",
    df["Cleaned_Review"].duplicated().sum()
)

print(
    "Empty reviews:",
    (df["Cleaned_Review"].str.len() == 0).sum()
)

print(
    "Missing ratings:",
    df["Rating"].isnull().sum()
)

print(
    "Missing sentiments:",
    df["Sentiment"].isnull().sum()
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nDataset:")
print("Records:", len(df))

print("\nSentiment:")
print(sentiment_counts)

print("\nAverage Rating:")
print(round(average_rating, 2))

print("\nAverage Review Word Count:")
print(round(df["Word_Count"].mean(), 2))

print("\nReports saved in:")
print(REPORT_DIR)

print("\nGenerated files:")

for file in REPORT_DIR.iterdir():
    print("-", file.name)

print("\n" + "=" * 70)
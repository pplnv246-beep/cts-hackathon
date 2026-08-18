import os
import pandas as pd


# ============================================================
# CUSTOMER FEEDBACK AI - ANALYTICS SERVICE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# ANALYZED DATASET
# ============================================================

ANALYZED_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "uploaded_analyzed_reviews.csv"
)


# ============================================================
# LOAD ANALYZED DATA
# ============================================================

def load_analysis_data():

    if not os.path.exists(ANALYZED_FILE):
        return pd.DataFrame()

    df = pd.read_csv(
        ANALYZED_FILE,
        low_memory=False,
        keep_default_na=False
    )

    return df


# ============================================================
# OVERALL ANALYTICS
# ============================================================

def get_overview():

    df = load_analysis_data()

    if df.empty or "Predicted_Sentiment" not in df.columns:
        return {
            "total_reviews": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "positive_percentage": 0,
            "negative_percentage": 0,
            "neutral_percentage": 0,
        }

    total_reviews = len(df)

    sentiment_counts = (
        df["Predicted_Sentiment"]
        .astype(str)
        .str.strip()
        .value_counts()
    )

    positive = int(
        sentiment_counts.get(
            "Positive",
            0
        )
    )

    negative = int(
        sentiment_counts.get(
            "Negative",
            0
        )
    )

    neutral = int(
        sentiment_counts.get(
            "Neutral",
            0
        )
    )

    positive_percentage = (
        positive / total_reviews * 100
        if total_reviews > 0
        else 0
    )

    negative_percentage = (
        negative / total_reviews * 100
        if total_reviews > 0
        else 0
    )

    neutral_percentage = (
        neutral / total_reviews * 100
        if total_reviews > 0
        else 0
    )

    return {
        "total_reviews": total_reviews,

        "positive": positive,

        "negative": negative,

        "neutral": neutral,

        "positive_percentage": round(
            positive_percentage,
            2
        ),

        "negative_percentage": round(
            negative_percentage,
            2
        ),

        "neutral_percentage": round(
            neutral_percentage,
            2
        )
    }


# ============================================================
# SENTIMENT DISTRIBUTION
# ============================================================

def get_sentiment_distribution():

    df = load_analysis_data()

    if df.empty or "Predicted_Sentiment" not in df.columns:
        return [
            {"sentiment": "Positive", "count": 0, "percentage": 0},
            {"sentiment": "Negative", "count": 0, "percentage": 0},
            {"sentiment": "Neutral", "count": 0, "percentage": 0},
        ]

    counts = (
        df["Predicted_Sentiment"]
        .astype(str)
        .str.strip()
        .value_counts()
    )

    total = len(df)

    result = []

    for sentiment in [
        "Positive",
        "Negative",
        "Neutral"
    ]:

        count = int(
            counts.get(
                sentiment,
                0
            )
        )

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        result.append(
            {
                "sentiment": sentiment,

                "count": count,

                "percentage": round(
                    percentage,
                    2
                )
            }
        )

    return result


# ============================================================
# CLEAN CONCERN VALUE
# ============================================================

def clean_concern_value(value):

    if value is None:
        return None

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    invalid_values = {
        "none",
        "nan",
        "null",
        "na",
        "n/a",
        "undefined"
    }

    if value.lower() in invalid_values:
        return None

    return value


# ============================================================
# GET VALID CONCERNS
# ============================================================

def get_valid_concerns(value):

    value = clean_concern_value(value)

    if value is None:
        return []

    concerns = []

    for concern in value.split(";"):

        concern = clean_concern_value(
            concern
        )

        if concern is not None:
            concerns.append(
                concern
            )

    return concerns


# ============================================================
# CONCERN DISTRIBUTION
# ============================================================

def get_concern_distribution():

    df = load_analysis_data()

    if df.empty or "Detected_Concerns" not in df.columns:
        return []

    concern_counts = {}

    for concerns in df[
        "Detected_Concerns"
    ]:

        valid_concerns = (
            get_valid_concerns(
                concerns
            )
        )

        for concern in valid_concerns:

            concern_counts[concern] = (
                concern_counts.get(
                    concern,
                    0
                ) + 1
            )

    total = len(df)

    result = []

    for concern, count in sorted(
        concern_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ):

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        result.append(
            {
                "concern": concern,

                "count": int(
                    count
                ),

                "percentage": round(
                    percentage,
                    2
                )
            }
        )

    return result


# ============================================================
# SENTIMENT VS CONCERN
# ============================================================

def get_concern_sentiment():

    df = load_analysis_data()

    if df.empty or "Detected_Concerns" not in df.columns or "Predicted_Sentiment" not in df.columns:
        return []

    data = []

    for _, row in df.iterrows():

        concerns = row.get(
            "Detected_Concerns",
            None
        )

        sentiment = row.get(
            "Predicted_Sentiment",
            "Unknown"
        )

        sentiment = str(
            sentiment
        ).strip()

        if sentiment not in {
            "Positive",
            "Negative",
            "Neutral"
        }:

            continue

        valid_concerns = (
            get_valid_concerns(
                concerns
            )
        )

        if not valid_concerns:
            continue

        for concern in valid_concerns:

            data.append(
                {
                    "Concern": concern,

                    "Sentiment": sentiment
                }
            )

    if not data:
        return []

    concern_df = pd.DataFrame(
        data
    )

    pivot = pd.crosstab(
        concern_df["Concern"],
        concern_df["Sentiment"]
    )

    result = []

    for concern in pivot.index:

        row = pivot.loc[concern]

        result.append(
            {
                "concern": concern,

                "positive": int(
                    row.get(
                        "Positive",
                        0
                    )
                ),

                "negative": int(
                    row.get(
                        "Negative",
                        0
                    )
                ),

                "neutral": int(
                    row.get(
                        "Neutral",
                        0
                    )
                )
            }
        )

    result.sort(
        key=lambda item:
            (
                item["negative"]
                + item["positive"]
                + item["neutral"]
            ),
        reverse=True
    )

    return result


# ============================================================
# CUSTOMER FEEDBACK TRENDS
# ============================================================

def get_trends():

    df = load_analysis_data()

    # --------------------------------------------------------
    # Find Review Date column
    # --------------------------------------------------------

    possible_date_columns = [
        "Review Date",
        "review date",
        "Review_Date",
        "review_date",
        "Date",
        "date"
    ]

    date_column = None

    for column in possible_date_columns:

        if column in df.columns:

            date_column = column

            break

    # --------------------------------------------------------
    # Case-insensitive search
    # --------------------------------------------------------

    if date_column is None:

        normalized_columns = {
            str(column).strip().lower():
                column
            for column in df.columns
        }

        for column in possible_date_columns:

            key = column.lower()

            if key in normalized_columns:

                date_column = (
                    normalized_columns[key]
                )

                break

    if date_column is None:

        return {
            "message":
                "Review Date column not found",

            "data": []
        }

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce",
        format="mixed"
    )

    valid_count = int(
        dates.notna().sum()
    )

    if valid_count == 0:

        return {
            "message":
                "No valid review dates found",

            "data": []
        }

    # --------------------------------------------------------
    # Create temporary dataframe
    # --------------------------------------------------------

    trend_df = pd.DataFrame(
        {
            "Review_Date": dates,

            "Sentiment":
                df[
                    "Predicted_Sentiment"
                ]
        }
    )

    trend_df = trend_df.dropna(
        subset=[
            "Review_Date"
        ]
    )

    # --------------------------------------------------------
    # Monthly period
    # --------------------------------------------------------

    trend_df["Month"] = (
        trend_df[
            "Review_Date"
        ]
        .dt.to_period("M")
        .astype(str)
    )

    # --------------------------------------------------------
    # Total reviews per month
    # --------------------------------------------------------

    monthly = (
        trend_df
        .groupby("Month")
        .size()
        .reset_index(
            name="reviews"
        )
    )

    # --------------------------------------------------------
    # Negative reviews per month
    # --------------------------------------------------------

    negative_monthly = (
        trend_df[
            trend_df[
                "Sentiment"
            ]
            .astype(str)
            .str.strip()
            .eq("Negative")
        ]
        .groupby("Month")
        .size()
        .reset_index(
            name="negative_reviews"
        )
    )

    # --------------------------------------------------------
    # Merge total + negative
    # --------------------------------------------------------

    monthly = monthly.merge(
        negative_monthly,
        on="Month",
        how="left"
    )

    monthly[
        "negative_reviews"
    ] = monthly[
        "negative_reviews"
    ].fillna(0)

    # --------------------------------------------------------
    # Build API result
    # --------------------------------------------------------

    result = []

    for _, row in monthly.iterrows():

        result.append(
            {
                "month":
                    row["Month"],

                "reviews":
                    int(
                        row["reviews"]
                    ),

                "negative_reviews":
                    int(
                        row[
                            "negative_reviews"
                        ]
                    )
            }
        )

    return {
        "date_column":
            date_column,

        "valid_dates":
            valid_count,

        "data":
            result
    }
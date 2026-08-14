import os
import pandas as pd


# ============================================================
# CUSTOMER FEEDBACK AI - AI BUSINESS SUMMARY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)


# ============================================================
# CURRENT ANALYZED DATASET
# ============================================================

ANALYZED_FILE = os.path.join(
    DATA_DIR,
    "uploaded_analyzed_reviews.csv"
)

MONTHLY_FILE = os.path.join(
    DATA_DIR,
    "complaint_monthly_trends.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "ai_summary.txt"
)


# ============================================================
# HELPERS
# ============================================================

def find_sentiment_column(df):

    possible_columns = [
        "Predicted_Sentiment",
        "Sentiment",
        "sentiment",
        "label",
        "Label"
    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


def calculate_percentage(count, total):

    if total == 0:
        return 0

    return round(
        (count / total) * 100,
        2
    )


# ============================================================
# GENERATE AI SUMMARY
# ============================================================

def generate_ai_summary():

    # ========================================================
    # CHECK ANALYZED FILE
    # ========================================================

    if not os.path.exists(ANALYZED_FILE):

        raise FileNotFoundError(
            "Analyzed dataset not found: "
            + ANALYZED_FILE
        )

    # ========================================================
    # LOAD CURRENT ANALYZED DATASET
    # ========================================================

    df = pd.read_csv(
        ANALYZED_FILE,
        low_memory=False,
        keep_default_na=False
    )

    total_reviews = len(df)

    if total_reviews == 0:

        return {
            "summary": "No reviews are available for analysis.",
            "dominant_sentiment": "Unknown",
            "top_concerns": [],
            "recommendations": [],
            "total_reviews": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "positive_count": 0,
            "negative_percentage": 0,
            "neutral_percentage": 0,
            "positive_percentage": 0,
            "trend_direction": "STABLE",
            "complaint_change_percentage": 0
        }

    # ========================================================
    # FIND SENTIMENT COLUMN
    # ========================================================

    sentiment_column = find_sentiment_column(df)

    if sentiment_column is None:

        raise ValueError(
            "No sentiment column found in the analyzed dataset."
        )

    # ========================================================
    # SENTIMENT ANALYSIS
    # ========================================================

    sentiment_series = (
        df[sentiment_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    negative_count = int(
        sentiment_series
        .eq("negative")
        .sum()
    )

    neutral_count = int(
        sentiment_series
        .eq("neutral")
        .sum()
    )

    positive_count = int(
        sentiment_series
        .eq("positive")
        .sum()
    )

    negative_percentage = calculate_percentage(
        negative_count,
        total_reviews
    )

    neutral_percentage = calculate_percentage(
        neutral_count,
        total_reviews
    )

    positive_percentage = calculate_percentage(
        positive_count,
        total_reviews
    )

    # ========================================================
    # DOMINANT SENTIMENT
    # ========================================================

    sentiment_counts = {
        "Negative": negative_count,
        "Neutral": neutral_count,
        "Positive": positive_count
    }

    dominant_sentiment = max(
        sentiment_counts,
        key=sentiment_counts.get
    )

    # ========================================================
    # CUSTOMER CONCERN ANALYSIS
    # ========================================================

    concern_counts = {}

    if "Detected_Concerns" in df.columns:

        for concerns in df["Detected_Concerns"]:

            concerns = str(
                concerns
            ).strip()

            if (
                not concerns
                or concerns.lower() in [
                    "none",
                    "nan"
                ]
            ):
                continue

            for concern in concerns.split(";"):

                concern = concern.strip()

                if not concern:
                    continue

                concern_counts[concern] = (
                    concern_counts.get(
                        concern,
                        0
                    ) + 1
                )

    # ========================================================
    # SORT CONCERNS
    # ========================================================

    concern_df = pd.DataFrame(
        list(
            concern_counts.items()
        ),
        columns=[
            "Concern",
            "Count"
        ]
    )

    if not concern_df.empty:

        concern_df = (
            concern_df
            .sort_values(
                by="Count",
                ascending=False
            )
            .reset_index(drop=True)
        )

    # ========================================================
    # TOP CONCERNS
    # ========================================================

    top_concerns = []

    if not concern_df.empty:

        for _, row in concern_df.head(5).iterrows():

            count = int(
                row["Count"]
            )

            top_concerns.append(
                {
                    "concern": str(
                        row["Concern"]
                    ),
                    "count": count,
                    "percentage": calculate_percentage(
                        count,
                        total_reviews
                    )
                }
            )

    # ========================================================
    # NEGATIVE REVIEW CONCERNS
    # ========================================================

    negative_df = df[
        sentiment_series == "negative"
    ].copy()

    negative_concern_counts = {}

    if (
        not negative_df.empty
        and "Detected_Concerns" in negative_df.columns
    ):

        for concerns in negative_df[
            "Detected_Concerns"
        ]:

            concerns = str(
                concerns
            ).strip()

            if (
                not concerns
                or concerns.lower() in [
                    "none",
                    "nan"
                ]
            ):
                continue

            for concern in concerns.split(";"):

                concern = concern.strip()

                if not concern:
                    continue

                negative_concern_counts[concern] = (
                    negative_concern_counts.get(
                        concern,
                        0
                    ) + 1
                )

    # ========================================================
    # SORT NEGATIVE CONCERNS
    # ========================================================

    negative_concern_df = pd.DataFrame(
        list(
            negative_concern_counts.items()
        ),
        columns=[
            "Concern",
            "Negative_Count"
        ]
    )

    if not negative_concern_df.empty:

        negative_concern_df = (
            negative_concern_df
            .sort_values(
                by="Negative_Count",
                ascending=False
            )
            .reset_index(drop=True)
        )

    top_negative_concerns = []

    if not negative_concern_df.empty:

        top_negative_concerns = (
            negative_concern_df
            .head(5)["Concern"]
            .tolist()
        )

    # ========================================================
    # COMPLAINT TREND
    # ========================================================

    trend_direction = "STABLE"
    complaint_change_percentage = 0

    if os.path.exists(MONTHLY_FILE):

        try:

            monthly = pd.read_csv(
                MONTHLY_FILE,
                low_memory=False,
                keep_default_na=False
            )

            if (
                not monthly.empty
                and "Complaints" in monthly.columns
                and "Month" in monthly.columns
            ):

                monthly = (
                    monthly
                    .sort_values("Month")
                    .reset_index(drop=True)
                )

                if len(monthly) >= 6:

                    recent = monthly.tail(3)

                    previous = monthly.iloc[-6:-3]

                elif len(monthly) >= 2:

                    midpoint = len(monthly) // 2

                    previous = monthly.iloc[
                        :midpoint
                    ]

                    recent = monthly.iloc[
                        midpoint:
                    ]

                else:

                    previous = pd.DataFrame()

                    recent = monthly

                previous_complaints = (
                    pd.to_numeric(
                        previous["Complaints"],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

                recent_complaints = (
                    pd.to_numeric(
                        recent["Complaints"],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )

                if previous_complaints > 0:

                    complaint_change_percentage = round(
                        (
                            (
                                recent_complaints
                                - previous_complaints
                            )
                            / previous_complaints
                        )
                        * 100,
                        2
                    )

                    if complaint_change_percentage > 10:

                        trend_direction = "INCREASING"

                    elif complaint_change_percentage < -10:

                        trend_direction = "DECREASING"

                    else:

                        trend_direction = "STABLE"

        except Exception:

            trend_direction = "STABLE"

            complaint_change_percentage = 0

    # ========================================================
    # BUSINESS ASSESSMENT
    # ========================================================

    if negative_percentage > positive_percentage:

        assessment = (
            "Customer sentiment is predominantly negative."
        )

    elif positive_percentage > negative_percentage:

        assessment = (
            "Customer sentiment is predominantly positive."
        )

    else:

        assessment = (
            "Customer sentiment is relatively balanced."
        )

    # ========================================================
    # BUSINESS SUMMARY
    # ========================================================

    summary_text = (
        f"Analyzed {total_reviews:,} customer reviews. "
        f"Overall sentiment is {dominant_sentiment.lower()}, "
        f"with {negative_percentage:.2f}% negative, "
        f"{neutral_percentage:.2f}% neutral, and "
        f"{positive_percentage:.2f}% positive reviews. "
        f"{assessment}"
    )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recommendation_map = {

        "Delivery":
            "Investigate delivery delays, shipping performance and logistics.",

        "Customer Service":
            "Improve customer service response times and issue resolution.",

        "Refund":
            "Review refund processing time and refund-related complaints.",

        "Return":
            "Analyze the return and replacement process for recurring issues.",

        "Price":
            "Review pricing and customer value perception.",

        "Payment":
            "Investigate payment failures and transaction-related problems.",

        "Packaging":
            "Improve packaging quality and damage prevention.",

        "Product Quality":
            "Investigate recurring product quality and defect issues.",

        "Product Features":
            "Review product features and performance expectations."
    }

    recommendations = []

    for concern in top_negative_concerns[:3]:

        if concern in recommendation_map:

            recommendations.append(
                recommendation_map[concern]
            )

    if trend_direction == "INCREASING":

        recommendations.append(
            "Recent complaint volume is increasing. "
            "Prioritize investigation of recurring negative concerns."
        )

    elif trend_direction == "DECREASING":

        recommendations.append(
            "Recent complaint volume is decreasing. "
            "Continue monitoring the improvements."
        )

    else:

        recommendations.append(
            "Complaint volume is relatively stable. "
            "Continue monitoring major customer concerns."
        )

    recommendations = list(
        dict.fromkeys(
            recommendations
        )
    )

    # ========================================================
    # SAVE TEXT SUMMARY
    # ========================================================

    report = []

    report.append("=" * 70)

    report.append(
        "CUSTOMER FEEDBACK AI - BUSINESS SUMMARY"
    )

    report.append("=" * 70)

    report.append("")

    report.append(
        "OVERALL CUSTOMER FEEDBACK"
    )

    report.append("-" * 40)

    report.append(
        f"Total reviews analyzed: "
        f"{total_reviews:,}"
    )

    report.append(
        f"Negative reviews: "
        f"{negative_count:,} "
        f"({negative_percentage:.2f}%)"
    )

    report.append(
        f"Neutral reviews: "
        f"{neutral_count:,} "
        f"({neutral_percentage:.2f}%)"
    )

    report.append(
        f"Positive reviews: "
        f"{positive_count:,} "
        f"({positive_percentage:.2f}%)"
    )

    report.append("")

    report.append(
        f"Dominant sentiment: "
        f"{dominant_sentiment}"
    )

    report.append(
        f"Complaint trend: "
        f"{trend_direction}"
    )

    report.append("")

    report.append(
        "TOP CUSTOMER CONCERNS"
    )

    report.append("-" * 40)

    for index, item in enumerate(
        top_concerns,
        start=1
    ):

        report.append(
            f"{index}. "
            f"{item['concern']} - "
            f"{item['count']:,} reviews "
            f"({item['percentage']:.2f}%)"
        )

    report.append("")

    report.append(
        "RECOMMENDATIONS"
    )

    report.append("-" * 40)

    for recommendation in recommendations:

        report.append(
            f"- {recommendation}"
        )

    report_text = "\n".join(
        report
    )

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report_text
        )

    # ========================================================
    # API RESPONSE
    # ========================================================

    return {

        "summary":
            summary_text,

        "dominant_sentiment":
            dominant_sentiment,

        "top_concerns":
            top_concerns,

        "recommendations":
            recommendations,

        "total_reviews":
            int(total_reviews),

        "negative_count":
            int(negative_count),

        "neutral_count":
            int(neutral_count),

        "positive_count":
            int(positive_count),

        "negative_percentage":
            float(negative_percentage),

        "neutral_percentage":
            float(neutral_percentage),

        "positive_percentage":
            float(positive_percentage),

        "trend_direction":
            trend_direction,

        "complaint_change_percentage":
            float(complaint_change_percentage)
    }


# ============================================================
# DIRECT SCRIPT EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "CUSTOMER FEEDBACK AI - AI SUMMARY ANALYSIS"
    )

    print("=" * 70)

    print()

    try:

        result = generate_ai_summary()

        print(
            result["summary"]
        )

        print()

        print(
            "Total reviews:",
            result["total_reviews"]
        )

        print(
            "Negative:",
            result["negative_count"],
            f"({result['negative_percentage']:.2f}%)"
        )

        print(
            "Neutral:",
            result["neutral_count"],
            f"({result['neutral_percentage']:.2f}%)"
        )

        print(
            "Positive:",
            result["positive_count"],
            f"({result['positive_percentage']:.2f}%)"
        )

        print()

        print(
            "Top concerns:"
        )

        for item in result["top_concerns"]:

            print(
                f"- {item['concern']}: "
                f"{item['count']:,} "
                f"({item['percentage']:.2f}%)"
            )

        print()

        print(
            "Recommendations:"
        )

        for recommendation in result["recommendations"]:

            print(
                f"- {recommendation}"
            )

        print()

        print(
            "Summary saved to:"
        )

        print(
            OUTPUT_FILE
        )

        print()

        print("=" * 70)

        print(
            "AI SUMMARY UPDATED SUCCESSFULLY"
        )

        print("=" * 70)

    except Exception as e:

        print(
            "ERROR:",
            e
        )
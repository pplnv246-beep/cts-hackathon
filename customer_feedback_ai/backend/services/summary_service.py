import os
import pandas as pd


# ============================================================
# CUSTOMER FEEDBACK AI
# AI SUMMARY AND BUSINESS RECOMMENDATION SERVICE
# ============================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)


ANALYZED_FILE = os.path.join(
    PROCESSED_DIR,
    "uploaded_analyzed_reviews.csv"
)


# ============================================================
# FIND SENTIMENT COLUMN
# ============================================================

def find_sentiment_column(df):

    possible_columns = [

        "Sentiment",
        "sentiment",

        "Predicted_Sentiment",
        "predicted_sentiment",

        "Sentiment_Label",
        "sentiment_label",

        "Predicted Sentiment",
        "predicted sentiment",

        "sentiment_prediction",
        "Sentiment_Prediction",

        "label",
        "Label",

        "target",
        "Target"

    ]


    # First check exact names

    for column in possible_columns:

        if column in df.columns:

            return column


    # Second check case-insensitively

    column_map = {

        str(column).strip().lower():
            column

        for column in df.columns

    }


    for column in possible_columns:

        key = column.strip().lower()

        if key in column_map:

            return column_map[key]


    # Third check columns containing
    # the word "sentiment"

    for column in df.columns:

        column_name = (
            str(column)
            .strip()
            .lower()
        )

        if "sentiment" in column_name:

            return column


    return None


# ============================================================
# FIND CONCERN COLUMN
# ============================================================

def find_concern_column(df):

    possible_columns = [

        "Detected_Concerns",
        "detected_concerns",

        "Detected Concerns",
        "detected concerns",

        "Concern",
        "concern",

        "Concerns",
        "concerns"

    ]


    for column in possible_columns:

        if column in df.columns:

            return column


    column_map = {

        str(column).strip().lower():
            column

        for column in df.columns

    }


    for column in possible_columns:

        key = column.strip().lower()

        if key in column_map:

            return column_map[key]


    for column in df.columns:

        column_name = (
            str(column)
            .strip()
            .lower()
        )

        if "concern" in column_name:

            return column


    return None


# ============================================================
# LOAD ANALYZED DATA
# ============================================================

def load_analyzed_data():

    if not os.path.exists(ANALYZED_FILE):

        raise FileNotFoundError(
            "No analyzed CSV file found. "
            "Please upload and analyze a CSV file first."
        )


    df = pd.read_csv(
        ANALYZED_FILE,
        low_memory=False
    )


    return df


# ============================================================
# GET SENTIMENT COUNTS
# ============================================================

def get_sentiment_counts(
    df,
    sentiment_column
):

    counts = {

        "Positive": 0,

        "Negative": 0,

        "Neutral": 0

    }


    if sentiment_column is None:

        return counts


    values = (
        df[sentiment_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


    for value in values:

        # ----------------------------------------------------
        # POSITIVE
        # ----------------------------------------------------

        if value in [

            "positive",
            "pos"

        ]:

            counts["Positive"] += 1


        # ----------------------------------------------------
        # NEGATIVE
        # ----------------------------------------------------

        elif value in [

            "negative",
            "neg"

        ]:

            counts["Negative"] += 1


        # ----------------------------------------------------
        # NEUTRAL
        # ----------------------------------------------------

        elif value in [

            "neutral",
            "neu"

        ]:

            counts["Neutral"] += 1


    return counts


# ============================================================
# GET CONCERN COUNTS
# ============================================================

def get_concern_counts(
    df,
    concern_column
):

    concern_counts = {}


    if concern_column is None:

        return concern_counts


    for value in (
        df[concern_column]
        .fillna("None")
    ):

        text = str(value).strip()


        if not text:

            continue


        if text.lower() == "none":

            continue


        concerns = text.split(";")


        for concern in concerns:

            concern = concern.strip()


            if not concern:

                continue


            concern_counts[concern] = (
                concern_counts.get(
                    concern,
                    0
                ) + 1
            )


    return dict(
        sorted(
            concern_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )


# ============================================================
# PERCENTAGE
# ============================================================

def calculate_percentage(
    count,
    total
):

    if total == 0:

        return 0.0


    return round(
        (count / total) * 100,
        2
    )


# ============================================================
# GENERATE SUMMARY
# ============================================================

def generate_summary():

    df = load_analyzed_data()


    total_reviews = len(df)


    # --------------------------------------------------------
    # FIND COLUMNS
    # --------------------------------------------------------

    sentiment_column = (
        find_sentiment_column(df)
    )


    concern_column = (
        find_concern_column(df)
    )


    # --------------------------------------------------------
    # SENTIMENT COUNTS
    # --------------------------------------------------------

    sentiment_counts = (
        get_sentiment_counts(
            df,
            sentiment_column
        )
    )


    # --------------------------------------------------------
    # CONCERN COUNTS
    # --------------------------------------------------------

    concern_counts = (
        get_concern_counts(
            df,
            concern_column
        )
    )


    positive = sentiment_counts["Positive"]

    negative = sentiment_counts["Negative"]

    neutral = sentiment_counts["Neutral"]


    positive_percentage = (
        calculate_percentage(
            positive,
            total_reviews
        )
    )


    negative_percentage = (
        calculate_percentage(
            negative,
            total_reviews
        )
    )


    neutral_percentage = (
        calculate_percentage(
            neutral,
            total_reviews
        )
    )


    # ========================================================
    # DOMINANT SENTIMENT
    # ========================================================

    if sum(sentiment_counts.values()) == 0:

        dominant_sentiment = "Unknown"

    else:

        dominant_sentiment = max(
            sentiment_counts,
            key=sentiment_counts.get
        )


    # ========================================================
    # TOP CONCERNS
    # ========================================================

    top_concerns = list(
        concern_counts.items()
    )[:5]


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_parts = []


    summary_parts.append(

        f"The dataset contains "
        f"{total_reviews:,} customer reviews."

    )


    if dominant_sentiment != "Unknown":

        summary_parts.append(

            f"{dominant_sentiment} sentiment "
            f"is currently dominant, with "
            f"{sentiment_counts[dominant_sentiment]:,} "
            f"reviews."

        )


    if negative_percentage >= 50:

        summary_parts.append(

            "More than half of the reviews "
            "are negative, indicating a "
            "significant level of customer "
            "dissatisfaction."

        )


    elif negative_percentage >= 30:

        summary_parts.append(

            "A considerable portion of the "
            "feedback is negative, indicating "
            "areas that require attention."

        )


    else:

        summary_parts.append(

            "Negative feedback represents "
            "a smaller portion of the overall "
            "customer feedback."

        )


    # ========================================================
    # CONCERN SUMMARY
    # ========================================================

    if top_concerns:

        top_concern = top_concerns[0]


        top_concern_name = (
            top_concern[0]
        )


        top_concern_count = (
            top_concern[1]
        )


        top_concern_percentage = (
            calculate_percentage(
                top_concern_count,
                total_reviews
            )
        )


        summary_parts.append(

            f"{top_concern_name} is the most "
            f"frequently detected customer concern, "
            f"appearing in "
            f"{top_concern_count:,} reviews "
            f"({top_concern_percentage}%)."

        )


        if len(top_concerns) >= 2:

            second_concern = (
                top_concerns[1]
            )


            summary_parts.append(

                f"The next major concern is "
                f"{second_concern[0]}, appearing "
                f"in {second_concern[1]:,} reviews."

            )


    summary = " ".join(
        summary_parts
    )


    # ========================================================
    # BUSINESS RECOMMENDATIONS
    # ========================================================

    recommendations = []


    # --------------------------------------------------------
    # SENTIMENT RECOMMENDATION
    # --------------------------------------------------------

    if negative_percentage >= 50:

        recommendations.append(

            "Prioritize the major sources of "
            "customer dissatisfaction because "
            "negative feedback represents more "
            "than half of the dataset."

        )


    elif negative_percentage >= 30:

        recommendations.append(

            "Investigate recurring negative "
            "feedback and prioritize the most "
            "frequently reported customer problems."

        )


    else:

        recommendations.append(

            "Continue monitoring negative feedback "
            "to identify emerging customer problems."

        )


    # ========================================================
    # CONCERN RECOMMENDATIONS
    # ========================================================

    recommendation_map = {

        "Delivery":

            "Investigate delivery delays, shipping "
            "performance and courier reliability.",


        "Product Quality":

            "Review product quality issues, defective "
            "items and manufacturing problems.",


        "Customer Service":

            "Improve customer-service response times "
            "and support resolution processes.",


        "Price":

            "Review pricing, perceived value and "
            "customer expectations around cost.",


        "Refund":

            "Simplify and accelerate refund processing "
            "to reduce customer frustration.",


        "Return":

            "Improve the return and replacement process "
            "and make the policy easier to understand.",


        "Packaging":

            "Review packaging quality and protection "
            "during transportation.",


        "Product Features":

            "Analyze product feature and performance "
            "feedback for possible improvements.",


        "Payment":

            "Investigate payment, billing and "
            "transaction issues to reduce checkout "
            "problems."

    }


    for concern, count in top_concerns:

        if concern in recommendation_map:

            recommendations.append(

                recommendation_map[
                    concern
                ]

            )


    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    unique_recommendations = []


    for recommendation in recommendations:

        if recommendation not in (
            unique_recommendations
        ):

            unique_recommendations.append(
                recommendation
            )


    recommendations = (
        unique_recommendations[:5]
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "total_reviews":
            total_reviews,


        "sentiment_column":
            sentiment_column,


        "concern_column":
            concern_column,


        "sentiment_distribution": {

            "positive":
                positive,

            "negative":
                negative,

            "neutral":
                neutral

        },


        "sentiment_percentages": {

            "positive":
                positive_percentage,

            "negative":
                negative_percentage,

            "neutral":
                neutral_percentage

        },


        "dominant_sentiment":
            dominant_sentiment,


        "top_concerns": [

            {

                "concern":
                    concern,

                "count":
                    count,

                "percentage":
                    calculate_percentage(
                        count,
                        total_reviews
                    )

            }

            for concern, count
            in top_concerns

        ],


        "summary":
            summary,


        "recommendations":
            recommendations

    }
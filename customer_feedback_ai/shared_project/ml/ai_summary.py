import os
import pandas as pd


# ============================================================
# CUSTOMER FEEDBACK AI - AI SUMMARY ANALYSIS
# ============================================================

# backend/ml/ai_summary.py
# Go from ml -> backend -> customer_feedback_ai
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

CONCERN_FILE = os.path.join(
    DATA_DIR,
    "concern_analyzed_reviews.csv"
)

MONTHLY_FILE = os.path.join(
    DATA_DIR,
    "complaint_monthly_trends.csv"
)

SPIKE_FILE = os.path.join(
    DATA_DIR,
    "complaint_spikes.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "ai_summary.txt"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - AI SUMMARY ANALYSIS")
print("=" * 70)

print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading analysis data...")

df = pd.read_csv(
    CONCERN_FILE,
    low_memory=False,
    keep_default_na=False
)

monthly = pd.read_csv(
    MONTHLY_FILE,
    low_memory=False,
    keep_default_na=False
)

spikes = pd.read_csv(
    SPIKE_FILE,
    low_memory=False,
    keep_default_na=False
)

print("Analysis data loaded successfully.")

print()


# ============================================================
# BASIC SENTIMENT ANALYSIS
# ============================================================

total_reviews = len(df)

negative_count = (
    df["Sentiment"]
    .astype(str)
    .str.lower()
    .eq("negative")
    .sum()
)

neutral_count = (
    df["Sentiment"]
    .astype(str)
    .str.lower()
    .eq("neutral")
    .sum()
)

positive_count = (
    df["Sentiment"]
    .astype(str)
    .str.lower()
    .eq("positive")
    .sum()
)


negative_percentage = (
    negative_count / total_reviews * 100
)

neutral_percentage = (
    neutral_count / total_reviews * 100
)

positive_percentage = (
    positive_count / total_reviews * 100
)


# ============================================================
# CONCERN ANALYSIS
# ============================================================

concern_counts = {}

for concerns in df["Detected_Concerns"]:

    concerns = str(concerns).strip()

    if not concerns or concerns.lower() in [
        "none",
        "nan"
    ]:
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

    concern_df = concern_df.sort_values(
        by="Count",
        ascending=False
    )

    concern_df["Percentage"] = (
        concern_df["Count"]
        / total_reviews
        * 100
    )


# ============================================================
# NEGATIVE CONCERN ANALYSIS
# ============================================================

negative_df = df[
    df["Sentiment"]
    .astype(str)
    .str.lower()
    .eq("negative")
].copy()


negative_concern_counts = {}


for concerns in negative_df[
    "Detected_Concerns"
]:

    concerns = str(
        concerns
    ).strip()

    if not concerns or concerns.lower() in [
        "none",
        "nan"
    ]:
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
    )


# ============================================================
# RECENT COMPLAINT TREND
# ============================================================

monthly = monthly.sort_values(
    "Month"
).reset_index(drop=True)


if len(monthly) >= 6:

    recent = monthly.tail(3)

    previous = monthly.iloc[-6:-3]

    recent_complaints = (
        recent["Complaints"].sum()
    )

    previous_complaints = (
        previous["Complaints"].sum()
    )

else:

    midpoint = len(monthly) // 2

    previous = monthly.iloc[:midpoint]

    recent = monthly.iloc[midpoint:]

    previous_complaints = (
        previous["Complaints"].sum()
    )

    recent_complaints = (
        recent["Complaints"].sum()
    )


if previous_complaints > 0:

    complaint_change_percentage = (
        (
            recent_complaints
            - previous_complaints
        )
        / previous_complaints
        * 100
    )

else:

    complaint_change_percentage = 0


if complaint_change_percentage > 10:

    trend_direction = "INCREASING"

elif complaint_change_percentage < -10:

    trend_direction = "DECREASING"

else:

    trend_direction = "STABLE"


# ============================================================
# TOP CONCERNS
# ============================================================

top_concerns = []

if not concern_df.empty:

    top_concerns = (
        concern_df
        .head(5)
        ["Concern"]
        .tolist()
    )


top_negative_concerns = []

if not negative_concern_df.empty:

    top_negative_concerns = (
        negative_concern_df
        .head(5)
        ["Concern"]
        .tolist()
    )


# ============================================================
# GENERATE SUMMARY
# ============================================================

summary = []


summary.append(
    "=" * 70
)

summary.append(
    "CUSTOMER FEEDBACK AI - BUSINESS SUMMARY"
)

summary.append(
    "=" * 70
)

summary.append("")


# ============================================================
# OVERALL CUSTOMER FEEDBACK
# ============================================================

summary.append(
    "OVERALL CUSTOMER FEEDBACK"
)

summary.append(
    "-" * 40
)

summary.append(
    f"Total reviews analyzed: "
    f"{total_reviews:,}"
)

summary.append(
    f"Negative reviews: "
    f"{negative_count:,} "
    f"({negative_percentage:.2f}%)"
)

summary.append(
    f"Neutral reviews: "
    f"{neutral_count:,} "
    f"({neutral_percentage:.2f}%)"
)

summary.append(
    f"Positive reviews: "
    f"{positive_count:,} "
    f"({positive_percentage:.2f}%)"
)

summary.append("")


# ============================================================
# SENTIMENT INTERPRETATION
# ============================================================

if negative_percentage > positive_percentage:

    summary.append(
        "Overall assessment: Customer sentiment "
        "is predominantly negative."
    )

elif positive_percentage > negative_percentage:

    summary.append(
        "Overall assessment: Customer sentiment "
        "is predominantly positive."
    )

else:

    summary.append(
        "Overall assessment: Customer sentiment "
        "is relatively balanced."
    )

summary.append("")


# ============================================================
# TOP CUSTOMER CONCERNS
# ============================================================

summary.append(
    "TOP CUSTOMER CONCERNS"
)

summary.append(
    "-" * 40
)

for index, row in concern_df.head(5).reset_index(
    drop=True
).iterrows():

    rank = index + 1

    summary.append(
        f"{rank}. {row['Concern']} - "
        f"{int(row['Count']):,} reviews "
        f"({row['Percentage']:.2f}%)"
    )

summary.append("")


# ============================================================
# MAJOR NEGATIVE CONCERNS
# ============================================================

summary.append(
    "MAJOR NEGATIVE CONCERNS"
)

summary.append(
    "-" * 40
)

for index, row in negative_concern_df.head(
    5
).reset_index(drop=True).iterrows():

    rank = index + 1

    summary.append(
        f"{rank}. {row['Concern']} - "
        f"{int(row['Negative_Count']):,} "
        f"negative reviews"
    )

summary.append("")


# ============================================================
# RECENT COMPLAINT TREND
# ============================================================

summary.append(
    "RECENT COMPLAINT TREND"
)

summary.append(
    "-" * 40
)

if len(monthly) > 0:

    summary.append(
        f"Comparison period 1: "
        f"{previous['Month'].iloc[0]} "
        f"to "
        f"{previous['Month'].iloc[-1]}"
    )

    summary.append(
        f"Comparison period 2: "
        f"{recent['Month'].iloc[0]} "
        f"to "
        f"{recent['Month'].iloc[-1]}"
    )

    summary.append(
        f"Previous period complaints: "
        f"{int(previous_complaints):,}"
    )

    summary.append(
        f"Recent period complaints: "
        f"{int(recent_complaints):,}"
    )

    summary.append(
        f"Complaint change: "
        f"{complaint_change_percentage:.2f}%"
    )

    summary.append(
        f"Trend: {trend_direction}"
    )

summary.append("")


# ============================================================
# COMPLAINT SPIKES
# ============================================================

summary.append(
    "COMPLAINT SPIKES"
)

summary.append(
    "-" * 40
)

if spikes.empty:

    summary.append(
        "No significant complaint spikes "
        "were detected."
    )

else:

    for _, row in spikes.iterrows():

        summary.append(
            f"- {row['Month']}: "
            f"{int(row['Complaints']):,} complaints"
        )

summary.append("")


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

summary.append(
    "BUSINESS INSIGHTS"
)

summary.append(
    "-" * 40
)

if top_negative_concerns:

    summary.append(
        f"- {top_negative_concerns[0]} "
        "is the leading negative concern."
    )


if len(top_negative_concerns) > 1:

    summary.append(
        f"- {top_negative_concerns[1]} "
        "is another major source of "
        "negative feedback."
    )


if trend_direction == "INCREASING":

    summary.append(
        "- Recent complaint volume is "
        "increasing and requires attention."
    )

elif trend_direction == "DECREASING":

    summary.append(
        "- Recent complaint volume is "
        "decreasing, indicating improvement."
    )

else:

    summary.append(
        "- Recent complaint volume is "
        "relatively stable."
    )

summary.append("")


# ============================================================
# RECOMMENDATIONS
# ============================================================

summary.append(
    "RECOMMENDATIONS"
)

summary.append(
    "-" * 40
)

for concern in top_negative_concerns[:3]:

    if concern == "Delivery":

        summary.append(
            "- Investigate delivery delays, "
            "shipping performance and logistics."
        )

    elif concern == "Customer Service":

        summary.append(
            "- Improve customer service response "
            "times and issue resolution."
        )

    elif concern == "Refund":

        summary.append(
            "- Review refund processing time "
            "and refund-related complaints."
        )

    elif concern == "Return":

        summary.append(
            "- Analyze the return and replacement "
            "process for recurring issues."
        )

    elif concern == "Price":

        summary.append(
            "- Review pricing and customer "
            "value perception."
        )

    elif concern == "Payment":

        summary.append(
            "- Investigate payment failures and "
            "transaction-related problems."
        )

    elif concern == "Packaging":

        summary.append(
            "- Improve packaging quality and "
            "damage prevention."
        )

    elif concern == "Product Quality":

        summary.append(
            "- Investigate recurring product "
            "quality and defect issues."
        )

    elif concern == "Product Features":

        summary.append(
            "- Review product features and "
            "performance expectations."
        )

summary.append("")


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_text = "\n".join(
    summary
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
        summary_text
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(
    summary_text
)

print()

print("=" * 70)

print(
    "AI SUMMARY ANALYSIS COMPLETED"
)

print("=" * 70)

print()

print(
    "Summary saved to:"
)

print(
    OUTPUT_FILE
)

print()
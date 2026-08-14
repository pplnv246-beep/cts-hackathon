import os
import pandas as pd


# ============================================================
# CUSTOMER FEEDBACK AI - COMPLAINT TREND DETECTION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "concern_analyzed_reviews.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MONTHLY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "complaint_monthly_trends.csv"
)

CONCERN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "concern_monthly_trends.csv"
)

SPIKE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "complaint_spikes.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CUSTOMER FEEDBACK AI - COMPLAINT TREND DETECTION")
print("=" * 70)

print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading concern-analyzed dataset...")

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


# ============================================================
# DATE PROCESSING
# ============================================================

print("Processing dates...")

date_columns = [
    "Date of Experience",
    "Review Date"
]

date_column = None

for column in date_columns:

    if column in df.columns:

        converted_dates = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        valid_dates = converted_dates.notna().sum()

        if valid_dates > 0:

            date_column = column

            df["Analysis_Date"] = converted_dates

            print(
                "Date column selected:",
                column
            )

            break


if date_column is None:

    raise ValueError(
        "No usable date column found."
    )


print()

print(
    "Valid dates:",
    df["Analysis_Date"].notna().sum()
)

print(
    "Invalid/missing dates:",
    df["Analysis_Date"].isna().sum()
)

print()


# ============================================================
# REMOVE INVALID DATES
# ============================================================

df = df[
    df["Analysis_Date"].notna()
].copy()


# ============================================================
# CREATE TIME PERIODS
# ============================================================

df["Month"] = (
    df["Analysis_Date"]
    .dt.to_period("M")
    .astype(str)
)

df["Week"] = (
    df["Analysis_Date"]
    .dt.to_period("W")
    .astype(str)
)


# ============================================================
# IDENTIFY NEGATIVE REVIEWS
# ============================================================

df["Is_Negative"] = (
    df["Sentiment"]
    .astype(str)
    .str.lower()
    .eq("negative")
)


# ============================================================
# IDENTIFY REVIEWS WITH CONCERNS
# ============================================================

df["Has_Concern"] = (
    df["Detected_Concerns"]
    .astype(str)
    .str.lower()
    .ne("none")
)


# ============================================================
# DEFINE COMPLAINT
# ============================================================

df["Is_Complaint"] = (
    df["Is_Negative"]
    & df["Has_Concern"]
)


# ============================================================
# MONTHLY SENTIMENT AND COMPLAINT TREND
# ============================================================

print("Calculating monthly trends...")

monthly = (
    df.groupby("Month")
    .agg(
        Total_Reviews=(
            "Review",
            "count"
        ),

        Negative_Reviews=(
            "Is_Negative",
            "sum"
        ),

        Positive_Reviews=(
            "Sentiment",
            lambda x:
            x.astype(str)
            .str.lower()
            .eq("positive")
            .sum()
        ),

        Neutral_Reviews=(
            "Sentiment",
            lambda x:
            x.astype(str)
            .str.lower()
            .eq("neutral")
            .sum()
        ),

        Reviews_With_Concerns=(
            "Has_Concern",
            "sum"
        ),

        Complaints=(
            "Is_Complaint",
            "sum"
        )
    )
    .reset_index()
)


# ============================================================
# CALCULATE RATES
# ============================================================

monthly["Complaint_Rate"] = (
    monthly["Complaints"]
    / monthly["Total_Reviews"]
    * 100
).round(2)


monthly["Negative_Rate"] = (
    monthly["Negative_Reviews"]
    / monthly["Total_Reviews"]
    * 100
).round(2)


monthly["Concern_Rate"] = (
    monthly["Reviews_With_Concerns"]
    / monthly["Total_Reviews"]
    * 100
).round(2)


# ============================================================
# MONTH-OVER-MONTH CHANGE
# ============================================================

monthly["Complaint_Change_Percent"] = (
    monthly["Complaints"]
    .pct_change()
    * 100
).round(2)


monthly["Negative_Change_Percent"] = (
    monthly["Negative_Reviews"]
    .pct_change()
    * 100
).round(2)


# ============================================================
# SAVE MONTHLY RESULTS
# ============================================================

monthly.to_csv(
    MONTHLY_OUTPUT,
    index=False
)


# ============================================================
# CONCERN-WISE MONTHLY TREND
# ============================================================

print("Calculating concern-wise trends...")

concern_rows = []

for _, row in df.iterrows():

    concerns = str(
        row["Detected_Concerns"]
    )

    if concerns.lower() == "none":
        continue

    for concern in concerns.split("; "):

        concern_rows.append(
            {
                "Month": row["Month"],
                "Concern": concern,
                "Sentiment": row["Sentiment"]
            }
        )


if concern_rows:

    concern_data = pd.DataFrame(
        concern_rows
    )

    concern_monthly = (
        concern_data
        .groupby(
            ["Month", "Concern"]
        )
        .agg(
            Total_Mentions=(
                "Concern",
                "count"
            ),

            Negative_Mentions=(
                "Sentiment",
                lambda x:
                x.astype(str)
                .str.lower()
                .eq("negative")
                .sum()
            )
        )
        .reset_index()
    )

    concern_monthly["Negative_Percentage"] = (
        concern_monthly["Negative_Mentions"]
        / concern_monthly["Total_Mentions"]
        * 100
    ).round(2)

    concern_monthly = (
        concern_monthly
        .sort_values(
            ["Concern", "Month"]
        )
    )

    concern_monthly["Change_Percent"] = (
        concern_monthly
        .groupby("Concern")
        ["Total_Mentions"]
        .pct_change()
        * 100
    ).round(2)

else:

    concern_monthly = pd.DataFrame()


concern_monthly.to_csv(
    CONCERN_OUTPUT,
    index=False
)


# ============================================================
# COMPLAINT SPIKE DETECTION
# ============================================================

print("Detecting complaint spikes...")

if len(monthly) >= 3:

    mean_complaints = monthly[
        "Complaints"
    ].mean()

    std_complaints = monthly[
        "Complaints"
    ].std()

    spike_threshold = (
        mean_complaints
        + 2 * std_complaints
    )

    spikes = monthly[
        monthly["Complaints"]
        > spike_threshold
    ].copy()

    spikes["Spike_Threshold"] = (
        spike_threshold
    )

else:

    spikes = pd.DataFrame()


spikes.to_csv(
    SPIKE_OUTPUT,
    index=False
)


# ============================================================
# DISPLAY MONTHLY RESULTS
# ============================================================

print()

print("=" * 70)
print("MONTHLY COMPLAINT TREND")
print("=" * 70)

print()

print(
    monthly[
        [
            "Month",
            "Total_Reviews",
            "Negative_Reviews",
            "Reviews_With_Concerns",
            "Complaints",
            "Complaint_Rate",
            "Negative_Rate",
            "Concern_Rate",
            "Complaint_Change_Percent"
        ]
    ].to_string(index=False)
)


# ============================================================
# TOP CONCERNS
# ============================================================

print()

print("=" * 70)
print("OVERALL CONCERN SUMMARY")
print("=" * 70)

print()

if not concern_monthly.empty:

    overall_concerns = (
        concern_monthly
        .groupby("Concern")
        ["Total_Mentions"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for concern, count in overall_concerns.items():

        print(
            f"{concern:<25}"
            f"{count:>6} mentions"
        )


# ============================================================
# TOP NEGATIVE CONCERNS
# ============================================================

print()

print("=" * 70)
print("TOP NEGATIVE CONCERNS")
print("=" * 70)

print()

if not concern_monthly.empty:

    negative_concerns = (
        concern_monthly
        .groupby("Concern")
        ["Negative_Mentions"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for concern, count in negative_concerns.items():

        print(
            f"{concern:<25}"
            f"{count:>6} negative reviews"
        )


# ============================================================
# COMPLAINT SPIKES
# ============================================================

print()

print("=" * 70)
print("COMPLAINT SPIKE DETECTION")
print("=" * 70)

print()

if spikes.empty:

    print(
        "No significant complaint spikes detected."
    )

else:

    print(
        "Potential complaint spikes detected:"
    )

    print()

    print(
        spikes[
            [
                "Month",
                "Complaints",
                "Complaint_Rate",
                "Complaint_Change_Percent"
            ]
        ].to_string(index=False)
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print()

print("=" * 70)
print("COMPLAINT TREND ANALYSIS COMPLETED")
print("=" * 70)

print()

print(
    "Monthly trend saved:"
)

print(MONTHLY_OUTPUT)

print()

print(
    "Concern trend saved:"
)

print(CONCERN_OUTPUT)

print()

print(
    "Complaint spike analysis saved:"
)

print(SPIKE_OUTPUT)

print()

print("=" * 70)
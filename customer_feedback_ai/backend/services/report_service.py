import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

from backend.services.analytics_service import (
    get_overview,
    get_concern_distribution,
    get_trends
)

from backend.services.summary_service import (
    generate_summary
)


# ============================================================
# CUSTOMER FEEDBACK AI
# PROFESSIONAL MANAGEMENT REPORT SERVICE
# ============================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "generated"
)


REPORT_FILE = os.path.join(
    REPORT_DIR,
    "customer_feedback_overall_report.pdf"
)


# ============================================================
# HELPERS
# ============================================================


def safe_number(value, default=0):

    try:
        return float(value)

    except (TypeError, ValueError):

        return default


def safe_int(value, default=0):

    try:
        return int(value)

    except (TypeError, ValueError):

        return default


def percentage(value, total):

    if not total:

        return 0.0

    return round(
        value / total * 100,
        2
    )


def clean_text(value):

    if value is None:

        return ""

    return str(value).strip()


# ============================================================
# GENERATE OVERALL REPORT
# ============================================================


def generate_overall_report():

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )


    # ========================================================
    # LOAD ANALYTICS
    # ========================================================

    overview = get_overview()

    concern_distribution = (
        get_concern_distribution()
    )

    trends = get_trends()

    summary = generate_summary()


    # ========================================================
    # BASIC METRICS
    # ========================================================

    total_reviews = safe_int(
        overview.get(
            "total_reviews",
            0
        )
    )


    positive_count = safe_int(
        overview.get(
            "positive",
            0
        )
    )


    negative_count = safe_int(
        overview.get(
            "negative",
            0
        )
    )


    neutral_count = safe_int(
        overview.get(
            "neutral",
            0
        )
    )


    positive_percentage = safe_number(
        overview.get(
            "positive_percentage",
            percentage(
                positive_count,
                total_reviews
            )
        )
    )


    negative_percentage = safe_number(
        overview.get(
            "negative_percentage",
            percentage(
                negative_count,
                total_reviews
            )
        )
    )


    neutral_percentage = safe_number(
        overview.get(
            "neutral_percentage",
            percentage(
                neutral_count,
                total_reviews
            )
        )
    )


    # ========================================================
    # DOMINANT SENTIMENT
    # ========================================================

    sentiment_values = {

        "Positive": positive_count,

        "Neutral": neutral_count,

        "Negative": negative_count

    }


    dominant_sentiment = max(
        sentiment_values,
        key=sentiment_values.get
    )


    # ========================================================
    # TOP CONCERNS
    # ========================================================

    valid_concerns = []

    for item in concern_distribution:

        if not isinstance(
            item,
            dict
        ):

            continue


        concern = clean_text(
            item.get(
                "concern",
                "Unknown"
            )
        )


        count = safe_int(
            item.get(
                "count",
                0
            )
        )


        item_percentage = safe_number(
            item.get(
                "percentage",
                percentage(
                    count,
                    total_reviews
                )
            )
        )


        valid_concerns.append({

            "concern":
                concern,

            "count":
                count,

            "percentage":
                item_percentage

        })


    valid_concerns = sorted(
        valid_concerns,
        key=lambda x: x["count"],
        reverse=True
    )


    top_concern = (
        valid_concerns[0]
        if valid_concerns
        else None
    )


    # ========================================================
    # TREND INFORMATION
    # ========================================================

    trend_data = []

    if isinstance(
        trends,
        dict
    ):

        trend_data = trends.get(
            "data",
            []
        )


    trend_direction = "STABLE"


    if isinstance(
        trends,
        dict
    ):

        trend_direction = str(
            trends.get(
                "trend_direction",
                trends.get(
                    "direction",
                    "STABLE"
                )
            )
        ).upper()


    # ========================================================
    # DOCUMENT
    # ========================================================

    document = SimpleDocTemplate(

        REPORT_FILE,

        pagesize=A4,

        rightMargin=18 * mm,

        leftMargin=18 * mm,

        topMargin=18 * mm,

        bottomMargin=18 * mm

    )


    styles = getSampleStyleSheet()


    # ========================================================
    # STYLES
    # ========================================================

    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=24,

        leading=28,

        spaceAfter=8,

        textColor=colors.HexColor(
            "#0f172a"
        )

    )


    subtitle_style = ParagraphStyle(

        "ReportSubtitle",

        parent=styles["Normal"],

        alignment=TA_CENTER,

        fontSize=11,

        leading=15,

        textColor=colors.HexColor(
            "#64748b"
        ),

        spaceAfter=8

    )


    heading_style = ParagraphStyle(

        "ReportHeading",

        parent=styles["Heading2"],

        fontSize=16,

        leading=20,

        spaceBefore=12,

        spaceAfter=10,

        textColor=colors.HexColor(
            "#0f172a"
        )

    )


    subheading_style = ParagraphStyle(

        "ReportSubHeading",

        parent=styles["Heading3"],

        fontSize=12,

        leading=16,

        spaceBefore=8,

        spaceAfter=6,

        textColor=colors.HexColor(
            "#334155"
        )

    )


    normal_style = ParagraphStyle(

        "ReportNormal",

        parent=styles["Normal"],

        fontSize=10,

        leading=15,

        spaceAfter=7,

        textColor=colors.HexColor(
            "#334155"
        )

    )


    small_style = ParagraphStyle(

        "ReportSmall",

        parent=styles["Normal"],

        fontSize=8,

        leading=11,

        textColor=colors.HexColor(
            "#64748b"
        )

    )


    metric_style = ParagraphStyle(

        "MetricValue",

        parent=styles["Normal"],

        fontSize=18,

        leading=22,

        alignment=TA_CENTER,

        textColor=colors.HexColor(
            "#0f172a"
        )

    )


    metric_label_style = ParagraphStyle(

        "MetricLabel",

        parent=styles["Normal"],

        fontSize=9,

        leading=12,

        alignment=TA_CENTER,

        textColor=colors.HexColor(
            "#64748b"
        )

    )


    story = []


    # ========================================================
    # TITLE
    # ========================================================

    story.append(

        Paragraph(
            "CUSTOMER FEEDBACK AI",
            title_style
        )

    )


    story.append(

        Paragraph(
            "Overall Customer Feedback Report",
            subtitle_style
        )

    )


    generated_at = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )


    story.append(

        Paragraph(
            f"Generated on {generated_at}",
            subtitle_style
        )

    )


    story.append(
        Spacer(
            1,
            8
        )
    )


    # ========================================================
    # REPORT PURPOSE
    # ========================================================

    story.append(

        Paragraph(
            "Management Report",
            heading_style
        )

    )


    story.append(

        Paragraph(
            "This report summarizes customer feedback at an "
            "organizational level. It highlights overall "
            "sentiment, major customer concerns, business "
            "findings, complaint trends and recommended "
            "actions.",
            normal_style
        )

    )


    # ========================================================
    # 1. EXECUTIVE SUMMARY
    # ========================================================

    story.append(

        Paragraph(
            "1. Executive Summary",
            heading_style
        )

    )


    if dominant_sentiment == "Negative":

        overall_statement = (
            "Customer sentiment is predominantly negative, "
            "indicating a significant level of customer "
            "dissatisfaction that requires management attention."
        )

    elif dominant_sentiment == "Positive":

        overall_statement = (
            "Customer sentiment is predominantly positive, "
            "indicating generally favorable customer experiences."
        )

    else:

        overall_statement = (
            "Customer sentiment is predominantly neutral, "
            "indicating that customer experiences are mixed or "
            "not strongly positive or negative."
        )


    story.append(

        Paragraph(
            overall_statement,
            normal_style
        )

    )


    story.append(

        Paragraph(
            f"A total of <b>{total_reviews:,}</b> customer "
            f"reviews were analyzed. The dominant sentiment "
            f"was <b>{dominant_sentiment}</b>.",
            normal_style
        )

    )


    if top_concern:

        story.append(

            Paragraph(
                f"The largest identified customer concern was "
                f"<b>{top_concern['concern']}</b>, appearing in "
                f"<b>{top_concern['count']:,}</b> reviews "
                f"({top_concern['percentage']:.2f}%).",
                normal_style
            )

        )


    # ========================================================
    # 2. SENTIMENT OVERVIEW
    # ========================================================

    story.append(

        Paragraph(
            "2. Sentiment Overview",
            heading_style
        )

    )


    sentiment_data = [

        [
            "Sentiment",
            "Review Count",
            "Percentage"
        ],

        [
            "Negative",
            f"{negative_count:,}",
            f"{negative_percentage:.2f}%"
        ],

        [
            "Neutral",
            f"{neutral_count:,}",
            f"{neutral_percentage:.2f}%"
        ],

        [
            "Positive",
            f"{positive_count:,}",
            f"{positive_percentage:.2f}%"
        ]

    ]


    sentiment_table = Table(

        sentiment_data,

        colWidths=[
            60 * mm,
            45 * mm,
            45 * mm
        ],

        repeatRows=1

    )


    sentiment_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1e293b")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cbd5e1")
            ),

            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER"
            ),

            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#f8fafc")
                ]
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])

    )


    story.append(
        sentiment_table
    )


    story.append(
        Spacer(
            1,
            8
        )
    )


    # ========================================================
    # KEY SENTIMENT FINDING
    # ========================================================

    story.append(

        Paragraph(
            f"<b>Management observation:</b> "
            f"{negative_percentage:.2f}% of analyzed reviews "
            f"are negative, compared with "
            f"{positive_percentage:.2f}% positive reviews.",
            normal_style
        )

    )


    # ========================================================
    # 3. KEY BUSINESS FINDINGS
    # ========================================================

    story.append(

        Paragraph(
            "3. Key Business Findings",
            heading_style
        )

    )


    findings = []


    if negative_percentage > positive_percentage:

        findings.append(
            "Negative customer feedback exceeds positive "
            "feedback and should be treated as the primary "
            "business improvement signal."
        )

    else:

        findings.append(
            "Positive customer feedback is at or above the "
            "negative feedback level, indicating generally "
            "favorable customer perception."
        )


    if top_concern:

        findings.append(
            f"{top_concern['concern']} is the largest "
            f"identified customer concern with "
            f"{top_concern['count']:,} related reviews."
        )


    if len(valid_concerns) >= 2:

        second_concern = valid_concerns[1]

        findings.append(
            f"{second_concern['concern']} is the second-largest "
            f"identified concern with "
            f"{second_concern['count']:,} related reviews."
        )


    if trend_direction in [
        "INCREASING",
        "UP",
        "RISING"
    ]:

        findings.append(
            "Complaint activity shows an increasing trend. "
            "Management should investigate the causes of the "
            "recent increase."
        )

    elif trend_direction in [
        "DECREASING",
        "DOWN",
        "FALLING"
    ]:

        findings.append(
            "Complaint activity shows a decreasing trend, "
            "which may indicate improvement in customer "
            "experience."
        )

    else:

        findings.append(
            "Complaint activity appears relatively stable. "
            "Major customer concerns should continue to be "
            "monitored."
        )


    for index, finding in enumerate(
        findings,
        start=1
    ):

        story.append(

            Paragraph(
                f"<b>{index}.</b> {finding}",
                normal_style
            )

        )


    # ========================================================
    # 4. TOP CUSTOMER CONCERNS
    # ========================================================

    story.append(

        Paragraph(
            "4. Top Customer Concerns",
            heading_style
        )

    )


    concern_rows = [

        [
            "Rank",
            "Concern",
            "Reviews",
            "Percentage"
        ]

    ]


    for index, item in enumerate(
        valid_concerns[:10],
        start=1
    ):

        concern_rows.append([

            str(index),

            item["concern"],

            f"{item['count']:,}",

            f"{item['percentage']:.2f}%"

        ])


    if len(
        concern_rows
    ) == 1:

        concern_rows.append([

            "-",

            "No concern data available",

            "-",

            "-"

        ])


    concern_table = Table(

        concern_rows,

        colWidths=[
            18 * mm,
            72 * mm,
            35 * mm,
            35 * mm
        ],

        repeatRows=1

    )


    concern_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1e293b")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cbd5e1")
            ),

            (
                "ALIGN",
                (0, 0),
                (0, -1),
                "CENTER"
            ),

            (
                "ALIGN",
                (2, 1),
                (-1, -1),
                "CENTER"
            ),

            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#f8fafc")
                ]
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )

        ])

    )


    story.append(
        concern_table
    )


    # ========================================================
    # 5. RECOMMENDED BUSINESS ACTIONS
    # ========================================================

    story.append(

        Paragraph(
            "5. Recommended Business Actions",
            heading_style
        )

    )


    recommendation_map = {

        "Delivery":
            "Investigate delivery delays, shipping "
            "performance and logistics operations.",

        "Customer Service":
            "Improve customer service response times, "
            "staff responsiveness and issue resolution.",

        "Refund":
            "Review refund processing times and identify "
            "recurring refund-related failures.",

        "Return":
            "Analyze the return and replacement process "
            "for recurring customer difficulties.",

        "Price":
            "Review pricing, customer value perception "
            "and competitive positioning.",

        "Payment":
            "Investigate payment failures and "
            "transaction-related problems.",

        "Packaging":
            "Improve packaging quality and strengthen "
            "damage-prevention processes.",

        "Product Quality":
            "Investigate recurring product defects and "
            "quality-related complaints.",

        "Product Features":
            "Review product features and performance "
            "against customer expectations."

    }


    recommendations = []


    for item in valid_concerns[:5]:

        concern = item["concern"]


        if concern in recommendation_map:

            recommendations.append(

                recommendation_map[
                    concern
                ]

            )


    if not recommendations:

        recommendations.append(
            "Continue monitoring customer sentiment "
            "and major feedback categories."
        )


    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        story.append(

            Paragraph(
                f"<b>{index}.</b> {recommendation}",
                normal_style
            )

        )


    # ========================================================
    # 6. COMPLAINT TREND
    # ========================================================

    story.append(
        PageBreak()
    )


    story.append(

        Paragraph(
            "6. Complaint Trend",
            heading_style
        )

    )


    story.append(

        Paragraph(
            f"<b>Current trend assessment:</b> "
            f"{trend_direction}",
            normal_style
        )

    )


    if trend_data:

        trend_rows = [

            [
                "Month",
                "Total Reviews",
                "Negative Reviews"
            ]

        ]


        for item in trend_data[-12:]:

            if not isinstance(
                item,
                dict
            ):

                continue


            trend_rows.append([

                clean_text(
                    item.get(
                        "month",
                        ""
                    )
                ),

                f"{safe_int(item.get('reviews', 0)):,}",

                f"{safe_int(item.get('negative_reviews', 0)):,}"

            ])


        if len(trend_rows) > 1:

            trend_table = Table(

                trend_rows,

                colWidths=[
                    55 * mm,
                    45 * mm,
                    55 * mm
                ],

                repeatRows=1

            )


            trend_table.setStyle(

                TableStyle([

                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1e293b")
                    ),

                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),

                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),

                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor("#cbd5e1")
                    ),

                    (
                        "ALIGN",
                        (1, 1),
                        (-1, -1),
                        "CENTER"
                    ),

                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            colors.white,
                            colors.HexColor("#f8fafc")
                        ]
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    )

                ])

            )


            story.append(
                trend_table
            )

        else:

            story.append(

                Paragraph(
                    "No monthly trend records are available.",
                    normal_style
                )

            )

    else:

        story.append(

            Paragraph(
                "No complaint trend data is available.",
                normal_style
            )

        )


    # ========================================================
    # 7. MANAGEMENT CONCLUSION
    # ========================================================

    story.append(

        Paragraph(
            "7. Management Conclusion",
            heading_style
        )

    )


    if dominant_sentiment == "Negative":

        conclusion = (
            f"The analysis indicates that customer "
            f"dissatisfaction is the dominant feedback pattern. "
            f"Management attention should initially focus on "
            f"the highest-volume customer concerns, particularly "
            f"{top_concern['concern'] if top_concern else 'the major identified concerns'}. "
            f"Improvements should be tracked through future "
            f"feedback analysis to determine whether sentiment "
            f"and complaint levels improve."
        )

    elif dominant_sentiment == "Positive":

        conclusion = (
            "Overall customer feedback is favorable. "
            "Management should preserve the factors driving "
            "positive experiences while continuing to monitor "
            "the major complaint categories for early warning "
            "signals."
        )

    else:

        conclusion = (
            "Customer feedback is largely neutral. Management "
            "should investigate the major concern categories "
            "and monitor changes in sentiment over time."
        )


    story.append(

        Paragraph(
            conclusion,
            normal_style
        )

    )


    # ========================================================
    # FINAL FOOTNOTE
    # ========================================================

    story.append(
        Spacer(
            1,
            20
        )
    )


    story.append(

        Paragraph(
            "This report was generated automatically by "
            "Customer Feedback AI from the analyzed customer "
            "feedback dataset.",
            small_style
        )

    )


    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )


    return REPORT_FILE
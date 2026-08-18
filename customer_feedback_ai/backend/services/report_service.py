import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether
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
    # DOCUMENT SETUP (COMPACT 12mm MARGINS FOR CONTINUOUS FLOW)
    # ========================================================

    document = SimpleDocTemplate(

        REPORT_FILE,

        pagesize=A4,

        rightMargin=12 * mm,

        leftMargin=12 * mm,

        topMargin=12 * mm,

        bottomMargin=12 * mm

    )


    styles = getSampleStyleSheet()


    # ========================================================
    # INCREASED FONT STYLES WITH TIGHT LEADING & CONTROLLED SPACING
    # ========================================================

    title_style = ParagraphStyle(

        "ReportTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=22,

        leading=25,

        spaceAfter=3,

        textColor=colors.HexColor(
            "#0f172a"
        )

    )


    subtitle_style = ParagraphStyle(

        "ReportSubtitle",

        parent=styles["Normal"],

        alignment=TA_CENTER,

        fontSize=11,

        leading=14,

        textColor=colors.HexColor(
            "#64748b"
        ),

        spaceAfter=2

    )


    heading_style = ParagraphStyle(

        "ReportHeading",

        parent=styles["Heading2"],

        fontSize=14.5,

        leading=17,

        spaceBefore=8,

        spaceAfter=3,

        keepWithNext=True,

        textColor=colors.HexColor(
            "#0f172a"
        )

    )


    subheading_style = ParagraphStyle(

        "ReportSubHeading",

        parent=styles["Heading3"],

        fontSize=12.5,

        leading=15,

        spaceBefore=6,

        spaceAfter=2,

        keepWithNext=True,

        textColor=colors.HexColor(
            "#334155"
        )

    )


    normal_style = ParagraphStyle(

        "ReportNormal",

        parent=styles["Normal"],

        fontSize=11,

        leading=14.5,

        spaceAfter=3,

        textColor=colors.HexColor(
            "#334155"
        )

    )


    small_style = ParagraphStyle(

        "ReportSmall",

        parent=styles["Normal"],

        fontSize=9.5,

        leading=12,

        textColor=colors.HexColor(
            "#64748b"
        )

    )


    story = []


    # ========================================================
    # TITLE & HEADER
    # ========================================================

    story.append(
        Paragraph("CUSTOMER FEEDBACK AI", title_style)
    )

    story.append(
        Paragraph("Overall Customer Feedback Report", subtitle_style)
    )

    generated_at = datetime.now().strftime("%d %B %Y, %I:%M %p")

    story.append(
        Paragraph(f"Generated on {generated_at}", subtitle_style)
    )

    story.append(Spacer(1, 3))


    # ========================================================
    # REPORT PURPOSE & EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph("Management Report", heading_style)
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

    story.append(
        Paragraph("1. Executive Summary", heading_style)
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

    story.append(Paragraph(overall_statement, normal_style))

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

    story.append(Spacer(1, 3))


    # ========================================================
    # 2. SENTIMENT OVERVIEW TABLE
    # ========================================================

    story.append(
        Paragraph("2. Sentiment Overview", heading_style)
    )

    sentiment_data = [
        ["Sentiment", "Review Count", "Percentage"],
        ["Negative", f"{negative_count:,}", f"{negative_percentage:.2f}%"],
        ["Neutral", f"{neutral_count:,}", f"{neutral_percentage:.2f}%"],
        ["Positive", f"{positive_count:,}", f"{positive_percentage:.2f}%"]
    ]

    sentiment_table = Table(
        sentiment_data,
        colWidths=[66 * mm, 60 * mm, 60 * mm],
        repeatRows=1
    )

    sentiment_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10.5),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(sentiment_table)

    story.append(Spacer(1, 2))

    story.append(
        Paragraph(
            f"<b>Management observation:</b> "
            f"{negative_percentage:.2f}% of analyzed reviews "
            f"are negative, compared with "
            f"{positive_percentage:.2f}% positive reviews.",
            normal_style
        )
    )

    story.append(Spacer(1, 3))


    # ========================================================
    # 3. KEY BUSINESS FINDINGS
    # ========================================================

    story.append(
        Paragraph("3. Key Business Findings", heading_style)
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

    if trend_direction in ["INCREASING", "UP", "RISING"]:
        findings.append(
            "Complaint activity shows an increasing trend. "
            "Management should investigate the causes of the "
            "recent increase."
        )
    elif trend_direction in ["DECREASING", "DOWN", "FALLING"]:
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

    for index, finding in enumerate(findings, start=1):
        story.append(
            Paragraph(f"<b>{index}.</b> {finding}", normal_style)
        )

    story.append(Spacer(1, 3))


    # ========================================================
    # 4. TOP CUSTOMER CONCERNS TABLE (CONTINUOUS TABLE SPLIT)
    # ========================================================

    story.append(
        Paragraph("4. Top Customer Concerns", heading_style)
    )

    concern_rows = [
        ["Rank", "Concern", "Reviews", "Percentage"]
    ]

    for index, item in enumerate(valid_concerns[:10], start=1):
        concern_rows.append([
            str(index),
            item["concern"],
            f"{item['count']:,}",
            f"{item['percentage']:.2f}%"
        ])

    if len(concern_rows) == 1:
        concern_rows.append(["-", "No concern data available", "-", "-"])

    concern_table = Table(
        concern_rows,
        colWidths=[18 * mm, 88 * mm, 40 * mm, 40 * mm],
        repeatRows=1
    )

    concern_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10.5),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),
            ("ALIGN", (3, 0), (3, -1), "RIGHT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(concern_table)

    story.append(Spacer(1, 3))


    # ========================================================
    # 5. RECOMMENDED BUSINESS ACTIONS
    # ========================================================

    story.append(
        Paragraph("5. Recommended Business Actions", heading_style)
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
            recommendations.append(recommendation_map[concern])

    if not recommendations:
        recommendations.append(
            "Continue monitoring customer sentiment "
            "and major feedback categories."
        )

    for index, recommendation in enumerate(recommendations, start=1):
        story.append(
            Paragraph(f"<b>{index}.</b> {recommendation}", normal_style)
        )

    story.append(Spacer(1, 3))


    # ========================================================
    # 6. COMPLAINT TREND (CONTINUOUS UNINTERRUPTED FLOW)
    # ========================================================

    story.append(
        Paragraph("6. Complaint Trend", heading_style)
    )

    story.append(
        Paragraph(f"<b>Current trend assessment:</b> {trend_direction}", normal_style)
    )

    if trend_data:
        trend_rows = [
            ["Month", "Total Reviews", "Negative Reviews"]
        ]

        for item in trend_data[-12:]:
            if not isinstance(item, dict):
                continue
            trend_rows.append([
                clean_text(item.get("month", "")),
                f"{safe_int(item.get('reviews', 0)):,}",
                f"{safe_int(item.get('negative_reviews', 0)):,}"
            ])

        if len(trend_rows) > 1:
            trend_table = Table(
                trend_rows,
                colWidths=[66 * mm, 60 * mm, 60 * mm],
                repeatRows=1
            )

            trend_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10.5),
                    ("ALIGN", (0, 0), (0, -1), "LEFT"),
                    ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                    ("ALIGN", (2, 0), (2, -1), "RIGHT"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6)
                ])
            )

            story.append(trend_table)
        else:
            story.append(
                Paragraph("No monthly trend records are available.", normal_style)
            )
    else:
        story.append(
            Paragraph("No complaint trend data is available.", normal_style)
        )

    story.append(Spacer(1, 3))


    # ========================================================
    # 7. MANAGEMENT CONCLUSION (FLOWS SEAMLESSLY)
    # ========================================================

    story.append(
        Paragraph("7. Management Conclusion", heading_style)
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

    story.append(Paragraph(conclusion, normal_style))

    story.append(Spacer(1, 6))

    story.append(
        Paragraph(
            "This report was generated automatically by "
            "Customer Feedback AI from the analyzed customer "
            "feedback dataset.",
            small_style
        )
    )


    # ========================================================
    # BUILD PDF DOCUMENT
    # ========================================================

    document.build(story)

    return REPORT_FILE
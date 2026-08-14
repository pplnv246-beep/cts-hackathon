import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.services.prediction_service import predict_sentiment
from backend.services.upload_service import save_uploaded_csv, load_csv
from backend.services.analysis_service import analyze_csv
from backend.services.prediction_service import predict_sentiment
from backend.services.upload_service import save_uploaded_csv, load_csv
from backend.services.analysis_service import analyze_csv
from backend.services.report_service import generate_overall_report
from backend.services.analytics_service import (
    get_overview,
    get_sentiment_distribution,
    get_concern_distribution,
    get_concern_sentiment,
    get_trends
)


# ============================================================
# CUSTOMER FEEDBACK AI - FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Customer Feedback AI",
    description="AI-powered customer feedback analysis system",
    version="1.0.0"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)

INDEX_FILE = os.path.join(
    FRONTEND_DIR,
    "index.html"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ReviewRequest(BaseModel):
    review: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Customer Feedback AI API is running",
        "dashboard": "/dashboard",
        "docs": "/docs"
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/dashboard")
def dashboard():

    if not os.path.exists(INDEX_FILE):

        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found."
        )

    return FileResponse(
        INDEX_FILE
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# SINGLE REVIEW PREDICTION
# ============================================================

@app.post("/predict")
def predict(request: ReviewRequest):

    if request.review is None:

        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty."
        )

    review = request.review.strip()

    if not review:

        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty."
        )

    try:

        result = predict_sentiment(
            review
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    return {
        "review": review,
        **result
    }


# ============================================================
# CSV UPLOAD
# ============================================================

@app.post("/upload")
async def upload_csv(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed."
        )

    file_content = await file.read()

    if not file_content:

        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file is empty."
        )

    try:

        file_path = save_uploaded_csv(
            file_content,
            file.filename
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to save uploaded file: {str(e)}"
        )

    try:

        df = load_csv(
            file_path
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to read CSV file. "
                "Please make sure it is a valid CSV file."
            )
        )

    return {
        "message": "CSV uploaded successfully",
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns)
    }


# ============================================================
# CSV ANALYSIS
# ============================================================

@app.post("/analyze")
async def analyze_uploaded_csv(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed."
        )

    file_content = await file.read()

    if not file_content:

        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file is empty."
        )

    try:

        file_path = save_uploaded_csv(
            file_content,
            file.filename
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to save uploaded file: {str(e)}"
        )

    try:

        result = analyze_csv(
            file_path
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

    return {
        "message": "CSV analysis completed successfully",
        "filename": file.filename,
        **result
    }


# ============================================================
# ANALYTICS - OVERVIEW
# ============================================================

@app.get("/analytics/overview")
def analytics_overview():

    try:

        return get_overview()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# ANALYTICS - SENTIMENT
# ============================================================

@app.get("/analytics/sentiment")
def analytics_sentiment():

    try:

        return {
            "data": get_sentiment_distribution()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# ANALYTICS - CONCERNS
# ============================================================

@app.get("/analytics/concerns")
def analytics_concerns():

    try:

        return {
            "data": get_concern_distribution()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# ANALYTICS - CONCERN VS SENTIMENT
# ============================================================

@app.get("/analytics/concern-sentiment")
def analytics_concern_sentiment():

    try:

        return {
            "data": get_concern_sentiment()
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# ANALYTICS - TRENDS
# ============================================================

@app.get("/analytics/trends")
def analytics_trends():

    try:

        return get_trends()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# AI SUMMARY
# ============================================================

@app.get("/ai-summary")
def ai_summary():

    try:

        from backend.ml.ai_summary import generate_ai_summary

        return generate_ai_summary()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"AI summary generation failed: {str(e)}"
        )


# ============================================================
# SERVE FRONTEND STATIC FILES
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True
    ),
    name="frontend"
)
# ============================================================
# DOWNLOAD OVERALL REPORT
# ============================================================

@app.get("/download-report")
def download_report():

    report_path = generate_overall_report()

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename="customer_feedback_overall_report.pdf"
    )
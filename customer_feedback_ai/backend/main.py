import os
import warnings

warnings.filterwarnings("ignore")

try:
    import threadpoolctl
    orig_find = getattr(threadpoolctl.ThreadpoolController, "_find_libraries_with_enum_process_module_ex", None)
    if orig_find:
        def safe_find(self):
            try:
                return orig_find(self)
            except Exception:
                return []
        threadpoolctl.ThreadpoolController._find_libraries_with_enum_process_module_ex = safe_find
except Exception:
    pass

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


from backend.services.upload_service import (
    save_uploaded_csv,
    load_csv
)

from backend.services.analysis_service import (
    analyze_csv
)

from backend.services.analytics_service import (
    get_overview,
    get_sentiment_distribution,
    get_concern_distribution,
    get_concern_sentiment,
    get_trends
)

from backend.services.report_service import (
    generate_overall_report
)


# ============================================================
# CUSTOMER FEEDBACK AI - FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Customer Feedback AI",

    description=(
        "AI-powered customer feedback "
        "analysis system"
    ),

    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "backend",
    "data",
    "processed"
)


STALE_ANALYSIS_FILES = {
    "uploaded_analyzed_reviews.csv",
    "cleaned_reviews.csv",
    "nlp_ready_reviews.csv",
    "complaint_monthly_trends.csv",
    "concern_monthly_trends.csv",
    "complaint_spikes.csv",
    "concern_spikes.csv",
    "ai_summary.txt"
}


FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)


INDEX_FILE = os.path.join(
    FRONTEND_DIR,
    "index.html"
)


def clear_stale_analysis_state():
    """Ensure a brand-new open starts from an empty state."""
    if not os.path.isdir(PROCESSED_DIR):
        return

    for file_name in os.listdir(PROCESSED_DIR):
        if file_name in STALE_ANALYSIS_FILES:
            full_path = os.path.join(PROCESSED_DIR, file_name)
            if os.path.isfile(full_path):
                os.remove(full_path)


@app.on_event("startup")
def startup_cleanup():
    clear_stale_analysis_state()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    if os.path.exists(INDEX_FILE):
        return FileResponse(INDEX_FILE)

    return {
        "message": "Customer Feedback AI API is running",
        "docs": "/docs"
    }



# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "message":
            "Customer Feedback AI API is running"

    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/dashboard")
def dashboard():

    if not os.path.exists(
        INDEX_FILE
    ):

        raise HTTPException(

            status_code=404,

            detail=
                "Dashboard file not found."

        )


    return FileResponse(
        INDEX_FILE
    )


# ============================================================
# CSV UPLOAD
# ============================================================

@app.post("/upload")
async def upload_csv(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(

            status_code=400,

            detail=
                "Only CSV files are allowed."

        )


    file_content = await file.read()


    if not file_content:

        raise HTTPException(

            status_code=400,

            detail=
                "Uploaded CSV file is empty."

        )


    try:

        file_path = save_uploaded_csv(

            file_content,

            file.filename

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
                f"Unable to save uploaded file: {str(e)}"

        )


    try:

        df = load_csv(
            file_path
        )

    except Exception:

        raise HTTPException(

            status_code=400,

            detail=(
                "Unable to read CSV file. "
                "Please make sure it is a valid CSV file."
            )

        )


    return {

        "message":
            "CSV uploaded successfully",

        "filename":
            file.filename,

        "rows":
            len(df),

        "columns":
            list(df.columns)

    }


# ============================================================
# CSV ANALYSIS
# ============================================================

@app.post("/analyze")
async def analyze_uploaded_csv(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(

            status_code=400,

            detail=
                "Only CSV files are allowed."

        )


    file_content = await file.read()


    if not file_content:

        raise HTTPException(

            status_code=400,

            detail=
                "Uploaded CSV file is empty."

        )


    try:

        file_path = save_uploaded_csv(

            file_content,

            file.filename

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
                f"Unable to save uploaded file: {str(e)}"

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

            detail=
                f"Analysis failed: {str(e)}"

        )


    return result


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

            "data":
                get_sentiment_distribution()

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ============================================================
# ANALYTICS - CUSTOMER CONCERNS
# ============================================================

@app.get("/analytics/concerns")
def analytics_concerns():

    try:

        return {

            "data":
                get_concern_distribution()

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ============================================================
# ANALYTICS - CONCERN VS SENTIMENT
# ============================================================

@app.get(
    "/analytics/concern-sentiment"
)
def analytics_concern_sentiment():

    try:

        return {

            "data":
                get_concern_sentiment()

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
# AI BUSINESS SUMMARY
# ============================================================

@app.get("/ai-summary")
def ai_summary():

    try:

        from backend.ml.ai_summary import (
            generate_ai_summary
        )

        return generate_ai_summary()

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "AI summary generation failed: "
                f"{str(e)}"
            )

        )


# ============================================================
# DOWNLOAD OVERALL PDF REPORT
# ============================================================

@app.get("/download-report")
def download_report():

    try:

        report_path = (
            generate_overall_report()
        )


        if not os.path.exists(
            report_path
        ):

            raise HTTPException(

                status_code=500,

                detail=
                    "Report file was not generated."

            )


        return FileResponse(

            path=report_path,

            media_type="application/pdf",

            filename=
                "customer_feedback_overall_report.pdf"

        )


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "Report generation failed: "
                f"{str(e)}"
            )

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
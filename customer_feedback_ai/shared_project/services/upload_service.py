import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "data",
    "uploads"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_csv(file, filename):
    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(file_path, "wb") as buffer:
        buffer.write(file)

    return file_path


def load_csv(file_path):

    try:
        # First try the normal fast parser
        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        return df

    except Exception:
        # Fallback for malformed/complex CSV files
        df = pd.read_csv(
            file_path,
            engine="python",
            on_bad_lines="warn"
        )

        return df
    @app.post("/analyze")
async def analyze_uploaded_csv(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed."
        )

    file_content = await file.read()

    file_path = save_uploaded_csv(
        file_content,
        file.filename
    )

    try:
        result = analyze_csv(file_path)

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
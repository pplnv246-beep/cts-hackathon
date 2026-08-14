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
        return pd.read_csv(
            file_path,
            low_memory=False
        )

    except Exception:

        return pd.read_csv(
            file_path,
            engine="python",
            on_bad_lines="warn"
        )
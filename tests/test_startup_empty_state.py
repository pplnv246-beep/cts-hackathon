import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from customer_feedback_ai.backend.services.analytics_service import get_overview
from customer_feedback_ai.backend.main import clear_stale_analysis_state


def test_clear_stale_analysis_state_removes_processed_csv(tmp_path, monkeypatch):
    processed_dir = tmp_path / "customer_feedback_ai" / "backend" / "data" / "processed"
    processed_dir.mkdir(parents=True)
    stale_file = processed_dir / "uploaded_analyzed_reviews.csv"
    stale_file.write_text("Predicted_Sentiment\nPositive\n")

    monkeypatch.setattr("customer_feedback_ai.backend.main.BASE_DIR", str(tmp_path))
    monkeypatch.setattr("customer_feedback_ai.backend.services.analytics_service.BASE_DIR", str(tmp_path))
    monkeypatch.setattr("customer_feedback_ai.backend.services.analytics_service.ANALYZED_FILE", str(stale_file))

    clear_stale_analysis_state()

    assert not stale_file.exists()


def test_get_overview_returns_zero_when_no_data_exists(tmp_path, monkeypatch):
    missing_file = tmp_path / "uploaded_analyzed_reviews.csv"
    monkeypatch.setattr("customer_feedback_ai.backend.services.analytics_service.ANALYZED_FILE", str(missing_file))

    overview = get_overview()

    assert overview == {
        "total_reviews": 0,
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "positive_percentage": 0,
        "negative_percentage": 0,
        "neutral_percentage": 0,
    }

import json
from pathlib import Path

import joblib
import numpy as np

from app.core.config import get_settings
from app.models.dataset import latest_feature_row_for_ticker


def _artifacts_paths() -> tuple[Path, Path]:
    settings = get_settings()
    artifacts_dir = Path(settings.model_artifacts_dir)
    return artifacts_dir / "baseline_model.joblib", artifacts_dir / "baseline_metadata.json"


def load_model_and_metadata() -> tuple[object, dict]:
    model_path, metadata_path = _artifacts_paths()
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model artifacts not found. Train model first.")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def predict_for_ticker(ticker: str) -> dict:
    model, metadata = load_model_and_metadata()
    feature_columns = metadata.get("feature_columns", [])

    row = latest_feature_row_for_ticker(ticker=ticker.upper(), window_hours=metadata.get("window_hours", 24))
    if not row:
        raise ValueError(f"No feature snapshot found for ticker {ticker.upper()}.")

    vector = []
    for col in feature_columns:
        value = row.get(col)
        if value is None:
            value = 0.0
        vector.append(float(value))

    x = np.array([vector], dtype=float)
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    if 1 in classes:
        probability_up = float(proba[classes.index(1)])
    else:
        probability_up = 0.0
    prediction = "up" if probability_up >= 0.5 else "down"
    confidence = probability_up if prediction == "up" else 1.0 - probability_up

    return {
        "ticker": ticker.upper(),
        "prediction": prediction,
        "probability_up": probability_up,
        "confidence": confidence,
        "window_end": row.get("window_end"),
        "features": {col: row.get(col) for col in feature_columns},
    }

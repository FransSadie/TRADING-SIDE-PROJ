import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.core.config import get_settings
from app.models.dataset import FEATURE_COLUMNS, load_training_dataframe, split_time_ordered


def _safe_roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float | None:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def _probability_up(model: object, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(shape=(x.shape[0],), dtype=float)


def train_and_save_baseline() -> dict:
    settings = get_settings()
    artifacts_dir = Path(settings.model_artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_dataframe(horizon_days=1, window_hours=24)
    if df.empty:
        raise RuntimeError("No training rows available. Run ingestion and feature pipeline first.")
    if len(df) < 3:
        raise RuntimeError(f"Not enough rows to train reliably (found {len(df)}).")

    train_df, val_df = split_time_ordered(df, train_ratio=0.8)
    if train_df.empty or val_df.empty:
        raise RuntimeError("Time split failed due to insufficient rows.")

    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["target_up"].to_numpy(dtype=int)
    x_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_val = val_df["target_up"].to_numpy(dtype=int)

    train_classes = np.unique(y_train)
    model_type = "logistic_regression"
    if len(train_classes) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model_type = "dummy_most_frequent"
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)

    val_proba = _probability_up(model, x_val)
    val_pred = (val_proba >= 0.5).astype(int)

    selected_returns = val_df.loc[val_pred == 1, "future_return"]
    hit_rate = float((selected_returns > 0).mean()) if len(selected_returns) > 0 else None
    avg_future_return = float(selected_returns.mean()) if len(selected_returns) > 0 else None

    metrics = {
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "recall": float(recall_score(y_val, val_pred, zero_division=0)),
        "f1": float(f1_score(y_val, val_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_val, val_proba),
        "val_rows": int(len(val_df)),
        "train_rows": int(len(train_df)),
        "signals_count": int((val_pred == 1).sum()),
        "signals_hit_rate": hit_rate,
        "signals_avg_future_return": avg_future_return,
    }

    model_path = artifacts_dir / "baseline_model.joblib"
    metadata_path = artifacts_dir / "baseline_metadata.json"

    joblib.dump(model, model_path)
    metadata = {
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "horizon_days": 1,
        "window_hours": 24,
        "train_classes": [int(v) for v in train_classes.tolist()],
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": metrics,
    }


def main() -> None:
    result = train_and_save_baseline()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

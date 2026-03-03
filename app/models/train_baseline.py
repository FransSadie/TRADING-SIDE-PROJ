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

PRICE_ONLY_COLUMNS = [
    "price_close",
    "return_1d",
    "price_return_5d",
    "rolling_volatility_20d",
    "volume_zscore_20d",
]


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


def _build_model(y_train: np.ndarray) -> tuple[object, str]:
    train_classes = np.unique(y_train)
    if len(train_classes) < 2:
        return DummyClassifier(strategy="most_frequent"), "dummy_most_frequent"
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), "logistic_regression"


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_proba),
    }


def _walk_forward_metrics(df, feature_cols: list[str], folds: int = 4) -> dict:
    if len(df) < 20:
        return {"folds": 0, "avg_accuracy": None, "avg_f1": None}

    fold_metrics = []
    step = max(5, len(df) // (folds + 1))
    for end_idx in range(step, len(df) - step + 1, step):
        train_df = df.iloc[:end_idx]
        val_df = df.iloc[end_idx : end_idx + step]
        if len(val_df) == 0:
            continue
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["target_up"].to_numpy(dtype=int)
        x_val = val_df[feature_cols].to_numpy(dtype=float)
        y_val = val_df["target_up"].to_numpy(dtype=int)

        model, _ = _build_model(y_train)
        model.fit(x_train, y_train)
        proba = _probability_up(model, x_val)
        pred = (proba >= 0.5).astype(int)
        fold_metrics.append(_classification_metrics(y_val, pred, proba))

    if not fold_metrics:
        return {"folds": 0, "avg_accuracy": None, "avg_f1": None}
    return {
        "folds": len(fold_metrics),
        "avg_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "avg_f1": float(np.mean([m["f1"] for m in fold_metrics])),
    }


def _baseline_comparison(train_df, val_df) -> dict:
    y_train = train_df["target_up"].to_numpy(dtype=int)
    y_val = val_df["target_up"].to_numpy(dtype=int)

    comparisons = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df[FEATURE_COLUMNS].to_numpy(dtype=float), y_train)
    dummy_proba = _probability_up(dummy, val_df[FEATURE_COLUMNS].to_numpy(dtype=float))
    dummy_pred = (dummy_proba >= 0.5).astype(int)
    comparisons["dummy_most_frequent"] = _classification_metrics(y_val, dummy_pred, dummy_proba)

    price_model, price_model_type = _build_model(y_train)
    price_model.fit(train_df[PRICE_ONLY_COLUMNS].to_numpy(dtype=float), y_train)
    price_proba = _probability_up(price_model, val_df[PRICE_ONLY_COLUMNS].to_numpy(dtype=float))
    price_pred = (price_proba >= 0.5).astype(int)
    comparisons[f"price_only_{price_model_type}"] = _classification_metrics(y_val, price_pred, price_proba)

    return comparisons


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
    model, model_type = _build_model(y_train)
    model.fit(x_train, y_train)

    val_proba = _probability_up(model, x_val)
    val_pred = (val_proba >= 0.5).astype(int)

    selected_returns = val_df.loc[val_pred == 1, "future_return"]
    hit_rate = float((selected_returns > 0).mean()) if len(selected_returns) > 0 else None
    avg_future_return = float(selected_returns.mean()) if len(selected_returns) > 0 else None

    metrics = {
        **_classification_metrics(y_val, val_pred, val_proba),
        "val_rows": int(len(val_df)),
        "train_rows": int(len(train_df)),
        "signals_count": int((val_pred == 1).sum()),
        "signals_hit_rate": hit_rate,
        "signals_avg_future_return": avg_future_return,
        "walk_forward": _walk_forward_metrics(df, FEATURE_COLUMNS, folds=4),
        "baselines": _baseline_comparison(train_df, val_df),
    }

    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = artifacts_dir / f"baseline_model_{version_id}.joblib"
    metadata_path = artifacts_dir / f"baseline_metadata_{version_id}.json"
    latest_model_path = artifacts_dir / "baseline_model.joblib"
    latest_metadata_path = artifacts_dir / "baseline_metadata.json"
    manifest_path = artifacts_dir / "current_model.json"

    joblib.dump(model, model_path)
    joblib.dump(model, latest_model_path)
    metadata = {
        "version_id": version_id,
        "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "horizon_days": 1,
        "window_hours": 24,
        "train_classes": [int(v) for v in train_classes.tolist()],
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    latest_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "version_id": version_id,
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "version_id": version_id,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": metrics,
    }


def main() -> None:
    result = train_and_save_baseline()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

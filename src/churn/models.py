"""Model zoo. Every model is a full Pipeline (preprocessing + classifier)."""
from __future__ import annotations

import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE
from .features import build_preprocessor


def build_models(
    numeric: list[str],
    categorical: list[str],
    pos_weight: float = 1.0,
    include_xgboost: bool = True,
    seed: int = RANDOM_STATE,
) -> dict[str, Pipeline]:
    def wrap(clf):
        return Pipeline([("prep", build_preprocessor(numeric, categorical)), ("clf", clf)])

    models: dict[str, Pipeline] = {
        "Logistic Regression": wrap(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        ),
        "Random Forest": wrap(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = wrap(
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=pos_weight,
                    eval_metric="logloss",
                    random_state=seed,
                )
            )
        except ImportError:
            warnings.warn("xgboost is not installed; skipping the XGBoost model.", stacklevel=2)
    return models

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from churn import evaluate as ev
from churn.features import (
    build_preprocessor,
    clean,
    column_groups,
    engineer,
    prepare_features,
    prepare_xy,
)
from churn.models import build_models
from churn.synthetic import make_synthetic
from churn import predict, train


def _df(n=600, seed=1):
    return make_synthetic(n, seed=seed)


def test_clean_turns_blanks_and_none_into_nan():
    out = clean(_df())
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    assert out["TotalCharges"].isna().sum() >= 0
    assert out["gender"].isna().any()
    assert not out["gender"].dropna().astype(str).str.strip().eq("").any()


def test_no_rows_dropped_for_missing_features():
    """Regression test: the original notebook called dropna() and silently lost rows."""
    raw = _df()
    X, y = prepare_xy(raw)
    assert len(X) == len(raw) == len(y)
    assert X.isna().any().any()  # missing values are kept and imputed later


def test_engineered_columns_and_identifier_removed():
    X, _ = prepare_xy(_df())
    assert "customerID" not in X.columns and "Location" not in X.columns
    assert {"TotalTickets", "Region", "State"} <= set(X.columns)


def test_invalid_target_rows_are_removed():
    raw = _df(100)
    raw.loc[0, "Churn"] = None
    X, y = prepare_xy(raw)
    assert len(X) == 99 and set(y.unique()) <= {0, 1}


def test_preprocessor_handles_unseen_category_and_all_missing_row():
    X, _ = prepare_xy(_df())
    num, cat = column_groups(X)
    prep = build_preprocessor(num, cat).fit(X)
    new = X.head(3).copy()
    new.loc[new.index[0], "Contract"] = "Brand new contract"
    new.iloc[1, :] = np.nan
    assert np.isfinite(prep.transform(new)).all()


def test_models_beat_chance_on_synthetic_data():
    X, y = prepare_xy(_df(1500, seed=3))
    num, cat = column_groups(X)
    split = int(len(X) * 0.75)
    models = build_models(num, cat, include_xgboost=False)
    for name, pipe in models.items():
        pipe.fit(X.iloc[:split], y.iloc[:split])
        m = ev.evaluate_model(pipe, X.iloc[split:], y.iloc[split:])
        assert m["roc_auc"] > 0.65, name


def test_top_decile_lift_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    prob = np.linspace(1, 0, 100)
    assert abs(ev.top_decile_lift(y, prob) - 10.0) < 1e-9


def test_feature_importance_has_readable_names():
    X, y = prepare_xy(_df())
    num, cat = column_groups(X)
    pipe = build_models(num, cat, include_xgboost=False)["Random Forest"].fit(X, y)
    imp = ev.feature_importance(pipe, top_n=5)
    assert len(imp) == 5 and imp["Importance"].is_monotonic_decreasing
    assert not imp["Feature"].str.contains("__").any()


def test_train_and_predict_cli_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        res = train.main([
            "--demo", "--no-xgboost", "--cv-folds", "2",
            "--reports-dir", str(tmp / "reports"), "--models-dir", str(tmp / "models"),
        ])
        assert (tmp / "reports" / "metrics.json").exists()
        assert (tmp / "reports" / "figures" / "roc_curves.png").exists()
        assert (tmp / "models" / "best_model.joblib").exists()
        assert json.loads((tmp / "reports" / "metrics.json").read_text())["meta"]["demo"] is True
        assert res["meta"]["best_model"] in res["models"]

        new = tmp / "new.csv"
        make_synthetic(50, seed=9).drop(columns=["Churn"]).to_csv(new, index=False)
        out = predict.main(["--model", str(tmp / "models" / "best_model.joblib"),
                            "--data", str(new), "--out", str(tmp / "scores.csv")])
        assert len(out) == 50 and out["churn_probability"].between(0, 1).all()
        assert set(out["risk_band"]) <= {"Low", "Medium", "High"}


def test_prepare_features_works_without_target():
    X = prepare_features(_df(50).drop(columns=["Churn"]))
    assert "TotalTickets" in X.columns
    assert len(engineer(clean(_df(10)))) == 10

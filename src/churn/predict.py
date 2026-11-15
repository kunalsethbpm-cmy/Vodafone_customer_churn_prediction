"""Score new customers with the saved model.

Usage:
    python -m churn.predict --model models/best_model.joblib --data new_customers.xlsx --out scores.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import ID_COL
from .data import load_data
from .features import prepare_features


def risk_band(prob: float, high: float, medium: float) -> str:
    return "High" if prob >= high else "Medium" if prob >= medium else "Low"


def main(argv: list[str] | None = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("churn_scores.csv"))
    p.add_argument("--high", type=float, default=0.6, help="probability for the High band (arbitrary default)")
    p.add_argument("--medium", type=float, default=0.3, help="probability for the Medium band (arbitrary default)")
    args = p.parse_args(argv)

    bundle = joblib.load(args.model)
    raw = load_data(args.data)
    prob = bundle["model"].predict_proba(prepare_features(raw))[:, 1]
    out = pd.DataFrame({"churn_probability": prob.round(4)})
    out["risk_band"] = [risk_band(v, args.high, args.medium) for v in prob]
    if ID_COL in raw.columns:
        out.insert(0, ID_COL, raw[ID_COL].values)
    out = out.sort_values("churn_probability", ascending=False)
    out.to_csv(args.out, index=False)
    print(f"Scored {len(out):,} customers with {bundle['name']} -> {args.out}")
    return out


if __name__ == "__main__":
    main()

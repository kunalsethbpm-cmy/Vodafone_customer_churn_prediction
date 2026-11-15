"""Synthetic stand-in data with the same schema as the Kaggle file.

ONLY for smoke tests, CI and trying the pipeline without downloading the dataset.
Metrics computed on this data say nothing about real customers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIONS = {
    "North": ["New York", "Massachusetts"],
    "South": ["Texas", "Florida"],
    "East": ["Illinois", "Ohio"],
    "West": ["California", "Washington"],
}


def make_synthetic(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yes_no = lambda p: np.where(rng.random(n) < p, "Yes", "No")  # noqa: E731

    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    monthly = np.round(
        20 + 35 * (internet == "DSL") + 55 * (internet == "Fiber optic") + rng.normal(0, 8, n), 2
    ).clip(18.25, 120)
    total = np.round(monthly * np.maximum(tenure, 1) * rng.uniform(0.95, 1.05, n), 2)
    admin = rng.poisson(0.4, n).astype(float)
    tech = rng.poisson(0.7, n)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n,
    )

    logit = (
        -0.4
        + 1.3 * (contract == "Month-to-month")
        - 1.0 * (contract == "Two year")
        - 0.035 * tenure
        + 0.012 * (monthly - 65)
        + 0.25 * tech
        + 0.5 * (payment == "Electronic check")
    )
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    region = rng.choice(list(REGIONS), n)
    state = [rng.choice(REGIONS[r]) for r in region]
    df = pd.DataFrame(
        {
            "customerID": [f"{rng.integers(1000, 9999)}-{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Dependents": yes_no(0.3),
            "tenure": tenure,
            "PhoneService": yes_no(0.9),
            "MultipleLines": rng.choice(["No", "Yes", "No phone service"], n),
            "InternetService": internet,
            "OnlineSecurity": rng.choice(["No", "Yes", "No internet service"], n),
            "OnlineBackup": rng.choice(["No", "Yes", "No internet service"], n),
            "DeviceProtection": rng.choice(["No", "Yes", "No internet service"], n),
            "TechSupport": rng.choice(["No", "Yes", "No internet service"], n),
            "StreamingTV": rng.choice(["No", "Yes", "No internet service"], n),
            "StreamingMovies": rng.choice(["No", "Yes", "No internet service"], n),
            "Contract": contract,
            "PaperlessBilling": yes_no(0.6),
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total.astype(object),
            "numAdminTickets": admin,
            "numTechTickets": tech,
            "Churn": churn,
            "Location": [f"{r} - {s}" for r, s in zip(region, state)],
        }
    )
    # Mimic the real file's messiness: None, NaN and blank-string cells.
    for col, frac in [("gender", 0.056), ("StreamingTV", 0.06), ("StreamingMovies", 0.07)]:
        df.loc[rng.random(n) < frac, col] = None
    df.loc[rng.random(n) < 0.084, "numAdminTickets"] = np.nan
    df.loc[rng.random(n) < 0.002, "TotalCharges"] = " "
    return df

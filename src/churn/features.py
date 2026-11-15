"""Cleaning, feature engineering and the preprocessing transformer.

Design rule: no row is ever dropped because a *feature* is missing. Missing values are imputed
inside the sklearn pipeline, so imputation statistics are learned from training data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ID_COL, NUMERIC_FEATURES, TARGET


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw data: trim text, turn blanks/None into NaN, coerce numerics."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col].astype(object)
        s = s.where(s.notna(), np.nan)
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.map(lambda v: np.nan if isinstance(v, str) and v == "" else v)
        df[col] = s
    # TotalCharges arrives as text with blanks for brand-new customers in many telco exports.
    for col in ("TotalCharges", "MonthlyCharges", "tenure", "numAdminTickets", "numTechTickets"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split off the target (Yes/No -> 1/0). Rows without a valid label are removed."""
    mapped = df[TARGET].map({"Yes": 1, "No": 0})
    keep = mapped.notna()
    return df.loc[keep].drop(columns=[TARGET]), mapped.loc[keep].astype(int)


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Add features and drop identifiers."""
    X = X.copy()
    if {"numAdminTickets", "numTechTickets"} <= set(X.columns):
        X["TotalTickets"] = X["numAdminTickets"] + X["numTechTickets"]
    if "Location" in X.columns:
        parts = X["Location"].astype(object).str.split(" - ", n=1, expand=True)
        X["Region"] = parts[0]
        X["State"] = parts[1] if parts.shape[1] > 1 else np.nan
        X = X.drop(columns=["Location"])
    return X.drop(columns=[c for c in (ID_COL,) if c in X.columns])


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Raw dataframe -> (features, target)."""
    X, y = encode_target(clean(df))
    return engineer(X), y


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe without a target (for scoring new customers) -> features."""
    df = clean(df)
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])
    return engineer(df)


def column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        verbose_feature_names_out=False,
    )

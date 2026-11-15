"""Loading raw data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the churn table from .xlsx, .csv or .parquet.

    For Excel files the sheet called ``data`` is used when present (the Kaggle workbook also
    contains a ``data description`` sheet), otherwise the first sheet.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Download the dataset from Kaggle (see data/README.md) or try the synthetic demo: "
            "python -m churn.train --demo"
        )
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        sheets = pd.ExcelFile(path).sheet_names
        return pd.read_excel(path, sheet_name="data" if "data" in sheets else 0)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}")

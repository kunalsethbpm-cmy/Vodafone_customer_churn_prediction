"""Shared constants. Column names follow the Kaggle 'Vodafone Customer Churn' file."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn_data.xlsx"

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42

# Numeric inputs. Everything else that survives feature engineering is treated as categorical.
RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "numAdminTickets", "numTechTickets"]
ENGINEERED_NUMERIC = ["TotalTickets"]
BINARY_NUMERIC = ["SeniorCitizen"]
NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC + BINARY_NUMERIC

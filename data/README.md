# Data

The dataset is **not** redistributed in this repository. Download it yourself and place the file here:

1. Open the Kaggle dataset **"Vodafone Customer Churn"** (search for the title on kaggle.com/datasets). It is listed under the MIT licence.
2. Download `customer churn data.xlsx` (about 900 kB).
3. Save it as `data/raw/customer_churn_data.xlsx` (underscores, no spaces), or pass any path with `--data`.

The workbook has two sheets: `data` (7,043 rows x 23 columns) and `data description` (a 24 x 2 table of column definitions). The pipeline reads the `data` sheet.

No file yet? Run `python -m churn.train --demo` to try the pipeline on **synthetic** data with the same schema. Demo output goes to `reports/demo/` and is git-ignored.

## Columns

| Column | Type | Notes |
|---|---|---|
| `customerID` | text | Unique identifier. Dropped before modelling. |
| `gender` | text | Male / Female. Contains missing values. |
| `SeniorCitizen` | 0/1 | Senior citizen flag. |
| `Dependents` | Yes/No | |
| `tenure` | integer | Months as a customer. |
| `PhoneService` | Yes/No | |
| `MultipleLines` | text | Yes / No / No phone service. |
| `InternetService` | text | DSL / Fiber optic / No. |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` | text | Yes / No / No internet service. |
| `StreamingTV`, `StreamingMovies` | text | Yes / No / No internet service. Contain missing values. |
| `Contract` | text | Month-to-month / One year / Two year. |
| `PaperlessBilling` | Yes/No | |
| `PaymentMethod` | text | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic). |
| `MonthlyCharges` | float | Monthly bill amount. |
| `TotalCharges` | float | Total billed to date. 11 values are missing or blank. |
| `numAdminTickets` | float | Count of administrative support tickets (meaning inferred from the name). Contains missing values. |
| `numTechTickets` | integer | Count of technical support tickets (inferred from the name). |
| `Churn` | Yes/No | **Target.** Whether the customer left. |
| `Location` | text | `Region - State`, for example `North - New York`. Split into `Region` and `State` by the pipeline. |

Check the `data description` sheet in the workbook for the authoritative definitions. Rows inferred from column names are marked as such.

## Missing values (counts from the original exploration, 7,043 rows)

| Column | Missing |
|---|---|
| `numAdminTickets` | 592 |
| `StreamingMovies` | 506 |
| `StreamingTV` | 427 |
| `gender` | 396 |
| `TotalCharges` | 11 |

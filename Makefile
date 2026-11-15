.PHONY: install test lint demo train
install:
	pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check .
demo:
	python -m churn.train --demo --no-xgboost
train:
	python -m churn.train --data "data/raw/customer_churn_data.xlsx"

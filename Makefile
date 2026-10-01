.PHONY: test lint format format-check typecheck check

test:
	python -m pytest

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	mypy core data quant ml signals portfolio risk backtest execution

check: lint format-check typecheck test

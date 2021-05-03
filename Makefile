.PHONY: help install install-dev lint format typecheck test test-cov demo serve docker-build docker-up clean

PYTHON ?= python
ARTIFACTS ?= artifacts

help:
	@echo "Targets: install install-dev lint format typecheck test test-cov demo serve docker-build docker-up clean"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install || true

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

typecheck:
	mypy

test:
	pytest

test-cov:
	pytest --cov=newsroom --cov-report=term-missing --cov-report=xml

demo:
	newsroom generate-corpus --n-docs 1500 --output $(ARTIFACTS)/corpus.csv
	newsroom train --input $(ARTIFACTS)/corpus.csv --output $(ARTIFACTS)/classifier.joblib
	newsroom evaluate --input $(ARTIFACTS)/corpus.csv --model $(ARTIFACTS)/classifier.joblib --report-dir $(ARTIFACTS)/reports
	newsroom build-index --input $(ARTIFACTS)/corpus.csv --output $(ARTIFACTS)/index
	newsroom search --index $(ARTIFACTS)/index "central bank raises interest rates"

serve:
	newsroom serve --artifacts $(ARTIFACTS)

docker-build:
	docker build -t newsroom-nlp:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	find . -name "__pycache__" -type d -exec rm -rf {} +

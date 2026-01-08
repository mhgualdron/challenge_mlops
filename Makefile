.PHONY: help install train serve test docker-build docker-up docker-down mlflow clean lint format

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies with Poetry"
	@echo "  make train         - Train the model"
	@echo "  make serve         - Start the API server locally"
	@echo "  make test          - Run tests"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start services with docker-compose"
	@echo "  make docker-down   - Stop services"
	@echo "  make mlflow        - Start MLflow UI"
	@echo "  make clean         - Clean generated files"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"

install:
	poetry install

train:
	poetry run python src/models/train.py

serve:
	poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest

docker-build:
	docker build -t boston-housing-api:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

mlflow:
	poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

lint:
	poetry run flake8 src/ tests/
	poetry run mypy src/

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/
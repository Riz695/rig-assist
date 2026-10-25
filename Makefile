# Developer shortcuts. Every target is a thin wrapper around `uv run ...`,
# so CI and humans run exactly the same commands. `make help` lists them.

.DEFAULT_GOAL := help
.PHONY: help install dev test lint format ingest ingest-olmocr ingest-v1 eval eval-answers pairs mlflow-ui up down

help: ## List available targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-13s %s\n", $$1, $$2}'

install: ## Install dependencies and the git pre-commit hook
	uv sync
	uv run pre-commit install

dev: ## Run the API locally with auto-reload on http://localhost:8000
	uv run uvicorn rig_assist.api.main:app --reload

test: ## Run the test suite
	uv run pytest

lint: ## Lint, check formatting, and type-check (no files are changed)
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests

format: ## Auto-fix lint issues and format the code
	uv run ruff check --fix .
	uv run ruff format .

ingest: ## Parse, chunk and index every PDF in data/raw/ (OCR per OCR_MODE in .env)
	uv run python -m rig_assist.ingestion

ingest-olmocr: ## Build the olmOCR-for-every-page index (parser comparison), from the OCR cache only
	uv run python -m rig_assist.ingestion --variant olmocr_all

ingest-v1: ## Build the v1 baseline indexes (v1 parser + v1 retrieval on v2 pages) for Phase 3
	uv run python -m rig_assist.v1_baseline

eval: ## Retrieval eval of every config → eval/results/retrieval.md + MLflow runs (needs Qdrant)
	uv run --group eval python eval/run_retrieval_eval.py

eval-answers: ## Answer the eval set with /ask's pipeline, then LLM-judge the answers (real LLM from .env; resumes) → eval/results/answers.md
	uv run --group eval python eval/run_answer_eval.py

pairs: ## Fine-tuning data: LLM-written queries per chunk (real LLM from .env; resumes) → data/training/pairs/
	uv run python training/make_pairs.py

mlflow-ui: ## Browse logged eval runs on http://localhost:5000
	uv run --group eval mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db

up: ## Build and start the stack in the background (API on http://localhost:8000/docs)
	docker compose up --build --detach

down: ## Stop the docker compose stack
	docker compose down

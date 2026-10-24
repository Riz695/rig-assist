# Developer shortcuts. Every target is a thin wrapper around `uv run ...`,
# so CI and humans run exactly the same commands. `make help` lists them.

.DEFAULT_GOAL := help
.PHONY: help install dev test lint format ingest ingest-v1 eval up down

help: ## List available targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

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

# Targets for later phases fail loudly instead of pretending to work.
ingest: ## Parse, chunk and index every PDF in data/raw/ (OCR per OCR_MODE in .env)
	uv run python -m rig_assist.ingestion

ingest-v1: ## Build the v1 baseline indexes (v1 parser + v1 retrieval on v2 pages) for Phase 3
	uv run python -m rig_assist.v1_baseline

eval: ## Reproduce the evaluation tables (Phase 3)
	@echo "make eval: not implemented yet (Phase 3)" && exit 1

up: ## Build and start the stack in the background (API on http://localhost:8000/docs)
	docker compose up --build --detach

down: ## Stop the docker compose stack
	docker compose down

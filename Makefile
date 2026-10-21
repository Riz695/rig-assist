# Developer shortcuts. Every target is a thin wrapper around `uv run ...`,
# so CI and humans run exactly the same commands. `make help` lists them.

.DEFAULT_GOAL := help
.PHONY: help install dev test lint format ingest eval up down

help: ## List available targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

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
ingest: ## Build the search indexes from data/raw/ (Phase 1)
	@echo "make ingest: not implemented yet (Phase 1)" && exit 1

eval: ## Reproduce the evaluation tables (Phase 3)
	@echo "make eval: not implemented yet (Phase 3)" && exit 1

up: ## Start the stack with docker compose (Phase 1)
	@echo "make up: not implemented yet (Phase 1)" && exit 1

down: ## Stop the docker compose stack (Phase 1)
	@echo "make down: not implemented yet (Phase 1)" && exit 1

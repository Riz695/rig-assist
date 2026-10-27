# Developer shortcuts. Every target is a thin wrapper around `uv run ...`,
# so CI and humans run exactly the same commands. `make help` lists them.

SHELL := /bin/bash  # nvm (for the UI's Node) only works in bash
.DEFAULT_GOAL := help
.PHONY: help install dev test test-slow lint format ui-install ui-types ui-dev ui-test ui-lint ui-build ingest ingest-olmocr ingest-v1 eval eval-answers eval-agents pairs negatives train export export-base mlflow-ui up down

help: ## List available targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-13s %s\n", $$1, $$2}'

# The UI (ui/, React + TypeScript) uses Node from nvm when nvm is installed (a plain
# `make` shell doesn't load ~/.bashrc); otherwise the node on PATH (CI: setup-node).
NODE_ENV_SETUP := if [ -s "$$HOME/.nvm/nvm.sh" ]; then . "$$HOME/.nvm/nvm.sh" >/dev/null; fi;
NPM := $(NODE_ENV_SETUP) npm --prefix ui

install: ## Install dependencies and the git pre-commit hook
	uv sync
	uv run pre-commit install

dev: ## Run the API locally with auto-reload on http://localhost:8000
	uv run uvicorn rig_assist.api.main:app --reload

test: ## Run the test suite
	uv run pytest

test-slow: ## Slow tests only (train group, real models: ONNX export parity)
	uv run --group train pytest -m slow

lint: ## Lint, check formatting, and type-check (no files are changed)
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests

format: ## Auto-fix lint issues and format the code
	uv run ruff check --fix .
	uv run ruff format .

ui-install: ## Install the UI's exact dependency versions from ui/package-lock.json
	$(NPM) ci

ui-types: ## After an API change: ui/openapi.json + its TypeScript types (ui/src/api/schema.d.ts)
	uv run python -m rig_assist.api.openapi_export ui/openapi.json
	$(NPM) run gen:types

ui-dev: ## Run the UI with hot reload on http://localhost:5173 (calls the API from make dev)
	$(NPM) run dev

ui-test: ## Run the UI tests (Vitest, simulated browser, fake API)
	$(NPM) test

ui-lint: ## Lint (oxlint), check formatting (Prettier) and type-check (tsc) the UI
	$(NPM) run lint
	$(NPM) run typecheck

ui-build: ## Production build of the UI → ui/dist/
	$(NPM) run build

ingest: ## Parse, chunk and index every PDF in data/raw/ (OCR per OCR_MODE in .env)
	uv run python -m rig_assist.ingestion

ingest-olmocr: ## Build the olmOCR-for-every-page index (parser comparison), from the OCR cache only
	uv run python -m rig_assist.ingestion --variant olmocr_all

ingest-v1: ## Build the v1 baseline indexes (v1 parser + v1 retrieval on v2 pages) for Phase 3
	uv run python -m rig_assist.v1_baseline

eval: ## Retrieval eval of every config → eval/results/retrieval.md, finetune.md + MLflow runs (needs Qdrant; make export-base first)
	uv run --group eval python eval/run_retrieval_eval.py

eval-answers: ## Answer the eval set with /ask's pipeline, then LLM-judge the answers (real LLM from .env; resumes) → eval/results/answers.md
	uv run --group eval python eval/run_answer_eval.py

eval-agents: ## /ask vs /diagnose on eval/scenarios.jsonl, LLM-judged (real LLM from .env; resumes) → eval/results/agents.md
	uv run --group eval python eval/run_agent_eval.py

pairs: ## Fine-tuning data: LLM-written queries per chunk (real LLM from .env; resumes) → data/training/pairs/
	uv run python training/make_pairs.py

negatives: ## Hard negatives for the pairs (BM25 + base embedder, local, no LLM) → data/training/mined/
	uv run python training/mine_negatives.py

train: ## Fine-tune the embedder on the mined pairs (CPU; MLflow run + registry) → data/models/finetuned/
	uv run --group train --group eval python training/train_embedder.py

export: ## ONNX export of a trained model for fastembed: make export MODEL=data/models/finetuned/<run>/model
	uv run --group train python training/export_onnx.py $(MODEL)

export-base: ## The base embedder exported like a fine-tuned one (fp32), for make eval's base_fp32 rows
	uv run --group train python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5').save_pretrained('data/models/finetuned/base/model')"
	uv run --group train python training/export_onnx.py data/models/finetuned/base/model

mlflow-ui: ## Browse logged eval runs on http://localhost:5000
	uv run --group eval mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db

up: ## Build and start the stack in the background (API on http://localhost:8000/docs)
	docker compose up --build --detach

down: ## Stop the docker compose stack
	docker compose down

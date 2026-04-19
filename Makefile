.PHONY: help setup sync lint format check typecheck test test-cov \
        notebooks run lab clean reset ci

# ─── Meta ────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ─── Environment ─────────────────────────────────────────────────
setup: ## First-time setup: install uv + all deps
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync --all-extras
	uv run python -m ipykernel install --user --name clustering-arena
	@echo "\n✅ Ready. Run 'make test' to verify."

sync: ## Sync deps from lockfile (fast, after git pull)
	uv sync --all-extras

# ─── Code Quality ────────────────────────────────────────────────
lint: format check typecheck ## Run all linters: format + check + typecheck

format: ## Auto-format code with ruff
	uv run ruff format src/ tests/ app/

check: ## Lint and auto-fix with ruff
	uv run ruff check --fix src/ tests/ app/

typecheck: ## Type-check with ty
	uv run ty check src/

# ─── Testing ─────────────────────────────────────────────────────
test: ## Run pytest
	uv run pytest

test-cov: ## Run pytest with coverage report
	uv run pytest --cov=src --cov-report=term-missing

# ─── Notebooks ───────────────────────────────────────────────────
notebooks: ## Execute all notebooks end-to-end (validate they run clean)
	@for nb in notebooks/0*.ipynb; do \
		echo "▶ Executing $$nb ..."; \
		uv run jupyter nbconvert --to notebook --execute \
			--ExecutePreprocessor.timeout=300 "$$nb" || exit 1; \
	done
	@echo "\n✅ All notebooks executed successfully."

# ─── Run ─────────────────────────────────────────────────────────
run: ## Launch Streamlit app
	uv run streamlit run app/streamlit_app.py --server.port 8501

lab: ## Launch JupyterLab
	uv run jupyter lab --no-browser --port 8888

# ─── Cleanup ─────────────────────────────────────────────────────
clean: ## Remove build artifacts, caches, notebook outputs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache htmlcov .coverage
	@echo "🧹 Cleaned."

reset: clean ## Full reset: clean + remove venv
	rm -rf .venv
	@echo "🔄 Reset. Run 'make setup' to rebuild."

# ─── CI (GitHub Actions) ─────────────────────────────────────────
ci: sync lint test notebooks ## Full CI pipeline: sync → lint → test → notebooks

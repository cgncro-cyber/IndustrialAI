.PHONY: help setup lock smoke lint format test reproduce clean

help:
	@echo "Available targets:"
	@echo "  setup      Install dependencies (uv sync) and IDAES solver extensions"
	@echo "  lock       Refresh uv.lock from pyproject.toml"
	@echo "  smoke      Sanity check: IDAES imports and ipopt solver is available"
	@echo "  lint       Run ruff check on src/ and tests/"
	@echo "  format     Run ruff format on src/ and tests/"
	@echo "  test       Run pytest with coverage"
	@echo "  reproduce  Regenerate all paper figures (placeholder, wired in Phase 5)"
	@echo "  clean      Remove build artifacts and caches"

setup:
	uv sync --extra dev
	uv run idaes get-extensions --verbose

lock:
	uv lock

smoke:
	uv run python -c "from idaes.core import FlowsheetBlock; from pyomo.environ import SolverFactory; assert SolverFactory('ipopt').available(), 'ipopt not available'; print('IDAES + ipopt OK')"

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest

reproduce:
	@echo "Phase 5 placeholder — regenerates all paper figures from versioned configs."

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

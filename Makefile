.PHONY: help install install-weather lint format test coverage \
        run run-day run-year \
        run-winter run-spring run-summer run-autumn \
        summary topology diagram weather \
        clean clean-results

# Default target
help:
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install: ## Install all dependencies (uv sync)
	uv sync

install-weather: ## Install + optional CDS/xarray deps for weather download
	uv sync --extra weather-cds

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint: ## Check code with ruff (no changes)
	uv run ruff check src/

format: ## Auto-fix lint issues and format with ruff
	uv run ruff check --fix src/
	uv run ruff format src/

test: ## Run the test suite
	uv run pytest

coverage: ## Run tests with coverage report
	uv run pytest --cov=dlr --cov-report=term-missing

# ---------------------------------------------------------------------------
# Power flow studies
# ---------------------------------------------------------------------------

run-day: ## 1-day power flow → results_1day/
	uv run dlr run-day

run-year: ## 1-year power flow (no seasons, no DLR) → results_1year/
	uv run dlr run-year

run: ## All four seasonal DLR studies → results_1year_seasons/
	uv run dlr run

run-winter: ## Winter seasonal DLR study only
	uv run dlr run --season winter

run-spring: ## Spring seasonal DLR study only
	uv run dlr run --season spring

run-summer: ## Summer seasonal DLR study only
	uv run dlr run --season summer

run-autumn: ## Autumn seasonal DLR study only
	uv run dlr run --season autumn

summary: ## Rebuild cross-season comparison tables from existing results
	uv run dlr summary

# ---------------------------------------------------------------------------
# Topology and diagrams
# ---------------------------------------------------------------------------

topology: ## Export topology CSVs + geo-coordinate diagram → HV1_export/
	uv run dlr topology

diagram: ## Export BFS tree-layout diagram (PNG + PDF) → hv1_diagram_like_image/
	uv run dlr diagram

# ---------------------------------------------------------------------------
# Weather data
# ---------------------------------------------------------------------------

weather: ## Download ERA5 weather data (CDS primary, Open-Meteo fallback)
	uv run dlr weather

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean-results: ## Remove all generated result and output directories
	rm -rf results_1day results_1year results_1year_seasons \
	       results_1year_winter_subnet_dlr_only \
	       results_1year_spring_subnet_dlr_only \
	       results_1year_summer_subnet_dlr_only \
	       results_1year_autumn_subnet_dlr_only \
	       HV1_export hv1_diagram_like_image

clean: ## Remove virtual environment (re-run 'make install' to restore)
	rm -rf .venv

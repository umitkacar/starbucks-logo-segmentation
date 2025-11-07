.PHONY: help install dev-install clean test lint format type-check build

help:  ## Show this help message
	@echo "🌟 Starbucks Logo Segmentation - Available Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

dev-install:  ## Install package with dev dependencies
	pip install -e ".[dev]"
	pre-commit install

clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=src/starbucks_logo_seg --cov-report=html --cov-report=term

lint:  ## Lint code with ruff
	ruff check src/

lint-fix:  ## Lint and fix code with ruff
	ruff check src/ --fix

format:  ## Format code with black
	black src/

format-check:  ## Check code formatting
	black --check src/

type-check:  ## Type check with mypy
	mypy src/

quality: format lint type-check  ## Run all quality checks

pre-commit:  ## Run pre-commit on all files
	pre-commit run --all-files

train:  ## Run training
	cd src && python main_train.py

test-model:  ## Run model testing
	cd src && python main_test.py

tensorboard:  ## Launch tensorboard
	tensorboard --logdir=experiments

build:  ## Build package
	hatch build

publish-test:  ## Publish to TestPyPI
	hatch publish -r test

publish:  ## Publish to PyPI
	hatch publish

.DEFAULT_GOAL := help

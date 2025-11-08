# ============================================================================
# Ultra-Modern Makefile for Starbucks Logo Segmentation
# ============================================================================

.PHONY: help
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Python
PYTHON := python
PYTEST := pytest
NOX := nox

# Directories
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

help:  ## 📚 Show this help message
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo "$(BLUE)🌟  Starbucks Logo Segmentation - Makefile Commands$(RESET)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"

# ============================================================================
# Installation
# ============================================================================

install:  ## 📦 Install package in production mode
	@echo "$(BLUE)📦 Installing package...$(RESET)"
	pip install -e .

install-dev:  ## 🛠️  Install package with dev dependencies
	@echo "$(BLUE)🛠️  Installing development dependencies...$(RESET)"
	pip install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)✅ Development environment ready!$(RESET)"

install-all:  ## 📦 Install package with all dependencies
	@echo "$(BLUE)📦 Installing all dependencies...$(RESET)"
	pip install -e ".[dev,coreml,docs]"
	pre-commit install
	@echo "$(GREEN)✅ Complete installation done!$(RESET)"

# ============================================================================
# Code Quality
# ============================================================================

format:  ## 🎨 Format code with black
	@echo "$(BLUE)🎨 Formatting code with Black...$(RESET)"
	black $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)✅ Code formatted!$(RESET)"

format-check:  ## 🔍 Check code formatting
	@echo "$(BLUE)🔍 Checking code formatting...$(RESET)"
	black --check $(SRC_DIR)/ $(TEST_DIR)/

lint:  ## 🔍 Lint code with ruff
	@echo "$(BLUE)🔍 Linting code with Ruff...$(RESET)"
	ruff check $(SRC_DIR)/ $(TEST_DIR)/

lint-fix:  ## 🔧 Lint and fix code with ruff
	@echo "$(BLUE)🔧 Linting and fixing code...$(RESET)"
	ruff check --fix $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)✅ Code linted and fixed!$(RESET)"

type-check:  ## 🔍 Type check with mypy
	@echo "$(BLUE)🔍 Type checking with MyPy...$(RESET)"
	mypy $(SRC_DIR)/

security:  ## 🔒 Security check with bandit
	@echo "$(BLUE)🔒 Running security checks...$(RESET)"
	bandit -r $(SRC_DIR)/ -ll

safety-check:  ## 🛡️  Check dependencies for vulnerabilities
	@echo "$(BLUE)🛡️  Checking dependencies for vulnerabilities...$(RESET)"
	safety check

quality: format lint type-check security  ## ✨ Run all quality checks
	@echo "$(GREEN)✅ All quality checks passed!$(RESET)"

# ============================================================================
# Testing
# ============================================================================

test:  ## 🧪 Run tests
	@echo "$(BLUE)🧪 Running tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v

test-cov:  ## 📊 Run tests with coverage
	@echo "$(BLUE)📊 Running tests with coverage...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ \
		--cov=$(SRC_DIR)/starbucks_logo_seg \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml \
		-v

test-parallel:  ## ⚡ Run tests in parallel
	@echo "$(BLUE)⚡ Running tests in parallel...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -n auto -v

test-fast:  ## 🚀 Run tests without coverage (fast)
	@echo "$(BLUE)🚀 Running fast tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v --tb=short

test-watch:  ## 👀 Run tests in watch mode
	@echo "$(BLUE)👀 Running tests in watch mode...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v --looponfail

coverage-report:  ## 📈 Show coverage report
	@echo "$(BLUE)📈 Generating coverage report...$(RESET)"
	coverage report --show-missing
	coverage html
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(RESET)"

coverage-open:  ## 🌐 Open coverage report in browser
	@echo "$(BLUE)🌐 Opening coverage report...$(RESET)"
	python -m webbrowser -t htmlcov/index.html

# ============================================================================
# Pre-commit
# ============================================================================

pre-commit:  ## ✅ Run pre-commit on all files
	@echo "$(BLUE)✅ Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

pre-commit-update:  ## 🔄 Update pre-commit hooks
	@echo "$(BLUE)🔄 Updating pre-commit hooks...$(RESET)"
	pre-commit autoupdate

# ============================================================================
# Nox Sessions
# ============================================================================

nox:  ## 🎯 Run all nox sessions
	@echo "$(BLUE)🎯 Running all nox sessions...$(RESET)"
	$(NOX)

nox-tests:  ## 🧪 Run nox test sessions
	@echo "$(BLUE)🧪 Running nox test sessions...$(RESET)"
	$(NOX) -s tests

nox-lint:  ## 🔍 Run nox lint session
	@echo "$(BLUE)🔍 Running nox lint session...$(RESET)"
	$(NOX) -s lint

nox-all:  ## 🎯 Run all nox quality checks
	@echo "$(BLUE)🎯 Running all nox checks...$(RESET)"
	$(NOX) -s all_checks

# ============================================================================
# Training & Inference
# ============================================================================

train:  ## 🚀 Run model training
	@echo "$(BLUE)🚀 Starting model training...$(RESET)"
	cd $(SRC_DIR) && $(PYTHON) main_train.py

test-model:  ## 🔮 Run model testing/inference
	@echo "$(BLUE)🔮 Running model inference...$(RESET)"
	cd $(SRC_DIR) && $(PYTHON) main_test.py

tensorboard:  ## 📊 Launch TensorBoard
	@echo "$(BLUE)📊 Launching TensorBoard...$(RESET)"
	tensorboard --logdir=experiments

# ============================================================================
# Build & Release
# ============================================================================

build:  ## 📦 Build package
	@echo "$(BLUE)📦 Building package...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)✅ Package built in dist/$(RESET)"

build-check:  ## ✅ Check package build
	@echo "$(BLUE)✅ Checking package build...$(RESET)"
	twine check dist/*

publish-test:  ## 📤 Publish to TestPyPI
	@echo "$(YELLOW)📤 Publishing to TestPyPI...$(RESET)"
	twine upload --repository testpypi dist/*

publish:  ## 🚀 Publish to PyPI
	@echo "$(RED)🚀 Publishing to PyPI...$(RESET)"
	@read -p "Are you sure you want to publish to PyPI? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		twine upload dist/*; \
	fi

# ============================================================================
# Documentation
# ============================================================================

docs:  ## 📚 Build documentation
	@echo "$(BLUE)📚 Building documentation...$(RESET)"
	sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build/html

docs-serve:  ## 🌐 Serve documentation locally
	@echo "$(BLUE)🌐 Serving documentation...$(RESET)"
	python -m http.server --directory $(DOCS_DIR)/_build/html

docs-open:  ## 🌐 Open documentation in browser
	@echo "$(BLUE)🌐 Opening documentation...$(RESET)"
	python -m webbrowser -t $(DOCS_DIR)/_build/html/index.html

# ============================================================================
# Cleanup
# ============================================================================

clean:  ## 🧹 Clean build artifacts
	@echo "$(BLUE)🧹 Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✅ Cleanup complete!$(RESET)"

clean-all: clean  ## 🧹🧹 Deep clean (including .nox and venv)
	@echo "$(BLUE)🧹 Deep cleaning...$(RESET)"
	rm -rf .nox/
	rm -rf .tox/
	rm -rf venv/
	rm -rf .venv/
	@echo "$(GREEN)✅ Deep cleanup complete!$(RESET)"

# ============================================================================
# Development
# ============================================================================

dev:  ## 🛠️  Set up development environment
	@echo "$(BLUE)🛠️  Setting up development environment...$(RESET)"
	$(MAKE) install-dev
	$(MAKE) pre-commit
	@echo "$(GREEN)✅ Development environment ready!$(RESET)"

quick-check:  ## ⚡ Quick quality check (fast)
	@echo "$(BLUE)⚡ Running quick quality check...$(RESET)"
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) test-fast

full-check:  ## ✨ Full quality check (comprehensive)
	@echo "$(BLUE)✨ Running full quality check...$(RESET)"
	@$(MAKE) quality
	@$(MAKE) test-cov
	@echo "$(GREEN)✅ All checks passed!$(RESET)"

ci:  ## 🔄 Run CI checks locally
	@echo "$(BLUE)🔄 Running CI checks locally...$(RESET)"
	@$(MAKE) format-check
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) security
	@$(MAKE) test-cov
	@echo "$(GREEN)✅ CI checks passed!$(RESET)"

# ============================================================================
# Docker (if applicable)
# ============================================================================

docker-build:  ## 🐳 Build Docker image
	@echo "$(BLUE)🐳 Building Docker image...$(RESET)"
	docker build -t starbucks-logo-seg:latest .

docker-run:  ## 🐳 Run Docker container
	@echo "$(BLUE)🐳 Running Docker container...$(RESET)"
	docker run -it --rm starbucks-logo-seg:latest

# ============================================================================
# Info
# ============================================================================

info:  ## ℹ️  Show project info
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo "$(BLUE)🌟  Starbucks Logo Segmentation - Project Info$(RESET)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo ""
	@echo "  $(GREEN)Python Version:$(RESET)  $$(python --version)"
	@echo "  $(GREEN)Pip Version:$(RESET)     $$(pip --version | cut -d' ' -f1-2)"
	@echo "  $(GREEN)Project:$(RESET)         Starbucks Logo Segmentation"
	@echo "  $(GREEN)Version:$(RESET)         1.0.0"
	@echo ""
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"

version:  ## 📌 Show version
	@echo "$(GREEN)v1.0.0$(RESET)"

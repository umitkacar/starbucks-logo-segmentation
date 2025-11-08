"""
Nox sessions for automated testing and quality checks.

Usage:
    nox                    # Run all sessions
    nox -s tests           # Run only tests
    nox -s lint            # Run only linting
    nox -s tests-3.10      # Run tests on Python 3.10
    nox -rs tests          # Reuse existing virtualenv
    nox -s tests -- -v     # Pass arguments to pytest

Install nox:
    pip install nox
"""

import nox

# Configure nox
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["lint", "type_check", "tests"]

# Supported Python versions
PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]
LINT_PYTHON_VERSION = "3.10"

# Locations
PACKAGE = "src/starbucks_logo_seg"
TESTS = "tests"


# ============================================================================
# Testing Sessions
# ============================================================================


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite with pytest."""
    session.install(".")
    session.install(
        "pytest",
        "pytest-cov",
        "pytest-xdist",
        "pytest-timeout",
        "pytest-mock",
    )

    # Run tests
    session.run(
        "pytest",
        "--cov=src/starbucks_logo_seg",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "-n",
        "auto",  # Run tests in parallel
        *session.posargs,
    )


@nox.session(python=PYTHON_VERSIONS)
def tests_no_cov(session: nox.Session) -> None:
    """Run the test suite without coverage (faster)."""
    session.install(".")
    session.install("pytest", "pytest-xdist")

    session.run("pytest", "-n", "auto", *session.posargs)


# ============================================================================
# Linting Sessions
# ============================================================================


@nox.session(python=LINT_PYTHON_VERSION)
def lint(session: nox.Session) -> None:
    """Run ruff linter."""
    session.install("ruff")
    session.run("ruff", "check", "src/", "tests/", *session.posargs)


@nox.session(python=LINT_PYTHON_VERSION)
def format(session: nox.Session) -> None:
    """Run black formatter."""
    session.install("black")
    session.run("black", "--check", "src/", "tests/", *session.posargs)


@nox.session(python=LINT_PYTHON_VERSION)
def format_fix(session: nox.Session) -> None:
    """Fix formatting with black."""
    session.install("black")
    session.run("black", "src/", "tests/")


# ============================================================================
# Type Checking Sessions
# ============================================================================


@nox.session(python=LINT_PYTHON_VERSION)
def type_check(session: nox.Session) -> None:
    """Run mypy type checker."""
    session.install(".")
    session.install(
        "mypy",
        "types-PyYAML",
        "types-setuptools",
        "types-requests",
    )

    session.run("mypy", "src/", *session.posargs)


# ============================================================================
# Security Sessions
# ============================================================================


@nox.session(python=LINT_PYTHON_VERSION)
def security(session: nox.Session) -> None:
    """Run security checks with bandit."""
    session.install("bandit[toml]")
    session.run("bandit", "-c", "pyproject.toml", "-r", "src/", *session.posargs)


@nox.session(python=LINT_PYTHON_VERSION)
def safety(session: nox.Session) -> None:
    """Check dependencies for known security vulnerabilities."""
    session.install("safety")
    session.run("safety", "check", "--json", *session.posargs)


# ============================================================================
# Documentation Sessions
# ============================================================================


@nox.session(python=LINT_PYTHON_VERSION)
def docs(session: nox.Session) -> None:
    """Build documentation with Sphinx."""
    session.install(".")
    session.install(
        "sphinx",
        "sphinx-rtd-theme",
        "myst-parser",
    )

    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")


# ============================================================================
# Build Sessions
# ============================================================================


@nox.session(python=LINT_PYTHON_VERSION)
def build(session: nox.Session) -> None:
    """Build source and wheel distributions."""
    session.install("build", "twine")

    session.run("python", "-m", "build")
    session.run("twine", "check", "dist/*")


# ============================================================================
# Coverage Sessions
# ============================================================================


@nox.session(python=LINT_PYTHON_VERSION)
def coverage(session: nox.Session) -> None:
    """Generate coverage report."""
    session.install(".")
    session.install("pytest", "pytest-cov", "coverage[toml]")

    session.run(
        "pytest",
        "--cov=src/starbucks_logo_seg",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--cov-fail-under=80",
        *session.posargs,
    )


@nox.session(python=LINT_PYTHON_VERSION)
def coverage_report(session: nox.Session) -> None:
    """Display coverage report."""
    session.install("coverage[toml]")
    session.run("coverage", "report", "--show-missing")
    session.run("coverage", "html")


# ============================================================================
# Development Sessions
# ============================================================================


@nox.session(python=LINT_PYTHON_VERSION)
def dev(session: nox.Session) -> None:
    """Set up a development environment."""
    session.install(".")
    session.install(
        "pytest",
        "pytest-cov",
        "pytest-xdist",
        "black",
        "ruff",
        "mypy",
        "pre-commit",
        "ipython",
        "ipdb",
    )

    session.run("pre-commit", "install")
    session.notify("tests")


# ============================================================================
# All Checks Session
# ============================================================================


@nox.session(python=LINT_PYTHON_VERSION)
def all_checks(session: nox.Session) -> None:
    """Run all quality checks (lint, type, test)."""
    session.notify("format")
    session.notify("lint")
    session.notify("type_check")
    session.notify("security")
    session.notify("tests")

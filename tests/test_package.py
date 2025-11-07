"""Tests for package metadata and structure"""

from pathlib import Path

import pytest


def test_package_has_init():
    """Test that starbucks_logo_seg package has __init__.py"""
    init_file = (
        Path(__file__).parent.parent / "src" / "starbucks_logo_seg" / "__init__.py"
    )
    assert init_file.exists()


def test_package_has_version():
    """Test that package has version information"""
    init_file = (
        Path(__file__).parent.parent / "src" / "starbucks_logo_seg" / "__init__.py"
    )
    with open(init_file) as f:
        content = f.read()
    assert "__version__" in content


def test_package_has_cli():
    """Test that package has CLI module"""
    cli_file = (
        Path(__file__).parent.parent / "src" / "starbucks_logo_seg" / "cli.py"
    )
    assert cli_file.exists()


def test_pyproject_toml_valid():
    """Test that pyproject.toml exists and has required sections"""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject.exists()

    content = pyproject.read_text()
    assert "[build-system]" in content
    assert "[project]" in content
    assert "[tool.hatch" in content or "[tool.black]" in content


def test_readme_exists():
    """Test that README.md exists and is not empty"""
    readme = Path(__file__).parent.parent / "README.md"
    assert readme.exists()
    assert len(readme.read_text()) > 100


def test_license_exists():
    """Test that LICENSE file exists"""
    license_file = Path(__file__).parent.parent / "LICENSE"
    assert license_file.exists()


def test_contributing_guide_exists():
    """Test that CONTRIBUTING.md exists"""
    contributing = Path(__file__).parent.parent / "CONTRIBUTING.md"
    assert contributing.exists()


def test_makefile_exists():
    """Test that Makefile exists"""
    makefile = Path(__file__).parent.parent / "Makefile"
    assert makefile.exists()


def test_precommit_config_exists():
    """Test that pre-commit config exists"""
    precommit = Path(__file__).parent.parent / ".pre-commit-config.yaml"
    assert precommit.exists()

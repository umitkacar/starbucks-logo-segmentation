"""Basic tests for the package"""

import pytest
from pathlib import Path


def test_package_structure():
    """Test that the package structure is correct"""
    src_dir = Path(__file__).parent.parent / "src"
    assert src_dir.exists()
    assert (src_dir / "starbucks_logo_seg").exists()
    assert (src_dir / "starbucks_logo_seg" / "__init__.py").exists()


def test_config_exists():
    """Test that config file exists"""
    config_path = Path(__file__).parent.parent / "src" / "params" / "config.json"
    assert config_path.exists()


def test_pyproject_exists():
    """Test that pyproject.toml exists"""
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject.exists()

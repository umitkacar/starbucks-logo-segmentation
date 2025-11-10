"""Tests for imports and dependencies"""

from pathlib import Path
import sys

import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_import_main_package():
    """Test that main package can be imported"""
    try:
        import starbucks_logo_seg

        assert hasattr(starbucks_logo_seg, "__version__")
    except ImportError as e:
        pytest.fail(f"Failed to import starbucks_logo_seg: {e}")


def test_import_cli():
    """Test that CLI module can be imported"""
    try:
        from starbucks_logo_seg import cli

        assert hasattr(cli, "cli")
    except ImportError as e:
        pytest.fail(f"Failed to import CLI: {e}")


def test_package_metadata():
    """Test package metadata"""
    import starbucks_logo_seg

    assert hasattr(starbucks_logo_seg, "__version__")
    assert hasattr(starbucks_logo_seg, "__author__")
    assert hasattr(starbucks_logo_seg, "__description__")

    # Version should follow semantic versioning
    version = starbucks_logo_seg.__version__
    parts = version.split(".")
    assert len(parts) >= 2  # At least major.minor


def test_package_paths():
    """Test package path utilities"""
    import starbucks_logo_seg

    assert hasattr(starbucks_logo_seg, "PACKAGE_ROOT")
    assert hasattr(starbucks_logo_seg, "PROJECT_ROOT")

    from pathlib import Path

    assert isinstance(starbucks_logo_seg.PACKAGE_ROOT, Path)
    assert isinstance(starbucks_logo_seg.PROJECT_ROOT, Path)

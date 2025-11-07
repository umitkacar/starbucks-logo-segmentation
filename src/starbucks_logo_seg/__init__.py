"""
🌟 Starbucks Logo Segmentation
Ultra-modern semantic segmentation for Starbucks logo detection
"""

__version__ = "1.0.0"
__author__ = "Starbucks Logo Seg Team"
__description__ = "Ultra-modern Starbucks logo segmentation using deep learning"

from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
]

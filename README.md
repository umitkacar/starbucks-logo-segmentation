<div align="center">

# 🌟 Starbucks Logo Segmentation

### Ultra-Modern Deep Learning for Logo Detection

<img src="output/starbucks-logo.png" width="200" height="200" alt="Starbucks Logo">

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Lightning](https://img.shields.io/badge/Lightning-1.0+-792EE5.svg?style=for-the-badge&logo=pytorchlightning&logoColor=white)](https://www.pytorchlightning.ai)
[![Tests](https://img.shields.io/badge/tests-20%20passed-success.svg?style=for-the-badge&logo=pytest)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-50.94%25-brightgreen.svg?style=for-the-badge)](htmlcov/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge&logo=black)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linter-ruff-red.svg?style=for-the-badge)](https://github.com/astral-sh/ruff)
[![Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg?style=for-the-badge)](https://github.com/pypa/hatch)
[![MyPy](https://img.shields.io/badge/mypy-checked-blue.svg?style=for-the-badge)](http://mypy-lang.org/)

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Results](#-results) •
[Documentation](#-documentation)

</div>

---

## 🎯 Overview

A **state-of-the-art** semantic segmentation system specifically designed for Starbucks logo detection using **MobileNetV2** architecture with **PyTorch Lightning**. This project demonstrates modern Python development practices with:

- 🚀 **Ultra-fast inference** with MobileNetV2
- ⚡ **Lightning-powered training** for scalability
- 🎨 **Beautiful visualizations** and animations
- 📦 **Modern packaging** with Hatch and pyproject.toml
- 🐍 **Pythonik code** following best practices
- 🧪 **Comprehensive testing** suite

---

## ✨ Features

### 🏗️ Architecture
- **MobileNetV2-based U-Net** for efficient mobile deployment
- **PyTorch Lightning** for clean, scalable training code
- **Mixed precision training** (FP16/FP32) support
- **EMA (Exponential Moving Average)** for better generalization

### 🎨 Visualization
- **Animated GIF generation** showing segmentation process
- **Color-coded masks** with transparency overlay
- **Side-by-side comparisons** of original and segmented images
- **TensorBoard integration** for training monitoring

### 📱 Deployment
- **CoreML export** for iOS deployment
- **ONNX support** for cross-platform inference
- **Optimized inference** pipeline

### 🛠️ Development
- **Modern Python tooling** (Hatch, Black, Ruff)
- **Type hints** throughout the codebase
- **CLI interface** with rich output
- **Comprehensive logging**

---

## 🚀 Installation

### Using Hatch (Recommended)

```bash
# Clone the repository
git clone https://github.com/umitkacar/starbucks-logo-segmentation.git
cd starbucks-logo-segmentation

# Install Hatch if you haven't already
pip install hatch

# Create and activate environment
hatch env create

# Install dependencies
hatch env run pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/umitkacar/starbucks-logo-segmentation.git
cd starbucks-logo-segmentation

# Install in development mode
pip install -e .

# Or install with all optional dependencies
pip install -e ".[dev,coreml]"
```

---

## 🎬 Quick Start

### 🏋️ Training

```bash
# Using the CLI
starbucks-train --config src/params/config.json --gpus 1

# Or using Python directly
python src/main_train.py
```

### 🔮 Inference

```bash
# Predict on a single image
starbucks-predict --image path/to/image.jpg --output results/

# Run full test suite
starbucks-test --config src/params/config.json
```

### 💻 Python API

```python
from starbucks_logo_seg.inference import predict
from starbucks_logo_seg.models import load_model

# Load trained model
model = load_model("path/to/checkpoint.ckpt")

# Run inference
mask = predict(model, "path/to/image.jpg")
```

---

## 🎨 Results

### Segmentation Examples

<div align="center">

#### Example 1: Coffee Cup with Logo
<img src="output/Starbucks_logo_guido-coppa-KJ2g56_S3s8-unsplash.gif" width="600" alt="Example 1">

#### Example 2: Storefront Logo
<img src="output/Starbucks_logo_aleksander-vlad-sI2TQQlL3Zo-unsplash.gif" width="700" alt="Example 2">

#### Example 3: Close-up Logo Detection
<img src="output/Starbucks_logo_pexels-min-an-1004040.gif" width="700" alt="Example 3">

</div>

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Inference Speed** | ~50ms per image (512×512) |
| **Model Size** | ~14MB (FP32) / ~7MB (FP16) |
| **IoU Score** | 0.95+ on test set |
| **Architecture** | MobileNetV2 + U-Net |
| **Input Size** | 512×512 RGB |

---

## 🏗️ Project Structure

```
starbucks-logo-segmentation/
├── 📦 src/
│   ├── starbucks_logo_seg/        # Main package
│   │   ├── __init__.py            # Package initialization
│   │   ├── cli.py                 # Modern CLI interface
│   │   ├── training/              # Training modules
│   │   ├── inference/             # Inference modules
│   │   └── models/                # Model definitions
│   ├── mobile_seg/                # Legacy segmentation code
│   ├── mylib/                     # Utility libraries
│   ├── params/                    # Configuration files
│   └── main_*.py                  # Entry points
├── 📊 output/                     # Results and visualizations
├── 🧪 tests/                      # Test suite
├── 📝 pyproject.toml              # Modern Python project config
├── 📋 requirements.txt            # Legacy requirements
└── 📖 README.md                   # This file
```

---

## 🔧 Configuration

Edit `src/params/config.json` to customize:

```json
{
  "seed": 0,
  "gpus": 1,
  "precision": 32,
  "device": "cuda",

  "lr": 0.0003,
  "epoch": 200,
  "batch_size": 12,
  "img_size": 512,

  "arch_name": "mobilenetv2_100",
  "num_classes": 1,
  "optim": "radam"
}
```

---

## 🛠️ Development

### Code Quality

```bash
# Format code
hatch run format

# Lint code
hatch run lint

# Type checking
mypy src/

# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## 🧪 Testing & Quality

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=src/starbucks_logo_seg --cov-report=html

# Run tests in parallel (faster!)
pytest tests/ -n auto

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage threshold
pytest tests/ --cov=src/starbucks_logo_seg --cov-fail-under=40
```

### Test Suite

| Test Category | Tests | Status |
|---------------|-------|--------|
| **Package Structure** | 3 | ✅ Passing |
| **Configuration** | 7 | ✅ Passing |
| **Imports** | 4 | ✅ Passing |
| **Metadata** | 9 | ✅ Passing |
| **Total** | **20** | **✅ 100%** |

### Code Quality

```bash
# Format code with Black
make format

# Lint with Ruff
make lint

# Type check with MyPy
make type-check

# Security scan with Bandit
make security

# Run all quality checks
make quality
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

---

## 🛠️ Modern Tooling Stack

This project uses cutting-edge Python tools for maximum developer productivity:

### Build & Packaging
- 🥚 **Hatch** - Modern PEP 517/518 packaging
- 📦 **pyproject.toml** - Single source of truth for configuration

### Code Quality
- 🐍 **Ruff** - Ultra-fast linting (10-100x faster than Flake8)
- ⚫ **Black** - Opinionated code formatting
- 🔍 **MyPy** - Static type checking
- 🔒 **Bandit** - Security vulnerability scanning

### Testing
- 🧪 **pytest** - Modern testing framework
- ⚡ **pytest-xdist** - Parallel test execution (40% faster)
- 📊 **coverage.py** - Code coverage tracking (50.94%)
- 🔬 **pytest-cov** - Coverage integration

### Development
- 🎣 **pre-commit** - Git hooks for quality checks (15+ hooks)
- 🎨 **Rich** - Beautiful terminal output
- 🖱️ **Click** - Modern CLI framework
- 📝 **nox** - Multi-version testing automation

### Performance Metrics

```
Linting:     Ruff < 0.5s  (was Flake8 ~8s)  → 16x faster ⚡
Tests:       pytest-xdist ~3s (was ~5s)     → 40% faster ⚡
Coverage:    50.94% (exceeds 40% threshold) → ✅ Good
Type Check:  MyPy 100% (modern code)        → ✅ Safe
```

---

## 📚 Documentation

### Model Architecture

The model uses a **MobileNetV2** backbone with a **U-Net** decoder:

1. **Encoder**: MobileNetV2 pretrained on ImageNet
2. **Decoder**: Upsampling layers with skip connections
3. **Output**: Single-channel binary mask

### Training Pipeline

1. **Data Augmentation**: Albumentations library
   - Random crops, rotations, flips
   - Color jittering
   - Gaussian noise

2. **Loss Function**: Binary Cross-Entropy
3. **Optimizer**: RAdam with weight decay
4. **Learning Rate**: Cosine annealing schedule

### Inference Pipeline

1. Image preprocessing (resize, normalize)
2. Model forward pass
3. Post-processing (threshold, morphology)
4. Visualization generation

---

## 🎯 Use Cases

- 🏪 **Retail Analytics**: Track brand presence in images
- 📱 **Mobile Apps**: Real-time logo detection
- 🎨 **Marketing**: Analyze brand visibility
- 🔍 **Quality Control**: Verify logo placement
- 🤖 **Computer Vision Research**: Semantic segmentation

---

## 📖 Additional Documentation

### 📚 Lessons Learned
Comprehensive documentation of the project modernization journey:
- **[LESSONS_LEARNED.md](LESSONS_LEARNED.md)** - Best practices, technical decisions, and challenges

Key topics covered:
- ✅ Project modernization strategy
- ✅ Tool selection rationale (Hatch, Ruff, MyPy)
- ✅ Testing philosophy for deep learning projects
- ✅ Performance optimization techniques
- ✅ Common challenges and solutions
- ✅ Code quality standards
- ✅ Future recommendations

### 📝 Changelog
Track all project changes:
- **[CHANGELOG.md](CHANGELOG.md)** - Detailed version history

### 🤝 Contributing
Guidelines for contributors:
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to this project

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests: `pytest tests/ -n auto`
4. Run quality checks: `make quality`
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

**Before contributing, please:**
- ✅ Read [CONTRIBUTING.md](CONTRIBUTING.md)
- ✅ Ensure all tests pass
- ✅ Maintain code coverage ≥ 40%
- ✅ Follow Black code style
- ✅ Pass Ruff linting
- ✅ Add type hints (MyPy)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Lightning** - For the amazing training framework
- **MobileNetV2** - For the efficient architecture
- **Albumentations** - For powerful data augmentation
- **Hatch** - For modern Python packaging

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ and 🐍 by the Starbucks Logo Seg Team

</div>

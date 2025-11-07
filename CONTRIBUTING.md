# 🤝 Contributing to Starbucks Logo Segmentation

First off, thank you for considering contributing to Starbucks Logo Segmentation! 🎉

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of PyTorch and computer vision

### Setting Up Development Environment

1. **Fork and Clone**

```bash
git clone https://github.com/YOUR-USERNAME/starbucks-logo-segmentation.git
cd starbucks-logo-segmentation
```

2. **Install Hatch**

```bash
pip install hatch
```

3. **Create Development Environment**

```bash
hatch env create
hatch shell
```

4. **Install Pre-commit Hooks**

```bash
pre-commit install
```

## 📝 Development Workflow

### Code Style

We use modern Python tooling to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

Run these before committing:

```bash
# Format code
hatch run format

# Lint code
hatch run lint

# Type checking
mypy src/
```

### Making Changes

1. **Create a Branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make Your Changes**

- Write clean, Pythonic code
- Add type hints
- Write docstrings for functions and classes
- Follow PEP 8 style guide

3. **Test Your Changes**

```bash
# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov
```

4. **Commit Your Changes**

```bash
git add .
git commit -m "feat: Add amazing new feature"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

5. **Push and Create Pull Request**

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub!

## 🐛 Reporting Bugs

Found a bug? Please create an issue with:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Any relevant logs or screenshots

## 💡 Suggesting Enhancements

Have an idea? Create an issue with:

- Clear description of the enhancement
- Why it would be useful
- Possible implementation approach

## 📋 Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add yourself to CONTRIBUTORS.md (if it exists)
4. Wait for code review
5. Address any feedback
6. Celebrate when it's merged! 🎉

## 🎯 Areas to Contribute

- **Model improvements**: New architectures, optimizations
- **Data augmentation**: New augmentation techniques
- **Documentation**: Tutorials, examples, improved docs
- **Testing**: More comprehensive test coverage
- **Performance**: Speed and memory optimizations
- **Features**: New functionality and tools

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Hatch Documentation](https://hatch.pypa.io/)

## ❓ Questions?

Feel free to open an issue for any questions!

---

Thank you for contributing! 🙏

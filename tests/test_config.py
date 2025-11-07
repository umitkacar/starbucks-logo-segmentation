"""Tests for configuration file"""

import json
from pathlib import Path

import pytest


def test_config_valid_json():
    """Test that config file is valid JSON"""
    config_path = Path(__file__).parent.parent / "src" / "params" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)


def test_config_required_keys():
    """Test that config has all required keys"""
    config_path = Path(__file__).parent.parent / "src" / "params" / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    required_keys = [
        "seed",
        "gpus",
        "device",
        "lr",
        "epoch",
        "batch_size",
        "img_size",
        "arch_name",
        "num_classes",
    ]

    for key in required_keys:
        assert key in config, f"Missing required key: {key}"


def test_config_values_valid():
    """Test that config values are valid"""
    config_path = Path(__file__).parent.parent / "src" / "params" / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Test numeric values
    assert isinstance(config["seed"], int)
    assert isinstance(config["epoch"], int)
    assert config["epoch"] > 0
    assert isinstance(config["batch_size"], int)
    assert config["batch_size"] > 0
    assert isinstance(config["img_size"], int)
    assert config["img_size"] > 0
    assert isinstance(config["lr"], (int, float))
    assert config["lr"] > 0

    # Test string values
    assert isinstance(config["arch_name"], str)
    assert isinstance(config["device"], str)
    assert config["device"] in ["cpu", "cuda", "mps"]


def test_config_model_architecture():
    """Test that model architecture settings are valid"""
    config_path = Path(__file__).parent.parent / "src" / "params" / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    assert "arch_name" in config
    assert "num_classes" in config
    assert isinstance(config["num_classes"], int)
    assert config["num_classes"] > 0

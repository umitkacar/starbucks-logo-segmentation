"""
🚀 Training Script for Starbucks Logo Segmentation
Modern PyTorch Lightning training with best practices
"""

import json
import logging
from logging import FileHandler, getLogger
from pathlib import Path
from time import time
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from rich.console import Console

from data import DataModule
from model import PLModule
from mylib.pytorch_lightning.logging import configure_logging

console = Console()


def load_config(config_path: str = "params/config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config = json.load(f)

    config["save_dir"] = Path("../experiments")
    return config


def setup_logger(log_dir: Path) -> logging.Logger:
    """
    Setup logger with file handler.

    Args:
        log_dir: Directory for log files

    Returns:
        Configured logger
    """
    logger = getLogger("lightning")
    logger.addHandler(FileHandler(log_dir / "train.log"))
    return logger


def create_callbacks(config: Dict[str, Any]) -> list:
    """
    Create training callbacks.

    Args:
        config: Configuration dictionary

    Returns:
        List of callbacks
    """
    monitor_metric = "ema_0_loss" if config.get("use_ema") else "val_0_loss"

    callbacks = [
        ModelCheckpoint(
            monitor=monitor_metric,
            save_last=True,
            save_top_k=3,
            mode="min",
            verbose=True,
            filename="{epoch}-{val_0_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Optional: Add early stopping
    if config.get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor=monitor_metric,
                patience=20,
                mode="min",
                verbose=True,
            ),
        )

    return callbacks


def main():
    """Main training function."""
    console.print("[bold cyan]🌟 Starbucks Logo Segmentation Training[/bold cyan]")
    console.print("[dim]Powered by PyTorch Lightning[/dim]\n")

    # Load configuration
    config = load_config()
    console.print("[green]✓[/green] Configuration loaded")
    console.print(f"[dim]Epochs: {config['epoch']}, Batch Size: {config['batch_size']}[/dim]")

    # Configure logging
    configure_logging()
    pl.seed_everything(config["seed"])
    console.print(f"[green]✓[/green] Random seed set to {config['seed']}")

    # Create TensorBoard logger
    tb_logger = pl.loggers.TensorBoardLogger(
        config["save_dir"],
        name="mobile_seg",
        version=str(int(time())),
    )

    log_dir = Path(tb_logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] Log directory created: {log_dir}")

    # Setup file logger
    logger = setup_logger(log_dir)
    logger.info(f"Training configuration: {config}")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["epoch"],
        gpus=config.get("gpus"),
        tpu_cores=config.get("num_tpu_cores"),
        precision=config["precision"],
        weights_save_path=config["save_dir"],
        resume_from_checkpoint=config.get("resume_from_checkpoint"),
        logger=tb_logger,
        callbacks=create_callbacks(config),
        deterministic=True,
        benchmark=True,
    )

    console.print("[green]✓[/green] Trainer initialized")
    console.print(
        f"[dim]Using {config.get('gpus', 0)} GPU(s), Precision: {config['precision']}-bit[/dim]",
    )

    # Create model and data module
    console.print("\n[bold yellow]Initializing model and dataset...[/bold yellow]")
    net = PLModule(config)
    dm = DataModule(config)

    console.print(f"[green]✓[/green] Model architecture: {config['arch_name']}")
    console.print("[green]✓[/green] Dataset prepared\n")

    # Start training
    console.print("[bold green]🚀 Starting training...[/bold green]\n")
    try:
        trainer.fit(net, datamodule=dm)
        console.print("\n[bold green]✅ Training completed successfully![/bold green]")
        console.print(f"[dim]Best model saved to: {log_dir}[/dim]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Training failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()

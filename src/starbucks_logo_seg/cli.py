"""
🎯 Command Line Interface for Starbucks Logo Segmentation
Modern CLI with rich output and user-friendly commands
"""

import sys

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    🌟 Starbucks Logo Segmentation CLI

    Ultra-modern tool for training and testing logo segmentation models.
    """
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="src/params/config.json",
    help="Path to configuration file",
)
@click.option(
    "--gpus",
    "-g",
    type=int,
    default=1,
    help="Number of GPUs to use",
)
def train(config: str, gpus: int):
    """
    🚀 Train the segmentation model

    Start training with the specified configuration.
    """
    console.print(f"[bold green]🚀 Starting training with config: {config}[/bold green]")
    console.print(f"[cyan]Using {gpus} GPU(s)[/cyan]")

    try:
        # Import here to avoid slow startup
        from .training.train import main as train_main

        train_main(config, gpus)
        console.print("[bold green]✅ Training completed successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ Training failed: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="src/params/config.json",
    help="Path to configuration file",
)
@click.option(
    "--image",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to input image",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="output",
    help="Output directory",
)
def predict(config: str, image: str, output: str):
    """
    🎨 Predict segmentation mask for an image

    Generate segmentation mask and visualization.
    """
    console.print(f"[bold cyan]🎨 Processing image: {image}[/bold cyan]")

    try:
        from .inference.predict import main as predict_main

        predict_main(config, image, output)
        console.print(f"[bold green]✅ Prediction saved to: {output}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ Prediction failed: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="src/params/config.json",
    help="Path to configuration file",
)
def test(config: str):
    """
    🧪 Test the trained model

    Run inference on test dataset.
    """
    console.print(f"[bold blue]🧪 Testing model with config: {config}[/bold blue]")

    try:
        from .inference.test import main as test_main

        test_main(config)
        console.print("[bold green]✅ Testing completed successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ Testing failed: {e}[/bold red]")
        sys.exit(1)


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()

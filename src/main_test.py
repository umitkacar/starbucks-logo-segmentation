"""
🎨 Inference Script for Starbucks Logo Segmentation
Modern visualization and GIF generation with best practices
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Dict

import albumentations as A
import cv2
import imageio
import imgviz
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from torch.utils.data import DataLoader

from mobile_seg.dataset import load_df, MaskDataset
from mobile_seg.modules.net import load_trained_model

console = Console()


def setup_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """
    Create output directories for results.

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary of output paths
    """
    dirs = {
        'output': output_dir,
        'visual': output_dir / "visual",
        'gif': output_dir / "gif_image"
    }

    for path in dirs.values():
        path.mkdir(exist_ok=True, parents=True)

    return dirs


def load_config(config_path: str = 'params/config.json') -> dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        return json.load(f)


def load_and_preprocess_image(
    image_path: Path,
    img_size: int = 512
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess image for inference.

    Args:
        image_path: Path to input image
        img_size: Target image size

    Returns:
        Tuple of (preprocessed tensor, original image array)
    """
    # Read and convert image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(
        img_rgb,
        (img_size, img_size),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert to tensor
    img_tensor = torch.from_numpy(
        img_resized.astype(np.float32).transpose((2, 0, 1)) / 255.0
    ).unsqueeze(0)

    return img_tensor, img_resized


def visualize_comparison(
    output_path: Path,
    original: np.ndarray,
    prediction: np.ndarray,
    save: bool = True
) -> None:
    """
    Create side-by-side visualization of original and predicted mask.

    Args:
        output_path: Path to save visualization
        original: Original image
        prediction: Predicted mask
        save: Whether to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(prediction, cmap=plt.cm.gray)
    axes[1].set_title('Predicted Mask', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()

    if save:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_overlay_visualization(
    original: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create overlay visualization with colored mask.

    Args:
        original: Original image
        mask: Predicted mask
        alpha: Transparency level

    Returns:
        Overlay visualization
    """
    mask_int = mask.astype(int)

    overlay = imgviz.label2rgb(
        mask_int,
        image=original,
        alpha=alpha,
        font_size=30,
        colormap=imgviz.label_colormap(n_label=256, value=255),
        loc="rb",
    )

    return overlay


def generate_gif(
    gif_dir: Path,
    output_path: Path,
    original: np.ndarray,
    overlay: np.ndarray,
    fps: int = 2
) -> None:
    """
    Generate animated GIF showing original and overlay.

    Args:
        gif_dir: Directory for temporary frames
        output_path: Path to save GIF
        original: Original image
        overlay: Overlay visualization
        fps: Frames per second
    """
    # Save frames
    frame1_path = gif_dir / "gif_frame_1.jpg"
    frame2_path = gif_dir / "gif_frame_2.jpg"

    plt.imsave(frame1_path, original)
    plt.imsave(frame2_path, overlay)

    # Create GIF
    frames = [imageio.imread(frame1_path), imageio.imread(frame2_path)]

    # Download imageio plugin if needed
    try:
        imageio.mimsave(output_path, frames, 'GIF-FI', fps=fps)
    except Exception:
        # Fallback to standard GIF
        imageio.mimsave(output_path, frames, fps=fps)


def process_image(
    model: torch.nn.Module,
    image_path: Path,
    config: dict,
    output_dirs: Dict[str, Path]
) -> None:
    """
    Process a single image and generate all visualizations.

    Args:
        model: Trained model
        image_path: Path to input image
        config: Configuration dictionary
        output_dirs: Output directories
    """
    console.print(f"[cyan]Processing: {image_path.name}[/cyan]")

    # Load and preprocess
    img_tensor, img_array = load_and_preprocess_image(
        image_path,
        config["img_size"]
    )

    # Inference
    with torch.no_grad():
        output = model(img_tensor.to(config["device"])).cpu()
        mask = output.squeeze().numpy()

    # Create visualizations
    vis_path = output_dirs['visual'] / f"Starbucks_logo_{image_path.stem}.png"
    visualize_comparison(vis_path, img_array, mask, save=True)

    # Create overlay
    overlay = create_overlay_visualization(img_array, mask)

    # Generate GIF
    gif_path = output_dirs['output'] / f'Starbucks_logo_{image_path.stem}.gif'
    generate_gif(output_dirs['gif'], gif_path, img_array, overlay)

    console.print(f"[green]✓[/green] Saved results to {gif_path}")


def main():
    """Main inference function."""
    console.print("[bold cyan]🎨 Starbucks Logo Segmentation - Inference[/bold cyan]")
    console.print("[dim]Generating visualizations and animations[/dim]\n")

    # Setup
    output_dirs = setup_output_dirs(Path('../output'))
    config = load_config()

    console.print(f"[green]✓[/green] Configuration loaded")
    console.print(f"[green]✓[/green] Output directories created\n")

    # Load model
    console.print("[yellow]Loading trained model...[/yellow]")
    model = load_trained_model(config).to(config["device"]).eval()
    console.print(f"[green]✓[/green] Model loaded successfully\n")

    # Example image path (update this to your actual image path)
    example_images = [
        Path("../../datahub/starbucks/image/bagir-bahana-6295IjcQkSQ-unsplash.jpg"),
    ]

    # Process images
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing images...",
            total=len(example_images)
        )

        for img_path in example_images:
            if img_path.exists():
                process_image(model, img_path, config, output_dirs)
            else:
                console.print(f"[yellow]⚠[/yellow] Image not found: {img_path}")

            progress.advance(task)

    console.print("\n[bold green]✅ Inference completed![/bold green]")
    console.print(f"[dim]Results saved to: {output_dirs['output']}[/dim]")


if __name__ == '__main__':
    main()

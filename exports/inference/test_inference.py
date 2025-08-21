"""
Simple test script for ObjectFormer ONNX inference.
Usage: python test_inference.py --image path/to/image.jpg --model path/to/model.onnx
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from onnx_inference_wrapper import ObjectFormerONNXInference


def visualize_results(image_path: str, results: dict, save_path: str = None):
    """Visualize inference results."""
    # Load original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Detection result
    detection_text = f"Tampered: {'Yes' if results['is_tampered'] else 'No'}\n"
    detection_text += f"Confidence: {results['detection_confidence']:.3f}"
    axes[1].text(
        0.1,
        0.5,
        detection_text,
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )
    axes[1].set_title("Detection Result")
    axes[1].axis("off")

    # Localization mask
    if results["confidence_mask"] is not None:
        axes[2].imshow(img_rgb, alpha=0.7)
        axes[2].imshow(results["confidence_mask"], alpha=0.5, cmap="jet")
        axes[2].set_title("Tampered Regions (Heatmap)")
    else:
        axes[2].text(
            0.5,
            0.5,
            "No localization\navailable",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[2].set_title("Localization Result")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test ObjectFormer ONNX inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="../onnx/objectformer_full.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization (optional)",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ONNX Runtime providers",
    )

    args = parser.parse_args()

    # Create results directory if it doesn't exist
    results_dir = Path("inference_results")
    results_dir.mkdir(exist_ok=True)

    # Validate inputs
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return

    try:
        # Initialize inference wrapper
        print("Loading ONNX model...")
        inferencer = ObjectFormerONNXInference(
            onnx_model_path=args.model, providers=args.providers
        )

        # Print model info
        model_info = inferencer.get_model_info()
        print(f"\nModel info:")
        print(f"- Providers: {model_info['providers']}")
        print(f"- Input shape: {model_info['inputs'][0]['shape']}")
        print(f"- Output count: {len(model_info['outputs'])}")

        # Run inference
        print(f"\nRunning inference on: {args.image}")
        results = inferencer.predict(
            args.image, threshold=args.threshold, return_original_size=True
        )

        # Print results
        print(f"\nResults:")
        print(f"- Is tampered: {results['is_tampered']}")
        print(f"- Detection confidence: {results['detection_confidence']:.4f}")
        print(f"- Raw logits: {results['raw_detection_logits']:.4f}")

        if results["binary_mask"] is not None:
            tampered_pixels = np.sum(results["binary_mask"])
            total_pixels = results["binary_mask"].size
            tampered_ratio = tampered_pixels / total_pixels
            print(f"- Tampered area: {tampered_ratio:.2%} of image")
        else:
            print("- No localization mask available")

        # Generate output filename if not provided
        if args.output is None:
            image_name = Path(args.image).stem
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = results_dir / f"{image_name}_{timestamp}_inference.png"

        # Visualize results
        print("\nGenerating visualization...")
        visualize_results(args.image, results, str(args.output))

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

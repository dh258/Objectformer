"""
Prediction script with confidence scoring for KYC reviewers.

This script generates predictions with confidence scores and visual outputs
for human review workflows. It supports batch processing and outputs results
in a reviewer-friendly format.

Usage:
    uv run python tools/predict_with_confidence.py \\
        --cfg configs/objectformer_doc_tamper_adamw.yaml \\
        --checkpoint path/to/checkpoint.pyth \\
        --input_dir path/to/images \\
        --output_dir path/to/results \\
        --threshold 0.4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ObjectFormer.utils.build_helper import build_model
from ObjectFormer.utils.checkpoint import load_test_checkpoint


def load_config(cfg_file):
    """Load config from YAML files."""
    with open('./configs/default.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    with open(cfg_file, 'r') as f:
        custom_cfg = yaml.safe_load(f)

    # Merge custom config into base config
    def merge_a_into_b(a, b):
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                merge_a_into_b(v, b[k])
            else:
                b[k] = v

    merge_a_into_b(custom_cfg, cfg)
    return cfg


def sigmoid(x):
    """Apply sigmoid activation to convert logits to probabilities."""
    return 1 / (1 + np.exp(-x))


def load_and_preprocess_image(image_path, img_size=320):
    """Load and preprocess image for model input."""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape[:2]

    # Resize
    img_resized = cv2.resize(img, (img_size, img_size))

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_resized / 255.0 - mean) / std

    # Convert to tensor
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()

    return img_tensor, original_shape


def create_overlay_visualization(image_path, mask, confidence, prediction, threshold, output_path):
    """Create visualization with heatmap overlay for reviewers."""
    # Load original image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize mask to match image
    h, w = img.shape[:2]
    mask_resized = cv2.resize(mask, (w, h))

    # Create heatmap
    heatmap = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Add text annotations
    status = "TAMPERED" if prediction else "AUTHENTIC"
    color = (255, 0, 0) if prediction else (0, 255, 0)

    cv2.putText(overlay, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(overlay, f"Confidence: {confidence:.2%}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(overlay, f"Threshold: {threshold}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Save
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), overlay_bgr)


def predict_batch(model, image_paths, cfg, threshold=0.4, output_dir=None):
    """
    Run predictions on a batch of images with confidence scoring.

    Args:
        model: Trained ObjectFormer model
        image_paths: List of image file paths
        cfg: Configuration dictionary
        threshold: Detection threshold (default: 0.4 for conservative flagging)
        output_dir: Directory to save visualizations (optional)

    Returns:
        List of dictionaries containing predictions and metadata
    """
    model.eval()
    results = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'visualizations').mkdir(exist_ok=True)
        (output_dir / 'masks').mkdir(exist_ok=True)

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Load and preprocess
                img_tensor, original_shape = load_and_preprocess_image(
                    img_path, cfg['DATASET']['IMG_SIZE']
                )

                # Prepare batch
                batch = {
                    'image': img_tensor.unsqueeze(0).cuda(),
                    'bin_label': torch.tensor([0]).cuda(),  # Dummy label
                    'mask': torch.zeros(1, original_shape[0], original_shape[1]).cuda()  # Dummy mask
                }

                # Forward pass
                detection_logit, localization_masks = model(batch)

                # Get detection confidence (apply sigmoid to logit)
                detection_prob = torch.sigmoid(detection_logit).item()
                prediction = detection_prob > threshold

                # Get localization mask
                loc_mask = torch.sigmoid(localization_masks[-1]).squeeze().cpu().numpy()

                # Prepare result
                result = {
                    'image_path': str(img_path),
                    'image_name': Path(img_path).name,
                    'prediction': 'tampered' if prediction else 'authentic',
                    'confidence': float(detection_prob),
                    'threshold': threshold,
                    'flagged_for_review': prediction,
                    'raw_logit': float(detection_logit.item()),
                    'max_mask_score': float(loc_mask.max()),
                    'mean_mask_score': float(loc_mask.mean()),
                }

                # Save visualizations if output directory provided
                if output_dir:
                    img_name = Path(img_path).stem

                    # Save overlay visualization
                    overlay_path = output_dir / 'visualizations' / f'{img_name}_overlay.jpg'
                    create_overlay_visualization(
                        img_path, loc_mask, detection_prob,
                        prediction, threshold, overlay_path
                    )

                    # Save raw mask
                    mask_path = output_dir / 'masks' / f'{img_name}_mask.png'
                    cv2.imwrite(str(mask_path), (loc_mask * 255).astype(np.uint8))

                    result['overlay_path'] = str(overlay_path)
                    result['mask_path'] = str(mask_path)

                results.append(result)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({
                    'image_path': str(img_path),
                    'image_name': Path(img_path).name,
                    'error': str(e)
                })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions with confidence scores for KYC review"
    )
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing images to process')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Detection threshold (default: 0.4 for conservative flagging)')
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png'],
                        help='Image file extensions to process')

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.cfg)

    # Override checkpoint path
    cfg['TEST']['CHECKPOINT_PATH'] = args.checkpoint

    # Build model
    print("Loading model...")
    model = build_model(cfg)
    load_test_checkpoint(cfg, model)
    model.cuda()
    print("Model loaded successfully.")

    # Find all images
    input_dir = Path(args.input_dir)
    image_paths = []
    for ext in args.extensions:
        image_paths.extend(input_dir.glob(f'**/*{ext}'))

    print(f"Found {len(image_paths)} images to process.")

    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return

    # Run predictions
    results = predict_batch(
        model, image_paths, cfg,
        threshold=args.threshold,
        output_dir=args.output_dir
    )

    # Save results to JSON
    output_path = Path(args.output_dir) / 'predictions.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate summary statistics
    total = len(results)
    flagged = sum(1 for r in results if r.get('flagged_for_review', False))
    errors = sum(1 for r in results if 'error' in r)
    avg_confidence_flagged = np.mean([
        r['confidence'] for r in results
        if r.get('flagged_for_review', False) and 'confidence' in r
    ]) if flagged > 0 else 0

    summary = {
        'total_images': total,
        'flagged_for_review': flagged,
        'authentic': total - flagged - errors,
        'errors': errors,
        'threshold': args.threshold,
        'flagged_percentage': f"{flagged/total*100:.1f}%" if total > 0 else "0%",
        'avg_confidence_flagged': float(avg_confidence_flagged)
    }

    summary_path = Path(args.output_dir) / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total images processed: {summary['total_images']}")
    print(f"Flagged for review: {summary['flagged_for_review']} ({summary['flagged_percentage']})")
    print(f"Authentic: {summary['authentic']}")
    print(f"Errors: {summary['errors']}")
    print(f"Detection threshold: {summary['threshold']}")
    if flagged > 0:
        print(f"Avg confidence (flagged): {summary['avg_confidence_flagged']:.2%}")
    print("="*60)
    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    if args.output_dir:
        print(f"Visualizations saved to: {Path(args.output_dir) / 'visualizations'}")


if __name__ == '__main__':
    main()

"""
Analyze predictions on validation set to understand model behavior.

This script runs predictions on the validation set and generates detailed
analysis including false positives, false negatives, and confidence distributions.

Usage:
    uv run python tools/analyze_validation_predictions.py \\
        --cfg configs/objectformer_doc_tamper_adamw.yaml \\
        --threshold 0.4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ObjectFormer.utils.build_helper import build_model, build_dataset, build_dataloader
from ObjectFormer.utils.checkpoint import load_test_checkpoint
from ObjectFormer.utils.distributed import init_process_group


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


def analyze_validation_set(model, val_loader, cfg, threshold=0.4):
    """
    Analyze model predictions on validation set.

    Returns detailed statistics including:
    - Confusion matrix
    - False positive/negative examples
    - Confidence distributions
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_confidences = []
    all_samples = []

    print("Running predictions on validation set...")

    with torch.no_grad():
        for batch_idx, samples in enumerate(val_loader):
            # Move to GPU
            samples = dict(map(
                lambda sample: (sample[0], sample[1].cuda(non_blocking=True)),
                samples.items()
            ))

            # Forward pass
            detection_logits, localization_masks = model(samples)

            # Get predictions
            confidences = torch.sigmoid(detection_logits).cpu().numpy()
            predictions = (confidences > threshold).astype(int)
            labels = samples['bin_label'].cpu().numpy()

            # Store results
            all_labels.extend(labels)
            all_preds.extend(predictions)
            all_confidences.extend(confidences)

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_confidences = np.array(all_confidences)

    # Calculate metrics
    tp = np.sum((all_labels == 1) & (all_preds == 1))
    tn = np.sum((all_labels == 0) & (all_preds == 0))
    fp = np.sum((all_labels == 0) & (all_preds == 1))
    fn = np.sum((all_labels == 1) & (all_preds == 0))

    accuracy = (tp + tn) / len(all_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Analyze false positives and false negatives
    fp_indices = np.where((all_labels == 0) & (all_preds == 1))[0]
    fn_indices = np.where((all_labels == 1) & (all_preds == 0))[0]

    results = {
        'total_samples': len(all_labels),
        'threshold': threshold,
        'confusion_matrix': {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'class_distribution': {
            'authentic': int(np.sum(all_labels == 0)),
            'tampered': int(np.sum(all_labels == 1))
        },
        'false_positives': {
            'count': int(fp),
            'indices': fp_indices.tolist(),
            'avg_confidence': float(all_confidences[fp_indices].mean()) if len(fp_indices) > 0 else 0
        },
        'false_negatives': {
            'count': int(fn),
            'indices': fn_indices.tolist(),
            'avg_confidence': float(all_confidences[fn_indices].mean()) if len(fn_indices) > 0 else 0
        },
        'confidence_stats': {
            'authentic_samples': {
                'mean': float(all_confidences[all_labels == 0].mean()),
                'std': float(all_confidences[all_labels == 0].std()),
                'min': float(all_confidences[all_labels == 0].min()),
                'max': float(all_confidences[all_labels == 0].max())
            },
            'tampered_samples': {
                'mean': float(all_confidences[all_labels == 1].mean()),
                'std': float(all_confidences[all_labels == 1].std()),
                'min': float(all_confidences[all_labels == 1].min()),
                'max': float(all_confidences[all_labels == 1].max())
            }
        }
    }

    return results, all_labels, all_preds, all_confidences


def create_visualizations(all_labels, all_confidences, threshold, output_dir):
    """Create visualization plots for analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Confidence distribution plot
    plt.figure(figsize=(10, 6))

    authentic_conf = all_confidences[all_labels == 0]
    tampered_conf = all_confidences[all_labels == 1]

    plt.hist(authentic_conf, bins=20, alpha=0.5, label='Authentic', color='green')
    plt.hist(tampered_conf, bins=20, alpha=0.5, label='Tampered', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')

    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Score Distribution by Ground Truth Label')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_dir / 'confidence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_dir / 'confidence_distribution.png'}")


def print_analysis_report(results):
    """Print detailed analysis report to console."""
    print("\n" + "="*80)
    print("VALIDATION SET ANALYSIS REPORT")
    print("="*80)

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {results['total_samples']}")
    print(f"  Authentic: {results['class_distribution']['authentic']} ({results['class_distribution']['authentic']/results['total_samples']*100:.1f}%)")
    print(f"  Tampered: {results['class_distribution']['tampered']} ({results['class_distribution']['tampered']/results['total_samples']*100:.1f}%)")

    print(f"\nDetection Threshold: {results['threshold']}")

    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"  True Positives (Tampered → Flagged):  {cm['true_positives']}")
    print(f"  True Negatives (Authentic → Clear):   {cm['true_negatives']}")
    print(f"  False Positives (Authentic → Flagged): {cm['false_positives']} ⚠️")
    print(f"  False Negatives (Tampered → Clear):    {cm['false_negatives']} ⚠️")

    print(f"\nPerformance Metrics:")
    m = results['metrics']
    print(f"  Accuracy:  {m['accuracy']:.3f}")
    print(f"  Precision: {m['precision']:.3f}")
    print(f"  Recall:    {m['recall']:.3f}")
    print(f"  F1 Score:  {m['f1_score']:.3f}")

    print(f"\nError Analysis:")
    if results['false_positives']['count'] > 0:
        print(f"  False Positives: {results['false_positives']['count']}")
        print(f"    → Legitimate docs incorrectly flagged for review")
        print(f"    → Avg confidence: {results['false_positives']['avg_confidence']:.3f}")
        print(f"    → Sample indices: {results['false_positives']['indices']}")
    else:
        print(f"  False Positives: 0 ✓")

    if results['false_negatives']['count'] > 0:
        print(f"  False Negatives: {results['false_negatives']['count']}")
        print(f"    → Tampered docs missed by detection")
        print(f"    → Avg confidence: {results['false_negatives']['avg_confidence']:.3f}")
        print(f"    → Sample indices: {results['false_negatives']['indices']}")
    else:
        print(f"  False Negatives: 0 ✓")

    print(f"\nConfidence Statistics:")
    auth = results['confidence_stats']['authentic_samples']
    tamp = results['confidence_stats']['tampered_samples']
    print(f"  Authentic samples:")
    print(f"    Mean: {auth['mean']:.3f} ± {auth['std']:.3f}")
    print(f"    Range: [{auth['min']:.3f}, {auth['max']:.3f}]")
    print(f"  Tampered samples:")
    print(f"    Mean: {tamp['mean']:.3f} ± {tamp['std']:.3f}")
    print(f"    Range: [{tamp['min']:.3f}, {tamp['max']:.3f}]")

    print(f"\nKYC MVP Assessment:")
    if results['false_negatives']['count'] == 0:
        print("  ✓ No tampered documents missed - safe for filtering")
    else:
        fn_rate = results['false_negatives']['count'] / results['class_distribution']['tampered']
        print(f"  ⚠️ False negative rate: {fn_rate:.1%} - some tampered docs may slip through")

    if results['false_positives']['count'] <= 2:
        print("  ✓ Low false positive rate - minimal review workload")
    else:
        fp_rate = results['false_positives']['count'] / results['class_distribution']['authentic']
        print(f"  ⚠️ False positive rate: {fp_rate:.1%} - {results['false_positives']['count']} legitimate docs flagged")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze model predictions on validation set"
    )
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Detection threshold (default: 0.4)')
    parser.add_argument('--output_dir', type=str, default='validation_analysis',
                        help='Directory to save analysis results')

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.cfg)

    # Initialize distributed process group (required for dataloader)
    init_process_group(0, cfg['NUM_GPUS'])

    # Build model
    print("Loading model...")
    model = build_model(cfg)
    load_test_checkpoint(cfg, model)
    model.cuda()
    print("Model loaded successfully.\n")

    # Build validation dataset
    print("Loading validation dataset...")
    val_dataset = build_dataset('val', cfg)
    val_loader = build_dataloader(val_dataset, 'val', cfg)
    print(f"Validation set: {len(val_dataset)} samples\n")

    # Run analysis
    results, all_labels, all_preds, all_confidences = analyze_validation_set(
        model, val_loader, cfg, threshold=args.threshold
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'validation_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Create visualizations
    create_visualizations(all_labels, all_confidences, args.threshold, output_dir)

    # Print report
    print_analysis_report(results)


if __name__ == '__main__':
    main()

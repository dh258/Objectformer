"""
ObjectFormer Training-to-Pretrained Conversion Script

This script converts training checkpoints (.pyth) to clean pretrained format (.pth)
suitable for transfer learning and model sharing.

Usage:
    python convert_training_to_pretrained.py --checkpoint path/to/ckpt_epoch_XXXXX.pyth --output path/to/pretrained_model.pth
"""

import argparse
import json
import os
import torch
from datetime import datetime


def extract_model_metadata(checkpoint):
    """Extract model architecture and training information from checkpoint."""
    cfg = checkpoint.get("cfg", {})

    # Extract architecture info from config
    model_cfg = cfg.get("MODEL", {})
    train_cfg = cfg.get("TRAIN", {})
    dataset_cfg = cfg.get("DATASET", {})

    metadata = {
        "architecture": "ObjectFormer",
        "model_type": model_cfg.get("META_ARCHITECTURE", "ObjectFormer"),
        "input_size": [288, 288],
        "num_classes": 2,  # Binary classification (authentic/tampered)
        "model_config": {
            "layers": model_cfg.get("LAYERS", [1, 1, 3, 1]),
            "channels": model_cfg.get("CHANNELS", [96, 192, 384, 768]),
            "heads": model_cfg.get("HEADS", [3, 6, 12, 24]),
            "img_size": model_cfg.get("IMG_SIZE", 380),
            "parts": model_cfg.get("PARTS", 64),
        },
        "training_info": {
            "epoch": checkpoint.get("epoch", -1),
            "dataset_root": dataset_cfg.get("ROOT_DIR", "unknown"),
            "batch_size": train_cfg.get("BATCH_SIZE", "unknown"),
            "learning_rate": cfg.get("OPTIMIZER", {}).get("BASE_LR", "unknown"),
            "max_epochs": train_cfg.get("MAX_EPOCH", "unknown"),
        },
        "conversion_info": {
            "converted_at": datetime.now().isoformat(),
            "original_checkpoint": None,  # Will be filled in convert function
            "script_version": "1.0",
        },
    }

    return metadata


def validate_model_state(model_state):
    """Validate the model state dictionary for completeness."""
    required_keys = [
        "obj_queries",
        "conv1.weight",
        "norm1.weight",
        "layer_0.rpn_qpos",
        "layer_3.last_enc.enc_attn.proj.weight",
    ]

    missing_keys = []
    for key in required_keys:
        if key not in model_state:
            missing_keys.append(key)

    if missing_keys:
        raise ValueError(f"Model state missing required keys: {missing_keys}")

    # Count total parameters
    total_params = sum(p.numel() for p in model_state.values() if hasattr(p, "numel"))
    print(f"✓ Model state validation passed. Total parameters: {total_params:,}")

    return True


def convert_training_to_pretrained(
    checkpoint_path, output_path, include_performance_metrics=False
):
    """
    Convert training checkpoint to clean pretrained format.

    Args:
        checkpoint_path (str): Path to training checkpoint (.pyth)
        output_path (str): Path to save pretrained model (.pth)
        include_performance_metrics (bool): Whether to include performance metrics in metadata
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load the training checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Validate checkpoint structure
    required_keys = ["model_state", "epoch", "cfg"]
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Required key '{key}' not found in checkpoint")

    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Extract model weights only
    model_state = checkpoint["model_state"]
    validate_model_state(model_state)

    # Extract metadata
    metadata = extract_model_metadata(checkpoint)
    metadata["conversion_info"]["original_checkpoint"] = os.path.basename(
        checkpoint_path
    )

    # Optionally include performance metrics placeholder
    if include_performance_metrics:
        metadata["performance"] = {
            "image_accuracy": "TBD",
            "pixel_f1": "TBD",
            "image_auc": "TBD",
            "pixel_auc": "TBD",
            "note": "Performance metrics need to be filled manually from evaluation results",
        }

    # Create pretrained model dictionary
    pretrained_model = {
        "model_state": model_state,
        "metadata": metadata,
    }

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save pretrained model
    print(f"Saving pretrained model to: {output_path}")
    torch.save(pretrained_model, output_path)

    # Save metadata as separate JSON file for easy inspection
    metadata_path = output_path.replace(".pth", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"✓ Conversion completed successfully!")
    print(f"  - Pretrained model: {output_path}")
    print(f"  - Metadata file: {metadata_path}")
    print(f"  - Model parameters: {len(model_state):,}")
    print(f"  - Original epoch: {checkpoint['epoch']}")

    return output_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert ObjectFormer training checkpoint to pretrained format"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        help="Path to training checkpoint (.pyth file)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to save pretrained model (.pth file)",
    )
    parser.add_argument(
        "--include-performance",
        action="store_true",
        help="Include performance metrics placeholders in metadata",
    )

    args = parser.parse_args()

    try:
        convert_training_to_pretrained(
            args.checkpoint, args.output, args.include_performance
        )
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

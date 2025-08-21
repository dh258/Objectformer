# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ObjectFormer is a PyTorch implementation for image manipulation detection and localization. It's designed as a defensive security tool that can identify tampered regions in images and classify whether an image has been manipulated. The model uses a transformer-based architecture with object-centric reasoning capabilities.

## Key Commands

### Training
```bash
# Basic training with default config
uv run python run.py --cfg configs/objectformer_bs24_lr2.5e-4.yaml

# Training with custom config  
uv run python run.py --cfg configs/your_config.yaml
```

### Testing/Evaluation
```bash
# Set TRAIN.ENABLE to False in config, then run:
uv run python run.py --cfg configs/objectformer_bs24_lr2.5e-4.yaml
```

### ONNX Export
```bash
# Export trained model to ONNX format for deployment
uv run python exports/export_to_onnx.py \
    --checkpoint exports/pretrained/objectformer_pretrained.pth \
    --output exports/onnx/objectformer_full.onnx \
    --validate

# Custom input size and batch size
uv run python exports/export_to_onnx.py \
    --checkpoint path/to/your/weights.pth \
    --output path/to/output.onnx \
    --input_size 320 \
    --batch_size 1 \
    --validate
```

### Environment Setup
Use Python 3.10 and use 'uv' as the package manager. The project uses pyproject.toml with locked dependencies for reproducible builds.

```bash
# Install dependencies (creates virtual env and installs from uv.lock)
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Verify CUDA availability
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Architecture Overview

### Core Components

**Model Architecture** (`ObjectFormer/models/objectformer.py:236`):
- `ObjectFormer`: Main model class implementing transformer-based architecture
- Multi-stage processing with encoder-decoder blocks (`OFBlock`)
- Object queries for reasoning about image regions
- Dual output: detection (binary classification) + localization (pixel-level masks)

**Dataset Pipeline** (`ObjectFormer/datasets/dataset.py:12`):
- `TamperingDataset`: Handles tab-separated annotation format
- Expected format: `image_path\tmask_path\tlabel` (label: 0=authentic, 1=tampered)
- Automatic zero mask generation for authentic images
- Supports albumentations for data augmentation

**Training Loop** (`tools/train.py:25`):
- Combined loss: BCELossWithLogits (detection) + DiceLoss (localization)
- Mixed precision training support with AMP
- Automatic checkpoint saving and resumption

**Evaluation** (`tools/test.py:22`):
- Multi-level metrics: Image-level accuracy/AUC/F1 + Pixel-level AUC/F1
- Threshold-based mask evaluation (configurable via `TEST.THRES`)

### Key Model Parameters

From `configs/objectformer_bs24_lr2.5e-4.yaml`:
- **Image size**: 224px input, processed at 380px internally
- **Architecture**: 4-stage transformer with [1,1,3,1] layers
- **Channels**: [96,192,384,768] progression
- **Multi-head attention**: [3,6,12,24] heads per stage
- **Object queries**: 64 parts per stage for reasoning

## Dataset Requirements

### File Structure
```
your_dataset/
├── images/          # JPG/PNG images
├── masks/           # Grayscale masks (0=authentic, 255=tampered)
├── train_split.txt  # Training annotations
├── val_split.txt    # Validation annotations  
└── test_split.txt   # Test annotations
```

### Annotation Format
Tab-separated format: `image_path\tmask_path\tlabel`
- Paths relative to ROOT_DIR
- Labels: 0=authentic, 1=tampered
- Mask paths only required for tampered images

### Minimum Test Setup
- 20-30 authentic + 20-30 tampered images
- Corresponding pixel-level masks for tampered images
- Set batch_size=4, MAX_EPOCH=5 for initial verification

## Configuration System

**Main config**: `configs/default.yaml` contains base parameters
**Model-specific**: Override specific parameters in configs like `objectformer_bs24_lr2.5e-4.yaml`

Key sections:
- `DATASET`: Root directory, splits, augmentation parameters
- `MODEL`: Architecture parameters, pretrained weights path
- `TRAIN`: Training settings, epochs, AMP, checkpointing  
- `OPTIMIZER`/`SCHEDULER`: Learning rate and optimization settings
- `LOSS`: Loss function weights and configurations

## Development Notes

### Model Registry System
Models and datasets use registry decorators:
- `@MODEL_REGISTRY.register()` for model classes
- `@DATASET_REGISTRY.register()` for dataset classes

### Key Utilities
- `ObjectFormer/utils/build_helper.py`: Factory functions for models, datasets, losses
- `ObjectFormer/utils/checkpoint.py`: Checkpoint saving/loading with auto-resume
- `ObjectFormer/utils/meters.py`: Training metrics and logging
- `tools/eval_utils.py`: Evaluation utilities and metric calculations

### Pretrained Models
Download ImageNet pretrained backbone from provided Google Drive link before training. Set path in config under `MODEL.PRETRAINED`.

## Security Context

This is a **defensive security tool** designed to:
- Detect image manipulation and forgery
- Localize tampered regions in images  
- Support forensic analysis and verification workflows
- Help identify potential disinformation through manipulated images

The model outputs both binary classification (tampered/authentic) and pixel-level localization masks for detailed analysis.
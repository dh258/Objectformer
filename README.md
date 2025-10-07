# ObjectFormer for Image Manipulation Detection and Localization

Official PyTorch implementation of [ObjectFormer for Image Manipulation Detection and Localization](https://arxiv.org/abs/2203.14681), CVPR 2022.

## Training and Evaluation

Please first download the imagenet-pretrained model from this [Google Drive](https://drive.google.com/file/d/1I-H58ldwwvoKD1GX0jdlNJGA9hqgYNKy/view?usp=sharing) link.

### Full Training

For full training from scratch:

```bash
uv run python run.py --cfg configs/objectformer_bs24_lr2.5e-4.yaml
```

### Fine-tuning on Small Datasets (NEW)

For fine-tuning on small datasets (~100 samples), use the freeze backbone strategy to prevent overfitting:

```bash
uv run python run.py --cfg configs/objectformer_doc_tamper_adamw.yaml
```

**Key features:**
- **Layer freezing**: Automatically freezes backbone layers (conv stem + stages 0-2)
- **Trainable heads**: Only trains `last_fc` and localization heads (`localizer_0/1/2`)
- **Parameter reduction**: ~90-95% of parameters frozen for efficient fine-tuning
- **Optimized hyperparameters**: Low LR (1e-5), high weight decay (0.01), small batch size (4)

**Configuration:**
```yaml
OPTIMIZER:
  FREEZE_STRATEGY: freeze_backbone  # Enable layer freezing
  BASE_LR: 0.00001                 # Very low LR for fine-tuning
  WEIGHT_DECAY: 0.01               # High regularization

TRAIN:
  MAX_EPOCH: 15                     # Short training to prevent overfitting
  BATCH_SIZE: 4                     # Small batches for small datasets
```

**Expected behavior:**
- Training converges within 5-10 epochs
- Validation loss should track training loss closely (gap < 0.1)
- If validation loss diverges, stop training earlier or increase weight decay

**Supported freeze strategies:**
- `none` (default): Train all parameters - backward compatible
- `freeze_backbone`: Freeze backbone, train only classification & localization heads

### Evaluation

For evaluation, set `TRAIN.ENABLE` to `False` in your config. For better performance on Pixel F1, adjust `TEST.THRES` (0.5 by default) on each testing dataset.

## Checkpoint Conversion

To convert training checkpoints to clean pretrained format suitable for transfer learning and sharing:

```bash
# Convert training checkpoint to pretrained format
uv run python exports/convert_training_to_pretrained.py \
    --checkpoint path/to/ckpt_epoch_XXXXX.pyth \
    --output exports/pretrained/your_model.pth

# Include performance metrics placeholders in metadata
uv run python exports/convert_training_to_pretrained.py \
    --checkpoint path/to/ckpt_epoch_XXXXX.pyth \
    --output exports/pretrained/your_model.pth \
    --include-performance
```

The conversion script:
- Extracts only model weights from training checkpoints (removes optimizer state, scheduler state, etc.)
- Creates clean pretrained format (.pth) suitable for transfer learning
- Generates metadata JSON file with model architecture and training information
- Validates model state completeness before conversion
- Preserves original training configuration and epoch information

**Output files:**
- `your_model.pth` - Clean pretrained model weights
- `your_model_metadata.json` - Model architecture and training metadata

## ONNX Export

To export a trained ObjectFormer model to ONNX format for deployment:

```bash
# Basic export with validation
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

The exported ONNX model includes both detection (binary classification) and localization (pixel-level masks) outputs. The script automatically extracts model architecture from pretrained weights metadata, eliminating the need for configuration files.

**Output format:**
- Detection: `[batch, 1]` - Binary probability for tampering detection
- Localization: 3 multi-scale masks `[batch, 1, H, W]` - Pixel-level tampering masks

## ONNX Inference

To run inference on images using the exported ONNX model:

```bash
# Test inference on a single image
uv run python exports/inference/test_inference.py \
    --image path/to/your/image.jpg \
    --model exports/onnx/objectformer_full.onnx \
    --threshold 0.5

# The inference wrapper automatically applies sigmoid to detection logits
# and provides both binary classification and pixel-level localization
```

**Python API usage:**
```python
from exports.inference.onnx_inference_wrapper import ObjectFormerONNXInference

# Initialize inference wrapper
inferencer = ObjectFormerONNXInference("exports/onnx/objectformer_full.onnx")

# Run inference
results = inferencer.predict("path/to/image.jpg", threshold=0.5)

# Results contain:
# - is_tampered: Boolean classification result
# - detection_confidence: Sigmoid probability [0-1]  
# - binary_mask: Binarized localization mask
# - confidence_mask: Raw localization confidence map
```

Note that we only release the checkpoints trained on the publicly available dataset ([CASIAV2](https://drive.google.com/file/d/1vd7o7JI-_EukyplskeuSWuham4iCWuTq/view?usp=sharing) and [IMD20](https://drive.google.com/file/d/1IWQvoMl9iaefCLbF5gXVbdB2ICfSXLPP/view?usp=sharing)). For a fair comparison with our method, please finetune it with your data.

## Testing

The repository includes a comprehensive test suite for validating the freeze strategy implementation.

### Running Tests

```bash
# Install dev dependencies (pytest, pytest-cov)
uv sync --extra dev

# Run all freeze strategy tests
uv run pytest tests/test_freeze_strategy.py -v

# Run with coverage report
uv run pytest tests/test_freeze_strategy.py --cov=ObjectFormer/utils/model_helpers --cov-report=html

# Run specific test
uv run pytest tests/test_freeze_strategy.py::TestFreezeBackboneStrategy::test_freeze_backbone_only_trains_heads -v
```

### Test Coverage

The freeze strategy implementation includes 10 comprehensive tests:

- **Basic Functionality** (3 tests): Default behavior, backward compatibility, error handling
- **Freeze Backbone Strategy** (3 tests): Layer selection, module state, gradient flow
- **Parameter Counting** (2 tests): Trainable parameter validation, helper utilities
- **Optimizer Integration** (2 tests): Training step validation, optimizer configuration

**Coverage**: 100% of freeze strategy code (`ObjectFormer/utils/model_helpers.py`)

## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{wang2022objectformer,
  title={Objectformer for image manipulation detection and localization},
  author={Wang, Junke and Wu, Zuxuan and Chen, Jingjing and Han, Xintong and Shrivastava, Abhinav and Lim, Ser-Nam and Jiang, Yu-Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

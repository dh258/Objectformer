# ObjectFormer Training Setup To-Do List

## 1. Dataset Preparation

### Minimum Test Dataset (for verification):
- **20-30 authentic document images** (JPG/PNG format)
- **20-30 tampered document images** with corresponding **pixel-level masks** (grayscale PNG, 0=authentic, 255=tampered)

### Dataset Structure:
```
your_dataset/
├── images/
│   ├── doc001.jpg
│   ├── doc002.jpg
│   └── ...
├── masks/
│   ├── doc001_mask.png  (for tampered images only)
│   ├── doc002_mask.png
│   └── ...
├── train_split.txt
├── val_split.txt
└── test_split.txt
```

### Annotation Files (tab-separated):
**Format**: `image_path\tmask_path\tlabel`
- Label: `0` = authentic, `1` = tampered
- Example:
```
images/doc001.jpg	masks/doc001_mask.png	0
images/doc002.jpg	masks/doc002_mask.png	1
```

## 2. Configuration Setup

### Update config file (`configs/your_config.yaml`):
```yaml
DATASET:
  ROOT_DIR: /path/to/your_dataset
  TRAIN_SPLIT: train_split.txt
  VAL_SPLIT: val_split.txt
  TEST_SPLIT: test_split.txt
  RETURN_MASK: True

DATALOADER:
  BATCH_SIZE: 4  # Small for test run
  NUM_WORKERS: 1

TRAIN:
  MAX_EPOCH: 5   # Few epochs for test
```

## 3. Environment Setup

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Verify CUDA availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. Test Run Verification

### Run training test:
```bash
python run.py --config configs/your_config.yaml
```

### Check for:
- [ ] Dataset loads without errors
- [ ] Model trains for 1-2 epochs
- [ ] Loss values decrease
- [ ] No CUDA memory errors
- [ ] Checkpoints save correctly

## 5. Tampering Generation (Optional)

### For automatic dataset expansion:
- [ ] Adapt document-specific manipulation techniques
- [ ] Generate pixel-level masks for tampered regions
- [ ] Validate generated samples manually

## 6. Full Training Preparation

### Scale up after successful test:
- **Recommended minimum**: 500+ authentic + 500+ tampered images
- Increase batch size based on GPU memory
- Set MAX_EPOCH to 40+ for full training
- Monitor validation metrics

## Important Notes

- **Test First**: Start with the 20-30 sample dataset to ensure everything works before scaling up!
- **Annotation Format**: Must be tab-separated, not space-separated
- **Mask Requirements**: Grayscale images with 0=authentic pixels, 255=tampered pixels
- **File Paths**: All paths in annotation files should be relative to ROOT_DIR

## Dataset Analysis Findings

### Current Pipeline:
- **Dataset Class**: `TamperingDataset` (ObjectFormer/datasets/dataset.py:12)
- **Annotation Format**: Tab-separated: `image_path\tmask_path\tlabel`
- **Binary Classification**: 0=authentic, 1=tampered
- **Mask Handling**: Zero masks for authentic, loaded masks for tampered
- **Loss Functions**: BCELossWithLogits (detection) + DiceLoss (localization)

### Model Outputs:
- **Detection**: Binary classification (tampered/authentic)
- **Localization**: Pixel-level tampering masks

The dataset is designed for defensive tampering detection, focusing on identifying and localizing manipulated regions in images.
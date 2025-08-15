# ObjectFormer Document Tampering Training Progress

## Problem Statement
Optimizing ObjectFormer for document tampering detection with focus on improving pixel-level F1 performance from initial score of 0.415.

## Key Context
- **Domain**: Document tampering detection (text edits, signature forgeries, photo replacements)
- **Challenge**: High class imbalance - tampered pixels typically <5% of image
- **Target**: Pixel F1 score >0.60 for production use

## Optimization Journey

### 1. Learning Rate Tuning
**Problem**: Training oscillations preventing convergence
- **Initial**: BASE_LR: 0.0001, MIN_LR: 0.000001 (100x ratio)
- **Issue**: Oscillating graphs, ineffective late-stage learning
- **Solution**: BASE_LR: 0.00004, MIN_LR: 0.000006 (5x ratio)
- **Result**: Stable training with meaningful updates throughout epochs

### 2. Resolution Strategy Evolution
**Progression**:
- **384px**: Baseline training (F1: 0.415, plateau at 0.45)
- **640px**: High detail capture (F1: 0.481, but 2-3x slower training)
- **512px**: Current optimal balance (efficiency + detail capture)

**Insight**: 640px broke F1=0.45 plateau but diminishing returns vs training time

### 3. Loss Function Configuration
**Current Stable Settings**:
```yaml
LOSS:
  LAMBDA_CLS: 0.3     # Detection weight
  LAMBDA_SEG: 2.4     # Localization weight (8:1 ratio)
  LOCALIZATION_LOSS:
    NAME: DiceLoss
    smooth: 0.5
    CLASS_WEIGHT: [0.2, 2.0]  # Background vs tampered pixel emphasis
```

### 4. Training Configuration
**Current Settings**:
- **Batch Size**: 4 (reduced from 8 for memory efficiency)
- **Image Size**: 512px training and testing
- **Optimizer**: AdamW with weight_decay: 0.001
- **Scheduler**: Cosine with warmup

## Current Performance
- **Pixel F1**: 0.481 (+0.066 improvement from 0.415)
- **Image F1**: 0.885 (excellent detection performance)
- **Pixel AUC**: 0.835 (strong localization capability)
- **Image AUC**: 0.950 (very strong detection)

## Key Technical Insights

### 1. Document Domain Characteristics
- Small tampered regions create extreme pixel imbalance
- Fine details (text, signatures) require sufficient resolution
- Class weights in DiceLoss essential for pixel-level learning

### 2. Learning Rate Sensitivity
- Document tampering highly sensitive to LR - too high causes oscillations
- MIN_LR ratio critical for late-stage convergence
- 5x ratio (BASE_LR:MIN_LR) maintains meaningful updates

### 3. Resolution Trade-offs
- 512px provides good balance of detail capture vs training efficiency
- 640px shows diminishing returns despite breaking plateau
- Multi-resolution strategy viable (train high, infer lower)

## Current Configuration File
`configs/objectformer_doc_tamper_conservative.yaml`

Key parameters:
```yaml
DATALOADER:
  BATCH_SIZE: 4

OPTIMIZER:
  BASE_LR: 0.00004
SCHEDULER:
  MIN_LR: 0.000006

DATASET:
  IMG_SIZE: 512
  RESIZE_PARAMS: [564, 564]
  RANDOMCROP_PARAMS: [512, 512]
  TEST_AUGMENTATIONS:
    RESIZE_PARAMS: [512, 512]
```

## Next Steps for Further Improvement

### Immediate Options:
1. **Loss Rebalancing**: Increase LAMBDA_SEG to 3.2+ (16:1 ratio) for stronger localization emphasis
2. **Class Weight Tuning**: Adjust DiceLoss CLASS_WEIGHT to [0.15, 2.5] for stronger tampered pixel focus
3. **Training Extension**: Increase MAX_EPOCH beyond 50 for full convergence

### Advanced Strategies:
1. **Focal Loss**: Consider FocalLoss for localization to better handle extreme imbalance
2. **Multi-scale Training**: Random scale augmentation for robustness
3. **Ensemble Methods**: Multiple model ensemble for production deployment

## Lessons Learned
1. **Conservative tuning first**: Stable training baseline before aggressive optimization
2. **Resolution matters**: Higher resolution helps with fine-grained tampering detection
3. **LR scheduling critical**: Document domain very sensitive to learning rate dynamics
4. **Incremental changes**: Test one parameter at a time to isolate effects

## Status
Training stable at 512px with F1=0.481. Ready for next optimization phase to push toward F1>0.55 target.
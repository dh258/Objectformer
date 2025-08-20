# Pixel F1 Improvement Strategy for ObjectFormer

## Problem Analysis

### Current Performance
- **Image F1**: Excellent (>0.9) - Model successfully detects tampered vs. authentic images
- **Pixel F1**: Poor (<0.500) - Model struggles with precise tamper localization
- **Gap indicates**: Good global tamper detection, poor boundary precision

### Root Cause Identification
The primary bottleneck is **RandomCrop augmentation** in the training pipeline, which creates spatial misalignment between images and ground truth masks.

## Why RandomCrop is Particularly Harmful for Tamper Localization

### 1. Pixel-Level Correspondence Loss
- **Training Process**: Resize to [434, 434] → RandomCrop to [380, 380]
- **Problem**: Exact tampered pixels may be partially cropped or repositioned
- **Result**: Model learns approximate tamper locations, not precise boundaries
- **Impact**: Creates the observed 0.500 pixel F1 ceiling

### 2. Tamper Pattern Fragmentation
- Tampered regions often have coherent spatial patterns (copy-paste boundaries, blend artifacts)
- RandomCrop breaks these patterns into fragments
- Model learns to detect fragments rather than complete tamper signatures

### 3. Context Dependency for Document Tampering
Document tampering requires:
- **Global consistency**: Font matching, text alignment, background uniformity
- **Relative positioning**: Signature placement, watermark locations, text flow
- **Full document context**: OCR inconsistencies, layout anomalies spanning entire page

RandomCrop destroys this global context, making the model "myopic."

### 4. Training-Inference Mismatch
- **Training**: Random 380×380 crops from 434×434 images
- **Inference**: Full 380×380 images
- **Problem**: Model never learns to process complete images during training

## Proposed Solutions

### Solution 1: Remove RandomCrop (Primary Fix)
**Confidence**: 80% | **Expected Improvement**: 10-20% pixel F1

#### Implementation
```yaml
# Current (problematic)
TRAIN_AUGMENTATIONS:
  COMPOSE: [
    [...],
    Resize,        # [434, 434]
    RandomCrop,    # [380, 380] ← REMOVE THIS
    Normalize
  ]

# Proposed (fixed)
TRAIN_AUGMENTATIONS:
  COMPOSE: [
    [...],
    Resize,        # [380, 380] - direct resize
    Normalize
  ]
  RESIZE_PARAMS: [380, 380]  # Direct resize to target size
```

#### Rationale
- Maintains perfect pixel-to-pixel correspondence between images and masks
- Preserves complete spatial context for forensic analysis
- Eliminates training-inference distribution mismatch

### Solution 2: Multi-Scale Inference
**Confidence**: 60-70% | **Expected Improvement**: 3-10% pixel F1

#### Implementation
```python
def multi_scale_inference(model, image, scales=[0.8, 1.0, 1.2]):
    predictions = []
    for scale in scales:
        # Resize input
        h, w = int(380 * scale), int(380 * scale)
        scaled_img = F.interpolate(image, size=(h, w), mode='bilinear')
        
        # Predict
        pred = model(scaled_img)
        
        # Resize back to original
        pred = F.interpolate(pred, size=(380, 380), mode='bilinear')
        predictions.append(pred)
    
    return torch.mean(torch.stack(predictions), dim=0)
```

#### Benefits
- Captures tamper features at multiple resolutions
- Helps detect both fine details and broader tampered regions
- Improves robustness to scale variations

### Solution 3: Post-Processing Pipeline
**Confidence**: 70-80% | **Expected Improvement**: 5-15% pixel F1

#### Implementation
```python
def refine_mask(mask_pred, min_area=100):
    # Convert to binary
    mask = (mask_pred > 0.5).astype(np.uint8)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill holes
    
    # Remove small components
    components = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, components[0]):
        if components[2][i, cv2.CC_STAT_AREA] < min_area:
            mask[components[1] == i] = 0
            
    return mask
```

#### Benefits
- Removes prediction noise and artifacts
- Fills small holes in tampered regions
- Eliminates spurious small detections
- Provides cleaner, more coherent masks

## Alternative Generalization Strategies

To maintain generalization benefits without spatial misalignment:

### 1. Multi-Scale Training
```yaml
# Train with multiple fixed scales instead of random crops
SCALES: [320, 380, 440]
```

### 2. Aspect Ratio Variation
```yaml
# Vary aspect ratios while maintaining full image context
ASPECT_RATIOS: [0.8, 1.0, 1.2]
```

### 3. Elastic Deformation
```yaml
# Slight geometric warping that preserves spatial relationships
ELASTIC_TRANSFORM: True
ELASTIC_ALPHA: 50
ELASTIC_SIGMA: 5
```

## Expected Combined Results

### Individual Improvements
- **RandomCrop removal**: +10-20% pixel F1
- **Multi-scale inference**: +3-10% pixel F1  
- **Post-processing**: +5-15% pixel F1

### Combined Impact
**Expected total improvement**: 15-35% pixel F1
**Target performance**: 0.650-0.700 pixel F1 (up from <0.500)

## Implementation Priority

1. **Immediate**: Remove RandomCrop from training config
2. **Short-term**: Implement post-processing pipeline
3. **Medium-term**: Add multi-scale inference capability
4. **Long-term**: Explore alternative generalization strategies

## Validation Approach

1. **Baseline**: Measure current pixel F1 performance
2. **Ablation**: Test each solution individually
3. **Combined**: Evaluate all solutions together
4. **Analysis**: Compare failure cases before/after improvements

This strategy addresses the fundamental spatial alignment issue while providing complementary improvements for robust tamper localization in document forensics applications.
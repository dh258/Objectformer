# ObjectFormer Post-Training Model Conversion Plan

## Overview

This document outlines the comprehensive plan for converting trained ObjectFormer checkpoints into various formats suitable for deployment, transfer learning, and inference optimization.

## Current Checkpoint Format Analysis

### Training Checkpoints (.pyth)
ObjectFormer currently saves training checkpoints with `.pyth` extension containing:
- Model state dict (`model_state`)
- Optimizer state (`optimizer_state`) 
- Scheduler state (`scheduler_state`)
- Mixed precision scaler state (`scaler_state`)
- Current epoch (`epoch`)
- Full configuration (`cfg`)

### Pretrained Models (.pth)
Existing pretrained models (e.g., `ckpt/imagenet_pre.pth`) contain:
- Clean model state dict only
- Minimal metadata

## Conversion Strategy

### Phase 1: Pretrained Format Conversion
**Goal:** Convert training checkpoint to clean pretrained format for future training

#### Tasks:
1. **Extract Model Weights Only**
   - Remove optimizer/scheduler states from checkpoint
   - Keep only the `model_state` dictionary
   - Validate weight shapes and parameter counts

2. **Save as Standard .pth Format**
   - Follow PyTorch conventions for pretrained models
   - Ensure compatibility with existing loading logic in `ObjectFormer/utils/checkpoint.py:102`

3. **Add Metadata**
   - Model architecture information (layers, channels, heads)
   - Training dataset details
   - Performance metrics (accuracy, F1, AUC)
   - Training configuration summary

4. **Validation**
   - Test loading with existing `load_checkpoint()` function
   - Verify model weights load correctly
   - Ensure no shape mismatches

#### Implementation:
```python
# Example structure for pretrained conversion
def convert_training_to_pretrained(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path)
    
    pretrained_model = {
        'model_state': checkpoint['model_state'],
        'metadata': {
            'architecture': 'ObjectFormer',
            'input_size': [224, 224],
            'num_classes': 2,  # binary classification
            'training_dataset': 'custom_tampering_dataset',
            'performance': {
                'image_accuracy': 0.xx,
                'pixel_f1': 0.xx,
                'image_auc': 0.xx
            }
        }
    }
    
    torch.save(pretrained_model, output_path)
```

### Phase 2: ONNX Export Pipeline
**Goal:** Create inference-ready ONNX models for deployment

#### Tasks:
1. **Model Preparation**
   - Switch model to evaluation mode
   - Handle dynamic input shapes
   - Remove training-specific layers/operations

2. **Input/Output Specification**
   - Define expected tensor shapes: `[batch, 3, 224, 224]`
   - Specify output formats:
     - Detection head: `[batch, 1]` (binary probability)
     - Localization head: `[batch, 1, H, W]` (pixel masks)

3. **Export Process**
   - Use `torch.onnx.export()` with proper operator versions
   - Ensure transformer layers are ONNX-compatible
   - Handle multi-head attention exports correctly

4. **Optimization Passes**
   - Apply ONNX Runtime optimizations
   - Quantization options (FP16, INT8)
   - Graph simplification

5. **Validation**
   - Compare PyTorch vs ONNX outputs
   - Numerical precision checks
   - Performance benchmarking

#### Export Variants:
- **Detection Only**: Binary classification model (smaller, faster)
- **Full Model**: Detection + localization (complete functionality)
- **Quantized Versions**: FP16/INT8 for edge deployment

### Phase 3: Deployment Assets
**Goal:** Complete inference package with utilities

#### Tasks:
1. **Inference Wrapper**
   - Simple API for image tampering detection
   - Preprocessing pipeline (resize, normalize)
   - Postprocessing (threshold application, mask generation)

2. **Performance Benchmarking**
   - Compare formats across hardware (CPU, GPU, mobile)
   - Memory usage analysis
   - Inference speed measurements

3. **Documentation**
   - Usage examples and tutorials
   - Performance characteristics
   - Deployment guides for different platforms

## Recommended File Structure

```
exports/
├── pretrained/
│   ├── objectformer_pretrained.pth          # Clean weights for transfer learning
│   ├── objectformer_metadata.json           # Model specifications
│   └── validation_report.txt                # Loading validation results
├── onnx/
│   ├── objectformer_detection.onnx          # Binary classification only
│   ├── objectformer_full.onnx               # Detection + localization  
│   ├── objectformer_detection_fp16.onnx     # Quantized detection model
│   ├── objectformer_full_fp16.onnx          # Quantized full model
│   └── model_specs.json                     # Input/output specifications
├── inference/
│   ├── inference_wrapper.py                 # Simple API wrapper
│   ├── preprocessing.py                     # Image preprocessing utilities
│   ├── postprocessing.py                    # Result processing utilities
│   └── benchmarks.json                      # Performance measurements
└── docs/
    ├── usage_guide.md                       # How to use exported models
    ├── performance_comparison.md             # Speed/accuracy trade-offs
    └── deployment_examples/                  # Platform-specific guides
        ├── pytorch_inference.py
        ├── onnx_inference.py
        └── mobile_deployment.md
```

## Implementation Priority

### High Priority (Phase 1)
1. ✅ Analyze current checkpoint format
2. 🔄 Create training-to-pretrained conversion script
3. 🔄 Implement validation testing

### Medium Priority (Phase 2) 
4. 🔄 Design ONNX export pipeline
5. 🔄 Create detection-only variant
6. 🔄 Implement full model export

### Low Priority (Phase 3)
7. 🔄 Build inference wrapper utilities
8. 🔄 Performance benchmarking
9. 🔄 Complete documentation

## Technical Considerations

### ONNX Compatibility
- Transformer attention mechanisms require ONNX opset 11+
- Dynamic shapes may need explicit handling
- Some PyTorch operations may need custom ONNX operators

### Performance Trade-offs
- **Pretrained (.pth)**: Best for transfer learning, PyTorch ecosystem
- **ONNX Full**: Cross-platform, good performance, larger size
- **ONNX Detection**: Fastest inference, detection only
- **Quantized Models**: Smallest size, slight accuracy loss

### Security Considerations
This conversion pipeline supports **defensive security applications**:
- Image manipulation detection
- Forensic analysis tools
- Content verification systems
- Anti-disinformation technology

## Next Steps

1. Implement Phase 1 pretrained conversion
2. Test with existing training checkpoints
3. Validate loading compatibility
4. Begin ONNX export development
5. Create comprehensive testing suite

---

*This plan supports defensive cybersecurity applications for detecting image manipulation and forgery.*
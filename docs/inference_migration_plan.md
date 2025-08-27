# ObjectFormer ONNX Inference Migration Plan

## Overview

This document outlines the plan to migrate the ObjectFormer ONNX inference implementation from this training repository to a specialized inference repository. The migration involves adapting the current inference script to work within an existing inference framework.

## Current Implementation Analysis

### ObjectFormer ONNX Inference Components

**Core Files:**
- `exports/inference/test_inference.py` - Main inference script with CLI interface
- `exports/inference/onnx_inference_wrapper.py` - ONNX model wrapper class
- `exports/onnx/objectformer_full_verihubs.onnx` - Exported ONNX model file
- `exports/pretrained/objectformer_full_verihubs_metadata.json` - Model metadata

**Key Features:**
1. **Model Loading**: `ObjectFormerONNXInference` class handles ONNX model initialization
2. **Input Preprocessing**: 
   - Automatic resizing to 288x288 pixels
   - Normalization and tensor conversion
   - Batch dimension handling
3. **Output Processing**:
   - Sigmoid activation for detection logits
   - Binary classification (tampered/authentic)
   - Pixel-level localization masks
   - Confidence scoring
4. **Visualization**: 
   - Side-by-side comparison (original, detection result, localization heatmap)
   - Tampered area percentage calculation
   - Result export to PNG files

**Model Specifications:**
- **Input**: `[batch_size, 3, 288, 288]` - RGB images
- **Outputs**: 
  - Detection scores (binary classification logits)
  - Localization masks (pixel-level tamper detection)
- **Architecture**: Transformer-based ObjectFormer
- **Task**: Image manipulation detection and localization

## Migration Requirements

### Pre-Migration Questions

**Existing Infrastructure Analysis:**
1. What is the current inference framework structure in the target repo?
2. How are models currently loaded and managed?
3. What input/output formats are expected?
4. Are there existing preprocessing pipelines to integrate with?
5. What dependency management system is used?

**Integration Strategy:**
1. Should ObjectFormer be added as a new model type or replace existing implementation?
2. Is the CLI interface needed or should it integrate with existing APIs?
3. Are visualization features required in the production environment?
4. How should model metadata be handled in the existing system?

**Performance Requirements:**
1. What are the expected inference speed requirements?
2. Should GPU acceleration be supported?
3. Are there memory constraints to consider?
4. Is batch processing needed?

### Migration Components

#### 1. Core Inference Engine
**Current Implementation:**
```python
class ObjectFormerONNXInference:
    def __init__(self, onnx_model_path, providers=['CPUExecutionProvider'])
    def predict(self, image_path, threshold=0.5, return_original_size=True)
    def get_model_info(self)
```

**Migration Needs:**
- Adapt to existing model loading patterns
- Integrate with current preprocessing pipelines
- Align output format with existing systems

#### 2. Preprocessing Pipeline
**Current Features:**
- Image loading from file paths
- Automatic resize to 288x288
- RGB conversion and normalization
- ONNX-compatible tensor formatting

**Migration Considerations:**
- Preserve ObjectFormer-specific requirements (288x288 input size)
- Integrate with existing image handling utilities
- Maintain preprocessing consistency for model accuracy

#### 3. Output Processing
**Current Outputs:**
```python
{
    'is_tampered': bool,
    'detection_confidence': float,
    'raw_detection_logits': float,
    'confidence_mask': numpy.ndarray,
    'binary_mask': numpy.ndarray
}
```

**Migration Needs:**
- Adapt output format to match existing API contracts
- Preserve essential information (classification + localization)
- Handle visualization requirements

#### 4. Model Metadata Handling
**Current Metadata:**
- Architecture parameters (layers, channels, heads, etc.)
- Training information (dataset, epochs, etc.)
- Conversion details (timestamp, version, etc.)

**Integration Requirements:**
- Align with existing model registry/catalog systems
- Preserve version tracking capabilities
- Maintain reproducibility information

## Migration Steps

### Phase 1: Analysis and Planning
1. **Examine Target Repository**
   - Analyze existing inference architecture
   - Identify integration points
   - Document current model loading patterns
   - Review API contracts and data formats

2. **Gap Analysis**
   - Compare preprocessing requirements
   - Identify output format differences
   - Assess dependency conflicts
   - Plan compatibility layers if needed

### Phase 2: Core Integration
1. **Model Wrapper Adaptation**
   - Adapt `ObjectFormerONNXInference` to existing patterns
   - Integrate with current model management system
   - Preserve ObjectFormer-specific requirements

2. **Preprocessing Integration**
   - Integrate ObjectFormer preprocessing with existing pipelines
   - Ensure 288x288 input size requirement is met
   - Maintain preprocessing consistency

3. **Output Format Alignment**
   - Adapt output structure to existing API contracts
   - Preserve classification and localization information
   - Handle backward compatibility if needed

### Phase 3: Testing and Validation
1. **Functionality Testing**
   - Verify model loading and initialization
   - Test preprocessing pipeline accuracy
   - Validate output format compatibility

2. **Accuracy Validation**
   - Compare inference results with original implementation
   - Test on known good/bad samples
   - Verify localization accuracy

3. **Performance Testing**
   - Measure inference speed
   - Test memory usage
   - Validate batch processing if applicable

### Phase 4: Documentation and Deployment
1. **Update Documentation**
   - Document new model integration
   - Update API documentation
   - Provide migration guide for existing users

2. **Deployment Preparation**
   - Package model files appropriately
   - Update dependency requirements
   - Prepare deployment scripts

## Technical Considerations

### Dependencies
**Current Requirements:**
- `onnxruntime` - ONNX model execution
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `torch` - Limited usage for tensor operations

**Migration Considerations:**
- Check compatibility with existing dependency versions
- Minimize new dependencies where possible
- Handle version conflicts gracefully

### Model File Management
**Current Structure:**
```
exports/
├── onnx/objectformer_full_verihubs.onnx           # ONNX model (main)
├── pretrained/objectformer_full_verihubs.pth      # PyTorch weights
└── pretrained/objectformer_full_verihubs_metadata.json  # Metadata
```

**Migration Needs:**
- Adapt to existing model storage patterns
- Preserve metadata accessibility
- Handle model versioning appropriately

### Performance Optimization
**Current Capabilities:**
- CPU execution (default)
- GPU acceleration support (configurable)
- Single image processing
- Dynamic batch sizing

**Optimization Opportunities:**
- Batch processing for multiple images
- Model caching for repeated usage
- Memory optimization for large images
- Provider-specific optimizations

## Success Criteria

### Functional Requirements
- [ ] Successful model loading in target environment
- [ ] Accurate inference results matching original implementation
- [ ] Proper integration with existing preprocessing pipelines
- [ ] Compatible output format with existing APIs

### Performance Requirements
- [ ] Inference speed within acceptable limits
- [ ] Memory usage within constraints
- [ ] Successful handling of various image sizes and formats

### Integration Requirements
- [ ] Seamless integration with existing model management
- [ ] Proper error handling and logging
- [ ] Documentation updated and comprehensive
- [ ] Backward compatibility maintained where applicable

## Next Steps

1. **Share target repository structure** for detailed analysis
2. **Review existing inference implementation** to identify integration patterns
3. **Clarify specific requirements** and constraints
4. **Begin Phase 1 analysis** once target repository is accessible

## Notes

- This migration focuses on the defensive security application of ObjectFormer for image tampering detection
- The model provides both binary classification (tampered/authentic) and pixel-level localization
- Current implementation has been validated on KTP (Indonesian ID card) and document datasets
- Maintain security focus - this is for defensive forensic analysis, not offensive manipulation

---

**Document Version**: 1.0  
**Created**: 2025-08-27  
**Last Updated**: 2025-08-27  
**Status**: Planning Phase
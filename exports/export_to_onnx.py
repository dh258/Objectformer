#!/usr/bin/env python3
"""
ONNX Export Script for ObjectFormer
Exports full model (detection + localization) to ONNX format
"""

import torch
import torch.onnx
import os
import sys
import argparse
from pathlib import Path

# Add ObjectFormer to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ObjectFormer.models.objectformer import ObjectFormer
import json


def export_objectformer_to_onnx(
    weights_path: str,
    output_path: str,
    metadata_path: str = None,
    input_shape: tuple = (1, 3, 288, 288),
    opset_version: int = 17
):
    """
    Export ObjectFormer model to ONNX format
    
    Args:
        weights_path: Path to pretrained model weights (.pth)
        output_path: Output path for ONNX model (full FP32 precision)
        metadata_path: Path to metadata JSON file (optional, auto-detected)
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version (17 recommended for transformers)
    """
    
    print(f"Loading pretrained weights from {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Extract or load metadata
    if 'metadata' in checkpoint:
        print("✓ Found metadata in .pth file")
        metadata = checkpoint['metadata']
    else:
        # Try to load external metadata file
        if metadata_path is None:
            metadata_path = weights_path.replace('.pth', '_metadata.json')
        
        if os.path.exists(metadata_path):
            print(f"✓ Loading metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            raise ValueError(f"No metadata found in .pth file or at {metadata_path}")
    
    # Build model from metadata
    print(f"Building model from metadata...")
    model_config = metadata['model_config']
    
    # Create model configuration dictionary
    num_stages = len(model_config["layers"])
    parts = model_config["parts"]
    
    model_cfg = {
        "PRETRAINED": None,
        "INPLANES": 64,  # Standard ObjectFormer inplanes
        "NUM_LAYERS": model_config["layers"],
        "NUM_CHS": model_config["channels"], 
        "NUM_STRIDES": [1, 2, 2, 2],  # Standard ObjectFormer strides  
        "NUM_HEADS": model_config["heads"],
        "NUM_PARTS": [parts] * num_stages,  # Parts per stage (e.g., [64, 64, 64, 64])
        "PATCH_SIZES": [8, 7, 7, 7],  # Standard ObjectFormer patch sizes
        "DROP_PATH": 0.0,  # No drop path for inference
        "NUM_ENC_HEADS": [1, 3, 6, 12],  # Standard ObjectFormer encoder heads
        "IMG_SIZE": model_config["img_size"],
        "USE_CLASSIFIER": True,
        "NUM_CLASSES": 1  # Binary classification with sigmoid (single output)
    }
    
    print(f"Model config: {model_cfg}")
    model = ObjectFormer(model_cfg)
    
    # Load model weights
    if 'model_state' in checkpoint:
        print("✓ Loading model state from checkpoint")
        model.load_state_dict(checkpoint['model_state'], strict=True)
    else:
        print("✓ Loading model weights directly")
        model.load_state_dict(checkpoint, strict=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    print(f"Creating dummy input with shape {input_shape}")
    dummy_input = torch.randn(input_shape)
    dummy_input_dict = {'img': dummy_input}
    
    # Trace the model to get output shapes
    with torch.no_grad():
        traced_output = model(dummy_input_dict)
        pred, masks = traced_output
        
        print(f"Model outputs:")
        if isinstance(pred, list):
            print(f"  Detection predictions (list): {len(pred)} items")
            for i, p in enumerate(pred):
                print(f"    pred[{i}]: {p.shape}")
        else:
            print(f"  Detection prediction: {pred.shape}")
            
        print(f"  Localization masks (list): {len(masks)} items")
        for i, mask in enumerate(masks):
            print(f"    mask[{i}]: {mask.shape}")
    
    # Define input and output names
    input_names = ['image']
    output_names = ['detection_scores', 'localization_masks']
    
    # Dynamic axes for batch dimension
    dynamic_axes = {
        'image': {0: 'batch_size'},
        'detection_scores': {0: 'batch_size'}, 
        'localization_masks': {0: 'batch_size'}
    }
    
    print(f"Exporting to ONNX format...")
    print(f"Output path: {output_path}")
    
    try:
        # Create a wrapper class to handle dictionary input for ONNX export
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, img_tensor):
                return self.model({'img': img_tensor})
        
        onnx_model = ONNXWrapper(model)
        
        torch.onnx.export(
            onnx_model,
            dummy_input,  # Pass tensor directly, not dictionary
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
            export_params=True
        )
        
        print(f"✅ ONNX export successful!")
        print(f"Model saved to: {output_path}")
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {str(e)}")
        return False


def validate_onnx_model(onnx_path: str, pytorch_model, dummy_input_dict, tolerance: float = 1e-5):
    """
    Validate ONNX model outputs match PyTorch model outputs
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        dummy_input_dict: Dummy input for testing
        tolerance: Numerical tolerance for comparison
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  ONNX Runtime not installed. Skipping validation.")
        return False
    
    print(f"Validating ONNX model...")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get PyTorch outputs
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_pred, pytorch_masks = pytorch_model(dummy_input_dict)
    
    # Prepare ONNX inputs
    onnx_inputs = {ort_session.get_inputs()[0].name: dummy_input_dict['img'].numpy()}
    
    # Get ONNX outputs
    onnx_outputs = ort_session.run(None, onnx_inputs)
    
    print(f"Comparing outputs...")
    
    # Compare detection outputs
    if isinstance(pytorch_pred, list):
        # Handle list of predictions (multi-stage)
        pytorch_pred_tensor = torch.stack(pytorch_pred, dim=1)  # [batch, stages]
    else:
        pytorch_pred_tensor = pytorch_pred
    
    onnx_pred = torch.from_numpy(onnx_outputs[0])
    
    pred_diff = torch.abs(pytorch_pred_tensor - onnx_pred).max().item()
    print(f"Detection prediction max difference: {pred_diff}")
    
    # Compare mask outputs (first mask only for simplicity)
    if len(pytorch_masks) > 0 and len(onnx_outputs) > 1:
        pytorch_mask = pytorch_masks[0]  # First mask
        onnx_mask = torch.from_numpy(onnx_outputs[1])
        
        mask_diff = torch.abs(pytorch_mask - onnx_mask).max().item()
        print(f"Localization mask max difference: {mask_diff}")
        
        if pred_diff < tolerance and mask_diff < tolerance:
            print(f"✅ Validation successful! Outputs match within tolerance ({tolerance})")
            return True
        else:
            print(f"❌ Validation failed! Differences exceed tolerance")
            return False
    else:
        if pred_diff < tolerance:
            print(f"✅ Detection validation successful!")
            return True
        else:
            print(f"❌ Detection validation failed!")
            return False


def main():
    parser = argparse.ArgumentParser(description='Export ObjectFormer to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to pretrained weights (.pth)')
    parser.add_argument('--metadata', help='Path to metadata JSON file (auto-detected if not provided)')
    parser.add_argument('--output', required=True, help='Output path for ONNX model (.onnx)')
    parser.add_argument('--input_size', default=288, type=int, help='Input image size')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--opset', default=17, type=int, help='ONNX opset version (17 recommended for transformers)')
    parser.add_argument('--validate', action='store_true', help='Validate ONNX outputs match PyTorch')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define input shape
    input_shape = (args.batch_size, 3, args.input_size, args.input_size)
    
    # Export to ONNX
    success = export_objectformer_to_onnx(
        weights_path=args.checkpoint,
        output_path=args.output,
        metadata_path=args.metadata,
        input_shape=input_shape,
        opset_version=args.opset
    )
    
    if success and args.validate:
        print("\n" + "="*50)
        print("VALIDATING ONNX MODEL")
        print("="*50)
        
        # Load model for validation (re-use same loading logic)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # Extract metadata
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
        else:
            metadata_path = args.metadata if args.metadata else args.checkpoint.replace('.pth', '_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Build model
        model_config = metadata['model_config']
        num_stages = len(model_config["layers"])
        parts = model_config["parts"]
        
        model_cfg = {
            "PRETRAINED": None,
            "INPLANES": 64,
            "NUM_LAYERS": model_config["layers"],
            "NUM_CHS": model_config["channels"], 
            "NUM_STRIDES": [1, 2, 2, 2],
            "NUM_HEADS": model_config["heads"],
            "NUM_PARTS": [parts] * num_stages,
            "PATCH_SIZES": [8, 7, 7, 7],
            "DROP_PATH": 0.0,
            "NUM_ENC_HEADS": [1, 3, 6, 12],
            "IMG_SIZE": model_config["img_size"],
            "USE_CLASSIFIER": True,
            "NUM_CLASSES": 1  # Binary classification with sigmoid (single output)
        }
        
        model = ObjectFormer(model_cfg)
        
        # Load weights
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        
        dummy_input = torch.randn(input_shape)
        dummy_input_dict = {'img': dummy_input}
        
        validate_onnx_model(args.output, model, dummy_input_dict)


if __name__ == "__main__":
    main()
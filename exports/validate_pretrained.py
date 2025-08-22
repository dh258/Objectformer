#!/usr/bin/env python3
"""
ObjectFormer Pretrained Model Validation Script

This script validates that converted pretrained models can be loaded correctly
by the existing ObjectFormer codebase loading mechanisms.

Usage:
    python validate_pretrained.py --model path/to/pretrained_model.pth --config path/to/config.yaml
"""

import argparse
import json
import os
import sys
import torch
from datetime import datetime

# Add ObjectFormer to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from ObjectFormer.utils.checkpoint import load_checkpoint
    from ObjectFormer.utils.build_helper import build_model
    from ObjectFormer.config.defaults import get_cfg
    OBJECTFORMER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ObjectFormer modules: {e}")
    print("Falling back to basic validation...")
    OBJECTFORMER_AVAILABLE = False


def validate_pretrained_format(model_path):
    """Validate the basic format of a pretrained model file."""
    print(f"Validating pretrained model format: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the pretrained model
    pretrained_model = torch.load(model_path, map_location='cpu')
    
    # Check required keys
    if not isinstance(pretrained_model, dict):
        raise ValueError("Pretrained model should be a dictionary")
    
    if 'model_state' not in pretrained_model:
        raise ValueError("Missing 'model_state' key in pretrained model")
    
    model_state = pretrained_model['model_state']
    
    if not isinstance(model_state, (dict, torch.nn.Module)):
        raise ValueError("'model_state' should be a dictionary or OrderedDict")
    
    # Validate key model components
    required_keys = [
        'obj_queries',
        'conv1.weight',
        'norm1.weight',
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in model_state:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"⚠️  Warning: Missing keys in model_state: {missing_keys}")
    else:
        print("✓ All required model keys present")
    
    # Count parameters
    total_params = sum(p.numel() for p in model_state.values() if hasattr(p, 'numel'))
    print(f"✓ Total model parameters: {total_params:,}")
    
    # Check metadata
    if 'metadata' in pretrained_model:
        metadata = pretrained_model['metadata']
        print("✓ Metadata present")
        print(f"  - Architecture: {metadata.get('architecture', 'Unknown')}")
        print(f"  - Input size: {metadata.get('input_size', 'Unknown')}")
        print(f"  - Training epoch: {metadata.get('training_info', {}).get('epoch', 'Unknown')}")
    else:
        print("⚠️  No metadata found in pretrained model")
    
    return True


def validate_with_objectformer_loader(model_path, config_path=None):
    """Validate using ObjectFormer's loading mechanism."""
    if not OBJECTFORMER_AVAILABLE:
        print("⚠️  Skipping ObjectFormer loader validation (modules not available)")
        return False
    
    print(f"Validating with ObjectFormer loader...")
    
    try:
        # Load config 
        if config_path and os.path.exists(config_path):
            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            print(f"✓ Loaded config from: {config_path}")
        else:
            # Use default config
            cfg = get_cfg()
            print("✓ Using default config")
        
        # Build model
        model = build_model(cfg)
        print("✓ Built model successfully")
        
        # Try to load the pretrained model using ObjectFormer's loader
        # Note: This simulates loading in test mode
        print("Testing pretrained model loading...")
        
        # Backup original test checkpoint path
        original_path = cfg.get('TEST', {}).get('CHECKPOINT_PATH', '')
        
        # Temporarily set our model path
        if 'TEST' not in cfg:
            cfg['TEST'] = {}
        cfg['TEST']['CHECKPOINT_PATH'] = model_path
        
        # Try loading with ObjectFormer's mechanism
        try:
            epoch = load_checkpoint(
                model_path,
                model,
                data_parallel=False,
                optimizer=None,
                scheduler=None,
                scaler=None,
                epoch_reset=True
            )
            print(f"✓ Successfully loaded with ObjectFormer loader (epoch: {epoch})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load with ObjectFormer loader: {e}")
            return False
        
        finally:
            # Restore original path
            cfg['TEST']['CHECKPOINT_PATH'] = original_path
    
    except Exception as e:
        print(f"❌ ObjectFormer validation failed: {e}")
        return False


def create_validation_report(model_path, results):
    """Create a validation report."""
    report = {
        'model_path': model_path,
        'validation_timestamp': datetime.now().isoformat(),
        'results': results,
        'status': 'PASSED' if all(results.values()) else 'FAILED'
    }
    
    report_path = model_path.replace('.pth', '_validation_report.json')
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✓ Validation report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Validate ObjectFormer pretrained model')
    parser.add_argument('--model', '-m', required=True,
                       help='Path to pretrained model (.pth file)')
    parser.add_argument('--config', '-c', 
                       help='Path to config file (optional)')
    parser.add_argument('--create-report', action='store_true',
                       help='Create validation report file')
    
    args = parser.parse_args()
    
    print("🔍 ObjectFormer Pretrained Model Validation")
    print("=" * 50)
    
    results = {}
    
    try:
        # Basic format validation
        print("\n1. Basic Format Validation")
        results['format_validation'] = validate_pretrained_format(args.model)
        print("✓ Format validation passed\n")
        
    except Exception as e:
        print(f"❌ Format validation failed: {e}\n")
        results['format_validation'] = False
    
    try:
        # ObjectFormer loader validation
        print("2. ObjectFormer Loader Validation")
        results['loader_validation'] = validate_with_objectformer_loader(args.model, args.config)
        print()
        
    except Exception as e:
        print(f"❌ Loader validation failed: {e}\n")
        results['loader_validation'] = False
    
    # Summary
    print("📋 Validation Summary")
    print("-" * 30)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test}: {status}")
    
    overall_status = all(results.values())
    print(f"\nOverall Status: {'✓ PASSED' if overall_status else '❌ FAILED'}")
    
    # Create report if requested
    if args.create_report:
        create_validation_report(args.model, results)
    
    return 0 if overall_status else 1


if __name__ == '__main__':
    exit(main())
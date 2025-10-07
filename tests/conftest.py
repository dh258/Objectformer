"""
Pytest configuration and shared fixtures for ObjectFormer tests.

This module provides reusable fixtures for testing the ObjectFormer
model and related components.
"""

import pytest
import torch


@pytest.fixture
def minimal_model_config():
    """
    Minimal model configuration for fast testing.

    Returns a configuration dict for ObjectFormer with minimal parameters
    to speed up test execution while maintaining architectural integrity.

    Returns:
        dict: Model configuration dictionary
    """
    return {
        'MODEL_NAME': 'ObjectFormer',
        'PRETRAINED': '',  # No pretrained weights for testing
        'INPLANES': 64,
        'NUM_LAYERS': [1, 1, 3, 1],
        'NUM_CHS': [96, 192, 384, 768],
        'NUM_STRIDES': [1, 2, 2, 2],
        'NUM_HEADS': [3, 6, 12, 24],
        'NUM_PARTS': [64, 64, 64, 64],
        'PATCH_SIZES': [8, 7, 7, 7],
        'DROP_PATH': 0.1,
        'NUM_ENC_HEADS': [1, 3, 6, 12],
        'NUM_CLASSES': 1,
        'use_classifier': True,
        'FIND_UNUSED_PARAMETERS': False,
    }


@pytest.fixture
def minimal_full_config(minimal_model_config):
    """
    Full minimal configuration including optimizer and training settings.

    Returns:
        dict: Complete configuration dictionary for testing training pipeline
    """
    return {
        'NUM_GPUS': 1,
        'RNG_SEED': 42,
        'MODEL': minimal_model_config,
        'OPTIMIZER': {
            'OPTIMIZER_METHOD': 'adamw',
            'BASE_LR': 0.001,
            'ADAM_BETAS': [0.9, 0.999],
            'EPS': 1e-8,
            'WEIGHT_DECAY': 0.01,
            'FREEZE_STRATEGY': 'none',  # Default strategy
        },
        'TRAIN': {
            'ENABLE': True,
            'MAX_EPOCH': 5,
            'AMP_ENABLE': False,
        },
    }


@pytest.fixture
def freeze_backbone_config(minimal_full_config):
    """
    Configuration with freeze_backbone strategy enabled.

    Returns:
        dict: Configuration with backbone freezing
    """
    config = minimal_full_config.copy()
    config['OPTIMIZER']['FREEZE_STRATEGY'] = 'freeze_backbone'
    return config


@pytest.fixture
def model_instance(minimal_model_config):
    """
    Create a fresh ObjectFormer model instance for testing.

    Returns:
        ObjectFormer: Instantiated model (on CPU for testing)
    """
    from ObjectFormer.models.objectformer import ObjectFormer
    model = ObjectFormer(minimal_model_config)
    return model


@pytest.fixture
def dummy_input():
    """
    Create dummy input tensor for model forward pass testing.

    Returns:
        dict: Dictionary with 'img' key containing dummy tensor
    """
    return {
        'img': torch.randn(1, 3, 320, 320)
    }


@pytest.fixture
def dummy_batch():
    """
    Create dummy batch with multiple samples.

    Returns:
        dict: Dictionary with 'img' key containing batch tensor
    """
    return {
        'img': torch.randn(2, 3, 320, 320)
    }

"""
Tests for layer freezing strategies in ObjectFormer.

This module tests the freeze strategy implementation used for fine-tuning
on small datasets. It validates that:
1. The correct layers are frozen/unfrozen
2. Frozen parameters don't receive gradients
3. Parameter counts are correct
4. Error handling works properly
"""

import pytest
import torch
from ObjectFormer.utils.model_helpers import (
    apply_freeze_strategy,
    get_trainable_param_count,
    get_frozen_param_count,
)


class TestFreezeStrategyBasics:
    """Test basic freeze strategy functionality."""

    def test_freeze_none_keeps_all_trainable(self, model_instance, minimal_full_config):
        """
        Test that 'none' strategy keeps all parameters trainable.

        This validates backward compatibility - existing configs without
        FREEZE_STRATEGY should work unchanged.
        """
        config = minimal_full_config.copy()
        config['OPTIMIZER']['FREEZE_STRATEGY'] = 'none'

        # Apply freeze strategy
        trainable_params = list(apply_freeze_strategy(model_instance, config))

        # All parameters should be trainable
        total_params = list(model_instance.parameters())
        assert len(trainable_params) == len(total_params)

        # Check all params have requires_grad=True
        for param in model_instance.parameters():
            assert param.requires_grad, "All parameters should be trainable with 'none' strategy"

    def test_freeze_none_is_default(self, model_instance):
        """
        Test that 'none' is the default strategy when not specified.

        This ensures backward compatibility with configs that don't have
        FREEZE_STRATEGY key.
        """
        config = {'OPTIMIZER': {}}  # No FREEZE_STRATEGY key

        trainable_params = list(apply_freeze_strategy(model_instance, config))
        total_params = list(model_instance.parameters())

        assert len(trainable_params) == len(total_params)

    def test_invalid_strategy_raises_error(self, model_instance, minimal_full_config):
        """
        Test that invalid strategy names raise ValueError.

        This validates error handling for typos or unsupported strategies.
        """
        config = minimal_full_config.copy()
        config['OPTIMIZER']['FREEZE_STRATEGY'] = 'invalid_strategy_name'

        with pytest.raises(ValueError, match="Unknown freeze strategy"):
            apply_freeze_strategy(model_instance, config)


class TestFreezeBackboneStrategy:
    """Test the freeze_backbone strategy in detail."""

    def test_freeze_backbone_only_trains_heads(self, model_instance, freeze_backbone_config):
        """
        Test that freeze_backbone only keeps head layers trainable.

        This is the core functionality test - validates that only
        last_fc and localizer heads are trainable.
        """
        trainable_params = list(apply_freeze_strategy(model_instance, freeze_backbone_config))

        # Get names of all trainable parameters
        trainable_names = [
            name for name, param in model_instance.named_parameters()
            if param.requires_grad
        ]

        # Check that expected modules are trainable
        assert any('last_fc' in name for name in trainable_names), \
            "last_fc should be trainable"
        assert any('localizer_0' in name for name in trainable_names), \
            "localizer_0 should be trainable"
        assert any('localizer_1' in name for name in trainable_names), \
            "localizer_1 should be trainable"
        assert any('localizer_2' in name for name in trainable_names), \
            "localizer_2 should be trainable"

        # Check that backbone layers are frozen
        # Note: We need to check for exact match 'conv1.' to avoid matching 'localizer_X.conv1'
        assert not any(name.startswith('conv1.') or name == 'conv1.weight' or name == 'conv1.bias' for name in trainable_names), \
            "conv1 (stem) should be frozen"
        assert not any('layer_0' in name for name in trainable_names), \
            "layer_0 should be frozen"
        assert not any('layer_1' in name for name in trainable_names), \
            "layer_1 should be frozen"
        assert not any('layer_2' in name for name in trainable_names), \
            "layer_2 should be frozen"

    def test_freeze_backbone_correct_modules(self, model_instance, freeze_backbone_config):
        """
        Test that exactly the right modules have requires_grad=True.

        More strict test than the previous one - checks exact module states.
        """
        apply_freeze_strategy(model_instance, freeze_backbone_config)

        # These should be trainable
        assert model_instance.last_fc.weight.requires_grad
        # Localizer is a composite module, check that at least one param is trainable
        assert any(p.requires_grad for p in model_instance.localizer_0.parameters())
        assert any(p.requires_grad for p in model_instance.localizer_1.parameters())
        assert any(p.requires_grad for p in model_instance.localizer_2.parameters())

        # These should be frozen
        assert not model_instance.conv1.weight.requires_grad
        assert not model_instance.norm1.weight.requires_grad

    def test_frozen_params_no_gradients(self, model_instance, freeze_backbone_config):
        """
        Test that frozen parameters don't accumulate gradients during backprop.

        This is critical - frozen layers should not be updated during training.
        """
        apply_freeze_strategy(model_instance, freeze_backbone_config)

        # Get initial weights of a frozen layer
        initial_conv1_weight = model_instance.conv1.weight.clone()

        # Forward pass
        dummy_input = {'img': torch.randn(1, 3, 320, 320)}
        detection, masks = model_instance(dummy_input)

        # Create dummy loss and backward
        loss = detection.sum() + sum(m.sum() for m in masks)
        loss.backward()

        # Check frozen layer has no gradient
        assert model_instance.conv1.weight.grad is None, \
            "Frozen layer should not have gradient"

        # Check trainable layer has gradient
        assert model_instance.last_fc.weight.grad is not None, \
            "Trainable layer should have gradient"


class TestParameterCounts:
    """Test parameter counting utilities."""

    def test_trainable_param_count(self, model_instance, freeze_backbone_config):
        """
        Test that trainable parameter count is significantly reduced.

        With freeze_backbone, we expect ~5-10% of parameters to be trainable.
        """
        # Get initial count (all trainable)
        initial_trainable = get_trainable_param_count(model_instance)

        # Apply freeze strategy
        apply_freeze_strategy(model_instance, freeze_backbone_config)

        # Get new count
        final_trainable = get_trainable_param_count(model_instance)
        final_frozen = get_frozen_param_count(model_instance)

        # Sanity checks
        assert final_trainable + final_frozen == initial_trainable, \
            "Total params should be conserved"

        # Check that we froze a significant portion
        assert final_trainable < initial_trainable * 0.2, \
            f"Expected <20% trainable, got {final_trainable/initial_trainable*100:.1f}%"

        assert final_frozen > initial_trainable * 0.8, \
            f"Expected >80% frozen, got {final_frozen/initial_trainable*100:.1f}%"

    def test_param_count_utilities(self, model_instance):
        """
        Test parameter counting utility functions.

        Validates that helper functions correctly count trainable/frozen params.
        """
        total = sum(p.numel() for p in model_instance.parameters())
        trainable = get_trainable_param_count(model_instance)
        frozen = get_frozen_param_count(model_instance)

        # Initially all should be trainable
        assert trainable == total
        assert frozen == 0

        # Manually freeze one layer
        for param in model_instance.conv1.parameters():
            param.requires_grad = False

        trainable_after = get_trainable_param_count(model_instance)
        frozen_after = get_frozen_param_count(model_instance)

        assert trainable_after < trainable
        assert frozen_after > frozen
        assert trainable_after + frozen_after == total


class TestIntegrationWithOptimizer:
    """Test integration with PyTorch optimizers."""

    def test_optimizer_only_updates_trainable(self, model_instance, freeze_backbone_config):
        """
        Test that optimizer only updates trainable parameters after a training step.

        This simulates a real training iteration and validates that frozen
        parameters remain unchanged.
        """
        # Apply freeze strategy and create optimizer
        trainable_params = apply_freeze_strategy(model_instance, freeze_backbone_config)
        optimizer = torch.optim.Adam(trainable_params, lr=0.001)

        # Store initial weights
        initial_conv1_weight = model_instance.conv1.weight.clone()
        initial_fc_weight = model_instance.last_fc.weight.clone()

        # Training step
        dummy_input = {'img': torch.randn(1, 3, 320, 320)}
        detection, masks = model_instance(dummy_input)
        loss = detection.sum() + sum(m.sum() for m in masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check frozen layer didn't change
        assert torch.allclose(model_instance.conv1.weight, initial_conv1_weight), \
            "Frozen layer should not be updated"

        # Check trainable layer changed
        assert not torch.allclose(model_instance.last_fc.weight, initial_fc_weight), \
            "Trainable layer should be updated"

    def test_optimizer_param_groups(self, model_instance, freeze_backbone_config):
        """
        Test that optimizer param_groups contains only trainable parameters.

        This ensures the optimizer is correctly configured.
        """
        trainable_params = list(apply_freeze_strategy(model_instance, freeze_backbone_config))
        optimizer = torch.optim.Adam(trainable_params, lr=0.001)

        # Count params in optimizer
        optimizer_param_count = sum(
            p.numel() for group in optimizer.param_groups
            for p in group['params']
        )

        # Should match trainable count
        trainable_count = get_trainable_param_count(model_instance)
        assert optimizer_param_count == trainable_count, \
            "Optimizer should contain exactly the trainable parameters"

import torch
import ObjectFormer.utils.logging as logging

logger = logging.get_logger(__name__)


def apply_freeze_strategy(model, cfg):
    """
    Apply parameter freezing based on config.

    This function implements layer freezing strategies for fine-tuning scenarios.
    It supports freezing backbone layers while keeping task-specific heads trainable.

    Args:
        model: Model instance (potentially DDP-wrapped)
        cfg: Config dict containing OPTIMIZER.FREEZE_STRATEGY

    Returns:
        Iterator of trainable parameters

    Raises:
        ValueError: If unknown freeze strategy is specified

    Supported Strategies:
        - 'none': Train all parameters (default, backward compatible)
        - 'freeze_backbone': Freeze backbone layers, train only classification
                           and localization heads (recommended for small datasets)

    Example:
        >>> cfg = {'OPTIMIZER': {'FREEZE_STRATEGY': 'freeze_backbone'}}
        >>> trainable_params = apply_freeze_strategy(model, cfg)
        >>> optimizer = torch.optim.Adam(trainable_params, lr=0.001)
    """
    strategy = cfg.get('OPTIMIZER', {}).get('FREEZE_STRATEGY', 'none')

    # Unwrap DDP if model is wrapped
    m = model.module if hasattr(model, 'module') else model

    if strategy == 'none':
        logger.info("=" * 80)
        logger.info("Freeze Strategy: none (training all parameters)")
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total trainable parameters: {total_params:,}")
        logger.info("=" * 80)
        return model.parameters()

    elif strategy == 'freeze_backbone':
        logger.info("=" * 80)
        logger.info("Freeze Strategy: freeze_backbone")
        logger.info("Freezing backbone layers (conv stem + stages 0-2)")
        logger.info("Training only: last_fc, localizer_0, localizer_1, localizer_2")

        # Freeze all parameters first
        for param in m.parameters():
            param.requires_grad = False

        # Collect trainable modules
        trainable_modules = []
        trainable_names = []

        # Unfreeze final classifier
        if hasattr(m, 'last_fc'):
            m.last_fc.requires_grad_(True)
            trainable_modules.append(m.last_fc)
            trainable_names.append('last_fc')

        # Unfreeze localizers (mask prediction heads)
        for i in range(3):  # localizer_0, localizer_1, localizer_2
            localizer_name = f'localizer_{i}'
            if hasattr(m, localizer_name):
                localizer = getattr(m, localizer_name)
                localizer.requires_grad_(True)
                trainable_modules.append(localizer)
                trainable_names.append(localizer_name)

        # Collect trainable parameters
        trainable_params = []
        for module in trainable_modules:
            trainable_params.extend(module.parameters())

        # Log statistics
        total_params = sum(p.numel() for p in m.parameters())
        trainable_param_count = sum(p.numel() for p in trainable_params)
        frozen_params = total_params - trainable_param_count
        trainable_percentage = (trainable_param_count / total_params) * 100

        logger.info(f"Trainable modules: {trainable_names}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_param_count:,} ({trainable_percentage:.2f}%)")
        logger.info(f"Frozen parameters: {frozen_params:,} ({100-trainable_percentage:.2f}%)")
        logger.info("=" * 80)

        return iter(trainable_params)

    elif strategy == 'freeze_stages_0_1_2':
        logger.info("=" * 80)
        logger.info("Freeze Strategy: freeze_stages_0_1_2")
        logger.info("Freezing: conv stem + layer_0 + layer_1 + layer_2")
        logger.info("Training: layer_3 + last_fc + localizer_0 + localizer_1 + localizer_2")

        # Freeze all parameters first
        for param in m.parameters():
            param.requires_grad = False

        # Collect trainable modules
        trainable_modules = []
        trainable_names = []

        # Unfreeze layer_3 (final backbone stage)
        if hasattr(m, 'layer_3'):
            m.layer_3.requires_grad_(True)
            trainable_modules.append(m.layer_3)
            trainable_names.append('layer_3')

        # Unfreeze final classifier
        if hasattr(m, 'last_fc'):
            m.last_fc.requires_grad_(True)
            trainable_modules.append(m.last_fc)
            trainable_names.append('last_fc')

        # Unfreeze localizers (mask prediction heads)
        for i in range(3):  # localizer_0, localizer_1, localizer_2
            localizer_name = f'localizer_{i}'
            if hasattr(m, localizer_name):
                localizer = getattr(m, localizer_name)
                localizer.requires_grad_(True)
                trainable_modules.append(localizer)
                trainable_names.append(localizer_name)

        # Collect trainable parameters
        trainable_params = []
        for module in trainable_modules:
            trainable_params.extend(module.parameters())

        # Log statistics
        total_params = sum(p.numel() for p in m.parameters())
        trainable_param_count = sum(p.numel() for p in trainable_params)
        frozen_params = total_params - trainable_param_count
        trainable_percentage = (trainable_param_count / total_params) * 100

        logger.info(f"Trainable modules: {trainable_names}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_param_count:,} ({trainable_percentage:.2f}%)")
        logger.info(f"Frozen parameters: {frozen_params:,} ({100-trainable_percentage:.2f}%)")
        logger.info("=" * 80)

        return iter(trainable_params)

    else:
        raise ValueError(
            f"Unknown freeze strategy: '{strategy}'. "
            f"Supported strategies: 'none', 'freeze_backbone', 'freeze_stages_0_1_2'"
        )


def get_trainable_param_count(model):
    """
    Get count of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_frozen_param_count(model):
    """
    Get count of frozen parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: Number of frozen parameters
    """
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

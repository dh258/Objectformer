import json
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import cv2

from ObjectFormer.utils.checkpoint import load_test_checkpoint
from ObjectFormer.utils.distributed import init_process_group
from ObjectFormer.utils import logging
from ObjectFormer.utils.build_helper import (
    build_dataloader,
    build_dataset,
    build_model,
)
from ObjectFormer.utils.meters import MetricLogger,TopKAccMetric,AucMetric,F1Metric
from tools.utils import parse_args, load_config, launch_func

logger = logging.get_logger(__name__)


def save_comparison_grid(image, gt_mask, pred_mask, save_path, image_idx, gt_label, pred_score, iou_score=None):
    """Save a comparison grid showing original image, ground truth, prediction, and overlay"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Image {image_idx} - GT Label: {gt_label}, Pred Score: {pred_score:.3f}' + 
                (f', IoU: {iou_score:.3f}' if iou_score is not None else ''), fontsize=14)
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth mask
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')
    
    # Predicted mask
    axes[1, 0].imshow(pred_mask, cmap='gray')
    axes[1, 0].set_title('Predicted Mask')
    axes[1, 0].axis('off')
    
    # Overlay visualization
    overlay = image.copy()
    if len(overlay.shape) == 3 and overlay.shape[2] == 3:
        # Create colored overlay: GT in green, Pred in red, Overlap in yellow
        gt_colored = np.zeros_like(overlay)
        pred_colored = np.zeros_like(overlay)
        
        gt_colored[gt_mask > 0.5] = [0, 255, 0]  # Green for GT
        pred_colored[pred_mask > 0.5] = [255, 0, 0]  # Red for prediction
        
        # Combine overlays
        combined_overlay = np.maximum(gt_colored, pred_colored)
        overlap_mask = (gt_mask > 0.5) & (pred_mask > 0.5)
        combined_overlay[overlap_mask] = [255, 255, 0]  # Yellow for overlap
        
        # Blend with original image
        alpha = 0.6
        overlay = cv2.addWeighted(overlay.astype(np.uint8), 1-alpha, 
                                combined_overlay.astype(np.uint8), alpha, 0)
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (GT: Green, Pred: Red, Overlap: Yellow)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_iou(gt_mask, pred_mask, threshold=0.5):
    """Calculate IoU between ground truth and predicted masks"""
    gt_binary = (gt_mask > threshold).astype(np.float32)
    pred_binary = (pred_mask > threshold).astype(np.float32)
    
    intersection = np.sum(gt_binary * pred_binary)
    union = np.sum(np.maximum(gt_binary, pred_binary))
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor(mean).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(image_tensor.device)
    
    # Denormalize
    image = image_tensor * std + mean
    image = torch.clamp(image, 0, 1)
    
    # Convert to numpy and transpose
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    return (image_np * 255).astype(np.uint8)


@torch.no_grad()
def perform_test_with_visualization(
    test_loader, model, cfg, cur_epoch=None, writer=None, mode='Test'
):
    metric_logger = MetricLogger(delimiter='  ')
    header = mode + ':'
    
    # Create output directory
    output_dir = os.path.join(cfg.get('OUTPUT_DIR', 'output'), 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'Saving visualizations to: {output_dir}')

    model.eval()

    il_acc1_metrics = TopKAccMetric(k=1, num_gpus=cfg['NUM_GPUS'])
    il_auc_metrics = AucMetric(cfg['NUM_GPUS'])
    pl_auc_metrics = AucMetric(cfg['NUM_GPUS'])
    il_f1_metrics = F1Metric(cfg['NUM_GPUS'])
    pl_f1_metrics = F1Metric(cfg['NUM_GPUS'])
    
    batch_idx = 0
    total_iou = 0.0
    num_samples = 0

    for samples in metric_logger.log_every(test_loader, 10, header):
        samples = dict(map(lambda sample:(sample[0],sample[1].cuda(non_blocking=True)),samples.items()))
        outputs = model(samples)
        
        # Calculate metrics (same as original test)
        il_acc1_metrics.update(samples['bin_label'], outputs[0])
        il_auc_metrics.update(samples['bin_label'], outputs[0])
        
        pl_auc_metrics.update(samples['mask'].reshape(-1), outputs[1][-1].reshape(-1))
        il_f1_metrics.update(samples['bin_label'], (outputs[0] > 0.5))
        
        mask_f1_thres = cfg["TEST"]["THRES"]
        mask_bin = (outputs[1][-1] > mask_f1_thres).int()
        pl_f1_metrics.update(samples['mask'].reshape(-1), mask_bin.reshape(-1))
        
        # Visualization part
        batch_size = samples['img'].shape[0]
        for i in range(batch_size):
            # Get individual samples
            image_tensor = samples['img'][i]
            gt_mask = samples['mask'][i].cpu().numpy().squeeze()
            pred_mask = outputs[1][-1][i].cpu().numpy().squeeze()
            gt_label = samples['bin_label'][i].cpu().item()
            pred_score = torch.sigmoid(outputs[0][i]).cpu().item()
            
            # Denormalize image for visualization
            image_np = denormalize_image(image_tensor)
            
            # Calculate IoU
            iou = calculate_iou(gt_mask, pred_mask, threshold=mask_f1_thres)
            total_iou += iou
            num_samples += 1
            
            # Save comparison grid
            save_path = os.path.join(output_dir, f'comparison_batch{batch_idx:03d}_sample{i:02d}.png')
            save_comparison_grid(
                image_np, gt_mask, pred_mask, save_path, 
                batch_idx * batch_size + i, gt_label, pred_score, iou
            )
            
        batch_idx += 1

    # Calculate average IoU
    avg_iou = total_iou / num_samples if num_samples > 0 else 0.0

    metric_logger.synchronize_between_processes()
    il_acc1_metrics.synchronize_between_processes()
    il_auc_metrics.synchronize_between_processes()
    pl_auc_metrics.synchronize_between_processes()
    il_f1_metrics.synchronize_between_processes()
    pl_f1_metrics.synchronize_between_processes()

    if writer and cur_epoch is not None:
        writer.add_scalar(tag='Image-level Acc', scalar_value=il_acc1_metrics.acc, global_step=cur_epoch)
        writer.add_scalar(tag='Image-level AUC', scalar_value=il_auc_metrics.auc, global_step=cur_epoch)
        writer.add_scalar(tag='Pixel-level AUC', scalar_value=pl_auc_metrics.auc, global_step=cur_epoch)
        writer.add_scalar(tag='Image-level F1', scalar_value=il_f1_metrics.f1, global_step=cur_epoch)
        writer.add_scalar(tag='Pixel-level F1', scalar_value=pl_f1_metrics.f1, global_step=cur_epoch)
        writer.add_scalar(tag='Average IoU', scalar_value=avg_iou, global_step=cur_epoch)

    
    logger.info(
        f'*** Image-level Acc: {il_acc1_metrics.acc:.3f}'
    )
    logger.info(
        f'*** Image-level Auc: {il_auc_metrics.auc:.3f}  Pixel-level Auc: {pl_auc_metrics.auc:.3f}'
    )
    logger.info(
        f'*** Image-level F1: {il_f1_metrics.f1:.3f}  Pixel-level F1: {pl_f1_metrics.f1:.3f}'
    )
    logger.info(
        f'*** Average IoU: {avg_iou:.3f}'
    )
    logger.info(
        f'*** Visualizations saved to: {output_dir} ({num_samples} images)'
    )


def test_with_visualization(local_rank, cfg):
    init_process_group(local_rank, cfg['NUM_GPUS'])
    np.random.seed(cfg['RNG_SEED'])
    torch.manual_seed(cfg['RNG_SEED'])
    logging.setup_logging(cfg, mode='test')
    logger.info(json.dumps(cfg, indent=4,ensure_ascii=False, sort_keys=False,separators=(',', ':')))

    model = build_model(cfg)
    load_test_checkpoint(cfg, model)
    
    test_dataset = build_dataset('test', cfg)
    test_loader = build_dataloader(test_dataset, 'test', cfg)

    logger.info('Testing model with visualization for {} iterations'.format(len(test_loader)))

    perform_test_with_visualization(test_loader, model, cfg)


def main():
    args = parse_args()
    cfg = load_config(args)
    
    # Force test mode
    cfg['TRAIN']['ENABLE'] = False
    cfg['TEST']['ENABLE'] = True
    
    launch_func(cfg=cfg, func=test_with_visualization)


if __name__ == '__main__':
    main()
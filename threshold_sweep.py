import json
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd

from ObjectFormer.utils.checkpoint import load_test_checkpoint
from ObjectFormer.utils.distributed import init_process_group
from ObjectFormer.utils import logging
from ObjectFormer.utils.build_helper import (
    build_dataloader,
    build_dataset,
    build_model,
)
from ObjectFormer.utils.meters import MetricLogger, TopKAccMetric, AucMetric, F1Metric
from tools.utils import parse_args, load_config, launch_func

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_threshold_sweep(
    test_loader, model, cfg, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
):
    """Perform threshold sweep and return metrics for each threshold"""
    metric_logger = MetricLogger(delimiter="  ")
    header = "Threshold Sweep:"

    model.eval()

    # Collect all predictions and ground truths
    all_bin_labels = []
    all_bin_preds = []
    all_mask_gts = []
    all_mask_preds = []

    logger.info(f"Collecting predictions from {len(test_loader)} batches...")

    for samples in metric_logger.log_every(test_loader, 10, header):
        samples = dict(
            map(
                lambda sample: (sample[0], sample[1].cuda(non_blocking=True)),
                samples.items(),
            )
        )
        outputs = model(samples)

        # Collect binary labels and predictions
        all_bin_labels.append(samples["bin_label"].cpu())
        all_bin_preds.append(outputs[0].cpu())

        # Collect mask ground truths and predictions
        all_mask_gts.append(samples["mask"].cpu())
        all_mask_preds.append(outputs[1][-1].cpu())

    # Concatenate all results
    all_bin_labels = torch.cat(all_bin_labels, dim=0)
    all_bin_preds = torch.cat(all_bin_preds, dim=0)
    all_mask_gts = torch.cat(all_mask_gts, dim=0).reshape(-1)
    all_mask_preds = torch.cat(all_mask_preds, dim=0).reshape(-1)

    logger.info(f"Collected {len(all_bin_labels)} samples, {len(all_mask_gts)} pixels")

    # Calculate image-level metrics (threshold-independent)
    il_acc1_metrics = TopKAccMetric(k=1, num_gpus=1)
    il_auc_metrics = AucMetric(1)
    il_f1_metrics = F1Metric(1)
    pl_auc_metrics = AucMetric(1)

    # Move tensors to GPU to match metric classes' expected device
    all_bin_labels_gpu = all_bin_labels.cuda()
    all_bin_preds_gpu = all_bin_preds.cuda()
    all_mask_gts_gpu = all_mask_gts.cuda()
    all_mask_preds_gpu = all_mask_preds.cuda()

    il_acc1_metrics.update(all_bin_labels_gpu, all_bin_preds_gpu)
    il_auc_metrics.update(all_bin_labels_gpu, all_bin_preds_gpu)
    il_f1_metrics.update(all_bin_labels_gpu, (all_bin_preds_gpu > 0.5))
    pl_auc_metrics.update(all_mask_gts_gpu, all_mask_preds_gpu)

    # Synchronize to compute metric attributes
    il_acc1_metrics.synchronize_between_processes()
    il_auc_metrics.synchronize_between_processes()
    il_f1_metrics.synchronize_between_processes()
    pl_auc_metrics.synchronize_between_processes()

    # Store results
    results = []

    logger.info("Computing metrics for each threshold...")
    for threshold in thresholds:
        # Calculate pixel-level F1 for this threshold
        pl_f1_metrics = F1Metric(1)
        mask_bin = (all_mask_preds_gpu > threshold).int()
        pl_f1_metrics.update(all_mask_gts_gpu, mask_bin)
        pl_f1_metrics.synchronize_between_processes()  # Required to compute f1 attribute

        # Calculate IoU for this threshold (using CPU tensors for efficiency)
        gt_binary = (all_mask_gts > 0.5).float()
        pred_binary = (all_mask_preds > threshold).float()

        intersection = torch.sum(gt_binary * pred_binary).item()
        union = torch.sum(torch.maximum(gt_binary, pred_binary)).item()
        iou = intersection / (union + 1e-8)

        # Calculate precision and recall
        tp = torch.sum(gt_binary * pred_binary).item()
        fp = torch.sum((1 - gt_binary) * pred_binary).item()
        fn = torch.sum(gt_binary * (1 - pred_binary)).item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        results.append(
            {
                "threshold": threshold,
                "pixel_f1": pl_f1_metrics.f1,
                "pixel_precision": precision,
                "pixel_recall": recall,
                "pixel_iou": iou,
            }
        )

        logger.info(
            f"Threshold {threshold:.1f}: F1={pl_f1_metrics.f1:.3f}, IoU={iou:.3f}, P={precision:.3f}, R={recall:.3f}"
        )

    # Add threshold-independent metrics to first result
    results[0].update(
        {
            "image_acc": il_acc1_metrics.acc,
            "image_auc": il_auc_metrics.auc,
            "image_f1": il_f1_metrics.f1,
            "pixel_auc": pl_auc_metrics.auc,
        }
    )

    return results


def save_results(results, output_dir):
    """Save results as CSV and create visualization plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save CSV
    csv_path = os.path.join(output_dir, "threshold_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot F1 scores
    ax1.plot(df["threshold"], df["pixel_f1"], "b-o", linewidth=2, markersize=6)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Pixel-level F1 Score")
    ax1.set_title("F1 Score vs Threshold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Find and mark best F1
    best_f1_idx = df["pixel_f1"].idxmax()
    best_threshold = df.loc[best_f1_idx, "threshold"]
    best_f1 = df.loc[best_f1_idx, "pixel_f1"]
    ax1.axvline(x=best_threshold, color="r", linestyle="--", alpha=0.7)
    ax1.text(
        best_threshold,
        best_f1 + 0.02,
        f"Best: {best_threshold:.1f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        color="red",
    )

    # Plot IoU
    ax2.plot(df["threshold"], df["pixel_iou"], "g-o", linewidth=2, markersize=6)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Pixel-level IoU")
    ax2.set_title("IoU vs Threshold")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Plot Precision and Recall
    ax3.plot(
        df["threshold"],
        df["pixel_precision"],
        "r-o",
        linewidth=2,
        markersize=6,
        label="Precision",
    )
    ax3.plot(
        df["threshold"],
        df["pixel_recall"],
        "b-o",
        linewidth=2,
        markersize=6,
        label="Recall",
    )
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Score")
    ax3.set_title("Precision & Recall vs Threshold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Plot F1 vs IoU scatter
    ax4.scatter(
        df["pixel_iou"], df["pixel_f1"], c=df["threshold"], cmap="viridis", s=100
    )
    ax4.set_xlabel("IoU")
    ax4.set_ylabel("F1 Score")
    ax4.set_title("F1 vs IoU (colored by threshold)")
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label("Threshold")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "threshold_sweep_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Plots saved to: {plot_path}")

    # Print summary
    logger.info("\n=== THRESHOLD SWEEP SUMMARY ===")
    logger.info(f"Image-level metrics (threshold-independent):")
    logger.info(f"  Accuracy: {results[0]['image_acc']:.3f}")
    logger.info(f"  AUC: {results[0]['image_auc']:.3f}")
    logger.info(f"  F1: {results[0]['image_f1']:.3f}")
    logger.info(f"Pixel-level AUC: {results[0]['pixel_auc']:.3f}")
    logger.info(
        f"\nBest pixel-level F1: {best_f1:.3f} at threshold {best_threshold:.1f}"
    )

    # Find best IoU
    best_iou_idx = df["pixel_iou"].idxmax()
    best_iou_threshold = df.loc[best_iou_idx, "threshold"]
    best_iou = df.loc[best_iou_idx, "pixel_iou"]
    logger.info(
        f"Best pixel-level IoU: {best_iou:.3f} at threshold {best_iou_threshold:.1f}"
    )

    return best_threshold, best_f1


def threshold_sweep(local_rank, cfg):
    init_process_group(local_rank, cfg["NUM_GPUS"])
    np.random.seed(cfg["RNG_SEED"])
    torch.manual_seed(cfg["RNG_SEED"])
    logging.setup_logging(cfg, mode="test")
    logger.info(
        json.dumps(
            cfg, indent=4, ensure_ascii=False, sort_keys=False, separators=(",", ":")
        )
    )

    model = build_model(cfg)
    load_test_checkpoint(cfg, model)

    test_dataset = build_dataset("test", cfg)
    test_loader = build_dataloader(test_dataset, "test", cfg)

    logger.info(f"Starting threshold sweep on {len(test_loader)} batches")

    # Define threshold range
    thresholds = [
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
    ]

    results = perform_threshold_sweep(test_loader, model, cfg, thresholds)

    # Save results using OUTPUT_DIR from config
    output_dir = os.path.join(cfg.get("OUTPUT_DIR", "output"), "threshold_sweep")
    best_threshold, best_f1 = save_results(results, output_dir)

    logger.info(f"\n=== RECOMMENDATION ===")
    logger.info(f"Update your config with: TEST.THRES: {best_threshold}")
    logger.info(f"This should improve pixel-level F1 from current to {best_f1:.3f}")


def main():
    args = parse_args()
    cfg = load_config(args)

    # Force test mode
    cfg["TRAIN"]["ENABLE"] = False
    cfg["TEST"]["ENABLE"] = True

    launch_func(cfg=cfg, func=threshold_sweep)


if __name__ == "__main__":
    main()

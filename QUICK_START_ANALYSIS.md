# Quick Start: Validation Analysis

## Step 1: Re-evaluate with Rebalanced Splits

Update config to run test mode:

```bash
# Edit configs/objectformer_doc_tamper_adamw.yaml
# Set: TRAIN.ENABLE: False
# Set: TEST.ENABLE: True
```

Run evaluation:

```bash
uv run python run.py --cfg configs/objectformer_doc_tamper_adamw.yaml
```

**Expected improvements:**
- Validation AUC should increase from 0.286 to 0.7-0.9
- More stable metrics with 10 authentic samples

---

## Step 2: Analyze Validation Predictions

```bash
uv run python tools/analyze_validation_predictions.py \
    --cfg configs/objectformer_doc_tamper_adamw.yaml \
    --threshold 0.4 \
    --output_dir validation_analysis
```

**Outputs:**
- `validation_analysis/validation_analysis.json` - Detailed metrics
- `validation_analysis/confidence_distribution.png` - Visualization
- Console report with confusion matrix

**What to check:**
- False negative count (should be 0-2 for MVP)
- False positive rate on authentic samples
- Confidence separation between classes

---

## Step 3: Generate Sample Predictions

```bash
# Test on a few validation images
uv run python tools/predict_with_confidence.py \
    --cfg configs/objectformer_doc_tamper_adamw.yaml \
    --checkpoint objectformer_doc_tamper_adamw_ktp_base_frozen/checkpoints/ckpt_epoch_00019.pyth \
    --input_dir training_dataset/images \
    --output_dir sample_predictions \
    --threshold 0.4
```

**Outputs:**
- `sample_predictions/predictions.json` - All predictions with confidence
- `sample_predictions/summary.json` - Statistics
- `sample_predictions/visualizations/` - Heatmap overlays
- `sample_predictions/masks/` - Raw masks

**What to check:**
- Review visualizations for authentic samples (should be mostly green/clear)
- Review visualizations for tampered samples (should show red regions)
- Validate confidence scores align with visual inspection

---

## Step 4: Threshold Tuning (Optional)

Test multiple thresholds:

```bash
for threshold in 0.3 0.35 0.4 0.45 0.5; do
    echo "Testing threshold: $threshold"
    uv run python tools/analyze_validation_predictions.py \
        --cfg configs/objectformer_doc_tamper_adamw.yaml \
        --threshold $threshold \
        --output_dir validation_analysis_thresh_$threshold
done
```

Compare results:
- Lower threshold (0.3-0.35): More conservative, higher recall
- Higher threshold (0.45-0.5): More aggressive, higher precision

**For KYC**: Choose threshold that minimizes false negatives while keeping false positive rate acceptable.

---

## Interpreting Results

### Good Signs for MVP ✅
- Validation AUC > 0.7
- False negative count ≤ 2 (on 21 tampered samples)
- False positive count ≤ 3 (on 10 authentic samples)
- Confidence separation: authentic mean < 0.4, tampered mean > 0.6

### Warning Signs ⚠️
- Validation AUC < 0.6
- False negative count > 3 (tampered docs slipping through)
- False positive count > 5 (too many legitimate docs flagged)
- Poor confidence separation (means overlap significantly)

### Critical Issues 🚫
- AUC < 0.5 (worse than random)
- False negative count > 5 (>25% of tampered docs missed)
- False positive count > 7 (>70% of authentic docs flagged)

---

## Next Steps Based on Results

### If metrics look good (✅):
1. Deploy to staging environment
2. Test with production-like KYC documents
3. Train KYC reviewers on confidence scores and visualizations
4. Set up monitoring for production false negative rate

### If metrics need improvement (⚠️):
1. Collect more training data (especially authentic samples)
2. Fine-tune for more epochs (try 5-10 more)
3. Adjust threshold based on business requirements
4. Consider data augmentation for better generalization

### If metrics are poor (🚫):
1. Review data quality (check for labeling errors)
2. Increase training data significantly (target 500+ samples)
3. Consider model architecture changes
4. Validate checkpoint loading is correct

---

## Quick Checklist

Before deploying to production:

- [ ] Validation AUC > 0.7
- [ ] Test set AUC > 0.85
- [ ] False negative rate < 10% on validation
- [ ] Confidence scoring script tested
- [ ] Visualizations reviewed by KYC team
- [ ] Threshold documented and justified
- [ ] Error analysis complete
- [ ] Production monitoring plan in place

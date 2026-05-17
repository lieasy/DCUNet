# 120-Segment Continual Unified Metric Supplement

Run from the project root:

```powershell
cd D:\VS-program\hs_recover_major
```

This supplement recomputes the full requested unified metrics:

- Subject-independent continual few-shot adaptation with `120` support segments.
- SNR robustness under the same continual few-shot setting for `120` support segments at `30` and `40` dB.
- Metrics include PCC, MSE, MAE, RMSE, SC, ePCC, Mean HR bias, and SD.

The script reuses the existing detailed metric evaluator and support/adaptation code from
`evaluate_missing_requested_metrics.py`.

## Full 120-Segment Supplement

```powershell
python code\evaluate_120seg_continual_unified_metrics.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --optimized-results-dir results\continual_subject5fold_adaptation_optimized `
  --output-dir results\continual_120seg_unified_metrics `
  --mode all `
  --support-list 120 `
  --noise-types white,pink,babble `
  --snr-list 30,40 `
  --support-strategy stratified_quality `
  --stage1-epochs 12 `
  --stage2-epochs 70 `
  --batch-size 16 `
  --adapt-batch-size 8 `
  --base-channels 32
```

Main outputs:

```text
results\continual_120seg_unified_metrics\requested_120seg_unified_metrics.csv
results\continual_120seg_unified_metrics\summary.json
results\continual_120seg_unified_metrics\continual_subject5fold_adaptation_optimized\requested_metrics.csv
results\continual_120seg_unified_metrics\continual_snr_robustness\requested_metrics.csv
```

Per-subject detailed outputs:

```text
results\continual_120seg_unified_metrics\continual_subject5fold_adaptation_optimized\fold_0\support_120\sample01_detailed_metrics.json
results\continual_120seg_unified_metrics\continual_snr_robustness\fold_0\support_120\sample01_snr_detailed_metrics.json
```

## Only Clean 120-Segment Metrics

```powershell
python code\evaluate_120seg_continual_unified_metrics.py `
  --mode clean `
  --support-list 120
```

## Only 120-Segment SNR 30/40 dB Metrics

```powershell
python code\evaluate_120seg_continual_unified_metrics.py `
  --mode snr `
  --support-list 120 `
  --snr-list 30,40
```

## Quick Smoke Test

```powershell
python code\evaluate_120seg_continual_unified_metrics.py `
  --output-dir results\debug_120seg_unified_metrics `
  --mode all `
  --support-list 120 `
  --snr-list 40 `
  --noise-types white `
  --max-folds 1 `
  --max-subjects 1 `
  --stage1-epochs 1 `
  --stage2-epochs 1 `
  --batch-size 8 `
  --adapt-batch-size 4 `
  --cpu
```

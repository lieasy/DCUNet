# Continual SNR Unified Metric Supplement

Run from the project root:

```powershell
cd D:\VS-program\hs_recover_major
```

This supplement is for conditions that already have old normalized
`PCC/MSE/MAE/RMSE` SNR robustness results but are missing the full unified metrics:

- `SC`
- `ePCC`
- `Mean HR [bpm]`
- `SD [bpm]`

By default it evaluates `72,96` support segments at `30,40` dB for
`white,pink,babble` noise.

## 72/96 Seg, 30/40 dB Full Unified Metrics

```powershell
python code\evaluate_continual_snr_unified_supplement.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --optimized-results-dir results\continual_subject5fold_adaptation_optimized `
  --snr-results-dir results\continual_snr_robustness `
  --output-dir results\continual_snr_unified_supplement_72_96_30_40 `
  --support-list 72,96 `
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
results\continual_snr_unified_supplement_72_96_30_40\requested_snr_unified_metrics.csv
results\continual_snr_unified_supplement_72_96_30_40\summary.json
results\continual_snr_unified_supplement_72_96_30_40\continual_snr_robustness\requested_metrics.csv
results\continual_snr_unified_supplement_72_96_30_40\continual_snr_robustness\condition_summary.csv
results\continual_snr_unified_supplement_72_96_30_40\continual_snr_robustness\condition_subject_metrics.csv
```

Per-subject detailed metrics:

```text
results\continual_snr_unified_supplement_72_96_30_40\continual_snr_robustness\fold_0\support_072\sample01_snr_detailed_metrics.json
results\continual_snr_unified_supplement_72_96_30_40\continual_snr_robustness\fold_0\support_096\sample01_snr_detailed_metrics.json
```

## Quick Smoke Test

```powershell
python code\evaluate_continual_snr_unified_supplement.py `
  --output-dir results\debug_snr_unified_supplement `
  --support-list 72 `
  --noise-types white `
  --snr-list 40 `
  --max-folds 1 `
  --max-subjects 1 `
  --stage1-epochs 1 `
  --stage2-epochs 1 `
  --batch-size 8 `
  --adapt-batch-size 4 `
  --cpu
```

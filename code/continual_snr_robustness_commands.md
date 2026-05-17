# Continual Few-Shot SNR Robustness Commands

Run from the project root:

```powershell
cd D:\VS-program\hs_recover_major
```

## Full Robustness Test

This test keeps the continual few-shot adaptation framework unchanged:

1. Load the existing subject-independent fold base model.
2. Select clean support segments for each held-out subject.
3. Adapt the model on clean support data.
4. Evaluate clean-support adapted models on noisy query inputs.

Noise is injected into the two input channels only. The target signal remains clean.

```powershell
python code\test_continual_snr_robustness.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --base-results-dir results\continual_subject5fold_adaptation `
  --output-dir results\continual_snr_robustness `
  --support-list 48,72,96 `
  --noise-types white,pink,babble `
  --snr-list 0,5,10,20,30,40 `
  --support-strategy stratified_quality `
  --stage1-epochs 12 `
  --stage2-epochs 70 `
  --batch-size 16 `
  --adapt-batch-size 8 `
  --base-channels 32
```

Main output:

```text
results\continual_snr_robustness\continual_snr_robustness_summary.json
```

Per-fold and per-support outputs:

```text
results\continual_snr_robustness\fold_0\summary.json
results\continual_snr_robustness\fold_0\support_048\summary.json
results\continual_snr_robustness\fold_0\support_048\sample01_snr_robustness.json
```

## Supplement Existing Results With 30/40 dB

If `results\continual_snr_robustness` already contains the 0/5/10/20 dB rows, run only the new conditions and merge them back into the same result tree:

```powershell
python code\test_continual_snr_robustness.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --base-results-dir results\continual_subject5fold_adaptation_optimized `
  --output-dir results\continual_snr_robustness `
  --support-list 48,72,96 `
  --noise-types white,pink,babble `
  --snr-list 30,40 `
  --merge-existing `
  --support-strategy stratified_quality `
  --stage1-epochs 12 `
  --stage2-epochs 70 `
  --batch-size 16 `
  --adapt-batch-size 8 `
  --base-channels 32
```

After this finishes, the rebuilt summaries contain the complete gradient:

```text
0 dB, 5 dB, 10 dB, 20 dB, 30 dB, 40 dB
```

## If Using Optimized Base Results

If the optimized base models already exist, switch `--base-results-dir`:

```powershell
python code\test_continual_snr_robustness.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --base-results-dir results\continual_subject5fold_adaptation_optimized `
  --output-dir results\continual_snr_robustness_optimized_base `
  --support-list 48,72,96 `
  --noise-types white,pink,babble `
  --snr-list 0,5,10,20,30,40
```

## Quick Smoke Test

Use this before the full run:

```powershell
python code\test_continual_snr_robustness.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --base-results-dir results\continual_subject5fold_adaptation `
  --output-dir results\debug_continual_snr_robustness `
  --max-folds 1 `
  --support-list 48 `
  --noise-types white,pink,babble `
  --snr-list 40,0 `
  --stage1-epochs 1 `
  --stage2-epochs 1 `
  --batch-size 8 `
  --adapt-batch-size 4
```

## Noise Conditions

```text
white   Gaussian white noise
pink    1/f pink noise generated in the frequency domain
babble  Multi-source babble-like interference mixed from other query segments
```

SNR levels:

```text
0 dB, 5 dB, 10 dB, 20 dB, 30 dB, 40 dB
```

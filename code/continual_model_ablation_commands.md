# Continual Few-Shot Model-Structure Ablation Commands

Run from the project root:

```powershell
cd D:\VS-program\hs_recover_major
```

## Full Suite

This keeps the subject-independent 5-fold continual few-shot adaptation framework unchanged. Each variant trains its own base model with the same fold split, then performs the same held-out-subject support adaptation.

```powershell
python code\train_continual_model_ablation_suite.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --output-dir results\continual_model_ablation_suite `
  --variants all `
  --folds 5 `
  --support-list 48,72,96,120 `
  --support-strategy stratified_quality `
  --base-epochs 180 `
  --base-min-epochs 40 `
  --base-early-stop 35 `
  --stage1-epochs 12 `
  --stage2-epochs 70 `
  --batch-size 16 `
  --adapt-batch-size 8 `
  --base-channels 32
```

Main output:

```text
results\continual_model_ablation_suite\continual_model_ablation_suite_summary.json
```

Per-variant output:

```text
results\continual_model_ablation_suite\<variant>\model_ablation_summary.json
results\continual_model_ablation_suite\<variant>\fold_0\base_model\best_base_model.pth
results\continual_model_ablation_suite\<variant>\fold_0\support_048\summary.json
```

## Variants

The suite contains one full reference model plus six model-structure ablations:

```text
full_model               Full current model reference.
no_multiscale_stem       Remove MultiScaleStem and use a single-scale stem.
no_residual_shortcut     Remove residual shortcuts inside convolution blocks.
no_encoder_decoder_skip  Remove UNet encoder-decoder skip feature transfer.
shallow_bottleneck       Remove one deep bottleneck residual block.
no_dropout               Remove dropout in residual blocks.
simple_head              Remove the output refinement residual head.
```

Run only the six ablations without the full reference:

```powershell
python code\train_continual_model_ablation_suite.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --output-dir results\continual_model_ablation_only `
  --variants no_multiscale_stem,no_residual_shortcut,no_encoder_decoder_skip,shallow_bottleneck,no_dropout,simple_head `
  --folds 5 `
  --support-list 48,72,96
```

## Add Only 120seg To Current Results

This reuses the base models already saved under `results\continual_model_ablation_suite`, runs only the 120-segment adaptation, and merges the new `support_120` summaries alongside the existing 48/72/96 results.

```powershell
python code\train_continual_model_ablation_suite.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --output-dir results\continual_model_ablation_suite `
  --variants all `
  --folds 5 `
  --support-list 120 `
  --support-strategy stratified_quality `
  --reuse-base-models `
  --merge-existing-summaries `
  --stage1-epochs 12 `
  --stage2-epochs 70 `
  --batch-size 16 `
  --adapt-batch-size 8 `
  --base-channels 32
```

If a subject has fewer than 120 usable support segments, the code automatically uses `min(120, total_segments - min_query_segments)`. With the default `--min-query-segments 20`, at least 20 query segments are kept for evaluation, and each JSON row records the actual `support_segments` used.

## Add 120seg Unified Metrics

The training summary contains PCC/RMSE/MAE only. Run this after the 120seg ablation run to recompute waveform-level unified metrics, including `SC`, `ePCC[%]`, and `HR-SD [bpm]`, and save them under the same ablation result directory.

```powershell
python code\evaluate_continual_unified_metrics.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --ablation-results-dir results\continual_model_ablation_suite `
  --output-dir results\continual_model_ablation_suite\unified_metrics_120seg `
  --mode clean `
  --variants all `
  --support-list 120 `
  --support-strategy stratified_quality `
  --stage1-epochs 12 `
  --stage2-epochs 70 `
  --batch-size 16 `
  --adapt-batch-size 8 `
  --base-channels 32
```

Useful output files:

```text
results\continual_model_ablation_suite\unified_metrics_120seg\clean_requested_metrics.csv
results\continual_model_ablation_suite\unified_metrics_120seg\clean_unified_summary.csv
results\continual_model_ablation_suite\unified_metrics_120seg\unified_metrics_summary.json
```

## Quick Smoke Test

Use this before the full run:

```powershell
python code\train_continual_model_ablation_suite.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --output-dir results\debug_continual_model_ablation_suite `
  --variants full_model,no_multiscale_stem `
  --max-folds 1 `
  --support-list 48 `
  --base-epochs 1 `
  --base-min-epochs 1 `
  --base-early-stop 0 `
  --stage1-epochs 1 `
  --stage2-epochs 1 `
  --batch-size 8 `
  --adapt-batch-size 4
```

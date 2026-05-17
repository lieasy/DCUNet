# Experiment Commands

Run these commands from the project root:

```powershell
cd D:\VS-program\hs_recover_major
```

## 1. Segment-mixed 5-fold baseline

This split is not subject-independent. It mixes all 2.5s segments first, then performs 5-fold cross validation, so segments from the same subject may appear in both train and test. Use it as the easier baseline to show the model structure can learn the recovery mapping.

```powershell
python code\train_segment_mixed_5fold_recovery.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --output-dir results\segment_mixed_5fold_recovery `
  --folds 5 `
  --epochs 180 `
  --min-epochs 40 `
  --early-stop 35 `
  --batch-size 16 `
  --base-channels 32 `
  --lr 2e-4 `
  --weight-decay 1e-6 `
  --stats-segments 4096
```

Quick smoke test:

```powershell
python code\train_segment_mixed_5fold_recovery.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --output-dir results\debug_segment_mixed_5fold_recovery `
  --max-folds 1 `
  --epochs 1 `
  --min-epochs 1 `
  --early-stop 0 `
  --batch-size 8
```

Main output:

```text
results\segment_mixed_5fold_recovery\segment_mixed_5fold_summary.json
```

## 2. Subject-independent 5-fold continual few-shot adaptation

This is the stricter cross-subject generalization experiment. Each fold holds out whole subjects, trains the base model on the remaining subjects, then evaluates few-shot adaptation on held-out subjects.

```powershell
python code\train_continual_subject5fold_adaptation_optimized.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --output-dir results\continual_subject5fold_adaptation_optimized `
  --folds 5 `
  --support-list 48,72,96 `
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
results\continual_subject5fold_adaptation_optimized\optimized_subject5fold_summary.json
```

## 3. Refine existing subject-independent results

Use this if the base models already exist and you only want to rerun stronger adaptation settings.

```powershell
python code\refine_continual_subject5fold_adaptation.py `
  --data-dir data\process_data\merged_htzx_lez_2.5s `
  --base-results-dir results\continual_subject5fold_adaptation `
  --output-dir results\continual_subject5fold_adaptation_refined `
  --support-list 72,96,120 `
  --support-strategy stratified_quality `
  --stage1-epochs 12 `
  --stage2-epochs 70 `
  --batch-size 16 `
  --adapt-batch-size 8
```

Main output:

```text
results\continual_subject5fold_adaptation_refined\refined_subject5fold_summary.json
```

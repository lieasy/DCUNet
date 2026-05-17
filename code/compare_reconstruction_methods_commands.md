# Five Comparison Reconstruction Methods

Unified entry:

```powershell
python code\train_compare_reconstruction_methods.py --experiment both --models all
```

## Methods

- `biosignal_gan`: adapted from `compare_program/biosignalGANs-main`, pix2pix/CycleGAN-style 1D generator.
- `p2e_wgan`: adapted from `compare_program/P2E-WGAN-ecg-ppg-reconstruction-main`.
- `cardiogan`: adapted from `compare_program/ppg2ecg-cardiogan-main`, attention U-Net generator.
- `ppg2ecg`: adapted from `compare_program/ppg2ecg-pytorch-master`.
- `rddm`: adapted from `compare_program/RDDM-main`, self-attention U-Net backbone trained as a direct reconstructor.

All methods use the current project's data format:

```text
data/process_data/merged_htzx_lez_2.5s/sample*_2.5s_filter_*.npy
shape = (segments, 3, 2500)
input = channel 2 + channel 3
target = channel 1
```

They also reuse the current robust normalization, `HeartSoundLoss`, support selection, and fold splitting logic. Final test/query evaluation now writes the detailed metrics:

```text
overall signal: PCC, MSE, MAE, RMSE, spectral convergence
envelope: envelope PCC, MSE, MAE, RMSE
S1: PCC, MSE, MAE, RMSE, average detected count
S2: PCC, MSE, MAE, RMSE, average detected count
heart rate: valid estimate count, mean HR bias, HR bias std
```

## Experiment 1: Subject Non-Independent

This is the segment-mixed 5-fold setting. Segments from the same subject may appear in train and test, matching `train_segment_mixed_5fold_recovery.py`.

Run all five:

```powershell
python code\train_compare_reconstruction_methods.py --experiment segment_mixed --models all
```

Run one method:

```powershell
python code\train_compare_reconstruction_methods.py --experiment segment_mixed --models p2e_wgan
```

## Experiment 2: Subject-Independent Few-Shot Continual Learning

This is the subject-level balanced 5-fold setting. Base subjects are used to train a base model, then each held-out subject is adapted with few support segments, matching `train_continual_subject5fold_adaptation_optimized.py`.

Run all five:

```powershell
python code\train_compare_reconstruction_methods.py --experiment continual --models all --support-list 48,72,96
```

Run one method:

```powershell
python code\train_compare_reconstruction_methods.py --experiment continual --models cardiogan --support-list 48,72,96
```

## Quick Debug Runs

Use these to check that a method can run before launching the full comparison:

```powershell
python code\train_compare_reconstruction_methods.py --experiment segment_mixed --models ppg2ecg --max-folds 1 --epochs 1 --min-epochs 1 --early-stop 0 --base-channels 8 --batch-size 4 --cpu
```

```powershell
python code\train_compare_reconstruction_methods.py --experiment continual --models ppg2ecg --max-folds 1 --base-epochs 1 --base-min-epochs 1 --base-early-stop 0 --stage1-epochs 1 --stage2-epochs 1 --support-list 48 --base-channels 8 --batch-size 4 --adapt-batch-size 4 --cpu
```

## Outputs

Default output root:

```text
results/compare_reconstruction_methods/
```

Important files:

```text
run_config.json
compare_summary.json
segment_mixed/<method>/segment_mixed_summary.json
continual/<method>/continual_summary.json
```

Detailed metric fields are saved as:

```text
segment_mixed/<method>/fold_*/summary.json -> detailed_test_metrics
continual/<method>/fold_*/support_*/sample*.json -> before_detailed_metrics / after_detailed_metrics / delta_detailed_metrics
continual/<method>/fold_*/support_*/summary.json -> before_detailed_aggregate / after_detailed_aggregate / delta_detailed_aggregate
```

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_continual_subject5fold_adaptation import (  # noqa: E402
    DEFAULT_DATA_DIR,
    HeartSoundLoss,
    HeartSoundUNet,
    SegmentDataset,
    choose_support,
    discover_subjects,
    make_loader,
    run_epoch,
    set_seed,
    set_trainable,
    split_support,
)
from test_continual_snr_robustness import NoisySegmentDataset  # noqa: E402
from train_compare_reconstruction_methods import build_model as build_compare_model  # noqa: E402
from train_compare_reconstruction_methods import detect_heart_sounds, estimate_heart_rate, extract_envelope  # noqa: E402
from train_compare_reconstruction_methods import parse_model_list  # noqa: E402
from train_continual_model_ablation_suite import make_model as build_ablation_model  # noqa: E402
from train_continual_model_ablation_suite import parse_list as parse_ablation_list  # noqa: E402
from train_segment_mixed_5fold_recovery import build_all_segment_indices, build_segment_folds  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "normalized_recovery_metrics"

BASE_METRIC_KEYS = [
    "pcc",
    "mse_raw",
    "mae_raw",
    "rmse_raw",
    "nmse_target_std",
    "nmae_target_std",
    "nrmse_target_std",
    "nmse_target_rms",
    "nrmse_target_rms",
    "nmae_target_absmean",
]

HR_METRIC_KEYS = [
    "hr_target_mean_bpm",
    "hr_pred_mean_bpm",
    "hr_bias_mean_bpm",
    "hr_mae_bpm",
    "hr_rmse_bpm",
    "hr_sd_bpm",
    "nhr_mae_target_mean",
    "nhr_rmse_target_mean",
    "nhr_sd_target_mean",
    "nhr_mae_percent",
    "nhr_rmse_percent",
    "nhr_sd_percent",
    "hr_valid_count",
    "hr_total_count",
]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value):
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


def safe_mean(values):
    arr = np.array([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else None


def safe_std(values):
    arr = np.array([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=np.float64)
    return float(np.std(arr)) if arr.size else None


def pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.std(a) <= 1e-8 or np.std(b) <= 1e-8:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))


def denorm_prediction(normalized: torch.Tensor, params: torch.Tensor, stats: dict) -> torch.Tensor:
    local_mean = params[:, 0].to(normalized.device).view(-1, 1, 1)
    local_std = params[:, 1].to(normalized.device).view(-1, 1, 1)
    global_mean = torch.tensor(float(stats["target_mean"]), device=normalized.device).view(1, 1, 1)
    global_std = torch.tensor(float(stats["target_std"]), device=normalized.device).view(1, 1, 1)
    scale = 0.7 / local_std + 0.3 / global_std
    shift = 0.7 * local_mean / local_std + 0.3 * global_mean / global_std
    return (normalized + shift) / scale


class DetailedMetricSegmentDataset(SegmentDataset):
    def __getitem__(self, idx):
        x, y, sample_id = super().__getitem__(idx)
        file_idx, segment_idx = self.indices[idx]
        raw_target = np.asarray(self.arrays[file_idx][segment_idx][0], dtype=np.float32)
        local_mean = float(np.median(raw_target))
        local_std = float(np.median(np.abs(raw_target - local_mean)) * 1.4826 + 1e-8)
        return x, y, torch.from_numpy(raw_target[None, :]), sample_id, torch.tensor([local_mean, local_std], dtype=torch.float32)


class NoisyDetailedMetricSegmentDataset(NoisySegmentDataset):
    def __getitem__(self, idx):
        x, y, sample_id = super().__getitem__(idx)
        file_idx, segment_idx = self.indices[idx]
        raw_target = np.asarray(self.arrays[file_idx][segment_idx][0], dtype=np.float32)
        local_mean = float(np.median(raw_target))
        local_std = float(np.median(np.abs(raw_target - local_mean)) * 1.4826 + 1e-8)
        return x, y, torch.from_numpy(raw_target[None, :]), sample_id, torch.tensor([local_mean, local_std], dtype=torch.float32)


def normalized_sample_metrics(pred: np.ndarray, target: np.ndarray, target_std: float) -> dict:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    err = pred - target
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    target_std = max(float(target_std), 1e-8)
    target_rms = max(float(np.sqrt(np.mean(target**2))), 1e-8)
    target_abs = max(float(np.mean(np.abs(target))), 1e-8)
    return {
        "pcc": pearson_np(pred, target),
        "mse_raw": mse,
        "mae_raw": mae,
        "rmse_raw": rmse,
        "nmse_target_std": mse / (target_std**2),
        "nmae_target_std": mae / target_std,
        "nrmse_target_std": rmse / target_std,
        "nmse_target_rms": mse / (target_rms**2),
        "nrmse_target_rms": rmse / target_rms,
        "nmae_target_absmean": mae / target_abs,
        "target_rms": target_rms,
        "target_absmean": target_abs,
    }


def evaluate_dataset(model, dataset, stats, args, device, desc: str) -> dict:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    values = {key: [] for key in BASE_METRIC_KEYS}
    values["target_rms"] = []
    values["target_absmean"] = []
    hr_target_values = []
    hr_pred_values = []
    hr_bias_values = []
    hr_abs_bias_values = []
    hr_total_count = 0
    model.eval()
    with torch.no_grad():
        for x, _y, raw_target, _sid, params in loader:
            x = x.to(device, non_blocking=True)
            raw_target = raw_target.to(device, non_blocking=True)
            params = params.to(device, non_blocking=True)
            pred = denorm_prediction(model(x), params, stats)
            pred_np = pred.detach().cpu().numpy()
            target_np = raw_target.detach().cpu().numpy()
            for item_idx in range(pred_np.shape[0]):
                sample = normalized_sample_metrics(
                    pred_np[item_idx, 0],
                    target_np[item_idx, 0],
                    float(stats["target_std"]),
                )
                for key, value in sample.items():
                    values[key].append(value)
                target_env = extract_envelope(target_np[item_idx, 0], args.sample_rate)
                pred_env = extract_envelope(pred_np[item_idx, 0], args.sample_rate)
                s1_target, _ = detect_heart_sounds(target_np[item_idx, 0], target_env, args.sample_rate)
                s1_pred, _ = detect_heart_sounds(pred_np[item_idx, 0], pred_env, args.sample_rate)
                hr_target = estimate_heart_rate(s1_target, args.sample_rate)
                hr_pred = estimate_heart_rate(s1_pred, args.sample_rate)
                hr_total_count += 1
                if hr_target is not None and hr_pred is not None:
                    bias = float(hr_pred - hr_target)
                    hr_target_values.append(float(hr_target))
                    hr_pred_values.append(float(hr_pred))
                    hr_bias_values.append(bias)
                    hr_abs_bias_values.append(abs(bias))
    out = {key: safe_mean(vals) for key, vals in values.items()}
    out.update({f"{key}_std": safe_std(vals) for key, vals in values.items() if key in BASE_METRIC_KEYS})
    out["n_segments"] = len(values["pcc"])
    out["target_std"] = float(stats["target_std"])
    target_hr_mean = safe_mean(hr_target_values)
    hr_rmse = None
    if hr_bias_values:
        hr_rmse = float(np.sqrt(np.mean(np.square(np.asarray(hr_bias_values, dtype=np.float64)))))
    hr_sd = safe_std(hr_bias_values)
    denom_hr = None if target_hr_mean is None else max(abs(float(target_hr_mean)), 1e-8)
    out.update(
        {
            "hr_target_mean_bpm": target_hr_mean,
            "hr_pred_mean_bpm": safe_mean(hr_pred_values),
            "hr_bias_mean_bpm": safe_mean(hr_bias_values),
            "hr_mae_bpm": safe_mean(hr_abs_bias_values),
            "hr_rmse_bpm": hr_rmse,
            "hr_sd_bpm": hr_sd,
            "nhr_mae_target_mean": None if denom_hr is None else safe_mean(hr_abs_bias_values) / denom_hr,
            "nhr_rmse_target_mean": None if denom_hr is None or hr_rmse is None else hr_rmse / denom_hr,
            "nhr_sd_target_mean": None if denom_hr is None or hr_sd is None else hr_sd / denom_hr,
            "nhr_mae_percent": None if denom_hr is None else safe_mean(hr_abs_bias_values) / denom_hr * 100.0,
            "nhr_rmse_percent": None if denom_hr is None or hr_rmse is None else hr_rmse / denom_hr * 100.0,
            "nhr_sd_percent": None if denom_hr is None or hr_sd is None else hr_sd / denom_hr * 100.0,
            "hr_valid_count": len(hr_bias_values),
            "hr_total_count": hr_total_count,
        }
    )
    print(
        f"{desc}: PCC={out['pcc']:.4f}, "
        f"NRMSE(std)={out['nrmse_target_std']:.4f}, "
        f"NMAE(abs)={out['nmae_target_absmean']:.4f}, "
        f"NHR-SD={out['nhr_sd_percent'] if out['nhr_sd_percent'] is not None else float('nan'):.2f}%"
    )
    return out


def adapt_model(args, model, support_train_set, support_val_set, device, sample_id):
    criterion = HeartSoundLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -float("inf")
    for stage_name, epochs, lr in [("decoder_head", args.stage1_epochs, args.stage1_lr), ("full", args.stage2_epochs, args.stage2_lr)]:
        if epochs <= 0:
            continue
        set_trainable(model, stage_name)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            continue
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.adapt_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.adapt_lr_patience, min_lr=args.adapt_min_lr
        )
        train_loader = make_loader(support_train_set, args.adapt_batch_size, True, device, args.seed + sample_id, args.num_workers)
        val_loader = make_loader(support_val_set, args.adapt_batch_size, False, device, args.seed + sample_id, args.num_workers)
        for epoch in range(1, epochs + 1):
            run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp, f"S{sample_id} {stage_name} {epoch}/{epochs}")
            val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"S{sample_id} support-val")
            scheduler.step(val_metrics["corr"])
            if val_metrics["corr"] > best_val + args.min_delta:
                best_val = val_metrics["corr"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    set_trainable(model, "full")
    return best_val


def append_row(rows, group: dict, metrics: dict):
    row = {**group}
    for key, value in metrics.items():
        row[key] = safe_float(value)
    rows.append(row)


def aggregate_rows(rows: list[dict], group_keys: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row.get(item) for item in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    metric_keys = (
        [
            key
            for key in rows[0].keys()
            if key not in set(group_keys) | {"fold", "sample_id"}
            and not key.endswith("_std")
        ]
        if rows
        else []
    )
    for key, subset in groups.items():
        item = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        item["n_rows"] = len(subset)
        for metric in metric_keys:
            vals = [row.get(metric) for row in subset]
            item[f"{metric}_mean"] = safe_mean(vals)
            item[f"{metric}_std"] = safe_std(vals)
        out.append(item)
    return sorted(out, key=lambda item: tuple(str(item.get(key)) for key in group_keys))


def build_eval_args(args, run_config: dict | None = None):
    cfg = dict(run_config or {})
    defaults = {
        "batch_size": args.batch_size,
        "adapt_batch_size": args.adapt_batch_size,
        "base_channels": args.base_channels,
        "support_val_ratio": args.support_val_ratio,
        "support_val_segments": args.support_val_segments,
        "min_query_segments": args.min_query_segments,
        "support_strategy": args.support_strategy,
        "stage1_epochs": args.stage1_epochs,
        "stage1_lr": args.stage1_lr,
        "stage2_epochs": args.stage2_epochs,
        "stage2_lr": args.stage2_lr,
        "adapt_weight_decay": args.adapt_weight_decay,
        "adapt_lr_patience": args.adapt_lr_patience,
        "adapt_min_lr": args.adapt_min_lr,
        "min_delta": args.min_delta,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "adapt_augment": args.adapt_augment,
        "segment_seconds": 2.5,
        "babble_sources": args.babble_sources,
        "sample_rate": args.sample_rate,
    }
    defaults.update({key: cfg[key] for key in defaults if key in cfg})
    defaults.update(
        {
            "batch_size": args.batch_size,
            "adapt_batch_size": args.adapt_batch_size,
            "num_workers": args.num_workers,
            "amp": args.amp,
        }
    )
    return argparse.Namespace(**defaults)


def load_subjects_arrays(data_dir: Path):
    subjects = discover_subjects(data_dir)
    arrays = [np.load(subject["file"], mmap_mode="r") for subject in subjects]
    return subjects, arrays


def evaluate_segment_mixed(args, subjects, arrays, device, rows):
    run_config = load_json(args.segment_results_dir / "run_config.json", {})
    eval_args = build_eval_args(args, run_config)
    all_indices = build_all_segment_indices(subjects)
    folds = build_segment_folds(all_indices, int(run_config.get("folds", args.folds)), int(run_config.get("seed", args.seed)))
    max_folds = min(args.max_folds or len(folds), len(folds))
    for fold_idx in range(max_folds):
        ckpt_path = args.segment_results_dir / f"fold_{fold_idx}" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"[skip] Missing segment checkpoint: {ckpt_path}")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = HeartSoundUNet(eval_args.base_channels).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        stats = checkpoint["normalization_stats"]
        dataset = DetailedMetricSegmentDataset(subjects, arrays, folds[fold_idx], stats, augment=False)
        metrics = evaluate_dataset(model, dataset, stats, eval_args, device, f"segment_mixed fold {fold_idx}")
        append_row(rows, {"protocol": "segment_mixed", "model": "DCUNet", "stage": "test", "fold": fold_idx}, metrics)


def evaluate_compare_segment_mixed(args, subjects, arrays, device, rows):
    models = parse_model_list(args.compare_models)
    for model_name in models:
        model_dir = args.compare_results_dir / "segment_mixed" / model_name
        run_config = load_json(args.compare_results_dir / "run_config.json", {})
        eval_args = build_eval_args(args, run_config)
        all_indices = build_all_segment_indices(subjects)
        folds = build_segment_folds(all_indices, int(run_config.get("folds", args.folds)), int(run_config.get("seed", args.seed)))
        max_folds = min(args.max_folds or len(folds), len(folds))
        for fold_idx in range(max_folds):
            ckpt_path = model_dir / f"fold_{fold_idx}" / "best_model.pth"
            if not ckpt_path.exists():
                print(f"[skip] Missing compare checkpoint: {ckpt_path}")
                continue
            checkpoint = torch.load(ckpt_path, map_location=device)
            model = build_compare_model(model_name, eval_args.base_channels).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            stats = checkpoint["normalization_stats"]
            dataset = DetailedMetricSegmentDataset(subjects, arrays, folds[fold_idx], stats, augment=False)
            metrics = evaluate_dataset(model, dataset, stats, eval_args, device, f"{model_name} segment fold {fold_idx}")
            append_row(rows, {"protocol": "segment_mixed", "model": model_name, "stage": "test", "fold": fold_idx}, metrics)


def evaluate_continual(args, subjects, arrays, device, rows):
    run_config = load_json(args.continual_results_dir / "run_config.json", {})
    eval_args = build_eval_args(args, run_config)
    fold_test_ids = run_config.get("fold_test_ids")
    if not fold_test_ids:
        raise ValueError(f"Missing fold_test_ids in {args.continual_results_dir / 'run_config.json'}")
    supports = parse_int_list(args.support_list)
    max_folds = min(args.max_folds or len(fold_test_ids), len(fold_test_ids))
    for fold_idx in range(max_folds):
        ckpt_path = args.continual_results_dir / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
        if not ckpt_path.exists():
            print(f"[skip] Missing continual base checkpoint: {ckpt_path}")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device)
        stats = checkpoint["normalization_stats"]
        for support_segments in supports:
            for sample_id in fold_test_ids[fold_idx]:
                support, query = choose_support(
                    subjects, arrays, int(sample_id), support_segments, eval_args.min_query_segments, eval_args.seed, eval_args.support_strategy
                )
                before_model = HeartSoundUNet(eval_args.base_channels).to(device)
                before_model.load_state_dict(checkpoint["model_state_dict"])
                query_dataset = DetailedMetricSegmentDataset(subjects, arrays, query, stats, augment=False)
                before = evaluate_dataset(
                    before_model, query_dataset, stats, eval_args, device, f"continual S{sample_id} {support_segments} before"
                )
                append_row(
                    rows,
                    {
                        "protocol": "continual_clean",
                        "model": "DCUNet",
                        "stage": "before",
                        "fold": fold_idx,
                        "sample_id": int(sample_id),
                        "support_segments": support_segments,
                    },
                    before,
                )
                support_train, support_val = split_support(
                    support, eval_args.seed + int(sample_id) + support_segments, eval_args.support_val_ratio, eval_args.support_val_segments
                )
                adapted_model = HeartSoundUNet(eval_args.base_channels).to(device)
                adapted_model.load_state_dict(checkpoint["model_state_dict"])
                adapt_model(
                    eval_args,
                    adapted_model,
                    SegmentDataset(subjects, arrays, support_train, stats, augment=eval_args.adapt_augment),
                    SegmentDataset(subjects, arrays, support_val, stats, augment=False),
                    device,
                    int(sample_id),
                )
                after = evaluate_dataset(
                    adapted_model, query_dataset, stats, eval_args, device, f"continual S{sample_id} {support_segments} after"
                )
                append_row(
                    rows,
                    {
                        "protocol": "continual_clean",
                        "model": "DCUNet",
                        "stage": "after",
                        "fold": fold_idx,
                        "sample_id": int(sample_id),
                        "support_segments": support_segments,
                    },
                    after,
                )


def evaluate_non_subject_snr(args, subjects, arrays, device, rows):
    run_config = load_json(args.segment_results_dir / "run_config.json", {})
    eval_args = build_eval_args(args, run_config)
    all_indices = build_all_segment_indices(subjects)
    folds = build_segment_folds(all_indices, int(run_config.get("folds", args.folds)), int(run_config.get("seed", args.seed)))
    max_folds = min(args.max_folds or len(folds), len(folds))
    for fold_idx in range(max_folds):
        ckpt_path = args.segment_results_dir / f"fold_{fold_idx}" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"[skip] Missing segment checkpoint: {ckpt_path}")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = HeartSoundUNet(eval_args.base_channels).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        stats = checkpoint["normalization_stats"]
        for noise_type in parse_str_list(args.noise_types):
            for snr_db in parse_float_list(args.snr_list):
                dataset = NoisyDetailedMetricSegmentDataset(
                    subjects,
                    arrays,
                    folds[fold_idx],
                    stats,
                    noise_type=noise_type,
                    snr_db=snr_db,
                    seed=eval_args.seed + fold_idx + int(snr_db * 17),
                    babble_sources=eval_args.babble_sources,
                )
                metrics = evaluate_dataset(model, dataset, stats, eval_args, device, f"non_subject_snr fold {fold_idx} {noise_type} {snr_db:g}")
                append_row(
                    rows,
                    {
                        "protocol": "segment_mixed_snr",
                        "model": "DCUNet",
                        "stage": "test",
                        "fold": fold_idx,
                        "noise_type": noise_type,
                        "snr_db": snr_db,
                    },
                    metrics,
                )


def evaluate_continual_snr(args, subjects, arrays, device, rows):
    run_config = load_json(args.continual_results_dir / "run_config.json", {})
    eval_args = build_eval_args(args, run_config)
    fold_test_ids = run_config.get("fold_test_ids")
    supports = parse_int_list(args.support_list)
    max_folds = min(args.max_folds or len(fold_test_ids), len(fold_test_ids))
    for fold_idx in range(max_folds):
        ckpt_path = args.continual_results_dir / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
        if not ckpt_path.exists():
            print(f"[skip] Missing continual base checkpoint: {ckpt_path}")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device)
        stats = checkpoint["normalization_stats"]
        for support_segments in supports:
            for sample_id in fold_test_ids[fold_idx]:
                support, query = choose_support(
                    subjects, arrays, int(sample_id), support_segments, eval_args.min_query_segments, eval_args.seed, eval_args.support_strategy
                )
                support_train, support_val = split_support(
                    support, eval_args.seed + int(sample_id) + support_segments, eval_args.support_val_ratio, eval_args.support_val_segments
                )
                models = {}
                before_model = HeartSoundUNet(eval_args.base_channels).to(device)
                before_model.load_state_dict(checkpoint["model_state_dict"])
                models["before"] = before_model
                after_model = HeartSoundUNet(eval_args.base_channels).to(device)
                after_model.load_state_dict(checkpoint["model_state_dict"])
                adapt_model(
                    eval_args,
                    after_model,
                    SegmentDataset(subjects, arrays, support_train, stats, augment=eval_args.adapt_augment),
                    SegmentDataset(subjects, arrays, support_val, stats, augment=False),
                    device,
                    int(sample_id),
                )
                models["after"] = after_model
                for stage, model in models.items():
                    for noise_type in parse_str_list(args.noise_types):
                        for snr_db in parse_float_list(args.snr_list):
                            dataset = NoisyDetailedMetricSegmentDataset(
                                subjects,
                                arrays,
                                query,
                                stats,
                                noise_type=noise_type,
                                snr_db=snr_db,
                                seed=eval_args.seed + int(sample_id) + support_segments + int(float(snr_db) * 17),
                                babble_sources=eval_args.babble_sources,
                            )
                            metrics = evaluate_dataset(
                                model, dataset, stats, eval_args, device, f"continual_snr S{sample_id} {support_segments} {stage} {noise_type} {snr_db:g}"
                            )
                            append_row(
                                rows,
                                {
                                    "protocol": "continual_snr",
                                    "model": "DCUNet",
                                    "stage": stage,
                                    "fold": fold_idx,
                                    "sample_id": int(sample_id),
                                    "support_segments": support_segments,
                                    "noise_type": noise_type,
                                    "snr_db": snr_db,
                                },
                                metrics,
                            )


def evaluate_ablation(args, subjects, arrays, device, rows):
    run_config = load_json(args.ablation_results_dir / "run_config.json", {})
    eval_args = build_eval_args(args, run_config)
    fold_test_ids = run_config.get("fold_test_ids")
    if not fold_test_ids:
        continual_config = load_json(args.continual_results_dir / "run_config.json", {})
        fold_test_ids = continual_config.get("fold_test_ids")
    variants = parse_ablation_list(args.ablation_variants)
    supports = parse_int_list(args.support_list)
    max_folds = min(args.max_folds or len(fold_test_ids), len(fold_test_ids))
    for variant in variants:
        for fold_idx in range(max_folds):
            ckpt_path = args.ablation_results_dir / variant / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
            if not ckpt_path.exists():
                print(f"[skip] Missing ablation checkpoint: {ckpt_path}")
                continue
            checkpoint = torch.load(ckpt_path, map_location=device)
            stats = checkpoint["normalization_stats"]
            for support_segments in supports:
                for sample_id in fold_test_ids[fold_idx]:
                    support, query = choose_support(
                        subjects, arrays, int(sample_id), support_segments, eval_args.min_query_segments, eval_args.seed, eval_args.support_strategy
                    )
                    support_train, support_val = split_support(
                        support, eval_args.seed + int(sample_id) + support_segments, eval_args.support_val_ratio, eval_args.support_val_segments
                    )
                    model = build_ablation_model(variant, eval_args.base_channels).to(device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    adapt_model(
                        eval_args,
                        model,
                        SegmentDataset(subjects, arrays, support_train, stats, augment=eval_args.adapt_augment),
                        SegmentDataset(subjects, arrays, support_val, stats, augment=False),
                        device,
                        int(sample_id),
                    )
                    dataset = DetailedMetricSegmentDataset(subjects, arrays, query, stats, augment=False)
                    metrics = evaluate_dataset(model, dataset, stats, eval_args, device, f"ablation {variant} S{sample_id} {support_segments}")
                    append_row(
                        rows,
                        {
                            "protocol": "ablation_clean",
                            "model": variant,
                            "stage": "after",
                            "fold": fold_idx,
                            "sample_id": int(sample_id),
                            "support_segments": support_segments,
                        },
                        metrics,
                    )


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = ["protocol", "model", "stage", "fold", "sample_id", "support_segments", "noise_type", "snr_db"]
    fields = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate saved recovery experiments and report normalized waveform errors. "
            "NMSE/NRMSE are reported with target_std and target RMS denominators; "
            "NMAE is reported with both target_std and mean(abs(target)) denominators."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--segment-results-dir", type=Path, default=PROJECT_ROOT / "results" / "segment_mixed_5fold_recovery")
    parser.add_argument("--compare-results-dir", type=Path, default=PROJECT_ROOT / "results" / "compare_reconstruction_methods")
    parser.add_argument("--continual-results-dir", type=Path, default=PROJECT_ROOT / "results" / "continual_subject5fold_adaptation_optimized")
    parser.add_argument("--ablation-results-dir", type=Path, default=PROJECT_ROOT / "results" / "continual_model_ablation_suite")
    parser.add_argument(
        "--protocols",
        type=str,
        default="segment,compare_segment,continual,non_subject_snr,continual_snr,ablation",
        help="Comma-separated: segment, compare_segment, continual, non_subject_snr, continual_snr, ablation.",
    )
    parser.add_argument("--compare-models", type=str, default="all")
    parser.add_argument("--ablation-variants", type=str, default="all")
    parser.add_argument("--support-list", type=str, default="72,96,120")
    parser.add_argument("--noise-types", type=str, default="white,pink,babble")
    parser.add_argument("--snr-list", type=str, default="10,20,30,40")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=0, help="Debug limit. 0 means all available folds.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--adapt-batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--support-val-ratio", type=float, default=0.25)
    parser.add_argument("--support-val-segments", type=int, default=8)
    parser.add_argument("--min-query-segments", type=int, default=20)
    parser.add_argument("--support-strategy", type=str, default="stratified_quality")
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-epochs", type=int, default=70)
    parser.add_argument("--stage2-lr", type=float, default=8e-5)
    parser.add_argument("--adapt-weight-decay", type=float, default=0.0)
    parser.add_argument("--adapt-lr-patience", type=int, default=8)
    parser.add_argument("--adapt-min-lr", type=float, default=5e-6)
    parser.add_argument("--min-delta", type=float, default=0.0002)
    parser.add_argument("--babble-sources", type=int, default=6)
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapt-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data_dir.is_absolute():
        args.data_dir = PROJECT_ROOT / args.data_dir
    for attr in ["output_dir", "segment_results_dir", "compare_results_dir", "continual_results_dir", "ablation_results_dir"]:
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, (PROJECT_ROOT / value).resolve())
    args.data_dir = args.data_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Data dir: {args.data_dir}")
    subjects, arrays = load_subjects_arrays(args.data_dir)
    print(f"Loaded {len(subjects)} subjects.")

    rows = []
    protocols = set(parse_str_list(args.protocols))
    if "segment" in protocols:
        evaluate_segment_mixed(args, subjects, arrays, device, rows)
    if "compare_segment" in protocols:
        evaluate_compare_segment_mixed(args, subjects, arrays, device, rows)
    if "continual" in protocols:
        evaluate_continual(args, subjects, arrays, device, rows)
    if "non_subject_snr" in protocols:
        evaluate_non_subject_snr(args, subjects, arrays, device, rows)
    if "continual_snr" in protocols:
        evaluate_continual_snr(args, subjects, arrays, device, rows)
    if "ablation" in protocols:
        evaluate_ablation(args, subjects, arrays, device, rows)

    summary_keys = ["protocol", "model", "stage", "support_segments", "noise_type", "snr_db"]
    summary_keys = [key for key in summary_keys if any(key in row for row in rows)]
    summary = aggregate_rows(rows, summary_keys)
    write_csv(args.output_dir / "normalized_metrics_rows.csv", rows)
    write_csv(args.output_dir / "normalized_metrics_summary.csv", summary)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Saved rows to: {args.output_dir / 'normalized_metrics_rows.csv'}")
    print(f"Saved summary to: {args.output_dir / 'normalized_metrics_summary.csv'}")


if __name__ == "__main__":
    main()

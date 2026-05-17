import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_continual_subject5fold_adaptation import (  # noqa: E402
    DEFAULT_DATA_DIR,
    HeartSoundLoss,
    SegmentDataset,
    discover_subjects,
    make_loader,
    run_epoch,
    set_seed,
)
from train_continual_model_ablation_suite import MODEL_VARIANTS, make_model, parse_list  # noqa: E402
from train_continual_subject5fold_adaptation_optimized import compute_stats, sanitize_args  # noqa: E402
from train_compare_reconstruction_methods import (  # noqa: E402
    detect_heart_sounds,
    estimate_heart_rate,
    extract_envelope,
    pearson_np,
)
from train_segment_mixed_5fold_recovery import (  # noqa: E402
    build_all_segment_indices,
    build_segment_folds,
    split_train_val,
    summarize_subject_overlap,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "segment_mixed_ablation_suite"


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


def spectral_convergence_batch(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 1024, hop_length: int = 256):
    pred_2d = pred.squeeze(1)
    target_2d = target.squeeze(1)
    n_fft = min(n_fft, pred_2d.shape[-1])
    window = torch.hann_window(n_fft, device=pred.device)
    pred_spec = torch.stft(pred_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True).abs()
    target_spec = torch.stft(target_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True).abs()
    numerator = torch.linalg.vector_norm(pred_spec - target_spec, dim=(1, 2))
    denominator = torch.linalg.vector_norm(target_spec, dim=(1, 2)).clamp_min(1e-8)
    return (numerator / denominator).detach().cpu().numpy().astype(float).tolist()


def normalized_waveform_metrics(pred: np.ndarray, target: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    err = pred - target
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    target_rms = max(float(np.sqrt(np.mean(target**2))), 1e-8)
    target_abs = max(float(np.mean(np.abs(target))), 1e-8)
    return {
        "pcc": pearson_np(pred, target),
        "nmse": mse / (target_rms**2),
        "nmae": mae / target_abs,
        "nrmse": rmse / target_rms,
    }


def evaluate_normalized_metrics(model, subjects, arrays, indices, stats, args, device, desc):
    dataset = DetailedMetricSegmentDataset(subjects, arrays, indices, stats, augment=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    values = {key: [] for key in ["pcc", "nmse", "nmae", "nrmse", "sc", "epcc"]}
    hr_target_values = []
    hr_bias_values = []
    hr_total = 0
    model.eval()
    with torch.no_grad():
        for x, _y, raw_target, _sid, params in loader:
            x = x.to(device, non_blocking=True)
            raw_target = raw_target.to(device, non_blocking=True)
            params = params.to(device, non_blocking=True)
            pred = denorm_prediction(model(x), params, stats)
            sc_values = spectral_convergence_batch(pred, raw_target)
            pred_np = pred.detach().cpu().numpy()
            target_np = raw_target.detach().cpu().numpy()
            for item_idx in range(pred_np.shape[0]):
                pred_item = pred_np[item_idx, 0]
                target_item = target_np[item_idx, 0]
                wave = normalized_waveform_metrics(pred_item, target_item)
                for key, value in wave.items():
                    values[key].append(value)
                values["sc"].append(sc_values[item_idx])

                target_env = extract_envelope(target_item, args.sample_rate)
                pred_env = extract_envelope(pred_item, args.sample_rate)
                target_env_norm = target_env / (np.max(target_env) + 1e-8)
                pred_env_norm = pred_env / (np.max(pred_env) + 1e-8)
                values["epcc"].append(pearson_np(pred_env_norm, target_env_norm))

                s1_target, _ = detect_heart_sounds(target_item, target_env, args.sample_rate)
                s1_pred, _ = detect_heart_sounds(pred_item, pred_env, args.sample_rate)
                hr_total += 1
                hr_target = estimate_heart_rate(s1_target, args.sample_rate)
                hr_pred = estimate_heart_rate(s1_pred, args.sample_rate)
                if hr_target is not None and hr_pred is not None:
                    hr_target_values.append(float(hr_target))
                    hr_bias_values.append(float(hr_pred - hr_target))

    target_hr_mean = safe_mean(hr_target_values)
    hr_sd = safe_std(hr_bias_values)
    nhr_sd = None if target_hr_mean is None or hr_sd is None else hr_sd / max(abs(target_hr_mean), 1e-8) * 100.0
    out = {key: safe_mean(vals) for key, vals in values.items()}
    out.update({f"{key}_std": safe_std(vals) for key, vals in values.items()})
    out.update(
        {
            "nhr_sd": nhr_sd,
            "hr_sd_bpm": hr_sd,
            "hr_target_mean_bpm": target_hr_mean,
            "hr_valid_count": len(hr_bias_values),
            "hr_total_count": hr_total,
            "n_segments": len(values["pcc"]),
        }
    )
    print(
        f"{desc}: PCC={out['pcc']:.4f}, NMSE={out['nmse']:.4f}, "
        f"NRMSE={out['nrmse']:.4f}, SC={out['sc']:.4f}, ePCC={out['epcc']:.4f}, "
        f"NHR-SD={out['nhr_sd'] if out['nhr_sd'] is not None else float('nan'):.2f}%"
    )
    return out


def metric_row(variant, fold_idx, metrics):
    return {
        "variant": variant,
        "fold": fold_idx,
        "PCC[%]": None if metrics.get("pcc") is None else metrics["pcc"] * 100.0,
        "NMSE": metrics.get("nmse"),
        "NMAE": metrics.get("nmae"),
        "NRMSE": metrics.get("nrmse"),
        "SC": metrics.get("sc"),
        "ePCC[%]": None if metrics.get("epcc") is None else metrics["epcc"] * 100.0,
        "NHR-SD[%]": metrics.get("nhr_sd"),
        "HR-SD[bpm]": metrics.get("hr_sd_bpm"),
        "n_segments": metrics.get("n_segments"),
        "hr_valid_count": metrics.get("hr_valid_count"),
        "hr_total_count": metrics.get("hr_total_count"),
    }


def aggregate_metric_rows(rows):
    out = []
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        item = {"variant": variant, "n_folds": len(subset)}
        for key in ["PCC[%]", "NMSE", "NMAE", "NRMSE", "SC", "ePCC[%]", "NHR-SD[%]", "HR-SD[bpm]"]:
            vals = [row.get(key) for row in subset]
            item[f"{key}_mean"] = safe_mean(vals)
            item[f"{key}_std"] = safe_std(vals)
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_one_variant_fold(args, variant, subjects, arrays, fold_idx, train_indices, val_indices, test_indices, device):
    fold_dir = args.output_dir / variant / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_model.pth"

    if args.reuse_existing and best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model = make_model(variant, args.base_channels).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        stats = checkpoint["normalization_stats"]
        return model, stats, float(checkpoint.get("best_val_corr", float("nan")))

    stats = compute_stats(arrays, train_indices, args.seed + fold_idx, args.stats_segments)
    (fold_dir / "normalization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    split_info = {
        "fold": fold_idx,
        "n_train_segments": len(train_indices),
        "n_val_segments": len(val_indices),
        "n_test_segments": len(test_indices),
        **summarize_subject_overlap(subjects, train_indices, val_indices, test_indices),
    }
    (fold_dir / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    train_loader = make_loader(SegmentDataset(subjects, arrays, train_indices, stats, augment=args.augment), args.batch_size, True, device, args.seed + fold_idx, args.num_workers)
    val_loader = make_loader(SegmentDataset(subjects, arrays, val_indices, stats, augment=False), args.batch_size, False, device, args.seed + fold_idx, args.num_workers)

    model = make_model(variant, args.base_channels).to(device)
    criterion = HeartSoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience, threshold=args.min_delta, min_lr=args.min_lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp
    best_val = -float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp, f"{variant} fold {fold_idx} train {epoch}/{args.epochs}")
        val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"{variant} fold {fold_idx} val")
        scheduler.step(val_metrics["corr"])
        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        (fold_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_metrics["corr"] > best_val + args.min_delta:
            best_val = val_metrics["corr"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalization_stats": stats,
                    "best_val_corr": best_val,
                    "variant": variant,
                    "fold": fold_idx,
                    "args": sanitize_args(args),
                },
                best_path,
            )
        else:
            bad_epochs += 1
        print(f"{variant} fold {fold_idx} epoch {epoch:03d}: val PCC {val_metrics['corr']:.4f}, patience {bad_epochs}")
        if args.early_stop > 0 and epoch >= args.min_epochs and bad_epochs >= args.early_stop:
            print(f"{variant} fold {fold_idx} early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["normalization_stats"], float(checkpoint["best_val_corr"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Subject-non-independent segment-mixed 5-fold model-structure ablation with normalized metrics."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", type=str, default="all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--early-stop", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0002)
    parser.add_argument("--stats-segments", type=int, default=4096)
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data_dir.is_absolute():
        args.data_dir = PROJECT_ROOT / args.data_dir
    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    subjects = discover_subjects(args.data_dir)
    arrays = [np.load(subject["file"], mmap_mode="r") for subject in subjects]
    all_indices = build_all_segment_indices(subjects)
    folds = build_segment_folds(all_indices, args.folds, args.seed)
    variants = parse_list(args.variants)
    max_folds = args.folds if args.max_folds <= 0 else min(args.max_folds, args.folds)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Variants: {variants}")

    run_config = sanitize_args(args)
    run_config["variants"] = variants
    run_config["variant_descriptions"] = {name: {k: v for k, v in cfg.items() if k != "factory"} for name, cfg in MODEL_VARIANTS.items()}
    run_config["n_subjects"] = len(subjects)
    run_config["n_segments"] = len(all_indices)
    run_config["split_mode"] = "segment_mixed_non_subject_independent"
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    fold_metric_rows = []
    for variant in variants:
        for fold_idx in range(max_folds):
            test_indices = folds[fold_idx]
            train_val_indices = [item for i, fold in enumerate(folds) if i != fold_idx for item in fold]
            train_indices, val_indices = split_train_val(train_val_indices, args.val_ratio, args.seed + fold_idx)
            model, stats, best_val = train_one_variant_fold(
                args, variant, subjects, arrays, fold_idx, train_indices, val_indices, test_indices, device
            )
            metrics = evaluate_normalized_metrics(model, subjects, arrays, test_indices, stats, args, device, f"{variant} fold {fold_idx} test")
            row = metric_row(variant, fold_idx, metrics)
            row["best_val_corr"] = best_val
            fold_metric_rows.append(row)
            variant_dir = args.output_dir / variant / f"fold_{fold_idx}"
            (variant_dir / "normalized_test_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    summary_rows = aggregate_metric_rows(fold_metric_rows)
    write_csv(args.output_dir / "fold_normalized_metrics.csv", fold_metric_rows)
    write_csv(args.output_dir / "segment_mixed_ablation_normalized_summary.csv", summary_rows)
    (args.output_dir / "segment_mixed_ablation_normalized_summary.json").write_text(
        json.dumps({"fold_rows": fold_metric_rows, "summary": summary_rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved summary to: {args.output_dir / 'segment_mixed_ablation_normalized_summary.csv'}")


if __name__ == "__main__":
    main()

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
from test_continual_snr_robustness import NoisySegmentDataset, parse_str_list  # noqa: E402
from train_compare_reconstruction_methods import (  # noqa: E402
    DETAILED_METRIC_KEYS,
    DetailedMetricSegmentDataset,
    aggregate_detailed_rows,
    calculate_heart_sound_metrics,
    detect_heart_sounds,
    evaluate_detailed_metrics,
    extract_envelope,
    flatten_detailed,
    segment_heart_sounds,
    signal_metrics_np,
    spectral_convergence_batch,
)
from train_continual_subject5fold_adaptation_optimized import (  # noqa: E402
    build_balanced_subject_folds,
    sanitize_args,
)
from train_segment_mixed_5fold_recovery import (  # noqa: E402
    build_all_segment_indices,
    build_segment_folds,
    split_train_val,
)


DEFAULT_SEGMENT_RESULTS = PROJECT_ROOT / "results" / "segment_mixed_5fold_recovery"
DEFAULT_OPTIMIZED_RESULTS = PROJECT_ROOT / "results" / "continual_subject5fold_adaptation_optimized"
DEFAULT_SNR_RESULTS = PROJECT_ROOT / "results" / "continual_snr_robustness"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "missing_requested_metrics_full"


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def safe_number(value):
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


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(checkpoint_path: Path, device, base_channels: int):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HeartSoundUNet(base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


class NoisyDetailedMetricSegmentDataset(NoisySegmentDataset):
    def __getitem__(self, idx):
        x, y, sample_id = super().__getitem__(idx)
        file_idx, segment_idx = self.indices[idx]
        raw_target = np.asarray(self.arrays[file_idx][segment_idx][0], dtype=np.float32)
        local_mean = float(np.median(raw_target))
        local_std = float(np.median(np.abs(raw_target - local_mean)) * 1.4826 + 1e-8)
        return x, y, torch.from_numpy(raw_target[None, :]), sample_id, torch.tensor([local_mean, local_std], dtype=torch.float32)


def denorm_prediction(normalized: torch.Tensor, params: torch.Tensor, stats: dict) -> torch.Tensor:
    local_mean = params[:, 0].to(normalized.device).view(-1, 1, 1)
    local_std = params[:, 1].to(normalized.device).view(-1, 1, 1)
    global_mean = torch.tensor(float(stats["target_mean"]), device=normalized.device).view(1, 1, 1)
    global_std = torch.tensor(float(stats["target_std"]), device=normalized.device).view(1, 1, 1)
    scale = 0.7 / local_std + 0.3 / global_std
    shift = 0.7 * local_mean / local_std + 0.3 * global_mean / global_std
    return (normalized + shift) / scale


def estimate_heart_rate(s1_indices, sample_rate: int = 1000):
    if len(s1_indices) < 2:
        return None
    intervals = np.diff(np.array(s1_indices, dtype=np.float64)) / sample_rate
    std = float(np.std(intervals))
    if std > 1e-8:
        intervals = intervals[np.abs(intervals - np.mean(intervals)) < 2 * std]
    if len(intervals) < 1:
        return None
    avg_interval = float(np.mean(intervals))
    return None if avg_interval <= 1e-8 else 60.0 / avg_interval


def evaluate_detailed_dataset(model, dataset, stats, args, device, desc: str):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    values = {
        key: []
        for key in DETAILED_METRIC_KEYS
        if key not in {"valid_hr_count", "total_hr_count", "mean_hr_bias", "std_hr_bias"}
    }
    hr_biases = []
    total_hr_count = 0
    with torch.no_grad():
        for x, _y, raw_target, _sid, params in loader:
            x = x.to(device, non_blocking=True)
            raw_target = raw_target.to(device, non_blocking=True)
            params = params.to(device, non_blocking=True)
            pred = denorm_prediction(model(x), params, stats)
            target = raw_target
            spec_vals = spectral_convergence_batch(pred, target)
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            for item_idx in range(pred_np.shape[0]):
                pred_item = pred_np[item_idx, 0]
                target_item = target_np[item_idx, 0]
                sig = signal_metrics_np(target_item, pred_item)
                values["corr"].append(sig["corr"])
                values["mse"].append(sig["mse"])
                values["mae"].append(sig["mae"])
                values["rmse"].append(sig["rmse"])
                values["spectral_convergence"].append(spec_vals[item_idx])

                target_env = extract_envelope(target_item, args.sample_rate)
                pred_env = extract_envelope(pred_item, args.sample_rate)
                target_env_norm = target_env / (np.max(target_env) + 1e-8)
                pred_env_norm = pred_env / (np.max(pred_env) + 1e-8)
                env = signal_metrics_np(target_env_norm, pred_env_norm)
                values["envelope_corr"].append(env["corr"])
                values["envelope_mse"].append(env["mse"])
                values["envelope_mae"].append(env["mae"])
                values["envelope_rmse"].append(env["rmse"])

                s1_target, s2_target = detect_heart_sounds(target_item, target_env, args.sample_rate)
                s1_pred, s2_pred = detect_heart_sounds(pred_item, pred_env, args.sample_rate)
                s1_metrics = calculate_heart_sound_metrics(
                    segment_heart_sounds(target_item, s1_target, args.heart_sound_window),
                    segment_heart_sounds(pred_item, s1_pred, args.heart_sound_window),
                )
                s2_metrics = calculate_heart_sound_metrics(
                    segment_heart_sounds(target_item, s2_target, args.heart_sound_window),
                    segment_heart_sounds(pred_item, s2_pred, args.heart_sound_window),
                )
                for prefix, item in [("s1", s1_metrics), ("s2", s2_metrics)]:
                    values[f"{prefix}_corr"].append(item["corr"])
                    values[f"{prefix}_mse"].append(item["mse"])
                    values[f"{prefix}_mae"].append(item["mae"])
                    values[f"{prefix}_rmse"].append(item["rmse"])
                    values[f"{prefix}_count"].append(item["count"])

                total_hr_count += 1
                hr_target = estimate_heart_rate(s1_target, args.sample_rate)
                hr_pred = estimate_heart_rate(s1_pred, args.sample_rate)
                if hr_target is not None and hr_pred is not None:
                    hr_biases.append(float(hr_pred - hr_target))

    out = {key: safe_mean(vals) for key, vals in values.items()}
    out["valid_hr_count"] = len(hr_biases)
    out["total_hr_count"] = total_hr_count
    out["mean_hr_bias"] = safe_mean(hr_biases)
    out["std_hr_bias"] = safe_std(hr_biases)
    print(
        f"{desc}: PCC={out['corr']:.4f}, MSE={out['mse']:.6f}, "
        f"ePCC={out['envelope_corr']:.4f}, HR={out['valid_hr_count']}/{out['total_hr_count']}"
    )
    return out


def adapt_model(args, model, support_train_set, support_val_set, device, sample_id):
    criterion = HeartSoundLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -float("inf")
    history = []
    for stage_name, epochs, lr in [
        ("decoder_head", args.stage1_epochs, args.stage1_lr),
        ("full", args.stage2_epochs, args.stage2_lr),
    ]:
        if epochs <= 0:
            continue
        set_trainable(model, stage_name)
        trainable = [param for param in model.parameters() if param.requires_grad]
        if not trainable:
            continue
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.adapt_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=args.adapt_lr_patience,
            min_lr=args.adapt_min_lr,
        )
        train_loader = make_loader(
            support_train_set,
            args.adapt_batch_size,
            True,
            device,
            args.seed + sample_id,
            args.num_workers,
        )
        val_loader = make_loader(
            support_val_set,
            args.adapt_batch_size,
            False,
            device,
            args.seed + sample_id,
            args.num_workers,
        )
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scaler,
                use_amp,
                f"S{sample_id} {stage_name} {epoch}/{epochs}",
            )
            val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"S{sample_id} val")
            scheduler.step(val_metrics["corr"])
            history.append(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "train_corr": train_metrics["corr"],
                    "val_corr": val_metrics["corr"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            if val_metrics["corr"] > best_val + args.min_delta:
                best_val = val_metrics["corr"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_val, history


def aggregate_phase_rows(rows, group_keys):
    groups = {}
    for row in rows:
        key = tuple(row.get(item) for item in group_keys)
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, subset in sorted(groups.items(), key=lambda item: tuple("" if x is None else str(x) for x in item[0])):
        out = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        out["n_rows"] = len(subset)
        if "support_segments" not in group_keys:
            vals = [row.get("support_segments") for row in subset]
            out["actual_support_segments_mean"] = safe_mean(vals)
            out["actual_support_segments_std"] = safe_std(vals)
            numeric_vals = [int(v) for v in vals if v is not None]
            out["actual_support_segments_min"] = min(numeric_vals) if numeric_vals else None
            out["actual_support_segments_max"] = max(numeric_vals) if numeric_vals else None
        for phase in ["before", "after", "delta"]:
            for metric in DETAILED_METRIC_KEYS:
                vals = [row.get(f"{phase}_{metric}") for row in subset]
                out[f"{phase}_{metric}_mean"] = safe_mean(vals)
                out[f"{phase}_{metric}_std"] = safe_std(vals)
        summaries.append(out)
    return summaries


def compact_requested_row(row, model, protocol, stage, support=None, noise_type=None, snr_db=None):
    prefix = f"{stage}_"
    return {
        "model": model,
        "protocol": protocol,
        "stage": stage,
        "support_segments": support,
        "noise_type": noise_type,
        "snr_db": snr_db,
        "PCC[%]": None if row.get(prefix + "corr_mean") is None else row[prefix + "corr_mean"] * 100,
        "MSE": row.get(prefix + "mse_mean"),
        "MAE": row.get(prefix + "mae_mean"),
        "RMSE": row.get(prefix + "rmse_mean"),
        "SC": row.get(prefix + "spectral_convergence_mean"),
        "ePCC[%]": None if row.get(prefix + "envelope_corr_mean") is None else row[prefix + "envelope_corr_mean"] * 100,
        "Mean HR [bpm]": row.get(prefix + "mean_hr_bias_mean"),
        "SD [bpm]": row.get(prefix + "std_hr_bias_mean"),
        "valid_hr_count": row.get(prefix + "valid_hr_count_mean"),
        "total_hr_count": row.get(prefix + "total_hr_count_mean"),
    }


def run_segment_mixed(args, subjects, arrays, device):
    rows = []
    all_indices = build_all_segment_indices(subjects)
    folds = build_segment_folds(all_indices, args.folds, args.seed)
    max_folds = args.folds if args.max_folds <= 0 else min(args.max_folds, args.folds)
    out_dir = args.output_dir / "segment_mixed_5fold_recovery"
    for fold_idx in range(max_folds):
        checkpoint_path = args.segment_results_dir / f"fold_{fold_idx}" / "best_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        test_indices = folds[fold_idx]
        model, checkpoint = load_model(checkpoint_path, device, args.base_channels)
        stats = checkpoint.get("normalization_stats")
        if stats is None:
            stats = load_json(args.segment_results_dir / f"fold_{fold_idx}" / "normalization_stats.json")
        detailed = evaluate_detailed_metrics(
            model,
            subjects,
            arrays,
            test_indices,
            stats,
            args,
            device,
            f"segment_mixed fold {fold_idx}",
        )
        row = {"fold": fold_idx, **flatten_detailed("detailed", detailed)}
        rows.append(row)
        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "detailed_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    aggregate = aggregate_detailed_rows(rows, "detailed")
    summary = {"rows": rows, "aggregate": aggregate}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "fold_metrics.csv", rows)
    compact = [
        {
            "model": "segment_mixed_5fold_recovery",
            "protocol": "segment_mixed",
            "stage": "test",
            "PCC[%]": aggregate["detailed_corr_mean"] * 100,
            "MSE": aggregate["detailed_mse_mean"],
            "MAE": aggregate["detailed_mae_mean"],
            "RMSE": aggregate["detailed_rmse_mean"],
            "SC": aggregate["detailed_spectral_convergence_mean"],
            "ePCC[%]": aggregate["detailed_envelope_corr_mean"] * 100,
            "Mean HR [bpm]": aggregate["detailed_mean_hr_bias_mean"],
            "SD [bpm]": aggregate["detailed_std_hr_bias_mean"],
            "valid_hr_count": aggregate["detailed_valid_hr_count_mean"],
            "total_hr_count": aggregate["detailed_total_hr_count_mean"],
        }
    ]
    write_csv(out_dir / "requested_metrics.csv", compact)
    return {"summary": summary, "requested": compact}


def evaluate_subject_clean(args, subjects, arrays, fold_idx, sample_id, support_segments, base_model_path, device):
    model, checkpoint = load_model(base_model_path, device, args.base_channels)
    stats = checkpoint["normalization_stats"]
    support, query = choose_support(
        subjects,
        arrays,
        sample_id,
        support_segments,
        args.min_query_segments,
        args.seed,
        args.support_strategy,
    )
    support_train, support_val = split_support(
        support,
        args.seed + sample_id + support_segments,
        args.support_val_ratio,
        args.support_val_segments,
    )
    query_set = DetailedMetricSegmentDataset(subjects, arrays, query, stats, augment=False)
    before = evaluate_detailed_dataset(model, query_set, stats, args, device, f"optimized fold {fold_idx} S{sample_id} before")
    support_train_set = SegmentDataset(subjects, arrays, support_train, stats, augment=args.adapt_augment)
    support_val_set = SegmentDataset(subjects, arrays, support_val, stats, augment=False)
    best_val, history = adapt_model(args, model, support_train_set, support_val_set, device, sample_id)
    after = evaluate_detailed_dataset(model, query_set, stats, args, device, f"optimized fold {fold_idx} S{sample_id} after")
    row = {
        "fold": fold_idx,
        "sample_id": sample_id,
        "requested_support_segments": support_segments,
        "support_segments": len(support),
        "query_segments": len(query),
        "support_seconds": len(support) * args.segment_seconds,
        "best_support_val_corr": best_val,
        **flatten_detailed("before", before),
        **flatten_detailed("after", after),
    }
    for key in DETAILED_METRIC_KEYS:
        before_val = row.get(f"before_{key}")
        after_val = row.get(f"after_{key}")
        row[f"delta_{key}"] = None if before_val is None or after_val is None else after_val - before_val
    return row, {"row": row, "history": history}


def run_optimized(args, subjects, arrays, device):
    result_config = load_json(args.optimized_results_dir / "run_config.json")
    folds = result_config.get("fold_test_ids") or build_balanced_subject_folds(subjects, args.folds)
    support_list = args.support_list
    rows = []
    max_folds = len(folds) if args.max_folds <= 0 else min(args.max_folds, len(folds))
    out_dir = args.output_dir / "continual_subject5fold_adaptation_optimized"
    for fold_idx in range(max_folds):
        base_model_path = args.optimized_results_dir / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
        if not base_model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {base_model_path}")
        test_ids = sorted(int(item) for item in folds[fold_idx])
        if args.max_subjects > 0:
            test_ids = test_ids[: args.max_subjects]
        for support_segments in support_list:
            support_dir = out_dir / f"fold_{fold_idx}" / f"support_{support_segments:03d}"
            support_dir.mkdir(parents=True, exist_ok=True)
            for sample_id in test_ids:
                row, detail = evaluate_subject_clean(
                    args, subjects, arrays, fold_idx, sample_id, support_segments, base_model_path, device
                )
                rows.append(row)
                (support_dir / f"sample{sample_id:02d}_detailed_metrics.json").write_text(
                    json.dumps(detail, indent=2), encoding="utf-8"
                )
    summary = aggregate_phase_rows(rows, ["requested_support_segments"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "subject_metrics.csv", rows)
    write_csv(out_dir / "support_summary.csv", summary)
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    compact = []
    for item in summary:
        for stage in ["before", "after"]:
            compact.append(
                compact_requested_row(
                    item,
                    "continual_subject5fold_adaptation_optimized",
                    "continual_subject_5fold",
                    stage,
                    support=item["requested_support_segments"],
                )
            )
    write_csv(out_dir / "requested_metrics.csv", compact)
    return {"summary": summary, "requested": compact}


def evaluate_subject_snr(args, subjects, arrays, fold_idx, sample_id, support_segments, base_model_path, device):
    model, checkpoint = load_model(base_model_path, device, args.base_channels)
    stats = checkpoint["normalization_stats"]
    support, query = choose_support(
        subjects,
        arrays,
        sample_id,
        support_segments,
        args.min_query_segments,
        args.seed,
        args.support_strategy,
    )
    support_train, support_val = split_support(
        support,
        args.seed + sample_id + support_segments,
        args.support_val_ratio,
        args.support_val_segments,
    )
    before_by_condition = {}
    for noise_type in args.noise_types:
        for snr_db in args.snr_list:
            dataset = NoisyDetailedMetricSegmentDataset(
                subjects,
                arrays,
                query,
                stats,
                noise_type=noise_type,
                snr_db=snr_db,
                seed=args.seed + sample_id + support_segments + int(float(snr_db) * 17),
                babble_sources=args.babble_sources,
            )
            before_by_condition[(noise_type, snr_db)] = evaluate_detailed_dataset(
                model,
                dataset,
                stats,
                args,
                device,
                f"SNR fold {fold_idx} S{sample_id} {noise_type} {snr_db:g}dB before",
            )

    support_train_set = SegmentDataset(subjects, arrays, support_train, stats, augment=args.adapt_augment)
    support_val_set = SegmentDataset(subjects, arrays, support_val, stats, augment=False)
    best_val, history = adapt_model(args, model, support_train_set, support_val_set, device, sample_id)

    rows = []
    for noise_type in args.noise_types:
        for snr_db in args.snr_list:
            dataset = NoisyDetailedMetricSegmentDataset(
                subjects,
                arrays,
                query,
                stats,
                noise_type=noise_type,
                snr_db=snr_db,
                seed=args.seed + sample_id + support_segments + int(float(snr_db) * 17),
                babble_sources=args.babble_sources,
            )
            after = evaluate_detailed_dataset(
                model,
                dataset,
                stats,
                args,
                device,
                f"SNR fold {fold_idx} S{sample_id} {noise_type} {snr_db:g}dB after",
            )
            before = before_by_condition[(noise_type, snr_db)]
            row = {
                "fold": fold_idx,
                "sample_id": sample_id,
                "requested_support_segments": support_segments,
                "support_segments": len(support),
                "query_segments": len(query),
                "support_seconds": len(support) * args.segment_seconds,
                "noise_type": noise_type,
                "snr_db": snr_db,
                "best_support_val_corr": best_val,
                **flatten_detailed("before", before),
                **flatten_detailed("after", after),
            }
            for key in DETAILED_METRIC_KEYS:
                before_val = row.get(f"before_{key}")
                after_val = row.get(f"after_{key}")
                row[f"delta_{key}"] = None if before_val is None or after_val is None else after_val - before_val
            rows.append(row)
    return rows, {"rows": rows, "history": history}


def run_snr(args, subjects, arrays, device):
    result_config = load_json(args.snr_results_dir / "run_config.json")
    folds = result_config.get("fold_test_ids") or build_balanced_subject_folds(subjects, args.folds)
    rows = []
    max_folds = len(folds) if args.max_folds <= 0 else min(args.max_folds, len(folds))
    out_dir = args.output_dir / "continual_snr_robustness"
    base_results = args.snr_base_results_dir or args.optimized_results_dir
    for fold_idx in range(max_folds):
        base_model_path = base_results / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
        if not base_model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {base_model_path}")
        test_ids = sorted(int(item) for item in folds[fold_idx])
        if args.max_subjects > 0:
            test_ids = test_ids[: args.max_subjects]
        for support_segments in args.support_list:
            support_dir = out_dir / f"fold_{fold_idx}" / f"support_{support_segments:03d}"
            support_dir.mkdir(parents=True, exist_ok=True)
            for sample_id in test_ids:
                sample_rows, detail = evaluate_subject_snr(
                    args, subjects, arrays, fold_idx, sample_id, support_segments, base_model_path, device
                )
                rows.extend(sample_rows)
                (support_dir / f"sample{sample_id:02d}_snr_detailed_metrics.json").write_text(
                    json.dumps(detail, indent=2), encoding="utf-8"
                )
    summary = aggregate_phase_rows(rows, ["requested_support_segments", "noise_type", "snr_db"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "condition_subject_metrics.csv", rows)
    write_csv(out_dir / "condition_summary.csv", summary)
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    compact = []
    for item in summary:
        for stage in ["before", "after"]:
            compact.append(
                compact_requested_row(
                    item,
                    "continual_snr_robustness",
                    "continual_snr_robustness",
                    stage,
                    support=item["requested_support_segments"],
                    noise_type=item["noise_type"],
                    snr_db=item["snr_db"],
                )
            )
    write_csv(out_dir / "requested_metrics.csv", compact)
    return {"summary": summary, "requested": compact}


def parse_args():
    parser = argparse.ArgumentParser(description="Re-evaluate missing detailed metrics for paper tables.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--segment-results-dir", type=Path, default=DEFAULT_SEGMENT_RESULTS)
    parser.add_argument("--optimized-results-dir", type=Path, default=DEFAULT_OPTIMIZED_RESULTS)
    parser.add_argument("--snr-results-dir", type=Path, default=DEFAULT_SNR_RESULTS)
    parser.add_argument("--snr-base-results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["segment", "optimized", "snr", "all"], default="all")
    parser.add_argument("--support-list", type=str, default="48,72,96")
    parser.add_argument("--noise-types", type=str, default="white,pink,babble")
    parser.add_argument("--snr-list", type=str, default="20,10,5,0")
    parser.add_argument("--support-strategy", choices=["random", "top_quality", "stratified_quality", "hybrid_quality"], default="stratified_quality")
    parser.add_argument("--support-val-ratio", type=float, default=0.25)
    parser.add_argument("--support-val-segments", type=int, default=8)
    parser.add_argument("--min-query-segments", type=int, default=20)
    parser.add_argument("--segment-seconds", type=float, default=2.5)
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--heart-sound-window", type=int, default=100)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-epochs", type=int, default=70)
    parser.add_argument("--stage2-lr", type=float, default=8e-5)
    parser.add_argument("--adapt-lr-patience", type=int, default=8)
    parser.add_argument("--adapt-min-lr", type=float, default=5e-6)
    parser.add_argument("--adapt-weight-decay", type=float, default=0.0)
    parser.add_argument("--min-delta", type=float, default=0.0002)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--adapt-batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--babble-sources", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapt-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def resolve_paths(args):
    for key in ["data_dir", "segment_results_dir", "optimized_results_dir", "snr_results_dir", "output_dir", "snr_base_results_dir"]:
        value = getattr(args, key)
        if value is None:
            continue
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        setattr(args, key, value.resolve())


def main():
    args = parse_args()
    resolve_paths(args)
    args.support_list = parse_int_list(args.support_list)
    args.noise_types = parse_str_list(args.noise_types)
    args.snr_list = parse_int_list(args.snr_list)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    subjects = discover_subjects(args.data_dir)
    arrays = [np.load(subject["file"], mmap_mode="r") for subject in subjects]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded {len(subjects)} subjects from {args.data_dir}")
    print(f"Output: {args.output_dir}")

    run_config = sanitize_args(args)
    run_config["note"] = (
        "Optimized and SNR after metrics are recomputed by rerunning per-subject support adaptation, "
        "because adapted per-subject checkpoints were not saved in the original result directories."
    )
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    final = {}
    requested_rows = []
    if args.mode in {"segment", "all"}:
        final["segment_mixed_5fold_recovery"] = run_segment_mixed(args, subjects, arrays, device)
        requested_rows.extend(final["segment_mixed_5fold_recovery"]["requested"])
    if args.mode in {"optimized", "all"}:
        final["continual_subject5fold_adaptation_optimized"] = run_optimized(args, subjects, arrays, device)
        requested_rows.extend(final["continual_subject5fold_adaptation_optimized"]["requested"])
    if args.mode in {"snr", "all"}:
        final["continual_snr_robustness"] = run_snr(args, subjects, arrays, device)
        requested_rows.extend(final["continual_snr_robustness"]["requested"])

    write_csv(args.output_dir / "requested_metrics_all.csv", requested_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"Saved requested metric table to: {args.output_dir / 'requested_metrics_all.csv'}")


if __name__ == "__main__":
    main()

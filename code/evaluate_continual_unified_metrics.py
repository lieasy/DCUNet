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
    choose_support,
    discover_subjects,
    make_loader,
    run_epoch,
    set_seed,
    set_trainable,
    split_support,
)
from test_continual_snr_robustness import NoisySegmentDataset, parse_str_list  # noqa: E402
from train_continual_model_ablation_suite import MODEL_VARIANTS, make_model, parse_list  # noqa: E402
from train_continual_subject5fold_adaptation_optimized import sanitize_args  # noqa: E402


DEFAULT_ABLATION_RESULTS = PROJECT_ROOT / "results" / "continual_model_ablation_suite"
DEFAULT_SNR_RESULTS = PROJECT_ROOT / "results" / "continual_snr_robustness"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "continual_unified_metrics"

METRIC_KEYS = [
    "pcc",
    "mae",
    "rmse",
    "spectral_convergence",
    "envelope_pcc",
    "mean_hr_bias",
    "std_hr_bias",
]

METRIC_LABELS = {
    "pcc": "Pearson相关系数 (PCC)",
    "mae": "平均绝对误差 (MAE)",
    "rmse": "均方根误差 (RMSE)",
    "spectral_convergence": "频谱收敛度",
    "envelope_pcc": "包络PCC",
    "mean_hr_bias": "平均心率偏差",
}
METRIC_LABELS["std_hr_bias"] = "HR-SD [bpm]"


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def safe_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def safe_mean(values):
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else None


def safe_std(values):
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    return float(np.std(arr)) if arr.size else None


def fmt_metric(value):
    return "nan" if value is None else f"{value:.4f}"


def pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.std(a) <= 1e-8 or np.std(b) <= 1e-8:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))


def hilbert_envelope_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    spectrum = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * h))


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if len(x) <= window:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(x, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def extract_envelope(signal: np.ndarray, sr: int = 1000, cutoff: float = 20.0) -> np.ndarray:
    envelope = hilbert_envelope_np(signal)
    # Approximate the old Hilbert + 20 Hz low-pass behavior without scipy.
    return moving_average(envelope, max(3, round(sr / cutoff)))


def simple_find_peaks(x: np.ndarray, height: float, distance: int, prominence: float) -> np.ndarray:
    peaks = []
    last_peak = -distance
    for idx in range(1, len(x) - 1):
        if x[idx] < height or x[idx] <= x[idx - 1] or x[idx] < x[idx + 1]:
            continue
        left = max(0, idx - distance // 2)
        right = min(len(x), idx + distance // 2 + 1)
        local_base = max(float(np.min(x[left:idx + 1])), float(np.min(x[idx:right])))
        if x[idx] - local_base < prominence:
            continue
        if idx - last_peak < distance and peaks:
            if x[idx] > x[peaks[-1]]:
                peaks[-1] = idx
                last_peak = idx
            continue
        peaks.append(idx)
        last_peak = idx
    return np.array(peaks, dtype=np.int64)


def refine_heart_sound_labels(peak_indices, s1_candidates, s2_candidates):
    all_peaks = sorted(int(x) for x in peak_indices)
    labels = []
    for peak in all_peaks:
        if peak in s1_candidates:
            labels.append("S1")
        elif peak in s2_candidates:
            labels.append("S2")
        else:
            labels.append("unknown")
    for idx in range(1, len(labels)):
        if labels[idx] == labels[idx - 1] and labels[idx] != "unknown":
            labels[idx - 1] = "S2" if labels[idx - 1] == "S1" else "S1"
    s1_indices = [all_peaks[idx] for idx, label in enumerate(labels) if label == "S1"]
    s2_indices = [all_peaks[idx] for idx, label in enumerate(labels) if label == "S2"]
    return s1_indices, s2_indices


def detect_heart_sounds(signal: np.ndarray, envelope: np.ndarray, sr: int = 1000):
    del signal
    envelope_norm = envelope / (np.max(envelope) + 1e-8)
    peak_indices = simple_find_peaks(envelope_norm, height=0.2, distance=sr // 4, prominence=0.1)
    if len(peak_indices) < 2:
        return [], []
    peak_values = envelope_norm[peak_indices]
    mean_peak = float(np.mean(peak_values))
    s1_candidates = {int(idx) for idx, val in zip(peak_indices, peak_values) if val > mean_peak * 1.1}
    s2_candidates = {int(idx) for idx, val in zip(peak_indices, peak_values) if mean_peak * 0.5 < val <= mean_peak * 1.1}
    return refine_heart_sound_labels(peak_indices, s1_candidates, s2_candidates)


def estimate_heart_rate(s1_indices, sr: int = 1000):
    if len(s1_indices) < 2:
        return None
    intervals = np.diff(np.array(s1_indices, dtype=np.float64)) / sr
    std = float(np.std(intervals))
    if std > 1e-8:
        intervals = intervals[np.abs(intervals - np.mean(intervals)) < 2 * std]
    if len(intervals) < 1:
        return None
    avg_interval = float(np.mean(intervals))
    return None if avg_interval <= 1e-8 else 60.0 / avg_interval


def spectral_convergence_batch(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 1024, hop_length: int = 256) -> list[float]:
    pred_2d = pred.squeeze(1)
    target_2d = target.squeeze(1)
    n_fft = min(n_fft, pred_2d.shape[-1])
    window = torch.hann_window(n_fft, device=pred.device)
    pred_spec = torch.stft(pred_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True).abs()
    target_spec = torch.stft(target_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True).abs()
    numerator = torch.linalg.vector_norm(pred_spec - target_spec, dim=(1, 2))
    denominator = torch.linalg.vector_norm(target_spec, dim=(1, 2)).clamp_min(1e-8)
    return (numerator / denominator).detach().cpu().numpy().astype(float).tolist()


class MetricSegmentDataset(SegmentDataset):
    def __getitem__(self, idx):
        x, y, sample_id = super().__getitem__(idx)
        file_idx, segment_idx = self.indices[idx]
        raw_target = np.asarray(self.arrays[file_idx][segment_idx][0], dtype=np.float32)
        local_mean = float(np.median(raw_target))
        local_std = float(np.median(np.abs(raw_target - local_mean)) * 1.4826 + 1e-8)
        return x, y, torch.from_numpy(raw_target[None, :]), sample_id, torch.tensor([local_mean, local_std], dtype=torch.float32)


class NoisyMetricSegmentDataset(NoisySegmentDataset):
    def __getitem__(self, idx):
        x, y, sample_id = super().__getitem__(idx)
        file_idx, segment_idx = self.indices[idx]
        raw_target = np.asarray(self.arrays[file_idx][segment_idx][0], dtype=np.float32)
        local_mean = float(np.median(raw_target))
        local_std = float(np.median(np.abs(raw_target - local_mean)) * 1.4826 + 1e-8)
        return x, y, torch.from_numpy(raw_target[None, :]), sample_id, torch.tensor([local_mean, local_std], dtype=torch.float32)


def denorm_target(normalized: torch.Tensor, raw_target: torch.Tensor, params: torch.Tensor, stats: dict) -> torch.Tensor:
    local_mean = params[:, 0].to(normalized.device).view(-1, 1, 1)
    local_std = params[:, 1].to(normalized.device).view(-1, 1, 1)
    global_mean = torch.tensor(float(stats["target_mean"]), device=normalized.device).view(1, 1, 1)
    global_std = torch.tensor(float(stats["target_std"]), device=normalized.device).view(1, 1, 1)
    scale = 0.7 / local_std + 0.3 / global_std
    shift = 0.7 * local_mean / local_std + 0.3 * global_mean / global_std
    denormed = (normalized + shift) / scale
    # Keep the target exactly equal to the original waveform, avoiding tiny numeric drift.
    return raw_target.to(normalized.device) if normalized.shape == raw_target.shape else denormed


def denorm_prediction(normalized: torch.Tensor, params: torch.Tensor, stats: dict) -> torch.Tensor:
    local_mean = params[:, 0].to(normalized.device).view(-1, 1, 1)
    local_std = params[:, 1].to(normalized.device).view(-1, 1, 1)
    global_mean = torch.tensor(float(stats["target_mean"]), device=normalized.device).view(1, 1, 1)
    global_std = torch.tensor(float(stats["target_std"]), device=normalized.device).view(1, 1, 1)
    scale = 0.7 / local_std + 0.3 / global_std
    shift = 0.7 * local_mean / local_std + 0.3 * global_mean / global_std
    return (normalized + shift) / scale


def make_metric_loader(dataset, batch_size, shuffle, device, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )


def evaluate_full_metrics(model, dataset, stats, args, device, desc: str):
    model.eval()
    loader = make_metric_loader(dataset, args.batch_size, False, device, args.seed, args.num_workers)
    values = {key: [] for key in METRIC_KEYS if key != "std_hr_bias"}
    hr_pred_values = []
    hr_target_values = []
    hr_bias_values = []
    with torch.no_grad():
        for x, _y, raw_target, _sid, params in loader:
            x = x.to(device, non_blocking=True)
            raw_target = raw_target.to(device, non_blocking=True)
            params = params.to(device, non_blocking=True)
            pred_norm = model(x)
            pred = denorm_prediction(pred_norm, params, stats)
            target = denorm_target(pred_norm, raw_target, params, stats)
            spec_vals = spectral_convergence_batch(pred, target)
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            for item_idx in range(pred_np.shape[0]):
                p = pred_np[item_idx, 0]
                t = target_np[item_idx, 0]
                mse = float(np.mean((p - t) ** 2))
                values["pcc"].append(pearson_np(p, t))
                values["mae"].append(float(np.mean(np.abs(p - t))))
                values["rmse"].append(float(np.sqrt(mse)))
                values["spectral_convergence"].append(spec_vals[item_idx])
                pred_env = extract_envelope(p, args.sample_rate)
                target_env = extract_envelope(t, args.sample_rate)
                pred_env_norm = pred_env / (np.max(pred_env) + 1e-8)
                target_env_norm = target_env / (np.max(target_env) + 1e-8)
                values["envelope_pcc"].append(pearson_np(pred_env_norm, target_env_norm))
                s1_pred, _ = detect_heart_sounds(p, pred_env, args.sample_rate)
                s1_target, _ = detect_heart_sounds(t, target_env, args.sample_rate)
                hr_pred = estimate_heart_rate(s1_pred, args.sample_rate)
                hr_target = estimate_heart_rate(s1_target, args.sample_rate)
                if hr_pred is not None:
                    hr_pred_values.append(hr_pred)
                if hr_target is not None:
                    hr_target_values.append(hr_target)
                hr_bias = None if hr_pred is None or hr_target is None else abs(hr_pred - hr_target)
                values["mean_hr_bias"].append(hr_bias)
                if hr_bias is not None:
                    hr_bias_values.append(hr_bias)
    out = {key: safe_mean(values[key]) for key in values}
    out["std_hr_bias"] = safe_std(hr_bias_values)
    out["n_segments"] = len(values["pcc"])
    out["hr_pred_mean"] = safe_mean(hr_pred_values)
    out["hr_target_mean"] = safe_mean(hr_target_values)
    print(f"{desc}: PCC={fmt_metric(out['pcc'])} MAE={fmt_metric(out['mae'])} RMSE={fmt_metric(out['rmse'])}")
    return out


def adapt_model(args, model, support_train_set, support_val_set, criterion, device, sample_id):
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -float("inf")
    history = []
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
            tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp, f"S{sample_id} {stage_name} {epoch}/{epochs}")
            va = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"S{sample_id} support-val")
            scheduler.step(va["corr"])
            history.append({"stage": stage_name, "epoch": epoch, "train_corr": tr["corr"], "val_corr": va["corr"], "lr": optimizer.param_groups[0]["lr"]})
            if va["corr"] > best_val + args.min_delta:
                best_val = va["corr"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_val, history


def wide_row(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": safe_float(metrics.get(key)) for key in METRIC_KEYS}


def evaluate_subject_clean(args, variant, fold_idx, sample_id, support_segments, subjects, arrays, base_model_path, device):
    checkpoint = torch.load(base_model_path, map_location=device)
    stats = checkpoint["normalization_stats"]
    support, query = choose_support(subjects, arrays, sample_id, support_segments, args.min_query_segments, args.seed, args.support_strategy)
    support_train, support_val = split_support(support, args.seed + sample_id + support_segments, args.support_val_ratio, args.support_val_segments)
    query_set = MetricSegmentDataset(subjects, arrays, query, stats, augment=False)
    support_train_set = SegmentDataset(subjects, arrays, support_train, stats, augment=args.adapt_augment)
    support_val_set = SegmentDataset(subjects, arrays, support_val, stats, augment=False)

    model = make_model(variant, args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    before = evaluate_full_metrics(model, query_set, stats, args, device, f"{variant} fold {fold_idx} S{sample_id} before")
    best_val, history = adapt_model(args, model, support_train_set, support_val_set, HeartSoundLoss(), device, sample_id)
    after = evaluate_full_metrics(model, query_set, stats, args, device, f"{variant} fold {fold_idx} S{sample_id} after")

    row = {
        "variant": variant,
        "fold": fold_idx,
        "sample_id": sample_id,
        "requested_support_segments": support_segments,
        "support_segments": len(support),
        "query_segments": len(query),
        "support_seconds": len(support) * args.segment_seconds,
        "best_support_val_corr": best_val,
        **wide_row("before", before),
        **wide_row("after", after),
    }
    for key in METRIC_KEYS:
        before_val = row[f"before_{key}"]
        after_val = row[f"after_{key}"]
        row[f"delta_{key}"] = None if before_val is None or after_val is None else after_val - before_val
    return row, {"row": row, "history": history}


def make_noisy_metric_dataset(subjects, arrays, query, stats, noise_type, snr_db, args, sample_id, support_segments):
    return NoisyMetricSegmentDataset(
        subjects,
        arrays,
        query,
        stats,
        noise_type=noise_type,
        snr_db=snr_db,
        seed=args.seed + sample_id + support_segments + int(float(snr_db) * 17),
        babble_sources=args.babble_sources,
    )


def evaluate_subject_snr(args, fold_idx, sample_id, support_segments, subjects, arrays, base_model_path, device):
    checkpoint = torch.load(base_model_path, map_location=device)
    stats = checkpoint["normalization_stats"]
    support, query = choose_support(subjects, arrays, sample_id, support_segments, args.min_query_segments, args.seed, args.support_strategy)
    support_train, support_val = split_support(support, args.seed + sample_id + support_segments, args.support_val_ratio, args.support_val_segments)
    support_train_set = SegmentDataset(subjects, arrays, support_train, stats, augment=args.adapt_augment)
    support_val_set = SegmentDataset(subjects, arrays, support_val, stats, augment=False)

    model = make_model(args.snr_variant, args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    before_by_condition = {}
    for noise_type in args.noise_types:
        for snr_db in args.snr_list:
            dataset = make_noisy_metric_dataset(subjects, arrays, query, stats, noise_type, snr_db, args, sample_id, support_segments)
            before_by_condition[(noise_type, snr_db)] = evaluate_full_metrics(
                model, dataset, stats, args, device, f"SNR fold {fold_idx} S{sample_id} {noise_type} {snr_db:g}dB before"
            )

    best_val, history = adapt_model(args, model, support_train_set, support_val_set, HeartSoundLoss(), device, sample_id)
    rows = []
    for noise_type in args.noise_types:
        for snr_db in args.snr_list:
            dataset = make_noisy_metric_dataset(subjects, arrays, query, stats, noise_type, snr_db, args, sample_id, support_segments)
            after = evaluate_full_metrics(model, dataset, stats, args, device, f"SNR fold {fold_idx} S{sample_id} {noise_type} {snr_db:g}dB after")
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
                **wide_row("before", before),
                **wide_row("after", after),
            }
            for key in METRIC_KEYS:
                before_val = row[f"before_{key}"]
                after_val = row[f"after_{key}"]
                row[f"delta_{key}"] = None if before_val is None or after_val is None else after_val - before_val
            rows.append(row)
    return rows, {"rows": rows, "history": history}


def aggregate_wide(rows, group_keys):
    summaries = []
    groups = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        groups.setdefault(key, []).append(row)
    for key, subset in sorted(groups.items(), key=lambda item: item[0]):
        out = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        out["n_subjects"] = len(subset)
        if "support_segments" not in group_keys:
            actual_support = [row.get("support_segments") for row in subset]
            out["actual_support_segments_mean"] = safe_mean(actual_support)
            out["actual_support_segments_std"] = safe_std(actual_support)
            actual_numeric = [int(value) for value in actual_support if value is not None]
            out["actual_support_segments_min"] = min(actual_numeric) if actual_numeric else None
            out["actual_support_segments_max"] = max(actual_numeric) if actual_numeric else None
        for phase in ["before", "after", "delta"]:
            for metric in METRIC_KEYS:
                vals = [row.get(f"{phase}_{metric}") for row in subset]
                out[f"{phase}_{metric}_mean"] = safe_mean(vals)
                out[f"{phase}_{metric}_std"] = safe_std(vals)
        summaries.append(out)
    return summaries


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


def metric_format_block(summary_row: dict) -> dict:
    return {
        METRIC_LABELS[key]: {
            "before_mean": summary_row.get(f"before_{key}_mean"),
            "after_mean": summary_row.get(f"after_{key}_mean"),
            "delta_mean": summary_row.get(f"delta_{key}_mean"),
        }
        for key in METRIC_KEYS
    }


def compact_requested_rows(summary: list[dict], protocol: str) -> list[dict]:
    rows = []
    for item in summary:
        support = item.get("requested_support_segments", item.get("support_segments"))
        for stage in ["before", "after"]:
            prefix = f"{stage}_"
            rows.append(
                {
                    "variant": item.get("variant"),
                    "protocol": protocol,
                    "stage": stage,
                    "support_segments": support,
                    "actual_support_segments_mean": item.get("actual_support_segments_mean"),
                    "actual_support_segments_min": item.get("actual_support_segments_min"),
                    "actual_support_segments_max": item.get("actual_support_segments_max"),
                    "PCC[%]": None if item.get(prefix + "pcc_mean") is None else item[prefix + "pcc_mean"] * 100.0,
                    "MAE": item.get(prefix + "mae_mean"),
                    "RMSE": item.get(prefix + "rmse_mean"),
                    "SC": item.get(prefix + "spectral_convergence_mean"),
                    "ePCC[%]": None
                    if item.get(prefix + "envelope_pcc_mean") is None
                    else item[prefix + "envelope_pcc_mean"] * 100.0,
                    "Mean HR [bpm]": item.get(prefix + "mean_hr_bias_mean"),
                    "HR-SD [bpm]": item.get(prefix + "std_hr_bias_mean"),
                }
            )
    return rows


def run_clean(args, subjects, arrays, fold_test_ids, device):
    rows = []
    max_folds = len(fold_test_ids) if args.max_folds <= 0 else min(args.max_folds, len(fold_test_ids))
    for variant in args.variants:
        for fold_idx in range(max_folds):
            base_model_path = args.ablation_results_dir / variant / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
            if not base_model_path.exists():
                raise FileNotFoundError(f"Missing base model: {base_model_path}")
            test_ids = sorted(int(item) for item in fold_test_ids[fold_idx])
            if args.max_subjects > 0:
                test_ids = test_ids[: args.max_subjects]
            for support_segments in args.support_list:
                support_dir = args.output_dir / "clean" / variant / f"fold_{fold_idx}" / f"support_{support_segments:03d}"
                support_dir.mkdir(parents=True, exist_ok=True)
                for sample_id in test_ids:
                    row, detail = evaluate_subject_clean(
                        args, variant, fold_idx, sample_id, support_segments, subjects, arrays, base_model_path, device
                    )
                    rows.append(row)
                    (support_dir / f"sample{sample_id:02d}_unified_metrics.json").write_text(
                        json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
    summary = aggregate_wide(rows, ["variant", "requested_support_segments"])
    write_csv(args.output_dir / "clean_unified_metrics.csv", rows)
    write_csv(args.output_dir / "clean_unified_summary.csv", summary)
    write_csv(args.output_dir / "clean_requested_metrics.csv", compact_requested_rows(summary, "continual_model_ablation"))
    return {"rows": rows, "summary": summary, "formatted_summary": [dict(row, metrics=metric_format_block(row)) for row in summary]}


def run_snr(args, subjects, arrays, fold_test_ids, device):
    rows = []
    max_folds = len(fold_test_ids) if args.max_folds <= 0 else min(args.max_folds, len(fold_test_ids))
    for fold_idx in range(max_folds):
        base_model_path = args.ablation_results_dir / args.snr_variant / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
        if not base_model_path.exists():
            raise FileNotFoundError(f"Missing SNR base model: {base_model_path}")
        test_ids = sorted(int(item) for item in fold_test_ids[fold_idx])
        if args.max_subjects > 0:
            test_ids = test_ids[: args.max_subjects]
        for support_segments in args.support_list:
            support_dir = args.output_dir / "snr" / f"fold_{fold_idx}" / f"support_{support_segments:03d}"
            support_dir.mkdir(parents=True, exist_ok=True)
            for sample_id in test_ids:
                sample_rows, detail = evaluate_subject_snr(
                    args, fold_idx, sample_id, support_segments, subjects, arrays, base_model_path, device
                )
                rows.extend(sample_rows)
                (support_dir / f"sample{sample_id:02d}_snr_unified_metrics.json").write_text(
                    json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8"
                )
    summary = aggregate_wide(rows, ["requested_support_segments", "noise_type", "snr_db"])
    write_csv(args.output_dir / "snr_unified_metrics.csv", rows)
    write_csv(args.output_dir / "snr_unified_summary.csv", summary)
    return {"rows": rows, "summary": summary, "formatted_summary": [dict(row, metrics=metric_format_block(row)) for row in summary]}


def load_fold_test_ids(args):
    config_path = args.ablation_results_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["fold_test_ids"]


def parse_args():
    parser = argparse.ArgumentParser(description="Re-evaluate continual few-shot models with unified waveform metrics.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ablation-results-dir", type=Path, default=DEFAULT_ABLATION_RESULTS)
    parser.add_argument("--snr-results-dir", type=Path, default=DEFAULT_SNR_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["clean", "snr", "both"], default="both")
    parser.add_argument("--variants", type=str, default="all")
    parser.add_argument("--snr-variant", choices=list(MODEL_VARIANTS.keys()), default="full_model")
    parser.add_argument("--support-list", type=str, default="48,72,96,120")
    parser.add_argument("--noise-types", type=str, default="white,pink,babble")
    parser.add_argument("--snr-list", type=str, default="20,10,5,0")
    parser.add_argument("--support-strategy", choices=["random", "top_quality", "stratified_quality", "hybrid_quality"], default="stratified_quality")
    parser.add_argument("--support-val-ratio", type=float, default=0.25)
    parser.add_argument("--support-val-segments", type=int, default=8)
    parser.add_argument("--min-query-segments", type=int, default=20)
    parser.add_argument("--segment-seconds", type=float, default=2.5)
    parser.add_argument("--sample-rate", type=int, default=1000)
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=0, help="Per-fold debugging limit. 0 means all subjects.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapt-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data_dir.is_absolute():
        args.data_dir = PROJECT_ROOT / args.data_dir
    if not args.ablation_results_dir.is_absolute():
        args.ablation_results_dir = PROJECT_ROOT / args.ablation_results_dir
    if not args.snr_results_dir.is_absolute():
        args.snr_results_dir = PROJECT_ROOT / args.snr_results_dir
    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir
    args.data_dir = args.data_dir.resolve()
    args.ablation_results_dir = args.ablation_results_dir.resolve()
    args.snr_results_dir = args.snr_results_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.variants = parse_list(args.variants)
    args.support_list = parse_int_list(args.support_list)
    args.noise_types = parse_str_list(args.noise_types)
    args.snr_list = parse_int_list(args.snr_list)
    set_seed(args.seed)

    fold_test_ids = load_fold_test_ids(args)
    subjects = discover_subjects(args.data_dir)
    arrays = [np.load(subject["file"], mmap_mode="r") for subject in subjects]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Output: {args.output_dir}")

    run_config = sanitize_args(args)
    run_config["metric_labels"] = METRIC_LABELS
    run_config["fold_test_ids"] = fold_test_ids
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    final = {"metric_labels": METRIC_LABELS}
    if args.mode in {"clean", "both"}:
        final["clean"] = run_clean(args, subjects, arrays, fold_test_ids, device)
    if args.mode in {"snr", "both"}:
        final["snr"] = run_snr(args, subjects, arrays, fold_test_ids, device)

    (args.output_dir / "unified_metrics_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved unified metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()

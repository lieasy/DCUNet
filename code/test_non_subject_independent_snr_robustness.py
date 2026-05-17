import argparse
import json
from pathlib import Path

import numpy as np
import torch

from refine_continual_subject5fold_adaptation import (
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
    HeartSoundLoss,
    HeartSoundUNet,
    discover_subjects,
    make_loader,
    run_epoch,
    set_seed,
)
from test_continual_snr_robustness import NoisySegmentDataset, parse_int_list, parse_str_list
from train_compare_reconstruction_methods import (
    DETAILED_METRIC_KEYS,
    calculate_heart_sound_metrics,
    denorm_prediction,
    detect_heart_sounds,
    estimate_heart_rate,
    extract_envelope,
    safe_mean,
    safe_number,
    safe_std,
    segment_heart_sounds,
    signal_metrics_np,
    spectral_convergence_batch,
)
from train_continual_subject5fold_adaptation_optimized import sanitize_args, split_base_indices


DEFAULT_BASE_RESULTS = PROJECT_ROOT / "results" / "continual_subject5fold_adaptation_optimized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "non_subject_independent_snr_robustness"
REQUESTED_METRIC_KEYS = [
    "corr",
    "mse",
    "mae",
    "rmse",
    "spectral_convergence",
    "envelope_corr",
    "mean_hr_bias",
    "std_hr_bias",
]


class NoisyDetailedMetricSegmentDataset(NoisySegmentDataset):
    def __getitem__(self, idx):
        x, y, sample_id = super().__getitem__(idx)
        file_idx, segment_idx = self.indices[idx]
        raw_target = np.asarray(self.arrays[file_idx][segment_idx][0], dtype=np.float32)
        local_mean = float(np.median(raw_target))
        local_std = float(np.median(np.abs(raw_target - local_mean)) * 1.4826 + 1e-8)
        return x, y, torch.from_numpy(raw_target[None, :]), sample_id, torch.tensor([local_mean, local_std], dtype=torch.float32)


def make_eval_loader(args, subjects, arrays, indices, stats, noise_type, snr_db, device, seed):
    dataset = NoisySegmentDataset(
        subjects,
        arrays,
        indices,
        stats,
        noise_type=noise_type,
        snr_db=snr_db,
        seed=seed,
        babble_sources=args.babble_sources,
    )
    return make_loader(dataset, args.batch_size, False, device, seed, args.num_workers)


def make_detailed_loader(args, subjects, arrays, indices, stats, noise_type, snr_db, device, seed):
    dataset = NoisyDetailedMetricSegmentDataset(
        subjects,
        arrays,
        indices,
        stats,
        noise_type=noise_type,
        snr_db=snr_db,
        seed=seed,
        babble_sources=args.babble_sources,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )


def evaluate_detailed_noise(model, args, subjects, arrays, indices, stats, noise_type, snr_db, device, seed, desc: str):
    model.eval()
    loader = make_detailed_loader(args, subjects, arrays, indices, stats, noise_type, snr_db, device, seed)
    values = {key: [] for key in DETAILED_METRIC_KEYS if key not in {"valid_hr_count", "total_hr_count", "mean_hr_bias", "std_hr_bias"}}
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
        f"{desc} detailed: PCC={out['corr'] if out['corr'] is not None else float('nan'):.4f}, "
        f"MSE={out['mse'] if out['mse'] is not None else float('nan'):.6f}, "
        f"SC={out['spectral_convergence'] if out['spectral_convergence'] is not None else float('nan'):.4f}, "
        f"ePCC={out['envelope_corr'] if out['envelope_corr'] is not None else float('nan'):.4f}, "
        f"HR valid={out['valid_hr_count']}/{out['total_hr_count']}"
    )
    return out


def evaluate_fold(args, subjects, arrays, fold_idx: int, base_ids: set[int], model_path: Path, device):
    checkpoint = torch.load(model_path, map_location=device)
    stats = checkpoint["normalization_stats"]
    _, _, base_test = split_base_indices(
        subjects,
        base_ids=base_ids,
        seed=args.seed + fold_idx,
        train_ratio=args.base_train_ratio,
        val_ratio=args.base_val_ratio,
    )

    model = HeartSoundUNet(args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = HeartSoundLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp

    rows = []
    for noise_type in args.noise_types:
        for snr_db in args.snr_list:
            loader = make_eval_loader(
                args,
                subjects,
                arrays,
                base_test,
                stats,
                noise_type,
                snr_db,
                device,
                args.seed + fold_idx * 10007 + int(snr_db * 31),
            )
            metrics = run_epoch(
                model,
                loader,
                criterion,
                device,
                None,
                scaler,
                use_amp,
                f"Non-subj-ind fold {fold_idx} {noise_type} {snr_db:g}dB",
            )
            detailed = evaluate_detailed_noise(
                model,
                args,
                subjects,
                arrays,
                base_test,
                stats,
                noise_type,
                snr_db,
                device,
                args.seed + fold_idx * 10007 + int(snr_db * 31),
                f"Non-subj-ind fold {fold_idx} {noise_type} {snr_db:g}dB",
            )
            rows.append(
                {
                    "fold": fold_idx,
                    "noise_type": noise_type,
                    "snr_db": snr_db,
                    "n_test_segments": len(base_test),
                    "detailed_metrics": detailed,
                    **{f"detailed_{key}": safe_number(detailed.get(key)) for key in DETAILED_METRIC_KEYS},
                    **metrics,
                }
            )
    return rows


def aggregate_condition(rows: list[dict]) -> dict:
    out = {"n_folds": len(rows)}
    for key in ["loss", "corr", "rmse", "mae"]:
        vals = np.array([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals))
    for key in REQUESTED_METRIC_KEYS:
        vals = np.array(
            [
                row[f"detailed_{key}"]
                for row in rows
                if row.get(f"detailed_{key}") is not None and not np.isnan(float(row[f"detailed_{key}"]))
            ],
            dtype=np.float64,
        )
        out[f"detailed_{key}_mean"] = float(np.mean(vals)) if vals.size else None
        out[f"detailed_{key}_std"] = float(np.std(vals)) if vals.size else None
    out["n_test_segments_mean"] = float(np.mean([row["n_test_segments"] for row in rows]))
    return out


def build_requested_metric_rows(condition_summaries: list[dict]) -> list[dict]:
    rows = []
    for item in condition_summaries:
        agg = item["aggregate"]
        rows.append(
            {
                "noise_type": item["noise_type"],
                "snr_db": item["snr_db"],
                "PCC_percent_mean": None if agg["detailed_corr_mean"] is None else agg["detailed_corr_mean"] * 100.0,
                "PCC_percent_std": None if agg["detailed_corr_std"] is None else agg["detailed_corr_std"] * 100.0,
                "MSE_mean": agg["detailed_mse_mean"],
                "MSE_std": agg["detailed_mse_std"],
                "MAE_mean": agg["detailed_mae_mean"],
                "MAE_std": agg["detailed_mae_std"],
                "RMSE_mean": agg["detailed_rmse_mean"],
                "RMSE_std": agg["detailed_rmse_std"],
                "SC_mean": agg["detailed_spectral_convergence_mean"],
                "SC_std": agg["detailed_spectral_convergence_std"],
                "ePCC_percent_mean": None if agg["detailed_envelope_corr_mean"] is None else agg["detailed_envelope_corr_mean"] * 100.0,
                "ePCC_percent_std": None if agg["detailed_envelope_corr_std"] is None else agg["detailed_envelope_corr_std"] * 100.0,
                "Mean_HR_bias_bpm_mean": agg["detailed_mean_hr_bias_mean"],
                "Mean_HR_bias_bpm_std": agg["detailed_mean_hr_bias_std"],
                "SD_HR_bias_bpm_mean": agg["detailed_std_hr_bias_mean"],
                "SD_HR_bias_bpm_std": agg["detailed_std_hr_bias_std"],
            }
        )
    return rows


def build_condition_summaries(rows: list[dict]) -> list[dict]:
    summaries = []
    noise_values = sorted({row["noise_type"] for row in rows})
    snr_values = sorted({float(row["snr_db"]) for row in rows})
    for noise_type in noise_values:
        for snr_db in snr_values:
            subset = [
                row
                for row in rows
                if row["noise_type"] == noise_type and float(row["snr_db"]) == snr_db
            ]
            if subset:
                summaries.append(
                    {
                        "noise_type": noise_type,
                        "snr_db": snr_db,
                        "aggregate": aggregate_condition(subset),
                        "per_fold": subset,
                    }
                )
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SNR robustness test for the non-subject-independent base-test split. "
            "Uses the optimized continual base models and evaluates held-out segments "
            "from the base/training subjects without few-shot adaptation."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--base-results-dir", type=Path, default=DEFAULT_BASE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--noise-types", type=str, default="white,pink,babble")
    parser.add_argument("--snr-list", type=str, default="10,20,30,40")
    parser.add_argument("--base-train-ratio", type=float, default=0.82)
    parser.add_argument("--base-val-ratio", type=float, default=0.09)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--babble-sources", type=int, default=6)
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--heart-sound-window", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = (PROJECT_ROOT / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir.resolve()
    args.base_results_dir = (
        PROJECT_ROOT / args.base_results_dir
    ).resolve() if not args.base_results_dir.is_absolute() else args.base_results_dir.resolve()
    args.output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    args.snr_list = parse_int_list(args.snr_list)
    args.noise_types = parse_str_list(args.noise_types)
    invalid_noise = [item for item in args.noise_types if item not in {"white", "pink", "babble"}]
    if invalid_noise:
        raise ValueError(f"Unknown noise types: {invalid_noise}. Valid: white,pink,babble")

    run_config_path = args.base_results_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing base run config: {run_config_path}")
    base_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    fold_test_ids = base_run_config["fold_test_ids"]
    all_ids = sorted(int(item) for fold in fold_test_ids for item in fold)

    subjects = discover_subjects(args.data_dir)
    arrays = [np.load(subject["file"], mmap_mode="r") for subject in subjects]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print("Split mode: non_subject_independent_base_test")
    print(f"Noise types: {args.noise_types}")
    print(f"SNR list: {args.snr_list}")
    print(f"Base results: {args.base_results_dir}")

    run_config = sanitize_args(args)
    run_config["split_mode"] = "non_subject_independent_base_test"
    run_config["protocol_note"] = (
        "Each optimized continual base model is evaluated on held-out base-test segments "
        "from subjects included in that fold's base/training subject pool; no few-shot adaptation is applied."
    )
    run_config["source_base_results"] = str(args.base_results_dir)
    run_config["fold_test_ids"] = fold_test_ids
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    max_folds = len(fold_test_ids) if args.max_folds <= 0 else min(args.max_folds, len(fold_test_ids))
    all_rows = []
    fold_summaries = []
    for fold_idx in range(max_folds):
        model_path = args.base_results_dir / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing base model: {model_path}")
        base_ids = set(all_ids) - {int(item) for item in fold_test_ids[fold_idx]}
        print(f"\n========== Non-subject-independent SNR fold {fold_idx} ==========")
        rows = evaluate_fold(args, subjects, arrays, fold_idx, base_ids, model_path, device)
        fold_summary = {
            "fold": fold_idx,
            "base_ids": sorted(base_ids),
            "held_out_subject_ids": sorted(int(item) for item in fold_test_ids[fold_idx]),
            "condition_summaries": build_condition_summaries(rows),
            "rows": rows,
        }
        fold_dir = args.output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "snr_robustness.json").write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")
        fold_summaries.append({k: v for k, v in fold_summary.items() if k != "rows"})
        all_rows.extend(rows)

    overall = build_condition_summaries(all_rows)
    final = {
        "split_mode": "non_subject_independent_base_test",
        "source_base_results": str(args.base_results_dir),
        "fold_summaries": fold_summaries,
        "overall_condition_summaries": overall,
        "requested_metric_rows": build_requested_metric_rows(overall),
    }
    summary_path = args.output_dir / "non_subject_independent_snr_robustness_summary.json"
    summary_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    requested_path = args.output_dir / "non_subject_independent_snr_requested_metrics.json"
    requested_path.write_text(json.dumps(final["requested_metric_rows"], indent=2), encoding="utf-8")

    print("\n========== Overall non-subject-independent SNR robustness summary ==========")
    for row in final["requested_metric_rows"]:
        print(
            f"{row['noise_type']} | {row['snr_db']:g}dB: "
            f"PCC {row['PCC_percent_mean']:.2f}%, "
            f"MSE {row['MSE_mean']:.6f}, "
            f"MAE {row['MAE_mean']:.6f}, "
            f"RMSE {row['RMSE_mean']:.6f}, "
            f"SC {row['SC_mean']:.4f}, "
            f"ePCC {row['ePCC_percent_mean']:.2f}%, "
            f"Mean HR bias {row['Mean_HR_bias_bpm_mean']:.2f} bpm, "
            f"SD {row['SD_HR_bias_bpm_mean']:.2f} bpm"
        )
    print(f"Saved summary to: {summary_path}")
    print(f"Saved requested metrics to: {requested_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from refine_continual_subject5fold_adaptation import (
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
    HeartSoundLoss,
    HeartSoundUNet,
    SegmentDataset,
    adapt_subject,
    aggregate,
    discover_subjects,
    make_loader,
    read_npy_shape,
    run_epoch,
    set_seed,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "continual_subject5fold_adaptation_optimized"


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center)) * 1.4826 + 1e-8)
    return center, scale


def compute_stats(arrays, train_indices, seed: int, max_segments: int):
    rng = np.random.default_rng(seed)
    take = min(max_segments, len(train_indices))
    selected = rng.choice(len(train_indices), size=take, replace=False)
    ch1, ch2, ch3 = [], [], []
    for idx in selected:
        file_idx, segment_idx = train_indices[int(idx)]
        wave = np.asarray(arrays[file_idx][segment_idx], dtype=np.float32)
        ch1.append(wave[0])
        ch2.append(wave[1])
        ch3.append(wave[2])
    target_mean, target_std = robust_stats(np.concatenate(ch1))
    ch2_mean, ch2_std = robust_stats(np.concatenate(ch2))
    ch3_mean, ch3_std = robust_stats(np.concatenate(ch3))
    return {
        "target_mean": target_mean,
        "target_std": target_std,
        "ch2_mean": ch2_mean,
        "ch2_std": ch2_std,
        "ch3_mean": ch3_mean,
        "ch3_std": ch3_std,
    }


def build_balanced_subject_folds(subjects: list[dict], n_folds: int) -> list[list[int]]:
    folds = [[] for _ in range(n_folds)]
    for subject in sorted(subjects, key=lambda row: int(row["segments"]), reverse=True):
        fold_idx = min(
            range(n_folds),
            key=lambda i: (
                sum(int(s["segments"]) for s in folds[i]),
                len(folds[i]),
                i,
            ),
        )
        folds[fold_idx].append(subject)
    return [[int(s["sample_id"]) for s in sorted(fold, key=lambda row: int(row["sample_id"]))] for fold in folds]


def split_base_indices(subjects, base_ids: set[int], seed: int, train_ratio: float, val_ratio: float):
    rng = np.random.default_rng(seed)
    indices = []
    for subject in subjects:
        if int(subject["sample_id"]) not in base_ids:
            continue
        file_idx = int(subject["file_idx"])
        for segment_idx in range(int(subject["segments"])):
            indices.append((file_idx, segment_idx))
    rng.shuffle(indices)
    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]


def train_base_model(args, subjects, arrays, base_ids: set[int], fold_idx: int, device):
    base_dir = args.output_dir / f"fold_{fold_idx}" / "base_model"
    base_dir.mkdir(parents=True, exist_ok=True)

    base_train, base_val, base_test = split_base_indices(
        subjects,
        base_ids=base_ids,
        seed=args.seed + fold_idx,
        train_ratio=args.base_train_ratio,
        val_ratio=args.base_val_ratio,
    )
    stats = compute_stats(arrays, base_train, args.seed + fold_idx, args.stats_segments)
    (base_dir / "normalization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    train_set = SegmentDataset(subjects, arrays, base_train, stats, augment=args.base_augment)
    val_set = SegmentDataset(subjects, arrays, base_val, stats, augment=False)
    test_set = SegmentDataset(subjects, arrays, base_test, stats, augment=False)
    train_loader = make_loader(train_set, args.batch_size, True, device, args.seed + fold_idx, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, device, args.seed + fold_idx, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, False, device, args.seed + fold_idx, args.num_workers)

    model = HeartSoundUNet(args.base_channels).to(device)
    criterion = HeartSoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        threshold=args.min_delta,
        min_lr=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp

    best_val = -float("inf")
    best_path = base_dir / "best_base_model.pth"
    bad_epochs = 0
    history = []
    for epoch in range(1, args.base_epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            f"Fold {fold_idx} Base {epoch}/{args.base_epochs}",
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"Fold {fold_idx} BaseVal")
        scheduler.step(val_metrics["corr"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        (base_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_metrics["corr"] > best_val + args.min_delta:
            best_val = val_metrics["corr"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normalization_stats": stats,
                    "best_val_corr": best_val,
                    "fold": fold_idx,
                    "args": sanitize_args(args),
                },
                best_path,
            )
        else:
            bad_epochs += 1

        print(
            f"Fold {fold_idx} base epoch {epoch:03d}: "
            f"train PCC {train_metrics['corr']:.4f}, val PCC {val_metrics['corr']:.4f}, "
            f"lr {optimizer.param_groups[0]['lr']:.2e}, patience {bad_epochs}"
        )
        if args.base_early_stop > 0 and epoch >= args.base_min_epochs and bad_epochs >= args.base_early_stop:
            print(f"Fold {fold_idx} base early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device, None, scaler, use_amp, f"Fold {fold_idx} BaseTest")
    (base_dir / "base_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    return best_path, test_metrics


def sanitize_args(args: argparse.Namespace):
    out = vars(args).copy()
    for key, value in list(out.items()):
        if isinstance(value, Path):
            out[key] = str(value)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized subject-level 5-fold continual adaptation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--support-list", type=str, default="48,72,96")
    parser.add_argument("--support-strategy", choices=["random", "top_quality", "stratified_quality", "hybrid_quality"], default="stratified_quality")
    parser.add_argument("--base-train-ratio", type=float, default=0.82)
    parser.add_argument("--base-val-ratio", type=float, default=0.09)
    parser.add_argument("--base-epochs", type=int, default=180)
    parser.add_argument("--base-min-epochs", type=int, default=40)
    parser.add_argument("--base-early-stop", type=int, default=35)
    parser.add_argument("--support-val-ratio", type=float, default=0.25)
    parser.add_argument("--support-val-segments", type=int, default=8)
    parser.add_argument("--min-query-segments", type=int, default=20)
    parser.add_argument("--segment-seconds", type=float, default=2.5)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-epochs", type=int, default=70)
    parser.add_argument("--stage2-lr", type=float, default=8e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--adapt-batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--adapt-weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--adapt-lr-patience", type=int, default=8)
    parser.add_argument("--adapt-min-lr", type=float, default=5e-6)
    parser.add_argument("--min-delta", type=float, default=0.0002)
    parser.add_argument("--stats-segments", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapt-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-base-results", type=Path, default=None)
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
    folds = build_balanced_subject_folds(subjects, args.folds)
    all_ids = sorted(int(subject["sample_id"]) for subject in subjects)
    support_list = [int(x.strip()) for x in args.support_list.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    run_config = sanitize_args(args)
    run_config["fold_test_ids"] = folds
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    all_by_support = {support: [] for support in support_list}
    fold_summaries = []
    max_folds = args.folds if args.max_folds <= 0 else min(args.max_folds, args.folds)
    reuse_base_results = args.reuse_base_results.resolve() if args.reuse_base_results else None

    for fold_idx in range(max_folds):
        test_ids = sorted(folds[fold_idx])
        base_ids = set(all_ids) - set(test_ids)
        fold_dir = args.output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n========== Optimized fold {fold_idx}: test {test_ids} ==========")

        if reuse_base_results is not None:
            base_model_path = reuse_base_results / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
            base_test_metrics = None
            print(f"Reusing base model: {base_model_path}")
        else:
            base_model_path, base_test_metrics = train_base_model(args, subjects, arrays, base_ids, fold_idx, device)

        fold_summary = {"fold": fold_idx, "test_ids": test_ids, "base_ids": sorted(base_ids), "support_summaries": []}
        for support_segments in support_list:
            rows = []
            support_dir = fold_dir / f"support_{support_segments:03d}"
            support_dir.mkdir(parents=True, exist_ok=True)
            for sample_id in test_ids:
                row = adapt_subject(args, subjects, arrays, base_model_path, sample_id, support_segments, device)
                light = {k: v for k, v in row.items() if k != "history"}
                light["fold"] = fold_idx
                (support_dir / f"sample{sample_id:02d}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                rows.append(light)
                all_by_support[support_segments].append(light)

            support_summary = {
                "fold": fold_idx,
                "support_segments": support_segments,
                "base_test_metrics": base_test_metrics,
                "aggregate": aggregate(rows),
                "per_subject": rows,
            }
            (support_dir / "summary.json").write_text(json.dumps(support_summary, indent=2), encoding="utf-8")
            fold_summary["support_summaries"].append(
                {"support_segments": support_segments, "aggregate": support_summary["aggregate"]}
            )
            print(
                f"Fold {fold_idx} support {support_segments}: "
                f"after PCC {support_summary['aggregate']['after_corr_mean']:.4f}, "
                f"delta {support_summary['aggregate']['delta_corr_mean']:.4f}"
            )
        fold_summaries.append(fold_summary)

    overall = []
    for support_segments, rows in all_by_support.items():
        overall.append(
            {
                "support_segments": support_segments,
                "support_seconds": support_segments * args.segment_seconds,
                "aggregate": aggregate(rows),
                "per_subject": rows,
            }
        )

    summary = {"fold_test_ids": folds[:max_folds], "fold_summaries": fold_summaries, "overall_support_summaries": overall}
    (args.output_dir / "optimized_subject5fold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n========== Optimized overall summary ==========")
    for item in overall:
        agg = item["aggregate"]
        print(
            f"Support {item['support_segments']} ({item['support_seconds']:.1f}s): "
            f"n={agg['n_subjects']}, after {agg['after_corr_mean']:.4f} +/- {agg['after_corr_std']:.4f}, "
            f"delta {agg['delta_corr_mean']:.4f}"
        )
    print(f"Saved summary to: {args.output_dir / 'optimized_subject5fold_summary.json'}")


if __name__ == "__main__":
    main()

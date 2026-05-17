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
    SegmentDataset,
    discover_subjects,
    make_loader,
    run_epoch,
    set_seed,
)
from train_continual_subject5fold_adaptation_optimized import compute_stats, sanitize_args


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "segment_mixed_5fold_recovery"


def build_all_segment_indices(subjects: list[dict]) -> list[tuple[int, int]]:
    indices = []
    for subject in subjects:
        file_idx = int(subject["file_idx"])
        for segment_idx in range(int(subject["segments"])):
            indices.append((file_idx, segment_idx))
    return indices


def build_segment_folds(indices: list[tuple[int, int]], n_folds: int, seed: int) -> list[list[tuple[int, int]]]:
    shuffled = list(indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    return [shuffled[i::n_folds] for i in range(n_folds)]


def split_train_val(indices: list[tuple[int, int]], val_ratio: float, seed: int):
    items = list(indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    n_val = int(round(len(items) * val_ratio))
    n_val = min(max(n_val, 1), len(items) - 1)
    return items[n_val:], items[:n_val]


def summarize_subject_overlap(subjects: list[dict], train_indices, val_indices, test_indices) -> dict:
    def to_ids(indices):
        return sorted({int(subjects[file_idx]["sample_id"]) for file_idx, _ in indices})

    train_ids = to_ids(train_indices)
    val_ids = to_ids(val_indices)
    test_ids = to_ids(test_indices)
    return {
        "train_subject_ids": train_ids,
        "val_subject_ids": val_ids,
        "test_subject_ids": test_ids,
        "n_train_subjects": len(train_ids),
        "n_val_subjects": len(val_ids),
        "n_test_subjects": len(test_ids),
        "shared_train_test_subjects": sorted(set(train_ids) & set(test_ids)),
        "shared_val_test_subjects": sorted(set(val_ids) & set(test_ids)),
    }


def train_one_fold(args, subjects, arrays, fold_idx: int, train_indices, val_indices, test_indices, device):
    fold_dir = args.output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

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

    train_set = SegmentDataset(subjects, arrays, train_indices, stats, augment=args.augment)
    val_set = SegmentDataset(subjects, arrays, val_indices, stats, augment=False)
    test_set = SegmentDataset(subjects, arrays, test_indices, stats, augment=False)
    train_loader = make_loader(train_set, args.batch_size, True, device, args.seed + fold_idx, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, device, args.seed + fold_idx, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, False, device, args.seed + fold_idx, args.num_workers)

    model = HeartSoundUNet(args.base_channels).to(device)
    criterion = HeartSoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    bad_epochs = 0
    history = []
    best_path = fold_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            f"Mixed fold {fold_idx} train {epoch}/{args.epochs}",
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"Mixed fold {fold_idx} val")
        scheduler.step(val_metrics["corr"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        (fold_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

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
            f"Mixed fold {fold_idx} epoch {epoch:03d}: "
            f"train PCC {train_metrics['corr']:.4f}, val PCC {val_metrics['corr']:.4f}, "
            f"lr {optimizer.param_groups[0]['lr']:.2e}, patience {bad_epochs}"
        )
        if args.early_stop > 0 and epoch >= args.min_epochs and bad_epochs >= args.early_stop:
            print(f"Mixed fold {fold_idx} early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device, None, scaler, use_amp, f"Mixed fold {fold_idx} test")
    fold_summary = {
        "fold": fold_idx,
        "best_val_corr": float(checkpoint["best_val_corr"]),
        "test_metrics": test_metrics,
        "split_info": split_info,
        "best_model": str(best_path),
    }
    (fold_dir / "summary.json").write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")
    return fold_summary


def aggregate_fold_metrics(fold_summaries: list[dict]) -> dict:
    out = {"n_folds": len(fold_summaries)}
    for key in ["loss", "corr", "rmse", "mae"]:
        values = np.array([row["test_metrics"][key] for row in fold_summaries], dtype=np.float64)
        out[f"test_{key}_mean"] = float(np.mean(values))
        out[f"test_{key}_std"] = float(np.std(values))
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment-mixed 5-fold recovery. This is not subject-independent: segments from the same subject may appear in train and test."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
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
    max_folds = args.folds if args.max_folds <= 0 else min(args.max_folds, args.folds)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded {len(subjects)} subjects and {len(all_indices)} total segments.")
    print("Split mode: segment-mixed 5-fold (not subject-independent).")

    run_config = sanitize_args(args)
    run_config["n_subjects"] = len(subjects)
    run_config["n_segments"] = len(all_indices)
    run_config["fold_segment_counts"] = [len(fold) for fold in folds[:max_folds]]
    run_config["split_mode"] = "segment_mixed_non_subject_independent"
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    fold_summaries = []
    for fold_idx in range(max_folds):
        test_indices = folds[fold_idx]
        train_val_indices = [item for i, fold in enumerate(folds) if i != fold_idx for item in fold]
        train_indices, val_indices = split_train_val(train_val_indices, args.val_ratio, args.seed + fold_idx)
        print(
            f"\n========== Segment-mixed 5-fold {fold_idx} =========="
            f"\nTrain segments: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}"
        )
        fold_summaries.append(
            train_one_fold(args, subjects, arrays, fold_idx, train_indices, val_indices, test_indices, device)
        )

    summary = {
        "split_mode": "segment_mixed_non_subject_independent",
        "fold_summaries": fold_summaries,
        "aggregate": aggregate_fold_metrics(fold_summaries),
    }
    (args.output_dir / "segment_mixed_5fold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    agg = summary["aggregate"]
    print("\n========== Segment-mixed 5-fold summary ==========")
    print(
        f"Test PCC {agg['test_corr_mean']:.4f} +/- {agg['test_corr_std']:.4f}, "
        f"RMSE {agg['test_rmse_mean']:.4f} +/- {agg['test_rmse_std']:.4f}, "
        f"MAE {agg['test_mae_mean']:.4f} +/- {agg['test_mae_std']:.4f}"
    )
    print(f"Saved summary to: {args.output_dir / 'segment_mixed_5fold_summary.json'}")


if __name__ == "__main__":
    main()

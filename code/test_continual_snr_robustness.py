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
    aggregate,
    choose_support,
    discover_subjects,
    make_loader,
    run_epoch,
    set_seed,
    set_trainable,
    split_support,
)
from train_continual_subject5fold_adaptation_optimized import sanitize_args


DEFAULT_BASE_RESULTS = PROJECT_ROOT / "results" / "continual_subject5fold_adaptation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "continual_snr_robustness"


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def make_pink_noise(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    channels, length = shape
    noise = np.empty(shape, dtype=np.float32)
    freqs = np.fft.rfftfreq(length)
    scale = np.ones_like(freqs, dtype=np.float64)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale[0] = 0.0
    for ch in range(channels):
        white = rng.standard_normal(length)
        spectrum = np.fft.rfft(white) * scale
        pink = np.fft.irfft(spectrum, n=length)
        noise[ch] = pink.astype(np.float32)
    return noise


class NoisySegmentDataset(SegmentDataset):
    def __init__(
        self,
        subjects,
        arrays,
        indices,
        stats,
        noise_type: str,
        snr_db: float,
        seed: int,
        babble_sources: int = 6,
    ):
        super().__init__(subjects, arrays, indices, stats, augment=False)
        self.noise_type = noise_type
        self.snr_db = float(snr_db)
        self.seed = int(seed)
        self.babble_sources = int(babble_sources)
        self._all_indices = list(indices)

    def __getitem__(self, idx):
        x, y, sample_id = super().__getitem__(idx)
        rng = np.random.default_rng(self.seed + idx * 1009 + int(self.snr_db * 31))
        x_np = x.numpy().astype(np.float32, copy=True)
        noise = self.make_noise(idx, x_np.shape, rng)
        x_np = self.add_noise_at_snr(x_np, noise, self.snr_db)
        return torch.from_numpy(x_np.astype(np.float32)), y, sample_id

    def make_noise(self, idx: int, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        if self.noise_type == "white":
            return rng.standard_normal(shape).astype(np.float32)
        if self.noise_type == "pink":
            return make_pink_noise(rng, shape)
        if self.noise_type == "babble":
            return self.make_babble_noise(idx, shape, rng)
        raise ValueError(f"Unknown noise type: {self.noise_type}")

    def make_babble_noise(self, idx: int, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        if not self._all_indices:
            return rng.standard_normal(shape).astype(np.float32)
        chosen = rng.choice(len(self._all_indices), size=max(1, self.babble_sources), replace=True)
        pieces = []
        for chosen_idx in chosen:
            file_idx, segment_idx = self._all_indices[int(chosen_idx)]
            wave = np.asarray(self.arrays[file_idx][segment_idx], dtype=np.float32)
            ch2 = self.norm(wave[1], self.stats["ch2_mean"], self.stats["ch2_std"])
            ch3 = self.norm(wave[2], self.stats["ch3_mean"], self.stats["ch3_std"])
            pieces.append(np.stack([ch2, ch3]).astype(np.float32))
        babble = np.mean(np.stack(pieces, axis=0), axis=0)
        babble = np.roll(babble, int(rng.integers(0, shape[-1])), axis=-1)
        return babble.astype(np.float32)

    @staticmethod
    def add_noise_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        noise = noise.astype(np.float32, copy=False)
        noise = noise - noise.mean(axis=-1, keepdims=True)
        signal_power = float(np.mean(signal**2))
        noise_power = float(np.mean(noise**2)) + 1e-12
        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(target_noise_power / noise_power)
        return signal + noise * scale


def make_eval_loader(args, subjects, arrays, query, stats, noise_type, snr_db, device, seed):
    dataset = NoisySegmentDataset(
        subjects,
        arrays,
        query,
        stats,
        noise_type=noise_type,
        snr_db=snr_db,
        seed=seed,
        babble_sources=args.babble_sources,
    )
    return make_loader(dataset, args.batch_size, False, device, seed, args.num_workers)


def evaluate_conditions(model, args, subjects, arrays, query, stats, device, sample_id, support_segments):
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
                query,
                stats,
                noise_type,
                snr_db,
                device,
                args.seed + sample_id + support_segments + int(snr_db * 17),
            )
            metrics = run_epoch(
                model,
                loader,
                criterion,
                device,
                None,
                scaler,
                use_amp,
                f"S{sample_id} {noise_type} {snr_db:g}dB",
            )
            rows.append({"noise_type": noise_type, "snr_db": snr_db, **metrics})
    return rows


def adapt_subject_then_eval_noise(args, subjects, arrays, base_model_path, sample_id, support_segments, device):
    checkpoint = torch.load(base_model_path, map_location=device)
    stats = checkpoint["normalization_stats"]
    support, query = choose_support(
        subjects, arrays, sample_id, support_segments, args.min_query_segments, args.seed, args.support_strategy
    )
    support_train, support_val = split_support(
        support, args.seed + sample_id + support_segments, args.support_val_ratio, args.support_val_segments
    )

    support_train_set = SegmentDataset(subjects, arrays, support_train, stats, augment=args.adapt_augment)
    support_val_set = SegmentDataset(subjects, arrays, support_val, stats, augment=False)
    criterion = HeartSoundLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp

    model = HeartSoundUNet(args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    before_rows = evaluate_conditions(model, args, subjects, arrays, query, stats, device, sample_id, support_segments)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -float("inf")
    history = []
    stages = [
        ("decoder_head", args.stage1_epochs, args.stage1_lr),
        ("full", args.stage2_epochs, args.stage2_lr),
    ]
    for stage_name, epochs, lr in stages:
        if epochs <= 0:
            continue
        set_trainable(model, stage_name)
        trainable = [p for p in model.parameters() if p.requires_grad]
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
        train_loader = make_loader(support_train_set, args.adapt_batch_size, True, device, args.seed + sample_id, args.num_workers)
        val_loader = make_loader(support_val_set, args.adapt_batch_size, False, device, args.seed + sample_id, args.num_workers)
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
            val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"S{sample_id} support-val")
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
    after_rows = evaluate_conditions(model, args, subjects, arrays, query, stats, device, sample_id, support_segments)

    merged = []
    before_by_condition = {(row["noise_type"], row["snr_db"]): row for row in before_rows}
    for after in after_rows:
        before = before_by_condition[(after["noise_type"], after["snr_db"])]
        merged.append(
            {
                "sample_id": sample_id,
                "support_segments": len(support),
                "query_segments": len(query),
                "support_seconds": len(support) * args.segment_seconds,
                "noise_type": after["noise_type"],
                "snr_db": after["snr_db"],
                "before_corr": before["corr"],
                "after_corr": after["corr"],
                "delta_corr": after["corr"] - before["corr"],
                "before_rmse": before["rmse"],
                "after_rmse": after["rmse"],
                "before_mae": before["mae"],
                "after_mae": after["mae"],
                "before_loss": before["loss"],
                "after_loss": after["loss"],
                "best_support_val_corr": best_val,
            }
        )
    return {"rows": merged, "history": history}


def aggregate_condition(rows: list[dict]) -> dict:
    out = {"n_subjects": len(rows)}
    keys = [
        "before_corr",
        "after_corr",
        "delta_corr",
        "before_rmse",
        "after_rmse",
        "before_mae",
        "after_mae",
        "before_loss",
        "after_loss",
    ]
    for key in keys:
        vals = np.array([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.nanmean(vals))
        out[f"{key}_std"] = float(np.nanstd(vals))
    return out


def build_condition_summaries(rows: list[dict]) -> list[dict]:
    summaries = []
    support_values = sorted({int(row["support_segments"]) for row in rows})
    noise_values = sorted({row["noise_type"] for row in rows})
    snr_values = sorted({float(row["snr_db"]) for row in rows})
    for support_segments in support_values:
        for noise_type in noise_values:
            for snr_db in snr_values:
                subset = [
                    row
                    for row in rows
                    if int(row["support_segments"]) == support_segments
                    and row["noise_type"] == noise_type
                    and float(row["snr_db"]) == snr_db
                ]
                if subset:
                    summaries.append(
                        {
                            "support_segments": support_segments,
                            "noise_type": noise_type,
                            "snr_db": snr_db,
                            "aggregate": aggregate_condition(subset),
                            "per_subject": subset,
                        }
                    )
    return summaries


def merge_condition_rows(existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    merged = {}
    for row in existing_rows:
        merged[(row["noise_type"], float(row["snr_db"]))] = row
    for row in new_rows:
        merged[(row["noise_type"], float(row["snr_db"]))] = row
    return [
        merged[key]
        for key in sorted(merged, key=lambda item: (item[0], float(item[1])))
    ]


def load_sample_rows(sample_path: Path, fold_idx: int | None = None) -> list[dict]:
    if not sample_path.exists():
        return []
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if fold_idx is not None:
        rows = [{**row, "fold": fold_idx} for row in rows]
    return rows


def collect_support_rows(support_dir: Path, test_ids: list[int], fold_idx: int) -> list[dict]:
    rows = []
    for sample_id in test_ids:
        sample_path = support_dir / f"sample{sample_id:02d}_snr_robustness.json"
        rows.extend(load_sample_rows(sample_path, fold_idx))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="SNR robustness test for continual few-shot adaptation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--base-results-dir", type=Path, default=DEFAULT_BASE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--support-list", type=str, default="48,72,96")
    parser.add_argument("--noise-types", type=str, default="white,pink,babble")
    parser.add_argument("--snr-list", type=str, default="0,5,10,20,30,40")
    parser.add_argument("--support-strategy", choices=["random", "top_quality", "stratified_quality", "hybrid_quality"], default="stratified_quality")
    parser.add_argument("--support-val-ratio", type=float, default=0.25)
    parser.add_argument("--support-val-segments", type=int, default=8)
    parser.add_argument("--min-query-segments", type=int, default=20)
    parser.add_argument("--segment-seconds", type=float, default=2.5)
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
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge newly evaluated SNR rows with existing sample JSON files before rebuilding summaries.",
    )
    parser.add_argument(
        "--only-120-30-40",
        action="store_true",
        help="Shortcut for supplementing only support-list=120 and snr-list=30,40 with merge-existing enabled.",
    )
    parser.add_argument(
        "--only-120-10-20",
        action="store_true",
        help="Shortcut for supplementing only support-list=120 and snr-list=10,20 with merge-existing enabled.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapt-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data_dir.is_absolute():
        args.data_dir = PROJECT_ROOT / args.data_dir
    if not args.base_results_dir.is_absolute():
        args.base_results_dir = PROJECT_ROOT / args.base_results_dir
    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir
    args.data_dir = args.data_dir.resolve()
    args.base_results_dir = args.base_results_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.only_120_30_40:
        args.support_list = "120"
        args.snr_list = "30,40"
        args.merge_existing = True
    if args.only_120_10_20:
        args.support_list = "120"
        args.snr_list = "10,20"
        args.merge_existing = True

    args.support_list = parse_int_list(args.support_list)
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

    subjects = discover_subjects(args.data_dir)
    arrays = [np.load(subject["file"], mmap_mode="r") for subject in subjects]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Noise types: {args.noise_types}")
    print(f"SNR list: {args.snr_list}")
    print(f"Base results: {args.base_results_dir}")

    run_config = sanitize_args(args)
    run_config["support_list"] = args.support_list
    run_config["snr_list"] = args.snr_list
    run_config["noise_types"] = args.noise_types
    run_config["source_base_results"] = str(args.base_results_dir)
    run_config["fold_test_ids"] = fold_test_ids
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    fold_summaries = []
    max_folds = len(fold_test_ids) if args.max_folds <= 0 else min(args.max_folds, len(fold_test_ids))
    for fold_idx in range(max_folds):
        base_model_path = args.base_results_dir / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
        if not base_model_path.exists():
            raise FileNotFoundError(f"Missing base model: {base_model_path}")
        test_ids = sorted(int(item) for item in fold_test_ids[fold_idx])
        print(f"\n========== SNR robustness fold {fold_idx}: test {test_ids} ==========")
        fold_rows = []
        for support_segments in args.support_list:
            support_dir = args.output_dir / f"fold_{fold_idx}" / f"support_{support_segments:03d}"
            support_dir.mkdir(parents=True, exist_ok=True)
            for sample_id in test_ids:
                result = adapt_subject_then_eval_noise(
                    args, subjects, arrays, base_model_path, sample_id, support_segments, device
                )
                sample_path = support_dir / f"sample{sample_id:02d}_snr_robustness.json"
                if args.merge_existing:
                    result["rows"] = merge_condition_rows(load_sample_rows(sample_path), result["rows"])
                    result["merged_existing"] = True
                sample_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

            support_rows = collect_support_rows(support_dir, test_ids, fold_idx)
            fold_rows.extend(support_rows)
            support_summary = {
                "fold": fold_idx,
                "support_segments": support_segments,
                "condition_summaries": build_condition_summaries(support_rows),
            }
            (support_dir / "summary.json").write_text(json.dumps(support_summary, indent=2), encoding="utf-8")

        fold_summary = {"fold": fold_idx, "test_ids": test_ids, "condition_summaries": build_condition_summaries(fold_rows)}
        (args.output_dir / f"fold_{fold_idx}" / "summary.json").write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")
        fold_summaries.append(fold_summary)

    all_rows = []
    for fold_summary in fold_summaries:
        fold_idx = int(fold_summary["fold"])
        test_ids = sorted(int(item) for item in fold_test_ids[fold_idx])
        for support_segments in args.support_list:
            support_dir = args.output_dir / f"fold_{fold_idx}" / f"support_{support_segments:03d}"
            all_rows.extend(collect_support_rows(support_dir, test_ids, fold_idx))

    overall = build_condition_summaries(all_rows)
    final = {
        "source_base_results": str(args.base_results_dir),
        "fold_summaries": fold_summaries,
        "overall_condition_summaries": overall,
    }
    (args.output_dir / "continual_snr_robustness_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    print("\n========== Overall SNR robustness summary ==========")
    for item in overall:
        agg = item["aggregate"]
        print(
            f"support {item['support_segments']} | {item['noise_type']} | {item['snr_db']:g}dB: "
            f"before {agg['before_corr_mean']:.4f}, "
            f"after {agg['after_corr_mean']:.4f} +/- {agg['after_corr_std']:.4f}, "
            f"delta {agg['delta_corr_mean']:.4f}, "
            f"RMSE {agg['after_rmse_mean']:.4f}"
        )
    print(f"Saved summary to: {args.output_dir / 'continual_snr_robustness_summary.json'}")


if __name__ == "__main__":
    main()

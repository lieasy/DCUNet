import argparse
import csv
import json
import sys
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_continual_subject5fold_adaptation import (  # noqa: E402
    DEFAULT_DATA_DIR,
    HeartSoundUNet,
    SegmentDataset,
    corr_1d,
    discover_subjects,
    make_loader,
    set_seed,
)
from train_segment_mixed_5fold_recovery import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_SEGMENT_RESULTS_DIR,
    build_all_segment_indices,
    build_segment_folds,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "demo_gallery"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def denorm_prediction(pred_norm: np.ndarray, raw_target: np.ndarray, stats: dict) -> np.ndarray:
    local_mean = float(np.median(raw_target))
    local_std = float(np.median(np.abs(raw_target - local_mean)) * 1.4826 + 1e-8)
    global_mean = float(stats["target_mean"])
    global_std = float(stats["target_std"])
    scale = 0.7 / local_std + 0.3 / global_std
    offset = 0.7 * local_mean / local_std + 0.3 * global_mean / global_std
    return ((pred_norm + offset) / scale).astype(np.float32)


def signal_metrics(pred: np.ndarray, target: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    err = pred - target
    left_pcc = corr_1d(np.asarray(left), target)
    right_pcc = corr_1d(np.asarray(right), target)
    best_input_pcc = max(abs(left_pcc), abs(right_pcc))
    return {
        "recovered_pcc": corr_1d(pred, target),
        "left_pcc": left_pcc,
        "right_pcc": right_pcc,
        "best_input_abs_pcc": best_input_pcc,
        "pcc_gain_over_best_input": corr_1d(pred, target) - best_input_pcc,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "target_std": float(np.std(target)),
    }


def load_checkpoint_model(fold_dir: Path, device: torch.device, base_channels: int) -> tuple[HeartSoundUNet, dict]:
    ckpt_path = fold_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    channels = int(ckpt_args.get("base_channels", base_channels))
    model = HeartSoundUNet(channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["normalization_stats"]


def predict_fold_candidates(args, subjects, arrays, fold_idx: int, test_indices, device: torch.device) -> list[dict]:
    fold_dir = args.results_dir / f"fold_{fold_idx}"
    model, stats = load_checkpoint_model(fold_dir, device, args.base_channels)
    dataset = SegmentDataset(subjects, arrays, test_indices, stats, augment=False)
    loader = make_loader(dataset, args.batch_size, False, device, args.seed + fold_idx, args.num_workers)
    rows = []
    cursor = 0

    with torch.no_grad():
        for x, _y, _sid in tqdm(loader, desc=f"Scoring fold {fold_idx}", leave=False):
            x = x.to(device, non_blocking=True)
            pred_norm = model(x).detach().cpu().numpy()[:, 0, :]
            batch_indices = test_indices[cursor : cursor + pred_norm.shape[0]]
            cursor += pred_norm.shape[0]
            for item_idx, (file_idx, segment_idx) in enumerate(batch_indices):
                wave_arr = np.asarray(arrays[file_idx][segment_idx], dtype=np.float32)
                target, left, right = wave_arr[0], wave_arr[1], wave_arr[2]
                pred = denorm_prediction(pred_norm[item_idx], target, stats)
                metrics = signal_metrics(pred, target, left, right)
                rows.append(
                    {
                        "fold": fold_idx,
                        "file_idx": int(file_idx),
                        "segment_idx": int(segment_idx),
                        "sample_id": int(subjects[file_idx]["sample_id"]),
                        **metrics,
                    }
                )
    return rows


def select_examples(rows: list[dict], top_k: int, min_pcc: float, min_gain: float, per_subject: int) -> list[dict]:
    filtered = [
        row
        for row in rows
        if row["recovered_pcc"] >= min_pcc
        and row["pcc_gain_over_best_input"] >= min_gain
        and row["target_std"] > 1e-8
    ]
    if not filtered:
        filtered = [row for row in rows if row["recovered_pcc"] >= min_pcc and row["target_std"] > 1e-8]
    filtered.sort(key=lambda r: (r["recovered_pcc"], r["pcc_gain_over_best_input"]), reverse=True)

    selected = []
    counts = {}
    for row in filtered:
        sid = int(row["sample_id"])
        if counts.get(sid, 0) >= per_subject:
            continue
        selected.append(row)
        counts[sid] = counts.get(sid, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def normalize_for_wav(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak < 1e-8:
        return np.zeros_like(x, dtype=np.int16)
    return np.clip(x / peak * 0.95, -1.0, 1.0).astype(np.float32)


def write_wav(path: Path, x: np.ndarray, sample_rate: int):
    y = (normalize_for_wav(x) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(y.tobytes())


def plot_example(path: Path, wave_arr: np.ndarray, pred: np.ndarray, sample_rate: int, title: str, metrics: dict):
    labels = ["Left ear raw", "Right ear raw", "Reference heart sound", "Recovered heart sound"]
    signals = [wave_arr[1], wave_arr[2], wave_arr[0], pred]
    colors = ["#2563eb", "#16a34a", "#111827", "#dc2626"]
    t = np.arange(wave_arr.shape[-1], dtype=np.float32) / float(sample_rate)

    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"{title} | recovered PCC={metrics['recovered_pcc']:.3f}, "
        f"gain={metrics['pcc_gain_over_best_input']:.3f}",
        fontsize=13,
    )
    for ax, label, sig, color in zip(axes, labels, signals, colors):
        sig = np.asarray(sig, dtype=np.float32)
        ax.plot(t, sig, color=color, linewidth=0.9)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.22, linewidth=0.6)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_assets(args, subjects, arrays, selected: list[dict]) -> list[dict]:
    image_dir = args.output_dir / "images"
    audio_dir = args.output_dir / "audio"
    image_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    enriched = []
    model_cache = {}
    for rank, row in enumerate(selected, start=1):
        fold_idx = int(row["fold"])
        if fold_idx not in model_cache:
            model_cache[fold_idx] = load_checkpoint_model(args.results_dir / f"fold_{fold_idx}", args.device, args.base_channels)
        model, stats = model_cache[fold_idx]
        wave_arr = np.asarray(arrays[int(row["file_idx"])][int(row["segment_idx"])], dtype=np.float32)
        dataset = SegmentDataset(subjects, arrays, [(int(row["file_idx"]), int(row["segment_idx"]))], stats, augment=False)
        x, _y, _sid = dataset[0]
        with torch.no_grad():
            pred_norm = model(x[None, ...].to(args.device)).detach().cpu().numpy()[0, 0]
        pred = denorm_prediction(pred_norm, wave_arr[0], stats)

        stem = f"rank{rank:02d}_sample{int(row['sample_id']):02d}_fold{fold_idx}_seg{int(row['segment_idx']):04d}"
        image_path = image_dir / f"{stem}.png"
        title = f"Sample {int(row['sample_id']):02d}, fold {fold_idx}, segment {int(row['segment_idx'])}"
        plot_example(image_path, wave_arr, pred, args.sample_rate, title, row)

        audio_paths = {
            "left_wav": audio_dir / f"{stem}_left.wav",
            "right_wav": audio_dir / f"{stem}_right.wav",
            "reference_wav": audio_dir / f"{stem}_reference.wav",
            "recovered_wav": audio_dir / f"{stem}_recovered.wav",
        }
        write_wav(audio_paths["left_wav"], wave_arr[1], args.sample_rate)
        write_wav(audio_paths["right_wav"], wave_arr[2], args.sample_rate)
        write_wav(audio_paths["reference_wav"], wave_arr[0], args.sample_rate)
        write_wav(audio_paths["recovered_wav"], pred, args.sample_rate)

        enriched_row = dict(row)
        enriched_row["image"] = str(image_path.relative_to(args.output_dir)).replace("\\", "/")
        for key, value in audio_paths.items():
            enriched_row[key] = str(value.relative_to(args.output_dir)).replace("\\", "/")
        enriched.append(enriched_row)
    return enriched


def write_manifest(output_dir: Path, rows: list[dict], selected: list[dict], config: dict):
    fieldnames = [
        "rank",
        "sample_id",
        "fold",
        "segment_idx",
        "recovered_pcc",
        "left_pcc",
        "right_pcc",
        "best_input_abs_pcc",
        "pcc_gain_over_best_input",
        "rmse",
        "mae",
        "target_std",
        "image",
        "left_wav",
        "right_wav",
        "reference_wav",
        "recovered_wav",
    ]
    with (output_dir / "selected_examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(selected, start=1):
            writer.writerow({key: row.get(key, rank if key == "rank" else "") for key in fieldnames})

    metric_keys = [
        "fold",
        "sample_id",
        "segment_idx",
        "recovered_pcc",
        "left_pcc",
        "right_pcc",
        "best_input_abs_pcc",
        "pcc_gain_over_best_input",
        "rmse",
        "mae",
        "target_std",
    ]
    with (output_dir / "all_candidate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in metric_keys})

    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_markdown(output_dir: Path, selected: list[dict]):
    lines = [
        "# Demo Gallery",
        "",
        "These examples were selected automatically from held-out segment-mixed test folds by high recovered PCC.",
        "",
    ]
    for rank, row in enumerate(selected, start=1):
        lines.extend(
            [
                f"## Example {rank}: sample {int(row['sample_id']):02d}, fold {int(row['fold'])}, segment {int(row['segment_idx'])}",
                "",
                f"- Recovered PCC: {row['recovered_pcc']:.4f}",
                f"- Left/right input PCC: {row['left_pcc']:.4f} / {row['right_pcc']:.4f}",
                f"- Gain over best input PCC: {row['pcc_gain_over_best_input']:.4f}",
                f"- RMSE / MAE: {row['rmse']:.4f} / {row['mae']:.4f}",
                "",
                f"![Waveform comparison]({row['image']})",
                "",
                f"- [Left ear audio]({row['left_wav']})",
                f"- [Right ear audio]({row['right_wav']})",
                f"- [Reference heart sound]({row['reference_wav']})",
                f"- [Recovered heart sound]({row['recovered_wav']})",
                "",
            ]
        )
    (output_dir / "demo_gallery.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build waveform/audio demo gallery by selecting high-PCC recovered heart-sound segments."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_SEGMENT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--min-pcc", type=float, default=0.90)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--per-subject", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.results_dir = resolve_path(args.results_dir)
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    subjects = discover_subjects(args.data_dir)
    arrays = [np.load(subject["file"], mmap_mode="r") for subject in subjects]
    all_indices = build_all_segment_indices(subjects)
    folds = build_segment_folds(all_indices, args.folds, args.seed)
    max_folds = args.folds if args.max_folds <= 0 else min(args.max_folds, args.folds)

    print(f"Using device: {args.device}")
    print(f"Loaded {len(subjects)} subjects and {len(all_indices)} total segments.")
    print(f"Scoring {max_folds} fold(s) from: {args.results_dir}")

    rows = []
    for fold_idx in range(max_folds):
        rows.extend(predict_fold_candidates(args, subjects, arrays, fold_idx, folds[fold_idx], args.device))

    selected = select_examples(rows, args.top_k, args.min_pcc, args.min_gain, args.per_subject)
    if not selected:
        raise RuntimeError("No examples selected. Try lowering --min-pcc or --min-gain.")

    selected = render_assets(args, subjects, arrays, selected)
    config = {
        "data_dir": str(args.data_dir),
        "results_dir": str(args.results_dir),
        "output_dir": str(args.output_dir),
        "folds": args.folds,
        "max_folds": max_folds,
        "top_k": args.top_k,
        "min_pcc": args.min_pcc,
        "min_gain": args.min_gain,
        "per_subject": args.per_subject,
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "device": str(args.device),
        "n_candidates": len(rows),
        "n_selected": len(selected),
    }
    write_manifest(args.output_dir, rows, selected, config)
    write_markdown(args.output_dir, selected)

    print(f"Selected {len(selected)} examples.")
    print(f"Wrote gallery to: {args.output_dir}")
    print(f"Open: {args.output_dir / 'demo_gallery.md'}")


if __name__ == "__main__":
    main()

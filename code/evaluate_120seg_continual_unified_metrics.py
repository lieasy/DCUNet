import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_missing_requested_metrics as metric_eval  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "continual_120seg_unified_metrics"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Supplement subject-independent continual few-shot results with full unified metrics "
            "for 120 support segments, plus 30/40 dB SNR robustness metrics."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=metric_eval.DEFAULT_DATA_DIR)
    parser.add_argument("--optimized-results-dir", type=Path, default=metric_eval.DEFAULT_OPTIMIZED_RESULTS)
    parser.add_argument(
        "--snr-results-dir",
        type=Path,
        default=None,
        help="Directory used only for fold_test_ids. Defaults to --optimized-results-dir.",
    )
    parser.add_argument("--snr-base-results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["clean", "snr", "all"], default="all")
    parser.add_argument("--support-list", type=str, default="120")
    parser.add_argument("--noise-types", type=str, default="white,pink,babble")
    parser.add_argument("--snr-list", type=str, default="30,40")
    parser.add_argument(
        "--support-strategy",
        choices=["random", "top_quality", "stratified_quality", "hybrid_quality"],
        default="stratified_quality",
    )
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


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main():
    args = parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.optimized_results_dir = resolve_path(args.optimized_results_dir)
    args.snr_results_dir = resolve_path(args.snr_results_dir or args.optimized_results_dir)
    args.snr_base_results_dir = resolve_path(args.snr_base_results_dir) if args.snr_base_results_dir else None
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.support_list = metric_eval.parse_int_list(args.support_list)
    args.noise_types = metric_eval.parse_str_list(args.noise_types)
    args.snr_list = metric_eval.parse_int_list(args.snr_list)
    metric_eval.set_seed(args.seed)

    subjects = metric_eval.discover_subjects(args.data_dir)
    arrays = [np.load(subject["file"], mmap_mode="r") for subject in subjects]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded {len(subjects)} subjects from {args.data_dir}")
    print(f"Support list: {args.support_list}")
    print(f"SNR list: {args.snr_list}")
    print(f"Output: {args.output_dir}")

    run_config = metric_eval.sanitize_args(args)
    run_config["note"] = (
        "120-segment subject-independent continual few-shot unified metrics and 30/40 dB "
        "SNR robustness supplement. Metrics are recomputed with the detailed evaluator in "
        "evaluate_missing_requested_metrics.py."
    )
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    final = {}
    requested_rows = []
    if args.mode in {"clean", "all"}:
        final["continual_subject5fold_adaptation_optimized"] = metric_eval.run_optimized(
            args, subjects, arrays, device
        )
        requested_rows.extend(final["continual_subject5fold_adaptation_optimized"]["requested"])
    if args.mode in {"snr", "all"}:
        final["continual_snr_robustness"] = metric_eval.run_snr(args, subjects, arrays, device)
        requested_rows.extend(final["continual_snr_robustness"]["requested"])

    metric_eval.write_csv(args.output_dir / "requested_120seg_unified_metrics.csv", requested_rows)
    metric_eval.write_csv(args.output_dir / "requested_metrics_all.csv", requested_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"Saved requested 120-segment metric table to: {args.output_dir / 'requested_120seg_unified_metrics.csv'}")


if __name__ == "__main__":
    main()

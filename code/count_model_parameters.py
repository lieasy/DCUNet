import argparse
import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_continual_subject5fold_adaptation import HeartSoundUNet  # noqa: E402
from train_compare_reconstruction_methods import MODEL_NAMES, build_model  # noqa: E402
from train_continual_model_ablation_suite import MODEL_VARIANTS, make_model  # noqa: E402
from train_continual_subject5fold_kd_student import UltraLightStudentUNet  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "model_parameter_counts"


def parse_features(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "non_trainable_params": int(total - trainable),
        "size_mb_fp32": float(total * 4 / (1024 * 1024)),
    }


def build_rows(args) -> list[dict]:
    rows = []

    models = {
        "DCUNet": HeartSoundUNet(args.base_channels),
        "DCUNet-KD": UltraLightStudentUNet(features=args.student_features, dropout=args.student_dropout),
    }
    for name, model in models.items():
        rows.append({"group": "proposed", "model": name, **count_parameters(model)})

    for variant in args.ablation_variants:
        model = make_model(variant, args.base_channels)
        rows.append(
            {
                "group": "ablation",
                "model": variant,
                "description": MODEL_VARIANTS[variant]["description"],
                **count_parameters(model),
            }
        )

    for name in args.compare_models:
        model = build_model(name, args.compare_base_channels)
        rows.append({"group": "comparison", "model": name, **count_parameters(model)})

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Count parameters for DCUNet, DCUNet-KD, ablation models, and comparison models.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--compare-base-channels", type=int, default=32)
    parser.add_argument("--student-features", type=parse_features, default=(24, 48, 96, 192))
    parser.add_argument("--student-dropout", type=float, default=0.1)
    parser.add_argument("--ablation-variants", type=str, default="all")
    parser.add_argument("--compare-models", type=str, default="all")
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.ablation_variants = (
        list(MODEL_VARIANTS.keys())
        if args.ablation_variants.strip().lower() == "all"
        else [item.strip() for item in args.ablation_variants.split(",") if item.strip()]
    )
    args.compare_models = (
        list(MODEL_NAMES)
        if args.compare_models.strip().lower() == "all"
        else [item.strip().lower() for item in args.compare_models.split(",") if item.strip()]
    )

    rows = build_rows(args)
    write_csv(args.output_dir / "model_parameter_counts.csv", rows)
    (args.output_dir / "model_parameter_counts.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"{row['group']:10s} {row['model']:28s} "
            f"total={row['total_params']:,} trainable={row['trainable_params']:,} size={row['size_mb_fp32']:.2f} MB"
        )
    print(f"Saved parameter counts to: {args.output_dir}")


if __name__ == "__main__":
    main()

import csv
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "requested_metric_supplement"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values):
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def safe_std(values):
    vals = [v for v in values if v is not None]
    return pstdev(vals) if len(vals) > 1 else (0.0 if len(vals) == 1 else None)


def metric_row(
    source_dir,
    model,
    protocol,
    stage,
    support=None,
    noise_type=None,
    snr_db=None,
    pcc=None,
    mse=None,
    mae=None,
    rmse=None,
    sc=None,
    epcc=None,
    mean_hr=None,
    sd_hr=None,
    n=None,
    mse_note="",
    missing_note="",
):
    return {
        "source_dir": source_dir,
        "model": model,
        "protocol": protocol,
        "stage": stage,
        "support_segments": support,
        "noise_type": noise_type,
        "snr_db": snr_db,
        "n": n,
        "PCC[%]": None if pcc is None else pcc * 100.0,
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "SC": sc,
        "ePCC[%]": None if epcc is None else epcc * 100.0,
        "Mean HR [bpm]": mean_hr,
        "SD [bpm]": sd_hr,
        "mse_note": mse_note,
        "missing_note": missing_note,
    }


def summarize_segment_mixed(rows):
    root = ROOT / "results" / "segment_mixed_5fold_recovery"
    data = load_json(root / "segment_mixed_5fold_summary.json")
    folds = data.get("fold_summaries", [])
    rmses = [f.get("test_metrics", {}).get("rmse") for f in folds]
    mses = [r * r for r in rmses if r is not None]
    ag = data.get("aggregate", {})
    rows.append(
        metric_row(
            source_dir=str(root.relative_to(ROOT)),
            model="segment_mixed_5fold_recovery",
            protocol="segment_mixed",
            stage="test",
            pcc=ag.get("test_corr_mean"),
            mse=safe_mean(mses),
            mae=ag.get("test_mae_mean"),
            rmse=ag.get("test_rmse_mean"),
            n=ag.get("n_folds"),
            mse_note="computed as mean(fold_rmse^2) from normalized RMSE",
            missing_note="SC/ePCC/Mean HR/SD are not present in saved JSON; rerun detailed evaluation to compute them.",
        )
    )


def summarize_optimized(rows):
    root = ROOT / "results" / "continual_subject5fold_adaptation_optimized"
    data = load_json(root / "optimized_subject5fold_summary.json")
    for item in data.get("overall_support_summaries", []):
        support = item.get("support_segments")
        per_subject = item.get("per_subject", [])
        ag = item.get("aggregate", {})
        for stage in ("before", "after"):
            rmses = [r.get(f"{stage}_rmse") for r in per_subject]
            mses = [r * r for r in rmses if r is not None]
            rows.append(
                metric_row(
                    source_dir=str(root.relative_to(ROOT)),
                    model="continual_subject5fold_adaptation_optimized",
                    protocol="continual_subject_5fold",
                    stage=stage,
                    support=support,
                    pcc=ag.get(f"{stage}_corr_mean"),
                    mse=safe_mean(mses),
                    mae=ag.get(f"{stage}_mae_mean"),
                    rmse=ag.get(f"{stage}_rmse_mean"),
                    n=ag.get("n_subjects"),
                    mse_note="computed as mean(subject_rmse^2) from normalized RMSE",
                    missing_note="SC/ePCC/Mean HR/SD are not present in saved JSON; rerun detailed evaluation to compute them.",
                )
            )


def summarize_snr(rows):
    root = ROOT / "results" / "continual_snr_robustness"
    data = load_json(root / "continual_snr_robustness_summary.json")
    for item in data.get("overall_condition_summaries", []):
        support = item.get("support_segments")
        noise_type = item.get("noise_type")
        snr_db = item.get("snr_db")
        per_subject = item.get("per_subject", [])
        ag = item.get("aggregate", {})
        for stage in ("before", "after"):
            rmses = [r.get(f"{stage}_rmse") for r in per_subject]
            mses = [r * r for r in rmses if r is not None]
            rows.append(
                metric_row(
                    source_dir=str(root.relative_to(ROOT)),
                    model="continual_snr_robustness",
                    protocol="continual_snr_robustness",
                    stage=stage,
                    support=support,
                    noise_type=noise_type,
                    snr_db=snr_db,
                    pcc=ag.get(f"{stage}_corr_mean"),
                    mse=safe_mean(mses),
                    mae=ag.get(f"{stage}_mae_mean"),
                    rmse=ag.get(f"{stage}_rmse_mean"),
                    n=ag.get("n_subjects"),
                    mse_note="computed as mean(subject_rmse^2) from normalized RMSE",
                    missing_note="SC/ePCC/Mean HR/SD are not present in saved JSON; rerun detailed evaluation to compute them.",
                )
            )


def summarize_kd(rows, dirname, model):
    root = ROOT / "results" / dirname
    data = load_json(root / "kd_student_subject5fold_summary.json")
    for item in data.get("overall_support_summaries", []):
        support = item.get("support_segments")
        ag = item.get("aggregate", {})
        for stage in ("before", "after"):
            rmse = ag.get(f"{stage}_rmse_mean")
            rows.append(
                metric_row(
                    source_dir=str(root.relative_to(ROOT)),
                    model=model,
                    protocol="continual_subject_5fold_kd",
                    stage=stage,
                    support=support,
                    pcc=ag.get(f"{stage}_pcc_mean"),
                    mse=None if rmse is None else rmse * rmse,
                    mae=ag.get(f"{stage}_mae_mean"),
                    rmse=rmse,
                    sc=ag.get(f"{stage}_spectral_convergence_mean"),
                    epcc=ag.get(f"{stage}_envelope_pcc_mean"),
                    mean_hr=ag.get(f"{stage}_mean_hr_bias_mean"),
                    sd_hr=ag.get(f"{stage}_mean_hr_bias_std"),
                    n=ag.get("n_subjects"),
                    mse_note="computed as aggregate_rmse_mean^2 because KD summaries do not save MSE",
                )
            )


def write_outputs(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_dir",
        "model",
        "protocol",
        "stage",
        "support_segments",
        "noise_type",
        "snr_db",
        "n",
        "PCC[%]",
        "MSE",
        "MAE",
        "RMSE",
        "SC",
        "ePCC[%]",
        "Mean HR [bpm]",
        "SD [bpm]",
        "mse_note",
        "missing_note",
    ]
    csv_path = OUT_DIR / "requested_metrics_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path = OUT_DIR / "requested_metrics_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main():
    rows = []
    summarize_snr(rows)
    summarize_optimized(rows)
    summarize_kd(rows, "continual_subject5fold_kd_student_reuse_teacher", "kd_student_reuse_teacher")
    summarize_kd(rows, "continual_subject5fold_kd_student_tuned", "kd_student_tuned")
    summarize_segment_mixed(rows)
    csv_path, json_path = write_outputs(rows)
    print(f"Wrote {len(rows)} rows")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()

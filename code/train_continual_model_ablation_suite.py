import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from refine_continual_subject5fold_adaptation import (
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
    HeartSoundLoss,
    HeartSoundUNet,
    MultiScaleStem,
    ResidualBlock,
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
from train_continual_subject5fold_adaptation_optimized import (
    build_balanced_subject_folds,
    compute_stats,
    sanitize_args,
    split_base_indices,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "continual_model_ablation_suite"
RECONSTRUCTION_METRIC_KEYS = [
    "spectral_convergence",
    "envelope_pcc",
    "sc",
    "epcc",
]


class SingleScaleStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 41, padding=20),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PlainConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlockNoDropout(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(out_channels, out_channels, 7, padding=3),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + self.shortcut(x))


class HeartSoundUNetNoMultiScaleStem(HeartSoundUNet):
    def __init__(self, base_channels=32):
        super().__init__(base_channels)
        self.stem = nn.Sequential(SingleScaleStem(2, base_channels), ResidualBlock(base_channels, base_channels))


class HeartSoundUNetNoResidualShortcut(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stem = nn.Sequential(MultiScaleStem(2, c1), PlainConvBlock(c1, c1))
        self.pool = nn.MaxPool1d(2)
        self.enc1 = PlainConvBlock(c1, c2)
        self.enc2 = PlainConvBlock(c2, c3)
        self.enc3 = PlainConvBlock(c3, c4)
        self.bottleneck = nn.Sequential(PlainConvBlock(c4, c4 * 2, 0.2), PlainConvBlock(c4 * 2, c4 * 2, 0.2))
        self.dec3 = PlainConvBlock(c4 * 2 + c4, c4)
        self.dec2 = PlainConvBlock(c4 + c3, c3)
        self.dec1 = PlainConvBlock(c3 + c2, c2)
        self.dec0 = PlainConvBlock(c2 + c1, c1)
        self.head = nn.Sequential(PlainConvBlock(c1, c1), nn.Conv1d(c1, 1, 1))

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x = self.bottleneck(self.pool(s3))
        x = self.dec3(self.upcat(x, s3))
        x = self.dec2(self.upcat(x, s2))
        x = self.dec1(self.upcat(x, s1))
        x = self.dec0(self.upcat(x, s0))
        return self.head(x)

    @staticmethod
    def upcat(x, skip):
        return torch.cat([F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False), skip], dim=1)


class HeartSoundUNetNoEncoderDecoderSkip(HeartSoundUNet):
    @staticmethod
    def upcat(x, skip):
        up = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return torch.cat([up, torch.zeros_like(skip)], dim=1)


class HeartSoundUNetShallowBottleneck(HeartSoundUNet):
    def __init__(self, base_channels=32):
        super().__init__(base_channels)
        c4 = base_channels * 8
        self.bottleneck = nn.Sequential(ResidualBlock(c4, c4 * 2, 0.2))


class HeartSoundUNetNoDropout(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stem = nn.Sequential(MultiScaleStem(2, c1), ResidualBlockNoDropout(c1, c1))
        self.pool = nn.MaxPool1d(2)
        self.enc1 = ResidualBlockNoDropout(c1, c2)
        self.enc2 = ResidualBlockNoDropout(c2, c3)
        self.enc3 = ResidualBlockNoDropout(c3, c4)
        self.bottleneck = nn.Sequential(ResidualBlockNoDropout(c4, c4 * 2), ResidualBlockNoDropout(c4 * 2, c4 * 2))
        self.dec3 = ResidualBlockNoDropout(c4 * 2 + c4, c4)
        self.dec2 = ResidualBlockNoDropout(c4 + c3, c3)
        self.dec1 = ResidualBlockNoDropout(c3 + c2, c2)
        self.dec0 = ResidualBlockNoDropout(c2 + c1, c1)
        self.head = nn.Sequential(ResidualBlockNoDropout(c1, c1), nn.Conv1d(c1, 1, 1))

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x = self.bottleneck(self.pool(s3))
        x = self.dec3(self.upcat(x, s3))
        x = self.dec2(self.upcat(x, s2))
        x = self.dec1(self.upcat(x, s1))
        x = self.dec0(self.upcat(x, s0))
        return self.head(x)

    @staticmethod
    def upcat(x, skip):
        return torch.cat([F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False), skip], dim=1)


class HeartSoundUNetSimpleHead(HeartSoundUNet):
    def __init__(self, base_channels=32):
        super().__init__(base_channels)
        self.head = nn.Conv1d(base_channels, 1, 1)


MODEL_VARIANTS = {
    "full_model": {
        "description": "完整当前模型：MultiScaleStem + Residual UNet + skip connections + deep bottleneck + dropout + refinement head.",
        "factory": HeartSoundUNet,
        "is_ablation": False,
    },
    "no_multiscale_stem": {
        "description": "去掉多尺度输入特征提取，MultiScaleStem 替换为单尺度卷积 stem。",
        "factory": HeartSoundUNetNoMultiScaleStem,
        "is_ablation": True,
    },
    "no_residual_shortcut": {
        "description": "去掉 ResidualBlock 中的 shortcut 残差连接，保留卷积层数和通道规模。",
        "factory": HeartSoundUNetNoResidualShortcut,
        "is_ablation": True,
    },
    "no_encoder_decoder_skip": {
        "description": "去掉 UNet 编码器到解码器的 skip feature 传递，仅保留上采样主干。",
        "factory": HeartSoundUNetNoEncoderDecoderSkip,
        "is_ablation": True,
    },
    "shallow_bottleneck": {
        "description": "去掉深层 bottleneck 的第二个残差块，削弱深层上下文建模。",
        "factory": HeartSoundUNetShallowBottleneck,
        "is_ablation": True,
    },
    "no_dropout": {
        "description": "去掉所有 ResidualBlock 内 dropout，考察正则化模块作用。",
        "factory": HeartSoundUNetNoDropout,
        "is_ablation": True,
    },
    "simple_head": {
        "description": "去掉输出端 refinement residual head，直接用 1x1 卷积输出恢复信号。",
        "factory": HeartSoundUNetSimpleHead,
        "is_ablation": True,
    },
}


def parse_list(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return list(MODEL_VARIANTS.keys())
    items = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in items if item not in MODEL_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid variants: {list(MODEL_VARIANTS)}")
    return items


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def make_model(variant: str, base_channels: int):
    return MODEL_VARIANTS[variant]["factory"](base_channels)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values):
    vals = [
        float(value)
        for value in values
        if value is not None and not math.isnan(float(value)) and not math.isinf(float(value))
    ]
    return float(np.mean(vals)) if vals else None


def safe_std(values):
    vals = [
        float(value)
        for value in values
        if value is not None and not math.isnan(float(value)) and not math.isinf(float(value))
    ]
    return float(np.std(vals)) if vals else None


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


def extract_envelope(signal: np.ndarray, sample_rate: int = 1000, cutoff: float = 20.0) -> np.ndarray:
    envelope = hilbert_envelope_np(signal)
    return moving_average(envelope, max(3, round(sample_rate / cutoff)))


def spectral_convergence_batch(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 1024, hop_length: int = 256):
    pred_2d = pred.squeeze(1)
    target_2d = target.squeeze(1)
    n_fft = min(n_fft, pred_2d.shape[-1])
    window = torch.hann_window(n_fft, device=pred.device)
    pred_spec = torch.stft(pred_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True).abs()
    target_spec = torch.stft(target_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True).abs()
    numerator = torch.linalg.vector_norm(pred_spec - target_spec, dim=(1, 2))
    denominator = torch.linalg.vector_norm(target_spec, dim=(1, 2)).clamp_min(1e-8)
    return (numerator / denominator).detach().cpu().numpy().astype(float).tolist()


class ReconstructionMetricSegmentDataset(SegmentDataset):
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


def evaluate_reconstruction_metrics(model, subjects, arrays, indices, stats, args, device, desc: str):
    model.eval()
    dataset = ReconstructionMetricSegmentDataset(subjects, arrays, indices, stats, augment=False)
    loader = make_loader(dataset, args.batch_size, False, device, args.seed, args.num_workers)
    sc_values = []
    epcc_values = []
    with torch.no_grad():
        for x, _y, raw_target, _sid, params in loader:
            x = x.to(device, non_blocking=True)
            raw_target = raw_target.to(device, non_blocking=True)
            params = params.to(device, non_blocking=True)
            pred = denorm_prediction(model(x), params, stats)
            target = raw_target
            sc_values.extend(spectral_convergence_batch(pred, target))
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            for item_idx in range(pred_np.shape[0]):
                pred_env = extract_envelope(pred_np[item_idx, 0], args.sample_rate)
                target_env = extract_envelope(target_np[item_idx, 0], args.sample_rate)
                pred_env_norm = pred_env / (np.max(pred_env) + 1e-8)
                target_env_norm = target_env / (np.max(target_env) + 1e-8)
                epcc_values.append(pearson_np(pred_env_norm, target_env_norm))
    out = {
        "spectral_convergence": safe_mean(sc_values),
        "envelope_pcc": safe_mean(epcc_values),
    }
    out["sc"] = out["spectral_convergence"]
    out["epcc"] = out["envelope_pcc"]
    print(
        f"{desc} metrics: SC={out['spectral_convergence'] if out['spectral_convergence'] is not None else float('nan'):.4f}, "
        f"ePCC={out['envelope_pcc'] if out['envelope_pcc'] is not None else float('nan'):.4f}"
    )
    return out


def flatten_reconstruction_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": metrics.get(key) for key in RECONSTRUCTION_METRIC_KEYS}


def aggregate_variant_rows(rows):
    out = aggregate(rows)
    for key in RECONSTRUCTION_METRIC_KEYS:
        for phase in ["before", "after", "delta"]:
            field = f"{phase}_{key}"
            vals = [row.get(field) for row in rows]
            out[f"{field}_mean"] = safe_mean(vals)
            out[f"{field}_std"] = safe_std(vals)
    return out


def merge_support_summaries(existing: list[dict], current: list[dict]) -> list[dict]:
    merged = {
        int(item["support_segments"]): item
        for item in existing
        if "support_segments" in item
    }
    for item in current:
        merged[int(item["support_segments"])] = item
    return [merged[key] for key in sorted(merged)]


def merge_fold_summaries(existing: list[dict], current: list[dict]) -> list[dict]:
    merged = {int(item["fold"]): item for item in existing if "fold" in item}
    for item in current:
        fold_idx = int(item["fold"])
        old = merged.get(fold_idx, {})
        combined = {**old, **item}
        combined["support_summaries"] = merge_support_summaries(
            old.get("support_summaries", []),
            item.get("support_summaries", []),
        )
        merged[fold_idx] = combined
    return [merged[key] for key in sorted(merged)]


def train_base_model_variant(args, variant, subjects, arrays, base_ids: set[int], fold_idx: int, device):
    base_dir = args.output_dir / variant / f"fold_{fold_idx}" / "base_model"
    base_dir.mkdir(parents=True, exist_ok=True)
    best_path = base_dir / "best_base_model.pth"
    metrics_path = base_dir / "base_test_metrics.json"

    if args.reuse_base_models and best_path.exists() and metrics_path.exists():
        print(f"{variant} fold {fold_idx}: reusing base model {best_path}")
        return best_path, load_json(metrics_path, {})

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

    model = make_model(variant, args.base_channels).to(device)
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
            f"{variant} fold {fold_idx} base {epoch}/{args.base_epochs}",
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"{variant} fold {fold_idx} base-val")
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
                    "variant": variant,
                    "fold": fold_idx,
                    "args": sanitize_args(args),
                },
                best_path,
            )
        else:
            bad_epochs += 1

        print(
            f"{variant} fold {fold_idx} base epoch {epoch:03d}: "
            f"train PCC {train_metrics['corr']:.4f}, val PCC {val_metrics['corr']:.4f}, "
            f"lr {optimizer.param_groups[0]['lr']:.2e}, patience {bad_epochs}"
        )
        if args.base_early_stop > 0 and epoch >= args.base_min_epochs and bad_epochs >= args.base_early_stop:
            print(f"{variant} fold {fold_idx} base early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device, None, scaler, use_amp, f"{variant} fold {fold_idx} base-test")
    (base_dir / "base_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    return best_path, test_metrics


def adapt_subject_variant(args, variant, subjects, arrays, base_model_path, sample_id, support_segments, device):
    checkpoint = torch.load(base_model_path, map_location=device)
    stats = checkpoint["normalization_stats"]
    support, query = choose_support(
        subjects, arrays, sample_id, support_segments, args.min_query_segments, args.seed, args.support_strategy
    )
    support_train, support_val = split_support(
        support, args.seed + sample_id + support_segments, args.support_val_ratio, args.support_val_segments
    )

    query_set = SegmentDataset(subjects, arrays, query, stats, augment=False)
    support_train_set = SegmentDataset(subjects, arrays, support_train, stats, augment=args.adapt_augment)
    support_val_set = SegmentDataset(subjects, arrays, support_val, stats, augment=False)

    model = make_model(variant, args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = HeartSoundLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp

    before = run_epoch(
        model,
        make_loader(query_set, args.batch_size, False, device, args.seed, args.num_workers),
        criterion,
        device,
        None,
        scaler,
        use_amp,
        f"{variant} S{sample_id} before",
    )
    before_reconstruction_metrics = evaluate_reconstruction_metrics(
        model,
        subjects,
        arrays,
        query,
        stats,
        args,
        device,
        f"{variant} S{sample_id} before",
    )
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
            tr = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scaler,
                use_amp,
                f"{variant} S{sample_id} {stage_name} {epoch}/{epochs}",
            )
            va = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"{variant} S{sample_id} support-val")
            scheduler.step(va["corr"])
            history.append(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "train_corr": tr["corr"],
                    "val_corr": va["corr"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            if va["corr"] > best_val + args.min_delta:
                best_val = va["corr"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    after = run_epoch(
        model,
        make_loader(query_set, args.batch_size, False, device, args.seed, args.num_workers),
        criterion,
        device,
        None,
        scaler,
        use_amp,
        f"{variant} S{sample_id} after",
    )
    after_reconstruction_metrics = evaluate_reconstruction_metrics(
        model,
        subjects,
        arrays,
        query,
        stats,
        args,
        device,
        f"{variant} S{sample_id} after",
    )
    delta_reconstruction_metrics = {}
    for key in RECONSTRUCTION_METRIC_KEYS:
        before_value = before_reconstruction_metrics.get(key)
        after_value = after_reconstruction_metrics.get(key)
        delta_reconstruction_metrics[key] = (
            None if before_value is None or after_value is None else after_value - before_value
        )
    return {
        "sample_id": sample_id,
        "support_segments": len(support),
        "query_segments": len(query),
        "support_seconds": len(support) * args.segment_seconds,
        "before_corr": before["corr"],
        "after_corr": after["corr"],
        "delta_corr": after["corr"] - before["corr"],
        "before_rmse": before["rmse"],
        "after_rmse": after["rmse"],
        "before_mae": before["mae"],
        "after_mae": after["mae"],
        "best_support_val_corr": best_val,
        "before_reconstruction_metrics": before_reconstruction_metrics,
        "after_reconstruction_metrics": after_reconstruction_metrics,
        "delta_reconstruction_metrics": delta_reconstruction_metrics,
        **flatten_reconstruction_metrics("before", before_reconstruction_metrics),
        **flatten_reconstruction_metrics("after", after_reconstruction_metrics),
        **flatten_reconstruction_metrics("delta", delta_reconstruction_metrics),
        "history": history,
    }


def run_variant(args, variant, subjects, arrays, folds, all_ids, support_list, device):
    all_by_support = {support: [] for support in support_list}
    fold_summaries = []
    max_folds = args.folds if args.max_folds <= 0 else min(args.max_folds, args.folds)

    for fold_idx in range(max_folds):
        test_ids = sorted(folds[fold_idx])
        base_ids = set(all_ids) - set(test_ids)
        print(f"\n========== {variant} fold {fold_idx}: test {test_ids} ==========")

        base_model_path, base_test_metrics = train_base_model_variant(
            args, variant, subjects, arrays, base_ids, fold_idx, device
        )

        fold_summary = {
            "fold": fold_idx,
            "test_ids": test_ids,
            "base_ids": sorted(base_ids),
            "base_test_metrics": base_test_metrics,
            "support_summaries": [],
        }
        for support_segments in support_list:
            support_dir = args.output_dir / variant / f"fold_{fold_idx}" / f"support_{support_segments:03d}"
            support_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for sample_id in test_ids:
                row = adapt_subject_variant(
                    args, variant, subjects, arrays, base_model_path, sample_id, support_segments, device
                )
                light = {k: v for k, v in row.items() if k != "history"}
                light["fold"] = fold_idx
                (support_dir / f"sample{sample_id:02d}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                rows.append(light)
                all_by_support[support_segments].append(light)

            support_summary = {
                "fold": fold_idx,
                "support_segments": support_segments,
                "aggregate": aggregate_variant_rows(rows),
                "per_subject": rows,
            }
            (support_dir / "summary.json").write_text(json.dumps(support_summary, indent=2), encoding="utf-8")
            fold_summary["support_summaries"].append(
                {"support_segments": support_segments, "aggregate": support_summary["aggregate"]}
            )
            print(
                f"{variant} fold {fold_idx} support {support_segments}: "
                f"after PCC {support_summary['aggregate']['after_corr_mean']:.4f}, "
                f"SC {support_summary['aggregate']['after_spectral_convergence_mean']:.4f}, "
                f"ePCC {support_summary['aggregate']['after_envelope_pcc_mean']:.4f}, "
                f"delta {support_summary['aggregate']['delta_corr_mean']:.4f}"
            )
        fold_summaries.append(fold_summary)

    overall = []
    for support_segments, rows in all_by_support.items():
        overall.append(
            {
                "support_segments": support_segments,
                "support_seconds": support_segments * args.segment_seconds,
                "aggregate": aggregate_variant_rows(rows),
                "per_subject": rows,
            }
        )

    summary = {
        "variant": variant,
        "description": MODEL_VARIANTS[variant]["description"],
        "is_ablation": MODEL_VARIANTS[variant]["is_ablation"],
        "fold_test_ids": folds[:max_folds],
        "fold_summaries": fold_summaries,
        "overall_support_summaries": overall,
    }
    variant_dir = args.output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_existing_summaries:
        existing = load_json(variant_dir / "model_ablation_summary.json", {})
        if existing:
            summary["fold_test_ids"] = existing.get("fold_test_ids", summary["fold_test_ids"])
            summary["fold_summaries"] = merge_fold_summaries(existing.get("fold_summaries", []), fold_summaries)
            summary["overall_support_summaries"] = merge_support_summaries(
                existing.get("overall_support_summaries", []),
                overall,
            )
    (variant_dir / "model_ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model-structure ablation suite under the subject-independent continual few-shot adaptation framework."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", type=str, default="all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--support-list", type=str, default="48,72,96,120")
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
    parser.add_argument("--sample-rate", type=int, default=1000)
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
    parser.add_argument(
        "--reuse-base-models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse existing fold base models and base-test metrics when present; useful for adding a new support size.",
    )
    parser.add_argument(
        "--merge-existing-summaries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge newly computed support summaries with existing summaries in the output directory.",
    )
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

    variants = parse_list(args.variants)
    support_list = parse_int_list(args.support_list)
    subjects = discover_subjects(args.data_dir)
    arrays = [np.load(subject["file"], mmap_mode="r") for subject in subjects]
    folds = build_balanced_subject_folds(subjects, args.folds)
    all_ids = sorted(int(subject["sample_id"]) for subject in subjects)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    existing_run_config = load_json(args.output_dir / "run_config.json", {}) if args.merge_existing_summaries else {}
    run_config = sanitize_args(args)
    run_config["variants"] = variants
    if existing_run_config:
        run_config["current_run_support_list"] = support_list
        run_config["support_list"] = sorted({*existing_run_config.get("support_list", []), *support_list})
    else:
        run_config["support_list"] = support_list
    run_config["fold_test_ids"] = folds
    run_config["model_variants"] = {
        name: {k: v for k, v in config.items() if k != "factory"} for name, config in MODEL_VARIANTS.items()
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Using device: {device}")
    print(f"Running model-structure variants: {variants}")
    print("Framework: subject-independent 5-fold continual few-shot adaptation.")

    existing_suite = (
        load_json(args.output_dir / "continual_model_ablation_suite_summary.json", {})
        if args.merge_existing_summaries
        else {}
    )
    suite = {
        "variant_order": existing_suite.get("variant_order", []),
        "variant_summaries": existing_suite.get("variant_summaries", {}),
    }
    for variant in variants:
        if variant not in suite["variant_order"]:
            suite["variant_order"].append(variant)
    for variant in variants:
        print(f"\n==================== Model ablation variant: {variant} ====================")
        suite["variant_summaries"][variant] = run_variant(
            args, variant, subjects, arrays, folds, all_ids, support_list, device
        )

    (args.output_dir / "continual_model_ablation_suite_summary.json").write_text(
        json.dumps(suite, indent=2), encoding="utf-8"
    )

    print("\n==================== Continual model ablation summary ====================")
    for variant in suite["variant_order"]:
        for item in suite["variant_summaries"][variant]["overall_support_summaries"]:
            agg = item["aggregate"]
            sc_text = (
                "nan"
                if agg.get("after_spectral_convergence_mean") is None
                else f"{agg['after_spectral_convergence_mean']:.4f}"
            )
            epcc_text = (
                "nan"
                if agg.get("after_envelope_pcc_mean") is None
                else f"{agg['after_envelope_pcc_mean']:.4f}"
            )
            print(
                f"{variant} support {item['support_segments']}: "
                f"before {agg['before_corr_mean']:.4f}, "
                f"after {agg['after_corr_mean']:.4f} +/- {agg['after_corr_std']:.4f}, "
                f"SC {sc_text}, "
                f"ePCC {epcc_text}, "
                f"delta {agg['delta_corr_mean']:.4f}"
            )
    print(f"Saved summary to: {args.output_dir / 'continual_model_ablation_suite_summary.json'}")


if __name__ == "__main__":
    main()

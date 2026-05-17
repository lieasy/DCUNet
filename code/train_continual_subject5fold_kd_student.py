import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_continual_unified_metrics import (  # noqa: E402
    METRIC_KEYS,
    MetricSegmentDataset,
    evaluate_full_metrics,
    safe_float,
    safe_mean,
    safe_std,
)
from refine_continual_subject5fold_adaptation import (  # noqa: E402
    DEFAULT_DATA_DIR,
    HeartSoundLoss,
    HeartSoundUNet,
    SegmentDataset,
    choose_support,
    discover_subjects,
    make_loader,
    run_epoch,
    set_seed,
    split_support,
)
from train_continual_subject5fold_adaptation_optimized import (  # noqa: E402
    build_balanced_subject_folds,
    compute_stats,
    sanitize_args,
    split_base_indices,
    train_base_model,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "continual_subject5fold_kd_student"


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class LightweightResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dropout=0.1):
        super().__init__()
        self.dw_conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dw_conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=False)
        self.dropout = nn.Dropout(dropout)
        hidden = max(4, out_channels // 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, hidden, 1),
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.dropout(self.act(self.bn1(self.dw_conv1(x))))
        out = self.dropout(self.bn2(self.dw_conv2(out)))
        out = out * self.attn(out)
        return self.act(out + residual)


class ChannelSplitAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.split = max(1, in_channels // reduction)

    def forward(self, x):
        weights = []
        for part in torch.split(x, self.split, dim=1):
            weights.append(torch.sigmoid(F.adaptive_avg_pool1d(part, 1) + F.adaptive_max_pool1d(part, 1)))
        return x * torch.cat(weights, dim=1)


class UltraLightStudentUNet(nn.Module):
    """Student framework copied from the previous KD notebook and adapted to this continual pipeline."""

    def __init__(self, input_channels=2, output_channels=1, features=(24, 48, 96, 192), dropout=0.1):
        super().__init__()
        features = list(features)
        self.features = features
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        self.ms_feature_extractor = nn.Sequential(
            DepthwiseSeparableConv(input_channels, 24, 21, padding=10),
            nn.LeakyReLU(0.1, inplace=False),
            DepthwiseSeparableConv(24, 48, 41, padding=20),
            nn.LeakyReLU(0.1, inplace=False),
            DepthwiseSeparableConv(48, 48, 81, padding=40),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv1d(48, features[0], 1),
        )
        self.channel_attn = ChannelSplitAttention(features[0])

        for i in range(len(features)):
            in_ch = features[i - 1] if i > 0 else features[0]
            self.encoder.append(LightweightResidualBlock(in_ch, features[i], kernel_size=7, dropout=dropout))
            self.encoder.append(LightweightResidualBlock(features[i], features[i], kernel_size=7, dropout=dropout))

        self.bottleneck = nn.Sequential(
            LightweightResidualBlock(features[-1], features[-1] * 2, kernel_size=7, dropout=0.2),
            LightweightResidualBlock(features[-1] * 2, features[-1] * 2, kernel_size=7, dropout=0.2),
            nn.Conv1d(features[-1] * 2, features[-1] * 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout(0.3),
        )

        for i in reversed(range(len(features))):
            skip_channels = features[i]
            prev_channels = features[-1] * 2 if i == len(features) - 1 else features[i + 1]
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(prev_channels + skip_channels, features[i], 1),
                    LightweightResidualBlock(features[i], features[i], kernel_size=7, dropout=dropout),
                )
            )
            self.decoder.append(LightweightResidualBlock(features[i], features[i], kernel_size=7, dropout=dropout))

        self.final_conv = nn.Sequential(
            LightweightResidualBlock(features[0], 64, kernel_size=7, dropout=dropout),
            LightweightResidualBlock(64, 32, kernel_size=7, dropout=dropout),
            nn.Conv1d(32, output_channels, 1),
        )
        self.refiner = nn.Sequential(
            DepthwiseSeparableConv(output_channels, 32, 21, padding=10),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv1d(32, output_channels, 1),
        )

    def forward(self, x, return_features=False):
        x = self.channel_attn(self.ms_feature_extractor(x))
        features = [x]
        skips = []
        for i in range(0, len(self.encoder), 2):
            x = self.encoder[i](x)
            x = self.encoder[i + 1](x)
            skips.append(x)
            features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        features.append(x)

        for i in range(0, len(self.decoder), 2):
            x = self.upsample(x)
            skip = skips[-(i // 2 + 1)]
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.decoder[i](x)
            x = self.decoder[i + 1](x)
            features.append(x)

        x = self.final_conv(x)
        x = x + self.refiner(x)
        return (x, features) if return_features else x


class StudentDistillationLoss(nn.Module):
    def __init__(
        self,
        hard_weight=0.55,
        soft_weight=0.30,
        corr_weight=0.15,
        temperature=3.0,
        teacher_gate_min=0.15,
        teacher_gate_max=0.75,
    ):
        super().__init__()
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.corr_weight = corr_weight
        self.temperature = temperature
        self.teacher_gate_min = teacher_gate_min
        self.teacher_gate_max = teacher_gate_max
        self.task_loss = HeartSoundLoss()

    @staticmethod
    def pair_corr(pred, target):
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        return (pred * target).sum(dim=-1) / torch.sqrt(
            (pred.square().sum(dim=-1) * target.square().sum(dim=-1)).clamp_min(1e-8)
        )

    def teacher_gate(self, teacher_out, target):
        corr = self.pair_corr(teacher_out.detach(), target).squeeze(1)
        gate = (corr - self.teacher_gate_min) / max(1e-6, self.teacher_gate_max - self.teacher_gate_min)
        return gate.clamp(0.0, 1.0).view(-1, 1, 1), corr

    def forward(self, student_out, teacher_out, target, kd_scale=1.0):
        hard = self.task_loss(student_out, target)
        gate, teacher_target_corr = self.teacher_gate(teacher_out, target)
        soft = (
            (student_out / self.temperature - teacher_out.detach() / self.temperature).square() * gate
        ).mean() * (self.temperature**2)
        student_teacher_corr = self.pair_corr(student_out, teacher_out.detach()).squeeze(1)
        corr = ((1.0 - student_teacher_corr) * gate.view(-1)).mean()
        loss = self.hard_weight * hard + kd_scale * (self.soft_weight * soft + self.corr_weight * corr)
        return loss, {
            "hard": float(hard.detach().cpu()),
            "soft": float(soft.detach().cpu()),
            "corr": float(corr.detach().cpu()),
            "teacher_target_corr": float(teacher_target_corr.mean().detach().cpu()),
            "teacher_gate": float(gate.mean().detach().cpu()),
            "kd_scale": float(kd_scale),
        }


def parse_features(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def load_teacher(base_model_path: Path, args, device):
    checkpoint = torch.load(base_model_path, map_location=device)
    teacher = HeartSoundUNet(args.base_channels).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher, checkpoint


def make_student(args, device):
    return UltraLightStudentUNet(features=args.student_features, dropout=args.student_dropout).to(device)


def run_kd_epoch(student, teacher, loader, criterion, device, optimizer, scaler, use_amp, desc, kd_scale=1.0):
    train = optimizer is not None
    student.train(train)
    teacher.eval()
    values = {
        "loss": [],
        "corr": [],
        "rmse": [],
        "mae": [],
        "hard": [],
        "soft": [],
        "kd_corr": [],
        "teacher_target_corr": [],
        "teacher_gate": [],
        "kd_scale": [],
    }
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y, _sid in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    teacher_out = teacher(x)
                student_out = student(x)
                loss, parts = criterion(student_out, teacher_out, y, kd_scale=kd_scale)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            corr, rmse, mae = batch_metrics(student_out.detach(), y)
            n = x.shape[0]
            values["loss"].extend([float(loss.detach().cpu())] * n)
            values["corr"].extend(corr.cpu().numpy().astype(float).tolist())
            values["rmse"].extend(rmse.cpu().numpy().astype(float).tolist())
            values["mae"].extend(mae.cpu().numpy().astype(float).tolist())
            values["hard"].extend([parts["hard"]] * n)
            values["soft"].extend([parts["soft"]] * n)
            values["kd_corr"].extend([parts["corr"]] * n)
            values["teacher_target_corr"].extend([parts["teacher_target_corr"]] * n)
            values["teacher_gate"].extend([parts["teacher_gate"]] * n)
            values["kd_scale"].extend([parts["kd_scale"]] * n)
    return {key: float(np.mean(vals)) for key, vals in values.items()}


def batch_metrics(pred, target):
    pc = pred - pred.mean(dim=-1, keepdim=True)
    tc = target - target.mean(dim=-1, keepdim=True)
    corr = (pc * tc).sum(dim=-1).squeeze(1) / torch.sqrt(
        (pc.square().sum(dim=-1).squeeze(1) * tc.square().sum(dim=-1).squeeze(1)).clamp_min(1e-8)
    )
    mse = (pred - target).square().mean(dim=(1, 2))
    mae = (pred - target).abs().mean(dim=(1, 2))
    return corr, torch.sqrt(mse), mae


def train_student_base_model(args, subjects, arrays, base_ids: set[int], fold_idx: int, teacher_model_path: Path, device):
    student_dir = args.output_dir / f"fold_{fold_idx}" / "student_base_model"
    student_dir.mkdir(parents=True, exist_ok=True)

    base_train, base_val, base_test = split_base_indices(
        subjects,
        base_ids=base_ids,
        seed=args.seed + fold_idx,
        train_ratio=args.base_train_ratio,
        val_ratio=args.base_val_ratio,
    )
    teacher, teacher_checkpoint = load_teacher(teacher_model_path, args, device)
    stats = teacher_checkpoint["normalization_stats"]
    (student_dir / "normalization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    train_set = SegmentDataset(subjects, arrays, base_train, stats, augment=args.base_augment)
    val_set = SegmentDataset(subjects, arrays, base_val, stats, augment=False)
    test_set = SegmentDataset(subjects, arrays, base_test, stats, augment=False)
    train_loader = make_loader(train_set, args.batch_size, True, device, args.seed + fold_idx, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, device, args.seed + fold_idx, args.num_workers)
    test_loader = make_loader(test_set, args.batch_size, False, device, args.seed + fold_idx, args.num_workers)

    student = make_student(args, device)
    criterion = StudentDistillationLoss(
        hard_weight=args.kd_hard_weight,
        soft_weight=args.kd_soft_weight,
        corr_weight=args.kd_corr_weight,
        temperature=args.kd_temperature,
        teacher_gate_min=args.teacher_gate_min,
        teacher_gate_max=args.teacher_gate_max,
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.student_base_lr, weight_decay=args.weight_decay)
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
    best_path = student_dir / "best_student_base_model.pth"
    bad_epochs = 0
    history = []
    for epoch in range(1, args.student_base_epochs + 1):
        train_metrics = run_kd_epoch(
            student,
            teacher,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            f"Fold {fold_idx} StudentKD {epoch}/{args.student_base_epochs}",
        )
        val_metrics = run_kd_epoch(
            student, teacher, val_loader, criterion, device, None, scaler, use_amp, f"Fold {fold_idx} StudentKDVal"
        )
        scheduler.step(val_metrics["corr"])
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        (student_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_metrics["corr"] > best_val + args.min_delta:
            best_val = val_metrics["corr"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "normalization_stats": stats,
                    "best_val_corr": best_val,
                    "fold": fold_idx,
                    "teacher_model_path": str(teacher_model_path),
                    "student_features": list(args.student_features),
                    "args": sanitize_args(args),
                },
                best_path,
            )
        else:
            bad_epochs += 1

        print(
            f"Fold {fold_idx} student epoch {epoch:03d}: "
            f"train PCC {train_metrics['corr']:.4f}, val PCC {val_metrics['corr']:.4f}, "
            f"KD soft {val_metrics['soft']:.4f}, lr {optimizer.param_groups[0]['lr']:.2e}, patience {bad_epochs}"
        )
        if (
            args.student_base_early_stop > 0
            and epoch >= args.student_base_min_epochs
            and bad_epochs >= args.student_base_early_stop
        ):
            print(f"Fold {fold_idx} student base early stopping at epoch {epoch}")
            break

    if args.student_hard_finetune_epochs > 0:
        hard_criterion = HeartSoundLoss()
        student.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=args.student_hard_finetune_lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr
        )
        for epoch in range(1, args.student_hard_finetune_epochs + 1):
            train_metrics = run_epoch(
                student,
                train_loader,
                hard_criterion,
                device,
                optimizer,
                scaler,
                use_amp,
                f"Fold {fold_idx} StudentHard {epoch}/{args.student_hard_finetune_epochs}",
            )
            val_metrics = run_epoch(
                student, val_loader, hard_criterion, device, None, scaler, use_amp, f"Fold {fold_idx} StudentHardVal"
            )
            scheduler.step(val_metrics["corr"])
            history.append(
                {
                    "epoch": f"hard_{epoch}",
                    "lr": optimizer.param_groups[0]["lr"],
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )
            (student_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if val_metrics["corr"] > best_val + args.min_delta:
                best_val = val_metrics["corr"]
                torch.save(
                    {
                        "model_state_dict": student.state_dict(),
                        "normalization_stats": stats,
                        "best_val_corr": best_val,
                        "fold": fold_idx,
                        "teacher_model_path": str(teacher_model_path),
                        "student_features": list(args.student_features),
                        "args": sanitize_args(args),
                    },
                    best_path,
                )
            print(
                f"Fold {fold_idx} student hard epoch {epoch:03d}: "
                f"train PCC {train_metrics['corr']:.4f}, val PCC {val_metrics['corr']:.4f}, "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )

    checkpoint = torch.load(best_path, map_location=device)
    student.load_state_dict(checkpoint["model_state_dict"])
    base_test_metrics = run_kd_epoch(
        student, teacher, test_loader, criterion, device, None, scaler, use_amp, f"Fold {fold_idx} StudentKDTest"
    )
    (student_dir / "student_base_test_metrics.json").write_text(json.dumps(base_test_metrics, indent=2), encoding="utf-8")
    return best_path, base_test_metrics


def set_student_trainable(model, mode):
    for param in model.parameters():
        param.requires_grad = mode == "full"
    if mode == "full":
        return
    if mode == "decoder_head":
        prefixes = ("decoder", "final_conv", "refiner")
    elif mode == "head_only":
        prefixes = ("final_conv", "refiner")
    else:
        raise ValueError(mode)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)


def adapt_student_subject(args, subjects, arrays, student_base_path, teacher_base_path, sample_id, support_segments, device):
    student_checkpoint = torch.load(student_base_path, map_location=device)
    stats = student_checkpoint["normalization_stats"]
    support, query = choose_support(
        subjects, arrays, sample_id, support_segments, args.min_query_segments, args.seed, args.support_strategy
    )
    support_train, support_val = split_support(
        support, args.seed + sample_id + support_segments, args.support_val_ratio, args.support_val_segments
    )

    query_set = MetricSegmentDataset(subjects, arrays, query, stats, augment=False)
    support_train_set = SegmentDataset(subjects, arrays, support_train, stats, augment=args.adapt_augment)
    support_val_set = SegmentDataset(subjects, arrays, support_val, stats, augment=False)
    metric_support_train_set = MetricSegmentDataset(subjects, arrays, support_train, stats, augment=False)
    metric_support_val_set = MetricSegmentDataset(subjects, arrays, support_val, stats, augment=False)

    student = make_student(args, device)
    student.load_state_dict(student_checkpoint["model_state_dict"])
    teacher, _teacher_checkpoint = load_teacher(teacher_base_path, args, device)
    kd_criterion = StudentDistillationLoss(
        hard_weight=args.adapt_kd_hard_weight,
        soft_weight=args.adapt_kd_soft_weight,
        corr_weight=args.adapt_kd_corr_weight,
        temperature=args.kd_temperature,
        teacher_gate_min=args.teacher_gate_min,
        teacher_gate_max=args.teacher_gate_max,
    ).to(device)
    hard_criterion = HeartSoundLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp

    before = evaluate_full_metrics(student, query_set, stats, args, device, f"S{sample_id} before")
    support_before = evaluate_full_metrics(
        student, metric_support_val_set, stats, args, device, f"S{sample_id} support-before"
    )

    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    best_val = -float("inf")
    history = []
    for stage_name, epochs, lr in [
        ("decoder_head", args.stage1_epochs, args.stage1_lr),
        ("full", args.stage2_epochs, args.stage2_lr),
    ]:
        if epochs <= 0:
            continue
        set_student_trainable(student, stage_name)
        trainable = [param for param in student.parameters() if param.requires_grad]
        if not trainable:
            continue
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.adapt_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.adapt_lr_patience, min_lr=args.adapt_min_lr
        )
        train_loader = make_loader(support_train_set, args.adapt_batch_size, True, device, args.seed + sample_id, args.num_workers)
        val_loader = make_loader(support_val_set, args.adapt_batch_size, False, device, args.seed + sample_id, args.num_workers)
        for epoch in range(1, epochs + 1):
            progress = 0.0 if epochs <= 1 else (epoch - 1) / (epochs - 1)
            kd_scale = args.adapt_kd_final_scale + (1.0 - args.adapt_kd_final_scale) * (0.5 + 0.5 * math.cos(math.pi * progress))
            tr = run_kd_epoch(
                student,
                teacher,
                train_loader,
                kd_criterion,
                device,
                optimizer,
                scaler,
                use_amp,
                f"S{sample_id} {stage_name} KD {epoch}/{epochs}",
                kd_scale=kd_scale,
            )
            va = run_epoch(
                student,
                val_loader,
                hard_criterion,
                device,
                None,
                scaler,
                use_amp,
                f"S{sample_id} support-val",
            )
            scheduler.step(va["corr"])
            history.append(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "train_corr": tr["corr"],
                    "train_loss": tr["loss"],
                    "train_kd_soft": tr["soft"],
                    "teacher_target_corr": tr["teacher_target_corr"],
                    "teacher_gate": tr["teacher_gate"],
                    "kd_scale": tr["kd_scale"],
                    "val_corr": va["corr"],
                    "val_rmse": va["rmse"],
                    "val_mae": va["mae"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            if va["corr"] > best_val + args.min_delta:
                best_val = va["corr"]
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    if args.calibration_epochs > 0:
        student.load_state_dict(best_state)
        set_student_trainable(student, args.calibration_mode)
        trainable = [param for param in student.parameters() if param.requires_grad]
        if trainable:
            optimizer = torch.optim.AdamW(trainable, lr=args.calibration_lr, weight_decay=args.adapt_weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=args.adapt_lr_patience, min_lr=args.adapt_min_lr
            )
            train_loader = make_loader(
                support_train_set, args.adapt_batch_size, True, device, args.seed + sample_id + 1009, args.num_workers
            )
            val_loader = make_loader(
                support_val_set, args.adapt_batch_size, False, device, args.seed + sample_id + 1009, args.num_workers
            )
            for epoch in range(1, args.calibration_epochs + 1):
                tr = run_epoch(
                    student,
                    train_loader,
                    hard_criterion,
                    device,
                    optimizer,
                    scaler,
                    use_amp,
                    f"S{sample_id} calibration {epoch}/{args.calibration_epochs}",
                )
                va = run_epoch(
                    student, val_loader, hard_criterion, device, None, scaler, use_amp, f"S{sample_id} calibration-val"
                )
                scheduler.step(va["corr"])
                history.append(
                    {
                        "stage": f"calibration_{args.calibration_mode}",
                        "epoch": epoch,
                        "train_corr": tr["corr"],
                        "train_loss": tr["loss"],
                        "val_corr": va["corr"],
                        "val_rmse": va["rmse"],
                        "val_mae": va["mae"],
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
                if va["corr"] > best_val + args.min_delta:
                    best_val = va["corr"]
                    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    after = evaluate_full_metrics(student, query_set, stats, args, device, f"S{sample_id} after")
    support_after = evaluate_full_metrics(student, metric_support_train_set, stats, args, device, f"S{sample_id} support-after")
    return {
        "sample_id": sample_id,
        "support_segments": len(support),
        "query_segments": len(query),
        "support_seconds": len(support) * args.segment_seconds,
        "before": before,
        "after": after,
        "delta": metric_delta(before, after),
        "support_before": support_before,
        "support_after": support_after,
        "best_support_val_corr": best_val,
        "history": history,
    }, best_state


def metric_delta(before: dict, after: dict) -> dict:
    out = {}
    for key in METRIC_KEYS:
        b = safe_float(before.get(key))
        a = safe_float(after.get(key))
        out[key] = None if a is None or b is None else a - b
    return out


def aggregate_full(rows):
    out = {"n_subjects": len(rows)}
    for phase in ["before", "after", "delta"]:
        for key in METRIC_KEYS:
            values = [safe_float(row.get(phase, {}).get(key)) for row in rows]
            out[f"{phase}_{key}_mean"] = safe_mean(values)
            out[f"{phase}_{key}_std"] = safe_std(values)
    return out


def fmt_metric(value):
    return "nan" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def parse_args():
    parser = argparse.ArgumentParser(description="KD student continual subject-level 5-fold few-shot adaptation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--teacher-results-dir", type=Path, default=None)
    parser.add_argument("--reuse-teacher-results", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--support-list", type=str, default="48,72,96")
    parser.add_argument(
        "--support-strategy",
        choices=["random", "top_quality", "stratified_quality", "hybrid_quality"],
        default="stratified_quality",
    )
    parser.add_argument("--base-train-ratio", type=float, default=0.82)
    parser.add_argument("--base-val-ratio", type=float, default=0.09)
    parser.add_argument("--base-epochs", type=int, default=180)
    parser.add_argument("--base-min-epochs", type=int, default=40)
    parser.add_argument("--base-early-stop", type=int, default=35)
    parser.add_argument("--student-base-epochs", type=int, default=160)
    parser.add_argument("--student-base-min-epochs", type=int, default=35)
    parser.add_argument("--student-base-early-stop", type=int, default=30)
    parser.add_argument("--student-hard-finetune-epochs", type=int, default=20)
    parser.add_argument("--student-hard-finetune-lr", type=float, default=8e-5)
    parser.add_argument("--support-val-ratio", type=float, default=0.25)
    parser.add_argument("--support-val-segments", type=int, default=8)
    parser.add_argument("--min-query-segments", type=int, default=20)
    parser.add_argument("--segment-seconds", type=float, default=2.5)
    parser.add_argument("--sample-rate", type=int, default=1000)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-epochs", type=int, default=70)
    parser.add_argument("--stage2-lr", type=float, default=8e-5)
    parser.add_argument("--calibration-epochs", type=int, default=20)
    parser.add_argument("--calibration-lr", type=float, default=3e-5)
    parser.add_argument("--calibration-mode", choices=["head_only", "decoder_head", "full"], default="full")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--adapt-batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--student-features", type=parse_features, default=(24, 48, 96, 192))
    parser.add_argument("--student-dropout", type=float, default=0.1)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--student-base-lr", type=float, default=4e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--adapt-weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--adapt-lr-patience", type=int, default=8)
    parser.add_argument("--adapt-min-lr", type=float, default=5e-6)
    parser.add_argument("--min-delta", type=float, default=0.0002)
    parser.add_argument("--stats-segments", type=int, default=4096)
    parser.add_argument("--kd-temperature", type=float, default=3.0)
    parser.add_argument("--kd-hard-weight", type=float, default=0.55)
    parser.add_argument("--kd-soft-weight", type=float, default=0.30)
    parser.add_argument("--kd-corr-weight", type=float, default=0.15)
    parser.add_argument("--adapt-kd-hard-weight", type=float, default=0.65)
    parser.add_argument("--adapt-kd-soft-weight", type=float, default=0.25)
    parser.add_argument("--adapt-kd-corr-weight", type=float, default=0.10)
    parser.add_argument("--adapt-kd-final-scale", type=float, default=0.15)
    parser.add_argument("--teacher-gate-min", type=float, default=0.15)
    parser.add_argument("--teacher-gate-max", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapt-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-adapted-models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def resolve_paths(args):
    for key in ["data_dir", "output_dir", "teacher_results_dir", "reuse_teacher_results"]:
        value = getattr(args, key)
        if value is None:
            continue
        setattr(args, key, (PROJECT_ROOT / value).resolve() if not value.is_absolute() else value.resolve())
    if args.teacher_results_dir is None:
        args.teacher_results_dir = args.output_dir / "teacher"


def teacher_path_for_fold(args, fold_idx):
    source = args.reuse_teacher_results if args.reuse_teacher_results is not None else args.teacher_results_dir
    return source / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"


def main():
    args = parse_args()
    resolve_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.teacher_results_dir.mkdir(parents=True, exist_ok=True)
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

    max_folds = args.folds if args.max_folds <= 0 else min(args.max_folds, args.folds)
    all_by_support = {support: [] for support in support_list}
    fold_summaries = []

    for fold_idx in range(max_folds):
        test_ids = sorted(folds[fold_idx])
        base_ids = set(all_ids) - set(test_ids)
        fold_dir = args.output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n========== KD student fold {fold_idx}: test {test_ids} ==========")

        teacher_model_path = teacher_path_for_fold(args, fold_idx)
        if args.reuse_teacher_results is None:
            teacher_args = deepcopy(args)
            teacher_args.output_dir = args.teacher_results_dir
            teacher_model_path, teacher_base_test_metrics = train_base_model(
                teacher_args, subjects, arrays, base_ids, fold_idx, device
            )
        else:
            teacher_base_test_metrics = None
            print(f"Reusing teacher model: {teacher_model_path}")
        if not teacher_model_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_model_path}")

        student_base_path, student_base_test_metrics = train_student_base_model(
            args, subjects, arrays, base_ids, fold_idx, teacher_model_path, device
        )

        fold_summary = {
            "fold": fold_idx,
            "test_ids": test_ids,
            "base_ids": sorted(base_ids),
            "teacher_model_path": str(teacher_model_path),
            "student_base_path": str(student_base_path),
            "teacher_base_test_metrics": teacher_base_test_metrics,
            "student_base_test_metrics": student_base_test_metrics,
            "support_summaries": [],
        }
        for support_segments in support_list:
            rows = []
            support_dir = fold_dir / f"support_{support_segments:03d}"
            support_dir.mkdir(parents=True, exist_ok=True)
            for sample_id in test_ids:
                row, adapted_state = adapt_student_subject(
                    args,
                    subjects,
                    arrays,
                    student_base_path,
                    teacher_model_path,
                    sample_id,
                    support_segments,
                    device,
                )
                light = {key: value for key, value in row.items() if key != "history"}
                light["fold"] = fold_idx
                sample_dir = support_dir / f"sample{sample_id:02d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                (sample_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                if args.save_adapted_models:
                    torch.save(
                        {
                            "model_state_dict": adapted_state,
                            "normalization_stats": torch.load(student_base_path, map_location="cpu")["normalization_stats"],
                            "student_base_path": str(student_base_path),
                            "teacher_model_path": str(teacher_model_path),
                            "sample_id": sample_id,
                            "support_segments": support_segments,
                        },
                        sample_dir / "best_adapted_student_model.pth",
                    )
                rows.append(light)
                all_by_support[support_segments].append(light)

            support_summary = {
                "fold": fold_idx,
                "support_segments": support_segments,
                "aggregate": aggregate_full(rows),
                "per_subject": rows,
            }
            (support_dir / "summary.json").write_text(json.dumps(support_summary, indent=2), encoding="utf-8")
            fold_summary["support_summaries"].append(
                {"support_segments": support_segments, "aggregate": support_summary["aggregate"]}
            )
            agg = support_summary["aggregate"]
            print(
                f"Fold {fold_idx} support {support_segments}: "
                f"after PCC {fmt_metric(agg['after_pcc_mean'])}, delta PCC {fmt_metric(agg['delta_pcc_mean'])}, "
                f"after MAE {fmt_metric(agg['after_mae_mean'])}, RMSE {fmt_metric(agg['after_rmse_mean'])}"
            )
        fold_summaries.append(fold_summary)

    overall = []
    for support_segments, rows in all_by_support.items():
        overall.append(
            {
                "support_segments": support_segments,
                "support_seconds": support_segments * args.segment_seconds,
                "aggregate": aggregate_full(rows),
                "per_subject": rows,
            }
        )

    summary = {"fold_test_ids": folds[:max_folds], "fold_summaries": fold_summaries, "overall_support_summaries": overall}
    summary_path = args.output_dir / "kd_student_subject5fold_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n========== KD student overall summary ==========")
    for item in overall:
        agg = item["aggregate"]
        print(
            f"Support {item['support_segments']} ({item['support_seconds']:.1f}s): "
            f"n={agg['n_subjects']}, PCC {fmt_metric(agg['after_pcc_mean'])} +/- {fmt_metric(agg['after_pcc_std'])}, "
            f"MAE {fmt_metric(agg['after_mae_mean'])}, RMSE {fmt_metric(agg['after_rmse_mean'])}, "
            f"SC {fmt_metric(agg['after_spectral_convergence_mean'])}, "
            f"EnvPCC {fmt_metric(agg['after_envelope_pcc_mean'])}, "
            f"HRBias {fmt_metric(agg['after_mean_hr_bias_mean'])}"
        )
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

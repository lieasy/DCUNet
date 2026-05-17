import argparse
import ast
import json
import math
import random
import re
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "process_data" / "merged_htzx_lez_2.5s"
DEFAULT_BASE_RESULTS = PROJECT_ROOT / "results" / "continual_subject5fold_adaptation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "continual_subject5fold_adaptation_refined"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def read_npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(path)
        major = f.read(1)[0]
        f.read(1)
        header_len = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        header = f.read(header_len).decode("latin1")
    return tuple(ast.literal_eval(header)["shape"])


def discover_subjects(data_dir: Path) -> list[dict]:
    rows = []
    for file_idx, path in enumerate(sorted(data_dir.glob("sample*_2.5s_filter_*.npy"), key=natural_key)):
        match = re.search(r"sample(\d+)_", path.name)
        if not match:
            continue
        shape = read_npy_shape(path)
        rows.append(
            {
                "file_idx": file_idx,
                "sample_id": int(match.group(1)),
                "segments": int(shape[0]),
                "file": str(path.resolve()),
            }
        )
    return rows


def corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False) - float(np.mean(a))
    b = b.astype(np.float64, copy=False) - float(np.mean(b))
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))


def segment_quality(wave: np.ndarray) -> float:
    target, ch2, ch3 = wave[0], wave[1], wave[2]
    c2 = abs(corr_1d(ch2, target))
    c3 = abs(corr_1d(ch3, target))
    energy_ratio = min(np.std(target), 0.5 * (np.std(ch2) + np.std(ch3))) / (
        max(np.std(target), 0.5 * (np.std(ch2) + np.std(ch3))) + 1e-8
    )
    return max(c2, c3, 0.5 * (c2 + c3)) * (0.5 + 0.5 * float(energy_ratio))


def choose_support(subjects, arrays, sample_id: int, support_segments: int, min_query_segments: int, seed: int, strategy: str):
    subject = next(s for s in subjects if int(s["sample_id"]) == sample_id)
    file_idx = int(subject["file_idx"])
    n = int(subject["segments"])
    max_support = max(1, n - min_query_segments)
    k = min(support_segments, max_support)
    rng = np.random.default_rng(seed + sample_id + support_segments)

    if strategy == "random":
        chosen = np.arange(n)
        rng.shuffle(chosen)
        chosen = chosen[:k].tolist()
    else:
        arr = arrays[file_idx]
        scores = np.array([segment_quality(np.asarray(arr[i], dtype=np.float32)) for i in range(n)])
        if strategy == "top_quality":
            chosen = np.argsort(scores)[::-1][:k].tolist()
        elif strategy == "stratified_quality":
            chosen = []
            for bin_indices in np.array_split(np.arange(n), k):
                chosen.append(int(bin_indices[np.argmax(scores[bin_indices])]))
        elif strategy == "hybrid_quality":
            n_top = max(1, int(round(k * 0.7)))
            top = np.argsort(scores)[::-1][:n_top].tolist()
            rest = [i for i in range(n) if i not in set(top)]
            rng.shuffle(rest)
            chosen = top + rest[: k - len(top)]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    support = [(file_idx, int(i)) for i in chosen[:k]]
    support_set = set(support)
    query = [(file_idx, i) for i in range(n) if (file_idx, i) not in support_set]
    return support, query


def split_support(support, seed: int, val_ratio: float, min_val: int):
    items = list(support)
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    if len(items) <= min_val + 2:
        return items, items
    n_val = min(max(min_val, int(round(len(items) * val_ratio))), len(items) - 2)
    return items[n_val:], items[:n_val]


class SegmentDataset(Dataset):
    def __init__(self, subjects, arrays, indices, stats, augment=False):
        self.subjects = subjects
        self.arrays = arrays
        self.indices = list(indices)
        self.stats = stats
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, segment_idx = self.indices[idx]
        wave = np.asarray(self.arrays[file_idx][segment_idx], dtype=np.float32)
        sample_id = int(self.subjects[file_idx]["sample_id"])
        target = self.norm(wave[0], self.stats["target_mean"], self.stats["target_std"])
        ch2 = self.norm(wave[1], self.stats["ch2_mean"], self.stats["ch2_std"])
        ch3 = self.norm(wave[2], self.stats["ch3_mean"], self.stats["ch3_std"])
        if self.augment:
            ch2, ch3, target = self.augment_wave(ch2, ch3, target)
        return (
            torch.from_numpy(np.stack([ch2, ch3]).astype(np.float32)),
            torch.from_numpy(target[None, :].astype(np.float32)),
            sample_id,
        )

    @staticmethod
    def norm(x, mean, std):
        local_mean = np.median(x)
        local_std = np.median(np.abs(x - local_mean)) * 1.4826 + 1e-8
        return 0.7 * ((x - local_mean) / local_std) + 0.3 * ((x - mean) / std)

    @staticmethod
    def augment_wave(ch2, ch3, target):
        if random.random() < 0.35:
            scale = random.uniform(0.85, 1.15)
            ch2, ch3, target = ch2 * scale, ch3 * scale, target * scale
        if random.random() < 0.35:
            noise = random.uniform(0.003, 0.015)
            ch2 = ch2 + np.random.randn(*ch2.shape).astype(np.float32) * noise
            ch3 = ch3 + np.random.randn(*ch3.shape).astype(np.float32) * noise
        return ch2, ch3, target


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 7, padding=3),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + self.shortcut(x))


class MultiScaleStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        q = out_channels // 4
        self.conv21 = nn.Conv1d(in_channels, q, 21, padding=10)
        self.conv41 = nn.Conv1d(in_channels, q, 41, padding=20)
        self.conv81 = nn.Conv1d(in_channels, q, 81, padding=40)
        self.conv1 = nn.Conv1d(in_channels, out_channels - 3 * q, 1)

    def forward(self, x):
        return torch.cat(
            [
                F.leaky_relu(self.conv21(x), 0.1),
                F.leaky_relu(self.conv41(x), 0.1),
                F.leaky_relu(self.conv81(x), 0.1),
                F.leaky_relu(self.conv1(x), 0.1),
            ],
            dim=1,
        )


class HeartSoundUNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stem = nn.Sequential(MultiScaleStem(2, c1), ResidualBlock(c1, c1))
        self.pool = nn.MaxPool1d(2)
        self.enc1 = ResidualBlock(c1, c2)
        self.enc2 = ResidualBlock(c2, c3)
        self.enc3 = ResidualBlock(c3, c4)
        self.bottleneck = nn.Sequential(ResidualBlock(c4, c4 * 2, 0.2), ResidualBlock(c4 * 2, c4 * 2, 0.2))
        self.dec3 = ResidualBlock(c4 * 2 + c4, c4)
        self.dec2 = ResidualBlock(c4 + c3, c3)
        self.dec1 = ResidualBlock(c3 + c2, c2)
        self.dec0 = ResidualBlock(c2 + c1, c1)
        self.head = nn.Sequential(ResidualBlock(c1, c1), nn.Conv1d(c1, 1, 1))

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


class HeartSoundLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    @staticmethod
    def corr_loss(pred, target):
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        corr = (pred * target).sum(dim=-1) / torch.sqrt(
            (pred.square().sum(dim=-1) * target.square().sum(dim=-1)).clamp_min(1e-8)
        )
        return 1.0 - corr.mean()

    def forward(self, pred, target):
        deriv = F.l1_loss(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1])
        return 0.25 * self.l1(pred, target) + 0.10 * self.mse(pred, target) + 0.50 * self.corr_loss(pred, target) + 0.15 * deriv


def make_loader(dataset, batch_size, shuffle, device, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle and len(dataset) >= batch_size,
        generator=generator,
    )


def batch_metrics(pred, target):
    pc = pred - pred.mean(dim=-1, keepdim=True)
    tc = target - target.mean(dim=-1, keepdim=True)
    corr = (pc * tc).sum(dim=-1).squeeze(1) / torch.sqrt(
        (pc.square().sum(dim=-1).squeeze(1) * tc.square().sum(dim=-1).squeeze(1)).clamp_min(1e-8)
    )
    mse = (pred - target).square().mean(dim=(1, 2))
    mae = (pred - target).abs().mean(dim=(1, 2))
    return corr, torch.sqrt(mse), mae


def run_epoch(model, loader, criterion, device, optimizer, scaler, use_amp, desc):
    train = optimizer is not None
    model.train(train)
    vals = {"loss": [], "corr": [], "rmse": [], "mae": []}
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y, _sid in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            corr, rmse, mae = batch_metrics(pred.detach(), y)
            n = x.shape[0]
            vals["loss"].extend([float(loss.item())] * n)
            vals["corr"].extend(corr.cpu().numpy().astype(float).tolist())
            vals["rmse"].extend(rmse.cpu().numpy().astype(float).tolist())
            vals["mae"].extend(mae.cpu().numpy().astype(float).tolist())
    return {k: float(np.mean(v)) for k, v in vals.items()}


def set_trainable(model, mode):
    for p in model.parameters():
        p.requires_grad = True
    if mode == "full":
        return
    for name, param in model.named_parameters():
        if mode == "decoder_head":
            param.requires_grad = name.startswith("dec") or name.startswith("head")
        elif mode == "head_only":
            param.requires_grad = name.startswith("head")
        else:
            raise ValueError(mode)


def adapt_subject(args, subjects, arrays, base_model_path, sample_id, support_segments, device):
    checkpoint = torch.load(base_model_path, map_location=device)
    stats = checkpoint["normalization_stats"]
    support, query = choose_support(
        subjects, arrays, sample_id, support_segments, args.min_query_segments, args.seed, args.support_strategy
    )
    support_train, support_val = split_support(support, args.seed + sample_id + support_segments, args.support_val_ratio, args.support_val_segments)

    query_set = SegmentDataset(subjects, arrays, query, stats, augment=False)
    support_train_set = SegmentDataset(subjects, arrays, support_train, stats, augment=args.adapt_augment)
    support_val_set = SegmentDataset(subjects, arrays, support_val, stats, augment=False)

    model = HeartSoundUNet(args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = HeartSoundLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp
    before = run_epoch(model, make_loader(query_set, args.batch_size, False, device, args.seed, args.num_workers), criterion, device, None, scaler, use_amp, f"S{sample_id} before")

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
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=args.adapt_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.adapt_lr_patience, min_lr=args.adapt_min_lr)
        train_loader = make_loader(support_train_set, args.adapt_batch_size, True, device, args.seed + sample_id, args.num_workers)
        val_loader = make_loader(support_val_set, args.adapt_batch_size, False, device, args.seed + sample_id, args.num_workers)
        for epoch in range(1, epochs + 1):
            tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp, f"S{sample_id} {stage_name} {epoch}/{epochs}")
            va = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"S{sample_id} support-val")
            scheduler.step(va["corr"])
            history.append({"stage": stage_name, "epoch": epoch, "train_corr": tr["corr"], "val_corr": va["corr"], "lr": optimizer.param_groups[0]["lr"]})
            if va["corr"] > best_val + args.min_delta:
                best_val = va["corr"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    after = run_epoch(model, make_loader(query_set, args.batch_size, False, device, args.seed, args.num_workers), criterion, device, None, scaler, use_amp, f"S{sample_id} after")
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
        "history": history,
    }


def aggregate(rows):
    out = {"n_subjects": len(rows)}
    for key in ["before_corr", "after_corr", "delta_corr", "before_rmse", "after_rmse", "before_mae", "after_mae"]:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Refine continual subject 5-fold adaptation using existing base models.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--base-results-dir", type=Path, default=DEFAULT_BASE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--support-list", type=str, default="72,96,120")
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapt-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = (PROJECT_ROOT / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir.resolve()
    args.base_results_dir = (PROJECT_ROOT / args.base_results_dir).resolve() if not args.base_results_dir.is_absolute() else args.base_results_dir.resolve()
    args.output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    subjects = discover_subjects(args.data_dir)
    arrays = [np.load(s["file"], mmap_mode="r") for s in subjects]
    run_config = json.loads((args.base_results_dir / "run_config.json").read_text(encoding="utf-8"))
    fold_test_ids = run_config["fold_test_ids"]
    support_list = [int(x) for x in args.support_list.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    all_by_support = {k: [] for k in support_list}
    max_folds = len(fold_test_ids) if args.max_folds <= 0 else min(args.max_folds, len(fold_test_ids))
    for fold_idx in range(max_folds):
        base_model_path = args.base_results_dir / f"fold_{fold_idx}" / "base_model" / "best_base_model.pth"
        test_ids = sorted(int(x) for x in fold_test_ids[fold_idx])
        print(f"\n========== Refine fold {fold_idx}: test {test_ids} ==========")
        for support_segments in support_list:
            rows = []
            out_dir = args.output_dir / f"fold_{fold_idx}" / f"support_{support_segments:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for sid in test_ids:
                row = adapt_subject(args, subjects, arrays, base_model_path, sid, support_segments, device)
                row_light = {k: v for k, v in row.items() if k != "history"}
                (out_dir / f"sample{sid:02d}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                rows.append(row_light)
                all_by_support[support_segments].append({**row_light, "fold": fold_idx})
            summary = {"fold": fold_idx, "support_segments": support_segments, "aggregate": aggregate(rows), "per_subject": rows}
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Fold {fold_idx} support {support_segments}: after {summary['aggregate']['after_corr_mean']:.4f}, delta {summary['aggregate']['delta_corr_mean']:.4f}")

    overall = []
    for support_segments, rows in all_by_support.items():
        overall.append({"support_segments": support_segments, "support_seconds": support_segments * args.segment_seconds, "aggregate": aggregate(rows), "per_subject": rows})
    final = {"source_base_results": str(args.base_results_dir), "support_strategy": args.support_strategy, "overall_support_summaries": overall}
    (args.output_dir / "refined_subject5fold_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\n========== Refined overall summary ==========")
    for item in overall:
        agg = item["aggregate"]
        print(f"Support {item['support_segments']}: n={agg['n_subjects']}, after {agg['after_corr_mean']:.4f} +/- {agg['after_corr_std']:.4f}, delta {agg['delta_corr_mean']:.4f}")
    print(f"Saved summary to: {args.output_dir / 'refined_subject5fold_summary.json'}")


if __name__ == "__main__":
    main()

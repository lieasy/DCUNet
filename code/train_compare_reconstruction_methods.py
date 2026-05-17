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
    SegmentDataset,
    aggregate,
    choose_support,
    discover_subjects,
    make_loader,
    run_epoch,
    set_seed,
    split_support,
)
from train_continual_subject5fold_adaptation_optimized import (
    build_balanced_subject_folds,
    compute_stats,
    sanitize_args,
    split_base_indices,
)
from train_segment_mixed_5fold_recovery import (
    build_all_segment_indices,
    build_segment_folds,
    split_train_val,
    summarize_subject_overlap,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "compare_reconstruction_methods"
MODEL_NAMES = [
    "biosignal_gan",
    "p2e_wgan",
    "cardiogan",
    "ppg2ecg",
    "rddm",
]

DETAILED_METRIC_KEYS = [
    "corr",
    "mse",
    "mae",
    "rmse",
    "spectral_convergence",
    "envelope_corr",
    "envelope_mse",
    "envelope_mae",
    "envelope_rmse",
    "s1_corr",
    "s1_mse",
    "s1_mae",
    "s1_rmse",
    "s1_count",
    "s2_corr",
    "s2_mse",
    "s2_mae",
    "s2_rmse",
    "s2_count",
    "valid_hr_count",
    "total_hr_count",
    "mean_hr_bias",
    "std_hr_bias",
]


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, norm="batch", dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
        ]
        if norm == "batch":
            layers.append(nn.BatchNorm1d(out_channels))
        elif norm == "instance":
            layers.append(nn.InstanceNorm1d(out_channels, affine=True))
        elif norm == "group":
            layers.append(nn.GroupNorm(1, out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
        ]
        if norm == "batch":
            layers.append(nn.BatchNorm1d(out_channels))
        elif norm == "instance":
            layers.append(nn.InstanceNorm1d(out_channels, affine=True))
        elif norm == "group":
            layers.append(nn.GroupNorm(1, out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Pix2Pix1DUNet(nn.Module):
    """P2E-WGAN / biosignalGANs style 1D pix2pix generator adapted to 2 input channels."""

    def __init__(self, in_channels=2, out_channels=1, base_channels=64, norm="instance"):
        super().__init__()
        c = base_channels
        self.down1 = ConvBlock(in_channels, c, 9, norm="none")
        self.down2 = ConvBlock(c, c * 2, 7, norm=norm)
        self.down3 = ConvBlock(c * 2, c * 4, 7, norm=norm, dropout=0.25)
        self.down4 = ConvBlock(c * 4, c * 8, 5, norm=norm, dropout=0.35)
        self.pool = nn.AvgPool1d(2)
        self.bottleneck = ConvBlock(c * 8, c * 8, 5, norm=norm, dropout=0.5)
        self.up3 = ConvBlock(c * 8 + c * 8, c * 4, 5, norm=norm)
        self.up2 = ConvBlock(c * 4 + c * 4, c * 2, 7, norm=norm)
        self.up1 = ConvBlock(c * 2 + c * 2, c, 7, norm=norm)
        self.up0 = ConvBlock(c + c, c, 9, norm=norm)
        self.final = nn.Conv1d(c, out_channels, 1)

    def forward(self, x):
        target_len = x.shape[-1]
        s0 = self.down1(x)
        s1 = self.down2(self.pool(s0))
        s2 = self.down3(self.pool(s1))
        s3 = self.down4(self.pool(s2))
        z = self.bottleneck(self.pool(s3))
        z = self.up3(torch.cat([F.interpolate(z, size=s3.shape[-1], mode="linear", align_corners=False), s3], dim=1))
        z = self.up2(torch.cat([F.interpolate(z, size=s2.shape[-1], mode="linear", align_corners=False), s2], dim=1))
        z = self.up1(torch.cat([F.interpolate(z, size=s1.shape[-1], mode="linear", align_corners=False), s1], dim=1))
        z = self.up0(torch.cat([F.interpolate(z, size=s0.shape[-1], mode="linear", align_corners=False), s0], dim=1))
        return self.final(z)[..., :target_len]


class PPG2ECGConvNet(nn.Module):
    """ppg2ecg-pytorch convolutional autoencoder adapted from 1-channel PPG to 2-channel heart-sound input."""

    def __init__(self, in_channels=2, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, c, 31, stride=2, padding=15),
            nn.PReLU(c),
            nn.Conv1d(c, c * 2, 31, padding=15),
            nn.PReLU(c * 2),
            nn.Conv1d(c * 2, c * 4, 31, stride=2, padding=15),
            nn.PReLU(c * 4),
            nn.Conv1d(c * 4, c * 8, 31, padding=15),
            nn.PReLU(c * 8),
            nn.Conv1d(c * 8, c * 16, 31, stride=2, padding=15),
            nn.PReLU(c * 16),
            nn.ConvTranspose1d(c * 16, c * 8, 31, stride=2, padding=15, output_padding=1),
            nn.PReLU(c * 8),
            nn.ConvTranspose1d(c * 8, c * 4, 31, padding=15),
            nn.PReLU(c * 4),
            nn.ConvTranspose1d(c * 4, c * 2, 31, stride=2, padding=15, output_padding=1),
            nn.PReLU(c * 2),
            nn.ConvTranspose1d(c * 2, c, 31, padding=15),
            nn.PReLU(c),
            nn.ConvTranspose1d(c, out_channels, 31, stride=2, padding=15, output_padding=1),
        )

    def forward(self, x):
        y = self.net(x)
        return F.interpolate(y, size=x.shape[-1], mode="linear", align_corners=False)


class AttentionGate1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 2)
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, current, skip):
        current = F.interpolate(current, size=skip.shape[-1], mode="linear", align_corners=False)
        return skip * self.gate(torch.cat([current, skip], dim=1))


class CardioGANAttentionUNet(nn.Module):
    """PyTorch adaptation of CardioGAN's attention U-Net generator."""

    def __init__(self, in_channels=2, out_channels=1, base_channels=48):
        super().__init__()
        c = base_channels
        self.pool = nn.AvgPool1d(2)
        self.enc0 = ConvBlock(in_channels, c, 15, norm="instance")
        self.enc1 = ConvBlock(c, c * 2, 15, norm="instance")
        self.enc2 = ConvBlock(c * 2, c * 4, 15, norm="instance")
        self.enc3 = ConvBlock(c * 4, c * 8, 15, norm="instance", dropout=0.3)
        self.mid = ConvBlock(c * 8, c * 8, 15, norm="instance", dropout=0.4)
        self.g3 = AttentionGate1D(c * 8)
        self.g2 = AttentionGate1D(c * 4)
        self.g1 = AttentionGate1D(c * 2)
        self.g0 = AttentionGate1D(c)
        self.dec3 = ConvBlock(c * 8 + c * 8, c * 4, 15, norm="instance")
        self.dec2 = ConvBlock(c * 4 + c * 4, c * 2, 15, norm="instance")
        self.dec1 = ConvBlock(c * 2 + c * 2, c, 15, norm="instance")
        self.dec0 = ConvBlock(c + c, c, 15, norm="instance")
        self.final = nn.Conv1d(c, out_channels, 1)

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        z = self.mid(self.pool(s3))
        z = self.dec3(torch.cat([F.interpolate(z, size=s3.shape[-1], mode="linear", align_corners=False), self.g3(z, s3)], dim=1))
        z = self.dec2(torch.cat([F.interpolate(z, size=s2.shape[-1], mode="linear", align_corners=False), self.g2(z, s2)], dim=1))
        z = self.dec1(torch.cat([F.interpolate(z, size=s1.shape[-1], mode="linear", align_corners=False), self.g1(z, s1)], dim=1))
        z = self.dec0(torch.cat([F.interpolate(z, size=s0.shape[-1], mode="linear", align_corners=False), self.g0(z, s0)], dim=1))
        return self.final(z)


class SelfAttention1D(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        seq = x.transpose(1, 2)
        attn, _ = self.attn(self.norm1(seq), self.norm1(seq), self.norm1(seq), need_weights=False)
        seq = seq + attn
        seq = seq + self.ff(seq)
        return seq.transpose(1, 2)


class RDDMUNet(nn.Module):
    """RDDM-inspired residual U-Net with self-attention, trained as a direct reconstructor for fair protocol matching."""

    def __init__(self, in_channels=2, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.pool = nn.MaxPool1d(2)
        self.inc = ConvBlock(in_channels, c, 5, norm="group")
        self.down1 = ConvBlock(c, c * 2, 5, norm="group")
        self.down2 = ConvBlock(c * 2, c * 4, 5, norm="group")
        self.down3 = ConvBlock(c * 4, c * 8, 5, norm="group")
        self.attn1 = SelfAttention1D(c * 2)
        self.attn2 = SelfAttention1D(c * 4)
        self.attn3 = SelfAttention1D(c * 8)
        self.mid = ConvBlock(c * 8, c * 8, 5, norm="group")
        self.up2 = ConvBlock(c * 8 + c * 8, c * 4, 5, norm="group")
        self.up1 = ConvBlock(c * 4 + c * 4, c * 2, 5, norm="group")
        self.up0 = ConvBlock(c * 2 + c * 2, c, 5, norm="group")
        self.head = nn.Conv1d(c + c, out_channels, 1)

    def forward(self, x):
        s0 = self.inc(x)
        s1 = self.attn1(self.down1(self.pool(s0)))
        s2 = self.attn2(self.down2(self.pool(s1)))
        s3 = self.attn3(self.down3(self.pool(s2)))
        z = self.mid(s3)
        z = self.up2(torch.cat([F.interpolate(z, size=s3.shape[-1], mode="linear", align_corners=False), s3], dim=1))
        z = self.up1(torch.cat([F.interpolate(z, size=s2.shape[-1], mode="linear", align_corners=False), s2], dim=1))
        z = self.up0(torch.cat([F.interpolate(z, size=s1.shape[-1], mode="linear", align_corners=False), s1], dim=1))
        z = F.interpolate(z, size=s0.shape[-1], mode="linear", align_corners=False)
        return self.head(torch.cat([z, s0], dim=1))


def build_model(name: str, base_channels: int) -> nn.Module:
    name = name.lower()
    if name == "biosignal_gan":
        return Pix2Pix1DUNet(base_channels=base_channels, norm="batch")
    if name == "p2e_wgan":
        return Pix2Pix1DUNet(base_channels=base_channels, norm="instance")
    if name == "cardiogan":
        return CardioGANAttentionUNet(base_channels=max(8, base_channels))
    if name == "ppg2ecg":
        return PPG2ECGConvNet(base_channels=base_channels)
    if name == "rddm":
        return RDDMUNet(base_channels=base_channels)
    raise ValueError(f"Unknown model: {name}. Choices: {MODEL_NAMES}")


def parse_model_list(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return list(MODEL_NAMES)
    names = [x.strip().lower() for x in text.split(",") if x.strip()]
    unknown = sorted(set(names) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Choices: {MODEL_NAMES}")
    return names


def safe_number(value):
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


def safe_mean(values):
    arr = np.array([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else None


def safe_std(values):
    arr = np.array([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=np.float64)
    return float(np.std(arr)) if arr.size else None


def pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.std(a) <= 1e-8 or np.std(b) <= 1e-8:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))


def signal_metrics_np(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mse = float(np.mean((a - b) ** 2))
    return {
        "corr": pearson_np(a, b),
        "mse": mse,
        "mae": float(np.mean(np.abs(a - b))),
        "rmse": float(np.sqrt(mse)),
    }


def hilbert_envelope_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    spectrum = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float64)
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


def simple_find_peaks(x: np.ndarray, height: float, distance: int, prominence: float) -> np.ndarray:
    peaks = []
    last_peak = -distance
    for idx in range(1, len(x) - 1):
        if x[idx] < height or x[idx] <= x[idx - 1] or x[idx] < x[idx + 1]:
            continue
        left = max(0, idx - distance // 2)
        right = min(len(x), idx + distance // 2 + 1)
        local_base = max(float(np.min(x[left : idx + 1])), float(np.min(x[idx:right])))
        if x[idx] - local_base < prominence:
            continue
        if idx - last_peak < distance and peaks:
            if x[idx] > x[peaks[-1]]:
                peaks[-1] = idx
                last_peak = idx
            continue
        peaks.append(idx)
        last_peak = idx
    return np.array(peaks, dtype=np.int64)


def refine_heart_sound_labels(peak_indices, s1_candidates, s2_candidates):
    all_peaks = sorted(int(x) for x in peak_indices)
    labels = []
    for peak in all_peaks:
        if peak in s1_candidates:
            labels.append("S1")
        elif peak in s2_candidates:
            labels.append("S2")
        else:
            labels.append("unknown")
    for idx in range(1, len(labels)):
        if labels[idx] == labels[idx - 1] and labels[idx] != "unknown":
            labels[idx - 1] = "S2" if labels[idx - 1] == "S1" else "S1"
    s1_indices = [all_peaks[idx] for idx, label in enumerate(labels) if label == "S1"]
    s2_indices = [all_peaks[idx] for idx, label in enumerate(labels) if label == "S2"]
    return s1_indices, s2_indices


def detect_heart_sounds(signal: np.ndarray, envelope: np.ndarray, sample_rate: int = 1000):
    del signal
    envelope_norm = envelope / (np.max(envelope) + 1e-8)
    peak_indices = simple_find_peaks(envelope_norm, height=0.2, distance=sample_rate // 4, prominence=0.1)
    if len(peak_indices) < 2:
        return [], []
    peak_values = envelope_norm[peak_indices]
    mean_peak = float(np.mean(peak_values))
    s1_candidates = {int(idx) for idx, val in zip(peak_indices, peak_values) if val > mean_peak * 1.1}
    s2_candidates = {int(idx) for idx, val in zip(peak_indices, peak_values) if mean_peak * 0.5 < val <= mean_peak * 1.1}
    return refine_heart_sound_labels(peak_indices, s1_candidates, s2_candidates)


def segment_heart_sounds(signal: np.ndarray, indices, window: int = 100) -> list[np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    segments = []
    half = window // 2
    for idx in indices:
        start = max(0, int(idx) - half)
        end = min(len(signal), int(idx) + half)
        segment = signal[start:end]
        if len(segment) == window:
            segments.append(segment)
    return segments


def calculate_heart_sound_metrics(target_segments, pred_segments) -> dict:
    n = min(len(target_segments), len(pred_segments))
    if n <= 0:
        return {"corr": None, "mse": None, "mae": None, "rmse": None, "count": 0}
    vals = {"corr": [], "mse": [], "mae": [], "rmse": []}
    for target_segment, pred_segment in zip(target_segments[:n], pred_segments[:n]):
        item = signal_metrics_np(target_segment, pred_segment)
        for key in vals:
            vals[key].append(item[key])
    return {key: safe_mean(value) for key, value in vals.items()} | {"count": n}


def estimate_heart_rate(s1_indices, sample_rate: int = 1000):
    if len(s1_indices) < 2:
        return None
    intervals = np.diff(np.array(s1_indices, dtype=np.float64)) / sample_rate
    std = float(np.std(intervals))
    if std > 1e-8:
        intervals = intervals[np.abs(intervals - np.mean(intervals)) < 2 * std]
    if len(intervals) < 1:
        return None
    avg_interval = float(np.mean(intervals))
    return None if avg_interval <= 1e-8 else 60.0 / avg_interval


def spectral_convergence_batch(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 1024, hop_length: int = 256) -> list[float]:
    pred_2d = pred.squeeze(1)
    target_2d = target.squeeze(1)
    n_fft = min(n_fft, pred_2d.shape[-1])
    window = torch.hann_window(n_fft, device=pred.device)
    pred_spec = torch.stft(pred_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True).abs()
    target_spec = torch.stft(target_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True).abs()
    numerator = torch.linalg.vector_norm(pred_spec - target_spec, dim=(1, 2))
    denominator = torch.linalg.vector_norm(target_spec, dim=(1, 2)).clamp_min(1e-8)
    return (numerator / denominator).detach().cpu().numpy().astype(float).tolist()


class DetailedMetricSegmentDataset(SegmentDataset):
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


def make_detailed_metric_loader(dataset, batch_size, device, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )


def evaluate_detailed_metrics(model, subjects, arrays, indices, stats, args, device, desc: str):
    model.eval()
    dataset = DetailedMetricSegmentDataset(subjects, arrays, indices, stats, augment=False)
    loader = make_detailed_metric_loader(dataset, args.batch_size, device, args.seed, args.num_workers)
    values = {key: [] for key in DETAILED_METRIC_KEYS if key not in {"valid_hr_count", "total_hr_count", "mean_hr_bias", "std_hr_bias"}}
    hr_biases = []
    total_hr_count = 0
    with torch.no_grad():
        for x, _y, raw_target, _sid, params in loader:
            x = x.to(device, non_blocking=True)
            raw_target = raw_target.to(device, non_blocking=True)
            params = params.to(device, non_blocking=True)
            pred = denorm_prediction(model(x), params, stats)
            target = raw_target
            spec_vals = spectral_convergence_batch(pred, target)
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            for item_idx in range(pred_np.shape[0]):
                pred_item = pred_np[item_idx, 0]
                target_item = target_np[item_idx, 0]
                sig = signal_metrics_np(target_item, pred_item)
                values["corr"].append(sig["corr"])
                values["mse"].append(sig["mse"])
                values["mae"].append(sig["mae"])
                values["rmse"].append(sig["rmse"])
                values["spectral_convergence"].append(spec_vals[item_idx])

                target_env = extract_envelope(target_item, args.sample_rate)
                pred_env = extract_envelope(pred_item, args.sample_rate)
                target_env_norm = target_env / (np.max(target_env) + 1e-8)
                pred_env_norm = pred_env / (np.max(pred_env) + 1e-8)
                env = signal_metrics_np(target_env_norm, pred_env_norm)
                values["envelope_corr"].append(env["corr"])
                values["envelope_mse"].append(env["mse"])
                values["envelope_mae"].append(env["mae"])
                values["envelope_rmse"].append(env["rmse"])

                s1_target, s2_target = detect_heart_sounds(target_item, target_env, args.sample_rate)
                s1_pred, s2_pred = detect_heart_sounds(pred_item, pred_env, args.sample_rate)
                s1_metrics = calculate_heart_sound_metrics(
                    segment_heart_sounds(target_item, s1_target, args.heart_sound_window),
                    segment_heart_sounds(pred_item, s1_pred, args.heart_sound_window),
                )
                s2_metrics = calculate_heart_sound_metrics(
                    segment_heart_sounds(target_item, s2_target, args.heart_sound_window),
                    segment_heart_sounds(pred_item, s2_pred, args.heart_sound_window),
                )
                for prefix, item in [("s1", s1_metrics), ("s2", s2_metrics)]:
                    values[f"{prefix}_corr"].append(item["corr"])
                    values[f"{prefix}_mse"].append(item["mse"])
                    values[f"{prefix}_mae"].append(item["mae"])
                    values[f"{prefix}_rmse"].append(item["rmse"])
                    values[f"{prefix}_count"].append(item["count"])

                total_hr_count += 1
                hr_target = estimate_heart_rate(s1_target, args.sample_rate)
                hr_pred = estimate_heart_rate(s1_pred, args.sample_rate)
                if hr_target is not None and hr_pred is not None:
                    hr_biases.append(float(hr_pred - hr_target))

    out = {key: safe_mean(vals) for key, vals in values.items()}
    out["valid_hr_count"] = len(hr_biases)
    out["total_hr_count"] = total_hr_count
    out["mean_hr_bias"] = safe_mean(hr_biases)
    out["std_hr_bias"] = safe_std(hr_biases)
    print(
        f"{desc} detailed: PCC={out['corr'] if out['corr'] is not None else float('nan'):.4f}, "
        f"MSE={out['mse'] if out['mse'] is not None else float('nan'):.6f}, "
        f"Envelope PCC={out['envelope_corr'] if out['envelope_corr'] is not None else float('nan'):.4f}, "
        f"HR valid={out['valid_hr_count']}/{out['total_hr_count']}"
    )
    return out


def flatten_detailed(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": safe_number(metrics.get(key)) for key in DETAILED_METRIC_KEYS}


def aggregate_detailed_rows(rows: list[dict], prefix: str = "detailed") -> dict:
    out = {"n_rows": len(rows)}
    for key in DETAILED_METRIC_KEYS:
        field = f"{prefix}_{key}"
        vals = [row.get(field) for row in rows if row.get(field) is not None]
        out[f"{field}_mean"] = safe_mean(vals)
        out[f"{field}_std"] = safe_std(vals)
    return out


def set_trainable_generic(model: nn.Module, mode: str):
    for param in model.parameters():
        param.requires_grad = True
    if mode == "full":
        return
    keywords = ("dec", "up", "head", "final", "out")
    for name, param in model.named_parameters():
        param.requires_grad = any(key in name for key in keywords)
    if not any(param.requires_grad for param in model.parameters()):
        for param in model.parameters():
            param.requires_grad = True


def train_segment_fold(args, model_name, subjects, arrays, fold_idx, train_indices, val_indices, test_indices, device):
    fold_dir = args.output_dir / "segment_mixed" / model_name / f"fold_{fold_idx}"
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

    train_loader = make_loader(SegmentDataset(subjects, arrays, train_indices, stats, args.augment), args.batch_size, True, device, args.seed + fold_idx, args.num_workers)
    val_loader = make_loader(SegmentDataset(subjects, arrays, val_indices, stats, False), args.batch_size, False, device, args.seed + fold_idx, args.num_workers)
    test_loader = make_loader(SegmentDataset(subjects, arrays, test_indices, stats, False), args.batch_size, False, device, args.seed + fold_idx, args.num_workers)
    model = build_model(model_name, args.base_channels).to(device)
    criterion = HeartSoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience, threshold=args.min_delta, min_lr=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp
    best_val = -float("inf")
    bad_epochs = 0
    history = []
    best_path = fold_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp, f"{model_name} mixed f{fold_idx} train {epoch}/{args.epochs}")
        val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"{model_name} mixed f{fold_idx} val")
        scheduler.step(val_metrics["corr"])
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        (fold_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_metrics["corr"] > best_val + args.min_delta:
            best_val = val_metrics["corr"]
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "normalization_stats": stats, "best_val_corr": best_val, "model_name": model_name, "args": sanitize_args(args)}, best_path)
        else:
            bad_epochs += 1
        print(f"{model_name} mixed fold {fold_idx} epoch {epoch:03d}: train PCC {train_metrics['corr']:.4f}, val PCC {val_metrics['corr']:.4f}")
        if args.early_stop > 0 and epoch >= args.min_epochs and bad_epochs >= args.early_stop:
            break
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device, None, scaler, use_amp, f"{model_name} mixed f{fold_idx} test")
    detailed_metrics = evaluate_detailed_metrics(
        model,
        subjects,
        arrays,
        test_indices,
        stats,
        args,
        device,
        f"{model_name} mixed f{fold_idx} test",
    )
    summary = {
        "fold": fold_idx,
        "model_name": model_name,
        "best_val_corr": float(checkpoint["best_val_corr"]),
        "test_metrics": test_metrics,
        "detailed_test_metrics": detailed_metrics,
        **flatten_detailed("detailed", detailed_metrics),
        "split_info": split_info,
        "best_model": str(best_path),
    }
    (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_base_model_generic(args, model_name, subjects, arrays, base_ids, fold_idx, device):
    base_dir = args.output_dir / "continual" / model_name / f"fold_{fold_idx}" / "base_model"
    base_dir.mkdir(parents=True, exist_ok=True)
    best_path = base_dir / "best_base_model.pth"
    metrics_path = base_dir / "base_test_metrics.json"
    if args.reuse_existing_base and best_path.exists() and metrics_path.exists():
        print(f"{model_name} base fold {fold_idx}: reusing {best_path}")
        return best_path, load_json(metrics_path, {})
    base_train, base_val, base_test = split_base_indices(subjects, base_ids, args.seed + fold_idx, args.base_train_ratio, args.base_val_ratio)
    stats = compute_stats(arrays, base_train, args.seed + fold_idx, args.stats_segments)
    (base_dir / "normalization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    train_loader = make_loader(SegmentDataset(subjects, arrays, base_train, stats, args.base_augment), args.batch_size, True, device, args.seed + fold_idx, args.num_workers)
    val_loader = make_loader(SegmentDataset(subjects, arrays, base_val, stats, False), args.batch_size, False, device, args.seed + fold_idx, args.num_workers)
    test_loader = make_loader(SegmentDataset(subjects, arrays, base_test, stats, False), args.batch_size, False, device, args.seed + fold_idx, args.num_workers)
    model = build_model(model_name, args.base_channels).to(device)
    criterion = HeartSoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience, threshold=args.min_delta, min_lr=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp
    best_val = -float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.base_epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp, f"{model_name} base f{fold_idx} {epoch}/{args.base_epochs}")
        val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"{model_name} base-val f{fold_idx}")
        scheduler.step(val_metrics["corr"])
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        (base_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val_metrics["corr"] > best_val + args.min_delta:
            best_val = val_metrics["corr"]
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "normalization_stats": stats, "best_val_corr": best_val, "model_name": model_name, "args": sanitize_args(args)}, best_path)
        else:
            bad_epochs += 1
        print(f"{model_name} base fold {fold_idx} epoch {epoch:03d}: train PCC {train_metrics['corr']:.4f}, val PCC {val_metrics['corr']:.4f}")
        if args.base_early_stop > 0 and epoch >= args.base_min_epochs and bad_epochs >= args.base_early_stop:
            break
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device, None, scaler, use_amp, f"{model_name} base-test f{fold_idx}")
    (base_dir / "base_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    return best_path, test_metrics


def adapt_subject_generic(args, model_name, subjects, arrays, base_model_path, sample_id, support_segments, device):
    checkpoint = torch.load(base_model_path, map_location=device)
    stats = checkpoint["normalization_stats"]
    support, query = choose_support(subjects, arrays, sample_id, support_segments, args.min_query_segments, args.seed, args.support_strategy)
    support_train, support_val = split_support(support, args.seed + sample_id + support_segments, args.support_val_ratio, args.support_val_segments)
    criterion = HeartSoundLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    use_amp = device.type == "cuda" and args.amp
    query_loader = make_loader(SegmentDataset(subjects, arrays, query, stats, False), args.batch_size, False, device, args.seed, args.num_workers)
    train_loader = make_loader(SegmentDataset(subjects, arrays, support_train, stats, args.adapt_augment), args.adapt_batch_size, True, device, args.seed + sample_id, args.num_workers)
    val_loader = make_loader(SegmentDataset(subjects, arrays, support_val, stats, False), args.adapt_batch_size, False, device, args.seed + sample_id, args.num_workers)
    model = build_model(model_name, args.base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    before = run_epoch(model, query_loader, criterion, device, None, scaler, use_amp, f"{model_name} S{sample_id} before")
    before_detailed = evaluate_detailed_metrics(
        model,
        subjects,
        arrays,
        query,
        stats,
        args,
        device,
        f"{model_name} S{sample_id} before",
    )
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -float("inf")
    history = []
    for stage_name, epochs, lr in [("decoder_head", args.stage1_epochs, args.stage1_lr), ("full", args.stage2_epochs, args.stage2_lr)]:
        if epochs <= 0:
            continue
        set_trainable_generic(model, stage_name)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=args.adapt_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.adapt_lr_patience, min_lr=args.adapt_min_lr)
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp, f"{model_name} S{sample_id} {stage_name} {epoch}/{epochs}")
            val_metrics = run_epoch(model, val_loader, criterion, device, None, scaler, use_amp, f"{model_name} S{sample_id} support-val")
            scheduler.step(val_metrics["corr"])
            history.append({"stage": stage_name, "epoch": epoch, "train_corr": train_metrics["corr"], "val_corr": val_metrics["corr"], "lr": optimizer.param_groups[0]["lr"]})
            if val_metrics["corr"] > best_val + args.min_delta:
                best_val = val_metrics["corr"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    after = run_epoch(model, query_loader, criterion, device, None, scaler, use_amp, f"{model_name} S{sample_id} after")
    after_detailed = evaluate_detailed_metrics(
        model,
        subjects,
        arrays,
        query,
        stats,
        args,
        device,
        f"{model_name} S{sample_id} after",
    )
    detailed_delta = {}
    for key in DETAILED_METRIC_KEYS:
        before_value = before_detailed.get(key)
        after_value = after_detailed.get(key)
        detailed_delta[key] = None if before_value is None or after_value is None else after_value - before_value
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
        "before_detailed_metrics": before_detailed,
        "after_detailed_metrics": after_detailed,
        "delta_detailed_metrics": detailed_delta,
        **flatten_detailed("before_detailed", before_detailed),
        **flatten_detailed("after_detailed", after_detailed),
        **flatten_detailed("delta_detailed", detailed_delta),
        "history": history,
    }


def aggregate_fold_metrics(rows):
    out = {"n_folds": len(rows)}
    for key in ["loss", "corr", "rmse", "mae"]:
        vals = np.array([row["test_metrics"][key] for row in rows], dtype=np.float64)
        out[f"test_{key}_mean"] = float(np.mean(vals))
        out[f"test_{key}_std"] = float(np.std(vals))
    out.update(aggregate_detailed_rows(rows, "detailed"))
    return out


def run_segment_mixed(args, model_names, subjects, arrays, device):
    all_indices = build_all_segment_indices(subjects)
    folds = build_segment_folds(all_indices, args.folds, args.seed)
    max_folds = args.folds if args.max_folds <= 0 else min(args.max_folds, args.folds)
    summaries = {}
    for model_name in model_names:
        fold_rows = []
        print(f"\n========== Segment-mixed comparison: {model_name} ==========")
        for fold_idx in range(max_folds):
            test_indices = folds[fold_idx]
            train_val_indices = [item for i, fold in enumerate(folds) if i != fold_idx for item in fold]
            train_indices, val_indices = split_train_val(train_val_indices, args.val_ratio, args.seed + fold_idx)
            fold_rows.append(train_segment_fold(args, model_name, subjects, arrays, fold_idx, train_indices, val_indices, test_indices, device))
        summary = {"model_name": model_name, "split_mode": "segment_mixed_non_subject_independent", "fold_summaries": fold_rows, "aggregate": aggregate_fold_metrics(fold_rows)}
        out_dir = args.output_dir / "segment_mixed" / model_name
        (out_dir / "segment_mixed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries[model_name] = summary
    return summaries


def merge_support_summaries(existing, new):
    if not existing:
        return new
    merged = deepcopy(existing)
    old_folds = {int(item["fold"]): item for item in merged.get("fold_summaries", [])}
    for new_fold in new.get("fold_summaries", []):
        fold_idx = int(new_fold["fold"])
        old_fold = old_folds.get(fold_idx)
        if old_fold is None:
            old_folds[fold_idx] = new_fold
            continue
        old_supports = {
            int(item["support_segments"]): item
            for item in old_fold.get("support_summaries", [])
            if "support_segments" in item
        }
        for support_item in new_fold.get("support_summaries", []):
            old_supports[int(support_item["support_segments"])] = support_item
        old_fold.update({k: v for k, v in new_fold.items() if k != "support_summaries"})
        old_fold["support_summaries"] = [old_supports[key] for key in sorted(old_supports)]

    old_overall = {
        int(item["support_segments"]): item
        for item in merged.get("overall_support_summaries", [])
        if "support_segments" in item
    }
    for item in new.get("overall_support_summaries", []):
        old_overall[int(item["support_segments"])] = item
    merged.update({k: v for k, v in new.items() if k not in {"fold_summaries", "overall_support_summaries"}})
    merged["fold_summaries"] = [old_folds[key] for key in sorted(old_folds)]
    merged["overall_support_summaries"] = [old_overall[key] for key in sorted(old_overall)]
    return merged


def run_continual(args, model_names, subjects, arrays, device):
    folds = build_balanced_subject_folds(subjects, args.folds)
    all_ids = sorted(int(subject["sample_id"]) for subject in subjects)
    support_list = [int(x.strip()) for x in args.support_list.split(",") if x.strip()]
    max_folds = args.folds if args.max_folds <= 0 else min(args.max_folds, args.folds)
    summaries = {}
    for model_name in model_names:
        print(f"\n========== Continual few-shot comparison: {model_name} ==========")
        all_by_support = {support: [] for support in support_list}
        fold_summaries = []
        for fold_idx in range(max_folds):
            test_ids = sorted(folds[fold_idx])
            base_ids = set(all_ids) - set(test_ids)
            base_model_path, base_test_metrics = train_base_model_generic(args, model_name, subjects, arrays, base_ids, fold_idx, device)
            fold_summary = {"fold": fold_idx, "test_ids": test_ids, "base_ids": sorted(base_ids), "support_summaries": []}
            for support_segments in support_list:
                rows = []
                support_dir = args.output_dir / "continual" / model_name / f"fold_{fold_idx}" / f"support_{support_segments:03d}"
                support_dir.mkdir(parents=True, exist_ok=True)
                for sample_id in test_ids:
                    sample_path = support_dir / f"sample{sample_id:02d}.json"
                    if args.reuse_existing_support and sample_path.exists():
                        row = load_json(sample_path, {})
                        print(f"{model_name} fold {fold_idx} support {support_segments} sample {sample_id}: reusing existing result")
                    else:
                        row = adapt_subject_generic(args, model_name, subjects, arrays, base_model_path, sample_id, support_segments, device)
                        sample_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                    light = {k: v for k, v in row.items() if k != "history"}
                    light["fold"] = fold_idx
                    rows.append(light)
                    all_by_support[support_segments].append(light)
                support_summary = {
                    "fold": fold_idx,
                    "support_segments": support_segments,
                    "base_test_metrics": base_test_metrics,
                    "aggregate": aggregate(rows),
                    "before_detailed_aggregate": aggregate_detailed_rows(rows, "before_detailed"),
                    "after_detailed_aggregate": aggregate_detailed_rows(rows, "after_detailed"),
                    "delta_detailed_aggregate": aggregate_detailed_rows(rows, "delta_detailed"),
                    "per_subject": rows,
                }
                (support_dir / "summary.json").write_text(json.dumps(support_summary, indent=2), encoding="utf-8")
                fold_summary["support_summaries"].append({"support_segments": support_segments, "aggregate": support_summary["aggregate"]})
                print(f"{model_name} fold {fold_idx} support {support_segments}: after PCC {support_summary['aggregate']['after_corr_mean']:.4f}, delta {support_summary['aggregate']['delta_corr_mean']:.4f}")
            fold_summaries.append(fold_summary)
        overall = [
            {
                "support_segments": support,
                "support_seconds": support * args.segment_seconds,
                "aggregate": aggregate(rows),
                "before_detailed_aggregate": aggregate_detailed_rows(rows, "before_detailed"),
                "after_detailed_aggregate": aggregate_detailed_rows(rows, "after_detailed"),
                "delta_detailed_aggregate": aggregate_detailed_rows(rows, "delta_detailed"),
                "per_subject": rows,
            }
            for support, rows in all_by_support.items()
        ]
        summary = {"model_name": model_name, "fold_test_ids": folds[:max_folds], "fold_summaries": fold_summaries, "overall_support_summaries": overall}
        out_dir = args.output_dir / "continual" / model_name
        if args.merge_existing_summary:
            existing = load_json(out_dir / "continual_summary.json", None)
            summary = merge_support_summaries(existing, summary)
        (out_dir / "continual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries[model_name] = summary
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Run five adapted comparison reconstruction methods on the current heart-sound project protocols.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment", choices=["segment_mixed", "continual", "both"], default="both")
    parser.add_argument("--models", type=str, default="all", help=f"Comma list or all. Choices: {','.join(MODEL_NAMES)}")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--support-list", type=str, default="48,72,96")
    parser.add_argument("--support-strategy", choices=["random", "top_quality", "stratified_quality", "hybrid_quality"], default="stratified_quality")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--base-train-ratio", type=float, default=0.82)
    parser.add_argument("--base-val-ratio", type=float, default=0.09)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--early-stop", type=int, default=35)
    parser.add_argument("--base-epochs", type=int, default=180)
    parser.add_argument("--base-min-epochs", type=int, default=40)
    parser.add_argument("--base-early-stop", type=int, default=35)
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--adapt-batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
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
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapt-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-existing-base",
        action="store_true",
        help="Reuse existing continual base models and base_test_metrics when supplementing new support sizes.",
    )
    parser.add_argument(
        "--merge-existing-summary",
        action="store_true",
        help="Merge newly generated support summaries into existing continual_summary.json/compare_summary.json instead of replacing older supports.",
    )
    parser.add_argument(
        "--reuse-existing-support",
        action="store_true",
        help="Reuse existing per-subject support results, useful when resuming a partially completed supplement run.",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir = (PROJECT_ROOT / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir.resolve()
    args.output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model_names = parse_model_list(args.models)
    subjects = discover_subjects(args.data_dir)
    arrays = [np.load(subject["file"], mmap_mode="r") for subject in subjects]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    run_config = sanitize_args(args)
    run_config["models"] = model_names
    run_config["n_subjects"] = len(subjects)
    run_config["n_segments"] = int(sum(int(s["segments"]) for s in subjects))
    run_config["model_sources"] = {
        "biosignal_gan": "compare_program/biosignalGANs-main, pix2pix/CycleGAN-style generator adapted to time-domain heart-sound reconstruction",
        "p2e_wgan": "compare_program/P2E-WGAN-ecg-ppg-reconstruction-main/models.py GeneratorUNet adapted to 2 input channels",
        "cardiogan": "compare_program/ppg2ecg-cardiogan-main/codes/module.py attention U-Net generator adapted from TensorFlow to PyTorch",
        "ppg2ecg": "compare_program/ppg2ecg-pytorch-master/modules/models.py PPG2ECG convolutional autoencoder adapted to 2 input channels",
        "rddm": "compare_program/RDDM-main/model.py self-attention U-Net backbone adapted as direct reconstructor",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    final = load_json(args.output_dir / "compare_summary.json", {}) if args.merge_existing_summary else {}
    if args.experiment in ("segment_mixed", "both"):
        final["segment_mixed"] = run_segment_mixed(args, model_names, subjects, arrays, device)
    if args.experiment in ("continual", "both"):
        final.setdefault("continual", {})
        final["continual"].update(run_continual(args, model_names, subjects, arrays, device))
    (args.output_dir / "compare_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"\nSaved comparison summary to: {args.output_dir / 'compare_summary.json'}")


if __name__ == "__main__":
    main()

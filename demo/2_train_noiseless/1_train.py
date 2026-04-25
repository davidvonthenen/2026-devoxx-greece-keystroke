#!/usr/bin/env python3
"""
Mel-spectrogram image classifier (multi-class) with:

- CoAtNet-style hybrid backbone:
    * MBConv (depthwise conv) stages for local spectrogram patterns
    * global relative self-attention stages for global context
- SpecAugment-style masking + time-shift augmentation (image-domain)
- Hyper-parameter grid-search

Reference:
Harrison, J., Toreini, E., & Mehrnezhad, M. (2023).
"A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards".
arXiv:2308.01074v1 [cs.CR]

Updates:
- Grid includes hidden_size up to 64 to better match CoAtNet-0-like channel widths.
- Added training accuracy tracking to monitor for 'collapse' described in paper.
- Safety checks for empty dataloaders.
"""

###############################################################################
# 1. Imports & global config, Device Selection
###############################################################################
import os, glob, random, itertools, warnings, math, shutil
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Device selection prefers MPS on Apple Silicon, then CUDA, then CPU.
# (Practical note) MPS makes MacBooks surprisingly usable for deep learning,
# but performance and operator support can vary by PyTorch version; if you
# see unexpected slowdowns, check for CPU fallbacks in the profiler/logs.
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">> Using Apple-Silicon MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(">> Using CUDA GPU")
else:
    print(">> Using CPU")

###############################################################################
# 2. Experiment-level constants / hyper-parameter grid
###############################################################################
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2

# Grid-search space.
# Paper Notes:
# - LR: 5e-4 was found optimal to prevent collapse (Table 2).
# - Batch Size: 16 was used (Table 3).
# - Hidden Size: Paper uses CoAtNet (typically starts at 64 channels).
# param_grid = {
#     "hidden_size":   [64],      # Updated from 16 to 64 to match CoAtNet-0 capacity
#     "learning_rate": [5e-4],    # Paper optimal value
#     "batch_size":    [16],      # Paper value
# }
param_grid = {
    "hidden_size":   [16, 32, 64],
    "learning_rate": [0.001, 0.0005, 0.0003],
    "batch_size":    [8, 16, 32],
}

# Paper used 1100 epochs.
MAX_EPOCHS = 1200
# MAX_EPOCHS = 4

# Early stopping still uses validation loss.
VAL_PATIENCE_DEFAULT = 750
# VAL_PATIENCE_DEFAULT = 2

# Optimizer / regularization
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0

# OneCycleLR schedule (paper uses linear annealing)
ONECYCLE_PCT_START = 0.10
ONECYCLE_DIV_FACTOR = 10.0
ONECYCLE_FINAL_DIV_FACTOR = 100.0

# Image/augmentation defaults (Paper: 64x64 inputs)
IMG_SIZE = 64
IN_CHANNELS = 3  # CoAtNet expects 3 channels

# SpecAugment-ish masking (image domain)
# Paper: "Time-shifted randomly by up to 40%"
TIMESHIFT_PCT = 0.40
# Paper: "Masking... random 10% of both time and frequency"
MASK_PCT = 0.10
MASKS_PER_AXIS = 2

# File types
IMAGE_EXTS = (".png")

# NOTE: In Python, (".png") is just a string, not a 1-tuple.
# That means iterating over IMAGE_EXTS yields characters ('.', 'p', 'n', 'g').
# _is_image_file() still works by coincidence (".png" in ".png" is True),
# but _glob_images() expects IMAGE_EXTS to be an iterable of extensions.
# If you intended a single-extension tuple, it should be: (".png",)

# Early-stop improvement check
MIN_DELTA_ABS = 1e-4
MIN_DELTA_REL = 2e-3

def improved(curr: float, best: float) -> bool:
    """Return True if `curr` is meaningfully better (lower) than `best`."""
    if not math.isfinite(best):
        return True
    min_delta = max(MIN_DELTA_ABS, MIN_DELTA_REL * max(abs(best), 1e-8))
    # Hybrid threshold: require either a small absolute improvement or a relative one.
    # This avoids treating tiny floating-point jitter as a real validation improvement.
    return (best - curr) > min_delta


###############################################################################
# 3. Utility helpers (splitting, file ops)
###############################################################################
def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def split_files_3way(files: List[Path], train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Shuffles and splits a list of Paths into train/val/test.
    Defensive for tiny class sizes.
    """
    files = list(files)
    random.shuffle(files)
    n = len(files)
    # Tiny-class handling: for very small `n`, we special-case the split so we
    # don't end up with an empty train set (model never sees the class) or an
    # empty validation set (early stopping becomes meaningless).

    if n == 0:
        return [], [], []
    if n == 1:
        return files, [], []
    if n == 2:
        return [files[0]], [files[1]], []
    if n == 3:
        return [files[0]], [files[1]], [files[2]]

    train_count = int(train_ratio * n)
    val_count = int(val_ratio * n)

    train_count = max(train_count, 1)
    val_count = max(val_count, 1)
    # If rounding makes test_count zero, we steal one sample back from train/val.
    # This keeps the test split non-empty for per-class sanity checks.

    test_count = n - train_count - val_count
    if test_count < 1:
        if train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        test_count = n - train_count - val_count
        test_count = max(test_count, 1)

    train_end = train_count
    val_end = train_count + val_count
    return files[:train_end], files[train_end:val_end], files[val_end:]


def list_label_dirs(data_root: Path) -> List[Path]:
    return sorted([p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")])


@dataclass(frozen=True)
class SplitPlan:
    label: str
    train: List[Path]
    val: List[Path]
    test: List[Path]


def build_split_plan(data_root: Path, train: float, val: float, test: float) -> List[SplitPlan]:
    """
    Stratified split: split *within each label folder* to preserve per-class ratios.
    """
    plans: List[SplitPlan] = []

    for label_dir in list_label_dirs(data_root):
        # Ignore pre-split folders if they exist.
        if label_dir.name in ("train", "val", "test"):
            continue

        files = sorted([p for p in label_dir.iterdir() if _is_image_file(p)])
        if not files:
            continue
        tr, va, te = split_files_3way(files, train, val, test)
        plans.append(SplitPlan(label=label_dir.name, train=tr, val=va, test=te))

    if not plans:
        raise RuntimeError(f"No label subfolders with image files found under: {data_root}")

    return plans


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def place_files(
    files: List[Path],
    dst_dir: Path,
    *,
    move: bool = False,
    overwrite: bool = False,
) -> int:
    """
    Copy/move files into dst_dir. Returns count written.
    """
    ensure_dir(dst_dir)
    written = 0

    for src in files:
        dst = dst_dir / src.name

        if dst.exists():
            if overwrite:
                dst.unlink()
            else:
                # Avoid silent collisions; two different files named the same is a human problem, not a filesystem one.
                raise FileExistsError(f"Destination exists: {dst} (use --overwrite to replace)")

        if move:
            shutil.move(str(src), str(dst))
        else:
            # copy2() preserves file metadata (mtime, etc.), which is nice when you
            # want the split folders to remain audit-friendly / reproducible.
            shutil.copy2(str(src), str(dst))

        written += 1

    return written


def write_splits(
    plans: List[SplitPlan],
    out_root: Path,
    *,
    move: bool = False,
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    Materialize split plan to disk:
        out_root/train/<label>/*
        out_root/val/<label>/*
        out_root/test/<label>/*
    """
    totals = {"train": 0, "val": 0, "test": 0}

    for plan in plans:
        train_dir = out_root / "train" / plan.label
        val_dir = out_root / "val" / plan.label
        test_dir = out_root / "test" / plan.label

        totals["train"] += place_files(plan.train, train_dir, move=move, overwrite=overwrite)
        totals["val"] += place_files(plan.val, val_dir, move=move, overwrite=overwrite)
        totals["test"] += place_files(plan.test, test_dir, move=move, overwrite=overwrite)

    return totals


def make_label_pairs(file_list: List[str], label: int):
    """Helper: [(path, lbl), ...] pairs used to build Datasets and minibatches."""
    return [(f, label) for f in file_list]

def _list_label_dirs(path: str) -> List[str]:
    """Return sorted list of subdirectories (labels), excluding hidden dirs."""
    if not os.path.isdir(path):
        return []
    labels = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full) and not name.startswith("."):
            labels.append(name)
    return sorted(labels)

def _glob_images(folder: str) -> List[str]:
    out = []
    # IMAGE_EXTS is intended to be a tuple/list of extensions (".png", ".jpg", ...).
    for ext in IMAGE_EXTS:
        out.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(out)

def build_pairs_and_labels(data_root: str) -> Tuple[List[Tuple[str,int]],
                                                       List[Tuple[str,int]],
                                                       List[Tuple[str,int]],
                                                       Dict[str,int],
                                                       List[str]]:
    """
    Build train/val/test (path, label_id) pairs and label mappings.
    """
    # Dynamic labels inferred from union across splits
    label_set = set()
    for split in ("train", "val", "test"):
        split_root = os.path.join(data_root, split)
        for lab in _list_label_dirs(split_root):
            label_set.add(lab)
    idx_to_label = sorted(label_set)
    # Sorting makes label indices stable across runs/machines.
    # That stability matters because we checkpoint idx_to_label for inference and
    # use numeric label ids when computing confusion matrices and reports.
    label_to_idx = {lab: i for i, lab in enumerate(idx_to_label)}

    train_pairs, val_pairs, test_pairs = [], [], []

    for lab in idx_to_label:
        # Train
        files = _glob_images(os.path.join(data_root, "train", lab))
        train_pairs.extend(make_label_pairs(files, label_to_idx[lab]))

        # Validate
        files = _glob_images(os.path.join(data_root, "val", lab))
        val_pairs.extend(make_label_pairs(files, label_to_idx[lab]))

        # Test
        files = _glob_images(os.path.join(data_root, "test", lab))
        test_pairs.extend(make_label_pairs(files, label_to_idx[lab]))

    for lst in (train_pairs, val_pairs, test_pairs):
        random.shuffle(lst)

    return train_pairs, val_pairs, test_pairs, label_to_idx, idx_to_label



###############################################################################
# 4. Image dataset + normalization + SpecAugment-style transforms
###############################################################################
def _pil_load_rgb(path: str, img_size: int) -> torch.Tensor:
    """
    Load an image as float tensor in [0,1], shape (C,H,W) with C=3.
    """
    img = Image.open(path).convert("RGB")
    # Many spectrogram PNGs are effectively grayscale; converting to RGB keeps the model's
    # 3-channel conv stem happy (and typically just duplicates the single channel).
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), resample=Image.BILINEAR)
        # Resizing is a trade-off: it standardizes shapes for batching, but it also
        # changes the effective time/frequency resolution encoded in the image.
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
    return t

def _compute_mean_std(paths: List[str], img_size: int, max_items: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean/std across all pixels in the dataset (streaming).
    """
    if not paths:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32), np.array([0.25, 0.25, 0.25], dtype=np.float32)

    if max_items is not None and len(paths) > max_items:
        paths = random.sample(paths, max_items)

    sum_ = torch.zeros(3, dtype=torch.float64)
    sumsq = torch.zeros(3, dtype=torch.float64)
    count = 0
    # float64 accumulators reduce numerical drift when summing millions of pixels.
    # `count` is the number of pixels per channel (H*W per image), not the number of images.

    for p in paths:
        x = _pil_load_rgb(p, img_size).double()  # (3,H,W)
        sum_ += x.sum(dim=(1, 2))
        sumsq += (x * x).sum(dim=(1, 2))
        count += x.shape[1] * x.shape[2]

    mean = sum_ / max(1, count)
    var = (sumsq / max(1, count)) - (mean * mean)
    # Clamp keeps us away from sqrt of tiny negative numbers caused by floating-point round-off.
    std = torch.sqrt(torch.clamp(var, min=1e-8))

    return mean.float().numpy(), std.float().numpy()

def _normalize(img: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    # Convert numpy stats to tensors on the same device/dtype as the image, then
    # broadcast over H/W. This keeps normalization cheap and avoids CPU↔device copies.
    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_t  = torch.tensor(std,  dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_t  = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
    return (img - mean_t) / std_t

def _time_shift(img: torch.Tensor, pct: float, fill: Optional[float] = None) -> torch.Tensor:
    """
    Shift along width (time axis) by up to pct of width.
    Fills vacated region with `fill` (default: mean of the image).
    """
    if pct <= 0:
        return img
    _, _, W = img.shape
    # Convention: width corresponds to time. Shifting simulates slight timing jitter
    # (keystroke alignment differences, microphone latency, etc.) without changing labels.
    max_shift = int(round(W * pct))
    if max_shift < 1:
        return img

    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return img

    if fill is None:
        fill = float(img.mean().item())

    out = img.clone()
    if shift > 0:
        out[..., :shift] = fill
        out[..., shift:] = img[..., :W-shift]
    else:
        s = -shift
        out[..., W-s:] = fill
        out[..., :W-s] = img[..., s:]
    return out

def _mask_axis(img: torch.Tensor, axis: int, mask_pct: float, masks: int, fill: Optional[float] = None) -> torch.Tensor:
    """
    Apply rectangular masks along one axis:
      axis=1 masks frequency (height) bands
      axis=2 masks time (width) bands
    """
    if mask_pct <= 0 or masks <= 0:
        return img
    C, H, W = img.shape
    L = H if axis == 1 else W
    max_width = max(1, int(round(L * mask_pct)))
    # Like SpecAugment: each mask is a contiguous band with random width up to `mask_pct`
    # of the axis length (time or frequency).
    if fill is None:
        fill = float(img.mean().item())
    out = img.clone()
    for _ in range(masks):
        width = random.randint(1, max_width)
        start = random.randint(0, max(0, L - width))
        if axis == 1:
            out[:, start:start+width, :] = fill
        else:
            out[:, :, start:start+width] = fill
    return out

class MelSpectrogramImageDataset(Dataset):
    """
    Dataset reading mel-spectrogram images and returning:
      image: (C,H,W) float32 normalized
      label: int64

    Training dataset computes mean/std and applies augmentations.
    Validation/test reuse training mean/std and disable augmentation.
    """
    def __init__(
        self,
        img_label_pairs: List[Tuple[str, int]],
        img_size: int = IMG_SIZE,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        compute_stats: bool = False,
        do_augmentation: bool = False,
        timeshift_pct: float = TIMESHIFT_PCT,
        mask_pct: float = MASK_PCT,
        masks_per_axis: int = MASKS_PER_AXIS,
    ):
        super().__init__()
        self.img_label_pairs = img_label_pairs
        self.img_size = img_size
        self.do_augmentation = do_augmentation
        self.timeshift_pct = timeshift_pct
        self.mask_pct = mask_pct
        self.masks_per_axis = masks_per_axis

        self.mean = mean
        self.std = std
        # Mean/std should be computed on the *training* split only. For val/test we pass
        # the training stats in to avoid leaking information across splits.
        # Normalization stats reflect the dataset distribution, so computing them on val/test would be leakage.
        if compute_stats or (self.mean is None or self.std is None):
            paths = [p for p, _ in img_label_pairs]
            self.mean, self.std = _compute_mean_std(paths, img_size, max_items=None)

    def __len__(self):
        return len(self.img_label_pairs)

    def __getitem__(self, idx):
        path, label = self.img_label_pairs[idx]
        img = _pil_load_rgb(path, self.img_size)  # (3,H,W) float
        # Assumes spectrogram layout: frequency on the y-axis (H) and time on the x-axis (W).
        # If your spectrogram images are transposed/flipped, swap the axis ids used below
        # (time_shift operates on width; mask_axis axis=1 is height/frequency, axis=2 is width/time).
        if self.do_augmentation:
            img = _time_shift(img, self.timeshift_pct)
            img = _mask_axis(img, axis=2, mask_pct=self.mask_pct, masks=self.masks_per_axis)  # time masks
            img = _mask_axis(img, axis=1, mask_pct=self.mask_pct, masks=self.masks_per_axis)  # freq masks

        img = _normalize(img, self.mean, self.std).to(torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return img, y


###############################################################################
# 5. CoAtNet-style building blocks (MBConv + relative attention)
###############################################################################
def _choose_num_heads(dim: int) -> int:
    # Heuristic: prefer more heads when possible, but keep per-head dimension reasonably large.
    # (Tiny head dims make attention noisy and slow without adding much representational power.)
    for h in (8, 4, 2, 1):
        if dim % h == 0 and (dim // h) >= 16:
            return h
    return 1

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None,
                 groups: int = 1, act: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        # `groups` lets this block act as either a standard conv (groups=1) or a depthwise conv
        # (groups=in_ch with in_ch==out_ch), which MBConv uses for cheap spatial mixing.
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(ch * se_ratio))
        self.fc1 = nn.Conv2d(ch, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, ch, kernel_size=1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class MBConv(nn.Module):
    """
    Mobile inverted bottleneck with depthwise conv + SE.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: int = 4,
                 se_ratio: float = 0.25, drop: float = 0.0):
        super().__init__()
        # Residual connection only when spatial size and channel count match.
        # (Downsampling blocks can't be residual-add without extra projection logic.)
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = in_ch * expansion
        # `expansion` controls the width of the internal bottleneck. Larger expansion increases
        # capacity (and compute) while keeping the depthwise conv cost relatively low.

        self.expand = ConvBNAct(in_ch, hidden, k=1, s=1, p=0, act=True) if expansion != 1 else nn.Identity()
        self.dwconv = ConvBNAct(hidden, hidden, k=3, s=stride, p=1, groups=hidden, act=True)  # depthwise
        self.se = SqueezeExcite(hidden, se_ratio=se_ratio)
        self.project = ConvBNAct(hidden, out_ch, k=1, s=1, p=0, act=False)
        self.drop = nn.Dropout(p=drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        out = self.expand(x)
        out = self.dwconv(out)
        out = self.se(out)
        out = self.project(out)
        out = self.drop(out)
        if self.use_residual:
            out = out + x
        return out

class RelativePositionBias(nn.Module):
    """
    Learnable 2D relative positional bias for global attention over an HxW grid.
    """
    def __init__(self, H: int, W: int, num_heads: int):
        super().__init__()
        self.H = H
        self.W = W
        self.num_heads = num_heads
        self.num_pos = (2 * H - 1) * (2 * W - 1)
        self.bias_table = nn.Parameter(torch.zeros(self.num_pos, num_heads))

        # Precompute pairwise relative position index for HxW tokens
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        try:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2,H,W)
        except TypeError:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # older torch
        coords_flat = coords.flatten(1)  # (2,N)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2,N,N)
        rel = rel.permute(1, 2, 0).contiguous()  # (N,N,2)
        rel[:, :, 0] += H - 1
        rel[:, :, 1] += W - 1
        rel[:, :, 0] *= (2 * W - 1)
        rel_index = rel.sum(-1)  # (N,N)
        # rel_index maps every (token_i, token_j) pair to an offset into bias_table.
        # persistent=False keeps checkpoints a bit smaller by not saving this deterministic buffer.
        self.register_buffer("rel_index", rel_index, persistent=False)

        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.bias_table, std=0.02)
        else:
            nn.init.normal_(self.bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        # returns (num_heads, N, N)
        N = self.H * self.W
        bias = self.bias_table[self.rel_index.view(-1)].view(N, N, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 rel_pos_bias: Optional[RelativePositionBias] = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rel_pos_bias = rel_pos_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # Pack Q/K/V together for a single matmul-friendly projection, then reshape into
        # (B, heads, tokens, head_dim).
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3,B,h,N,hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,h,N,N)
        if self.rel_pos_bias is not None:
            attn = attn + self.rel_pos_bias().unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock2D(nn.Module):
    """
    Transformer-style block operating on a 2D feature map via flattening.
    Uses global attention with 2D relative position bias.
    """
    def __init__(self, dim: int, H: int, W: int, num_heads: int, drop: float = 0.0,
                 attn_drop: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.H, self.W = H, W
        self.norm1 = nn.LayerNorm(dim)
        self.rpb = RelativePositionBias(H, W, num_heads)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop, rel_pos_bias=self.rpb)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        if H != self.H or W != self.W:
            raise ValueError(f"TransformerBlock2D expected {(self.H, self.W)}, got {(H, W)}")
        t = x.flatten(2).transpose(1, 2)  # (B,N,C)
        # flatten(2) produces tokens in row-major order (h0w0, h0w1, ...). The relative
        # position bias table is built assuming this same ordering.

        t = t + self.attn(self.norm1(t))
        t = t + self.mlp(self.norm2(t))

        out = t.transpose(1, 2).reshape(B, C, H, W)
        return out


###############################################################################
# 6. CoAtNet + Transformer hybrid classifier
###############################################################################
class CoAtNetTransformerClassifier(nn.Module):
    """
    CoAtNet-style image classifier tuned for mel-spectrogram images (e.g., 64x64).

    Structure mirrors the paper's description:
      - 2 depthwise-convolutional stages (MBConv)
      - 2 global relative attention stages (TransformerBlock2D)
      - global average pool + linear classification head
    """
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout

        # Channel plan: keep attention width modest across the grid.
        stem_dim = hidden_size
        conv2_dim = hidden_size * 2
        attn_dim = conv2_dim
        # Keeping attn_dim constant across attention stages avoids extra projection layers.

        # Downsampling plan for img_size=64:
        #   stem: 64 -> 64
        #   conv1: 64 -> 32
        #   conv2: 32 -> 16
        #   attn1: 16 -> 8
        #   attn2:  8 -> 4
        if img_size % 16 != 0:
            raise ValueError(f"img_size must be divisible by 16 (got {img_size})")

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, stem_dim, k=3, s=1, p=1, act=True),
            ConvBNAct(stem_dim, stem_dim, k=3, s=1, p=1, act=True),
        )

        self.conv1 = MBConv(stem_dim, stem_dim, stride=2, expansion=4, drop=dropout)
        self.conv2 = MBConv(stem_dim, conv2_dim, stride=2, expansion=4, drop=dropout)

        self.attn1_down = ConvBNAct(conv2_dim, attn_dim, k=3, s=2, p=1, act=True)  # 16->8
        self.attn2_down = ConvBNAct(attn_dim,  attn_dim, k=3, s=2, p=1, act=True)  # 8->4

        # Attention stage 1 at (img_size/8) x (img_size/8)
        H1 = img_size // 8
        W1 = img_size // 8
        heads = _choose_num_heads(attn_dim)
        self.attn1 = nn.ModuleList([
            TransformerBlock2D(attn_dim, H1, W1, num_heads=heads, drop=dropout, attn_drop=dropout)
        ])

        # Attention stage 2 at (img_size/16) x (img_size/16)
        H2 = img_size // 16
        W2 = img_size // 16
        self.attn2 = nn.ModuleList([
            TransformerBlock2D(attn_dim, H2, W2, num_heads=heads, drop=dropout, attn_drop=dropout)
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(attn_dim, num_classes),
        )

        # Attached by training code for inference consistency:
        # self.mean, self.std, self.idx_to_label

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns logits: (B, num_classes)
        """
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn1_down(x)
        for blk in self.attn1:
            x = blk(x)
        x = self.attn2_down(x)
        for blk in self.attn2:
            x = blk(x)
        return self.head(x)


###############################################################################
# 7. Training / evaluation helpers
###############################################################################
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    num_epochs: int,
    checkpoint_path: str,
    val_patience: int = VAL_PATIENCE_DEFAULT,
):
    """
    Train with early stopping on validation loss.
    Uses OneCycleLR with linear ramps (pct_start controls the warmup fraction).
    This is a practical proxy for the paper's 'linear annealing' description, not a bitwise match.
    Checkpoints are written only when validation loss improves significantly.
    """
    criterion = nn.CrossEntropyLoss()
    # CrossEntropyLoss expects raw, unnormalized logits (no softmax) and int class ids.
    optim = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    # OneCycleLR is stepped per batch; uses 'max_lr' semantics.
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=ONECYCLE_PCT_START,
        anneal_strategy="linear",
        div_factor=ONECYCLE_DIV_FACTOR,
        final_div_factor=ONECYCLE_FINAL_DIV_FACTOR,
    )

    best_val_loss = float("inf")
    no_improve_val = 0
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        # ---- training ----
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # set_to_none=True can be slightly faster and reduces memory writes versus zeroing.
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            # Gradient clipping guards against rare exploding updates (especially with attention blocks).
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optim.step()
            # OneCycleLR is defined per-batch, so we step it once per optimizer update.
            scheduler.step()
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_count += yb.size(0)

        avg_train_loss = total_loss / max(1, len(train_loader))
        train_acc = total_correct / max(1, total_count)
        # Tracking train_acc is a quick sanity check for 'collapse' (e.g., predicting one class forever).
        train_losses.append(avg_train_loss)

        # ---- validation ----
        model.eval()
        vtotal = 0.0
        y_true, y_pred = [], []
        # We accumulate predictions on CPU for sklearn metrics; for large datasets you might
        # stream metrics instead to avoid holding everything in memory.
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vtotal += criterion(logits, yb).item()
                y_true.extend(yb.cpu().numpy().tolist())
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

        avg_val_loss = vtotal / max(1, len(val_loader))
        val_losses.append(avg_val_loss)
        val_acc = accuracy_score(y_true, y_pred) if len(y_true) else float("nan")

        lr_now = optim.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{num_epochs}  loss={avg_train_loss:.4f}  tr_acc={train_acc:.3f}  "
              f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.3f}  lr={lr_now:.2e}")

        if improved(avg_val_loss, best_val_loss):
            best_val_loss = avg_val_loss
            no_improve_val = 0

            # Checkpoint includes enough metadata to rehydrate the model AND apply the same
            # normalization + label mapping at inference time (critical for reproducibility).
            checkpoint = {
                "model_state_dict": model.state_dict(),
                # Model hyperparams
                "img_size": getattr(model, "img_size", None),
                "in_channels": getattr(model, "in_channels", None),
                "hidden_size": getattr(model, "hidden_size", None),
                "num_classes": getattr(model, "num_classes", None),
                "dropout": getattr(model, "dropout", None),
                "batch_size": getattr(model, "batch_size", None),
                # Normalization stats for inference
                "mean": getattr(model, "mean", None),
                "std": getattr(model, "std", None),
                # Label mapping for inference
                "idx_to_label": getattr(model, "idx_to_label", None),
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            no_improve_val += 1

        if no_improve_val >= val_patience:
            print(">> Early-stop (val-loss plateau).")
            break

    # Load the best model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Reattach aux inference state
    model.mean = checkpoint.get("mean")
    model.std = checkpoint.get("std")
    model.idx_to_label = checkpoint.get("idx_to_label")

    return train_losses, val_losses


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    idx_to_label: List[str],
    split_name: str = "Test",
):
    """
    Multi-class evaluation with accuracy + macro/weighted F1, plus confusion matrix.
    """
    model.eval()
    y_true, y_pred = [], []
    prob_chunks = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            y_true.extend(yb.cpu().numpy().tolist())
            prob_chunks.append(probs.detach().cpu().numpy())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    if len(prob_chunks) > 0:
        y_prob = np.concatenate(prob_chunks, axis=0).astype(np.float32) 
    else:
        y_prob = np.zeros((0, len(idx_to_label)), dtype=np.float32)

    if y_true.size == 0:
        print(f"\n>> {split_name} metrics: (no samples)")
        return {
            "acc": float("nan"),
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
            "f1_macro": float("nan"),
            "f1_weighted": float("nan"),
            "cm": None,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    labels = list(range(len(idx_to_label)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # confusion_matrix rows=true labels, cols=predicted labels (sklearn convention).

    print(f"\n>> {split_name} metrics:")
    print(f"   Accuracy     : {acc:.3f}")
    print(f"   Macro P/R/F1 : {p_macro:.3f} / {r_macro:.3f} / {f1_macro:.3f}")
    print(f"   Weighted F1  : {f1_weighted:.3f}")
    print(f"   Confusion Matrix [rows=true, cols=pred] (size={len(idx_to_label)}x{len(idx_to_label)}):")
    print(cm)

    print("\n" + classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=idx_to_label,
        digits=3,
        zero_division=0
    ))

    return {
        "acc": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


###############################################################################
# 8. Single-image inference (normalization- & label-aware)
###############################################################################
def infer_single_image(
    img_path: str,
    model: nn.Module,
    device: torch.device,
) -> Tuple[str, float]:
    """
    Load a single spectrogram image, apply saved normalization, run model.

    Returns: (predicted_label_name, confidence)
    """
    img_size = getattr(model, "img_size", IMG_SIZE)
    mean = getattr(model, "mean", None)
    std  = getattr(model, "std", None)
    if mean is None or std is None:
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std  = np.array([0.25, 0.25, 0.25], dtype=np.float32)

    x = _pil_load_rgb(img_path, img_size)
    x = _normalize(x, mean, std).to(torch.float32)
    xb = x.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[0]
        pred_i = int(torch.argmax(probs))

    idx_to_label = getattr(model, "idx_to_label", None)
    pred_name = idx_to_label[pred_i] if idx_to_label and 0 <= pred_i < len(idx_to_label) else str(pred_i)
    return pred_name, float(probs[pred_i])

# Backward-compatible alias: older code calls this name (despite it not being a CSV).
# The argument is still an image path; the name is historical baggage.
def infer_single_csv(csv_path: str, model: nn.Module, device: torch.device) -> Tuple[str, float]:
    return infer_single_image(csv_path, model, device)


###############################################################################
# 9. Train with a hyper-parameter combo
###############################################################################
def train_with_params(
    data_root: str,
    hparams: Dict[str, Any],
    train_pairs: List[Tuple[str,int]],
    val_pairs:   List[Tuple[str,int]],
    test_pairs:  List[Tuple[str,int]],
    idx_to_label: List[str],
    count: int,
    pos: int,
) -> Dict[str, Any]:
    """
    Build train/val/test datasets and loaders, instantiate model, train, evaluate.
    """
    bs = hparams["batch_size"]

    # Training dataset computes normalization stats and applies augmentation
    train_ds = MelSpectrogramImageDataset(
        train_pairs,
        img_size=IMG_SIZE,
        compute_stats=True,
        do_augmentation=True,
    )

    # Reuse mean/std for val/test to avoid leakage
    val_ds = MelSpectrogramImageDataset(
        val_pairs,
        img_size=IMG_SIZE,
        mean=train_ds.mean,
        std=train_ds.std,
        compute_stats=False,
        do_augmentation=False,
    )
    test_ds = MelSpectrogramImageDataset(
        test_pairs,
        img_size=IMG_SIZE,
        mean=train_ds.mean,
        std=train_ds.std,
        compute_stats=False,
        do_augmentation=False,
    )

    # num_workers=0 keeps the data pipeline single-process, which tends to be friendlier on
    # macOS (spawn semantics) and avoids surprising nondeterminism during experimentation.
    train_dl = DataLoader(train_ds, bs, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   bs, shuffle=False, num_workers=0, pin_memory=False)
    test_dl  = DataLoader(test_ds,  bs, shuffle=False, num_workers=0, pin_memory=False)

    num_classes = len(idx_to_label)

    model = CoAtNetTransformerClassifier(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        hidden_size=hparams["hidden_size"],
        num_classes=num_classes,
        dropout=0.1,
    ).to(device)

    # Attach normalization + labels for checkpointing/inference
    model.batch_size   = bs
    model.mean         = train_ds.mean.astype(np.float32)
    model.std          = train_ds.std.astype(np.float32)
    model.idx_to_label = idx_to_label

    ckpt_name = (f"{pos}_{count}_best_w{hparams['hidden_size']}_lr{hparams['learning_rate']}_bs{bs}.pth")

    train_losses, val_losses = train_model(
        model, train_dl, val_dl,
        learning_rate=hparams["learning_rate"],
        num_epochs=MAX_EPOCHS,
        checkpoint_path=ckpt_name,
        val_patience=VAL_PATIENCE_DEFAULT,
    )

    val_metrics  = test_model(model, val_dl,  idx_to_label, "Validation")
    test_metrics = test_model(model, test_dl, idx_to_label, "Test")

    # Reload smoke-test
    reloaded = torch.load(ckpt_name, map_location=device, weights_only=False)
    model.load_state_dict(reloaded["model_state_dict"])
    model.mean = reloaded.get("mean", model.mean)
    model.std  = reloaded.get("std",  model.std)
    model.idx_to_label = reloaded.get("idx_to_label", model.idx_to_label)

    if test_metrics.get("acc") == test_metrics.get("acc"):
        print(f">> Test accuracy: {test_metrics['acc'] * 100:.2f}%")
    else:
        print(">> Test accuracy: (no test samples)")

    print(">> Per-file test predictions:")
    for img_path, lbl in test_pairs:
        pred, conf = infer_single_image(img_path, model, device)
        true_name = idx_to_label[lbl] if 0 <= lbl < len(idx_to_label) else str(lbl)
        print(f"  {os.path.basename(img_path):50s} → pred={pred:20s}  conf={conf:0.3f}  true={true_name}")

    return {
        "model": model,
        "ckpt_name": ckpt_name,
        "val":  val_metrics,
        "test": test_metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


###############################################################################
# 10. Grid-search driver
###############################################################################
def grid_search_experiments(root="data", pos: int = 0):
    """
    Run a Cartesian grid search over `param_grid` hyperparameters.

    Supports:
      - Pre-split: data/{train,val,test}/<label>/*.png  (active/default code path)
      - Unsplit:   data/<label>/*.png  (enable by uncommenting the split_root block below)
      - Note: the call site below expects `split_root` to exist when using the unsplit path.

    For each configuration:
      • trains/evaluates, and
      • writes out the validation/test file lists next to the checkpoint.
    Tracks the best test accuracy and copies its checkpoint with the score suffix.
    """
    data_root = Path(root)
    if not data_root.exists() or not data_root.is_dir():
        raise RuntimeError(f"Data root folder does not exist or is not a directory: {data_root}")

    keys = list(param_grid)

    # # create a timestamped split folder
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # split_root = Path(f"{data_root}_{timestamp}")

    # print("== Gathering image files and building splits ...")
    # plans = build_split_plan(data_root, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    # total_files = 0
    # for plan in plans:
    #     n = len(plan.train) + len(plan.val) + len(plan.test)
    #     total_files += n
    #     print(f"[{plan.label}] total={n:4d}  train={len(plan.train):4d}  val={len(plan.val):4d}  test={len(plan.test):4d}")

    # print()
    # totals = write_splits(plans, split_root)

    # print("Overall totals:")
    # print(f"  train={totals['train']}  val={totals['val']}  test={totals['test']}  "
    #         f"(sum={sum(totals.values())}, labels={len(plans)}, files={total_files})")

    # Build (path, label_id) pairs and label mapping from a pre-split folder (train/val/test).
    train_pairs, val_pairs, test_pairs, label_to_idx, idx_to_label = build_pairs_and_labels(str(data_root))

    print(f"\nLabel mapping (label_id -> label_name):")
    for i, label in enumerate(idx_to_label):
        print(f"  {i}: {label}")

    cnt = 1
    best_acc = -1.0
    best_model = ""

    for values in itertools.product(*param_grid.values()):
        hparams = dict(zip(keys, values))
        header = ", ".join(f"{k}={v}" for k, v in hparams.items())

        print("\n" + "=" * 80)
        print(">>> Training with", header)
        print("=" * 80)

        results = train_with_params(
            # IMPORTANT: `split_root` is only defined if you enable the (currently commented) split-writing block above.
            # If your dataset is already pre-split under `data/`, `split_root` should conceptually be `data_root`.
            data_root=str(split_root),
            hparams=hparams,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            idx_to_label=idx_to_label,
            count=cnt,
            pos=pos,
        )

        # Persist the validation and test filenames next to the checkpoint for auditing.
        with open(f"{results['ckpt_name']}_dataset.txt", "w", encoding="utf-8") as f:
            f.write("Labels (idx -> name):\n")
            for i, name in enumerate(idx_to_label):
                f.write(f"  {i:03d}: {name}\n")

            f.write("\nValidation files:\n")
            for p, y in val_pairs:
                f.write(f"  • {os.path.basename(p)}\t(label={idx_to_label[y]})\n")

            f.write("\nTest files:\n")
            for p, y in test_pairs:
                f.write(f"  • {os.path.basename(p)}\t(label={idx_to_label[y]})\n")

        with open(f"{results['ckpt_name']}_results.txt", "w", encoding="utf-8") as f:
            f.write(("Overall results:\n"))
            test_acc = results["test"].get("acc", float("nan"))
            f.write(f"  Test accuracy: {test_acc}\n")
            f.write(f"\n")

            f.write("\nValidation metrics:\n")
            for k, v in results["val"].items():
                f.write(f"  {k}: {v}\n")
            f.write("\nTest metrics:\n")
            for k, v in results["test"].items():
                f.write(f"  {k}: {v}\n")

        test_acc = results["test"].get("acc", float("nan"))
        if test_acc == test_acc and test_acc > best_acc:
            best_acc = test_acc
            best_model = results["ckpt_name"]

        cnt += 1

    if best_model:
        print("\nBest model found:")
        print("  ", best_model)
        print(f"  acc={best_acc:.6f}")
        shutil.copy(best_model, f"{best_model}_{best_acc:.6f}_BEST")


###############################################################################
# 11. Main
###############################################################################
if __name__ == "__main__":
    # Seed important RNGs for reproducibility.
    random.seed(47)
    np.random.seed(47)
    torch.manual_seed(47)

    # Note: 'pos' is an experiment repetition index.
    # If you enable the auto-splitting block, each run can create a new random per-label split;
    # with pre-split data/{train,val,test}, repetitions mostly vary init/augmentation/minibatch order.
    for pos in range(2):
        print(f"\n\n### STARTING REPEATED EXPERIMENT RUN {pos+1}/2 ###")
        grid_search_experiments("data", pos)

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
- Default hidden_size increased to 64 to better match CoAtNet-0 capacity.
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
# MPS makes local training feasible on Apple Silicon, but kernel coverage/perf can differ vs CUDA/CPU.
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
# NOTE: `param_grid` is iterated in insertion order (Python 3.7+). That order becomes the grid-search sweep order.
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
IN_CHANNELS = 3  # Spectrogram PNGs are stored as RGB; model assumes 3-channel input.

# SpecAugment-ish masking (image domain)
# Paper: "Time-shifted randomly by up to 40%"
TIMESHIFT_PCT = 0.40
# Paper: "Masking... random 10% of both time and frequency"
MASK_PCT = 0.10
MASKS_PER_AXIS = 2

# File types / allowed image suffixes.
# NOTE: `IMAGE_EXTS` is set as a *string* (not a tuple).
# - `_is_image_file()` uses `suffix in IMAGE_EXTS`, which works as a substring check for '.png'.
# - `_glob_images()` iterates `for ext in IMAGE_EXTS`, which will iterate characters when IMAGE_EXTS is a string.
#   If you ever mean to support multiple extensions, use a tuple like: IMAGE_EXTS = ('.png', '.jpg')
IMAGE_EXTS = (".png")

# Early-stop improvement check
MIN_DELTA_ABS = 1e-4
MIN_DELTA_REL = 2e-3

def improved(curr: float, best: float) -> bool:
    """Return True if `curr` is meaningfully better (lower) than `best`."""
    # We require a minimum improvement to avoid checkpoint churn from floating-point noise.
    # The threshold is the max of an absolute floor and a relative fraction of the current best loss.
    if not math.isfinite(best):
        return True
    min_delta = max(MIN_DELTA_ABS, MIN_DELTA_REL * max(abs(best), 1e-8))
    return (best - curr) > min_delta


###############################################################################
# 3. Utility helpers (splitting, file ops)
###############################################################################
def _is_image_file(p: Path) -> bool:
    # `suffix.lower()` includes the leading dot (e.g., '.png').
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def split_files_3way(files: List[Path], train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    # Tiny classes are common in acoustic datasets; we special-case n<=3 to avoid empty splits.
    Shuffles and splits a list of Paths into train/val/test.
    Defensive for tiny class sizes.
    """
    files = list(files)
    random.shuffle(files)
    n = len(files)

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


# GroupCounts/SplitPlan exist to make split logic explicit and easier to audit/reproduce.
@dataclass(frozen=True)
class GroupCounts:
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test
    

@dataclass(frozen=True)
class SplitPlan:
    label: str
    train: List[Path]
    val: List[Path]
    test: List[Path]



def parse_groups_csv(s: str) -> List[str]:
    # Group keys are matched as substrings against filenames (see `_match_group`).
    # Lowercasing here ensures matching is case-insensitive.
    groups = [g.strip().lower() for g in s.split(",") if g.strip()]
    if not groups:
        raise ValueError("--groups produced no valid entries")
    # Preserve order while deduping
    seen = set()
    out = []
    for g in groups:
        if g not in seen:
            out.append(g)
            seen.add(g)
    return out


def parse_group_split(spec: str) -> Tuple[str, GroupCounts]:
    """
    # This is designed for CLI usage (e.g., 'hp=4,1,1') but kept here as a reusable parser.
    Parse 'name=a,b,c' -> ('name', GroupCounts(a,b,c)).
    """
    if "=" not in spec:
        raise ValueError(f"Invalid --group-split '{spec}'. Expected format: group=train,val,test")
    name, rhs = spec.split("=", 1)
    name = name.strip().lower()
    parts = [p.strip() for p in rhs.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid --group-split '{spec}'. Expected 3 comma-separated ints: train,val,test")
    try:
        tr, va, te = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as e:
        raise ValueError(f"Invalid --group-split '{spec}'. train,val,test must be ints") from e
    if tr < 0 or va < 0 or te < 0:
        raise ValueError(f"Invalid --group-split '{spec}'. Counts must be >= 0")
    return name, GroupCounts(tr, va, te)


def _match_group(filename_lower: str, groups: List[str]) -> Optional[str]:
    # A filename can match at most one group key; overlapping keys (e.g., 'mac' and 'macbook')
    # will trigger an error so you don't silently mislabel data.
    hits = [g for g in groups if g in filename_lower]
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        return None
    # Multiple matches implies overlapping group keys.
    raise ValueError(f"Ambiguous group match for '{filename_lower}': {hits}")


def list_label_dirs(data_root: Path) -> List[Path]:
    return sorted([p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")])

def build_split_plan_by_groups(
    data_root: Path,
    *,
    groups: List[str],
    counts: GroupCounts,
    group_size: int,
    shuffle_within_group: bool,
    seed: int,
    strict: bool,
    allow_ungrouped: bool,
) -> List[SplitPlan]:
    """
    For each label directory:
      - bucket files by group substring
      - within each group, select N files for train/val/test by fixed counts
    """
    rng = random.Random(seed)
    plans: List[SplitPlan] = []

    for label_dir in list_label_dirs(data_root):
        if label_dir.name in ("train", "val", "test"):
            continue

        all_files = sorted([p for p in label_dir.iterdir() if _is_image_file(p)])
        if not all_files:
            continue

        grouped: Dict[str, List[Path]] = {g: [] for g in groups}
        ungrouped: List[Path] = []

        for p in all_files:
            g = _match_group(p.name.lower(), groups)
            if g is None:
                ungrouped.append(p)
            else:
                grouped[g].append(p)

        if ungrouped and not allow_ungrouped:
            sample = ", ".join([u.name for u in ungrouped[:10]])
            more = "" if len(ungrouped) <= 10 else f" (+{len(ungrouped) - 10} more)"
            raise RuntimeError(
                f"[{label_dir.name}] Found files that match no group key: {sample}{more}. "
                f"Add a key to --groups or use --allow-ungrouped."
            )

        train_files: List[Path] = []
        val_files: List[Path] = []
        test_files: List[Path] = []

        for g in groups:
            files_g = sorted(grouped.get(g, []))

            # `strict=True` is a guardrail for curated datasets where each group should have an exact size.
            if strict and len(files_g) != group_size:
                raise RuntimeError(
                    f"[{label_dir.name}] Group '{g}' expected {group_size} files but found {len(files_g)}: "
                    f"{[p.name for p in files_g]}"
                )

            # Shuffle *within* each group so train/val/test pick different samples per run (subject to `seed`).
            if shuffle_within_group:
                rng.shuffle(files_g)

            # In strict mode, we expect the split counts to match the group size exactly.
            # In non-strict mode, we only require we don't ask for more than we have.
            # In strict mode we also require the requested split counts to sum to the expected group size.
            if strict and counts.total != group_size:
                raise RuntimeError(
                    f"Strict mode requires train+val+test == --group-size "
                    f"(got {counts.total} vs group_size={group_size})."
                )

            # if counts.total > len(files_g):
            #     raise RuntimeError(
            #         f"[{label_dir.name}] Group '{g}' split counts sum to {counts.total} "
            #         f"but group has {len(files_g)} files (counts: train={counts.train} val={counts.val} test={counts.test})."
            #     )

            tr_end = counts.train
            va_end = counts.train + counts.val
            te_end = counts.train + counts.val + counts.test

            train_files.extend(files_g[:tr_end])
            val_files.extend(files_g[tr_end:va_end])
            test_files.extend(files_g[va_end:te_end])

        plans.append(SplitPlan(label=label_dir.name, train=train_files, val=val_files, test=test_files))

    if not plans:
        raise RuntimeError(f"No label subfolders with audio files found under: {data_root}")

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
    # Collect all image paths under `folder`. The glob pattern is '*<ext>'.
    # IMPORTANT: this assumes IMAGE_EXTS is an iterable of extensions; see the note where IMAGE_EXTS is defined.
    out = []
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

    # Shuffle to de-correlate minibatches from filesystem ordering (DataLoader shuffle handles this too, but this helps even when shuffle=False).
    for lst in (train_pairs, val_pairs, test_pairs):
        random.shuffle(lst)

    return train_pairs, val_pairs, test_pairs, label_to_idx, idx_to_label



###############################################################################
# 4. Image dataset + normalization + SpecAugment-style transforms
###############################################################################
def _pil_load_rgb(path: str, img_size: int) -> torch.Tensor:
    """
    # Spectrograms are stored as PNGs; `.convert('RGB')` guarantees a consistent 3-channel tensor even if the source is grayscale.
    # Resizing is done here (not in the model) so all downstream code can assume a fixed spatial grid.
    Load an image as float tensor in [0,1], shape (C,H,W) with C=3.
    """
    img = Image.open(path).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
    return t

def _compute_mean_std(paths: List[str], img_size: int, max_items: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    # Computes channel-wise stats by streaming over images (no need to hold all pixels in memory).
    # Accumulators use float64 to reduce numerical error when summing lots of pixels.
    Compute per-channel mean/std across all pixels in the dataset (streaming).
    """
    if not paths:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32), np.array([0.25, 0.25, 0.25], dtype=np.float32)

    if max_items is not None and len(paths) > max_items:
        paths = random.sample(paths, max_items)

    sum_ = torch.zeros(3, dtype=torch.float64)
    sumsq = torch.zeros(3, dtype=torch.float64)
    count = 0

    for p in paths:
        x = _pil_load_rgb(p, img_size).double()  # (3,H,W)
        sum_ += x.sum(dim=(1, 2))
        sumsq += (x * x).sum(dim=(1, 2))
        count += x.shape[1] * x.shape[2]

    mean = sum_ / max(1, count)
    var = (sumsq / max(1, count)) - (mean * mean)
    std = torch.sqrt(torch.clamp(var, min=1e-8))

    return mean.float().numpy(), std.float().numpy()

def _normalize(img: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    # Convert numpy stats to tensors on the same device/dtype as the image for fast fused ops.
    # Shape is (C,1,1) so broadcasting works over HxW.
    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_t  = torch.tensor(std,  dtype=img.dtype, device=img.device).view(-1, 1, 1)
    # Guard against degenerate channels where std==0 (would otherwise produce inf/NaN).
    std_t  = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
    return (img - mean_t) / std_t

def _time_shift(img: torch.Tensor, pct: float, fill: Optional[float] = None) -> torch.Tensor:
    """
    # Convention: in a spectrogram image, width ~= time and height ~= frequency.
    Shift along width (time axis) by up to pct of width.
    Fills vacated region with `fill` (default: mean of the image).
    """
    if pct <= 0:
        return img
    _, _, W = img.shape
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
    # This is SpecAugment in image form: zero/mean out contiguous time or frequency bands.
    # `axis` refers to the tensor dimension after (C,H,W): axis=1 => H (frequency), axis=2 => W (time).
    Apply rectangular masks along one axis:
      axis=1 masks frequency (height) bands
      axis=2 masks time (width) bands
    """
    if mask_pct <= 0 or masks <= 0:
        return img
    C, H, W = img.shape
    L = H if axis == 1 else W
    max_width = max(1, int(round(L * mask_pct)))
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
    # NOTE: mean/std are computed over *this dataset's* image list when `compute_stats=True`.
    # You should only compute stats on the training split and reuse them for val/test to avoid leakage.
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
        # If stats were not provided, compute them from the provided image list (optionally subsample via max_items).
        if compute_stats or (self.mean is None or self.std is None):
            paths = [p for p, _ in img_label_pairs]
            self.mean, self.std = _compute_mean_std(paths, img_size, max_items=None)

    def __len__(self):
        return len(self.img_label_pairs)

    def __getitem__(self, idx):
        path, label = self.img_label_pairs[idx]
        img = _pil_load_rgb(path, self.img_size)  # (3,H,W) float
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
    # Heuristic: prefer more heads when divisible, but keep head_dim >= 16 to avoid overly tiny attention heads.
    for h in (8, 4, 2, 1):
        if dim % h == 0 and (dim // h) >= 16:
            return h
    return 1

# Small helper block used throughout the CNN backbone: Conv2d -> BatchNorm -> SiLU (or Identity).
# `groups` enables depthwise convolutions when set to `in_channels`.
class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None,
                 groups: int = 1, act: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SqueezeExcite(nn.Module):
    # Squeeze-and-Excitation: global pool -> bottleneck MLP -> channel-wise gating.
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
    # MBConv is the EfficientNet-style block: 1x1 expand -> 3x3 depthwise -> SE -> 1x1 project.
    # Residual connection is only used when spatial size and channel count match (stride==1 and in_ch==out_ch).
    Mobile inverted bottleneck with depthwise conv + SE.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: int = 4,
                 se_ratio: float = 0.25, drop: float = 0.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = in_ch * expansion

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
        # `persistent=False` keeps this large index tensor out of checkpoints; it is deterministic from (H,W).
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
        # Compute Q,K,V in one matmul then reshape to (3, B, heads, N, head_dim) for batched attention.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3,B,h,N,hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention. Shape stays (B, heads, N, N).
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,h,N,N)
        if self.rel_pos_bias is not None:
            # Add learnable relative position bias (per-head) to attention logits before softmax.
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
        # Flatten HxW to a token sequence of length N=H*W for attention, then transpose to (B,N,C).
        t = x.flatten(2).transpose(1, 2)  # (B,N,C)

        # Pre-norm Transformer: LayerNorm -> attention -> residual.
        t = t + self.attn(self.norm1(t))
        # Follow with MLP (feed-forward network) and another residual.
        t = t + self.mlp(self.norm2(t))

        out = t.transpose(1, 2).reshape(B, C, H, W)
        return out


###############################################################################
# 6. CoAtNet + Transformer hybrid classifier
###############################################################################
class CoAtNetTransformerClassifier(nn.Module):
    """
    # The attention stages are `ModuleList`s so you can deepen the model by appending more TransformerBlock2D blocks per stage.
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

        # Downsampling plan for img_size=64:
        #   stem: 64 -> 64
        #   conv1: 64 -> 32
        #   conv2: 32 -> 16
        #   attn1: 16 -> 8
        #   attn2:  8 -> 4
        # Downsampling happens 4 times (stride-2 conv/MBConv), so we need the final grid to be an integer size.
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
        # Choose an attention head count that divides `attn_dim` while keeping head_dim reasonably large.
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

    Learning-rate schedule:
      - Uses OneCycleLR stepped per batch with `anneal_strategy='linear'`.
      - This is a practical stand-in for the paper's "linear annealing" description, while still providing warmup + decay.

    Checkpointing:
      - Writes a checkpoint only when validation loss improves by `improved()` (min abs/rel delta).
      - Stores extra metadata (normalization stats + label mapping) so inference can be done without re-deriving them.
    """
    # Standard multi-class classification loss over logits.
    criterion = nn.CrossEntropyLoss()
    # Adam with weight decay acts as L2 regularization on weights (not on activations).
    optim = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    # OneCycleLR is stepped per batch; uses 'max_lr' semantics.
    # `OneCycleLR` needs an estimate of total steps. Guard against empty dataloaders.
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
            # `set_to_none=True` can reduce memory writes; gradients are allocated on first backward pass.
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            # Gradient clipping improves stability when the LR schedule peaks.
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optim.step()
            # OneCycleLR is stepped per *batch* (not per epoch).
            scheduler.step()
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_count += yb.size(0)

        avg_train_loss = total_loss / max(1, len(train_loader))
        train_acc = total_correct / max(1, total_count)
        train_losses.append(avg_train_loss)

        # ---- validation ----
        model.eval()
        vtotal = 0.0
        y_true, y_pred = [], []
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

        # Save only meaningfully improved checkpoints to avoid thrashing the filesystem.
        if improved(avg_val_loss, best_val_loss):
            best_val_loss = avg_val_loss
            no_improve_val = 0

            # Checkpoint payload is intentionally self-describing: it includes model hyperparams + preprocessing metadata.
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
    # `weights_only=False` is explicit for newer PyTorch versions that default to safer weight-only loading.
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
            # Softmax converts logits to class probabilities per sample.
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
    # Macro averages weight each class equally; weighted averages weight by support (class frequency).
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    labels = list(range(len(idx_to_label)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

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
    # Fallback stats are intentionally conservative; real deployments should prefer the saved training stats.
    if mean is None or std is None:
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std  = np.array([0.25, 0.25, 0.25], dtype=np.float32)

    x = _pil_load_rgb(img_path, img_size)
    x = _normalize(x, mean, std).to(torch.float32)
    # Add batch dimension: model expects (B,C,H,W).
    xb = x.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[0]
        pred_i = int(torch.argmax(probs))

    idx_to_label = getattr(model, "idx_to_label", None)
    # Map predicted class index back to a human-readable label if we have the mapping.
    pred_name = idx_to_label[pred_i] if idx_to_label and 0 <= pred_i < len(idx_to_label) else str(pred_i)
    return pred_name, float(probs[pred_i])

# Backward-compatible alias: older code calls this name.
def infer_single_csv(csv_path: str, model: nn.Module, device: torch.device) -> Tuple[str, float]:
    # Parameter name is a legacy artifact; the path still points to an image file.
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

    # `num_workers=0` avoids multiprocessing nondeterminism and is the safest default across OSes.
    # If you crank this up, you'll also want deterministic worker seeding for reproducible experiments.
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

    # Encode the hyper-parameter combo in the filename so experiment artifacts are self-identifying.
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
    # This catches cases where training succeeded but serialization missed required metadata.
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

    Active data layout (what the current code path expects):
      - Pre-split: <root>/{train,val,test}/<label>/*.png

    Optional / commented-out path:
      - The block below (currently commented) can materialize a timestamped `split_root` from an unsplit
        directory by using group-aware sampling (`build_split_plan_by_groups`).

    For each configuration:
      • trains + evaluates, and
      • writes out the validation/test file lists next to the checkpoint for auditing.

    Best model tracking:
      - Tracks the best *test* accuracy and copies its checkpoint with a score suffix.
    """
    data_root = Path(root)
    if not data_root.exists() or not data_root.is_dir():
        raise RuntimeError(f"Data root folder does not exist or is not a directory: {data_root}")

    # Keep a stable key order so we can zip keys<->values coming from itertools.product below.
    keys = list(param_grid)

    # # create a timestamped split folder
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # split_root = Path(f"{data_root}_{timestamp}")

    # print("== Gathering image files and building splits ...")
    # # train, val, test
    # counts = GroupCounts(4, 1, 1)

    # plans = build_split_plan_by_groups(
    #     data_root,
    #     groups=["hp","lenovo","mac","messanger","msi","zoom"],
    #     counts=counts,
    #     group_size=6,
    #     shuffle_within_group=True,
    #     seed=47,
    #     strict=False,
    #     allow_ungrouped=False,
    # )

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

    # Build (path, label_id) pairs and label mapping from the directory under `root`.
    # (If you enable the commented split-materialization block, you likely want to point this at `split_root` instead.)
    train_pairs, val_pairs, test_pairs, label_to_idx, idx_to_label = build_pairs_and_labels(str(data_root))

    print(f"\nLabel mapping (label_id -> label_name):")
    for i, label in enumerate(idx_to_label):
        print(f"  {i}: {label}")

    cnt = 1
    best_acc = -1.0
    best_model = ""

    # Cartesian product over all hyper-parameter value lists. Each iteration is one experiment run.
    for values in itertools.product(*param_grid.values()):
        hparams = dict(zip(keys, values))
        header = ", ".join(f"{k}={v}" for k, v in hparams.items())

        print("\n" + "=" * 80)
        print(">>> Training with", header)
        print("=" * 80)

        # NOTE: `data_root` passed below is not used inside `train_with_params()` (kept for compatibility).
        # Also note: `split_root` must be defined if you want to train on a freshly materialized split directory.
        results = train_with_params(
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
        # This is especially useful for dataset governance: you can later prove exactly which samples were evaluated.
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
        # Duplicate the checkpoint with an accuracy suffix so you don't have to grep logs to find the winner.
        shutil.copy(best_model, f"{best_model}_{best_acc:.6f}_BEST")


###############################################################################
# 11. Main
###############################################################################
if __name__ == "__main__":
    # Seed important RNGs for reproducibility.
    # This makes runs repeatable given identical hardware/software and single-worker DataLoaders.
    # For strict determinism on CUDA, you'd also set torch.backends.cudnn.deterministic/benchmark, etc.
    random.seed(47)
    np.random.seed(47)
    torch.manual_seed(47)

    # Note: 'pos' indicates the experiment run number. 
    # Since specific seeding per fold isn't used, this performs repeated random subsampling (Monte Carlo CV).
    # Repeat the whole grid twice. Because we don't reseed inside the loop, each run sees different shuffles
    # while remaining deterministic relative to the initial seed (useful for rough stability checks).
    for pos in range(2):
        print(f"\n\n### STARTING REPEATED EXPERIMENT RUN {pos+1}/2 ###")
        grid_search_experiments("data", pos)

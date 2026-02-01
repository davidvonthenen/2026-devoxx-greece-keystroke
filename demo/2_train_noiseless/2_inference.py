#!/usr/bin/env python3
"""
Minimal inference runner for the mel-spectrogram PNG classifier trained with the
CoAtNet-style MBConv + 2D relative-attention backbone.

Usage:
  python infer_melspec_model.py --checkpoint /path/to/model.pth --image /path/to/spec.png
  python infer_melspec_model.py --checkpoint model.pth --image spec.png --topk 5
  python infer_melspec_model.py --checkpoint model.pth --image /path/to/folder_of_pngs

Notes:
- Expects checkpoints produced by the provided training script (stores: model_state_dict,
  img_size, in_channels, hidden_size, num_classes, dropout, mean, std, idx_to_label).
- Preprocessing matches training: RGB, resize to img_size, normalize with saved mean/std.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Device selection (MPS -> CUDA -> CPU), with optional override
# -----------------------------------------------------------------------------
def select_device(device_str: Optional[str] = None) -> torch.device:
    if device_str:
        d = device_str.lower().strip()
        if d in ("cpu",):
            return torch.device("cpu")
        if d in ("cuda", "gpu"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if d in ("mps",):
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        return torch.device(device_str)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Image loading + normalization (matches training code)
# -----------------------------------------------------------------------------
DEFAULT_IMG_SIZE = 64

def _pil_load_rgb(path: Union[str, Path], img_size: int) -> torch.Tensor:
    """
    Load an image as float tensor in [0,1], shape (C,H,W) with C=3.
    """
    img = Image.open(str(path)).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
    return t

def _normalize(img: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean = np.asarray(mean, dtype=np.float32).reshape(3)
    std  = np.asarray(std,  dtype=np.float32).reshape(3)

    mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_t  = torch.tensor(std,  dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_t  = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
    return (img - mean_t) / std_t


# -----------------------------------------------------------------------------
# Model definition (inference-only)
# -----------------------------------------------------------------------------
def _choose_num_heads(dim: int) -> int:
    for h in (8, 4, 2, 1):
        if dim % h == 0 and (dim // h) >= 16:
            return h
    return 1

class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(ch * se_ratio))
        self.fc1 = nn.Conv2d(ch, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class MBConv(nn.Module):
    """
    Mobile inverted bottleneck with depthwise conv + SE.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expansion: int = 4,
        se_ratio: float = 0.25,
        drop: float = 0.0,
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden = in_ch * expansion

        self.expand = ConvBNAct(in_ch, hidden, k=1, s=1, p=0, act=True) if expansion != 1 else nn.Identity()
        self.dwconv = ConvBNAct(hidden, hidden, k=3, s=stride, p=1, groups=hidden, act=True)
        self.se = SqueezeExcite(hidden, se_ratio=se_ratio)
        self.project = ConvBNAct(hidden, out_ch, k=1, s=1, p=0, act=False)
        self.drop = nn.Dropout(p=drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        try:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        except TypeError:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flat = coords.flatten(1)  # (2,N)

        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2,N,N)
        rel = rel.permute(1, 2, 0).contiguous()  # (N,N,2)
        rel[:, :, 0] += H - 1
        rel[:, :, 1] += W - 1
        rel[:, :, 0] *= (2 * W - 1)
        rel_index = rel.sum(-1)  # (N,N)

        # Not saved in state_dict (persistent=False) in the training code.
        self.register_buffer("rel_index", rel_index, persistent=False)

        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.bias_table, std=0.02)
        else:
            nn.init.normal_(self.bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        N = self.H * self.W
        bias = self.bias_table[self.rel_index.view(-1)].view(N, N, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()  # (heads, N, N)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rel_pos_bias: Optional[RelativePositionBias] = None,
    ):
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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(
        self,
        dim: int,
        H: int,
        W: int,
        num_heads: int,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.H, self.W = H, W
        self.norm1 = nn.LayerNorm(dim)
        self.rpb = RelativePositionBias(H, W, num_heads)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop, rel_pos_bias=self.rpb)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.H or W != self.W:
            raise ValueError(f"TransformerBlock2D expected {(self.H, self.W)}, got {(H, W)}")
        t = x.flatten(2).transpose(1, 2)  # (B,N,C)
        t = t + self.attn(self.norm1(t))
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)

class CoAtNetTransformerClassifier(nn.Module):
    """
    CoAtNet-style image classifier tuned for mel-spectrogram images (e.g., 64x64).

    Structure:
      - 2 depthwise-convolutional stages (MBConv)
      - 2 global relative attention stages (TransformerBlock2D)
      - global average pool + linear head
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

        stem_dim = hidden_size
        conv2_dim = hidden_size * 2
        attn_dim = conv2_dim

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

        H1 = img_size // 8
        W1 = img_size // 8
        heads = _choose_num_heads(attn_dim)
        self.attn1 = nn.ModuleList([
            TransformerBlock2D(attn_dim, H1, W1, num_heads=heads, drop=dropout, attn_drop=dropout)
        ])

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

        # These are attached from the checkpoint for consistent inference:
        # self.mean, self.std, self.idx_to_label

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# -----------------------------------------------------------------------------
# Inference function (verified; small robustness tweaks for mean/std types)
# -----------------------------------------------------------------------------
def infer_single_image(img_path: Union[str, Path], model: nn.Module, device: torch.device) -> Tuple[str, float]:
    """
    Load a single spectrogram image, apply saved normalization, run model.

    Returns:
      (predicted_label_name, confidence)
    """
    img_size = int(getattr(model, "img_size", DEFAULT_IMG_SIZE))

    mean = getattr(model, "mean", None)
    std = getattr(model, "std", None)
    if mean is None or std is None:
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std  = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    else:
        mean = np.asarray(mean, dtype=np.float32)
        std  = np.asarray(std, dtype=np.float32)

    x = _pil_load_rgb(img_path, img_size)
    x = _normalize(x, mean, std).to(torch.float32)
    xb = x.unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[0]
        pred_i = int(torch.argmax(probs).item())

    idx_to_label = getattr(model, "idx_to_label", None)
    if isinstance(idx_to_label, (list, tuple)) and 0 <= pred_i < len(idx_to_label):
        pred_name = str(idx_to_label[pred_i])
    else:
        pred_name = str(pred_i)

    return pred_name, float(probs[pred_i].item())


def topk_from_logits(logits: torch.Tensor, k: int) -> List[Tuple[int, float]]:
    probs = torch.softmax(logits, dim=1)[0]
    k = min(k, probs.numel())
    vals, idxs = torch.topk(probs, k=k, largest=True, sorted=True)
    return [(int(i.item()), float(v.item())) for i, v in zip(idxs, vals)]


# -----------------------------------------------------------------------------
# Checkpoint loader
# -----------------------------------------------------------------------------
def _torch_load_compat(path: Union[str, Path], map_location: torch.device):
    """
    Compatibility shim: some torch versions don't accept weights_only=...
    """
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)

def load_model_from_checkpoint(ckpt_path: Union[str, Path], device: torch.device) -> CoAtNetTransformerClassifier:
    ckpt = _torch_load_compat(ckpt_path, map_location=device)

    img_size = int(ckpt.get("img_size", DEFAULT_IMG_SIZE) or DEFAULT_IMG_SIZE)
    in_channels = int(ckpt.get("in_channels", 3) or 3)
    hidden_size = ckpt.get("hidden_size", None)
    if hidden_size is None:
        raise ValueError("Checkpoint missing 'hidden_size' (required to rebuild the model).")

    dropout = float(ckpt.get("dropout", 0.1) if ckpt.get("dropout", None) is not None else 0.1)

    idx_to_label = ckpt.get("idx_to_label", None)
    num_classes = ckpt.get("num_classes", None)
    if num_classes is None:
        if isinstance(idx_to_label, (list, tuple)):
            num_classes = len(idx_to_label)
        else:
            raise ValueError("Checkpoint missing 'num_classes' and 'idx_to_label'; cannot infer class count.")

    model = CoAtNetTransformerClassifier(
        img_size=img_size,
        in_channels=in_channels,
        hidden_size=int(hidden_size),
        num_classes=int(num_classes),
        dropout=dropout,
    ).to(device)

    state = ckpt.get("model_state_dict", None)
    if state is None:
        raise ValueError("Checkpoint missing 'model_state_dict'.")

    model.load_state_dict(state, strict=True)

    # Attach normalization + label mapping for inference
    model.mean = ckpt.get("mean", None)
    model.std = ckpt.get("std", None)
    model.idx_to_label = idx_to_label

    return model


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def list_images(path: Union[str, Path]) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        out: List[Path] = []
        for child in sorted(p.iterdir()):
            if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                out.append(child)
        return out
    return []

def main() -> int:
    ap = argparse.ArgumentParser(description="Run inference for CoAtNet+Transformer mel-spectrogram classifier.")
    ap.add_argument("--checkpoint", "-c", default="1_27_best_w64_lr0.0003_bs32.pth", help="Path to .pth checkpoint produced by training script.")
    ap.add_argument("--image", "-i", default="s-09.png", help="Path to spectrogram image file OR directory of images.")
    ap.add_argument("--device", default=None, help="Override device: cpu | cuda | mps (or torch device string).")
    ap.add_argument("--topk", type=int, default=3, help="Print top-k predictions (default: 1).")
    args = ap.parse_args()

    device = select_device(args.device)
    print(f">> Device: {device}")

    model = load_model_from_checkpoint(args.checkpoint, device=device)

    img_paths = list_images(args.image)
    if not img_paths:
        raise FileNotFoundError(f"No image(s) found at: {args.image}")

    # Run inference
    for img_path in img_paths:
        img_size = int(getattr(model, "img_size", DEFAULT_IMG_SIZE))
        mean = getattr(model, "mean", np.array([0.5, 0.5, 0.5], dtype=np.float32))
        std  = getattr(model, "std",  np.array([0.25, 0.25, 0.25], dtype=np.float32))

        x = _pil_load_rgb(img_path, img_size)
        x = _normalize(x, np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)).to(torch.float32)
        xb = x.unsqueeze(0).to(device)

        model.eval()
        with torch.inference_mode():
            logits = model(xb)

        idx_to_label = getattr(model, "idx_to_label", None)
        topk = max(1, int(args.topk))

        if topk == 1:
            pred_name, conf = infer_single_image(img_path, model, device)
            print(f"{img_path.name} -> pred={pred_name}  conf={conf:.4f}")
        else:
            pairs = topk_from_logits(logits, k=topk)
            pretty = []
            for cls_i, prob in pairs:
                if isinstance(idx_to_label, (list, tuple)) and 0 <= cls_i < len(idx_to_label):
                    name = str(idx_to_label[cls_i])
                else:
                    name = str(cls_i)
                pretty.append(f"{name}:{prob:.4f}")
            print(f"{img_path.name} -> " + "  ".join(pretty))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

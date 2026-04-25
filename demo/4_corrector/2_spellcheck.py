#!/usr/bin/env python3
"""
infer_and_spellcheck_words.py

Pipeline:
1) Load checkpoint + run top-k inference on a list of spectrogram PNGs (each image = one keypress).
2) Treat 'space' as correct (hard delimiter):
   - If a position's top-1 predicted token == space_token, lock it to ONLY space_token.
3) Split the sequence into word segments separated by space.
4) For each word segment:
   - Enumerate all token combinations within the segment
   - Convert to a word string
   - Call API Ninjas spellcheck on the word (robust retries/backoff + cache)
   - If spellchecker found NO corrections (and API was healthy), print to console AND append to complete_words.txt
   - Keep top candidates:
       * Always preserve unique UNCORRECTED words by raw form (even if other typos correct to same word).
       * Then fill remaining slots with best corrected candidates (deduped by corrected form).
5) Combine word options across segments (spaces fixed) into full candidate phrases.
6) Write results to CSV.

API:
  GET https://api.api-ninjas.com/v1/spellcheck?text=...
  Header: X-Api-Key: <key>

Usage:
  export NINJAS_API_KEY="YOUR_KEY"
  python infer_and_spellcheck_words.py --checkpoint model.pth --topk 3 --lock-space-top1 --keep-per-word 25

Notes:
- If your "space" image isn't top-1 == "space", you can force-lock indices via --lock-space-indices.
- complete_words.txt is reset each run (unless you pass --append-complete-words).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

import requests


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
    img = Image.open(str(path)).convert("RGB")
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
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
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None,
                 groups: int = 1, act: bool = True):
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
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: int = 4,
                 se_ratio: float = 0.25, drop: float = 0.0):
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
        coords_flat = coords.flatten(1)

        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += H - 1
        rel[:, :, 1] += W - 1
        rel[:, :, 0] *= (2 * W - 1)
        rel_index = rel.sum(-1)

        self.register_buffer("rel_index", rel_index, persistent=False)

        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.bias_table, std=0.02)
        else:
            nn.init.normal_(self.bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        N = self.H * self.W
        bias = self.bias_table[self.rel_index.view(-1)].view(N, N, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 rel_pos_bias: Optional[RelativePositionBias] = None):
        super().__init__()
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
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
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
    def __init__(self, dim: int, H: int, W: int, num_heads: int,
                 drop: float = 0.0, attn_drop: float = 0.0, mlp_ratio: float = 4.0):
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
        t = x.flatten(2).transpose(1, 2)
        t = t + self.attn(self.norm1(t))
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)

class CoAtNetTransformerClassifier(nn.Module):
    def __init__(self, img_size: int, in_channels: int, hidden_size: int, num_classes: int, dropout: float = 0.1):
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

        self.attn1_down = ConvBNAct(conv2_dim, attn_dim, k=3, s=2, p=1, act=True)
        self.attn2_down = ConvBNAct(attn_dim,  attn_dim, k=3, s=2, p=1, act=True)

        H1 = img_size // 8
        W1 = img_size // 8
        heads = _choose_num_heads(attn_dim)
        self.attn1 = nn.ModuleList([TransformerBlock2D(attn_dim, H1, W1, num_heads=heads, drop=dropout, attn_drop=dropout)])

        H2 = img_size // 16
        W2 = img_size // 16
        self.attn2 = nn.ModuleList([TransformerBlock2D(attn_dim, H2, W2, num_heads=heads, drop=dropout, attn_drop=dropout)])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(attn_dim, num_classes),
        )

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
# Checkpoint loader
# -----------------------------------------------------------------------------
def _torch_load_compat(path: Union[str, Path], map_location: torch.device):
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

    model.mean = ckpt.get("mean", None)
    model.std = ckpt.get("std", None)
    model.idx_to_label = idx_to_label

    return model


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateToken:
    token: str
    prob: float

def topk_probs_for_image(img_path: Union[str, Path], model: nn.Module, device: torch.device, k: int) -> List[CandidateToken]:
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

    k = max(1, min(k, int(probs.numel())))
    vals, idxs = torch.topk(probs, k=k, largest=True, sorted=True)

    idx_to_label = getattr(model, "idx_to_label", None)
    out: List[CandidateToken] = []
    for cls_i, prob in zip(idxs.tolist(), vals.tolist()):
        if isinstance(idx_to_label, (list, tuple)) and 0 <= cls_i < len(idx_to_label):
            label = str(idx_to_label[cls_i])
        else:
            label = str(cls_i)
        out.append(CandidateToken(token=label, prob=float(prob)))
    return out


# -----------------------------------------------------------------------------
# Combination generation (streaming)
# -----------------------------------------------------------------------------
def iter_all_combinations(per_pos: Sequence[Sequence[CandidateToken]]) -> Iterator[Tuple[List[str], float]]:
    if not per_pos:
        return
    sizes = [len(x) for x in per_pos]
    if any(s == 0 for s in sizes):
        return

    idxs = [0] * len(per_pos)

    while True:
        toks: List[str] = []
        logp = 0.0
        for pos, i in enumerate(idxs):
            c = per_pos[pos][i]
            toks.append(c.token)
            logp += math.log(max(c.prob, 1e-12))
        yield toks, logp

        carry = len(idxs) - 1
        while carry >= 0:
            idxs[carry] += 1
            if idxs[carry] < sizes[carry]:
                break
            idxs[carry] = 0
            carry -= 1
        if carry < 0:
            return


# -----------------------------------------------------------------------------
# Robust spellcheck client with retry + cache
# -----------------------------------------------------------------------------
SPELLCHECK_URL = "https://api.api-ninjas.com/v1/spellcheck"

class SpellcheckClient:
    def __init__(
        self,
        api_key: str,
        timeout_s: float = 20.0,
        retries: int = 6,
        backoff_base: float = 0.5,
        backoff_max: float = 10.0,
        jitter: float = 0.25,
        cache_max: int = 200_000,
    ):
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self.jitter = float(jitter)
        self.cache_max = int(cache_max)
        self._cache: Dict[str, Dict] = {}

        self._session = requests.Session()
        self._session.headers.update({"X-Api-Key": self.api_key})

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        delay = delay * (1.0 + random.uniform(-self.jitter, self.jitter))
        if delay > 0:
            time.sleep(max(0.0, delay))

    def _cache_get(self, text: str) -> Optional[Dict]:
        return self._cache.get(text)

    def _cache_put(self, text: str, resp: Dict) -> None:
        if self.cache_max <= 0:
            return
        if len(self._cache) >= self.cache_max:
            for k in list(self._cache.keys())[: max(1, self.cache_max // 10)]:
                self._cache.pop(k, None)
        self._cache[text] = resp

    def spellcheck(self, text: str) -> Dict:
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        params = {"text": text}
        last_err: Optional[str] = None

        for attempt in range(self.retries + 1):
            try:
                r = self._session.get(SPELLCHECK_URL, params=params, timeout=self.timeout_s)

                if r.status_code == 401:
                    raise RuntimeError("401 Unauthorized: bad API key (X-Api-Key).")

                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    if attempt < self.retries:
                        self._sleep_backoff(attempt)
                        continue
                    break

                if not r.ok:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

                resp = r.json()
                self._cache_put(text, resp)
                return resp

            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    self._sleep_backoff(attempt)
                    continue
                break

        fallback = {"original": text, "corrected": text, "corrections": [], "error": last_err or "unknown"}
        self._cache_put(text, fallback)
        return fallback


# -----------------------------------------------------------------------------
# Complete word sink: prints + writes to file (deduped, thread-safe)
# -----------------------------------------------------------------------------
class CompleteWordSink:
    def __init__(self, path: Union[str, Path], append: bool = False):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seen: set[str] = set()

        mode = "a" if append else "w"
        self._fh = self.path.open(mode, encoding="utf-8", newline="\n")
        if append:
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if s:
                        self._seen.add(s)
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
            finally:
                self._fh.close()

    def write_word(self, word: str) -> bool:
        word = (word or "").strip()
        if not word:
            return False

        with self._lock:
            if word in self._seen:
                return False
            self._seen.add(word)

            print(f"[complete] {word}")
            self._fh.write(word + "\n")
            self._fh.flush()
            return True


# -----------------------------------------------------------------------------
# Word-level factoring using space as delimiter
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class WordOption:
    word_raw: str
    word_corrected: str
    log_prob: float
    num_corrections: int
    api_error: str = ""

def lock_space_positions(
    per_pos: List[List[CandidateToken]],
    space_token: str,
    lock_by_top1: bool,
    force_lock_indices: Sequence[int],
) -> List[List[CandidateToken]]:
    locked: List[List[CandidateToken]] = []
    force_set = set(int(i) for i in force_lock_indices)

    for i, cands in enumerate(per_pos):
        if i in force_set:
            p_space = next((c.prob for c in cands if c.token == space_token), cands[0].prob)
            locked.append([CandidateToken(space_token, float(p_space))])
            continue

        if lock_by_top1 and cands and cands[0].token == space_token:
            locked.append([CandidateToken(space_token, float(cands[0].prob))])
        else:
            locked.append(cands)
    return locked

def split_into_word_segments(
    per_pos: Sequence[Sequence[CandidateToken]],
    space_token: str,
) -> Tuple[List[List[List[CandidateToken]]], List[int]]:
    word_segments: List[List[List[CandidateToken]]] = []
    current: List[List[CandidateToken]] = []
    space_positions: List[int] = []

    for idx, cands in enumerate(per_pos):
        if len(cands) == 1 and cands[0].token == space_token:
            space_positions.append(idx)
            if current:
                word_segments.append(current)
                current = []
        else:
            current.append(list(cands))

    if current:
        word_segments.append(current)

    return word_segments, space_positions

def process_one_word_segment(
    client: SpellcheckClient,
    segment: List[List[CandidateToken]],
    keep_per_word: int,
    sleep_ms: int,
    workers: int,
    complete_sink: CompleteWordSink,
) -> List[WordOption]:
    """
    Updated logic:
    - Preserve UNCORRECTED words (n_corr==0, no API error, corrected==raw) as their own kept entries
      keyed by raw spelling.
    - Dedup corrected candidates by corrected spelling (like before).
    - Final kept list: all uncorrected (sorted by logp), then fill remaining slots with corrected.
    """
    combos = list(iter_all_combinations(segment))
    combos.sort(key=lambda x: x[1], reverse=True)

    best_uncorrected_by_raw: Dict[str, WordOption] = {}
    best_corrected_by_corrected: Dict[str, WordOption] = {}

    def do_one(tokens: List[str], logp: float) -> WordOption:
        word = "".join(tokens)
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        resp = client.spellcheck(word)
        corrected = str(resp.get("corrected", word))
        corrections = resp.get("corrections", []) or []
        n_corr = len(corrections)
        api_err = str(resp.get("error", "")) if "error" in resp else ""

        is_healthy = ("error" not in resp)
        is_uncorrected = is_healthy and (n_corr == 0) and (corrected == word)

        if is_uncorrected:
            complete_sink.write_word(corrected)

        return WordOption(
            word_raw=word,
            word_corrected=corrected,
            log_prob=logp,
            num_corrections=n_corr,
            api_error=api_err,
        )

    def consider(opt: WordOption) -> None:
        # Classify and keep best per key
        is_healthy = (opt.api_error == "")
        is_uncorrected = is_healthy and (opt.num_corrections == 0) and (opt.word_raw == opt.word_corrected)

        if is_uncorrected:
            key = opt.word_raw
            prev = best_uncorrected_by_raw.get(key)
            if prev is None or opt.log_prob > prev.log_prob:
                best_uncorrected_by_raw[key] = opt
        else:
            key = opt.word_corrected
            prev = best_corrected_by_corrected.get(key)
            if prev is None or opt.log_prob > prev.log_prob:
                best_corrected_by_corrected[key] = opt

    if workers <= 1:
        for tokens, logp in combos:
            consider(do_one(tokens, logp))
    else:
        workers = max(1, int(workers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(do_one, tokens, logp) for tokens, logp in combos]
            for fut in as_completed(futs):
                consider(fut.result())

    # Build final kept list: uncorrected first, then corrected fill
    keep_per_word = max(1, int(keep_per_word))

    uncorrected = list(best_uncorrected_by_raw.values())
    uncorrected.sort(key=lambda o: o.log_prob, reverse=True)

    corrected = list(best_corrected_by_corrected.values())
    corrected.sort(key=lambda o: o.log_prob, reverse=True)

    if len(uncorrected) >= keep_per_word:
        return uncorrected[:keep_per_word]

    remaining = keep_per_word - len(uncorrected)
    return uncorrected + corrected[:remaining]

def iter_phrase_combinations(
    word_options: Sequence[Sequence[WordOption]],
    space_char: str = " ",
    max_phrases: int = 0,
) -> Iterator[Tuple[str, str, float, int, str]]:
    if not word_options:
        return
    sizes = [len(w) for w in word_options]
    if any(s == 0 for s in sizes):
        return

    idxs = [0] * len(word_options)
    emitted = 0

    while True:
        raws: List[str] = []
        cors: List[str] = []
        lp = 0.0
        nc = 0
        errs: List[str] = []
        for j, i in enumerate(idxs):
            opt = word_options[j][i]
            raws.append(opt.word_raw)
            cors.append(opt.word_corrected)
            lp += opt.log_prob
            nc += opt.num_corrections
            if opt.api_error:
                errs.append(opt.api_error)

        yield space_char.join(raws), space_char.join(cors), lp, nc, " | ".join(errs)
        emitted += 1
        if max_phrases and emitted >= max_phrases:
            return

        carry = len(idxs) - 1
        while carry >= 0:
            idxs[carry] += 1
            if idxs[carry] < sizes[carry]:
                break
            idxs[carry] = 0
            carry -= 1
        if carry < 0:
            return


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Top-k key combos with space as fixed delimiter + robust spellcheck pruning.")
    ap.add_argument("--checkpoint", "-c", default="0_16_best_w32_lr0.0003_bs8.pth", help="Path to .pth checkpoint.")
    ap.add_argument("--device", default=None, help="Override device: cpu | cuda | mps (or torch device string).")
    ap.add_argument("--topk", type=int, default=2, help="Top-k predictions per position.")
    ap.add_argument("--min-prob", type=float, default=0.0, help="Drop per-position candidates below this probability.")
    ap.add_argument("--space-token", default="space", help="Token string to treat as space delimiter.")
    ap.add_argument("--lock-space-top1", action="store_true", default=True, help="Lock positions whose top-1 token == space-token to space only.")
    ap.add_argument("--lock-space-indices", nargs="*", default=[], help="Force-lock these indices as space (0-based).")
    ap.add_argument("--keep-per-word", type=int, default=25, help="Keep top-N candidates per word segment (uncorrected preserved).")
    ap.add_argument("--max-phrases", type=int, default=0, help="Cap final phrase combinations (0 = no cap).")
    ap.add_argument("--out", default="spellchecked_phrases.csv", help="Output CSV path.")
    ap.add_argument("--api-key", default=None, help="API Ninjas key (else uses env NINJAS_API_KEY).")
    ap.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel API calls within word segment processing.")
    ap.add_argument("--sleep-ms", type=int, default=0, help="Sleep between requests per worker.")
    ap.add_argument("--final-spellcheck", action="store_true", help="Also spellcheck the FULL corrected phrase (extra API calls).")

    # retry knobs
    ap.add_argument("--retries", type=int, default=6, help="Retries for transient HTTP failures.")
    ap.add_argument("--backoff-base", type=float, default=0.5, help="Exponential backoff base seconds.")
    ap.add_argument("--backoff-max", type=float, default=10.0, help="Max backoff seconds.")
    ap.add_argument("--jitter", type=float, default=0.25, help="Backoff jitter fraction.")
    ap.add_argument("--cache-max", type=int, default=200_000, help="In-memory cache size for spellcheck results.")

    # complete words output
    ap.add_argument("--complete-words-file", default="complete_words.txt", help="Where to write words with zero corrections.")
    ap.add_argument("--append-complete-words", action="store_true", help="Append to complete_words.txt instead of resetting.")

    ap.add_argument("--images", nargs="*", default=None, help="Override images list.")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("NINJAS_API_KEY")
    if not api_key:
        print("Missing API key. Provide --api-key or set NINJAS_API_KEY.", file=sys.stderr)
        return 2

    device = select_device(args.device)
    print(f">> Device: {device}")

    model = load_model_from_checkpoint(args.checkpoint, device=device)

    if args.images:
        images = [Path(p) for p in args.images]
    else:
        images = [
            Path("./examples/hlenovo5.png"),
            Path("./examples/elenovo6.png"),
            Path("./examples/llenovo3.png"),
            Path("./examples/llenovo3.png"),
            Path("./examples/olenovo3.png"),
            Path("./examples/spacelenovo1.png"),
            Path("./examples/glenovo4.png"),
            Path("./examples/rlenovo5.png"),
            Path("./examples/elenovo6.png"),
            Path("./examples/elenovo6.png"),
            Path("./examples/clenovo1.png"),
            Path("./examples/elenovo6.png"),
        ]

    missing = [p for p in images if not p.exists()]
    if missing:
        print("Missing image files:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        return 2

    topk = max(1, int(args.topk))
    min_prob = float(args.min_prob)

    # 1) Per-position candidates
    per_pos: List[List[CandidateToken]] = []
    for img in images:
        cands = topk_probs_for_image(img, model, device, k=topk)
        cands = [c for c in cands if c.prob >= min_prob] or cands[:1]
        per_pos.append(cands)

    # 2) Lock spaces (assumed correct)
    force_idx = [int(x) for x in args.lock_space_indices] if args.lock_space_indices else []
    per_pos = lock_space_positions(per_pos, args.space_token, bool(args.lock_space_top1), force_idx)

    print("\n>> Per-position candidates (after space-locking):")
    for i, (img, cands) in enumerate(zip(images, per_pos)):
        pretty = "  ".join([f"{c.token}:{c.prob:.4f}" for c in cands])
        print(f"  [{i:02d}] {img.name} -> {pretty}")

    # 3) Split into word segments
    word_segments, space_positions = split_into_word_segments(per_pos, space_token=args.space_token)
    print(f"\n>> Locked space positions: {space_positions}")
    print(f">> Word segments: {len(word_segments)}")

    # 4) Robust spellcheck client
    client = SpellcheckClient(
        api_key=api_key,
        timeout_s=float(args.timeout),
        retries=int(args.retries),
        backoff_base=float(args.backoff_base),
        backoff_max=float(args.backoff_max),
        jitter=float(args.jitter),
        cache_max=int(args.cache_max),
    )

    # 5) Complete word sink (prints + file)
    complete_sink = CompleteWordSink(
        path=args.complete_words_file,
        append=bool(args.append_complete_words),
    )
    print(f"\n>> Complete words will be written to: {Path(args.complete_words_file).resolve()}")

    try:
        # 6) Per-word pruning
        keep_per_word = max(1, int(args.keep_per_word))
        word_opts_all: List[List[WordOption]] = []

        for wi, seg in enumerate(word_segments):
            combos_est = 1
            for pos in seg:
                combos_est *= max(1, len(pos))
            print(f"\n>> Word {wi+1}/{len(word_segments)}: positions={len(seg)}  raw_combos={combos_est:,}")

            opts = process_one_word_segment(
                client=client,
                segment=seg,
                keep_per_word=keep_per_word,
                sleep_ms=max(0, int(args.sleep_ms)),
                workers=max(1, int(args.workers)),
                complete_sink=complete_sink,
            )

            print(f">> Kept word options: {len(opts)} (uncorrected preserved)")
            for o in opts[:10]:
                tag = f"  api_err={o.api_error}" if o.api_error else ""
                if o.word_raw != o.word_corrected:
                    print(f"    {o.word_raw} -> {o.word_corrected}  logp={o.log_prob:.3f}  corr={o.num_corrections}{tag}")
                else:
                    print(f"    {o.word_raw}  logp={o.log_prob:.3f}  corr={o.num_corrections}{tag}")

            word_opts_all.append(opts)

        # 7) Combine word options into phrases
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "rank",
                "raw_phrase",
                "corrected_phrase",
                "log_prob_sum",
                "prob_approx",
                "total_word_corrections",
                "spellcheck_errors",
                "final_corrected_phrase",
                "final_num_corrections",
            ])

            rank = 0
            for raw_phrase, corrected_phrase, logp_sum, total_corr, errs in iter_phrase_combinations(
                word_opts_all,
                space_char=" ",
                max_phrases=int(args.max_phrases),
            ):
                rank += 1
                prob_approx = math.exp(logp_sum) if logp_sum > -700 else 0.0

                final_corrected = ""
                final_ncorr = ""
                if args.final_spellcheck:
                    resp = client.spellcheck(corrected_phrase)
                    final_corrected = str(resp.get("corrected", corrected_phrase))
                    final_ncorr = str(len(resp.get("corrections", []) or []))

                w.writerow([
                    rank,
                    raw_phrase,
                    corrected_phrase,
                    f"{logp_sum:.8f}",
                    f"{prob_approx:.10e}",
                    total_corr,
                    errs,
                    final_corrected,
                    final_ncorr,
                ])

                if rank % 500 == 0:
                    f.flush()
                    print(f">> wrote {rank:,} phrases...", file=sys.stderr)

        print(f"\n>> Wrote CSV: {out_path.resolve()}")
        return 0

    finally:
        complete_sink.close()


if __name__ == "__main__":
    raise SystemExit(main())

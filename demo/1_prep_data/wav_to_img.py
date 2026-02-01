#!/usr/bin/env python3
"""
wav_to_melspectrogram_images.py

Recursively converts WAV files into mel-spectrogram images using
`librosa.feature.melspectrogram`.

The attached paper uses mel-spectrograms as a visual feature representation
and reports using 64 mel bands with an FFT window of 1024 samples and hop
lengths around 500/255 samples (depending on clip length), yielding square
images for downstream image models.

Key points
----------
- Preserves the input directory structure under an output root.
- Produces image files only (no CSV output).
- Designed for very short clips (< 1s), e.g., keystrokes.
- Includes Data Augmentation (Time-Shift + SpecAugment) as per [arXiv:2308.01074v1].

Reference (docs):
- https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html

Example
-------
  # Create Paper-accurate 64x64 images with augmentation enabled
  python wav_to_melspectrogram_images.py data --output images \
      --target-mels 64 --target-frames 64 --scale 1 --augment
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

import librosa
import librosa.display  # needed for "paper" rendering

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt


def iter_wavs(input_root: Path) -> Iterable[Path]:
    """Yield .wav files under input_root, recursively."""
    for root, _, files in os.walk(input_root):
        for fn in files:
            if fn.lower().endswith(".wav"):
                yield Path(root) / fn


def safe_power_to_db(S: np.ndarray, top_db: float) -> np.ndarray:
    """
    Convert a power spectrogram to dB robustly.

    librosa.power_to_db can produce -inf if the input is all zeros.
    For silent clips, return a constant floor at -top_db.
    """
    max_val = float(np.max(S)) if S.size else 0.0
    if max_val <= 0.0 or not np.isfinite(max_val):
        return np.full_like(S, -top_db, dtype=np.float32)
    return librosa.power_to_db(S, ref=max_val, top_db=top_db).astype(np.float32)


def augment_audio(y: np.ndarray, sr: int, max_shift_pct: float = 0.4) -> np.ndarray:
    """
    [Paper Cite: 151] "signals were time-shifted randomly by up to 40% in either direction."
    Performs a circular roll of the audio array.
    """
    if len(y) == 0:
        return y
    
    # Calculate max shift in samples
    max_shift_samples = int(len(y) * max_shift_pct)
    if max_shift_samples == 0:
        return y

    # Random integer between -shift and +shift
    shift_amt = random.randint(-max_shift_samples, max_shift_samples)
    return np.roll(y, shift_amt)


def augment_spectrogram(S_db: np.ndarray, mean_val: float, mask_pct: float = 0.1) -> np.ndarray:
    """
    [Paper Cite: 154, 155] "masking... taking a random 10% of both the time and frequency axis 
    and setting all values... to the mean" (SpecAugment).
    
    S_db shape: (n_mels, n_frames)
    """
    n_mels, n_frames = S_db.shape
    out = S_db.copy()

    # Frequency Masking
    n_freq_mask = int(n_mels * mask_pct)
    if n_freq_mask > 0:
        f0 = random.randint(0, n_mels - n_freq_mask)
        out[f0 : f0 + n_freq_mask, :] = mean_val

    # Time Masking
    n_time_mask = int(n_frames * mask_pct)
    if n_time_mask > 0:
        t0 = random.randint(0, n_frames - n_time_mask)
        out[:, t0 : t0 + n_time_mask] = mean_val

    return out


def compute_melspec_db(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
    power: float,
    top_db: float,
) -> np.ndarray:
    """
    Compute a log-mel spectrogram in dB.

    Returns
    -------
    S_db: np.ndarray [shape=(n_mels, n_frames)]
        Values in [-top_db, 0] (for non-silent input).
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )
    return safe_power_to_db(S, top_db=top_db)


def fix_spectrogram_size(
    S_db: np.ndarray,
    *,
    target_mels: Optional[int],
    target_frames: Optional[int],
    pad_value_db: float,
) -> np.ndarray:
    """
    Optionally pad/trim a spectrogram to a fixed size.

    Notes
    -----
    - This affects the *bin grid* (data density) along each axis.
    - If you want variable-sized images, leave targets as None.
    """
    out = S_db

    if target_mels is not None and target_mels > 0:
        # Pad/trim on the mel axis (axis=0)
        out = librosa.util.fix_length(out, size=target_mels, axis=0, constant_values=pad_value_db)

    if target_frames is not None and target_frames > 0:
        # Pad/trim on the time axis (axis=1)
        out = librosa.util.fix_length(out, size=target_frames, axis=1, constant_values=pad_value_db)

    return out


def render_matrix_style(
    S_db: np.ndarray,
    out_path: Path,
    *,
    top_db: float,
    cmap: str,
    scale: int,
) -> None:
    """
    Save the spectrogram as a "pure image matrix":
    - no axes
    - no colorbar
    - no titles

    The output pixel dimensions are:
      (n_mels * scale) x (n_frames * scale)

    This is typically what you want for ML pipelines.
    """
    # Normalize dB in [-top_db, 0] -> [0, 1]
    S_norm = (S_db + float(top_db)) / float(top_db)
    S_norm = np.clip(S_norm, 0.0, 1.0)

    # Upscale by integer factor (nearest-neighbor) to get higher pixel resolution.
    if scale > 1:
        S_norm = np.repeat(np.repeat(S_norm, scale, axis=0), scale, axis=1)

    # plt.imsave writes the array as an image without figure overhead.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, S_norm, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0)


def render_paper_style(
    y: np.ndarray,
    sr: int,
    S_db: np.ndarray,
    hop_length: int,
    out_path: Path,
    *,
    cmap: str,
    dpi: int,
    with_waveform: bool,
    title: Optional[str],
) -> None:
    """
    Save a mel-spectrogram in a "paper-like" style using librosa.display.specshow:
    - time axis in seconds
    - mel frequency axis
    - colorbar in dB
    - optional waveform panel on top

    This matches the typical visual style seen in academic figures.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if with_waveform:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), dpi=dpi, gridspec_kw={"height_ratios": [1, 2]}
        )
        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_xlabel("")
        ax1.set_ylabel("Amplitude")
        if title:
            ax1.set_title(title)

        # Mel-spectrogram
        img = librosa.display.specshow(
            S_db,
            x_axis="time",
            y_axis="mel",
            sr=sr,
            hop_length=hop_length,
            ax=ax2,
            cmap=cmap,
        )
        ax2.set_ylabel("Mel frequency")
        cbar = fig.colorbar(img, ax=ax2, format="%+2.0f dB")
        cbar.set_label("dB")
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=dpi)
        img = librosa.display.specshow(
            S_db,
            x_axis="time",
            y_axis="mel",
            sr=sr,
            hop_length=hop_length,
            ax=ax,
            cmap=cmap,
        )
        if title:
            ax.set_title(title)
        ax.set_ylabel("Mel frequency")
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.set_label("dB")
        fig.tight_layout()

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def wav_to_image(
    wav_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    sample_rate: Optional[int],
    preemphasis: float,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
    power: float,
    top_db: float,
    target_mels: Optional[int],
    target_frames: Optional[int],
    style: str,
    cmap: str,
    scale: int,
    dpi: int,
    with_waveform: bool,
    add_title: bool,
    overwrite: bool,
    augment: bool,
) -> Tuple[Path, bool]:
    """
    Convert one WAV file into an image.

    Returns
    -------
    (out_path, written)
    """
    rel = wav_path.relative_to(input_root)
    out_path = (output_root / rel).with_suffix(".png")

    if out_path.exists() and not overwrite:
        return out_path, False

    # Load audio (mono)
    y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)

    if y.size == 0:
        # Create an empty/silent placeholder image
        S_db = np.full((n_mels, 1), -top_db, dtype=np.float32)
    else:
        # 1. Augmentation: Time Shift [Cite: 151]
        if augment:
            y = augment_audio(y, sr)

        # Optional preemphasis (0 disables it)
        if preemphasis and preemphasis > 0.0:
            y = librosa.effects.preemphasis(y, coef=float(preemphasis))

        S_db = compute_melspec_db(
            y,
            sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            power=power,
            top_db=top_db,
        )

        # 2. Augmentation: SpecAugment / Masking [Cite: 154, 155]
        if augment:
            # Mask with the mean value of the current spectrogram
            mean_val = float(np.mean(S_db))
            S_db = augment_spectrogram(S_db, mean_val=mean_val)

    # Optional fixed sizing for dataset consistency [Cite: 153]
    if target_mels is not None or target_frames is not None:
        S_db = fix_spectrogram_size(
            S_db,
            target_mels=target_mels,
            target_frames=target_frames,
            pad_value_db=-top_db,
        )

    title = wav_path.name if add_title else None

    if style == "matrix":
        render_matrix_style(S_db, out_path, top_db=top_db, cmap=cmap, scale=scale)
    elif style == "paper":
        render_paper_style(
            y=y,
            sr=sr,
            S_db=S_db,
            hop_length=hop_length,
            out_path=out_path,
            cmap=cmap,
            dpi=dpi,
            with_waveform=with_waveform,
            title=title,
        )
    else:
        raise ValueError(f"Unknown style: {style}")

    return out_path, True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert WAV files into mel-spectrogram images (no CSV output)."
    )
    p.add_argument(
        "inputdir",
        nargs="?",
        default="data",
        help="Root directory to scan for .wav files (default: data).",
    )
    p.add_argument(
        "--output",
        default="melspec_images",
        help="Output root directory for images (default: melspec_images).",
    )

    # Audio / spectrogram parameters
    p.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="Target sample rate for loading audio. Use 0 to keep original SR (default: 48000).",
    )
    p.add_argument("--preemphasis", type=float, default=0.0, help="Preemphasis coef (0 disables it).")
    p.add_argument("--n-fft", type=int, default=1024, help="FFT window size (default: 1024).")
    p.add_argument("--hop-length", type=int, default=256, help="Hop length in samples (default: 256; Paper uses 255/500).")
    p.add_argument("--n-mels", type=int, default=64, help="Number of mel bands (default: 64).")
    p.add_argument("--fmin", type=float, default=0.0, help="Minimum frequency (Hz) (default: 0).")
    p.add_argument(
        "--fmax",
        type=float,
        default=0.0,
        help="Maximum frequency (Hz). Use 0 to set to Nyquist (default: 0).",
    )
    p.add_argument("--power", type=float, default=2.0, help="Power for melspectrogram (default: 2.0).")
    p.add_argument(
        "--top-db",
        type=float,
        default=80.0,
        help="Dynamic range for dB scaling (default: 80).",
    )

    # Optional fixed sizing (useful if you want square/consistent tensors)
    p.add_argument(
        "--target-mels",
        type=int,
        default=0,
        help="If >0, pad/trim spectrogram mel axis to this size (default: 0 = off).",
    )
    p.add_argument(
        "--target-frames",
        type=int,
        default=0,
        help="If >0, pad/trim spectrogram time axis to this size (default: 0 = off).",
    )

    # Rendering
    p.add_argument(
        "--style",
        choices=("matrix", "paper"),
        default="matrix",
        help="Image style: 'matrix' (no axes) or 'paper' (axes + colorbar) (default: matrix).",
    )
    p.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap name (default: magma).",
    )
    p.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Integer upscaling factor for 'matrix' style images (default: 1). Use 1 for ML inputs.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for 'paper' style rendering (default: 300).",
    )
    p.add_argument(
        "--with-waveform",
        action="store_true",
        help="(paper style only) Include waveform panel above the mel-spectrogram.",
    )
    p.add_argument(
        "--title",
        action="store_true",
        help="Add filename as the plot title (paper style only).",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images if they already exist.",
    )
    
    # Augmentation
    p.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation (time shift + masking) as per the research paper.",
    )

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    input_root = Path(args.inputdir).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not input_root.is_dir():
        raise SystemExit(f"Input directory not found: {input_root}")

    sample_rate: Optional[int] = args.sr if args.sr and args.sr > 0 else None
    fmax: Optional[float] = args.fmax if args.fmax and args.fmax > 0 else None

    target_mels = args.target_mels if args.target_mels and args.target_mels > 0 else None
    target_frames = args.target_frames if args.target_frames and args.target_frames > 0 else None

    wavs = list(iter_wavs(input_root))
    if not wavs:
        print(f"No .wav files found under: {input_root}")
        return 0

    written_count = 0
    skipped_count = 0

    for wav_path in wavs:
        out_path, written = wav_to_image(
            wav_path,
            input_root=input_root,
            output_root=output_root,
            sample_rate=sample_rate,
            preemphasis=args.preemphasis,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            fmin=args.fmin,
            fmax=fmax,
            power=args.power,
            top_db=args.top_db,
            target_mels=target_mels,
            target_frames=target_frames,
            style=args.style,
            cmap=args.cmap,
            scale=max(1, int(args.scale)),
            dpi=int(args.dpi),
            with_waveform=bool(args.with_waveform),
            add_title=bool(args.title),
            overwrite=bool(args.overwrite),
            augment=bool(args.augment),
        )

        if written:
            written_count += 1
            print(f"[saved]  {out_path}")
        else:
            skipped_count += 1
            print(f"[skip]   {out_path} (exists)")

    print(f"\nDone. Wrote {written_count} image(s), skipped {skipped_count}.")
    print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

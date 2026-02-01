#!/usr/bin/env python3
"""
scan_wavs.py

Recursively scan a root folder for WAV files and report:
  1) Any files whose sampling rate != expected (default 48000 Hz)
  2) Any files whose channel count != expected (default 1)
  3) The longest WAV file by duration (centisecond precision)

Behavior:
  - Prints full file paths of mismatching WAV files to STDOUT.
  - Prints a short summary (including the longest file) to STDERR.

Dependencies:
  - librosa (required)
  - soundfile (optional, but recommended). Falls back to Python's wave module.

Usage:
  python scan_wavs.py /path/to/root
  python scan_wavs.py /path/to/root --sr 48000 --channels 1
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path
from typing import Iterator, Optional, Tuple

import librosa

try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None


def iter_wav_files(root: Path) -> Iterator[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".wav"):
                yield Path(dirpath) / name


def read_wav_metadata(path: Path) -> Tuple[int, int, int]:
    """
    Return (samplerate_hz, channels, frames).

    Prefer soundfile (handles more WAV variants). Fall back to wave for PCM WAVs.
    """
    if sf is not None:
        info = sf.info(str(path))
        return int(info.samplerate), int(info.channels), int(info.frames)

    with wave.open(str(path), "rb") as wf:
        return int(wf.getframerate()), int(wf.getnchannels()), int(wf.getnframes())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively scan WAV files for sampling rate / channel mismatches and report the longest file."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="data",
        help="Root folder to scan (default: current directory).",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Expected sampling rate in Hz (default: 44100).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Expected number of channels (default: 2).",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    expected_sr = int(args.sr)
    expected_ch = int(args.channels)

    total = 0
    mismatches = 0

    longest_path: Optional[Path] = None
    longest_duration_s: float = -1.0

    for wav_path in iter_wav_files(root):
        total += 1

        # Use librosa for samplerate (fast header probe).
        try:
            sr_librosa = int(librosa.get_samplerate(str(wav_path)))
        except Exception as e:
            print(f"ERROR reading samplerate via librosa: {wav_path} ({e})", file=sys.stderr)
            continue

        # Use metadata for channels/frames (and samplerate cross-check).
        try:
            sr_meta, channels, frames = read_wav_metadata(wav_path)
        except Exception as e:
            print(f"ERROR reading WAV metadata: {wav_path} ({e})", file=sys.stderr)
            continue

        # If they disagree, prefer container metadata for checks/duration.
        sr = sr_meta if sr_meta != sr_librosa else sr_librosa

        duration_s = (frames / float(sr)) if sr > 0 else 0.0
        if duration_s > longest_duration_s:
            longest_duration_s = duration_s
            longest_path = wav_path

        if sr != expected_sr or channels != expected_ch:
            mismatches += 1
            print(str(wav_path))

    if total == 0:
        print(f"No .wav files found under: {root}", file=sys.stderr)
        return 2

    print(f"\nScanned: {total} WAV files", file=sys.stderr)
    print(f"Mismatches (sr != {expected_sr} or channels != {expected_ch}): {mismatches}", file=sys.stderr)

    if longest_path is not None:
        print("\nLongest WAV file:", file=sys.stderr)
        print(str(longest_path), file=sys.stderr)
        print(f"Duration: {longest_duration_s:.2f} seconds", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

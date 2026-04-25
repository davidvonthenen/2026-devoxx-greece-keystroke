# Performs Data Preparation for a Dataset of Your Choice (Single or Multiple Keyboards)

This project performs data preparation for a dataset (single or multiple keyboards).

## Prerequisites

- Python 3.12+
- An H100 (or better)

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

## Usage

### Step 1: Data Prep

Download the dataset. Find links in the `DOWNLOAD_DATASET_README.txt`.

### Step 2: Convert WAV Files to Spectrographic Images

```bash
python wav_to_img.py --output images --target-mels 64 --target-frames 64 --hop-length 255 --scale 1
```

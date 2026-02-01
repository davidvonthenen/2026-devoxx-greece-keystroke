# Train A Multi Keyboard Acoustic Classifier

This project trains a acoustic keyboard classifier for multiple keyboards.

## Prerequisites

- Python 3.12+
- An H100 (or better)

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

## Usage

### Step 1: Get Your Spectrographic Images From the Previous Step

If you haven't converted the WAV files from the single keyboard dataset, you can download this already converted dataset. Please see `DOWNLOAD_DATASET_README.txt` for more details.

Running the following command below, will train our acoustic classifier model for a single keyboard.

```bash
python 1_train.py
```

### Step 2: Run Inference On Your Model

**NOTE:** If you don't have access to that kind of hardware, download this prebuild model from `DOWNLOAD_MODEL_README.txt`.

To run inference, run the following command:

```bash
python 2_inference.py
```

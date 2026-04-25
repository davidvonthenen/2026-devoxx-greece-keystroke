# Correct The Incorrect Cliassification

This attempt to demonstrate an imperfect inference on keyboard keystroke identification and trying to correct it.

## Prerequisites

- Python 3.12+
- An H100 (or better)

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

## Usage

### Step 1: Run Inference On Your Model

**NOTE:** If you don't have access to that kind of hardware, download this prebuild model from `DOWNLOAD_MODEL_README.txt`.

This attempt to demonstrate an imperfect inference on keyboard keystroke identification. To run our demo, run the following command:

```bash
python 1_inference.py
```

### Step 2: 

Let's simulate correcting the error by running the following script:

```bash
python 2_spellcheck.py
```

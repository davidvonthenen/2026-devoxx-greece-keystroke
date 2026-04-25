# Devoxx Greece 2026: The Sound of Your Secrets: Teaching Your Model to Spy, So You Can Learn to Defend

Welcome to the landing page for the session `The Sound of Your Secrets: Teaching Your Model to Spy, So You Can Learn to Defend` at `Devoxx Greece 2026`.

## What to Expect

This repo intends to provide an introduction to:

- Perform the data prep for either the single or multiple keyboard dataset
- Build out a classifier model for single keyboard sounds
- Build out a classifier model for multiple keyboard sounds
- Perform "data correction"

## Hardware Prerequisites

The training for either the single or multiple keyboard projects will require a GPU (and of the H100 or better variety). There is simply no getting around that.

If you don't have access to this kind of hardware, you can at least download the pre-built models for inference.

## Software Prerequisites

- A Linux or Mac-based Developer’s Laptop 
  - Windows Users should use a VM or Cloud Instance
- Python Installed: version 3.12 or higher
- (Recommended) Using a miniconda or venv virtual environment
- Basic familiarity with shell operations

## Participation Options

There are 4 projects to make this happen. Step 2 and 3 are dependent on Step 1.

- [demo/1_prep_data](./demos/1_prep_data/README.md)
- [demo/2_train_noiseless](./demos/2_train_noiseless/README.md)
- [demo/3_train_mka-dependent](./demos/3_train_mka-dependent/README.md)
- [demo/4_corrector](./demos/4_corrector/README.md)
- [demo 5: Masked Language Model](https://github.com/davidvonthenen/2026-scale-23x-slm/tree/main/demo/3_MLM)

The instructions and purpose for each demo is contained within their respective folders.

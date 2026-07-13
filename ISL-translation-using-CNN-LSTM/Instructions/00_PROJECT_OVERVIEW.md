# ISL Translator — Project Overview

## Goal
Build a word-level Indian Sign Language (ISL) recognition system that takes a video clip of a sign and outputs the corresponding English word/gloss. Optionally extend to sentence-level translation.

## Environment
- **Platform:** Google Colab (T4 GPU, 15GB VRAM)
- **Storage:** Google Drive (mount at `/content/drive/MyDrive/isl_project/`)
- **Framework:** PyTorch with fp16 AMP
- **Python:** 3.10+

## Repository Structure to Create
```
isl_project/
├── data/
│   ├── raw/                    # Original video files (on Drive)
│   ├── keypoints/              # Extracted .npy keypoint files (on Drive)
│   └── splits/                 # train.txt, val.txt, test.txt
├── src/
│   ├── extract.py              # MediaPipe keypoint extraction
│   ├── dataset.py              # PyTorch Dataset class
│   ├── augment.py              # Augmentation functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bilstm.py           # Baseline BiLSTM model
│   │   └── spoter.py           # SPOTER Transformer model
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation script
│   └── utils.py                # Helpers (label maps, checkpointing)
├── configs/
│   ├── bilstm.yaml             # BiLSTM hyperparameters
│   └── spoter.yaml             # SPOTER hyperparameters
├── notebooks/
│   ├── 01_extraction.ipynb     # Run keypoint extraction
│   ├── 02_train.ipynb          # Training notebook
│   └── 03_inference.ipynb      # Demo inference
├── checkpoints/                # Saved model weights (on Drive)
├── logs/                       # Training logs
└── requirements.txt
```

## Document Map
Read these docs in order:
1. `00_PROJECT_OVERVIEW.md` — this file
2. `01_DATASET.md` — dataset acquisition and structure
3. `02_PREPROCESSING.md` — keypoint extraction pipeline
4. `03_MODELS.md` — all model architectures with full code
5. `04_TRAINING.md` — training loop, configs, logging
6. `05_EVALUATION.md` — metrics and evaluation scripts
7. `06_NOTEBOOKS.md` — Colab notebook specifications
8. `07_EXTENSION_SLT.md` — how to extend to sentence-level

## Execution Order
1. Read ALL docs before writing any code
2. Create directory structure
3. Implement `extract.py` and run extraction once → save to Drive
4. Implement `dataset.py` + `augment.py`
5. Implement and train BiLSTM baseline
6. Implement and train SPOTER
7. Evaluate, compare, iterate

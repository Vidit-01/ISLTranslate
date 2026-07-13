# AGENT INSTRUCTIONS

You are implementing an Indian Sign Language (ISL) word/sentence-level recognition system. Read all docs completely before writing any code.

## Your Task

Build the complete ISL translation pipeline as specified across the documentation files in this directory.

## Reading Order (mandatory)

1. `00_PROJECT_OVERVIEW.md` — project structure, file map
2. `01_DATASET.md` — dataset acquisition, label maps, splits
3. `02_PREPROCESSING.md` — MediaPipe extraction, augmentation, normalization
4. `03_MODELS.md` — BiLSTM, SPOTER, SLTModel implementations + Dataset class
5. `04_TRAINING.md` — configs, training loop
6. `05_EVALUATION.md` — evaluation script, inference utils
7. `06_NOTEBOOKS.md` — Colab notebook specifications
8. `07_EXTENSION_SLT.md` — sentence-level extension (implement last)

## Implementation Rules

1. **Read all docs first.** Do not write a single line of code until you have read all 8 docs.
2. **Implement in order:** directory structure → extract.py → augment.py → dataset.py → models/ → train.py → evaluate.py → notebooks
3. **Do not modify the model architectures** specified in `03_MODELS.md` unless there is a concrete error (shape mismatch, etc.)
4. **All paths use the Drive root:** `/content/drive/MyDrive/isl_project/` — never hardcode local paths
5. **No print statements** inside training loops except via tqdm descriptions
6. **All files go in `src/`** unless they are configs (→ `configs/`) or notebooks (→ `notebooks/`)
7. **Code must be resume-safe:** extraction skips existing files, training loads from latest checkpoint if it exists

## Error Handling Rules

- If MediaPipe fails on a frame: zero-fill that frame's keypoints (already handled in `extract.py`)
- If a video file fails entirely: log to `failed.txt`, continue — never crash the extraction loop
- If CUDA OOM: reduce batch size in the config, do not change model architecture
- Surface all other errors loudly — no silent fallbacks

## Validation Checkpoints

After each major step, verify:

| Step | Check |
|---|---|
| After extraction | `verify_keypoints()` reports 0 broken files |
| After dataset | `ds[0]` returns shape `(64, 543)` tensor |
| After BiLSTM 1 epoch | Val loss decreasing, no NaN |
| After SPOTER 1 epoch | Val loss ≤ BiLSTM val loss |
| After 50 epochs | BiLSTM val acc ≥ 30% (sanity) |
| After full training | SPOTER val acc ≥ 65% |

## What NOT to Implement

- Do not implement a real-time webcam loop (Colab doesn't support it cleanly)
- Do not implement beam search decoder (greedy is sufficient for now)
- Do not implement the SLT model until word-level is validated
- Do not use `torchvision` video loaders — all video loading is through OpenCV in `extract.py`

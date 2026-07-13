# Dataset

## Primary recommended Dataset: ISL Words with Landmarks (Kaggle)

This is a processed version of the INCLUDE dataset, containing 80 classes of common ISL words. It is pre-processed for ViViT and includes MediaPipe landmarks.

- **URL:** [Kaggle: Indian Sign Language words with Landmarks](https://www.kaggle.com/datasets/kaushikyh/indian-sign-language-words-with-landmarks)
- **Method:** Download via Kaggle CLI or direct download.
- **Structure:** `ProcessedData_vivit/{word}/{video}.mp4`
- **Features:** 32 frames per video, 224x224 RGB, includes hand and pose landmarks.

## Alternative Dataset: INCLUDE (IIT Bombay)

The full original dataset.

- **URL:** https://iitbacin.github.io/INCLUDE-dataset/
- **Method:** Fill request form, access via Drive link (~2-3 days).
- **Size:** ~4500 videos, 263 classes.

## Fallback Dataset: WLASL

WLASL is publicly available and has an identical pipeline. Use it to build and validate the full pipeline while waiting for INCLUDE.

- **URL:** https://github.com/dxli94/WLASL
- **Size:** 21,000+ videos, 2000 ASL word classes
- **Download:** Videos via provided download script in repo
- **Use:** Pretrain on WLASL → fine-tune on INCLUDE (transfer learning)

## Label Map

Build `label_map.json` mapping class name → integer index:

```python
import os, json

def build_label_map(video_dir):
    classes = sorted(os.listdir(video_dir))  # alphabetical for reproducibility
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    with open("data/splits/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    return label_map
```

## Train/Val/Test Splits

Split per class (not randomly across all videos) to avoid signer leakage if possible.

```python
import random, os

def create_splits(video_dir, label_map, train_ratio=0.7, val_ratio=0.15):
    train, val, test = [], [], []
    for label in os.listdir(video_dir):
        if label not in label_map:
            continue
        videos = [f"{label}/{v}" for v in os.listdir(f"{video_dir}/{label}")]
        random.shuffle(videos)
        n = len(videos)
        t = int(n * train_ratio)
        v = int(n * val_ratio)
        train.extend([(p, label_map[label]) for p in videos[:t]])
        val.extend([(p, label_map[label]) for p in videos[t:t+v]])
        test.extend([(p, label_map[label]) for p in videos[t+v:]])

    def write_split(split, path):
        with open(path, "w") as f:
            for path_, label in split:
                f.write(f"{path_} {label}\n")

    write_split(train, "data/splits/train.txt")
    write_split(val,   "data/splits/val.txt")
    write_split(test,  "data/splits/test.txt")
```

## requirements.txt

```
mediapipe==0.10.21
opencv-python-headless==4.11.0.86
scipy==1.13.0
pyyaml>=6.0
tqdm>=4.65.0
scikit-learn>=1.3.0
seaborn>=0.12.0
sacrebleu>=2.3.0          # SLT only
```

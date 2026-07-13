# Colab Notebooks

## Setup (Run at Top of Every Notebook)

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone/sync repo
import os
PROJECT_ROOT = '/content/drive/MyDrive/isl_project'
os.makedirs(PROJECT_ROOT, exist_ok=True)

# Add src to path
import sys
sys.path.insert(0, f'{PROJECT_ROOT}/src')

# Install deps
!pip install mediapipe opencv-python-headless scipy tqdm pyyaml scikit-learn seaborn -q
```

---

## `notebooks/01_extraction.ipynb`

**Purpose:** One-time keypoint extraction. Run once, save to Drive.

```python
# Cell 1: Setup (see above)

# Cell 2: Verify GPU (extraction itself is CPU-bound, but good to check)
import torch
print(torch.cuda.is_available())

# Cell 3: Build label map
import os, json
VIDEO_DIR = f'{PROJECT_ROOT}/data/raw/INCLUDE/Videos'
SPLITS_DIR = f'{PROJECT_ROOT}/data/splits'
os.makedirs(SPLITS_DIR, exist_ok=True)

from utils import build_label_map  # implement in utils.py
label_map = build_label_map(VIDEO_DIR)
print(f"Classes: {len(label_map)}")

# Cell 4: Create splits
from dataset import create_splits
create_splits(VIDEO_DIR, label_map)

# Cell 5: Extract keypoints (will take ~2 hours for INCLUDE)
from extract import preprocess_dataset
NPY_DIR = f'{PROJECT_ROOT}/data/keypoints'

preprocess_dataset(
    video_dir=VIDEO_DIR,
    save_dir=NPY_DIR,
    label_map=label_map
)

# Cell 6: Verify extraction
from extract import verify_keypoints
verify_keypoints(NPY_DIR)

# Cell 7: Compute normalization stats
import numpy as np, os
from extract import compute_stats

train_files = open(f'{SPLITS_DIR}/train.txt').read().splitlines()
mean, std = compute_stats(NPY_DIR, train_files)
print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
print(f"Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
```

---

## `notebooks/02_train.ipynb`

**Purpose:** Train BiLSTM then SPOTER, save checkpoints.

```python
# Cell 1: Setup

# Cell 2: Quick dataset sanity check
from dataset import ISLDataset
from torch.utils.data import DataLoader

SPLITS_DIR = f'{PROJECT_ROOT}/data/splits'
NPY_DIR = f'{PROJECT_ROOT}/data/keypoints'

ds = ISLDataset(
    npy_dir=NPY_DIR,
    split_file=f'{SPLITS_DIR}/train.txt',
    mean_path=f'{SPLITS_DIR}/mean.npy',
    std_path=f'{SPLITS_DIR}/std.npy',
    augment=True
)
print(f"Train samples: {len(ds)}")
x, y = ds[0]
print(f"Sample shape: {x.shape}, label: {y}")  # expect (64, 543), int

# Cell 3: Train BiLSTM
from train import train
train(f'{PROJECT_ROOT}/configs/bilstm.yaml')

# Cell 4: Train SPOTER
train(f'{PROJECT_ROOT}/configs/spoter.yaml')

# Cell 5: Monitor training (run periodically)
import json, matplotlib.pyplot as plt

def plot_log(log_path, title):
    with open(log_path) as f:
        log = json.load(f)
    epochs = [e['epoch'] for e in log]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [e['train_loss'] for e in log], label='train')
    plt.plot(epochs, [e['val_loss'] for e in log], label='val')
    plt.legend(); plt.title(f'{title} Loss')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [e['val_acc'] for e in log], label='val top-1')
    plt.plot(epochs, [e['val_top5'] for e in log], label='val top-5')
    plt.legend(); plt.title(f'{title} Accuracy')
    plt.tight_layout(); plt.show()

plot_log(f'{PROJECT_ROOT}/checkpoints/bilstm/log.json', 'BiLSTM')
```

---

## `notebooks/03_inference.ipynb`

**Purpose:** Demo inference on a single video or webcam stream.

```python
# Cell 1: Setup

# Cell 2: Load model
import torch, json, numpy as np
from models.spoter import SPOTER
from utils import predict_video

SPLITS_DIR = f'{PROJECT_ROOT}/data/splits'
CKPT = f'{PROJECT_ROOT}/checkpoints/spoter/best.pt'
CONFIG = f'{PROJECT_ROOT}/configs/spoter.yaml'

import yaml
with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SPOTER(
    input_dim=cfg['input_dim'], d_model=cfg['d_model'],
    nhead=cfg['nhead'], num_encoder_layers=cfg['num_encoder_layers'],
    dim_feedforward=cfg['dim_feedforward'],
    num_classes=cfg['num_classes'], dropout=0.0
)
ckpt = torch.load(CKPT, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device).eval()

mean = np.load(f'{SPLITS_DIR}/mean.npy')
std = np.load(f'{SPLITS_DIR}/std.npy')

with open(f'{SPLITS_DIR}/label_map.json') as f:
    label_map = json.load(f)
label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

# Cell 3: Predict on a video file
VIDEO_PATH = '/content/test_sign.mp4'  # upload your video
results = predict_video(model, VIDEO_PATH, mean, std, label_names, device, top_k=5)
for rank, (label, conf) in enumerate(results, 1):
    print(f"  {rank}. {label}: {conf*100:.1f}%")

# Cell 4: Upload and test from Colab file picker
from google.colab import files
uploaded = files.upload()
for fname in uploaded:
    results = predict_video(model, fname, mean, std, label_names, device)
    print(f"\nPredictions for {fname}:")
    for r, (l, c) in enumerate(results, 1):
        print(f"  {r}. {l}: {c*100:.1f}%")
```

---

## Colab Tips

**Avoid session timeouts during extraction:**
```python
# Run in a cell to prevent Colab from disconnecting
import IPython
display(IPython.display.Javascript('''
  function ClickConnect(){
    document.querySelector("colab-toolbar-button#connect").click()
  }
  setInterval(ClickConnect, 60000)
'''))
```

**Save checkpoint manually if runtime is about to die:**
```python
# Emergency checkpoint save
torch.save(model.state_dict(), f'{PROJECT_ROOT}/checkpoints/emergency.pt')
```

**Resume training from checkpoint:**
```python
ckpt = torch.load(f'{PROJECT_ROOT}/checkpoints/spoter/epoch_50.pt')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt['epoch'] + 1
# Then modify train() to accept start_epoch argument
```

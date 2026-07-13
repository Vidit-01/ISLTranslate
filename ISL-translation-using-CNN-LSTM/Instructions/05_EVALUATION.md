# Evaluation & Inference

## `src/evaluate.py` — Full Implementation

```python
import os
import json
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset import ISLDataset
from models.bilstm import BiLSTMClassifier
from models.spoter import SPOTER


def load_model(cfg, checkpoint_path, device):
    if cfg['model'] == 'bilstm':
        model = BiLSTMClassifier(
            input_dim=cfg['input_dim'], hidden=cfg['hidden'],
            num_layers=cfg['num_layers'], num_classes=cfg['num_classes'],
            dropout=0.0  # No dropout at eval
        )
    else:
        model = SPOTER(
            input_dim=cfg['input_dim'], d_model=cfg['d_model'],
            nhead=cfg['nhead'], num_encoder_layers=cfg['num_encoder_layers'],
            dim_feedforward=cfg['dim_feedforward'],
            num_classes=cfg['num_classes'], dropout=0.0
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


@torch.no_grad()
def run_evaluation(model, loader, device, label_names=None):
    all_preds, all_labels, all_probs = [], [], []

    for x, y in tqdm(loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    top1 = (all_preds == all_labels).mean()

    # Top-5
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5 = np.array([all_labels[i] in top5_preds[i] for i in range(len(all_labels))]).mean()

    print(f"Top-1 Accuracy: {top1:.4f} ({top1*100:.1f}%)")
    print(f"Top-5 Accuracy: {top5:.4f} ({top5*100:.1f}%)")
    print()

    if label_names:
        print(classification_report(all_labels, all_preds, target_names=label_names))

    return all_preds, all_labels, all_probs, top1, top5


def plot_confusion_matrix(all_labels, all_preds, label_names, save_path, top_n=30):
    """Plot confusion matrix for top_n most confused classes."""
    cm = confusion_matrix(all_labels, all_preds)

    # Select top_n classes by error count
    errors = cm.sum(axis=1) - cm.diagonal()
    top_classes = np.argsort(errors)[-top_n:]
    cm_sub = cm[np.ix_(top_classes, top_classes)]
    names_sub = [label_names[i] for i in top_classes]

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_sub, xticklabels=names_sub, yticklabels=names_sub,
                fmt='d', cmap='Blues', annot=len(top_classes) <= 20)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Top {top_n} most confused classes)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(log_path, save_path):
    with open(log_path) as f:
        log = json.load(f)

    epochs = [e['epoch'] for e in log]
    train_acc = [e['train_acc'] for e in log]
    val_acc = [e['val_acc'] for e in log]
    val_top5 = [e['val_top5'] for e in log]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [e['train_loss'] for e in log], label='Train')
    axes[0].plot(epochs, [e['val_loss'] for e in log], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Loss')

    axes[1].plot(epochs, train_acc, label='Train Top-1')
    axes[1].plot(epochs, val_acc, label='Val Top-1')
    axes[1].plot(epochs, val_top5, label='Val Top-5', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Accuracy')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def evaluate_cli(config_path, checkpoint_path, split='test'):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_cfg = cfg['data']
    splits_dir = data_cfg['splits_dir']

    ds = ISLDataset(
        npy_dir=data_cfg['npy_dir'],
        split_file=os.path.join(splits_dir, f'{split}.txt'),
        mean_path=os.path.join(splits_dir, 'mean.npy'),
        std_path=os.path.join(splits_dir, 'std.npy'),
        augment=False
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    with open(os.path.join(splits_dir, 'label_map.json')) as f:
        label_map = json.load(f)
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

    model = load_model(cfg, checkpoint_path, device)
    preds, labels, probs, top1, top5 = run_evaluation(model, loader, device, label_names)

    ckpt_dir = cfg['checkpoint']['dir']
    plot_confusion_matrix(labels, preds, label_names,
                          os.path.join(ckpt_dir, 'confusion_matrix.png'))
    log_path = os.path.join(ckpt_dir, 'log.json')
    if os.path.exists(log_path):
        plot_training_curves(log_path, os.path.join(ckpt_dir, 'curves.png'))

    results = {'top1': float(top1), 'top5': float(top5), 'split': split}
    with open(os.path.join(ckpt_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='test')
    args = parser.parse_args()
    evaluate_cli(args.config, args.checkpoint, args.split)
```

---

## `src/utils.py` — Inference Helper

```python
import numpy as np
import torch
from torch.cuda.amp import autocast

from extract import extract_keypoints, normalize_sequence


def predict_video(model, video_path, mean, std, label_names, device, top_k=5):
    """
    Run inference on a single video file.

    Returns:
        List of (label_name, confidence) tuples, top_k results
    """
    model.eval()

    kp = extract_keypoints(video_path)      # (T, 543)
    kp = normalize_sequence(kp)             # (64, 543)
    kp = (kp - mean) / std                  # normalize

    x = torch.FloatTensor(kp).unsqueeze(0).to(device)  # (1, 64, 543)

    with torch.no_grad(), autocast():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_ids = probs.topk(top_k)
    results = [(label_names[i.item()], p.item()) for i, p in zip(top_ids, top_probs)]
    return results
```

---

## Metrics Reference

| Metric | Target (INCLUDE, 263 classes) |
|---|---|
| Top-1 Accuracy | ≥ 65% (BiLSTM baseline) |
| Top-1 Accuracy | ≥ 75% (SPOTER) |
| Top-5 Accuracy | ≥ 90% (SPOTER) |

If you are below these numbers: check augmentation is on, check normalization is applied, verify keypoint extraction didn't silently fail (check `failed.txt`), and confirm val split has no overlap with train.

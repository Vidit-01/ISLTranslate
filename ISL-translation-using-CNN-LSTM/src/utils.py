import os
import json
import random
import numpy as np
import torch
from torch.cuda.amp import autocast

from extract import extract_keypoints, normalize_sequence


def build_label_map(video_dir):
    """Build label_map.json mapping class name -> integer index."""
    classes = sorted(os.listdir(video_dir))  # alphabetical for reproducibility
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "splits", "label_map.json"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(label_map, f, indent=2)
    return label_map


def create_splits(video_dir, label_map, train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test split files."""
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

    splits_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "splits"
    )
    os.makedirs(splits_dir, exist_ok=True)

    def write_split(split, path):
        with open(path, "w") as f:
            for path_, label in split:
                f.write(f"{path_} {label}\n")

    write_split(train, os.path.join(splits_dir, "train.txt"))
    write_split(val,   os.path.join(splits_dir, "val.txt"))
    write_split(test,  os.path.join(splits_dir, "test.txt"))


def compute_stats(npy_dir, train_files):
    """Compute mean/std over training keypoints for normalization."""
    all_data = []
    for fname in train_files:
        arr = np.load(os.path.join(npy_dir, fname))  # (64, 543)
        all_data.append(arr)
    all_data = np.stack(all_data)  # (N, 64, 543)
    mean = all_data.mean(axis=(0, 1))  # (543,)
    std = all_data.std(axis=(0, 1)) + 1e-6

    splits_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "splits"
    )
    os.makedirs(splits_dir, exist_ok=True)
    np.save(os.path.join(splits_dir, "mean.npy"), mean)
    np.save(os.path.join(splits_dir, "std.npy"), std)
    return mean, std


def predict_video(model, video_path, model_path, mean, std, label_names, device, top_k=5):
    """
    Run inference on a single video file using MediaPipe Task API.

    Returns:
        List of (label_name, confidence) tuples, top_k results
    """
    model.eval()

    kp = extract_keypoints(video_path, model_path)      # (T, 543)
    kp = normalize_sequence(kp)                        # (64, 543)
    kp = (kp - mean) / std                             # normalize

    x = torch.FloatTensor(kp).unsqueeze(0).to(device)  # (1, 64, 543)

    with torch.no_grad(), autocast():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_ids = probs.topk(top_k)
    results = [(label_names[i.item()], p.item()) for i, p in zip(top_ids, top_probs)]
    return results

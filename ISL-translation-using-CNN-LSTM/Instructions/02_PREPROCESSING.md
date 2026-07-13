# Preprocessing Pipeline

## Overview

Extract MediaPipe Holistic keypoints from every video once, save as `.npy` files to Drive. All training loads `.npy` directly — no video decoding at training time.

**Keypoint vector per frame: 543 dimensions**
- Pose: 33 landmarks × 4 (x, y, z, visibility) = 132
- Left hand: 21 landmarks × 3 (x, y, z) = 63
- Right hand: 21 landmarks × 3 = 63
- Face (subset): 95 landmarks × 3 = 285

## `src/extract.py` — Full Implementation

```python
import os
import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d
from tqdm import tqdm

mp_holistic = mp.solutions.holistic

TARGET_LEN = 64  # normalized sequence length for all videos
KEYPOINT_DIM = 543


def extract_landmarks(results):
    """Extract and concatenate all landmark groups into a flat vector."""
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility]
                  for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(132)
    )
    lh = (
        np.array([[lm.x, lm.y, lm.z]
                  for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(63)
    )
    rh = (
        np.array([[lm.x, lm.y, lm.z]
                  for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(63)
    )
    # Use first 95 face landmarks (lips, eyebrows, nose — grammatically relevant)
    face = (
        np.array([[lm.x, lm.y, lm.z]
                  for lm in results.face_landmarks.landmark[:95]]).flatten()
        if results.face_landmarks else np.zeros(285)
    )
    return np.concatenate([pose, lh, rh, face])  # (543,)


def extract_keypoints(video_path):
    """Extract per-frame keypoints from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    sequence = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            sequence.append(extract_landmarks(results))

    cap.release()
    if len(sequence) == 0:
        raise ValueError(f"No frames extracted from: {video_path}")
    return np.array(sequence, dtype=np.float32)  # (T, 543)


def normalize_sequence(seq, target_len=TARGET_LEN):
    """Temporally resample sequence to fixed length via linear interpolation."""
    T = seq.shape[0]
    if T == target_len:
        return seq
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, seq, axis=0, kind='linear')
    return f(x_new).astype(np.float32)  # (target_len, 543)


def preprocess_dataset(video_dir, save_dir, label_map, split_file=None):
    """
    Extract keypoints for all videos and save as .npy files.

    Args:
        video_dir: Root directory containing class subdirectories of videos
        save_dir: Directory to save .npy files
        label_map: dict mapping class_name -> int label
        split_file: Optional path to split txt; if None process everything
    """
    os.makedirs(save_dir, exist_ok=True)
    failed = []

    if split_file:
        with open(split_file) as f:
            entries = [line.strip().split() for line in f]
        items = [(os.path.join(video_dir, p), int(lbl)) for p, lbl in entries]
    else:
        items = []
        for label_name in os.listdir(video_dir):
            if label_name not in label_map:
                continue
            label_id = label_map[label_name]
            class_dir = os.path.join(video_dir, label_name)
            for vid in os.listdir(class_dir):
                if vid.endswith(('.mp4', '.avi', '.mov')):
                    items.append((os.path.join(class_dir, vid), label_id))

    for video_path, label_id in tqdm(items, desc="Extracting keypoints"):
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(save_dir, f"{vid_name}_{label_id}.npy")

        if os.path.exists(save_path):
            continue  # resume-safe

        try:
            kp = extract_keypoints(video_path)
            kp = normalize_sequence(kp)
            np.save(save_path, kp)
        except Exception as e:
            print(f"FAILED: {video_path} — {e}")
            failed.append(video_path)

    if failed:
        with open(os.path.join(save_dir, "failed.txt"), "w") as f:
            f.write("\n".join(failed))
    print(f"Done. Failed: {len(failed)}/{len(items)}")


def verify_keypoints(save_dir, expected_shape=(64, 543)):
    """Sanity check all saved .npy files."""
    broken = []
    for fname in os.listdir(save_dir):
        if not fname.endswith('.npy'):
            continue
        arr = np.load(os.path.join(save_dir, fname))
        if arr.shape != expected_shape:
            broken.append((fname, arr.shape))
    if broken:
        print(f"Shape mismatches: {broken}")
    else:
        print(f"All {len(os.listdir(save_dir))} files OK, shape {expected_shape}")
```

## `src/augment.py` — Full Implementation

```python
import numpy as np
from scipy.interpolate import interp1d

TARGET_LEN = 64


def time_warp(seq, factor_range=(0.8, 1.2)):
    """Stretch or compress sequence in time."""
    factor = np.random.uniform(*factor_range)
    T = seq.shape[0]
    new_len = max(2, int(T * factor))
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, new_len)
    f = interp1d(x_old, seq, axis=0, kind='linear')
    warped = f(x_new)
    # Resample back to TARGET_LEN
    x_old2 = np.linspace(0, 1, new_len)
    x_new2 = np.linspace(0, 1, TARGET_LEN)
    f2 = interp1d(x_old2, warped, axis=0, kind='linear')
    return f2(x_new2).astype(np.float32)


def spatial_jitter(seq, std=0.01):
    """Add Gaussian noise to keypoint coordinates."""
    return seq + np.random.normal(0, std, seq.shape).astype(np.float32)


def mirror(seq):
    """
    Flip sign horizontally (mirror x-coordinates).
    Swaps left/right hand landmarks too.
    Coordinate indices for x: every 3rd starting from 0 for xyz groups,
    every 4th starting from 0 for xyzv groups (pose).
    """
    seq = seq.copy()
    # Flip pose x (indices 0,4,8,...,128 — every 4th)
    seq[:, 0:132:4] = 1.0 - seq[:, 0:132:4]
    # Flip left hand x (indices 132,135,...,186 — every 3rd)
    seq[:, 132:195:3] = 1.0 - seq[:, 132:195:3]
    # Flip right hand x
    seq[:, 195:258:3] = 1.0 - seq[:, 195:258:3]
    # Swap left and right hands
    lh = seq[:, 132:195].copy()
    rh = seq[:, 195:258].copy()
    seq[:, 132:195] = rh
    seq[:, 195:258] = lh
    # Flip face x
    seq[:, 258::3] = 1.0 - seq[:, 258::3]
    return seq


def drop_frames(seq, drop_prob=0.1):
    """Randomly zero out entire frames to simulate occlusion."""
    mask = np.random.rand(seq.shape[0]) > drop_prob
    seq = seq.copy()
    seq[~mask] = 0.0
    return seq


def augment(seq, p_warp=0.5, p_jitter=0.7, p_mirror=0.5, p_drop=0.3):
    """Apply random augmentations to a keypoint sequence."""
    if np.random.rand() < p_warp:
        seq = time_warp(seq)
    if np.random.rand() < p_jitter:
        seq = spatial_jitter(seq)
    if np.random.rand() < p_mirror:
        seq = mirror(seq)
    if np.random.rand() < p_drop:
        seq = drop_frames(seq)
    return seq
```

## Normalization

Compute per-dimension mean and std over training set, apply z-score normalization:

```python
def compute_stats(npy_dir, train_files):
    """Compute mean/std over training keypoints for normalization."""
    all_data = []
    for fname in train_files:
        arr = np.load(os.path.join(npy_dir, fname))  # (64, 543)
        all_data.append(arr)
    all_data = np.stack(all_data)  # (N, 64, 543)
    mean = all_data.mean(axis=(0, 1))  # (543,)
    std = all_data.std(axis=(0, 1)) + 1e-6
    np.save("data/splits/mean.npy", mean)
    np.save("data/splits/std.npy", std)
    return mean, std
```

Load mean/std in Dataset and apply: `seq = (seq - mean) / std`

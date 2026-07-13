import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from augment import augment as apply_augment


class ISLDataset(Dataset):
    def __init__(self, npy_dir, split_file, mean_path, std_path, augment=False):
        """
        Args:
            npy_dir: Directory containing .npy keypoint files
            split_file: Path to split txt (format: "filename_labelid.npy" per line
                        OR "relative/video/path labelid" — see note below)
            mean_path: Path to mean.npy for normalization
            std_path: Path to std.npy for normalization
            augment: Whether to apply augmentation
        """
        self.npy_dir = npy_dir
        self.augment = augment
        self.mean = np.load(mean_path)  # (543,)
        self.std = np.load(std_path)    # (543,)

        # Build file list from directory (naming convention: vidname_labelid.npy)
        self.samples = []
        for fname in os.listdir(npy_dir):
            if not fname.endswith('.npy'):
                continue
            label_id = int(fname.rsplit('_', 1)[-1].replace('.npy', ''))
            self.samples.append((fname, label_id))

        # If split_file given, filter to only those files
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                allowed = set(line.strip() for line in f)
            self.samples = [(f, l) for f, l in self.samples if f in allowed]

        assert len(self.samples) > 0, f"No samples found in {npy_dir}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        seq = np.load(os.path.join(self.npy_dir, fname))  # (64, 543)

        if self.augment:
            seq = apply_augment(seq)

        seq = (seq - self.mean) / self.std  # z-score normalize
        return torch.FloatTensor(seq), torch.tensor(label, dtype=torch.long)

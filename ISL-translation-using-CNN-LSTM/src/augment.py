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

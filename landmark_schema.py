from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LANDMARK_SPECS: tuple[tuple[str, int], ...] = (
    ("face", 468),
    ("pose", 33),
    ("left_hand", 21),
    ("right_hand", 21),
)
LANDMARK_TYPES = tuple(name for name, _ in LANDMARK_SPECS)
ROWS_PER_FRAME = sum(count for _, count in LANDMARK_SPECS)
DIM_NAMES = ("x", "y", "z")
FEATURE_IDXS = np.array(
    list(range(468, 489))
    + list(range(489, 522))
    + list(range(522, 543))
    + [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375],
    dtype=np.int32,
)
N_FEATURE_POINTS = len(FEATURE_IDXS)


def build_landmark_template() -> pd.DataFrame:
    rows = []
    for type_name, count in LANDMARK_SPECS:
        for landmark_index in range(count):
            rows.append({"type": type_name, "landmark_index": landmark_index})
    return pd.DataFrame(rows, columns=["type", "landmark_index"])


XYZ_SKEL = build_landmark_template()


def _unwrap_landmark_group(group: object) -> Sequence[object] | None:
    if group is None:
        return None
    if isinstance(group, (list, tuple)):
        if not group:
            return None
        first = group[0]
        if hasattr(first, "landmark"):
            return first.landmark
        return group
    if hasattr(group, "landmark"):
        return group.landmark
    return None


def normalize_landmark_group(group: object) -> Sequence[object] | None:
    return _unwrap_landmark_group(group)


def _landmarks_to_dataframe(landmarks: Sequence[object] | None, type_name: str) -> pd.DataFrame:
    if not landmarks:
        return pd.DataFrame(columns=["landmark_index", "x", "y", "z", "type"])

    rows = []
    for landmark_index, point in enumerate(landmarks):
        rows.append(
            {
                "landmark_index": landmark_index,
                "x": getattr(point, "x", np.nan),
                "y": getattr(point, "y", np.nan),
                "z": getattr(point, "z", np.nan),
                "type": type_name,
            }
        )
    return pd.DataFrame(rows, columns=["landmark_index", "x", "y", "z", "type"])


def results_to_frame_dataframe(results: object, frame: int) -> pd.DataFrame:
    groups = []
    for type_name in LANDMARK_TYPES:
        group = _unwrap_landmark_group(getattr(results, f"{type_name}_landmarks", None))
        group_df = _landmarks_to_dataframe(group, type_name)
        if not group_df.empty:
            groups.append(group_df)

    if groups:
        landmarks = pd.concat(groups, ignore_index=True)
    else:
        landmarks = pd.DataFrame(columns=["landmark_index", "x", "y", "z", "type"])
    return XYZ_SKEL.merge(landmarks, on=["type", "landmark_index"], how="left").assign(frame=frame)


def frame_dataframe_to_array(frame_df: pd.DataFrame) -> np.ndarray:
    return frame_df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)


def load_relevant_data_subset(pq_path: str | Path) -> np.ndarray:
    data = pd.read_parquet(pq_path, columns=list(DIM_NAMES))
    if len(data) % ROWS_PER_FRAME != 0:
        raise ValueError(
            f"Parquet file {pq_path} has {len(data)} landmark rows, which is not divisible by {ROWS_PER_FRAME}."
        )

    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(DIM_NAMES))
    return data.astype(np.float32)


def pad_or_trim_sequence(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    if sequence.ndim != 3:
        raise ValueError(f"Expected a 3D tensor [frames, rows, dims], got shape {sequence.shape}.")

    frames = sequence.shape[0]
    if frames == max_frames:
        return sequence.astype(np.float32, copy=False)

    if frames > max_frames:
        return sequence[-max_frames:].astype(np.float32, copy=False)

    padded = np.zeros((max_frames, sequence.shape[1], sequence.shape[2]), dtype=np.float32)
    padded[-frames:] = sequence.astype(np.float32, copy=False)
    return padded

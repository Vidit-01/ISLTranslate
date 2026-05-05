from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from landmark_schema import (
    DIM_NAMES,
    FEATURE_IDXS,
    N_FEATURE_POINTS,
    ROWS_PER_FRAME,
    frame_dataframe_to_array,
    load_relevant_data_subset,
    pad_or_trim_sequence,
    results_to_frame_dataframe,
)


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)


def ensure_model_asset(model_path: Path, model_url: str) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe holistic model to {model_path}...")
    from urllib.request import urlretrieve

    urlretrieve(model_url, model_path)
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Indian Sign Language recognition model.")
    parser.add_argument("--manifest", required=True, help="CSV file with columns path,label")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to store model and metadata")
    parser.add_argument("--cache-dir", default=None, help="Optional directory for cached landmark sequences")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL, help="Holistic Landmarker task download URL")
    parser.add_argument("--model-path", default="holistic_landmarker.task", help="Local holistic task path")
    parser.add_argument("--max-frames", type=int, default=128, help="Frames per sequence after padding/truncation")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth frame when extracting from video")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", nargs="*", default=None, help="Optional exact labels/words to train on")
    parser.add_argument("--max-classes", type=int, default=None, help="Use the top N labels by sample count")
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="Cap samples per selected label")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"path", "label"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing required columns: {sorted(missing)}")

    manifest = manifest.copy()
    manifest["path"] = manifest["path"].astype(str)
    manifest["label"] = manifest["label"].astype(str)
    if "split" in manifest.columns:
        manifest["split"] = manifest["split"].astype(str)
    return manifest


def filter_manifest(
    manifest: pd.DataFrame,
    labels: list[str] | None,
    max_classes: int | None,
    max_samples_per_class: int | None,
    seed: int,
) -> pd.DataFrame:
    manifest = manifest.copy()
    counts = manifest["label"].value_counts()

    if labels:
        requested = {label.casefold(): label for label in labels}
        available = {label.casefold(): label for label in counts.index}
        matched = [available[key] for key in requested if key in available]
        missing = [label for key, label in requested.items() if key not in available]
        if missing:
            print(f"Requested labels not found: {missing}")
            print("Available labels preview:")
            print(counts.head(40))
        if len(matched) < 2:
            raise ValueError("Fewer than 2 requested labels were found. Adjust --labels to match your manifest labels.")
        manifest = manifest[manifest["label"].isin(matched)]
    elif max_classes is not None:
        keep = counts.head(max_classes).index
        manifest = manifest[manifest["label"].isin(keep)]

    if max_samples_per_class is not None:
        manifest = (
            manifest.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(n=min(max_samples_per_class, len(group)), random_state=seed))
            .reset_index(drop=True)
        )

    return manifest.sample(frac=1, random_state=seed).reset_index(drop=True)


def split_manifest(manifest: pd.DataFrame, validation_split: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(manifest) < 2:
        raise ValueError("Need at least 2 samples to create a train/validation split.")

    class_counts = manifest["label"].value_counts()
    n_classes = len(class_counts)
    n_val = max(1, int(np.ceil(validation_split * len(manifest))))
    if len(manifest) - n_val < 1:
        n_val = 1

    can_stratify = (
        n_classes > 1
        and class_counts.min() >= 2
        and n_val >= n_classes
        and (len(manifest) - n_val) >= n_classes
    )

    if not can_stratify:
        print("Using non-stratified split because the dataset is small or has classes with only one sample.")
        print(class_counts.head(30))

    return train_test_split(
        manifest,
        test_size=n_val,
        random_state=seed,
        stratify=manifest["label"] if can_stratify else None,
    )


def cache_path_for(sample_path: Path, cache_dir: Path) -> Path:
    digest = hashlib.sha1(str(sample_path).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{sample_path.stem}-{digest}.npy"


def load_parquet_sequence(pq_path: Path) -> np.ndarray:
    sequence = load_relevant_data_subset(pq_path)
    return sequence


def extract_sequence_from_video(
    video_path: Path,
    detector: object,
    frame_stride: int = 1,
    start_timestamp_ms: int = 0,
) -> tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    frames: list[np.ndarray] = []
    last_timestamp_ms = start_timestamp_ms
    target_size = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_stride > 1 and frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        if target_size is None:
            target_size = (frame.shape[1], frame.shape[0])
        elif frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = max(last_timestamp_ms + 1, start_timestamp_ms + int((frame_idx / fps) * 1000))
        last_timestamp_ms = timestamp_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = detector.detect_for_video(mp_image, timestamp_ms)
        frame_df = results_to_frame_dataframe(result, frame_idx)
        frames.append(frame_dataframe_to_array(frame_df))
        frame_idx += 1

    cap.release()

    if not frames:
        return np.zeros((0, ROWS_PER_FRAME, len(DIM_NAMES)), dtype=np.float32), last_timestamp_ms + 1

    return np.stack(frames).astype(np.float32), last_timestamp_ms + 1


def load_sequence_for_sample(
    sample_path: Path,
    detector: object,
    cache_dir: Path | None,
    frame_stride: int,
    start_timestamp_ms: int = 0,
) -> tuple[np.ndarray, int]:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_path_for(sample_path, cache_dir)
        if cached.exists():
            return np.load(cached), start_timestamp_ms

    suffix = sample_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        sequence = load_parquet_sequence(sample_path)
        next_timestamp_ms = start_timestamp_ms
    else:
        sequence, next_timestamp_ms = extract_sequence_from_video(
            sample_path,
            detector,
            frame_stride=frame_stride,
            start_timestamp_ms=start_timestamp_ms,
        )

    if cache_dir is not None:
        np.save(cached, sequence)

    return sequence, next_timestamp_ms


def build_sequences(
    manifest: pd.DataFrame,
    detector: object,
    max_frames: int,
    cache_dir: Path | None,
    frame_stride: int,
    label_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    label_names = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    next_timestamp_ms = 0

    for row in manifest.itertuples(index=False):
        sample_path = Path(row.path)
        if not sample_path.is_absolute():
            sample_path = Path.cwd() / sample_path
        sequence, next_timestamp_ms = load_sequence_for_sample(
            sample_path,
            detector,
            cache_dir,
            frame_stride,
            start_timestamp_ms=next_timestamp_ms,
        )
        if sequence.shape[0] == 0:
            continue

        sequences.append(pad_or_trim_sequence(sequence, max_frames)[:, FEATURE_IDXS, :])
        labels.append(label_to_id[str(row.label)])

    if not sequences:
        raise RuntimeError("No usable samples were found in the manifest.")

    return np.stack(sequences), np.asarray(labels, dtype=np.int64), label_names


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(d_model),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn = self.attention(inputs, inputs, training=training)
        x = self.norm1(inputs + self.dropout1(attn, training=training))
        ffn = self.ffn(x, training=training)
        return self.norm2(x + self.dropout2(ffn, training=training))


def build_model(max_frames: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(max_frames, N_FEATURE_POINTS, len(DIM_NAMES)), name="inputs")
    x = tf.where(tf.math.is_nan(inputs), 0.0, inputs)
    frame_mask = tf.cast(tf.reduce_any(tf.not_equal(x, 0.0), axis=[2, 3]), tf.float32)
    x = tf.reshape(x, [-1, max_frames, N_FEATURE_POINTS * len(DIM_NAMES)])

    d_model = 256
    x = tf.keras.layers.Dense(d_model, activation="gelu")(x)
    positions = tf.range(start=0, limit=max_frames, delta=1)
    x = x + tf.keras.layers.Embedding(max_frames, d_model)(positions)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = TransformerBlock(d_model=d_model, num_heads=4, ff_dim=512)(x)
    x = TransformerBlock(d_model=d_model, num_heads=4, ff_dim=512)(x)
    x = tf.expand_dims(frame_mask, axis=-1) * x
    x = tf.reduce_sum(x, axis=1) / tf.maximum(tf.reduce_sum(frame_mask, axis=1, keepdims=True), 1.0)
    x = tf.keras.layers.Dense(128, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="outputs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def write_artifacts(
    output_dir: Path,
    labels: list[str],
    max_frames: int,
    history: tf.keras.callbacks.History,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_payload = {
        "labels": labels,
        "label_to_id": {label: idx for idx, label in enumerate(labels)},
    }
    (output_dir / "labels.json").write_text(json.dumps(label_payload, indent=2), encoding="utf-8")
    (output_dir / "training_config.json").write_text(
        json.dumps(
            {
                "max_frames": max_frames,
                "rows_per_frame": ROWS_PER_FRAME,
                "feature_indices": FEATURE_IDXS.tolist(),
                "n_feature_points": N_FEATURE_POINTS,
                "dims": list(DIM_NAMES),
                "history": history.history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def export_tflite(model: tf.keras.Model, output_dir: Path) -> Path:
    class InferenceModule(tf.Module):
        def __init__(self, keras_model):
            super().__init__()
            self.keras_model = keras_model

        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=[model.input_shape[1], N_FEATURE_POINTS, len(DIM_NAMES)],
                    dtype=tf.float32,
                    name="inputs",
                )
            ]
        )
        def __call__(self, inputs):
            outputs = self.keras_model(tf.expand_dims(inputs, axis=0), training=False)[0]
            return {"outputs": outputs}

    inference_module = InferenceModule(model)
    concrete_func = inference_module.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], inference_module)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    model_path = output_dir / "model.tflite"
    model_path.write_bytes(tflite_model)
    return model_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    detector_path = ensure_model_asset(Path(args.model_path), args.model_url)
    base_options = python.BaseOptions(model_asset_path=str(detector_path))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    manifest = load_manifest(Path(args.manifest))
    manifest = filter_manifest(
        manifest,
        labels=args.labels,
        max_classes=args.max_classes,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
    )
    labels = sorted(manifest["label"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    if "split" in manifest.columns:
        split_col = manifest["split"].fillna("").str.lower()
        train_manifest = manifest[split_col.isin({"train", "training"})].copy()
        val_manifest = manifest[split_col.isin({"val", "validation", "dev"})].copy()
        if train_manifest.empty or val_manifest.empty:
            raise ValueError("When a split column is provided, it must contain both train and validation samples.")
    else:
        train_manifest, val_manifest = split_manifest(manifest, args.validation_split, args.seed)

    with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as detector:
        X_train, y_train, labels = build_sequences(
            train_manifest,
            detector,
            max_frames=args.max_frames,
            cache_dir=cache_dir,
            frame_stride=args.frame_stride,
            label_to_id=label_to_id,
        )
        X_val, y_val, _ = build_sequences(
            val_manifest,
            detector,
            max_frames=args.max_frames,
            cache_dir=cache_dir,
            frame_stride=args.frame_stride,
            label_to_id=label_to_id,
        )

    model = build_model(args.max_frames, len(labels))
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(output_dir / "keras_model.keras")
    write_artifacts(output_dir, labels, args.max_frames, history)
    export_tflite(model, output_dir)
    print(f"Training complete. Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()

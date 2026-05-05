from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from landmark_schema import frame_dataframe_to_array, pad_or_trim_sequence, results_to_frame_dataframe


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained SignSense model on a video.")
    parser.add_argument("--video", required=True, help="Path to the video file to classify")
    parser.add_argument("--artifacts-dir", default="training_artifacts", help="Directory containing model.tflite and labels.json")
    parser.add_argument("--holistic-model", default="holistic_landmarker.task", help="Path to MediaPipe holistic .task file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to print")
    parser.add_argument("--window-stride", type=int, default=16, help="Frame stride for averaging predictions over long videos")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_model_asset(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe holistic model to {model_path}...")
    urlretrieve(MODEL_URL, model_path)
    return model_path


def create_prediction_runner(interpreter: tf.lite.Interpreter):
    signatures = interpreter.get_signature_list()
    if "serving_default" in signatures:
        return interpreter.get_signature_runner("serving_default")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    input_dtype = input_details[0]["dtype"]

    def predict(inputs):
        inputs = np.asarray(inputs, dtype=input_dtype)
        if tuple(inputs.shape) != tuple(input_details[0]["shape"]):
            interpreter.resize_tensor_input(input_index, inputs.shape, strict=False)
            interpreter.allocate_tensors()
        interpreter.set_tensor(input_index, inputs)
        interpreter.invoke()
        return {"outputs": interpreter.get_tensor(output_index)}

    return predict


def load_labels(path: Path) -> list[str]:
    payload = load_json(path)
    if "labels" in payload:
        return [str(label) for label in payload["labels"]]
    if isinstance(payload, list):
        return [str(label) for label in payload]
    return [str(label) for _, label in sorted(payload.items())]


def extract_video_sequence(video_path: Path, detector) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    last_timestamp_ms = 0
    target_size = None
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if target_size is None:
            target_size = (frame.shape[1], frame.shape[0])
        elif frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(last_timestamp_ms + 1, int((frame_idx / fps) * 1000))
        last_timestamp_ms = timestamp_ms
        result = detector.detect_for_video(mp_image, timestamp_ms)
        frames.append(frame_dataframe_to_array(results_to_frame_dataframe(result, frame_idx)))
        frame_idx += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames were decoded from {video_path}")
    return np.stack(frames).astype(np.float32)


def make_model_inputs(sequence: np.ndarray, max_frames: int, feature_indices: np.ndarray | None, window_stride: int) -> list[np.ndarray]:
    if sequence.shape[0] <= max_frames:
        inputs = [pad_or_trim_sequence(sequence, max_frames)]
    else:
        starts = list(range(0, sequence.shape[0] - max_frames + 1, max(1, window_stride)))
        last_start = sequence.shape[0] - max_frames
        if starts[-1] != last_start:
            starts.append(last_start)
        inputs = [sequence[start : start + max_frames] for start in starts]

    if feature_indices is not None:
        inputs = [item[:, feature_indices, :] for item in inputs]
    return inputs


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    artifacts_dir = Path(args.artifacts_dir)
    config = load_json(artifacts_dir / "training_config.json")
    labels = load_labels(artifacts_dir / "labels.json")

    max_frames = int(config.get("max_frames", 64))
    feature_indices = config.get("feature_indices")
    feature_indices = np.asarray(feature_indices, dtype=np.int32) if feature_indices else None

    holistic_model = ensure_model_asset(Path(args.holistic_model))
    base_options = python.BaseOptions(model_asset_path=str(holistic_model))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    with vision.HolisticLandmarker.create_from_options(options) as detector:
        sequence = extract_video_sequence(video_path, detector)

    inputs_list = make_model_inputs(sequence, max_frames, feature_indices, args.window_stride)

    interpreter = tf.lite.Interpreter(model_path=str(artifacts_dir / "model.tflite"))
    interpreter.allocate_tensors()
    pred_fn = create_prediction_runner(interpreter)
    predictions = [np.asarray(pred_fn(inputs=inputs)["outputs"]).reshape(-1) for inputs in inputs_list]
    outputs = np.mean(np.stack(predictions), axis=0)

    order = outputs.argsort()[::-1][: args.top_k]
    print(f"Video: {video_path}")
    print(f"Frames decoded: {sequence.shape[0]}")
    print(f"Windows evaluated: {len(inputs_list)}")
    print(f"Input tensor: {inputs_list[0].shape}")
    print(f"Prediction: {labels[int(order[0])] if labels else int(order[0])}")
    print("Top predictions:")
    for rank, idx in enumerate(order, start=1):
        label = labels[int(idx)] if int(idx) < len(labels) else str(int(idx))
        print(f"{rank}. {label}: {float(outputs[idx]):.4f}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()

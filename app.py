import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from urllib.request import urlretrieve

from landmark_schema import (
    load_relevant_data_subset,
    normalize_landmark_group,
    pad_or_trim_sequence,
    results_to_frame_dataframe,
)

load_dotenv()

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
MODEL_PATH = Path(os.getenv("HOLISTIC_LANDMARKER_MODEL_PATH", "holistic_landmarker.task"))
MODEL_TFLITE_PATH = Path(os.getenv("MODEL_TFLITE_PATH", "model.tflite"))
for candidate in (Path("training_artifacts/model.tflite"), Path("artifacts/model.tflite")):
    if not MODEL_TFLITE_PATH.exists() and candidate.exists():
        MODEL_TFLITE_PATH = candidate


def resolve_artifact_path(env_name, default_name):
    env_value = os.getenv(env_name)
    candidates = []
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(Path(default_name))
    candidates.append(Path("training_artifacts") / default_name)
    candidates.append(Path("artifacts") / default_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(candidates[0])


LABEL_MAP_PATH = resolve_artifact_path("LABEL_MAP_PATH", "labels.json")
TRAINING_CONFIG_PATH = resolve_artifact_path("TRAINING_CONFIG_PATH", "training_config.json")


def load_model_max_frames():
    if TRAINING_CONFIG_PATH.exists():
        payload = json.loads(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "max_frames" in payload:
            return int(payload["max_frames"])
    return int(os.getenv("MODEL_MAX_FRAMES", "128"))


MODEL_MAX_FRAMES = load_model_max_frames()


def load_feature_indices():
    if TRAINING_CONFIG_PATH.exists():
        payload = json.loads(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "feature_indices" in payload:
            return np.asarray(payload["feature_indices"], dtype=np.int32)
    return None


FEATURE_IDXS = load_feature_indices()

# Initialize the TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path=str(MODEL_TFLITE_PATH))
interpreter.allocate_tensors()


def create_prediction_runner(interpreter):
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
        expected_shape = tuple(input_details[0]["shape"])
        if tuple(inputs.shape) != expected_shape:
            interpreter.resize_tensor_input(input_index, inputs.shape, strict=False)
            interpreter.allocate_tensors()
        interpreter.set_tensor(input_index, inputs)
        interpreter.invoke()
        return {"outputs": interpreter.get_tensor(output_index)}

    return predict


pred_fn = create_prediction_runner(interpreter)

def load_label_mapping():
    if LABEL_MAP_PATH.exists():
        payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "labels" in payload:
            labels = payload["labels"]
        elif isinstance(payload, list):
            labels = payload
        else:
            labels = list(payload.values())
        return {idx: label for idx, label in enumerate(labels)}

    csv_candidates = [Path("labels.csv"), Path("train.csv")]
    for candidate in csv_candidates:
        if candidate.exists():
            frame = pd.read_csv(candidate)
            label_column = "label" if "label" in frame.columns else "sign"
            labels = sorted(frame[label_column].astype(str).unique().tolist())
            return {idx: label for idx, label in enumerate(labels)}

    return {}


ORD2SIGN = load_label_mapping()
SIGN2ORD = {label: idx for idx, label in ORD2SIGN.items()}
print(f"Loaded model: {MODEL_TFLITE_PATH}")
print(f"Loaded labels: {LABEL_MAP_PATH} ({len(ORD2SIGN)} classes)")

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def ensure_model_asset(model_path, model_url):
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe holistic model to {model_path}...")
    urlretrieve(model_url, model_path)
    return model_path


def _connection_indices(connection):
    if hasattr(connection, "start") and hasattr(connection, "end"):
        return connection.start, connection.end
    if isinstance(connection, (tuple, list)) and len(connection) >= 2:
        return connection[0], connection[1]
    raise TypeError(f"Unsupported connection type: {type(connection)!r}")


def draw_landmark_group(image, landmarks, connections, point_color, line_color=None, point_radius=2, line_thickness=2):
    landmarks = normalize_landmark_group(landmarks)
    if not landmarks:
        return

    height, width = image.shape[:2]
    line_color = line_color or point_color

    for connection in connections or []:
        start_idx, end_idx = _connection_indices(connection)
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue

        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(image, start_point, end_point, line_color, line_thickness)

    for landmark in landmarks:
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(image, point, point_radius, point_color, -1)


def annotate_holistic_frame(image, results):
    annotated_image = image.copy()

    draw_landmark_group(
        annotated_image,
        results.face_landmarks,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        point_color=(255, 210, 120),
        line_color=(255, 170, 60),
        point_radius=1,
        line_thickness=1,
    )
    draw_landmark_group(
        annotated_image,
        results.pose_landmarks,
        vision.PoseLandmarksConnections.POSE_LANDMARKS,
        point_color=(0, 255, 0),
        line_color=(0, 180, 0),
        point_radius=3,
        line_thickness=2,
    )
    draw_landmark_group(
        annotated_image,
        results.left_hand_landmarks,
        vision.HandLandmarksConnections.HAND_CONNECTIONS,
        point_color=(255, 0, 0),
        line_color=(200, 0, 0),
        point_radius=3,
        line_thickness=2,
    )
    draw_landmark_group(
        annotated_image,
        results.right_hand_landmarks,
        vision.HandLandmarksConnections.HAND_CONNECTIONS,
        point_color=(0, 0, 255),
        line_color=(0, 0, 200),
        point_radius=3,
        line_thickness=2,
    )

    return annotated_image

def create_frame_landmark_df(results, frame, xyz=None):
    return results_to_frame_dataframe(results, frame)

def get_display_message_from_api(recognised_words):
    GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY) 
    
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
            Objective:
            You have developed an isolated American Sign Language (ASL) word recognition model. At the end of each run, the model stores the recognized words in a list. However, the words may not necessarily be in the correct order. Your objective is to utilize these recognized words to construct a coherent and meaningful English sentence. The resulting sentence should be as simple as possible while still accurately conveying the intended meaning.

            Instructions:

            - Input: You will be provided with a Python list containing the recognized ASL words from your model. The contents of this list may vary depending on the output of your model.
            - Processing: Rearrange the words in the list to form a grammatically correct and logically valid English sentence. Take into consideration the context and logical flow of the sentence. Always ignore the word "TV".
            - Output: Generate a concise English sentence that accurately conveys the meaning of the recognized ASL words.

            Considerations:

            - Simplicity: Aim for simplicity in your sentence structure and vocabulary.
            - Clarity: Ensure that the sentence is clear and understandable.
            - Relevance: The sentence should reflect the meaning conveyed by the ASL words.
            - Grammar: Maintain proper grammar and syntax in the sentence.

            Example:

            Input: recognized_words = cat mat
            output: cat on the mat

            Here is the actual input for which you have to produce the relevant output: recognised_words = {' '.join(recognised_words)}
            """
    
    response = model.generate_content(prompt)
    
    return response.text

def do_capture_loop(pred_fn):
    all_landmarks = []
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()  # Check if the camera is working and get a frame to read dimensions
    if not ret:
        print("Failed to grab frame")
        cap.release()
        return
    
    frame_height, frame_width = frame.shape[:2]
    scale_factor = 1.0  # Scale the image to fill the window more
    scaled_height = int(frame_height * scale_factor)
    scaled_width = int(frame_width * scale_factor)
    display_width = scaled_width + 1200  # Extra width for text
    display_height = scaled_height  # Match the height of the camera feed

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5  # Larger font size
    text_thickness = 3
    
    start_time = time.monotonic()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    font_scale_d = 1.1 # Increased font scale for larger text
    last_prediction_time = 0
    escape_pressed = False
    display_message = "Press Escape to toggle message display"
    unique_signs = []
    sign_name = ""

    model_path = ensure_model_asset(MODEL_PATH, MODEL_URL)
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    with vision.HolisticLandmarker.create_from_options(options) as holistic:
        while cap.isOpened():
            current_time = time.monotonic()
            elapsed_time = int(current_time - start_time)
            timestamp_ms = int((current_time - start_time) * 1000)

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Scaling up the camera feed
            image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = holistic.detect_for_video(mp_image, timestamp_ms)
            landmarks = create_frame_landmark_df(results, elapsed_time)
            all_landmarks.append(landmarks)

            image = annotate_holistic_frame(image_rgb, results)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if current_time - last_prediction_time >= 3:
                if all_landmarks:
                    concatenated_landmarks = pd.concat(all_landmarks).reset_index(drop=True)
                    concatenated_landmarks.to_parquet("out.parquet")
                    xyz_np = pad_or_trim_sequence(load_relevant_data_subset("out.parquet"), MODEL_MAX_FRAMES)
                    if FEATURE_IDXS is not None:
                        xyz_np = xyz_np[:, FEATURE_IDXS, :]
                    try:
                        p = pred_fn(inputs=xyz_np)
                    except (ValueError, RuntimeError):
                        p = pred_fn(inputs=np.expand_dims(xyz_np, axis=0))
                    sign = p['outputs'].argmax()
                    sign_name = ORD2SIGN.get(int(sign), str(int(sign)))
                    if sign_name not in unique_signs:
                        unique_signs.append(sign_name)

                    last_prediction_time = current_time
                    all_landmarks = []  # Reset landmarks

            if sign_name == "" or sign_name == "TV":
                sign_name = "No Movement Detected"

            # UI Improvements
            display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            display[:scaled_height, :scaled_width] = image

            # Draw the text
            cv2.putText(display, f"Sign: {sign_name}", (scaled_width + 10, 100), font, font_scale, (0, 255, 0), text_thickness)
            cv2.putText(display, f"Time: {elapsed_time}s", (scaled_width + 10, 200), font, font_scale, (0, 0, 255), text_thickness)

            if escape_pressed:
                cv2.putText(display, display_message, (scaled_width + 10, 300), font, font_scale_d, (255, 255, 0), text_thickness)

            cv2.imshow("MediaPipe Holistic", display)


            key = cv2.waitKey(5)
            if key & 0xFF == 27:
                escape_pressed = not escape_pressed
                display_message = get_display_message_from_api(unique_signs) if escape_pressed else "Press Escape to toggle message display"
                if escape_pressed: 
                    unique_signs = []
            elif key & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    do_capture_loop(pred_fn)

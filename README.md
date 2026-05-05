# SignSense: Transformer based ASL Recognition Model

This repository contains a sign-language recognition pipeline for real-time gesture classification. The model uses MediaPipe Holistic Landmarker features and a sequence classifier to interpret signs, and the Gemini API to turn recognized words into sentences. The training pipeline now supports Indian Sign Language datasets through a manifest-driven workflow.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup and Testing](#setup-and-testing)
- [Demo](#demo)
- [License](#license)

## Project Overview
This project aims to provide a robust solution for real-time ASL recognition using a transformer-based deep learning model. The model captures live video input, processes the frames to detect and recognize ASL gestures, and constructs meaningful sentences from the recognized words using the Gemini-Pro LLM API. This tool can significantly enhance communication for individuals who use ASL.

## Features
- **Real-time Sign Recognition**: Detect and recognize gestures in real-time.
- **Sequence Model Architecture**: Uses a landmark sequence classifier trained on MediaPipe features.
- **Sentence Construction**: Integrates with Gemini-Pro LLM API to build sentences from recognized signs.
- **Live Video Input**: Supports live video input for seamless ASL translation.

## Tech Stack
- **Python**: Core programming language used for development.
- **TensorFlow**: Deep learning framework for building and training the transformer model.
- **OpenCV**: Library for real-time computer vision tasks, used for video capture and preprocessing.
- **MediaPipe Tasks**: Framework for building multimodal machine learning pipelines, used for holistic, face, hand, and pose tracking.
- **Gemini-Pro LLM API**: API for generating sentences from recognized ASL words.

## Training

For Colab, open [ISL_Colab_Training.ipynb](/D:/Vidit31/Papers/IPD/SignSense/ISL_Colab_Training.ipynb). It includes dataset download, manifest creation, MediaPipe landmark extraction, model training, TFLite export, and artifact download in one notebook.

The notebook defaults to a smaller 10-word experiment so the first Colab run does not have to process every video in the dataset. Edit `SELECTED_WORDS` in the configuration cell to change the words.

To train on an Indian Sign Language dataset, create a CSV manifest with at least these columns:

- `path`: path to each sample video or precomputed landmark parquet file
- `label`: the sign label for that sample

Then run:

```bash
python train_isl.py --manifest path/to/manifest.csv --output-dir artifacts
```

For a faster local smoke test, train only 10 classes and cap the number of videos per class:

```bash
python train_isl.py --manifest path/to/manifest.csv --output-dir artifacts --max-classes 10 --max-samples-per-class 25
```

The trainer writes `model.tflite`, `labels.json`, and `training_config.json` into the output directory. The app reads the label map automatically and will use `artifacts/model.tflite` if present.

## Setup and Testing

To run the project on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/DEV-D-GR8/SignSense.git
   cd SignSense

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   The app will automatically download the latest MediaPipe Holistic Landmarker `.task` bundle on first run. If you want to store it elsewhere, set `HOLISTIC_LANDMARKER_MODEL_PATH` to an existing local file.

3. **Run the Application:**
   ```bash
   powershell -ExecutionPolicy Bypass -File .\run_app.ps1
   ```

   `run_app.ps1` creates a Python 3.12 virtual environment in `.venv312`, installs the runtime dependencies from `requirements-runtime.txt`, and starts the app. The app automatically uses `training_artifacts/model.tflite`, `training_artifacts/labels.json`, and `training_artifacts/training_config.json` when that folder exists.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more information.

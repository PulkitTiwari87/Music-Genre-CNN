# 🎵 Music Genre Classification using CNN

> **Deep Learning project for automatic music genre classification using Mel Spectrograms, Convolutional Neural Networks (CNNs), and SpecAugment.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-purple)
![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-yellow?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Genres](#-genres)
- [System Architecture](#-system-architecture)
- [End-to-End Pipeline](#-end-to-end-pipeline)
- [Audio Preprocessing](#-audio-preprocessing)
- [Mel Spectrograms](#-mel-spectrograms)
- [Dataset Preparation](#-dataset-preparation)
- [Train Validation Test Split](#-train-validation-test-split)
- [Model Development](#-model-development)
- [CNN V1](#-cnn-v1)
- [CNN V2](#-cnn-v2)
- [CNN V3](#-cnn-v3)
- [SpecAugment](#-specaugment)
- [Training Strategy](#-training-strategy)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [CNN V3 Classification Report](#-cnn-v3-classification-report)
- [Confusion Matrix Analysis](#-confusion-matrix-analysis)
- [Song-Level Prediction](#-song-level-prediction)
- [Why Song-Level Accuracy Matters](#-why-song-level-accuracy-matters)
- [Model Comparison](#-model-comparison)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Web Application](#-web-application)
- [Google Colab](#-google-colab)
- [Saved Models](#-saved-models)
- [Requirements](#-requirements)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Application Architecture](#-application-architecture)
- [Reproducibility](#-reproducibility)
- [Academic / Project Significance](#-academic--project-significance)
- [Author](#-author)

---

# 🎯 Project Overview

Music Genre Classification is a machine learning problem in which an audio recording is automatically assigned to one of a predefined set of musical genres.

This project implements a **deep learning-based music genre classification system using Convolutional Neural Networks (CNNs)**.

Instead of feeding raw audio directly into the CNN, the audio is transformed into a **Mel Spectrogram**. A Mel Spectrogram represents how the frequency content of an audio signal changes over time and provides an image-like representation that can be effectively processed by CNNs.

The project was developed progressively through three CNN versions:

```text
CNN V1
  ↓
73.38% segment accuracy

CNN V2
  ↓
81.85% segment accuracy
86.67% song-level accuracy

CNN V3 + SpecAugment
  ↓
83.26% segment accuracy
90.00% song-level accuracy
```

The final model, **CNN V3**, achieved:

> ## 🏆 90.00% song-level accuracy

on the held-out test set.

---

# ❓ Problem Statement

Traditional music applications need to understand the characteristics of songs in order to organize, search, recommend, and personalize music.

Manual genre classification is:

- Time-consuming
- Subjective
- Difficult to scale
- Dependent on human judgment

The objective of this project is to build an automated system capable of analyzing an audio file and predicting its musical genre using deep learning.

---

# 🎯 Objectives

The major objectives of the project are:

1. Obtain and organize the GTZAN music genre dataset.
2. Load and preprocess audio files using Python.
3. Convert audio signals into Mel Spectrogram representations.
4. Prepare the spectrograms as CNN-compatible tensors.
5. Train multiple CNN architectures.
6. Compare different model versions.
7. Improve generalization using SpecAugment.
8. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrices.
9. Evaluate predictions at both segment level and complete-song level.
10. Save the final trained model for future inference.
11. Prepare the model for integration into a web/API-based application.

---

# ✨ Key Features

- 🎵 Automatic music genre classification
- 🎧 Audio signal processing using Librosa
- 📊 Mel Spectrogram feature extraction
- 🧠 CNN-based deep learning
- 🛡️ SpecAugment-based regularization
- 📈 Training and validation monitoring
- 📋 Classification reports
- 🔥 Confusion matrix visualization
- 🎶 Song-level prediction
- 💾 Trained model checkpointing
- ☁️ Google Colab + GPU training
- 📁 Experiment tracking
- 🚀 Ready for API/frontend integration

---

# 📚 Dataset

## GTZAN Genre Collection

The project uses the **GTZAN Genre Collection**, a commonly used benchmark dataset for automatic music genre classification.

The dataset contains:

- **1,000 audio tracks**
- **10 genres**
- **100 tracks per genre**
- Approximately **30 seconds per track**

The dataset used during development was obtained from the Kaggle GTZAN dataset distribution.

The audio files are organized by genre:

```text
genres_original/
│
├── blues/
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

### Dataset is NOT included in this repository

The original audio dataset is intentionally excluded from GitHub because of its size and dataset licensing/distribution considerations.

---

# 🎼 Genres

The model predicts 10 classes:

| Label | Genre |
|---:|---|
| 0 | Blues |
| 1 | Classical |
| 2 | Country |
| 3 | Disco |
| 4 | Hip-Hop |
| 5 | Jazz |
| 6 | Metal |
| 7 | Pop |
| 8 | Reggae |
| 9 | Rock |

The exact class ordering is determined by the label encoder used during training.

---

# 🏗️ System Architecture

The complete ML pipeline is:

```text
                    ┌─────────────────┐
                    │   Audio File    │
                    │   WAV / MP3*    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Audio Loading   │
                    │    Librosa      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Resampling      │
                    │   22,050 Hz     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Audio Segments  │
                    │   3 seconds     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Mel Spectrogram │
                    │ 128 Mel bands   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Normalization / │
                    │ dB Conversion   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   SpecAugment   │
                    │     CNN V3      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ CNN Prediction  │
                    │ 10 probabilities│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Average Segment │
                    │ Probabilities   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Final Genre     │
                    └─────────────────┘
```

---

# 🔄 End-to-End Pipeline

The project follows these stages:

```text
Dataset
   ↓
Audio Loading
   ↓
Resampling
   ↓
Segmentation
   ↓
Mel Spectrogram Extraction
   ↓
Feature Dataset
   ↓
Train / Validation / Test Split
   ↓
CNN V1
   ↓
CNN V2
   ↓
CNN V3
   ↓
SpecAugment
   ↓
Model Evaluation
   ↓
Segment Predictions
   ↓
Song-Level Probability Aggregation
   ↓
Final Genre Prediction
```

---

# 🎧 Audio Preprocessing

The audio is loaded using **Librosa**.

The project uses:

```python
sr = 22050
```

which means the audio is resampled to:

> **22,050 Hz**

A sample audio file was approximately:

```text
Sample rate: 22050 Hz
Samples:     661794
Duration:    ~30 seconds
```

The audio is then divided into smaller segments for model training.

---

# 📊 Mel Spectrograms

Raw audio is a one-dimensional time-domain signal.

Example:

```text
Amplitude
   │
 1 │     /\       /\
   │ /\ /  \  /\ /  \
 0 │/  V    \/  V    \__
   └────────────────────── Time
```

A CNN is much more effective when the audio is represented as a 2D time-frequency representation.

Therefore, the project converts audio into a Mel Spectrogram.

Conceptually:

```text
Frequency
   ↑
   │ ███████████
   │ ███████████████
   │ █████████████████
   │ ███████████████
   │ █████████
   └──────────────────→ Time
```

The project uses:

```python
n_mels = 128
```

The resulting feature representation used by the CNN is:

```text
(128, 130, 1)
```

where:

- `128` = Mel frequency bands
- `130` = time frames
- `1` = single channel

The Mel Spectrogram is converted to decibels using:

```python
librosa.power_to_db(
    mel_spectrogram,
    ref=np.max
)
```

---

# 🧮 Feature Extraction

The core feature extraction function follows this process:

```python
audio, sr = librosa.load(
    file_path,
    sr=22050
)

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=128
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)
```

The extracted spectrogram is then reshaped into a CNN-compatible tensor.

---

# 🗂️ Dataset Preparation

The dataset contains:

```text
10 genres × 100 songs = 1000 songs
```

The audio is converted into multiple segments.

The resulting feature dataset used during development contained approximately:

```text
Training:
6985 samples

Validation:
1497 samples

Testing:
1499 samples
```

Each feature has the shape:

```text
128 × 130 × 1
```

---

# 🧪 Train / Validation / Test Split

The data is divided into:

```text
Training Set
Validation Set
Testing Set
```

The test set is kept separate from training and validation so that the final model can be evaluated on unseen examples.

An important part of the project is also evaluating at the **song level**, rather than relying only on individual 3-second segments.

---

# 🧠 Model Development

Three CNN versions were developed and compared.

This was done deliberately so that model improvements could be measured experimentally rather than assuming that a more complex architecture would perform better.

---

# 🥉 CNN V1

CNN V1 was the initial baseline architecture.

### Architecture

```text
Input
  ↓
Conv2D (32)
  ↓
Batch Normalization
  ↓
MaxPooling2D
  ↓
Conv2D (64)
  ↓
Batch Normalization
  ↓
MaxPooling2D
  ↓
Conv2D (128)
  ↓
Batch Normalization
  ↓
MaxPooling2D
  ↓
Flatten
  ↓
Dense (256)
  ↓
Dropout
  ↓
Dense (10)
  ↓
Softmax
```

### Parameters

Approximately:

```text
6.52 million parameters
```

### Result

```text
Segment Accuracy: 73.38%
```

CNN V1 established the initial baseline.

---

# 🥈 CNN V2

CNN V2 improved the architecture by using **Global Average Pooling** instead of directly flattening the convolutional feature maps.

### Architecture

```text
Input
  ↓
Conv2D (32)
  ↓
Batch Normalization
  ↓
MaxPooling
  ↓
Conv2D (64)
  ↓
Batch Normalization
  ↓
MaxPooling
  ↓
Conv2D (128)
  ↓
Batch Normalization
  ↓
MaxPooling
  ↓
Global Average Pooling
  ↓
Dense
  ↓
Dropout
  ↓
Dense (10)
  ↓
Softmax
```

### Results

```text
Segment Accuracy: 81.85%

Song-Level Accuracy: 86.67%
```

This was a significant improvement over CNN V1.

---

# 🥇 CNN V3 — Final Model

CNN V3 was designed to improve generalization and feature learning.

The architecture adds:

- SpecAugment
- A deeper convolutional block
- 256 filters in the final convolutional block
- L2 regularization
- Global Average Pooling
- Dropout
- Early stopping
- Learning-rate reduction

### Architecture

```text
Input (128 × 130 × 1)
        ↓
   SpecAugment
        ↓
Conv2D (32)
        ↓
BatchNorm
        ↓
MaxPooling
        ↓
Conv2D (64)
        ↓
BatchNorm
        ↓
MaxPooling
        ↓
Conv2D (128)
        ↓
BatchNorm
        ↓
MaxPooling
        ↓
Conv2D (256)
        ↓
BatchNorm
        ↓
MaxPooling
        ↓
GlobalAveragePooling
        ↓
Dense (128)
        ↓
Dropout (0.4)
        ↓
Dense (10)
        ↓
Softmax
```

---

# 🛡️ SpecAugment

SpecAugment is applied to the Mel Spectrogram during training.

Two types of masking are used:

### Frequency masking

Random frequency bands are hidden.

```text
Before:

████████████████
████████████████
████████████████
████████████████

After:

████████████████
████████░░░░████
████████░░░░████
████████████████
```

### Time masking

Random time sections are hidden.

```text
Before:

████████████████████

After:

████████░░░░████████
```

The goal is to prevent the model from becoming overly dependent on very specific local patterns.

SpecAugment is applied only during training and not during inference.

---

# ⚙️ Training Strategy

CNN V3 was trained using:

### Optimizer

```text
Adam
```

Initial learning rate:

```text
0.001
```

### Loss

```text
Categorical Crossentropy
```

### Batch size

```text
32
```

### Maximum epochs

```text
30
```

### Early stopping

Training stops when validation loss stops improving for several epochs.

The best model weights are restored.

### ReduceLROnPlateau

The learning rate is reduced when validation loss stops improving.

### Regularization

CNN V3 uses:

```text
L2 regularization
Dropout = 0.4
SpecAugment
```

---

# 📈 CNN V3 Training Result

Training stopped early at epoch 28 (about 185 s on a Colab GPU):

```text
Epoch 28: Early stopping
Restoring model weights from the end of the best epoch: 22   (best validation loss)
```

The model that was evaluated is the checkpoint with the best validation accuracy, saved at epoch 21:

```text
Best validation accuracy: 82.30%  (0.82298, epoch 21)
```

These numbers are extracted from the notebook's saved outputs by `analytics/extract_notebook.py`.

---

# 📊 Evaluation

The model is evaluated at two different levels:

## 1. Segment-level evaluation

Each 3-second audio segment is classified individually.

Metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## 2. Song-level evaluation

All segments belonging to the same song are classified.

Their probability vectors are averaged:

```text
Segment 1 → [probabilities]
Segment 2 → [probabilities]
Segment 3 → [probabilities]
...
Segment N → [probabilities]

              ↓

Average probabilities

              ↓

Highest probability genre
```

This provides the final prediction for the complete song.

---

# 🏆 Results

## Final CNN V3 Performance

```text
Segment Accuracy: 83.26%

Macro Precision: 83.43%
Macro Recall:    83.26%
Macro F1-Score:  83.31%

Song-Level Accuracy: 90.00%
```

The most important application-level metric is:

# 🏆 90.00% Song-Level Accuracy

---

# 📋 CNN V3 Classification Report

| Genre | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Blues | 0.9556 | 0.8600 | 0.9053 | 150 |
| Classical | 0.9467 | 0.9467 | 0.9467 | 150 |
| Country | 0.7619 | 0.7467 | 0.7542 | 150 |
| Disco | 0.7628 | 0.7933 | 0.7778 | 150 |
| Hip-Hop | 0.8581 | 0.8523 | 0.8552 | 149 |
| Jazz | 0.8839 | 0.9133 | 0.8984 | 150 |
| Metal | 0.8808 | 0.8867 | 0.8837 | 150 |
| Pop | 0.7192 | 0.7000 | 0.7095 | 150 |
| Reggae | 0.8808 | 0.8867 | 0.8837 | 150 |
| Rock | 0.6937 | 0.7400 | 0.7161 | 150 |

Overall:

```text
Accuracy = 83.26%
Macro F1 = 83.31%
Weighted F1 = 83.30%
```

---

# 📊 Confusion Matrix Analysis

The segment-level confusion matrix shows that CNN V3 performs particularly well on:

- Classical
- Blues
- Jazz
- Metal
- Reggae

The more difficult genres include:

- Pop
- Rock
- Country
- Disco

This is expected because some genres have overlapping musical characteristics.

For example:

```text
Rock ↔ Pop
Country ↔ Disco
Country ↔ Pop
Hip-Hop ↔ Metal
```

The model is learning meaningful genre characteristics, but some musical styles naturally share similar spectral patterns.

---

# 🎶 Song-Level Prediction

A major feature of this project is that the system does not simply classify one arbitrary 3-second portion and call that the genre of the complete song.

Instead:

```text
30-second song
       ↓
Multiple segments
       ↓
Mel Spectrogram for each segment
       ↓
CNN prediction for each segment
       ↓
Probability aggregation
       ↓
Final song prediction
```

For example:

```text
Segment 1 → Rock 0.82
Segment 2 → Rock 0.91
Segment 3 → Rock 0.88
Segment 4 → Pop 0.06
...
       ↓
Aggregated prediction
       ↓
Rock
```

This improves robustness because an individual segment may not contain enough information to represent the entire song.

---

# 🎯 Why Song-Level Accuracy Matters

For a real music application, the user does not want:

> "This 3-second section is probably Rock."

The user wants:

> "This song is Rock."

Therefore, song-level evaluation is a better representation of the intended application.

The final model achieved:

```text
135 / 150 × 100 = 90%
```

on the song-level test evaluation (150 held-out songs, 15 per genre).

---

# 📊 Model Comparison

| Model | Main Improvement | Segment Accuracy | Song Accuracy |
|---|---|---:|---:|
| CNN V1 | Baseline CNN | 73.38% | — |
| CNN V2 | Global Average Pooling | 81.85% | 86.67% |
| **CNN V3** | **SpecAugment + deeper CNN + regularization** | **83.26%** | **90.00%** |

### Improvement

CNN V2 → CNN V3:

```text
86.67% → 90.00%

Improvement = +3.33 percentage points
```

CNN V1 → CNN V3:

```text
73.38% → 83.26%

Improvement = +9.88 percentage points
```

At the practical song-level evaluation:

```text
CNN V2 → 86.67%
CNN V3 → 90.00%
```

Therefore, **CNN V3 is selected as the final model**.

---

# 📁 Project Structure

```text
Music-Genre-CNN/
│
├── Music_Genre_CNN.ipynb      # data prep, CNN V1/V2/V3 training, evaluation
├── final_results.json         # final metrics
│
├── src/
│   ├── preprocessing.py       # audio → Mel segments (same pipeline as the notebook)
│   ├── predict.py             # model loading, SpecAugment layer, song-level prediction
│   └── api.py                 # FastAPI service (also serves the built frontend)
│
├── frontend/                  # React + Vite + TypeScript web UI
├── tests/                     # pytest suite
│
├── models/
│   ├── music_genre_cnn_final_v3.keras   # final CNN V3 (~5 MB, tracked in Git)
│   └── README.md
│
├── Dockerfile
├── requirements.txt           # notebook / training dependencies
├── requirements-api.txt       # API dependencies
├── LICENSE
└── README.md
```

The original audio dataset and the larger V1/V2 model files are not committed.

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/PulkitTiwari87/Music-Genre-CNN.git
cd Music-Genre-CNN
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Core libraries used:

```text
Python
NumPy
Pandas
TensorFlow
Keras
Librosa
Scikit-learn
Matplotlib
Seaborn
tqdm
KaggleHub
```

`requirements.txt` (for the notebook) contains the list below; the web API has its own pinned `requirements-api.txt`.

```text
numpy
pandas
matplotlib
seaborn
librosa
scikit-learn
tensorflow
tqdm
kagglehub
```

---

# ▶️ Running the Project

## Option 1 — Google Colab

The project was developed and trained in Google Colab.

Recommended environment:

```text
Google Colab
+
T4 GPU
```

Open `Music_Genre_CNN.ipynb` in Colab (use the badge at the top of the notebook), then run the cells in sequence.

---

## Option 2 — Local Machine

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset separately.

Set the dataset path:

```python
DATASET_PATH = "path/to/genres_original"
```

Run the preprocessing and training sections.

A GPU is recommended for training.

---

# 🌐 Web Application

A FastAPI service wraps the trained CNN V3, and a React + TypeScript (Vite) page lets you upload a song and see the predicted genre with per-genre probabilities. In production FastAPI serves the built frontend, so there is one process and one port.

## Run locally

Requires Python 3.12 and Node 22.13+.

```bash
pip install -r requirements-api.txt
cd frontend && npm ci && npm run build && cd ..
uvicorn src.api:app
```

Open <http://127.0.0.1:8000>.

For development with hot reload, run `uvicorn src.api:app --reload` and, in `frontend/`, `npm run dev` (<http://localhost:5173>, proxies `/api` to port 8000).

## API

| Endpoint | Description |
|---|---|
| `GET /api/health` | `200` when the model is loaded, `503` otherwise |
| `POST /api/predict` | Multipart field `file`: WAV, MP3, FLAC or OGG, at most 25 MB and at least 3 s. Only the first 60 seconds are analysed. |

```json
{
  "genre": "rock",
  "confidence": 0.63,
  "probabilities": { "blues": 0.02, "classical": 0.0, "...": "..." },
  "segments": 10,
  "duration_seconds": 30.0
}
```

Errors: `413` file too large, `415` not decodable as audio, `422` empty/too short/missing file, `503` model not loaded.

The service decodes uploads in memory (nothing is written to disk). Authentication and rate limiting are not included; put the service behind a gateway if you expose it publicly.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/music_genre_cnn_final_v3.keras` | Model file to serve |
| `PORT` | `8000` | Port used by the Docker image (Render sets its own) |
| `CORS_ORIGINS` | *(unset)* | Comma-separated UI origins allowed to call the API. Only needed when the UI is hosted elsewhere. |
| `VITE_API_URL` | *(unset)* | Build-time, frontend only: base URL of the API (e.g. `https://my-api.onrender.com`). Unset = same origin. |

## Deployment (Render + Vercel)

TensorFlow is far too large for a Vercel function, so the API runs on **Render** (from the `Dockerfile`) and the static UI runs on **Vercel** (root directory `frontend`):

1. Render web service (Docker): set `CORS_ORIGINS` to the Vercel URL.
2. Vercel project: set `VITE_API_URL` to the Render URL, then redeploy.

The free Render plan has 512 MB RAM. It runs this service, but with limited headroom (measured about 450 MB steady and ~500 MB at the worst-case upload; memory is flat across requests, there is no leak), and a 30 s clip takes roughly 10 s to analyse. Upgrade the plan if you need more. It also sleeps after 15 minutes idle (the first request then takes about a minute).

## Docker

```bash
docker build -t music-genre-cnn .
docker run -p 8000:8000 music-genre-cnn
```

> Status: the Dockerfile is provided but has not been build-tested yet.

## Tests

```bash
pip install pytest httpx librosa   # librosa is only used by the parity tests
pytest                             # preprocessing parity with the notebook, model loading, API
cd frontend && npm test       # UI behaviour
```

The notebook computed features with librosa. `src/preprocessing.py` reproduces them with numpy, soundfile and soxr only, because librosa's numba/scipy stack costs ~130 MB of RAM. The parity test executes the notebook's own `extract_segments` cell and checks the features match (decoding is bit-identical; features differ by at most ~2e-6 on a 0–1 scale; the model's prediction differs by ~1e-7).

---

# ☁️ Google Colab Workflow

The project was developed using Google Colab because it provides access to GPU acceleration.

The workflow was:

```text
Google Colab
    ↓
Kaggle dataset
    ↓
Feature extraction
    ↓
NumPy feature dataset
    ↓
CNN training
    ↓
GPU acceleration
    ↓
Model evaluation
    ↓
Google Drive model storage
    ↓
GitHub notebook
```

The trained models were stored in Google Drive during development.

---

# 💾 Saved Models

The model-development process produced three versions:

```text
CNN_V1_best.keras
CNN_V2_best.keras
CNN_V3_best.keras
```

The final production candidate is:

```text
music_genre_cnn_final_v3.keras
```

The final model (~5 MB) is committed as `models/music_genre_cnn_final_v3.keras` and is what the API serves. The V1 model (~75 MB) is not committed; for larger models use Git LFS, Hugging Face Hub, or object storage.

---

# 🔐 `.gitignore`

Large and generated files are excluded from Git (`*.keras`, `*.h5`, `*.npz`, `*.npy`, `*.pkl`, `data/`, `datasets/`, virtual environments, `node_modules/`), with one exception for the final serving model, `models/music_genre_cnn_final_v3.keras`.

The original GTZAN audio files must not be uploaded to this repository.

---

# ⚠️ Limitations

Although the final model achieved 90% song-level accuracy on the project test set, several limitations remain.

### 1. Dataset size

The project uses only 1,000 songs.

### 2. Genre overlap

Some genres share musical characteristics.

### 3. Dataset bias

GTZAN is a relatively small benchmark dataset and may not represent the full diversity of modern music.

### 4. Audio format

The model was trained on 30-second WAV clips. The API also accepts MP3, FLAC and OGG, and analyses at most the first 60 seconds of a file, but accuracy on compressed or full-length recordings was not measured.

### 5. Generalization

Performance on songs outside the GTZAN distribution may be lower.

### 6. Segment-level vs song-level performance

Segment-level accuracy is lower than song-level accuracy because individual short segments may not contain enough information to identify the genre reliably.

### 7. Model size

The CNN is suitable for experimentation and deployment, but further optimization may be required for lightweight mobile or edge deployment.

---

# 🔮 Future Improvements

Possible improvements include:

## 🎧 Better Audio Support

- Benchmark accuracy on MP3/FLAC/OGG and full-length songs (the API accepts them, but they are not evaluated)
- M4A/AAC support (needs ffmpeg)

## 🧠 Advanced Models

Experiment with:

- CRNN
- LSTM
- GRU
- ResNet
- EfficientNet
- Audio Spectrogram Transformers
- Transfer learning

## 🎵 More Audio Features

Combine Mel Spectrograms with:

- MFCC
- Chroma features
- Spectral centroid
- Spectral bandwidth
- Spectral contrast
- Zero-crossing rate
- Tempo

## 🧪 Better Augmentation

Experiment with:

- Time stretching
- Pitch shifting
- Noise injection
- Time masking
- Frequency masking
- Mixup

## 🚀 Production hardening

- Authentication and rate limiting for the API
- Background jobs for very long audio files
- Measuring accuracy on real-world (non-GTZAN) music

## ☁️ Cloud Deployment

Possible deployment options include:

- Docker
- AWS
- Google Cloud
- Azure
- Render
- Railway
- Hugging Face Spaces

---

# 🏗️ Application Architecture

The trained model is served as a full-stack application (see [Web Application](#-web-application)).

```text
┌──────────────────────────┐
│       React Frontend     │
│                          │
│  Upload Music File       │
└────────────┬─────────────┘
             │
             │ HTTP POST
             ▼
┌──────────────────────────┐
│       FastAPI Backend    │
│                          │
│ /api/predict             │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    Audio Preprocessor    │
│                          │
│ Librosa                  │
│ Resampling               │
│ Segmentation             │
│ Mel Spectrogram          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│       CNN V3 Model       │
│                          │
│ Genre probabilities      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Song-Level Aggregation   │
│                          │
│ Average probabilities    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│       Prediction         │
│                          │
│ Genre: Rock              │
│ Confidence: 90%          │
└──────────────────────────┘
```

---

# 🔁 Reproducibility

To reproduce the experiment:

1. Obtain the GTZAN dataset.
2. Configure the dataset path.
3. Install the required Python libraries.
4. Run the feature extraction pipeline.
5. Generate the Mel Spectrogram features.
6. Split the data into training, validation, and testing sets.
7. Train CNN V1.
8. Train CNN V2.
9. Train CNN V3 with SpecAugment.
10. Evaluate segment-level performance.
11. Evaluate song-level performance.
12. Save the best model.

For deterministic experiments, random seeds should be configured for Python, NumPy, and TensorFlow.

---

# 📌 Important ML Design Decisions

## Why CNN?

CNNs are effective at detecting local spatial patterns.

A Mel Spectrogram behaves similarly to an image:

```text
X-axis → Time
Y-axis → Frequency
Pixel value → Energy
```

CNN filters can therefore learn patterns related to:

- Rhythm
- Harmonic structures
- Frequency distributions
- Instrument characteristics
- Temporal-spectral patterns

---

## Why Mel Spectrogram?

Human hearing is not linearly sensitive to frequency.

The Mel scale provides a perceptually motivated representation of frequency.

Therefore, Mel Spectrograms provide a useful representation for music/audio classification.

---

## Why Multiple CNN Versions?

Instead of presenting only one model, this project follows an experimental approach:

```text
Baseline
   ↓
Improvement
   ↓
Regularization
   ↓
Final Model
```

This makes it possible to measure whether each architectural change actually improves performance.

---

## Why Global Average Pooling?

CNN V1 used Flatten, which produced a large number of parameters in the dense layer.

Global Average Pooling summarizes spatial feature maps and reduces the number of parameters.

This can help:

- Reduce overfitting
- Reduce model complexity
- Improve generalization

---

## Why SpecAugment?

SpecAugment modifies spectrograms during training by masking portions of the time and frequency dimensions.

This forces the model to learn more robust features rather than memorizing exact spectrogram patterns.

---

# 📈 Final Conclusion

This project demonstrates an end-to-end deep learning workflow for music genre classification.

The system progresses from raw audio to Mel Spectrograms and then uses CNN-based deep learning to classify music into 10 genres.

Three model versions were developed:

```text
CNN V1 → 73.38% segment accuracy

CNN V2 → 81.85% segment accuracy
          86.67% song accuracy

CNN V3 → 83.26% segment accuracy
          90.00% song accuracy
```

The final CNN V3 model combines:

- Convolutional feature extraction
- Batch normalization
- Max pooling
- Global average pooling
- L2 regularization
- Dropout
- SpecAugment
- Early stopping
- Learning-rate scheduling

The final system achieved:

> **90.00% song-level accuracy on the held-out test set.**

The project provides a foundation for building a complete music intelligence application capable of accepting user-uploaded audio and returning predicted genres with confidence scores.

---

# 👨‍💻 Author

## Pulkit Tiwari

**B.Tech Computer Science & Engineering**  
**UPES, Dehradun**

Areas of interest:

- Machine Learning
- Deep Learning
- Full-Stack Development
- Cybersecurity
- Artificial Intelligence

---

# ⭐ Project Status

```text
[████████████████████] 100% ML Model Development

✅ Dataset integration
✅ Audio preprocessing
✅ Mel Spectrogram extraction
✅ Feature generation
✅ Train/validation/test split
✅ CNN V1
✅ CNN V2
✅ CNN V3
✅ SpecAugment
✅ Model evaluation
✅ Classification report
✅ Confusion matrices
✅ Song-level evaluation
✅ Final model selection

✅ FastAPI inference API
✅ React frontend
✅ User audio upload and prediction
✅ Automated tests (pytest + Vitest)
🚧 Docker image (Dockerfile provided, not build-tested)
🚧 Hosted deployment
```

---

## ⭐ If you found this project useful

Consider starring the repository and following the project as it evolves from a research notebook into a complete deployed music genre classification application.

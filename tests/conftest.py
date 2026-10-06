import io

import numpy as np
import pytest
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)

from src.predict import SpecAugment, load_model
from src.preprocessing import SAMPLE_RATE

tf.get_logger().setLevel("ERROR")


def make_audio(seconds, sr=SAMPLE_RATE, seed=0):
    """A few sine tones plus noise: deterministic, non-silent test audio."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    wave = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)
    return (wave + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


def encode(audio, fmt, sr=SAMPLE_RATE):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def stub_model_path(tmp_path_factory):
    """CNN V3 architecture (notebook cell 136) with random weights, saved as .keras.

    Exercises the real save/load path incl. the SpecAugment custom layer. It says nothing
    about accuracy; that is covered by the real-model tests when models/ has the file.
    """
    blocks = [
        layer
        for filters in (32, 64, 128, 256)
        for layer in (
            Conv2D(filters, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
        )
    ]
    model = Sequential(
        [Input(shape=(128, 130, 1)), SpecAugment(12, 15)]
        + blocks
        + [
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(10, activation="softmax"),
        ]
    )
    path = tmp_path_factory.mktemp("model") / "stub.keras"
    model.save(path)
    return path


@pytest.fixture(scope="session")
def stub_model(stub_model_path):
    return load_model(stub_model_path)

"""Load the trained CNN V3 and predict a song's genre by averaging segment probabilities."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .preprocessing import GENRES, SAMPLE_RATE, extract_segments


class SpecAugment(layers.Layer):
    """Frequency + time masking (training only). Required to deserialize the model."""

    def __init__(self, freq_mask_param=12, time_mask_param=15, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    @staticmethod
    def _keep_mask(size, max_width, axis_shape):
        width = tf.random.uniform([], 0, max_width + 1, dtype=tf.int32)
        start = tf.random.uniform(
            [], 0, tf.maximum(1, size - width + 1), dtype=tf.int32
        )
        idx = tf.range(size)
        masked = tf.logical_and(idx >= start, idx < start + width)
        return tf.reshape(tf.cast(~masked, tf.float32), axis_shape)

    def call(self, inputs, training=None):
        if training is False:
            return inputs
        # inputs: (batch, frequency, time, channels)
        shape = tf.shape(inputs)
        x = inputs * tf.cast(
            self._keep_mask(shape[1], self.freq_mask_param, [1, -1, 1, 1]), inputs.dtype
        )
        return x * tf.cast(
            self._keep_mask(shape[2], self.time_mask_param, [1, 1, -1, 1]), inputs.dtype
        )


def load_model(path):
    return tf.keras.models.load_model(path, custom_objects={"SpecAugment": SpecAugment})


def predict_song(model, audio):
    """Classify a waveform (22,050 Hz mono). Returns genre, confidence and all probabilities."""
    segments = extract_segments(audio)
    probabilities = np.mean(model.predict(segments, batch_size=8, verbose=0), axis=0)
    best = int(np.argmax(probabilities))
    return {
        "genre": GENRES[best],
        "confidence": float(probabilities[best]),
        "probabilities": {g: float(p) for g, p in zip(GENRES, probabilities)},
        "segments": len(segments),
        "duration_seconds": round(len(audio) / SAMPLE_RATE, 2),
    }

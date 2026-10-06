"""Audio -> Mel-spectrogram segments, identical to the notebook's `extract_segments`."""

import librosa
import numpy as np

SAMPLE_RATE = 22050
SEGMENT_DURATION = 3
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Class order of the model output (LabelEncoder sorts names alphabetically).
GENRES = (
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
)


def load_audio(source, max_seconds=None):
    """Load a path or file-like object as mono audio at SAMPLE_RATE."""
    audio, _ = librosa.load(source, sr=SAMPLE_RATE, duration=max_seconds)
    return audio


def extract_segments(audio):
    """Split audio into non-overlapping 3 s segments and return their Mel features.

    Returns float32 array of shape (n_segments, N_MELS, 130, 1), scaled to [0, 1].
    Raises ValueError if the audio is shorter than one segment.
    """
    segments = []
    for start in range(0, len(audio) - SEGMENT_SAMPLES + 1, SEGMENT_SAMPLES):
        mel = librosa.feature.melspectrogram(
            y=audio[start : start + SEGMENT_SAMPLES],
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        mel_db = np.clip(librosa.power_to_db(mel, ref=np.max), -80, 0)
        segments.append(((mel_db + 80) / 80).astype(np.float32))

    if not segments:
        raise ValueError(
            f"Audio is too short: need at least {SEGMENT_DURATION} seconds."
        )
    return np.array(segments)[..., np.newaxis]

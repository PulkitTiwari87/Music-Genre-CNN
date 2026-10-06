"""Audio -> Mel-spectrogram segments, matching the notebook's `extract_segments`.

The notebook used librosa. This module reproduces the same numbers with numpy, soundfile
and soxr only: librosa pulls in scipy/numba (~130 MB RSS), which does not fit Render's
512 MB free tier next to TensorFlow. tests/test_preprocessing.py checks parity against
the notebook's own code and against librosa.
"""

import math

import numpy as np
import soundfile as sf
import soxr

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
    """Load a path or file-like object as mono float32 audio at SAMPLE_RATE."""
    with sf.SoundFile(source) as f:
        native_rate = f.samplerate
        frames = -1 if max_seconds is None else int(max_seconds * native_rate)
        # Downmix block by block so a stereo copy of the whole file is never held in memory.
        blocks = f.blocks(
            blocksize=5 * native_rate, frames=frames, dtype="float32", always_2d=True
        )
        mono = [block.mean(axis=1) for block in blocks]
    audio = np.concatenate(mono) if mono else np.zeros(0, np.float32)
    if native_rate != SAMPLE_RATE:
        wanted = math.ceil(len(audio) * SAMPLE_RATE / native_rate)
        audio = soxr.resample(audio, native_rate, SAMPLE_RATE, quality="HQ")
        audio = np.pad(audio[:wanted], (0, max(0, wanted - len(audio))))
    return audio.astype(np.float32, copy=False)


def _hz_to_mel(freq):
    """Slaney mel scale: linear below 1 kHz, logarithmic above."""
    freq = np.asarray(freq, dtype=np.float64)
    log_part = 15.0 + np.log(np.maximum(freq, 1000.0) / 1000.0) / (math.log(6.4) / 27.0)
    return np.where(freq >= 1000.0, log_part, freq / (200.0 / 3.0))


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    log_part = 1000.0 * np.exp((math.log(6.4) / 27.0) * (mel - 15.0))
    return np.where(mel >= 15.0, log_part, mel * (200.0 / 3.0))


def _mel_filterbank():
    """(N_MELS, 1 + N_FFT // 2) triangular filters with Slaney area normalisation."""
    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, 1 + N_FFT // 2)
    mel_freqs = _mel_to_hz(
        np.linspace(_hz_to_mel(0.0), _hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2)
    )
    ramps = np.subtract.outer(mel_freqs, fft_freqs)
    lower = -ramps[:-2] / np.diff(mel_freqs)[:-1, None]
    upper = ramps[2:] / np.diff(mel_freqs)[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    return weights * (2.0 / (mel_freqs[2 : N_MELS + 2] - mel_freqs[:N_MELS]))[:, None]


_MEL_FILTERS = _mel_filterbank().astype(np.float32)
# Periodic Hann window, as used by librosa's STFT.
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def _mel_db(segment):
    """Power Mel spectrogram in dB relative to its own peak, clipped to [-80, 0]."""
    padded = np.pad(
        segment, N_FFT // 2
    )  # centred frames, zero padding (librosa >= 0.10)
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    power = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2  # (time, freq)
    mel = _MEL_FILTERS @ power.T.astype(np.float32)  # (mel, time)
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db -= 10.0 * np.log10(max(1e-10, float(mel.max())))
    return np.clip(db, -80.0, 0.0)


def extract_segments(audio):
    """Split audio into non-overlapping 3 s segments and return their Mel features.

    Returns float32 array of shape (n_segments, N_MELS, 130, 1), scaled to [0, 1].
    Raises ValueError if the audio is shorter than one segment.
    """
    segments = [
        ((_mel_db(audio[start : start + SEGMENT_SAMPLES]) + 80) / 80).astype(np.float32)
        for start in range(0, len(audio) - SEGMENT_SAMPLES + 1, SEGMENT_SAMPLES)
    ]
    if not segments:
        raise ValueError(
            f"Audio is too short: need at least {SEGMENT_DURATION} seconds."
        )
    return np.array(segments)[..., np.newaxis]

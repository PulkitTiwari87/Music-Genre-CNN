import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.preprocessing import GENRES, SAMPLE_RATE, extract_segments, load_audio
from tests.conftest import make_audio

ROOT = Path(__file__).resolve().parent.parent


def test_segments_shape_range_and_count():
    segments = extract_segments(make_audio(30.0))
    assert segments.shape == (10, 128, 130, 1)
    assert segments.dtype == np.float32
    assert 0.0 <= segments.min() and segments.max() <= 1.0


def test_trailing_partial_segment_is_dropped():
    assert extract_segments(make_audio(7.9)).shape[0] == 2


def test_too_short_audio_raises():
    with pytest.raises(ValueError, match="too short"):
        extract_segments(make_audio(2.9))


def test_load_audio_resamples_to_22050_mono(tmp_path):
    path = tmp_path / "stereo44k.wav"
    sf.write(path, np.stack([make_audio(4, 44100)] * 2, axis=1), 44100)
    audio = load_audio(path)
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(4 * SAMPLE_RATE, abs=2)


def test_load_audio_respects_max_seconds(tmp_path):
    path = tmp_path / "long.wav"
    sf.write(path, make_audio(10), SAMPLE_RATE)
    assert len(load_audio(path, max_seconds=5)) == pytest.approx(5 * SAMPLE_RATE, abs=2)


def test_genres_match_label_encoder_order_and_results_file():
    assert list(GENRES) == sorted(GENRES)
    results = json.loads((ROOT / "final_results.json").read_text())
    assert list(GENRES) == results["genres"]


def test_matches_notebook_extract_segments(tmp_path):
    """Run the notebook's own `extract_segments` cell and compare feature-for-feature."""
    notebook = json.loads((ROOT / "Music_Genre_CNN.ipynb").read_text(encoding="utf-8"))
    # The notebook redefines this function; the last definition is the one that was trained with.
    cell = [
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code"
        and "def extract_segments(file_path)" in "".join(c["source"])
    ][-1]
    namespace = {}
    exec("import numpy as np\nimport librosa\n" + cell, namespace)  # noqa: S102

    path = tmp_path / "clip.wav"
    sf.write(path, make_audio(10.5), SAMPLE_RATE)

    expected = np.array(namespace["extract_segments"](str(path)))[..., np.newaxis]
    actual = extract_segments(load_audio(path))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-6)

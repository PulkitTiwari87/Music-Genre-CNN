from pathlib import Path

import numpy as np
import pytest

from src.predict import load_model, predict_song
from src.preprocessing import GENRES
from tests.conftest import make_audio

REAL_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "music_genre_cnn_final_v3.keras"
)


def assert_valid_prediction(result, expected_segments):
    assert result["genre"] in GENRES
    assert list(result["probabilities"]) == list(GENRES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert result["confidence"] == pytest.approx(max(result["probabilities"].values()))
    assert result["probabilities"][result["genre"]] == result["confidence"]
    assert result["segments"] == expected_segments


def test_stub_model_loads_with_specaugment_and_predicts(stub_model):
    assert stub_model.input_shape == (None, 128, 130, 1)
    assert stub_model.output_shape == (None, 10)
    result = predict_song(stub_model, make_audio(30.0))
    assert_valid_prediction(result, 10)
    assert result["duration_seconds"] == 30.0


def test_inference_is_deterministic_so_specaugment_is_off(stub_model):
    audio = make_audio(9.0)
    first = predict_song(stub_model, audio)["probabilities"]
    second = predict_song(stub_model, audio)["probabilities"]
    assert first == second


def test_short_audio_is_rejected(stub_model):
    with pytest.raises(ValueError):
        predict_song(stub_model, make_audio(1.0))


@pytest.mark.skipif(
    not REAL_MODEL.exists(), reason="trained model not present in models/"
)
def test_real_model_loads_and_predicts():
    model = load_model(REAL_MODEL)
    assert model.input_shape == (None, 128, 130, 1)
    assert model.output_shape == (None, 10)
    assert (
        model.count_params() == 423_946
    )  # as reported by the notebook's model.summary()
    assert_valid_prediction(predict_song(model, make_audio(30.0)), 10)
    np.testing.assert_allclose(
        model.predict(np.zeros((1, 128, 130, 1), np.float32), verbose=0).sum(),
        1.0,
        atol=1e-5,
    )

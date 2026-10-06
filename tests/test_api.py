import pytest
from fastapi.testclient import TestClient

from src import api
from tests.conftest import encode, make_audio


@pytest.fixture(scope="module")
def client(stub_model_path):
    with TestClient(api.create_app(stub_model_path)) as c:
        yield c


def post(client, data, name="clip.wav", content_type="audio/wav"):
    return client.post("/api/predict", files={"file": (name, data, content_type)})


def test_health_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert len(response.json()["genres"]) == 10


@pytest.mark.parametrize("fmt", ["WAV", "FLAC", "OGG", "MP3"])
def test_predict_supported_formats(client, fmt):
    response = post(client, encode(make_audio(10.0), fmt), name=f"clip.{fmt.lower()}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["segments"] == 3
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["genre"] in body["probabilities"]


def test_garbage_file_is_unsupported_media(client):
    assert post(client, b"this is not audio" * 100, "x.mp3").status_code == 415


def test_empty_file_rejected(client):
    assert post(client, b"").status_code == 422


def test_too_short_audio_rejected(client):
    response = post(client, encode(make_audio(1.0), "WAV"))
    assert response.status_code == 422
    assert "too short" in response.json()["detail"]


def test_missing_file_field_rejected(client):
    assert client.post("/api/predict").status_code == 422


def test_oversized_upload_rejected(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 1024)
    assert post(client, b"\0" * 2048).status_code == 413


def test_model_missing_gives_503(tmp_path):
    with TestClient(api.create_app(tmp_path / "missing.keras")) as c:
        assert c.get("/api/health").status_code == 503
        assert post(c, encode(make_audio(5.0), "WAV")).status_code == 503

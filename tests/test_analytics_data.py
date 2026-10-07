"""Every number the /brag and /model pages show must trace back to the notebook or the model file."""

import json
from pathlib import Path

import pytest

from analytics import extract_notebook as ex

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "frontend" / "src" / "data"


@pytest.fixture(scope="module")
def notebook_json():
    return json.loads((DATA / "notebook.json").read_text(encoding="utf-8"))


def test_committed_notebook_json_is_what_the_extractor_produces():
    expected = json.dumps(ex.build(), indent=1, ensure_ascii=False) + "\n"
    assert (DATA / "notebook.json").read_text(encoding="utf-8") == expected, (
        "frontend/src/data/notebook.json is stale: run `python analytics/extract_notebook.py`"
    )


def test_results_agree_with_final_results_json(notebook_json):
    final = json.loads((ROOT / "final_results.json").read_text())
    results = notebook_json["results"]
    assert results["v1"]["segment_accuracy"] == pytest.approx(
        final["models"]["CNN_V1"]["segment_accuracy"], abs=1e-3
    )
    assert results["v2"]["segment_accuracy"] == pytest.approx(
        final["models"]["CNN_V2"]["segment_accuracy"], abs=1e-9
    )
    assert results["v2"]["song_accuracy"] == pytest.approx(
        final["models"]["CNN_V2"]["song_accuracy"], abs=1e-4
    )
    assert results["v3"]["segment_accuracy"] == pytest.approx(
        final["models"]["CNN_V3"]["segment_accuracy"], abs=1e-12
    )
    assert (
        results["v3"]["song_accuracy"]
        == final["models"]["CNN_V3"]["song_accuracy"]
        == final["final_song_accuracy"]
    )
    assert notebook_json["dataset"]["classes"] == final["genres"]


@pytest.mark.parametrize(
    "key,total",
    [
        ("v1_segment", 1499),
        ("v2_segment", 1499),
        ("v3_segment", 1499),
        ("v2_song", 150),
        ("v3_song", 150),
    ],
)
def test_confusion_matrices_are_internally_consistent(notebook_json, key, total):
    record = notebook_json["confusion"][key]
    matrix = record["matrix"]
    assert sum(map(sum, matrix)) == total == record["total"]
    assert all(len(row) == 10 for row in matrix) and len(matrix) == 10
    assert all(cell >= 0 for row in matrix for cell in row)
    accuracy = sum(matrix[i][i] for i in range(10)) / total
    assert accuracy == pytest.approx(record["accuracy"])
    version = key.split("_")[0]
    level = "song_accuracy" if key.endswith("song") else "segment_accuracy"
    assert accuracy == pytest.approx(notebook_json["results"][version][level], abs=5e-5)


def test_v3_matrix_reproduces_the_classification_report(notebook_json):
    matrix = notebook_json["confusion"]["v3_segment"]["matrix"]
    report = notebook_json["reports"]["v3_segment"]["per_class"]
    for i, name in enumerate(notebook_json["dataset"]["classes"]):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        assert support == report[name]["support"]
        assert matrix[i][i] / support == pytest.approx(report[name]["recall"], abs=6e-5)
        assert matrix[i][i] / predicted == pytest.approx(
            report[name]["precision"], abs=6e-5
        )


def test_a_misread_matrix_would_be_rejected():
    source = json.loads(ex.CONFUSION.read_text())
    entry = source["matrices"]["v3_song"]
    entry["matrix"][0][0] += 1  # one wrongly transcribed digit
    with pytest.raises(ex.CheckError):
        ex.validate_confusion("v3_song", entry, source["labels"], {}, {"v3_song": 0.9})


def test_dataset_and_training_facts(notebook_json):
    data = notebook_json["dataset"]
    assert (
        data["total_songs"]
        == sum(data["songs"].values())
        == sum(data["songs_per_class"].values())
        == 1000
    )
    assert data["songs"] == {"train": 700, "val": 150, "test": 150}
    assert data["segments"] == {"train": 6985, "val": 1497, "test": 1499}
    for run in notebook_json["training"].values():
        assert [e["epoch"] for e in run["epochs"]] == list(
            range(1, run["epochs_run"] + 1)
        )
        assert run["epochs_run"] <= run["max_epochs"]
    v3 = notebook_json["training"]["v3"]
    assert v3["best_val_loss_epoch"] == 22 and v3["best_val_accuracy_epoch"] == 21
    assert v3["best_checkpoint_val_accuracy"] == pytest.approx(0.82298)


def test_a_committed_eval_export_is_real_and_matches_the_recorded_run(notebook_json):
    """analytics/export_analytics.py output may be committed, but never synthetic test data."""
    path = DATA.parent.parent / "public" / "analytics" / "eval.json"
    if not path.exists():
        pytest.skip(
            "no eval.json committed yet (the pages show 'Not available' for those panels)"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["provenance"]["synthetic"] is False, (
        "eval.json is synthetic test data; delete it"
    )
    check = data["split_check"]
    assert check["matches"] is True, (
        "export was made on a split that differs from the evaluated one"
    )
    assert check["fingerprint_ok"] is True
    assert abs(check["delta_segments"]) <= 5
    assert check["n_test_segments"] == notebook_json["dataset"]["segments"]["test"]
    assert check["n_test_songs"] == notebook_json["dataset"]["songs"]["test"]
    assert data["classes"] == notebook_json["dataset"]["classes"]

    # The export's matrices must agree with the figures transcribed from the notebook, up to the
    # recorded drift (one flipped prediction moves two cells by one each).
    def distance(a, b):
        return sum(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))

    for level, key in (("segment", "v3_segment"), ("song", "v3_song")):
        gap = distance(
            data[level]["confusion"], notebook_json["confusion"][key]["matrix"]
        )
        assert gap <= 2 * 5, (
            f"{level} confusion differs from the notebook figure by {gap} cells"
        )


def test_split_fingerprint_is_what_the_notebook_printed():
    from analytics.export_analytics import EXPECTED_FIRST_TEST_FILES

    text = ex.output_text(ex.load_notebook(), 116)
    printed = [line.split()[1] for line in text.splitlines()[1:6]]
    assert list(EXPECTED_FIRST_TEST_FILES) == printed


def test_exporter_pins_the_deployed_models_weights():
    """The exporter refuses any other model, so the pinned hash must be the committed model's."""
    model_path = ROOT / "models" / "music_genre_cnn_final_v3.keras"
    if not model_path.exists():
        pytest.skip("trained model not present")
    from analytics.export_analytics import (
        DEPLOYED_MODEL_WEIGHTS_SHA256,
        weights_fingerprint,
    )
    from src.predict import load_model

    assert weights_fingerprint(load_model(model_path)) == DEPLOYED_MODEL_WEIGHTS_SHA256


def test_model_json_matches_the_real_model_file():
    model_path = ROOT / "models" / "music_genre_cnn_final_v3.keras"
    if not model_path.exists():
        pytest.skip("trained model not present")
    from src.predict import load_model

    described = json.loads((DATA / "model.json").read_text())
    model = load_model(model_path)
    assert model.count_params() == described["parameters"]["total"]
    assert len(model.layers) == len(described["layers"])
    assert described["input_shape"] == list(model.input_shape[1:])
    assert described["file_size_bytes"] == model_path.stat().st_size
    nb = json.loads((DATA / "notebook.json").read_text())
    assert nb["models"]["v3"]["total"] == described["parameters"]["total"]

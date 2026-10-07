"""The Colab exporter must reproduce sklearn's numbers from the model's own probabilities."""

import base64
import json

import numpy as np
import pytest
import soundfile as sf

pd = pytest.importorskip("pandas")
pytest.importorskip("librosa")
pytest.importorskip("sklearn")

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from analytics import export_analytics as ex
from src.preprocessing import GENRES

SPLITS = {"train": 3, "val": 2, "test": 4}  # songs per genre; 6 s each = 2 segments


def make_song(path, genre_index, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(6 * ex.SAMPLE_RATE) / ex.SAMPLE_RATE
    wave = sum(
        np.sin(2 * np.pi * (150 + 90 * genre_index) * (h + 1) * t) / (h + 1)
        for h in range(3)
    )
    sf.write(
        path,
        0.2 * wave + (0.02 + 0.01 * genre_index) * rng.standard_normal(len(t)),
        ex.SAMPLE_RATE,
    )


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("audio")
    frames = {}
    seed = 0
    for split, per_genre in SPLITS.items():
        rows = []
        for g, genre in enumerate(GENRES):
            for n in range(per_genre):
                path = root / f"{genre}.{split}{n}.wav"
                make_song(path, g, seed)
                rows.append({"file_path": str(path), "genre": genre})
                seed += 1
        frames[split] = pd.DataFrame(rows)
    segs = np.concatenate(
        [ex._segments(ex._load_audio(p)) for p in frames["test"]["file_path"]]
    )
    X_test = segs[..., np.newaxis]
    y_onehot = np.concatenate(
        [
            np.repeat(np.eye(10)[[GENRES.index(g)]], 2, axis=0)
            for g in frames["test"]["genre"]
        ]
    )
    return frames, X_test, y_onehot


@pytest.fixture(scope="module")
def exported(dataset, stub_model, tmp_path_factory):
    frames, X_test, y_onehot = dataset
    out = tmp_path_factory.mktemp("out") / "analytics_eval.json"
    ex.export_analytics(
        stub_model,
        X_test,
        y_onehot,
        frames["train"],
        frames["val"],
        frames["test"],
        list(GENRES),
        out_path=str(out),
        expected_segment_accuracy=None,
        log=lambda *_: None,
        provenance={"synthetic": True, "source": "unit test"},
    )
    return json.loads(out.read_text())


def test_metrics_match_sklearn_on_the_models_own_probabilities(
    exported, dataset, stub_model
):
    _, X_test, y_onehot = dataset
    prob = stub_model.predict(X_test, verbose=0).astype(np.float64)
    y_true = y_onehot.argmax(1)
    seg = exported["segment"]

    assert seg["n"] == len(y_true) == 80
    assert seg["accuracy"] == pytest.approx((prob.argmax(1) == y_true).mean(), abs=1e-6)
    assert (
        seg["confusion"]
        == confusion_matrix(y_true, prob.argmax(1), labels=range(10)).tolist()
    )
    for k, genre in enumerate(GENRES):
        assert seg["roc"]["per_class"][genre]["auc"] == pytest.approx(
            roc_auc_score(y_true == k, prob[:, k]), abs=1e-5
        )
        assert seg["pr"]["per_class"][genre]["ap"] == pytest.approx(
            average_precision_score(y_true == k, prob[:, k]), abs=1e-5
        )
    assert seg["roc"]["macro_auc"] == pytest.approx(
        roc_auc_score(y_true, prob, multi_class="ovr", average="macro"), abs=1e-5
    )
    assert seg["pr"]["macro_ap"] == pytest.approx(
        np.mean([average_precision_score(y_true == k, prob[:, k]) for k in range(10)]),
        abs=1e-5,
    )


def test_curves_are_valid_roc_and_pr_curves(exported):
    for level in ("segment", "song"):
        for roc in exported[level]["roc"]["per_class"].values():
            assert (
                roc["fpr"][0] == 0
                and roc["fpr"][-1] == 1
                and roc["tpr"][0] == 0
                and roc["tpr"][-1] == 1
            )
            assert np.all(np.diff(roc["fpr"]) >= 0) and np.all(np.diff(roc["tpr"]) >= 0)
            area = np.trapezoid(roc["tpr"], roc["fpr"])
            assert area == pytest.approx(roc["auc"], abs=2e-3)
        for pr in exported[level]["pr"]["per_class"].values():
            assert pr["recall"][0] == 1 and pr["precision"][-1] == 1
            assert pr["baseline"] == pytest.approx(
                pr["positives"] / exported[level]["n"], abs=1e-4
            )


def test_song_level_is_the_mean_of_segment_probabilities(exported, dataset, stub_model):
    frames, X_test, _ = dataset
    prob = stub_model.predict(X_test, verbose=0)
    songs = exported["song"]["songs"]
    assert len(songs) == 40 == exported["song"]["n"]
    for i, entry in enumerate(songs):
        expected = prob[2 * i : 2 * i + 2].mean(axis=0)
        np.testing.assert_allclose(entry["probabilities"], expected, atol=1e-4)
        assert entry["predicted"] == GENRES[int(expected.argmax())]
        assert entry["actual"] == frames["test"]["genre"].iloc[i]
    assert (
        exported["song"]["confusion"]
        == confusion_matrix(
            [GENRES.index(g) for g in frames["test"]["genre"]],
            [GENRES.index(s["predicted"]) for s in songs],
            labels=range(10),
        ).tolist()
    )


def test_confidence_histograms_account_for_every_sample(exported):
    for level in ("segment", "song"):
        conf = exported[level]["confidence"]
        n_correct = round(exported[level]["accuracy"] * exported[level]["n"])
        assert sum(conf["correct"]) == n_correct
        assert sum(conf["wrong"]) == exported[level]["n"] - n_correct
        assert sum(b["n"] for b in conf["reliability"]) == exported[level]["n"]


def test_images_embeddings_and_stats_are_consistent(exported):
    ds = exported["dataset_stats"]
    assert ds["n"] == 10 * sum(SPLITS.values()) and ds["failed"] == []
    assert set(ds["split"]) == set(SPLITS)
    for key in ("mfcc_pca", "mfcc_tsne"):
        assert len(exported["embeddings"][key]["x"]) == ds["n"]
    assert len(exported["embeddings"]["model_tsne"]["x"]) == 80
    assert -1 <= exported["embeddings"]["silhouette"]["cnn_embedding"] <= 1
    assert len(exported["mfcc_correlation"]["matrix"]) == 20
    for item in exported["examples"]:
        assert item["actual"] == item["predicted"]
    for item in exported["errors"]:
        assert item["actual"] != item["predicted"]
    for item in exported["examples"] + exported["errors"]:
        assert len(base64.b64decode(item["mel"]["data"])) == 128 * 130
        assert len(base64.b64decode(item["mfcc"]["data"])) == 20 * 130
        assert len(item["waveform"]["min"]) == len(item["waveform"]["max"]) == 600
    assert len(exported["class_mean_mel"]) == 10


def test_provenance_flags_synthetic_runs(exported):
    assert exported["provenance"]["synthetic"] is True
    assert exported["schema_version"] == 1


def test_refuses_to_export_when_the_split_is_not_the_evaluated_one(
    dataset, stub_model, tmp_path
):
    frames, X_test, y_onehot = dataset
    with pytest.raises(RuntimeError, match="not the split"):
        ex.export_analytics(
            stub_model,
            X_test,
            y_onehot,
            frames["train"],
            frames["val"],
            frames["test"],
            list(GENRES),
            out_path=str(tmp_path / "x.json"),
            log=lambda *_: None,
        )
    assert not (tmp_path / "x.json").exists()


def _run(dataset, stub_model, tmp_path, **kwargs):
    frames, X_test, y_onehot = dataset
    return ex.export_analytics(
        stub_model,
        X_test,
        y_onehot,
        frames["train"],
        frames["val"],
        frames["test"],
        list(GENRES),
        out_path=str(tmp_path / "out.json"),
        log=lambda *_: None,
        **kwargs,
    )


def test_one_flipped_segment_is_tolerated_and_recorded(
    dataset, stub_model, tmp_path, exported
):
    n = exported["segment"]["n"]
    recorded = (
        exported["segment"]["accuracy"] + 1 / n
    )  # as if one more segment had been right originally
    result = _run(
        dataset,
        stub_model,
        tmp_path,
        expected_segment_accuracy=recorded,
        expected_first_files=None,
    )
    check = result["split_check"]
    assert check["matches"] is True and check["exact"] is False
    assert (
        check["delta_segments"] == -1
        and check["max_drift_segments"] == ex.MAX_DRIFT_SEGMENTS
    )


def test_exact_reproduction_is_flagged_exact(dataset, stub_model, tmp_path, exported):
    recorded = exported["segment"]["accuracy"]
    result = _run(
        dataset,
        stub_model,
        tmp_path,
        expected_segment_accuracy=recorded,
        expected_first_files=None,
    )
    assert (
        result["split_check"]["exact"] is True
        and result["split_check"]["delta_segments"] == 0
    )


def test_larger_drift_is_refused(dataset, stub_model, tmp_path, exported):
    n = exported["segment"]["n"]
    recorded = exported["segment"]["accuracy"] + (ex.MAX_DRIFT_SEGMENTS + 3) / n
    with pytest.raises(RuntimeError, match="differs from the recorded"):
        _run(
            dataset,
            stub_model,
            tmp_path,
            expected_segment_accuracy=recorded,
            expected_first_files=None,
        )


def test_a_different_song_order_is_refused_even_if_accuracy_matches(
    dataset, stub_model, tmp_path, exported
):
    recorded = exported["segment"]["accuracy"]
    with pytest.raises(RuntimeError, match="not the split the notebook evaluated"):
        _run(
            dataset,
            stub_model,
            tmp_path,
            expected_segment_accuracy=recorded,
            expected_first_files=("rock.nonexistent.wav",),
        )


def test_the_fingerprint_matches_when_the_split_starts_with_the_recorded_songs(
    dataset, stub_model, tmp_path, exported
):
    frames, _, _ = dataset
    first = tuple(
        p.replace("\\", "/").split("/")[-1] for p in frames["test"]["file_path"].head(3)
    )
    result = _run(
        dataset,
        stub_model,
        tmp_path,
        expected_segment_accuracy=exported["segment"]["accuracy"],
        expected_first_files=first,
    )
    assert result["split_check"]["fingerprint_ok"] is True


def test_refuses_when_test_df_does_not_match_x_test(dataset, stub_model, tmp_path):
    frames, X_test, y_onehot = dataset
    shuffled = frames["test"].iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="does not match X_test"):
        ex.export_analytics(
            stub_model,
            X_test,
            y_onehot,
            frames["train"],
            frames["val"],
            shuffled,
            list(GENRES),
            out_path=str(tmp_path / "y.json"),
            expected_segment_accuracy=None,
            log=lambda *_: None,
        )

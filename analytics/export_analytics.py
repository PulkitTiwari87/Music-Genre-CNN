"""Export the evaluation analytics that power the /brag and /model pages.

Run this in the Colab notebook once `X_test`, `y_test_cat`, `train_df`, `val_df`, `test_df`,
`label_encoder` and the `SpecAugment` class exist. Use the DEPLOYED model from the repo, not
`best_model_v3`: re-running the notebook re-trains the network, so that variable is a different model.

    !wget -q -O export_analytics.py https://raw.githubusercontent.com/PulkitTiwari87/Music-Genre-CNN/main/analytics/export_analytics.py
    !wget -q -O deployed_v3.keras https://raw.githubusercontent.com/PulkitTiwari87/Music-Genre-CNN/main/models/music_genre_cnn_final_v3.keras
    import importlib, export_analytics
    importlib.reload(export_analytics)   # picks up a re-downloaded file in the same runtime
    deployed = tf.keras.models.load_model("deployed_v3.keras", custom_objects={"SpecAugment": SpecAugment})
    export_analytics.export_analytics(
        model=deployed,
        X_test=X_test,
        y_test_onehot=y_test_cat,
        train_df=train_df, val_df=val_df, test_df=test_df,
        class_names=list(label_encoder.classes_),
    )
    from google.colab import files
    files.download("analytics_eval.json")

Then save the downloaded file as frontend/public/analytics/eval.json and commit it.

Nothing here is estimated: every number is computed from the model's predicted
probabilities on the real test set, or from the real audio files. Before writing anything
the function checks that it is describing the reported results:
  0. the model's weights must hash to DEPLOYED_MODEL_WEIGHTS_SHA256 (the deployed CNN V3);
  1. test_df must start with the songs the notebook printed (cell 116): same split, same order;
  2. the segment accuracy must match the recorded 0.8325550556182861 to within
     MAX_DRIFT_SEGMENTS segments (a newer librosa can flip a borderline segment; any drift
     is recorded in `split_check`);
  3. the re-extracted features must equal X_test.
If any check fails it stops.

Output schema (version 1), all arrays are plain JSON lists unless noted:
  provenance, split_check, classes
  segment / song: n, accuracy, per_class{precision,recall,f1,support}, macro, weighted,
                  confusion[[]], roc{per_class,micro,macro}, pr{per_class,micro,macro_ap},
                  confidence{edges,correct,wrong,reliability,ece,...},
                  high_confidence_errors[]            (song also has `songs[]`)
  dataset_stats   per-song duration/rms/peak/centroid/zcr + MFCC mean/std for all songs
  embeddings      mfcc_pca, mfcc_tsne, (mfcc_umap), model_pca, model_tsne, silhouette
  mfcc_correlation, mfcc_class_means
  examples        one correctly classified test song per genre (waveform envelope,
                  model-input Mel spectrogram, MFCC, probabilities)
  errors          every misclassified test song with its model-input spectrogram
  class_mean_mel  mean model-input spectrogram of each class (test segments)
  training_history  (if `history` is given)
uint8 images are base64 strings of row-major bytes (shape stored next to them).
"""

import base64
import datetime
import json
import platform
from itertools import pairwise

import numpy as np

SAMPLE_RATE = 22050
SEGMENT_SECONDS = 3
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS
N_MELS, N_FFT, HOP_LENGTH = 128, 2048, 512
EXPECTED_SEGMENT_ACCURACY = (
    0.8325550556182861  # recorded by the notebook (cell 152 / final_results.json)
)
# First test songs the notebook printed (cell 116, `song_results_df.head()`): a fingerprint of the
# split AND its file order. It only matches if test_df is the split the model was evaluated on.
# SHA-256 of the weights of the deployed model (models/music_genre_cnn_final_v3.keras). Re-running the
# notebook RE-TRAINS the network and gives different weights, so analytics must be computed on this file,
# never on whatever `best_model_v3` happens to be in memory.
DEPLOYED_MODEL_WEIGHTS_SHA256 = (
    "803f0a2a71908ce142fe889b6dee6203173acdbab5d55c1a51a73e2cbabe88fc"
)
MODEL_URL = "https://raw.githubusercontent.com/PulkitTiwari87/Music-Genre-CNN/main/models/music_genre_cnn_final_v3.keras"
EXPECTED_FIRST_TEST_FILES = (
    "disco.00082.wav",
    "pop.00050.wav",
    "rock.00061.wav",
    "metal.00073.wav",
    "jazz.00094.wav",
)
# Re-running in a newer Colab (different librosa/TensorFlow/Keras) can flip a borderline segment.
# A few segments of drift is numerical noise; more means this is not the evaluated model/features.
MAX_DRIFT_SEGMENTS = 5
SCHEMA_VERSION = 1
MAX_ERROR_EXAMPLES = (
    24  # misclassified songs that carry embedded spectrograms (bounds file size)
)


# ------------------------------------------------------------------ small helpers
def _round(values, digits=4):
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def _b64(array):
    return base64.b64encode(
        np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    ).decode("ascii")


def _uint8_image(array01):
    return np.clip(np.round(np.asarray(array01) * 255), 0, 255).astype(np.uint8)


def _load_audio(path):
    import librosa

    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    return audio


def _segments(audio):
    """Identical to the notebook's extract_segments (cell 31): (n, 128, 130) in [0, 1]."""
    import librosa

    out = []
    for start in range(0, len(audio) - SEGMENT_SAMPLES + 1, SEGMENT_SAMPLES):
        mel = librosa.feature.melspectrogram(
            y=audio[start : start + SEGMENT_SAMPLES],
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        mel_db = np.clip(librosa.power_to_db(mel, ref=np.max), -80, 0)
        out.append(((mel_db + 80) / 80).astype(np.float32))
    return np.array(out)


def _curve(x, y, limit=400):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) > limit:
        keep = np.unique(np.linspace(0, len(x) - 1, limit).round().astype(int))
        x, y = x[keep], y[keep]
    return _round(x), _round(y)


# ----------------------------------------------------------- classification block
def classification_block(y_true, prob, classes):
    """Metrics, ROC, PR and confidence analysis from true labels and class probabilities."""
    from sklearn.metrics import (
        auc,
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        precision_recall_fscore_support,
        roc_curve,
    )

    n_classes = len(classes)
    y_true = np.asarray(y_true)
    pred = prob.argmax(axis=1)
    onehot = np.eye(n_classes)[y_true]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=range(n_classes), zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0
    )[:3]
    weighted = precision_recall_fscore_support(
        y_true, pred, average="weighted", zero_division=0
    )[:3]

    roc, pr = {}, {}
    fprs, tprs, aucs, aps = [], [], [], []
    for k, name in enumerate(classes):
        positives = int(onehot[:, k].sum())
        if positives == 0 or positives == len(y_true):
            roc[name] = pr[name] = None
            continue
        fpr, tpr, _ = roc_curve(onehot[:, k], prob[:, k])
        class_auc = float(auc(fpr, tpr))
        fprs.append(fpr), tprs.append(tpr), aucs.append(class_auc)
        x, y = _curve(fpr, tpr)
        roc[name] = {
            "fpr": x,
            "tpr": y,
            "auc": round(class_auc, 5),
            "positives": positives,
        }

        prec, rec, _ = precision_recall_curve(onehot[:, k], prob[:, k])
        ap = float(average_precision_score(onehot[:, k], prob[:, k]))
        aps.append(ap)
        x, y = _curve(rec, prec)
        pr[name] = {
            "recall": x,
            "precision": y,
            "ap": round(ap, 5),
            "positives": positives,
            "baseline": round(positives / len(y_true), 5),
        }

    # micro-average: pool every (sample, class) decision
    micro_fpr, micro_tpr, _ = roc_curve(onehot.ravel(), prob.ravel())
    micro_prec, micro_rec, _ = precision_recall_curve(onehot.ravel(), prob.ravel())
    # macro-average ROC: average the per-class TPR on the union of FPR points
    grid = np.unique(np.concatenate(fprs))
    macro_tpr = np.mean([np.interp(grid, f, t) for f, t in zip(fprs, tprs)], axis=0)
    x, y = _curve(micro_fpr, micro_tpr)
    micro_roc = {"fpr": x, "tpr": y, "auc": round(float(auc(micro_fpr, micro_tpr)), 5)}
    x, y = _curve(grid, macro_tpr)
    macro_roc = {"fpr": x, "tpr": y, "auc": round(float(auc(grid, macro_tpr)), 5)}
    micro_pr_x, micro_pr_y = _curve(micro_rec, micro_prec)

    confidence = prob.max(axis=1)
    correct = pred == y_true
    edges = np.linspace(0, 1, 21)
    reliability, ece = [], 0.0
    bins = np.linspace(0, 1, 11)
    for lo, hi in pairwise(bins):
        mask = (
            (confidence > lo) & (confidence <= hi)
            if lo > 0
            else (confidence >= lo) & (confidence <= hi)
        )
        if mask.any():
            acc_bin, conf_bin = (
                float(correct[mask].mean()),
                float(confidence[mask].mean()),
            )
            ece += mask.mean() * abs(acc_bin - conf_bin)
            reliability.append(
                {
                    "lo": round(lo, 2),
                    "hi": round(hi, 2),
                    "n": int(mask.sum()),
                    "accuracy": round(acc_bin, 4),
                    "mean_confidence": round(conf_bin, 4),
                }
            )
    return {
        "n": len(y_true),
        "accuracy": round(float(correct.mean()), 6),
        "per_class": {
            name: {
                "precision": round(float(precision[k]), 4),
                "recall": round(float(recall[k]), 4),
                "f1": round(float(f1[k]), 4),
                "support": int(support[k]),
            }
            for k, name in enumerate(classes)
        },
        "macro": dict(zip(("precision", "recall", "f1"), _round(macro))),
        "weighted": dict(zip(("precision", "recall", "f1"), _round(weighted))),
        "confusion": confusion_matrix(y_true, pred, labels=range(n_classes)).tolist(),
        "roc": {
            "per_class": roc,
            "micro": micro_roc,
            "macro": macro_roc,
            "macro_auc": round(float(np.mean(aucs)), 5),
        },
        "pr": {
            "per_class": pr,
            "micro": {
                "recall": micro_pr_x,
                "precision": micro_pr_y,
                "ap": round(
                    float(average_precision_score(onehot.ravel(), prob.ravel())), 5
                ),
            },
            "macro_ap": round(float(np.mean(aps)), 5),
        },
        "confidence": {
            "edges": _round(edges, 2),
            "correct": np.histogram(confidence[correct], edges)[0].tolist(),
            "wrong": np.histogram(confidence[~correct], edges)[0].tolist(),
            "mean_correct": round(float(confidence[correct].mean()), 4)
            if correct.any()
            else None,
            "mean_wrong": round(float(confidence[~correct].mean()), 4)
            if (~correct).any()
            else None,
            "reliability": reliability,
            "ece": round(float(ece), 4),
        },
    }


# ----------------------------------------------------------------- embeddings
def _tsne(x, seed):
    from sklearn.manifold import TSNE

    perplexity = float(min(30, max(2, (len(x) - 1) // 3)))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x)


def _embedding(points, **extra):
    points = np.asarray(points)
    return {"x": _round(points[:, 0], 3), "y": _round(points[:, 1], 3), **extra}


def weights_fingerprint(model):
    """Deterministic SHA-256 over every weight tensor (float32, layer order)."""
    import hashlib

    digest = hashlib.sha256()
    for weight in model.get_weights():
        digest.update(np.ascontiguousarray(weight, dtype=np.float32).tobytes())
    return digest.hexdigest()


# ------------------------------------------------------------------------ main
def export_analytics(
    model,
    X_test,
    y_test_onehot,
    train_df,
    val_df,
    test_df,
    class_names,
    out_path="analytics_eval.json",
    history=None,
    provenance=None,
    seed=42,
    expected_segment_accuracy=EXPECTED_SEGMENT_ACCURACY,
    expected_first_files=EXPECTED_FIRST_TEST_FILES,
    expected_weights_sha256=DEPLOYED_MODEL_WEIGHTS_SHA256,
    log=print,
):
    import os

    import keras
    import librosa
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    # 0. model identity: the analytics must describe the DEPLOYED model, not a re-trained one
    if expected_weights_sha256 is not None:
        found = weights_fingerprint(model)
        if found != expected_weights_sha256:
            raise RuntimeError(
                "This model is not the deployed CNN V3: its weights differ (fingerprint "
                f"{found[:12]}… vs {expected_weights_sha256[:12]}…). Re-running the notebook re-trains the "
                "network, so best_model_v3 in memory is a different model. Load the deployed one:\n"
                f"  !wget -q -O deployed_v3.keras {MODEL_URL}\n"
                "  model = tf.keras.models.load_model('deployed_v3.keras', "
                "custom_objects={'SpecAugment': SpecAugment})"
            )
        log("model check OK: weights are those of the deployed CNN V3")

    classes = list(class_names)
    index = {name: i for i, name in enumerate(classes)}
    y_true = np.argmax(y_test_onehot, axis=1)

    # 1. segment-level predictions (the model's real probabilities)
    prob = np.asarray(model.predict(X_test, batch_size=32, verbose=0), dtype=np.float64)
    segment_accuracy = float((prob.argmax(axis=1) == y_true).mean())
    split_check = {
        "segment_accuracy": segment_accuracy,
        "expected": expected_segment_accuracy,
        "n_test_segments": len(X_test),
        "n_test_songs": len(test_df),
    }
    if expected_segment_accuracy is not None:
        # 1) identity: the split (and its order) must be the one the notebook evaluated
        first_files = [
            str(p).replace("\\", "/").split("/")[-1]
            for p in test_df["file_path"].head(len(expected_first_files or ()))
        ]
        fingerprint_ok = expected_first_files is None or first_files == list(
            expected_first_files
        )
        if not fingerprint_ok:
            raise RuntimeError(
                f"test_df is not the split the notebook evaluated: it starts with {first_files} but the "
                f"recorded run started with {list(expected_first_files)}. The file order, and so the split, "
                "differs; curves from songs the model trained on would be inflated. Refusing to export."
            )
        # 2) numerics: same model and features should reproduce the accuracy, up to a few flipped segments
        delta = round((segment_accuracy - expected_segment_accuracy) * len(X_test))
        split_check.update(
            fingerprint_ok=True,
            delta_segments=delta,
            exact=delta == 0,
            max_drift_segments=MAX_DRIFT_SEGMENTS,
        )
        if abs(delta) > MAX_DRIFT_SEGMENTS:
            raise RuntimeError(
                f"Segment accuracy {segment_accuracy:.6f} differs from the recorded "
                f"{expected_segment_accuracy:.6f} by {delta} segments (allowed: {MAX_DRIFT_SEGMENTS}). "
                "X_test is not the split this model was evaluated on (or the wrong model is loaded); "
                "refusing to export curves that would not describe the reported results."
            )
        split_check["matches"] = True
        if delta == 0:
            log(
                f"split check OK: same songs, accuracy {segment_accuracy:.6f} equals the recorded value"
            )
        else:
            log(
                f"split check OK with numerical drift: same songs, but {abs(delta)} segment(s) predicted "
                "differently from the original run (library/hardware versions); recorded in split_check."
            )

    # 2. map segments back to songs by re-extracting them, and verify the mapping
    log(f"re-extracting {len(test_df)} test songs ...")
    counts, firsts, waves, chunks = [], [], {}, []
    for row in test_df.itertuples():
        audio = _load_audio(row.file_path)
        segs = _segments(audio)
        counts.append(len(segs))
        firsts.append(segs[0])
        chunks.append(segs)
        waves[row.file_path] = audio
    stacked = np.concatenate(chunks)
    if len(stacked) != len(X_test) or not np.allclose(
        stacked, np.asarray(X_test)[..., 0], atol=1e-5
    ):
        raise RuntimeError(
            "Re-extracted test features differ from X_test: test_df does not match X_test."
        )
    offsets = np.concatenate([[0], np.cumsum(counts)])
    song_prob = np.array([prob[a:b].mean(axis=0) for a, b in pairwise(offsets)])
    song_true = np.array([index[g] for g in test_df["genre"]])
    segment_song = np.repeat(np.arange(len(counts)), counts)
    files = [p.replace("\\", "/").split("/")[-1] for p in test_df["file_path"]]

    segment = classification_block(y_true, prob, classes)
    song = classification_block(song_true, song_prob, classes)
    song_pred = song_prob.argmax(axis=1)
    song_conf = song_prob.max(axis=1)

    def top3(p):
        return [
            {"genre": classes[j], "p": round(float(p[j]), 4)}
            for j in np.argsort(p)[::-1][:3]
        ]

    song["songs"] = [
        {
            "file": files[i],
            "actual": classes[song_true[i]],
            "predicted": classes[song_pred[i]],
            "confidence": round(float(song_conf[i]), 4),
            "segments": int(counts[i]),
            "probabilities": _round(song_prob[i]),
        }
        for i in range(len(files))
    ]
    wrong_songs = [
        int(i) for i in np.argsort(-song_conf) if song_pred[i] != song_true[i]
    ]
    song["high_confidence_errors"] = [
        dict(song["songs"][i], top3=top3(song_prob[i])) for i in wrong_songs[:10]
    ]
    seg_pred, seg_conf = prob.argmax(axis=1), prob.max(axis=1)
    wrong_segments = [
        int(i) for i in np.argsort(-seg_conf) if seg_pred[i] != y_true[i]
    ][:10]
    segment["high_confidence_errors"] = [
        {
            "file": files[segment_song[i]],
            "segment_index": int(i - offsets[segment_song[i]]),
            "actual": classes[y_true[i]],
            "predicted": classes[seg_pred[i]],
            "confidence": round(float(seg_conf[i]), 4),
        }
        for i in wrong_segments
    ]

    # 3. dataset statistics + MFCC feature space over every song
    log("computing dataset statistics ...")
    full = [
        (r.file_path, r.genre, split)
        for df, split in ((train_df, "train"), (val_df, "val"), (test_df, "test"))
        for r in df.itertuples()
    ]
    rows, failed = [], []
    for path, genre, split in full:
        try:
            audio = waves.get(path)
            audio = _load_audio(path) if audio is None else audio
            if len(audio) == 0:
                raise ValueError("empty audio")
            mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)
            rows.append(
                {
                    "genre": index[genre],
                    "split": split,
                    "duration": len(audio) / SAMPLE_RATE,
                    "rms": float(np.sqrt(np.mean(audio**2))),
                    "peak": float(np.abs(audio).max()),
                    "centroid": float(
                        librosa.feature.spectral_centroid(
                            y=audio, sr=SAMPLE_RATE
                        ).mean()
                    ),
                    "zcr": float(librosa.feature.zero_crossing_rate(audio).mean()),
                    "mfcc_mean": mfcc.mean(axis=1),
                    "mfcc_std": mfcc.std(axis=1),
                }
            )
        except Exception as error:  # noqa: BLE001  a corrupt file must not abort the export
            failed.append(
                {
                    "file": path.replace("\\", "/").split("/")[-1],
                    "error": str(error)[:120],
                }
            )

    def column(key, digits=4):
        return _round([r[key] for r in rows], digits)

    mfcc_mean = np.array([r["mfcc_mean"] for r in rows])
    mfcc_std = np.array([r["mfcc_std"] for r in rows])
    genre_idx = np.array([r["genre"] for r in rows])
    dataset_stats = {
        "n": len(rows),
        "failed": failed,
        "genre": genre_idx.tolist(),
        "split": [r["split"] for r in rows],
        "duration": column("duration", 3),
        "rms": column("rms", 5),
        "peak": column("peak", 4),
        "centroid": column("centroid", 1),
        "zcr": column("zcr", 5),
    }

    log("PCA / t-SNE on MFCC features ...")
    features = StandardScaler().fit_transform(np.hstack([mfcc_mean, mfcc_std]))
    pca = PCA(n_components=2, random_state=seed).fit(features)
    embeddings = {
        "mfcc_pca": _embedding(
            pca.transform(features),
            genre=genre_idx.tolist(),
            explained_variance=_round(pca.explained_variance_ratio_, 4),
        ),
        "mfcc_tsne": _embedding(_tsne(features, seed), genre=genre_idx.tolist()),
        "mfcc_feature_description": "20 MFCC means + 20 MFCC standard deviations per song, standardised",
    }
    try:
        import umap

        embeddings["mfcc_umap"] = _embedding(
            umap.UMAP(n_components=2, random_state=seed).fit_transform(features),
            genre=genre_idx.tolist(),
        )
    except ImportError:
        embeddings["mfcc_umap"] = None

    # 4. what the trained CNN learned: its 128-d penultimate layer on the test segments
    dense = [layer for layer in model.layers if type(layer).__name__ == "Dense"]
    penultimate = keras.Model(model.inputs, dense[-2].output)
    hidden = np.asarray(
        penultimate.predict(X_test, batch_size=32, verbose=0), dtype=np.float64
    )
    model_pca = PCA(n_components=2, random_state=seed).fit(hidden)
    embeddings["model_pca"] = _embedding(
        model_pca.transform(hidden),
        genre=y_true.tolist(),
        predicted=seg_pred.tolist(),
        explained_variance=_round(model_pca.explained_variance_ratio_, 4),
    )
    embeddings["model_tsne"] = _embedding(
        _tsne(hidden, seed), genre=y_true.tolist(), predicted=seg_pred.tolist()
    )
    embeddings["silhouette"] = {
        "mfcc_features": round(float(silhouette_score(features, genre_idx)), 4),
        "cnn_embedding": round(float(silhouette_score(hidden, y_true)), 4),
        "note": "Silhouette of genre labels: -1 (overlapping) to 1 (well separated). "
        "MFCC space is all songs; CNN embedding is the 128-d layer on the test segments.",
    }
    corr = np.corrcoef(mfcc_mean, rowvar=False)
    z_means = np.array(
        [mfcc_mean[genre_idx == k].mean(axis=0) for k in range(len(classes))]
    )
    z_means = (z_means - mfcc_mean.mean(axis=0)) / mfcc_mean.std(axis=0)

    # 5. examples (one confident, correct song per genre) and every error, with real spectrograms
    log("building examples ...")

    def song_view(i):
        audio = waves[test_df["file_path"].iloc[i]]
        bins = np.array_split(audio[: (len(audio) // 600) * 600], 600)
        mel = firsts[i]
        mfcc = librosa.feature.mfcc(S=mel * 80 - 80, n_mfcc=20)
        lo, hi = float(mfcc.min()), float(mfcc.max())
        return {
            "file": files[i],
            "actual": classes[song_true[i]],
            "predicted": classes[song_pred[i]],
            "confidence": round(float(song_conf[i]), 4),
            "duration": round(len(audio) / SAMPLE_RATE, 2),
            "probabilities": _round(song_prob[i]),
            "top3": top3(song_prob[i]),
            "waveform": {
                "min": _round([b.min() for b in bins], 3),
                "max": _round([b.max() for b in bins], 3),
            },
            "mel": {
                "shape": list(mel.shape),
                "data": _b64(_uint8_image(mel)),
                "scale": "uint8 = round(255 * x), x in [0, 1]",
            },
            "mfcc": {
                "shape": list(mfcc.shape),
                "data": _b64(_uint8_image((mfcc - lo) / ((hi - lo) or 1))),
                "min": round(lo, 3),
                "max": round(hi, 3),
            },
        }

    examples = []
    for k in range(len(classes)):
        candidates = [
            i for i in range(len(files)) if song_true[i] == k and song_pred[i] == k
        ]
        if candidates:
            examples.append(song_view(max(candidates, key=lambda i: song_conf[i])))
    errors = [
        song_view(i) for i in wrong_songs[:MAX_ERROR_EXAMPLES]
    ]  # most confident mistakes first
    song["n_errors"] = len(wrong_songs)
    x_test = np.asarray(X_test)
    class_mean_mel = [
        {
            "genre": classes[k],
            "data": _b64(_uint8_image(x_test[y_true == k, :, :, 0].mean(axis=0))),
        }
        for k in range(len(classes))
        if (y_true == k).any()
    ]

    result = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "generator": "analytics/export_analytics.py",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "synthetic": False,
            "source": "Colab notebook run",
            "model_weights_sha256": weights_fingerprint(model),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "librosa": librosa.__version__,
                "keras": keras.__version__,
            },
            **(provenance or {}),
        },
        "split_check": split_check,
        "classes": classes,
        "segment": segment,
        "song": song,
        "dataset_stats": dataset_stats,
        "embeddings": embeddings,
        "mfcc_correlation": {
            "matrix": _round(corr, 3),
            "labels": [f"MFCC {i + 1}" for i in range(20)],
        },
        "mfcc_class_means": {
            "matrix": _round(z_means, 3),
            "classes": classes,
            "note": "Mean MFCC per genre, z-scored per coefficient across all songs.",
        },
        "examples": examples,
        "errors": errors,
        "class_mean_mel": class_mean_mel,
        "mel_shape": [int(X_test.shape[1]), int(X_test.shape[2])],
    }
    if history is not None:
        hist = getattr(history, "history", history)
        result["training_history"] = {k: _round(v, 6) for k, v in hist.items()}

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, separators=(",", ":"))
    size = os.path.getsize(out_path) / 1e6
    log(
        f"wrote {out_path} ({size:.2f} MB): segment acc {segment['accuracy']:.4f}, "
        f"song acc {song['accuracy']:.4f}, macro ROC-AUC {segment['roc']['macro_auc']:.4f}, "
        f"macro AP {segment['pr']['macro_ap']:.4f}, {len(wrong_songs)} misclassified songs"
    )
    return result

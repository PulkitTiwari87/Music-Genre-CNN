"""Describe the real deployed model (models/music_genre_cnn_final_v3.keras) as frontend/src/data/model.json.

Layer types, shapes, parameter counts and hyper-parameters are read from the saved model
itself; the totals are cross-checked against the model.summary() printed in the notebook.
Latency is measured here, on whatever machine runs this script, and labelled as such.

    python analytics/extract_model.py
"""

import json
import os
import platform
import re
import statistics
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from analytics.extract_notebook import ANSI, load_notebook, output_text
from src.predict import load_model, predict_song

MODEL_PATH = ROOT / "models" / "music_genre_cnn_final_v3.keras"
OUT = ROOT / "frontend" / "src" / "data" / "model.json"


def layer_details(layer):
    config = layer.get_config()
    info = {"name": layer.name, "type": type(layer).__name__}
    for key in (
        "filters",
        "kernel_size",
        "strides",
        "padding",
        "activation",
        "pool_size",
        "units",
        "rate",
        "freq_mask_param",
        "time_mask_param",
    ):
        if key in config:
            value = config[key]
            info[key] = list(value) if isinstance(value, tuple) else value
    regularizer = config.get("kernel_regularizer")
    if regularizer:
        info["l2"] = regularizer["config"]["l2"]
    if type(layer).__name__ == "BatchNormalization":
        info["momentum"] = config["momentum"]
        info["epsilon"] = config["epsilon"]
    return info


def parameter_counts(layer):
    trainable = sum(int(np.prod(w.shape)) for w in layer.trainable_weights)
    total = sum(int(np.prod(w.shape)) for w in layer.weights)
    return total, trainable, total - trainable


def notebook_summary_totals():
    text = ANSI.sub("", output_text(load_notebook(), 145))
    pick = lambda label: int(
        re.search(rf"{label}: ([\d,]+)", text).group(1).replace(",", "")
    )
    return {
        "total": pick("Total params"),
        "trainable": pick("Trainable params"),
        "non_trainable": pick("Non-trainable params"),
    }


def timing(fn, repeats):
    fn()  # warm-up (graph tracing)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": round(statistics.median(samples), 1),
        "min_ms": round(min(samples), 1),
        "runs": repeats,
    }


def main():
    model = load_model(MODEL_PATH)
    shape = tuple(model.input_shape[1:])
    layers, previous = [], shape
    for layer in model.layers:
        output = tuple(layer.output.shape[1:])
        total, trainable, non_trainable = parameter_counts(layer)
        layers.append(
            {
                **layer_details(layer),
                "input_shape": list(previous),
                "output_shape": list(output),
                "params": total,
                "trainable_params": trainable,
                "non_trainable_params": non_trainable,
            }
        )
        previous = output

    totals = {
        "total": model.count_params(),
        "trainable": sum(l["trainable_params"] for l in layers),
        "non_trainable": sum(l["non_trainable_params"] for l in layers),
    }
    expected = notebook_summary_totals()
    if totals != expected:
        sys.exit(
            f"CHECK FAILED: model parameters {totals} differ from the notebook summary {expected}"
        )

    segments = np.random.default_rng(0).random((1, *shape), dtype=np.float32)
    ten = np.repeat(segments, 10, axis=0)
    audio = (
        np.random.default_rng(1).standard_normal(22050 * 30).astype(np.float32) * 0.1
    )
    with zipfile.ZipFile(MODEL_PATH) as archive:
        saved_with = json.loads(archive.read("metadata.json"))

    data = {
        "generated_by": "analytics/extract_model.py",
        "model_file": MODEL_PATH.name,
        "file_size_bytes": MODEL_PATH.stat().st_size,
        "name": "CNN V3 (SpecAugment)",
        "type": "Sequential convolutional neural network",
        "task": "10-class music genre classification",
        "input_shape": list(shape),
        "output_shape": list(model.output_shape[1:]),
        "classes": 10,
        "parameters": {**totals, "notebook_summary_matches": True},
        "framework": {
            "saved_with_keras": saved_with["keras_version"],
            "saved_on": saved_with["date_saved"],
            "measured_with_tensorflow": tf.__version__,
            "measured_with_keras": tf.keras.__version__,
        },
        "layers": layers,
        "latency": {
            "measured_on": f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
            f"{platform.system()}, TensorFlow {tf.__version__} (CPU only). "
            "A development machine, not the production host.",
            "one_segment": timing(lambda: model.predict(segments, verbose=0), 30),
            "ten_segments": timing(
                lambda: model.predict(ten, batch_size=8, verbose=0), 15
            ),
            "full_30s_song": timing(lambda: predict_song(model, audio), 5),
            "note": "full_30s_song includes feature extraction and 10 segment predictions.",
        },
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT.relative_to(ROOT)}: {totals['total']:,} parameters, {len(layers)} layers"
    )


if __name__ == "__main__":
    main()

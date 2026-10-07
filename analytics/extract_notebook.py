"""Extract the facts recorded in Music_Genre_CNN.ipynb into frontend/src/data/notebook.json.

Everything written comes from the notebook's saved outputs and source code, or from
analytics/confusion_matrices.json (figures transcribed from the notebook and validated
here). Nothing is estimated. Any check that fails aborts the run.

    python analytics/extract_notebook.py            # (re)generate notebook.json
    python analytics/extract_notebook.py --check    # fail if the committed file is stale
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "Music_Genre_CNN.ipynb"
CONFUSION = ROOT / "analytics" / "confusion_matrices.json"
FINAL_RESULTS = ROOT / "final_results.json"
OUT = ROOT / "frontend" / "src" / "data" / "notebook.json"

ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Notebook cell numbers (0-based) that hold each experiment's training run / evaluation.
TRAIN_CELLS = {"v1": 80, "v2": 102, "v3": 150}
REPORT_CELLS = {"v3_segment": 154, "v2_song": 120}


class CheckError(Exception):
    pass


def check(condition, message):
    if not condition:
        raise CheckError(message)


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source(nb, index):
    return "".join(nb["cells"][index]["source"])


def output_text(nb, index):
    parts = []
    for out in nb["cells"][index].get("outputs", []):
        if out["output_type"] == "stream":
            parts.append("".join(out["text"]))
        elif "data" in out and "text/plain" in out["data"]:
            parts.append("".join(out["data"]["text/plain"]))
    return ANSI.sub("", "\n".join(parts))


# --------------------------------------------------------------------------- dataset
def parse_dataset(nb):
    classes_text = output_text(nb, 17)
    per_class = {
        m[0]: int(m[1])
        for m in re.findall(r"^(\w+)\s+(\d+)$", classes_text, re.MULTILINE)
    }
    check(
        len(per_class) == 10 and set(per_class.values()) == {100},
        "class counts changed",
    )

    split = re.search(
        r"Training: (\d+)\s+Validation: (\d+)\s+Testing: (\d+)", output_text(nb, 21)
    )
    songs = dict(zip(("train", "val", "test"), map(int, split.groups())))
    check(sum(songs.values()) == 1000, "split does not add up to 1000 songs")

    split_text = output_text(nb, 22)
    per_split_class = {}
    for name, block in zip(
        ("train", "val", "test"), re.split(r"\n\s*\n(?=[A-Z]+\n)", split_text)
    ):
        counts = {
            m[0]: int(m[1]) for m in re.findall(r"^(\w+)\s+(\d+)$", block, re.MULTILINE)
        }
        check(
            len(counts) == 10 and len(set(counts.values())) == 1,
            f"{name} split is not stratified",
        )
        per_split_class[name] = next(iter(counts.values()))
    check(
        per_split_class == {"train": 70, "val": 15, "test": 15},
        f"unexpected per-class split sizes {per_split_class}",
    )

    segments = {}
    for name, cell in (("train", 36), ("val", 38), ("test", 40)):
        shape = re.search(
            rf"X_{name} shape: \((\d+), 128, 130\)", output_text(nb, cell)
        )
        segments[name] = int(shape.group(1))

    audio = re.search(
        r"Sample rate: (\d+)\s+Number of samples: (\d+)\s+Duration: ([\d.]+)",
        output_text(nb, 8),
    )
    corrupt = sorted(
        set(re.findall(r"Error processing \S*?/(\w+\.\d+\.wav)", output_text(nb, 36)))
    )
    per_song = re.search(r"Number of segments: (\d+)", output_text(nb, 25))

    return {
        "name": "GTZAN Genre Collection",
        "source": "Kaggle: andradaolteanu/gtzan-dataset-music-genre-classification",
        "file_format": "wav",
        "total_songs": 1000,
        "classes": sorted(per_class),
        "songs_per_class": per_class,
        "songs": songs,
        "songs_per_class_per_split": per_split_class,
        "segments": segments,
        "segments_per_song": int(per_song.group(1)),
        "sample_rate": int(audio.group(1)),
        "example_clip": {
            "file": "rock/rock.00000.wav",
            "samples": int(audio.group(2)),
            "seconds": float(audio.group(3)),
        },
        "segment_seconds": 3,
        "segment_shape": [128, 130, 1],
        "corrupt_files": corrupt,
        "split_method": "train_test_split(test_size=0.30, stratify=genre, random_state=42), "
        "then the remaining 30% split 50/50 (stratified, random_state=42)",
        "split_unit": "song (file), so segments of one song never cross splits",
        "evidence_cells": [8, 16, 17, 19, 20, 21, 22, 36, 38, 40],
    }


# ------------------------------------------------------------------------- training
EPOCH_START = re.compile(r"^Epoch (\d+)/(\d+)\s*$", re.MULTILINE)
EPOCH_LINE = re.compile(
    r"(\d+)s (\d+)(?:ms|us)/step - accuracy: ([\d.]+) - loss: ([\d.]+) "
    r"- val_accuracy: ([\d.]+) - val_loss: ([\d.]+)(?: - learning_rate: ([\d.e+-]+))?"
)


def parse_training(nb, cell):
    text = output_text(nb, cell)
    epochs = {}
    current = None
    for line in text.splitlines():
        start = EPOCH_START.match(line)
        if start:
            current = int(start.group(1))
            continue
        match = EPOCH_LINE.search(line)
        if match and current is not None:
            seconds, _, acc, loss, vacc, vloss, lr = match.groups()
            epochs[current] = {
                "epoch": current,
                "accuracy": float(acc),
                "loss": float(loss),
                "val_accuracy": float(vacc),
                "val_loss": float(vloss),
                "learning_rate": float(lr) if lr else None,
                "seconds": int(seconds),
            }
    ordered = [epochs[k] for k in sorted(epochs)]
    check(
        ordered and [e["epoch"] for e in ordered] == list(range(1, len(ordered) + 1)),
        f"cell {cell}: epoch log has gaps",
    )

    checkpoints = [
        (int(m.group(1)), float(m.group(2)))
        for m in re.finditer(
            r"Epoch (\d+): val_accuracy improved from [\w.]+ to ([\d.]+?)(?:,|\s)", text
        )
    ]
    improved = [epoch for epoch, _ in checkpoints]
    restored = re.search(
        r"Restoring model weights from the end of the best epoch: (\d+)", text
    )
    stopped = re.search(r"Epoch (\d+): early stopping", text)
    lr_drops = [
        {"epoch": int(m.group(1)), "to": float(m.group(2))}
        for m in re.finditer(
            r"Epoch (\d+): ReduceLROnPlateau reducing learning rate to (\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            text,
        )
    ]
    best_val_acc = max(e["val_accuracy"] for e in ordered)
    return {
        "notebook_cell": cell,
        "epochs": ordered,
        "epochs_run": len(ordered),
        "max_epochs": int(EPOCH_START.search(text).group(2)),
        "best_val_accuracy": best_val_acc,
        "best_val_accuracy_epoch": max(improved) if improved else None,
        "best_checkpoint_val_accuracy": checkpoints[-1][1] if checkpoints else None,
        "checkpoint_epochs": improved,
        "best_val_loss_epoch": int(restored.group(1)) if restored else None,
        "early_stopped_at_epoch": int(stopped.group(1)) if stopped else None,
        "learning_rate_reductions": lr_drops,
        "total_seconds_reported": sum(e["seconds"] for e in ordered),
    }


# -------------------------------------------------------------------------- reports
ROW = re.compile(
    r"^\s*(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", re.MULTILINE
)
ACC = re.compile(r"^\s*accuracy\s+([\d.]+)\s+(\d+)\s*$", re.MULTILINE)
AVG = re.compile(
    r"^\s*(macro|weighted) avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$",
    re.MULTILINE,
)


def parse_report(nb, cell, classes):
    text = output_text(nb, cell)
    rows = {m[0]: m for m in ROW.findall(text) if m[0] in classes}
    check(set(rows) == set(classes), f"cell {cell}: report is missing classes")
    per_class = {
        name: {
            "precision": float(rows[name][1]),
            "recall": float(rows[name][2]),
            "f1": float(rows[name][3]),
            "support": int(rows[name][4]),
        }
        for name in classes
    }
    accuracy = ACC.search(text)
    averages = {
        m[0]: {"precision": float(m[1]), "recall": float(m[2]), "f1": float(m[3])}
        for m in AVG.findall(text)
    }
    return {
        "notebook_cell": cell,
        "per_class": per_class,
        "accuracy": float(accuracy.group(1)),
        "total_support": int(accuracy.group(2)),
        "macro": averages["macro"],
        "weighted": averages["weighted"],
    }


# ---------------------------------------------------------------------- confusion
def validate_confusion(key, entry, labels, reports, accuracies):
    matrix = entry["matrix"]
    check(len(matrix) == 10 and all(len(r) == 10 for r in matrix), f"{key}: not 10x10")
    row_sums = [sum(r) for r in matrix]
    col_sums = [sum(matrix[i][j] for i in range(10)) for j in range(10)]
    total = sum(row_sums)
    diag = [matrix[i][i] for i in range(10)]

    is_song = key.endswith("song")
    expected_support = (
        [15] * 10 if is_song else [150, 150, 150, 150, 149, 150, 150, 150, 150, 150]
    )
    check(
        row_sums == expected_support,
        f"{key}: row sums {row_sums} != supports {expected_support}",
    )
    check(
        abs(sum(diag) / total - accuracies[key]) < 5e-5,
        f"{key}: diagonal accuracy {sum(diag) / total:.6f} != notebook {accuracies[key]:.6f}",
    )

    report = reports.get(key)
    if report:
        for i, name in enumerate(labels):
            stats = report["per_class"][name]
            check(row_sums[i] == stats["support"], f"{key}/{name}: support mismatch")
            check(
                abs(diag[i] / row_sums[i] - stats["recall"]) < 6e-5,
                f"{key}/{name}: recall mismatch",
            )
            check(
                abs(diag[i] / col_sums[i] - stats["precision"]) < 6e-5,
                f"{key}/{name}: precision mismatch",
            )
    return {
        "labels": labels,
        "matrix": matrix,
        "total": total,
        "accuracy": sum(diag) / total,
        "notebook_cell": entry["notebook_cell"],
        "provenance": "Read from the notebook's heatmap figure; validated against "
        + (
            "the notebook's classification report (precision, recall, support) and accuracy."
            if report
            else "row supports and the accuracy printed by the notebook."
        ),
        "validated_against_report": bool(report),
    }


# ------------------------------------------------------------------------ models
def parse_models(nb):
    """Parameter counts from each model.summary() and file sizes printed by the notebook."""

    def params(cell):
        text = output_text(nb, cell)
        pick = lambda label: int(
            re.search(rf"{label}: ([\d,]+)", text).group(1).replace(",", "")
        )
        return {
            "total": pick("Total params"),
            "trainable": pick("Trainable params"),
            "non_trainable": pick("Non-trainable params"),
            "notebook_cell": cell,
        }

    sizes = dict(
        re.findall(r"(CNN_V\d)_best\.keras\s+\S+\s+([\d.]+) MB", output_text(nb, 131))
    )
    models = {"v1": params(63), "v2": params(96), "v3": params(145)}
    check(
        models["v1"]["total"] == 6518922
        and models["v2"]["total"] == 111370
        and models["v3"]["total"] == 423946,
        "model parameter counts changed",
    )
    for key in ("v1", "v2"):
        models[key]["file_mb"] = float(sizes[f"CNN_{key.upper()}"])
    return models


# ----------------------------------------------------------------------- results
def parse_results(nb):
    final = json.loads(FINAL_RESULTS.read_text(encoding="utf-8"))
    v1 = re.search(r"CNN V1 Test Accuracy: ([\d.]+)", output_text(nb, 105))
    v2 = re.search(r"V2 Test Accuracy: ([\d.]+)", output_text(nb, 105))
    v3_loss = re.search(r"V3 Test Loss: ([\d.]+)", output_text(nb, 152))
    check(
        v1 and v2 and v3_loss, "could not find test accuracies in the notebook outputs"
    )
    measured = {
        "v1": {"segment_accuracy": float(v1.group(1))},
        "v2": {
            "segment_accuracy": float(v2.group(1)),
            "song_accuracy": float(
                re.search(r"([\d.]+)%", output_text(nb, 117)).group(1)
            )
            / 100,
        },
        "v3": {
            "segment_accuracy": final["models"]["CNN_V3"]["segment_accuracy"],
            "song_accuracy": final["models"]["CNN_V3"]["song_accuracy"],
            "test_loss": float(v3_loss.group(1)),
        },
    }
    check(
        abs(
            measured["v2"]["segment_accuracy"]
            - final["models"]["CNN_V2"]["segment_accuracy"]
        )
        < 1e-9,
        "final_results.json disagrees with the notebook output for CNN V2",
    )
    check(
        abs(
            measured["v1"]["segment_accuracy"]
            - final["models"]["CNN_V1"]["segment_accuracy"]
        )
        < 1e-3,
        "final_results.json disagrees with the notebook output for CNN V1",
    )
    return measured


# ------------------------------------------------------------------------ config
def evidence(nb, cell, pattern):
    check(
        re.search(pattern, source(nb, cell)),
        f"config evidence missing in cell {cell}: {pattern}",
    )
    return {"cell": cell, "pattern": pattern}


def build_config(nb):
    def item(value, cell, pattern):
        return {"value": value, "evidence": evidence(nb, cell, pattern)}

    code = "\n".join(
        source(nb, i) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
    )
    check(
        not re.search(r"set_seed|random\.seed|manual_seed|enable_op_determinism", code),
        "a global seed was found; update the config",
    )

    shared = {
        "sample_rate": item(22050, 31, r"SAMPLE_RATE = 22050"),
        "segment_seconds": item(3, 31, r"SEGMENT_DURATION = 3"),
        "n_mels": item(128, 31, r"N_MELS = 128"),
        "n_fft": item(2048, 31, r"N_FFT = 2048"),
        "hop_length": item(512, 31, r"HOP_LENGTH = 512"),
        "db_scaling": item(
            "power_to_db(ref=max), clip to [-80, 0] dB, (x + 80) / 80",
            31,
            r"mel_db \+ 80\) / 80",
        ),
        "loss": item(
            "categorical_crossentropy", 148, r'loss="categorical_crossentropy"'
        ),
        "batch_size": item(32, 150, r"batch_size=32"),
        "max_epochs": item(30, 150, r"epochs=30"),
        "split": item("70 / 15 / 15 by song, stratified", 19, r"test_size=0\.30"),
        "split_seed": item(42, 19, r"random_state=42"),
        "global_seed": {
            "value": None,
            "evidence": "No TensorFlow/NumPy/Python seed is set anywhere in the notebook.",
        },
    }
    v3 = {
        "optimizer": item("Adam", 148, r"Adam\("),
        "learning_rate": item(0.001, 148, r"learning_rate=0\.001"),
        "dropout": item(0.4, 136, r"Dropout\(\s*0\.4"),
        "l2": item(1e-4, 136, r"l2\(1e-4\)"),
        "augmentation": item(
            "SpecAugment: frequency mask up to 12 bins, time mask up to 15 frames (training only)",
            136,
            r"freq_mask_param=12,\s*time_mask_param=15",
        ),
        "early_stopping": item(
            "monitor val_loss, patience 6, restore best weights",
            149,
            r"patience=6[\s\S]*restore_best_weights=True",
        ),
        "lr_schedule": item(
            "ReduceLROnPlateau: val_loss, factor 0.5, patience 2, min 1e-6",
            149,
            r"factor=0\.5,\s*patience=2,\s*min_lr=1e-6",
        ),
        "checkpoint": item(
            "Best val_accuracy checkpoint is the model that was evaluated",
            149,
            r'monitor="val_accuracy"[\s\S]*save_best_only=True',
        ),
    }
    v2 = {
        "optimizer": item(
            "Adam (Keras default learning rate 0.001)", 65, r'optimizer="adam"'
        ),
        "dropout": item(0.4, 95, r"Dropout\(0\.4\)"),
        "l2": item(1e-4, 95, r"l2\(1e-4\)"),
        "early_stopping": item(
            "monitor val_loss, patience 5, restore best weights", 100, r"patience=5"
        ),
    }
    v1 = {
        "optimizer": item(
            "Adam (Keras default learning rate 0.001)", 65, r'optimizer="adam"'
        ),
        "dropout": item(0.5, 60, r"Dropout\(0\.5\)"),
        "early_stopping": item(
            "monitor val_loss, patience 5, restore best weights", 77, r"patience=5"
        ),
    }
    return {"shared": shared, "v1": v1, "v2": v2, "v3": v3}


# --------------------------------------------------------------------------- main
def build():
    nb = load_notebook()
    dataset = parse_dataset(nb)
    classes = dataset["classes"]
    results = parse_results(nb)

    reports = {
        key: parse_report(nb, cell, classes) for key, cell in REPORT_CELLS.items()
    }
    accuracies = {
        "v1_segment": results["v1"]["segment_accuracy"],
        "v2_segment": results["v2"]["segment_accuracy"],
        "v2_song": results["v2"]["song_accuracy"],
        "v3_segment": results["v3"]["segment_accuracy"],
        "v3_song": results["v3"]["song_accuracy"],
    }
    check(
        abs(reports["v3_segment"]["accuracy"] - accuracies["v3_segment"]) < 5e-5,
        "V3 report accuracy mismatch",
    )
    check(
        abs(reports["v2_song"]["accuracy"] - accuracies["v2_song"]) < 5e-5,
        "V2 song report accuracy mismatch",
    )

    confusion_source = json.loads(CONFUSION.read_text(encoding="utf-8"))
    check(
        confusion_source["labels"] == classes,
        "confusion labels differ from dataset classes",
    )
    confusion = {
        key: validate_confusion(key, entry, classes, reports, accuracies)
        for key, entry in confusion_source["matrices"].items()
    }

    tf_info = re.search(r"TensorFlow version: ([\d.]+)", output_text(nb, 66))
    training = {key: parse_training(nb, cell) for key, cell in TRAIN_CELLS.items()}
    check(
        training["v3"]["best_val_loss_epoch"] == 22
        and training["v3"]["early_stopped_at_epoch"] == 28,
        "V3 training run changed; re-check the README claims",
    )

    return {
        "generated_by": "analytics/extract_notebook.py",
        "source_notebook": NOTEBOOK.name,
        "environment": {
            "tensorflow_in_colab": tf_info.group(1) if tf_info else None,
            "accelerator": nb["metadata"].get("accelerator"),
            "notebook_cells": len(nb["cells"]),
        },
        "dataset": dataset,
        "training": training,
        "models": parse_models(nb),
        "results": results,
        "reports": reports,
        "confusion": confusion,
        "config": build_config(nb),
    }


def main():
    try:
        data = build()
    except CheckError as error:
        sys.exit(f"CHECK FAILED: {error}")
    text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        sys.exit(
            0
            if current == text
            else "notebook.json is stale: run analytics/extract_notebook.py"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text) // 1024} KB)")


if __name__ == "__main__":
    main()

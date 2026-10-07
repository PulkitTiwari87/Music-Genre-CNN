# Analytics pipeline

The `/brag` and `/model` pages contain no hand-typed results. Everything they show comes from three
generated files; anything that cannot be computed from them reads
**"Not available from current experiment"**.

| File | Produced by | Needs | Contains |
|---|---|---|---|
| `frontend/src/data/notebook.json` | `python analytics/extract_notebook.py` | `Music_Genre_CNN.ipynb`, `analytics/confusion_matrices.json`, `final_results.json` | dataset counts, V1/V2/V3 training logs, classification reports, confusion matrices, configuration (with notebook cell evidence), parameter counts |
| `frontend/src/data/model.json` | `python analytics/extract_model.py` | `models/music_genre_cnn_final_v3.keras` | layers, shapes, parameters, file size, measured latency |
| `frontend/public/analytics/eval.json` | `analytics/export_analytics.py` **in Colab** | the trained model, the test split and the audio | ROC/PR curves, probabilities, confidence, errors, PCA/t-SNE, histograms, spectrogram/MFCC examples |

The first two are committed. The third needs the audio and the exact test split, which only the Colab
notebook has, so it is generated there and committed by hand.

## What is verified automatically

`extract_notebook.py` aborts (and `tests/test_analytics_data.py` fails) if:

- a confusion-matrix row does not sum to its class support, or its diagonal does not reproduce the
  accuracy printed by the notebook, or it disagrees with the notebook's classification report;
- a configuration value (optimizer, learning rate, dropout …) is no longer present in the notebook source;
- the committed `notebook.json` differs from what the extractor produces now.

The confusion matrices in `confusion_matrices.json` were **read from the heatmap figures stored in
the notebook** (the notebook did not save the numbers). Those checks are what make that transcription
trustworthy, and a test proves a single wrong digit is rejected.

`extract_model.py` checks its parameter totals against the `model.summary()` printed in the notebook.

## Generating `eval.json` (Colab)

Run after the CNN V3 evaluation cells, so that `best_model_v3`, `X_test`, `y_test_cat`, `train_df`,
`val_df`, `test_df` and `label_encoder` exist:

```python
!wget -q -O export_analytics.py https://raw.githubusercontent.com/PulkitTiwari87/Music-Genre-CNN/main/analytics/export_analytics.py
import importlib, export_analytics
importlib.reload(export_analytics)   # re-reads the file just downloaded, even if it was imported before

export_analytics.export_analytics(
    model=best_model_v3,
    X_test=X_test,
    y_test_onehot=y_test_cat,
    train_df=train_df, val_df=val_df, test_df=test_df,
    class_names=list(label_encoder.classes_),
    history=history_v3,   # optional
)

from google.colab import files
files.download("analytics_eval.json")
```

Save the download as `frontend/public/analytics/eval.json`, commit and push.

Safety checks inside the exporter, before anything is written:

1. **Split identity.** `test_df` must start with the five songs the notebook printed in cell 116
   (`disco.00082`, `pop.00050`, `rock.00061`, `metal.00073`, `jazz.00094`). If the notebook was re-run and the file
   order, and therefore the split, changed, it stops: curves from songs the model trained on would be inflated.
2. **Numerics.** The model's segment accuracy on `X_test` must match the recorded `0.8325550556182861` to within
   5 segments. A newer Colab (different librosa/TensorFlow/Keras) can flip a borderline segment; a drift of up to 5
   is accepted and recorded in `split_check` (and shown on the page), anything larger means the wrong model or
   features and the export stops.
3. The test features re-extracted from the audio must equal `X_test`, so every segment is tied to the
   right song.

What it computes, all from the model's predicted probabilities and the real audio:

- per-class one-vs-rest **ROC** (FPR, TPR, AUC), micro- and macro-average, macro AUC;
- per-class **precision-recall** (AP), micro-average, macro AP;
- confusion matrix, precision/recall/F1/support, macro and weighted averages;
- confidence histograms (correct vs wrong), reliability diagram, expected calibration error, the
  highest-confidence mistakes;
- song-level versions of all of the above (segment probabilities averaged per song);
- per-song duration, RMS, peak, spectral centroid, zero-crossing rate and 20 MFCC means/stds for all 1,000
  songs; PCA and t-SNE (and UMAP if `umap-learn` is installed) of that feature space, and PCA/t-SNE of the
  CNN's own 128-d layer on the test segments; silhouette scores for both;
- MFCC correlation matrix and per-genre MFCC profile;
- one correctly classified example per genre (waveform envelope, model-input Mel spectrogram, MFCC) and the
  most confident misclassified songs, each with the actual spectrogram.

MFCCs appear only in this exploratory analysis. The CNN is trained on the Mel spectrogram, not on MFCCs.
Audio is never exported (GTZAN is a licensed research dataset), so there is no playback.

If a `synthetic: true` export is ever committed, the pages show a red banner, and
`pytest` fails (`test_a_committed_eval_export_is_real_and_matches_the_recorded_run`).

Requires Python 3.10+ (`itertools.pairwise`), librosa, scikit-learn and Keras, all present in Colab.

# Models

`music_genre_cnn_final_v3.keras` is the trained CNN V3 (SpecAugment) used by the API.
It is ~5 MB, so it is tracked in Git (see the exception in `.gitignore`); every other
`*.keras` file stays ignored.

| Property | Value |
|---|---|
| Input | `(128, 130, 1)` Mel spectrogram of a 3 s segment, scaled to `[0, 1]` |
| Output | 10 softmax probabilities, alphabetical genre order (`blues` … `rock`) |
| Parameters | 423,946 |
| Saved with | Keras 3.13.2 (Colab) |
| Custom layer | `SpecAugment` (see `src/predict.py`) |

Load it with `src.predict.load_model`, which registers the custom layer.
To serve a different file, set the `MODEL_PATH` environment variable.

To regenerate it, run `Music_Genre_CNN.ipynb` in Colab; the final cells save
`music_genre_cnn_final_v3.keras` to Google Drive.

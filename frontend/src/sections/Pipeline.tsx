import { useMemo, useState } from "react";
import { notebook } from "../data";
import type { EvalData } from "../data/types";
import { decodeBytes, genreColor, genreLabel, pct } from "../lib/format";
import { HBars, Spectrogram, Waveform } from "../viz/charts";
import { EvalGate, Reveal, Source } from "../viz/ui";

interface Stage {
  title: string;
  summary: string;
  detail: string;
  shape?: string;
  cells: string;
}

const d = notebook.dataset;
const cfg = notebook.config;

/** Every step below exists in Music_Genre_CNN.ipynb / src/preprocessing.py; cell numbers are the evidence. */
export const STAGES: Stage[] = [
  {
    title: "Dataset",
    summary: `${d.total_songs} GTZAN songs, ${d.classes.length} genres`,
    detail: `Downloaded from Kaggle (${d.source}) as ${d.file_format} files, 100 songs per genre, about 30 s each.`,
    shape: `${d.total_songs} files`,
    cells: "cells 2–8, 16–17",
  },
  {
    title: "Cleaning",
    summary: "Skip unreadable files, nothing else",
    detail: `There is no filtering, de-duplication or trimming. Feature extraction wraps each file in try/except; the only failure was ${d.corrupt_files.join(", ") || "none"}, which was skipped (it was in the training split).`,
    cells: "cells 34, 36",
  },
  {
    title: "Split by song",
    summary: `${d.songs.train} / ${d.songs.val} / ${d.songs.test} songs, stratified`,
    detail: `train_test_split with test_size 0.30 then 50/50, stratified by genre, random_state ${String(cfg.shared.split_seed.value)}. Splitting happens before segmentation, so the segments of one song never appear in two splits.`,
    shape: `${d.songs_per_class_per_split.train} / ${d.songs_per_class_per_split.val} / ${d.songs_per_class_per_split.test} songs per genre`,
    cells: "cells 19–28",
  },
  {
    title: "Load and resample",
    summary: `Mono, ${d.sample_rate.toLocaleString("en-US")} Hz`,
    detail: `librosa.load(path, sr=${d.sample_rate}) decodes the file, mixes down to mono and resamples. A 30 s clip becomes about ${d.example_clip.samples.toLocaleString("en-US")} samples.`,
    shape: `${d.example_clip.samples.toLocaleString("en-US")} samples`,
    cells: "cells 8, 31",
  },
  {
    title: "Segmentation",
    summary: `${d.segment_seconds}-second windows, no overlap`,
    detail: `The waveform is cut into consecutive ${d.segment_seconds} s windows (${d.sample_rate * d.segment_seconds} samples); a partial last window is dropped. Each 30 s song gives ${d.segments_per_song} segments, which turns ${d.songs.train} training songs into ${d.segments.train.toLocaleString("en-US")} training samples.`,
    shape: `${d.segments_per_song} × ${d.sample_rate * d.segment_seconds} samples`,
    cells: "cells 25, 31",
  },
  {
    title: "Mel spectrogram",
    summary: `${String(cfg.shared.n_mels.value)} mel bands, FFT ${String(cfg.shared.n_fft.value)}, hop ${String(cfg.shared.hop_length.value)}`,
    detail: "A short-time Fourier transform turns each window into frequency content over time; triangular mel filters group the frequencies on a perceptual scale. The result is a 2-D image of energy over time and pitch, which is what a CNN is good at reading.",
    shape: "128 mel × 130 frames",
    cells: "cells 31, 33",
  },
  {
    title: "Decibels and normalisation",
    summary: "power → dB, clip to [−80, 0], scale to [0, 1]",
    detail: "power_to_db(ref=max) expresses every value relative to the segment's loudest point, the range is clipped to 80 dB, and (x + 80) / 80 maps it to 0–1 so every input has the same scale whatever the recording level.",
    shape: "float32 in [0, 1]",
    cells: "cell 31",
  },
  {
    title: "Tensors and labels",
    summary: "Add channel axis; one-hot genre labels",
    detail: "A channel axis is appended, giving (samples, 128, 130, 1). LabelEncoder assigns class numbers alphabetically (blues = 0 … rock = 9) and to_categorical one-hot encodes them.",
    shape: `(N, ${d.segment_shape.join(", ")}) and (N, 10)`,
    cells: "cells 42–48",
  },
  {
    title: "Training",
    summary: "Adam, categorical cross-entropy, ≤ 30 epochs",
    detail: `Batch size ${String(cfg.shared.batch_size.value)}, learning rate ${String(cfg.v3.learning_rate.value)}, early stopping, learning-rate reduction and a best-validation-accuracy checkpoint. SpecAugment masks random frequency and time bands in training only.`,
    cells: "cells 136–150",
  },
  {
    title: "Model",
    summary: "CNN V3: 4 conv blocks → global average pooling → dense",
    detail: "Four Conv + BatchNorm + MaxPool blocks (32, 64, 128, 256 filters), global average pooling, a 128-unit dense layer with dropout, and a 10-way softmax. See the architecture diagram on the model page.",
    shape: "423,946 parameters",
    cells: "cells 135–137",
  },
  {
    title: "Prediction",
    summary: "Softmax per segment, then average per song",
    detail: "Each segment gets 10 probabilities. For a whole song the probability vectors of all its segments are averaged and the largest entry is the predicted genre.",
    shape: "10 probabilities",
    cells: "cells 115, 156",
  },
  {
    title: "Evaluation",
    summary: "Segment level and song level on the held-out test split",
    detail: `Accuracy, classification report and confusion matrix on the ${d.songs.test} test songs (${d.segments.test.toLocaleString("en-US")} segments), plus the song-level accuracy from averaged probabilities.`,
    cells: "cells 152–160",
  },
];

export function PipelineStages({ initial = 5 }: { initial?: number | null }) {
  const [open, setOpen] = useState<number | null>(initial);
  return (
    <div className="panel-block">
      <ol className="pipeline">
        {STAGES.map((stage, i) => (
          <Reveal as="li" key={stage.title} delay={Math.min(i, 8) * 45} className={open === i ? "open" : ""}>
            <button type="button" aria-expanded={open === i} onClick={() => setOpen(open === i ? null : i)}>
              <span className="pipe-index">{String(i + 1).padStart(2, "0")}</span>
              <span className="pipe-title">{stage.title}</span>
              <span className="pipe-summary">{stage.summary}</span>
            </button>
            {open === i && (
              <div className="pipe-detail">
                <p>{stage.detail}</p>
                <p className="pipe-meta">
                  {stage.shape && (
                    <span>
                      Output: <code>{stage.shape}</code>
                    </span>
                  )}
                  <span>Evidence: notebook {stage.cells}</span>
                </p>
              </div>
            )}
          </Reveal>
        ))}
      </ol>
      <Source>Music_Genre_CNN.ipynb source and outputs; the same pipeline is reimplemented in src/preprocessing.py for the API.</Source>
    </div>
  );
}

/* ------------------------------------------------------------------ signal explorer */
function SignalPanel({ data }: { data: EvalData }) {
  const [index, setIndex] = useState(0);
  const example = data.examples[Math.min(index, data.examples.length - 1)];
  const [rows, cols] = data.mel_shape;
  const mel = useMemo(() => (example ? decodeBytes(example.mel.data) : new Uint8Array()), [example]);
  const mfcc = useMemo(() => (example ? decodeBytes(example.mfcc.data) : new Uint8Array()), [example]);
  if (!example) return <p className="hint">No correctly classified example songs in this export.</p>;
  const first3 = Math.min(100, (3 / example.duration) * 100);
  return (
    <div className="panel-block">
      <div className="chips" role="group" aria-label="Example song">
        {data.examples.map((e, i) => (
          <button key={e.file} type="button" aria-pressed={index === i} onClick={() => setIndex(i)}>
            <i style={{ background: genreColor(e.actual) }} />
            {genreLabel(e.actual)}
          </button>
        ))}
      </div>
      <p className="hint">
        <code>{example.file}</code> · actual {genreLabel(example.actual)} · {example.duration.toFixed(1)} s · predicted{" "}
        <strong>{genreLabel(example.predicted)}</strong> with {pct(example.confidence)} confidence
      </p>
      <div className="signal-flow">
        <figure>
          <div className="wave-wrap">
            <Waveform min={example.waveform.min} max={example.waveform.max} ariaLabel={`Waveform of ${example.file}`} />
            <span className="wave-window" style={{ width: `${first3}%` }} />
          </div>
          <figcaption>
            <b>1 · Waveform</b> of the whole song. The highlighted first 3 s is one model input.
          </figcaption>
        </figure>
        <span className="flow-arrow" aria-hidden="true">→</span>
        <figure>
          <Spectrogram data={mel} rows={rows} cols={cols} ariaLabel={`Mel spectrogram of the first 3 seconds of ${example.file}`} />
          <figcaption>
            <b>2 · Mel spectrogram = model input</b> ({rows} × {cols}, 0–1). Time runs left to right, low pitch at the bottom.
          </figcaption>
        </figure>
        <span className="flow-arrow" aria-hidden="true">⇢</span>
        <figure className="aside">
          <Spectrogram data={mfcc} rows={example.mfcc.shape[0]} cols={example.mfcc.shape[1]} ariaLabel={`MFCC of the first 3 seconds of ${example.file}`} />
          <figcaption>
            <b>MFCC</b> (20 × {example.mfcc.shape[1]}): shown for comparison only. The CNN does <em>not</em> use MFCCs.
          </figcaption>
        </figure>
      </div>
      <h4>Prediction for this song (mean of its segment probabilities)</h4>
      <HBars
        ariaLabel={`Predicted probabilities for ${example.file}`}
        max={1}
        format={(v) => pct(v, 1)}
        items={data.classes
          .map((c, i) => ({ label: genreLabel(c), value: example.probabilities[i], color: genreColor(c), muted: c !== example.predicted }))
          .sort((a, b) => b.value - a.value)}
      />
      <Source>analytics/eval.json: real test song ({data.split_check.n_test_songs} test songs), features computed with the notebook's own extraction code.</Source>
    </div>
  );
}

export function SignalSection() {
  return (
    <EvalGate what="Waveform, spectrogram and MFCC views need the audio of representative songs, which only the Colab notebook can read.">
      {(data) => <SignalPanel data={data} />}
    </EvalGate>
  );
}

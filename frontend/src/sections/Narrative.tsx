import { useMemo, useState } from "react";
import { CLASSES, model, notebook } from "../data";
import type { EvalData } from "../data/types";
import { rankBy, statsFromMatrix, summarizeFit, topConfusions } from "../lib/analysis";
import { decodeBytes, fixed, genreColor, genreLabel, int, pct } from "../lib/format";
import { HBars, LineChart, Spectrogram } from "../viz/charts";
import { EvalGate, Source } from "../viz/ui";

/* ------------------------------------------------------------------- decisions */
const r = notebook.results;
const m = notebook.models;
const fitV2 = summarizeFit(notebook.training.v2);
const fitV3 = summarizeFit(notebook.training.v3);

interface Decision {
  q: string;
  a: string;
  evidence: string;
}

export const DECISIONS: Decision[] = [
  {
    q: "Why a convolutional network on spectrograms?",
    a: `A Mel spectrogram is a ${notebook.dataset.segment_shape.slice(0, 2).join(" × ")} image of energy over pitch and time. Convolution filters look for local patterns (harmonic stacks, drum textures) wherever they occur and share their weights across the image, so the whole model needs only ${int(model.parameters.total)} parameters.`,
    evidence: "cells 31, 136",
  },
  {
    q: "Why mel bands, decibels and 0–1 scaling?",
    a: "The mel scale spaces frequencies the way hearing does, decibels compress the huge dynamic range of audio, and clipping at −80 dB then dividing by 80 gives every input the same 0–1 range regardless of recording level.",
    evidence: "cell 31",
  },
  {
    q: "Why 3-second segments?",
    a: `They give the network a fixed input size and multiply the data: ${notebook.dataset.songs.train} training songs become ${int(notebook.dataset.segments.train)} training samples. At prediction time the segment probabilities are averaged, which lifts accuracy from ${pct(r.v3.segment_accuracy, 2)} per segment to ${pct(r.v3.song_accuracy, 2)} per song (+${((r.v3.song_accuracy - r.v3.segment_accuracy) * 100).toFixed(1)} points).`,
    evidence: "cells 31, 156, 158",
  },
  {
    q: "Why split by song before cutting segments?",
    a: "Segments of one song sound alike. If they were shuffled across train and test, the model would be graded on near-copies of what it trained on. The split is done on files first (stratified, so every genre has 70/15/15 songs), and only then segmented.",
    evidence: "cells 19–28",
  },
  {
    q: "Why global average pooling instead of Flatten?",
    a: `V1 flattened its last feature map into a Dense layer and had ${int(m.v1.total)} parameters; V2 averaged each map instead and needed ${int(m.v2.total)} (${(m.v1.total / m.v2.total).toFixed(0)}× fewer). Segment accuracy rose from ${pct(r.v1.segment_accuracy, 2)} to ${pct(r.v2.segment_accuracy, 2)}. The same step added L2 and 'same' padding, so the gain is the combined effect of those changes.`,
    evidence: "cells 60, 95, 81, 105",
  },
  {
    q: "Why SpecAugment, L2, dropout and early stopping?",
    a: `Training accuracy runs far ahead of validation accuracy, so overfitting is the main risk: V2 ended at ${pct(fitV2.finalTrainAcc)} train vs ${pct(fitV2.finalValAcc)} validation, V3 at ${pct(fitV3.finalTrainAcc)} vs ${pct(fitV3.finalValAcc)}. Masking random frequency and time bands (SpecAugment), L2 on the kernels, dropout 0.4, early stopping and learning-rate reduction all aim at that gap; the gap is still ${(fitV3.gap * 100).toFixed(0)} points in V3, so they limit it rather than remove it.`,
    evidence: "cells 136, 149, 150",
  },
  {
    q: "Why Adam and categorical cross-entropy?",
    a: "Softmax outputs with one-hot labels are the textbook pairing with cross-entropy, and Adam at its default learning rate (0.001) is a robust starting point. The notebook records no hyper-parameter search, so these are sensible defaults, not tuned values.",
    evidence: "cells 65, 148",
  },
  {
    q: "Why evaluate at two levels?",
    a: "Validation drives early stopping and checkpoint choice; the test split is touched only for the final numbers. Segment accuracy measures the network, song accuracy measures the product: a user uploads a song, not a 3-second slice.",
    evidence: "cells 152–160",
  },
];

export function DecisionList() {
  return (
    <div className="panel-block">
      <dl className="decisions">
        {DECISIONS.map((item) => (
          <div key={item.q} className="decision">
            <dt>{item.q}</dt>
            <dd>
              {item.a}
              <span className="evidence">notebook {item.evidence}</span>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/* --------------------------------------------------------------------- lessons */
export function Lessons() {
  const stats = notebook.reports.v3_segment.per_class;
  const ranked = rankBy(stats, "f1");
  const best = ranked[0];
  const worst = ranked[ranked.length - 1];
  const mistakes = topConfusions(notebook.confusion.v3_segment.matrix, CLASSES, 1)[0];
  const songStats = statsFromMatrix(notebook.confusion.v3_song.matrix, CLASSES);
  const songWorst = rankBy(songStats, "recall").slice(-2).reverse();
  const lat = model.latency.full_30s_song.median_ms;
  const items = [
    {
      title: "Strongest finding",
      body: `Averaging segment probabilities is worth ${((r.v3.song_accuracy - r.v3.segment_accuracy) * 100).toFixed(1)} points: ${pct(r.v3.segment_accuracy, 1)} per segment becomes ${pct(r.v3.song_accuracy, 1)} per song. ${genreLabel(best)} is the cleanest class (F1 ${fixed(stats[best].f1, 3)}).`,
    },
    {
      title: "Weakest area",
      body: `${genreLabel(worst)} has the lowest segment F1 (${fixed(stats[worst].f1, 3)}); the most frequent single mistake is ${genreLabel(mistakes.from)} predicted as ${genreLabel(mistakes.to)} (${mistakes.count} of ${notebook.dataset.segments.test} segments). At song level the lowest recall is ${songWorst.map((g) => `${genreLabel(g)} ${pct(songStats[g].recall, 0)}`).join(" and ")}.`,
    },
    {
      title: "Engineering trade-off",
      body: `Song-level voting needs ${notebook.dataset.segments_per_song} model passes per song instead of one: about ${lat} ms for a 30 s song on the development CPU, still fast, in exchange for the accuracy gain. Replacing Flatten with global pooling cut the model from ${int(m.v1.total)} to ${int(m.v3.total)} parameters (${(model.file_size_bytes / 1024 / 1024).toFixed(1)} MB on disk for V3).`,
    },
    {
      title: "Limitation",
      body: `One dataset (${int(notebook.dataset.total_songs)} songs), one split and one training run per model: there is no run-to-run variance, and a 150-song test set moves 0.67 points per song. Accuracy on music outside GTZAN, or on compressed audio, was not measured.`,
    },
    {
      title: "Next improvement",
      body: "The README lists the candidates: CRNN/ResNet or transfer-learning backbones, more augmentation (time-stretch, pitch-shift, mixup) and multiple seeds or k-fold evaluation to put error bars on every number. They are plans, not results.",
    },
  ];
  return (
    <div className="lessons">
      {items.map((item) => (
        <article key={item.title}>
          <h3>{item.title}</h3>
          <p>{item.body}</p>
        </article>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- class explorer */
function ClassEval({ data, genre }: { data: EvalData; genre: string }) {
  const index = data.classes.indexOf(genre);
  const roc = data.song.roc.per_class[genre];
  const pr = data.song.pr.per_class[genre];
  const songs = (data.song.songs ?? []).filter((s) => s.actual === genre);
  const example = data.examples.find((e) => e.actual === genre);
  const [rows, cols] = data.mel_shape;
  const mel = useMemo(() => (example ? decodeBytes(example.mel.data) : null), [example]);
  return (
    <div className="two-up">
      <div>
        <h4>Song-level curves for {genreLabel(genre)}</h4>
        {roc && pr ? (
          <LineChart
            height={260}
            xDomain={[0, 1]}
            yDomain={[0, 1.02]}
            xLabel="false positive rate / recall"
            yLabel="TPR / precision"
            diagonal
            formatX={(v) => v.toFixed(1)}
            formatY={(v) => v.toFixed(1)}
            ariaLabel={`ROC and precision-recall curves for ${genre}`}
            series={[
              { name: `ROC (AUC ${fixed(roc.auc, 3)})`, color: genreColor(genre), points: roc.fpr.map((x, i) => [x, roc.tpr[i]]) },
              { name: `PR (AP ${fixed(pr.ap, 3)})`, color: "#f0abfc", points: pr.recall.map((x, i) => [x, pr.precision[i]]) },
            ]}
          />
        ) : (
          <p className="hint">No curves for this class in the export.</p>
        )}
        {mel && example && (
          <figure className="mini-spec">
            <Spectrogram data={mel} rows={rows} cols={cols} ariaLabel={`Example ${genre} spectrogram`} />
            <figcaption>
              {example.file}: predicted {genreLabel(example.predicted)} at {pct(example.confidence)}
            </figcaption>
          </figure>
        )}
      </div>
      <div>
        <h4>
          Its {songs.length} test songs (class #{index + 1})
        </h4>
        <div className="table-scroll short">
          <table className="data">
            <thead>
              <tr>
                <th>File</th>
                <th>Predicted</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {songs.map((s) => (
                <tr key={s.file} className={s.predicted === genre ? "" : "bad"}>
                  <td>{s.file}</td>
                  <td>{genreLabel(s.predicted)}</td>
                  <td>{pct(s.confidence)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export function ClassExplorer() {
  const songStats = useMemo(() => statsFromMatrix(notebook.confusion.v3_song.matrix, CLASSES), []);
  const [genre, setGenre] = useState(rankBy(songStats, "f1").slice(-1)[0]);
  const idx = CLASSES.indexOf(genre);
  const seg = notebook.reports.v3_segment.per_class[genre];
  const song = songStats[genre];
  const songRow = notebook.confusion.v3_song.matrix[idx];
  const intoClass = CLASSES.map((g, i) => ({ g, n: notebook.confusion.v3_song.matrix[i][idx] })).filter((x) => x.g !== genre && x.n > 0);
  return (
    <div className="panel-block">
      <div className="chips" role="group" aria-label="Genre to inspect">
        {CLASSES.map((g) => (
          <button key={g} type="button" aria-pressed={genre === g} onClick={() => setGenre(g)}>
            <i style={{ background: genreColor(g) }} />
            {genreLabel(g)}
          </button>
        ))}
      </div>
      <div className="two-up">
        <div>
          <h4>{genreLabel(genre)}: metrics</h4>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th />
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Segments</td>
                  <td>{fixed(seg.precision, 3)}</td>
                  <td>{fixed(seg.recall, 3)}</td>
                  <td>{fixed(seg.f1, 3)}</td>
                  <td>{seg.support}</td>
                </tr>
                <tr>
                  <td>Songs</td>
                  <td>{fixed(song.precision, 3)}</td>
                  <td>{fixed(song.recall, 3)}</td>
                  <td>{fixed(song.f1, 3)}</td>
                  <td>{song.support}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h4>Where its test songs end up</h4>
          <HBars
            ariaLabel={`Predictions for actual ${genre} songs`}
            format={(v) => `${v} of ${song.support}`}
            max={song.support}
            items={CLASSES.map((g, i) => ({ label: genreLabel(g), value: songRow[i], color: g === genre ? "#22c55e" : genreColor(g), muted: songRow[i] === 0 }))}
          />
          <p className="hint">
            {intoClass.length
              ? `Songs of other genres predicted as ${genreLabel(genre)}: ${intoClass.map((x) => `${genreLabel(x.g)} ${x.n}`).join(", ")}.`
              : `No song of another genre was predicted as ${genreLabel(genre)}.`}
          </p>
        </div>
      </div>
      <EvalGate what="Per-class curves and the list of its test songs need the model's probabilities on the test set." banner={false}>
        {(data) => <ClassEval data={data} genre={genre} />}
      </EvalGate>
      <Source>notebook report and song-level confusion matrix; curves and songs from analytics/eval.json when present.</Source>
    </div>
  );
}

/* ------------------------------------------------------------------- reproduce */
export const EXPORT_CELL = `# Run AFTER the CNN V3 evaluation cells (best_model_v3, X_test, y_test_cat,
# train_df, val_df, test_df and label_encoder must exist).
!wget -q -O export_analytics.py https://raw.githubusercontent.com/PulkitTiwari87/Music-Genre-CNN/main/analytics/export_analytics.py
import export_analytics

export_analytics.export_analytics(
    model=best_model_v3,
    X_test=X_test,
    y_test_onehot=y_test_cat,
    train_df=train_df, val_df=val_df, test_df=test_df,
    class_names=list(label_encoder.classes_),
    history=history_v3,   # optional
)

from google.colab import files
files.download("analytics_eval.json")`;

export function Reproduce() {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(EXPORT_CELL);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }
  return (
    <div className="panel-block" id="generate">
      <p>
        Three generated files feed this page. Nothing on it is typed in by hand, and anything that cannot be computed from them shows{" "}
        <em>Not available from current experiment</em>.
      </p>
      <ol className="steps">
        <li>
          <code>python analytics/extract_notebook.py</code> reads the saved outputs and source of <code>Music_Genre_CNN.ipynb</code> and writes{" "}
          <code>frontend/src/data/notebook.json</code>: dataset counts, the three training logs, classification reports, configuration and the confusion
          matrices. It aborts if a matrix row does not sum to its class support, if a diagonal does not reproduce the notebook&apos;s printed accuracy, or if a
          configuration value is no longer in the notebook source.
        </li>
        <li>
          <code>python analytics/extract_model.py</code> loads <code>models/music_genre_cnn_final_v3.keras</code> and writes{" "}
          <code>frontend/src/data/model.json</code>: layers, shapes, parameters (checked against the notebook&apos;s <code>model.summary()</code>) and measured
          latency.
        </li>
        <li>
          The ROC and PR curves, probabilities, confidence, error samples, embeddings and spectrogram examples need your trained model and audio, so they come
          from a Colab cell. Paste it at the end of the notebook, download <code>analytics_eval.json</code>, save it as{" "}
          <code>frontend/public/analytics/eval.json</code> and commit. The exporter stops if the test split is not the one the model was evaluated on (its
          segment accuracy must equal the recorded 0.8325550556182861).
        </li>
      </ol>
      <div className="code-wrap">
        <button type="button" className="ghost copy" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
        <pre aria-label="Colab export cell">
          <code>{EXPORT_CELL}</code>
        </pre>
      </div>
      <Source>analytics/ in the repository; see analytics/README.md.</Source>
    </div>
  );
}

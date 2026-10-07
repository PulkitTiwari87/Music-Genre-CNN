import { useMemo, useState } from "react";
import { CLASSES, notebook } from "../data";
import type { EvalBlock, EvalData, SongView } from "../data/types";
import { rankBy, rowNormalize, statsFromMatrix, topConfusions } from "../lib/analysis";
import { decodeBytes, fixed, genreColor, genreLabel, pct } from "../lib/format";
import { useEval } from "../data/useEval";
import { Columns, Heatmap, HBars, LineChart, Spectrogram, type Series } from "../viz/charts";
import { Counter, EvalGate, Source, Stat } from "../viz/ui";

type Level = "segment" | "song";

const LEVEL_TEXT: Record<Level, string> = {
  segment: "3-second segments (1,499 test samples)",
  song: "whole songs (150 test songs, probabilities averaged over segments)",
};

function LevelToggle({ level, onChange }: { level: Level; onChange: (level: Level) => void }) {
  return (
    <div className="toggle" role="group" aria-label="Evaluation level">
      {(["song", "segment"] as Level[]).map((value) => (
        <button key={value} type="button" aria-pressed={level === value} onClick={() => onChange(value)}>
          {value === "song" ? "Song level" : "Segment level"}
        </button>
      ))}
    </div>
  );
}

const zip = (x: number[], y: number[]): [number, number][] => x.map((v, i) => [v, y[i]]);

/* ------------------------------------------------------------- headline numbers */
export function MetricCards() {
  const r = notebook.results.v3;
  const seg = notebook.reports.v3_segment;
  const state = useEval();
  const ready = state.status === "ready" && !state.data.provenance.synthetic ? state.data : null;
  return (
    <div className="panel-block">
      <dl className="stats metrics">
        <Stat label="Song accuracy" hint="150 held-out test songs">
          <Counter value={r.song_accuracy * 100} format={(v) => `${v.toFixed(1)}%`} />
        </Stat>
        <Stat label="Segment accuracy" hint="1,499 test segments">
          <Counter value={r.segment_accuracy * 100} format={(v) => `${v.toFixed(2)}%`} />
        </Stat>
        <Stat label="Macro F1 (segments)" hint={`weighted ${fixed(seg.weighted.f1, 4)}`}>
          {fixed(seg.macro.f1, 4)}
        </Stat>
        <Stat label="Macro precision / recall" hint="segment level">
          {fixed(seg.macro.precision, 4)} / {fixed(seg.macro.recall, 4)}
        </Stat>
        <Stat label="Macro ROC-AUC" hint={ready ? "song level, one-vs-rest" : "needs eval.json"}>
          {ready ? fixed(ready.song.roc.macro_auc, 3) : "n/a"}
        </Stat>
        <Stat label="Macro PR-AUC (AP)" hint={ready ? "song level" : "needs eval.json"}>
          {ready ? fixed(ready.song.pr.macro_ap, 3) : "n/a"}
        </Stat>
        <Stat label="Test loss" hint="categorical cross-entropy">{fixed(r.test_loss, 4)}</Stat>
      </dl>
      {!ready && (
        <p className="hint">
          ROC-AUC and PR-AUC show n/a: {state.status === "ready" ? "the loaded eval.json is flagged synthetic, so its numbers are not used here" : "they need the model's probabilities, which only the Colab export can produce"}.
        </p>
      )}
      <Source>notebook cells 152–159 (accuracy, loss, classification report); ROC/PR from analytics/eval.json.</Source>
    </div>
  );
}

/* ------------------------------------------------------------ ROC/PR explainer */
export function RocExplainer() {
  return (
    <div className="explainer">
      <article>
        <h3>What is a ROC curve?</h3>
        <p>
          A classifier outputs a probability, and you choose a threshold to turn it into a yes/no answer. For one genre against all the others, the ROC curve
          plots the share of true songs caught (true-positive rate) against the share of other songs wrongly flagged (false-positive rate) as that threshold
          sweeps from strict to lenient.
        </p>
      </article>
      <article>
        <h3>What does AUC mean?</h3>
        <p>
          AUC is the area under that curve: the chance that a randomly picked song of the genre gets a higher score for it than a randomly picked song of another
          genre. 0.5 is a coin flip and 1.0 is perfect separation. A high AUC means the model ranks songs well even before any threshold is chosen; a low AUC
          means it cannot tell the genre apart.
        </p>
      </article>
      <article>
        <h3>Ten classes, one-vs-rest</h3>
        <p>
          With ten genres there are ten ROC curves, each treating one genre as &quot;positive&quot; and the other nine as &quot;negative&quot;. The{" "}
          <b>macro</b> average gives every genre equal weight; the <b>micro</b> average pools every song-genre decision, so bigger classes would count more (here all
          are equal).
        </p>
      </article>
      <article>
        <h3>Why precision-recall too?</h3>
        <p>
          Precision-recall curves show how many flagged songs are right as you demand more coverage, and average precision (AP) summarises them. They are the
          stricter view when positives are rare; with ten balanced classes the no-skill precision is about 0.10, so even 0.5 is well above chance.
        </p>
      </article>
    </div>
  );
}

/* ------------------------------------------------------------ confusion matrix */
export function ConfusionPanel({ keys = ["v3_song", "v3_segment"] as const }) {
  const [key, setKey] = useState<(typeof keys)[number]>(keys[0]);
  const [mode, setMode] = useState<"counts" | "rows">("counts");
  const [selected, setSelected] = useState<number | null>(null);
  const record = notebook.confusion[key];
  const labels = record.labels;
  const display = useMemo(() => (mode === "rows" ? rowNormalize(record.matrix) : record.matrix), [mode, record]);
  const stats = useMemo(() => statsFromMatrix(record.matrix, labels), [record, labels]);
  const byRecall = rankBy(stats, "recall");
  const confusions = topConfusions(record.matrix, labels, 4);
  const unit = key.endsWith("song") ? "songs" : "segments";
  const row = selected === null ? null : record.matrix[selected];

  return (
    <div className="panel-block">
      <div className="controls">
        <div className="toggle" role="group" aria-label="Which evaluation">
          {keys.map((k) => (
            <button key={k} type="button" aria-pressed={key === k} onClick={() => { setKey(k); setSelected(null); }}>
              {k.endsWith("song") ? "CNN V3 · songs" : "CNN V3 · segments"}
            </button>
          ))}
        </div>
        <div className="toggle" role="group" aria-label="Cell values">
          <button type="button" aria-pressed={mode === "counts"} onClick={() => setMode("counts")}>Counts</button>
          <button type="button" aria-pressed={mode === "rows"} onClick={() => setMode("rows")}>% of actual class</button>
        </div>
      </div>
      <Heatmap
        rowLabels={labels.map(genreLabel)}
        colLabels={labels.map(genreLabel)}
        matrix={display}
        max={mode === "rows" ? 1 : Math.max(...record.matrix.flat())}
        format={(v) => (mode === "rows" ? String(Math.round(v * 100)) : String(v))}
        rowTitle="actual genre"
        colTitle="predicted genre"
        ariaLabel={`Confusion matrix, ${key.endsWith("song") ? "song" : "segment"} level, CNN V3`}
        describe={(r, c) => {
          const total = record.matrix[r].reduce((a, b) => a + b, 0);
          return `Actual ${genreLabel(labels[r])}, predicted ${genreLabel(labels[c])}: ${record.matrix[r][c]} of ${total} ${unit} (${pct(record.matrix[r][c] / total)})`;
        }}
        selectedRow={selected}
        onRowSelect={setSelected}
      />
      {row && selected !== null && (
        <div className="row-detail">
          <h4>Where {genreLabel(labels[selected])} {unit} go</h4>
          <HBars
            ariaLabel={`Predictions for actual ${labels[selected]}`}
            format={(v) => `${v} ${unit}`}
            items={labels.map((l, i) => ({ label: genreLabel(l), value: row[i], color: i === selected ? "#22c55e" : genreColor(l), muted: row[i] === 0 }))}
          />
        </div>
      )}
      <ul className="findings">
        <li>
          <strong>Strongest:</strong> {byRecall.slice(0, 3).map((l) => `${genreLabel(l)} (${pct(stats[l].recall, 0)})`).join(", ")}
        </li>
        <li>
          <strong>Hardest:</strong> {byRecall.slice(-3).reverse().map((l) => `${genreLabel(l)} (${pct(stats[l].recall, 0)})`).join(", ")}
        </li>
        <li>
          <strong>Most common mistakes:</strong>{" "}
          {confusions.map((c) => `${genreLabel(c.from)} → ${genreLabel(c.to)} (${c.count})`).join(", ")}
        </li>
      </ul>
      <Source>
        {record.provenance} (notebook cell {record.notebook_cell})
      </Source>
    </div>
  );
}

/* ------------------------------------------------------------ per-class metrics */
export function PerClassPanel() {
  const [level, setLevel] = useState<Level>("segment");
  const [metric, setMetric] = useState<"precision" | "recall" | "f1">("f1");
  const stats = useMemo(
    () =>
      level === "segment"
        ? notebook.reports.v3_segment.per_class
        : statsFromMatrix(notebook.confusion.v3_song.matrix, CLASSES),
    [level],
  );
  const ordered = rankBy(stats, metric);
  const weakest = ordered[ordered.length - 1];
  return (
    <div className="panel-block">
      <div className="controls">
        <LevelToggle level={level} onChange={setLevel} />
        <div className="toggle" role="group" aria-label="Metric">
          {(["precision", "recall", "f1"] as const).map((m) => (
            <button key={m} type="button" aria-pressed={metric === m} onClick={() => setMetric(m)}>
              {m === "f1" ? "F1" : m[0].toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <HBars
        ariaLabel={`${metric} by class, ${level} level`}
        max={1}
        format={(v) => fixed(v, 3)}
        items={ordered.map((name) => ({
          label: genreLabel(name),
          value: stats[name][metric],
          color: name === weakest || stats[name][metric] < 0.8 ? "#f59e0b" : genreColor(name),
          note: `n=${stats[name].support}`,
        }))}
      />
      <p className="hint">Amber bars are below 0.80; each class has support n shown beside its value. {LEVEL_TEXT[level]}.</p>
      <Source>
        {level === "segment"
          ? `notebook classification report (cell ${notebook.reports.v3_segment.notebook_cell})`
          : "computed from the validated song-level confusion matrix (the notebook printed only accuracy at song level)"}
      </Source>
    </div>
  );
}

/* ------------------------------------------------------------------ ROC and PR */
function curveSeries(block: EvalBlock, selected: string, kind: "roc" | "pr"): Series[] {
  const out: Series[] = [];
  for (const name of Object.keys(block.roc.per_class)) {
    const roc = block.roc.per_class[name];
    const pr = block.pr.per_class[name];
    if (!roc || !pr) continue;
    const points = kind === "roc" ? zip(roc.fpr, roc.tpr) : zip(pr.recall, pr.precision);
    const isSelected = selected === name || selected === "all";
    out.push({ name: genreLabel(name), color: genreColor(name), points, faint: !isSelected, width: selected === name ? 3 : 1.8 });
  }
  if (kind === "roc") {
    out.push({ name: "micro-average", color: "#ffffff", points: zip(block.roc.micro.fpr, block.roc.micro.tpr), dashed: true, width: 2 });
    out.push({ name: "macro-average", color: "#f0abfc", points: zip(block.roc.macro.fpr, block.roc.macro.tpr), dashed: true, width: 2 });
  } else {
    out.push({ name: "micro-average", color: "#ffffff", points: zip(block.pr.micro.recall, block.pr.micro.precision), dashed: true, width: 2 });
  }
  return out;
}

function CurvePanel({ data, kind }: { data: EvalData; kind: "roc" | "pr" }) {
  const [level, setLevel] = useState<Level>("song");
  const [selected, setSelected] = useState("all");
  const block = data[level];
  const names = data.classes.filter((c) => block.roc.per_class[c] && block.pr.per_class[c]);
  const series = useMemo(() => curveSeries(block, selected, kind), [block, selected, kind]);
  const scores = names.map((name) => ({
    name,
    value: kind === "roc" ? block.roc.per_class[name]!.auc : block.pr.per_class[name]!.ap,
  }));
  const headline = kind === "roc" ? block.roc.macro_auc : block.pr.macro_ap;
  const micro = kind === "roc" ? block.roc.micro.auc : block.pr.micro.ap;
  const chosen = selected !== "all" ? scores.find((s) => s.name === selected) : null;
  const baseline = block.pr.per_class[names[0]]?.baseline;
  return (
    <div className="panel-block">
      <div className="controls">
        <LevelToggle level={level} onChange={setLevel} />
      </div>
      <div className="chips" role="group" aria-label="Class to inspect">
        <button type="button" aria-pressed={selected === "all"} onClick={() => setSelected("all")}>All classes</button>
        {names.map((name) => (
          <button key={name} type="button" aria-pressed={selected === name} onClick={() => setSelected(name)}>
            <i style={{ background: genreColor(name) }} />
            {genreLabel(name)}
          </button>
        ))}
      </div>
      <div className="curve-layout">
        <LineChart
          series={series}
          xDomain={[0, 1]}
          yDomain={[0, 1.02]}
          xLabel={kind === "roc" ? "False positive rate" : "Recall"}
          yLabel={kind === "roc" ? "True positive rate" : "Precision"}
          diagonal={kind === "roc"}
          height={380}
          formatX={(v) => v.toFixed(1)}
          formatY={(v) => v.toFixed(1)}
          ariaLabel={`${kind === "roc" ? "One-vs-rest ROC curves" : "Precision-recall curves"} at ${level} level`}
          description={`Macro ${kind === "roc" ? "ROC-AUC" : "average precision"} ${fixed(headline, 3)}, micro ${fixed(micro, 3)}.`}
        />
        <div className="curve-scores">
          <p className="big-score">
            <span>{fixed(headline, 3)}</span>
            {kind === "roc" ? "macro ROC-AUC" : "macro average precision"}
          </p>
          <p className="hint">
            micro {kind === "roc" ? "AUC" : "AP"} {fixed(micro, 3)}
            {chosen && (
              <>
                {" · "}
                <strong>{genreLabel(chosen.name)}</strong> {fixed(chosen.value, 3)}
              </>
            )}
          </p>
          <HBars
            ariaLabel={`${kind === "roc" ? "AUC" : "Average precision"} by class`}
            max={1}
            format={(v) => fixed(v, 3)}
            items={scores
              .slice()
              .sort((a, b) => b.value - a.value)
              .map((s) => ({ label: genreLabel(s.name), value: s.value, color: genreColor(s.name), muted: selected !== "all" && selected !== s.name }))}
          />
        </div>
      </div>
      {kind === "pr" && baseline !== undefined && (
        <p className="hint">
          Each class makes up {pct(baseline, 0)} of the {level} test set, so a classifier with no skill would sit at a precision of about {fixed(baseline, 2)}.
        </p>
      )}
      <Source>
        analytics/eval.json: per-class one-vs-rest curves from the model's predicted probabilities on the test split ({LEVEL_TEXT[level]}).
      </Source>
    </div>
  );
}

export function RocSection() {
  return (
    <EvalGate what="ROC curves need the model's predicted probabilities on the test set; the notebook only kept hard predictions.">
      {(data) => <CurvePanel data={data} kind="roc" />}
    </EvalGate>
  );
}

export function PrSection() {
  return (
    <EvalGate what="Precision-recall curves need the model's predicted probabilities on the test set; the notebook only kept hard predictions.">
      {(data) => <CurvePanel data={data} kind="pr" />}
    </EvalGate>
  );
}

/* ------------------------------------------------------------------- confidence */
function ConfidencePanel({ data }: { data: EvalData }) {
  const [level, setLevel] = useState<Level>("song");
  const block = data[level];
  const conf = block.confidence;
  const wrong = block.n - conf.correct.reduce((a, b) => a + b, 0);
  return (
    <div className="panel-block">
      <div className="controls">
        <LevelToggle level={level} onChange={setLevel} />
      </div>
      <div className="two-up">
        <div>
          <h4>Confidence of correct vs wrong predictions</h4>
          <Columns
            edges={conf.edges}
            xLabel="top-class probability"
            formatEdge={(v) => v.toFixed(1)}
            ariaLabel={`Histogram of top-class probability for correct and wrong ${level} predictions`}
            series={[
              { name: "correct", color: "#22c55e", values: conf.correct },
              { name: "wrong", color: "#ef4444", values: conf.wrong },
            ]}
          />
          <p className="hint">
            Mean confidence when correct {conf.mean_correct === null ? "n/a" : fixed(conf.mean_correct, 3)}; when wrong{" "}
            {conf.mean_wrong === null ? "n/a (no errors)" : fixed(conf.mean_wrong, 3)}. {wrong} of {block.n} predictions wrong.
          </p>
        </div>
        <div>
          <h4>Confidence vs actual correctness</h4>
          <LineChart
            height={300}
            xDomain={[0, 1]}
            yDomain={[0, 1]}
            xLabel="mean confidence in bin"
            yLabel="accuracy in bin"
            diagonal
            formatX={(v) => v.toFixed(1)}
            formatY={(v) => v.toFixed(1)}
            ariaLabel={`Reliability diagram for ${level} predictions`}
            description={`Expected calibration error ${fixed(conf.ece, 3)}.`}
            series={[{ name: "model", color: "#5b9cff", points: conf.reliability.map((b) => [b.mean_confidence, b.accuracy]) }]}
          />
          <p className="hint">
            On the dashed diagonal, confidence equals accuracy. Expected calibration error {fixed(conf.ece, 3)}; bins hold{" "}
            {conf.reliability.map((b) => b.n).join(", ")} predictions.
          </p>
        </div>
      </div>
      <h4>High-confidence mistakes</h4>
      {block.high_confidence_errors.length === 0 ? (
        <p className="hint">The model made no mistakes at this level.</p>
      ) : (
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>File</th>
                <th>Actual</th>
                <th>Predicted</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {block.high_confidence_errors.slice(0, 8).map((e, i) => (
                <tr key={`${e.file}${e.segment_index ?? ""}${i}`}>
                  <td>{e.file}{e.segment_index !== undefined && ` · segment ${e.segment_index}`}</td>
                  <td>{genreLabel(e.actual)}</td>
                  <td>{genreLabel(e.predicted)}</td>
                  <td>{pct(e.confidence)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Source>analytics/eval.json (model probabilities on the test split).</Source>
    </div>
  );
}

export function ConfidenceSection() {
  return (
    <EvalGate what="Confidence analysis needs the model's predicted probabilities on the test set.">
      {(data) => <ConfidencePanel data={data} />}
    </EvalGate>
  );
}

/* ----------------------------------------------------------------- error analysis */
function correlation(a: Uint8Array, b: Uint8Array): number {
  const n = a.length;
  let sa = 0;
  let sb = 0;
  for (let i = 0; i < n; i += 1) {
    sa += a[i];
    sb += b[i];
  }
  const ma = sa / n;
  const mb = sb / n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i += 1) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return num / Math.sqrt(da * db || 1);
}

function ErrorsPanel({ data }: { data: EvalData }) {
  const [index, setIndex] = useState(0);
  const [rows, cols] = data.mel_shape;
  const means = useMemo(
    () => Object.fromEntries(data.class_mean_mel.map((m) => [m.genre, decodeBytes(m.data)])),
    [data],
  );
  const errors: SongView[] = data.errors;
  if (errors.length === 0) return <p className="hint">The model made no song-level mistakes on the test set.</p>;
  const error = errors[Math.min(index, errors.length - 1)];
  const mel = decodeBytes(error.mel.data);
  const toActual = means[error.actual] ? correlation(mel, means[error.actual]) : null;
  const toPredicted = means[error.predicted] ? correlation(mel, means[error.predicted]) : null;
  return (
    <div className="panel-block">
      <p className="hint">
        {data.song.n_errors ?? errors.length} of {data.song.n} test songs were misclassified; the {errors.length} most confident are shown.
      </p>
      <div className="chips" role="group" aria-label="Misclassified song">
        {errors.map((e, i) => (
          <button key={e.file} type="button" aria-pressed={index === i} onClick={() => setIndex(i)}>
            {e.file}
          </button>
        ))}
      </div>
      <div className="error-case">
        <div>
          <h4>
            {error.file}: actual <b style={{ color: genreColor(error.actual) }}>{genreLabel(error.actual)}</b>, predicted{" "}
            <b style={{ color: genreColor(error.predicted) }}>{genreLabel(error.predicted)}</b> at {pct(error.confidence)}
          </h4>
          <p className="hint">
            Top predictions: {error.top3.map((t) => `${genreLabel(t.genre)} ${pct(t.p, 0)}`).join(" · ")}
          </p>
        </div>
        <div className="mel-trio">
          <figure>
            <Spectrogram data={mel} rows={rows} cols={cols} ariaLabel={`Model input of ${error.file}`} />
            <figcaption>This song (first 3 s)</figcaption>
          </figure>
          {means[error.actual] && (
            <figure>
              <Spectrogram data={means[error.actual]} rows={rows} cols={cols} ariaLabel={`Mean ${error.actual} spectrogram`} />
              <figcaption>Mean {genreLabel(error.actual)} (actual)</figcaption>
            </figure>
          )}
          {means[error.predicted] && (
            <figure>
              <Spectrogram data={means[error.predicted]} rows={rows} cols={cols} ariaLabel={`Mean ${error.predicted} spectrogram`} />
              <figcaption>Mean {genreLabel(error.predicted)} (predicted)</figcaption>
            </figure>
          )}
        </div>
        {toActual !== null && toPredicted !== null && (
          <p className="finding">
            Pixel correlation of this segment with the class-mean spectrogram: actual {genreLabel(error.actual)} {fixed(toActual, 2)}, predicted{" "}
            {genreLabel(error.predicted)} {fixed(toPredicted, 2)}.{" "}
            {toPredicted > toActual
              ? "On this crude measure the segment really does look more like the predicted genre than its own."
              : "On this crude measure the segment still looks more like its own genre; the model used finer cues than the mean spectrogram."}
          </p>
        )}
      </div>
      <Source>analytics/eval.json: misclassified test songs; spectrograms are the exact model input.</Source>
    </div>
  );
}

export function ErrorsSection() {
  return (
    <EvalGate what="Error analysis needs the per-song predictions and spectrograms of the test set.">
      {(data) => <ErrorsPanel data={data} />}
    </EvalGate>
  );
}

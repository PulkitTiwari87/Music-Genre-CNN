import { useMemo, useState } from "react";
import { model, notebook } from "../data";
import type { ConfigItem } from "../data/types";
import { summarizeFit } from "../lib/analysis";
import { bytesToMB, fixed, int, pct } from "../lib/format";
import { HBars, LineChart } from "../viz/charts";
import { Source } from "../viz/ui";

type RunKey = "v1" | "v2" | "v3";

const NAMES: Record<RunKey, string> = {
  v1: "CNN V1 · baseline, Flatten",
  v2: "CNN V2 · global average pooling",
  v3: "CNN V3 · SpecAugment (final)",
};

/* ------------------------------------------------------------ training curves */
export function TrainingPanel({ initial = "v3" as RunKey }) {
  const [key, setKey] = useState<RunKey>(initial);
  const run = notebook.training[key];
  const fit = useMemo(() => summarizeFit(run), [run]);
  const epochs = run.epochs;
  const accuracyValues = epochs.flatMap((e) => [e.accuracy, e.val_accuracy]);
  const lossValues = epochs.flatMap((e) => [e.loss, e.val_loss]);
  const accLo = Math.floor(Math.min(...accuracyValues) * 10) / 10;
  const lossHi = Math.ceil(Math.max(...lossValues) * 10) / 10;
  const markers = [
    ...(run.best_val_loss_epoch ? [{ x: run.best_val_loss_epoch, label: `lowest val loss (epoch ${run.best_val_loss_epoch})` }] : []),
    ...(run.best_val_accuracy_epoch ? [{ x: run.best_val_accuracy_epoch, label: `evaluated checkpoint (epoch ${run.best_val_accuracy_epoch})` }] : []),
  ];
  const xDomain: [number, number] = [1, run.epochs_run];
  const train = (pick: (e: (typeof epochs)[number]) => number): [number, number][] => epochs.map((e) => [e.epoch, pick(e)]);

  return (
    <div className="panel-block">
      <div className="controls">
        <div className="toggle" role="group" aria-label="Model run">
          {(Object.keys(NAMES) as RunKey[]).map((k) => (
            <button key={k} type="button" aria-pressed={key === k} onClick={() => setKey(k)}>
              {k.toUpperCase()}
            </button>
          ))}
        </div>
        <span className="hint">{NAMES[key]}</span>
      </div>
      <div className="two-up">
        <div>
          <h4>Accuracy</h4>
          <LineChart
            height={300}
            xDomain={xDomain}
            yDomain={[accLo, 1]}
            xLabel="epoch"
            yLabel="accuracy"
            hover
            markers={markers}
            formatY={(v) => v.toFixed(1)}
            ariaLabel={`Training and validation accuracy per epoch, ${NAMES[key]}`}
            description={`Final training accuracy ${pct(fit.finalTrainAcc)}, validation accuracy ${pct(fit.finalValAcc)} after ${fit.epochs} epochs.`}
            series={[
              { name: "training", color: "#5b9cff", points: train((e) => e.accuracy) },
              { name: "validation", color: "#f5a623", points: train((e) => e.val_accuracy) },
            ]}
          />
        </div>
        <div>
          <h4>Loss</h4>
          <LineChart
            height={300}
            xDomain={xDomain}
            yDomain={[0, lossHi]}
            xLabel="epoch"
            yLabel="loss"
            hover
            markers={markers}
            formatY={(v) => v.toFixed(1)}
            ariaLabel={`Training and validation loss per epoch, ${NAMES[key]}`}
            description={`Lowest validation loss ${fixed(fit.minValLoss.val_loss, 3)} at epoch ${fit.minValLoss.epoch}.`}
            series={[
              { name: "training", color: "#5b9cff", points: train((e) => e.loss) },
              { name: "validation", color: "#f5a623", points: train((e) => e.val_loss) },
            ]}
          />
        </div>
      </div>
      <ul className="findings">
        <li>
          <strong>Convergence:</strong> ran {run.epochs_run} of {run.max_epochs} epochs; early stopping fired at epoch {run.early_stopped_at_epoch}. The
          learning rate was halved {run.learning_rate_reductions.length} times (epochs {run.learning_rate_reductions.map((r) => r.epoch).join(", ")}), down to{" "}
          {run.learning_rate_reductions.length ? run.learning_rate_reductions[run.learning_rate_reductions.length - 1].to.toExponential(1) : "n/a"}.
        </li>
        <li>
          <strong>Fit:</strong> final training accuracy {pct(fit.finalTrainAcc)} vs validation {pct(fit.finalValAcc)} (gap {pct(fit.gap)}): {fit.verdict}.
        </li>
        <li>
          <strong>Which weights were evaluated:</strong> the checkpoint with the best validation <em>accuracy</em> (epoch {run.best_val_accuracy_epoch},{" "}
          {run.best_checkpoint_val_accuracy !== null ? fixed(run.best_checkpoint_val_accuracy, 5) : "n/a"}). Early stopping separately restored the lowest
          validation <em>loss</em> weights (epoch {run.best_val_loss_epoch}) in memory.
        </li>
        <li>
          <strong>Time:</strong> {run.total_seconds_reported} s of reported epoch time on a Colab {notebook.environment.accelerator ?? "GPU"} (TensorFlow{" "}
          {notebook.environment.tensorflow_in_colab}).
        </li>
      </ul>
      <details>
        <summary>Epoch-by-epoch table ({epochs.length} rows)</summary>
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                <th>Epoch</th>
                <th>Train acc</th>
                <th>Val acc</th>
                <th>Train loss</th>
                <th>Val loss</th>
                <th>Learning rate</th>
              </tr>
            </thead>
            <tbody>
              {epochs.map((e) => (
                <tr key={e.epoch}>
                  <td>{e.epoch}</td>
                  <td>{fixed(e.accuracy, 4)}</td>
                  <td>{fixed(e.val_accuracy, 4)}</td>
                  <td>{fixed(e.loss, 4)}</td>
                  <td>{fixed(e.val_loss, 4)}</td>
                  <td>{e.learning_rate?.toExponential(2) ?? "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <Source>notebook cell {run.notebook_cell} training log, parsed by analytics/extract_notebook.py.</Source>
    </div>
  );
}

/* ------------------------------------------------------------ configuration */
const show = (item: ConfigItem) => (typeof item.value === "number" ? String(item.value) : item.value === null ? "not set" : item.value);
const proof = (item: ConfigItem) => (typeof item.evidence === "string" ? item.evidence : `cell ${item.evidence.cell}`);

export function ConfigTable() {
  const { shared, v3 } = notebook.config;
  const rows: [string, ConfigItem][] = [
    ["Optimizer", v3.optimizer],
    ["Learning rate", v3.learning_rate],
    ["Loss", shared.loss],
    ["Batch size", shared.batch_size],
    ["Maximum epochs", shared.max_epochs],
    ["Dropout", v3.dropout],
    ["Weight regularisation (L2)", v3.l2],
    ["Augmentation", v3.augmentation],
    ["Early stopping", v3.early_stopping],
    ["Learning-rate schedule", v3.lr_schedule],
    ["Model selection", v3.checkpoint],
    ["Dataset split", shared.split],
    ["Split random seed", shared.split_seed],
    ["Global random seed", shared.global_seed],
    ["Weight decay", { value: "none (no weight-decay optimizer; L2 on kernels only)", evidence: "CNN V3 definition, cell 136" }],
  ];
  return (
    <div className="panel-block">
      <div className="table-scroll">
        <table className="data config">
          <thead>
            <tr>
              <th>Setting</th>
              <th>Value (CNN V3)</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, item]) => (
              <tr key={label}>
                <td>{label}</td>
                <td>{show(item)}</td>
                <td>{proof(item)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">
        Each value was found by regular-expression match in the notebook source when the page data was generated; if the notebook changes and a value no
        longer matches, generation fails instead of showing a stale number.
      </p>
      <Source>notebook source, via analytics/extract_notebook.py.</Source>
    </div>
  );
}

/* ------------------------------------------------------------ V1 → V3 comparison */
export function ComparisonPanel() {
  const r = notebook.results;
  const m = notebook.models;
  const t = notebook.training;
  const rows = [
    {
      key: "v1" as const,
      name: "CNN V1",
      change: "baseline: 3 conv blocks, Flatten, Dense 256, Dropout 0.5",
      segment: r.v1.segment_accuracy,
      song: null as number | null,
    },
    {
      key: "v2" as const,
      name: "CNN V2",
      change: "Global average pooling instead of Flatten, 'same' padding, Dense 128, L2, Dropout 0.4",
      segment: r.v2.segment_accuracy,
      song: r.v2.song_accuracy,
    },
    {
      key: "v3" as const,
      name: "CNN V3",
      change: "adds a 4th conv block (256 filters) and SpecAugment",
      segment: r.v3.segment_accuracy,
      song: r.v3.song_accuracy,
    },
  ];
  const songSE = Math.sqrt((r.v3.song_accuracy * (1 - r.v3.song_accuracy)) / notebook.dataset.songs.test);
  const extraSongs = Math.round((r.v3.song_accuracy - r.v2.song_accuracy) * notebook.dataset.songs.test);
  return (
    <div className="panel-block">
      <div className="table-scroll">
        <table className="data compare">
          <thead>
            <tr>
              <th>Model</th>
              <th>What changed</th>
              <th>Parameters</th>
              <th>File</th>
              <th>Epochs</th>
              <th>Segment acc.</th>
              <th>Song acc.</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} className={row.key === "v3" ? "hl" : ""}>
                <td>
                  <strong>{row.name}</strong>
                </td>
                <td>{row.change}</td>
                <td>{int(m[row.key].total)}</td>
                <td>{row.key === "v3" ? bytesToMB(model.file_size_bytes) : `${m[row.key].file_mb} MB`}</td>
                <td>{t[row.key].epochs_run}</td>
                <td>{pct(row.segment, 2)}</td>
                <td>{row.song === null ? "not evaluated" : pct(row.song, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="two-up">
        <div>
          <h4>Segment accuracy</h4>
          <HBars
            ariaLabel="Segment-level test accuracy by model"
            max={1}
            format={(v) => pct(v, 2)}
            items={rows.map((row) => ({ label: row.name, value: row.segment, color: row.key === "v3" ? "#22c55e" : "#5b9cff" }))}
          />
        </div>
        <div>
          <h4>Song accuracy</h4>
          <HBars
            ariaLabel="Song-level test accuracy by model"
            max={1}
            format={(v) => pct(v, 2)}
            items={rows
              .filter((row) => row.song !== null)
              .map((row) => ({ label: row.name, value: row.song as number, color: row.key === "v3" ? "#22c55e" : "#5b9cff" }))}
          />
        </div>
      </div>
      <ul className="findings">
        <li>
          <strong>V1 → V2:</strong> {int(m.v1.total)} → {int(m.v2.total)} parameters ({(m.v1.total / m.v2.total).toFixed(0)}× fewer) and segment accuracy{" "}
          {pct(r.v1.segment_accuracy, 2)} → {pct(r.v2.segment_accuracy, 2)} (+{((r.v2.segment_accuracy - r.v1.segment_accuracy) * 100).toFixed(2)} points).
        </li>
        <li>
          <strong>V2 → V3:</strong> segment +{((r.v3.segment_accuracy - r.v2.segment_accuracy) * 100).toFixed(2)} points; song-level +{extraSongs} songs out of{" "}
          {notebook.dataset.songs.test}. One song is {(100 / notebook.dataset.songs.test).toFixed(2)} points and the standard error of a 90% accuracy on 150
          songs is about ±{(songSE * 100).toFixed(1)} points, so the song-level gain is suggestive, not conclusive. V3 changed two things at once, so the gain
          cannot be credited to SpecAugment alone.
        </li>
      </ul>
      <Source>
        notebook outputs (cells 81, 96, 105, 117, 131, 145, 152, 158) via analytics/extract_notebook.py. Only one test split and one training run per
        model were recorded; no run-to-run variance is available.
      </Source>
    </div>
  );
}

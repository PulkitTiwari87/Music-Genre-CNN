import { useMemo, type CSSProperties } from "react";
import { model, notebook } from "../data";
import { bytesToMB, int, pct } from "../lib/format";
import { Link } from "../router";
import { useFilmData } from "./data";
import { Chapter, at } from "./engine";

const d = notebook.dataset;
const cfg = notebook.config;

function useRows() {
  const { data } = useFilmData();
  return useMemo(() => {
    const aug = model.layers.find((l) => l.type === "SpecAugment");
    const convs = model.layers.filter((l) => l.type === "Conv2D");
    const r = notebook.results.v3;
    const skipped = d.corrupt_files.length;
    return [
      {
        word: "Data",
        line: `GTZAN · ${int(d.total_songs)} songs · ${d.classes.length} genres · split by song ${d.songs_per_class_per_split.train}/${d.songs_per_class_per_split.val}/${d.songs_per_class_per_split.test}, stratified · ${skipped} unreadable file${skipped === 1 ? "" : "s"} skipped`,
      },
      {
        word: "Features",
        line: `${int(d.sample_rate)} Hz mono · ${d.segment_seconds} s windows · ${String(cfg.shared.n_mels.value)} Mel bands × ${d.segment_shape[1]} frames · decibels scaled to [0, 1]`,
      },
      {
        word: "Model",
        line: `${convs.length} conv blocks, ${convs.map((c) => c.filters).join(" → ")} filters · global average pooling · ${int(model.parameters.total)} parameters · ${bytesToMB(model.file_size_bytes)}`,
      },
      {
        word: "Training",
        line: `Adam ${String(cfg.v3.learning_rate.value)} · batch ${String(cfg.shared.batch_size.value)} · SpecAugment (≤ ${aug?.freq_mask_param} bands, ≤ ${aug?.time_mask_param} frames) · early stopping · LR schedule · best-checkpoint · ${notebook.training.v3.epochs_run} epochs`,
      },
      {
        word: "Evaluation",
        line: `${pct(r.song_accuracy, 1)} song accuracy · F1 · ROC-AUC · PR-AUC · confusion matrices at song and slice level${data ? " · calibration" : ""}`,
      },
      {
        word: "Analysis",
        line: data
          ? `${data.errors.length} misclassified songs inspected · confidence of every mistake · class embeddings · mean-spectrogram test · a reproducible export`
          : "error inspection, class embeddings and a reproducible export",
      },
    ];
  }, [data]);
}

const CHAIN = [
  { name: "Data", note: "split by song, so no leakage into the test set" },
  { name: "Feature engineering", note: "one Mel-spectrogram pipeline shared by the notebook and the API" },
  { name: "Model design", note: "three versions compared; V3 kept" },
  { name: "Training", note: "regularised, scheduled and checkpointed" },
  { name: "Evaluation", note: "song and slice level, ROC, PR, calibration" },
  { name: "Error analysis", note: "every miss inspected, with a computed evidence list" },
  { name: "Interactive exploration", note: "a web app: upload a song, get a genre" },
] as const;

function EngineeringStage() {
  const rows = useRows();
  return (
    <>
      <div className="en-a f-pad">
        <div className="rv" style={at(0, 0.04, [0.5, 0.54])}>
          <p className="f-kicker">
            <b>15</b>The engineering
          </p>
        </div>
        <ol className="en-rows">
          {rows.map((r, i) => (
            <li key={r.word} className="rv" style={at(0.03 + i * 0.065, 0.08 + i * 0.065, [0.5, 0.54], { "--dy-out": "-20px" })}>
              <h3 className="en-word">{r.word}</h3>
              <p className="f-copy">{r.line}</p>
            </li>
          ))}
        </ol>
      </div>

      <div className="en-b f-pad">
        <div className="rv" style={at(0.56, 0.62, [0.98, 1])}>
          <p className="f-kicker">
            <b>16</b>Built, not just trained
          </p>
          <h2 className="f-big en-title">
            Not a number.
            <br />
            <span className="f-accent">A pipeline.</span>
          </h2>
        </div>

        <ol className="en-chain" style={{ "--fill": "clamp(0, (var(--p) - 0.62) / 0.28, 1)" } as CSSProperties}>
          {CHAIN.map((c, i) => {
            const start = 0.62 + (i / CHAIN.length) * 0.28;
            return (
              <li key={c.name} className="rv" style={at(start, start + 0.05, [0.98, 1], { "--dy": "14px" })}>
                <i aria-hidden="true" />
                <h3>{c.name}</h3>
                <p>{c.note}</p>
                {i === CHAIN.length - 1 && (
                  <p className="en-links">
                    <Link to="/">Try it live</Link>
                    <Link to="/model">Model details</Link>
                  </p>
                )}
              </li>
            );
          })}
        </ol>
      </div>
    </>
  );
}

export default function EngineeringScene() {
  return (
    <Chapter id="engineering" label="The engineering" vh={540} mobileVh={500}>
      <EngineeringStage />
    </Chapter>
  );
}

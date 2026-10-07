import { useMemo, useState, type CSSProperties } from "react";
import { model, notebook } from "../data";
import type { EvalData } from "../data/types";
import { summarizeFit } from "../lib/analysis";
import { decodeBytes, fixed, genreColor, genreLabel, pct } from "../lib/format";
import { Spectrogram } from "../viz/charts";
import { NeedsEval, useFilmData, type Sample } from "./data";
import { Chapter, at, useStep } from "./engine";

const ROWS = model.input_shape[0];
const COLS = model.input_shape[1];

/* ---------------------------------------------------------------- evidence */
const pearson = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i += 1) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let s = 0;
  let x = 0;
  let y = 0;
  for (let i = 0; i < n; i += 1) {
    s += (a[i] - ma) * (b[i] - mb);
    x += (a[i] - ma) ** 2;
    y += (b[i] - mb) ** 2;
  }
  return s / Math.sqrt(x * y);
};

const ranks = (v: number[]) => {
  const order = v.map((x, i) => [x, i] as const).sort((p, q) => p[0] - q[0]);
  const out = new Array<number>(v.length);
  order.forEach(([, i], r) => {
    out[i] = r;
  });
  return out;
};

/** How similar are the genres' average spectrograms, and does similarity predict which genres get confused? */
function meanSpectrogramEvidence(data: EvalData) {
  const means = data.class_mean_mel.map((c) => ({ genre: c.genre, v: decodeBytes(c.data) }));
  const labels = notebook.confusion.v3_segment.labels;
  const matrix = notebook.confusion.v3_segment.matrix;
  const pairs: { a: string; b: string; corr: number; confused: number }[] = [];
  for (let i = 0; i < means.length; i += 1) {
    for (let j = i + 1; j < means.length; j += 1) {
      const ia = labels.indexOf(means[i].genre);
      const ib = labels.indexOf(means[j].genre);
      pairs.push({ a: means[i].genre, b: means[j].genre, corr: pearson(means[i].v, means[j].v), confused: matrix[ia][ib] + matrix[ib][ia] });
    }
  }
  const corrs = pairs.map((p) => p.corr);
  const spearman = pearson(ranks(corrs), ranks(pairs.map((p) => p.confused)));
  const mostConfused = [...pairs].sort((p, q) => q.confused - p.confused)[0];
  const bySimilarity = [...pairs].sort((p, q) => q.corr - p.corr);
  return {
    min: Math.min(...corrs),
    max: Math.max(...corrs),
    mean: corrs.reduce((s, c) => s + c, 0) / corrs.length,
    spearman,
    mostConfused,
    similarityRank: bySimilarity.indexOf(mostConfused) + 1,
    pairCount: pairs.length,
  };
}

/* ------------------------------------------------------------ part A: errors */
function MeanMel({ data, genre }: { data: EvalData; genre: string }) {
  const item = data.class_mean_mel.find((c) => c.genre === genre);
  const bytes = useMemo(() => (item ? decodeBytes(item.data) : null), [item]);
  if (!bytes) return null;
  return (
    <figure className="fl-mean">
      <Spectrogram data={bytes} rows={ROWS} cols={COLS} ariaLabel={`Average ${genre} spectrogram`} className="fl-mel" />
      <figcaption className="f-tag">average {genreLabel(genre).toLowerCase()}</figcaption>
    </figure>
  );
}

function ErrorCard({ sample, data }: { sample: Sample; data: EvalData }) {
  const v = sample.view;
  return (
    <div className="fl-card" key={v.file}>
      <p className="fl-verdict">
        <span style={{ color: genreColor(v.actual) }}>{genreLabel(v.actual)}</span> <i>→</i> <span style={{ color: genreColor(v.predicted) }}>{genreLabel(v.predicted)}</span>
      </p>
      <p className="f-tag">
        {v.file} · <b className="fl-sure">{pct(v.confidence, 1)} sure</b>
      </p>
      <div className="fl-mels">
        <figure className="fl-main">
          <Spectrogram data={sample.mel} rows={ROWS} cols={COLS} ariaLabel={`Spectrogram of the misclassified song ${v.file}`} className="fl-mel" />
          <figcaption className="f-tag">this song</figcaption>
        </figure>
        <MeanMel data={data} genre={v.actual} />
        <MeanMel data={data} genre={v.predicted} />
      </div>
      <ul className="fl-top3">
        {v.top3.map((t) => (
          <li key={t.genre}>
            <span>{genreLabel(t.genre)}</span>
            <i style={{ "--w": t.p, "--c": genreColor(t.genre) } as CSSProperties} />
            <em>{pct(t.p, 1)}</em>
          </li>
        ))}
      </ul>
    </div>
  );
}

const AUTO = [0.1, 0.18, 0.26] as const;
const GONE = [0.4] as const;

function StruggleStage({ data, errors }: { data: EvalData; errors: Sample[] }) {
  const auto = Math.max(0, useStep(AUTO));
  const gone = useStep(GONE) >= 0;
  const [picked, setPicked] = useState<number | null>(null);
  const index = Math.min(picked ?? auto, errors.length - 1);
  const conf = data.song.confidence;
  const sure = errors.filter((e) => e.view.confidence >= 0.5).length;
  const verySure = errors.filter((e) => e.view.confidence >= 0.9).length;

  return (
    <div className={`fl-a f-pad${gone ? " gone" : ""}`}>
      <div className="fl-head rv" style={at(0, 0.05, [0.34, 0.4])}>
        <p className="f-kicker">
          <b>13</b>Where it struggles
        </p>
        <h2 className="f-big fl-title">
          Wrong,
          <br />
          <span className="f-warm">and sure.</span>
        </h2>
        <p className="f-copy">
          Every one of the {errors.length} misclassified test songs, placed by how confident the model was. On average it is <b>{pct(conf.mean_correct ?? 0, 1)}</b> sure when right
          and <b>{pct(conf.mean_wrong ?? 0, 1)}</b> when wrong: less sure, not unsure. {sure} of {errors.length} mistakes came with at least 50% confidence
          {verySure > 0 ? `, ${verySure} with over 90%` : ""}.
        </p>
      </div>

      <div className="fl-stagecard rv" style={at(0.06, 0.12, [0.34, 0.4])}>
        <ErrorCard sample={errors[index]} data={data} />
      </div>

      <div className="fl-strip rv" style={at(0.06, 0.12, [0.34, 0.4])}>
        <div className="fl-track" role="group" aria-label="Misclassified songs by confidence">
          {errors.map((e, i) => (
            <button
              key={e.view.file}
              type="button"
              className="fl-dot"
              aria-pressed={i === index}
              aria-label={`${e.view.file}: ${genreLabel(e.view.actual)} predicted as ${genreLabel(e.view.predicted)}, ${pct(e.view.confidence, 1)} sure`}
              style={{ left: `${e.view.confidence * 100}%`, bottom: `${1.9 + (i % 4) * 1.5}rem`, "--c": genreColor(e.view.actual) } as CSSProperties}
              onClick={() => setPicked(i)}
            />
          ))}
          <span className="fl-axis" aria-hidden="true">
            <b style={{ left: "0%" }}>0%</b>
            <b style={{ left: "50%" }}>50%</b>
            <b style={{ left: "100%" }}>100% sure</b>
          </span>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- part B: why */
function WhyStage({ data }: { data: EvalData }) {
  const ev = useMemo(() => meanSpectrogramEvidence(data), [data]);
  const fit = useMemo(() => summarizeFit(notebook.training.v3), []);
  const perSong = 100 / notebook.dataset.songs.test;
  const rows = [
    {
      head: "Averages can't separate genres.",
      body: (
        <>
          The average spectrograms of all genres correlate {fixed(ev.min, 2)}–{fixed(ev.max, 2)} (mean {fixed(ev.mean, 2)}). And the pair the model mixes up most,{" "}
          <b>
            {genreLabel(ev.mostConfused.a)} ↔ {genreLabel(ev.mostConfused.b)}
          </b>
          , is only the {ev.similarityRank}th most similar of {ev.pairCount}: similarity of averages does not predict confusion (rank correlation {fixed(ev.spearman, 2)}).
          The difference lives in finer detail.
        </>
      ),
    },
    {
      head: "The network's own map still overlaps.",
      body: (
        <>
          Silhouette <b>{fixed(data.embeddings.silhouette.cnn_embedding, 3)}</b> in its 128-D layer, where 1 would be perfectly separated and 0 is no structure. Structured,
          but the genre boundaries are soft.
        </>
      ),
    },
    {
      head: "700 songs is not much.",
      body: (
        <>
          {notebook.dataset.songs_per_class_per_split.train} training songs per genre. Training accuracy ends at <b>{pct(fit.finalTrainAcc, 1)}</b> against{" "}
          <b>{pct(fit.finalValAcc, 1)}</b> on validation: the network partly memorises.
        </>
      ),
    },
    {
      head: "One split, one run.",
      body: (
        <>
          {notebook.dataset.songs.test} test songs: each song moves accuracy by <b>{perSong.toFixed(2)} points</b>. One dataset (GTZAN), one split, one training run, so there are
          no error bars on any number in this film.
        </>
      ),
    },
  ];
  return (
    <div className="fl-b f-pad">
      <div className="rv" style={at(0.42, 0.48, [0.97, 1])}>
        <p className="f-kicker">
          <b>14</b>Why
        </p>
        <h2 className="f-mega fl-why">
          Why<span className="f-warm">?</span>
        </h2>
        <p className="f-note">Observations from this experiment, not proofs of cause.</p>
      </div>
      <ol className="fl-rows">
        {rows.map((r, i) => (
          <li key={r.head} className="rv" style={at(0.48 + i * 0.1, 0.54 + i * 0.1, [0.97, 1])}>
            <span className="f-num">{String(i + 1).padStart(2, "0")}</span>
            <div>
              <h3>{r.head}</h3>
              <p className="f-copy">{r.body}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

function FailuresStage() {
  const { status, data, errors } = useFilmData();
  if (!data || errors.length === 0) {
    return (
      <div className="f-pad">
        <NeedsEval status={status} what="The failure analysis uses every misclassified test song from the Colab export." />
      </div>
    );
  }
  return (
    <>
      <StruggleStage data={data} errors={errors} />
      <WhyStage data={data} />
    </>
  );
}

export default function FailuresScene() {
  return (
    <Chapter id="failures" label="Where it struggles, and why" vh={560} mobileVh={520}>
      <FailuresStage />
    </Chapter>
  );
}

import { useCallback } from "react";
import { model } from "../data";
import { Spectrogram } from "../viz/charts";
import { NeedsEval, useFilmData, type Sample } from "./data";
import { Chapter, ProgressCanvas, at, lerp, useStep } from "./engine";

const ROWS = model.input_shape[0];
const COLS = model.input_shape[1];

const PROPERTIES = [
  { word: "Frequency", note: "which pitches are present, and how strongly", probe: "one instant: energy in each of 128 Mel bands" },
  { word: "Rhythm", note: "when the energy lands", probe: "total energy across the 3 seconds: the pulse" },
  { word: "Timbre", note: "the colour of a sound", probe: "average spectrum of a classical and a metal song" },
  { word: "Structure", note: "how it all unfolds in time", probe: "the whole 30 s clip, as a waveform envelope" },
  { word: "Harmony", note: "pitches stacked in repeating patterns", probe: "stacked horizontal ridges in the spectrogram" },
] as const;

const BREAKS = [0.16, 0.28, 0.4, 0.52, 0.64] as const;

const mix = (v: number): [number, number, number] => [lerp(143, 255, v), lerp(120, 196, v), lerp(255, 130, v)];
const rgb = (c: number[], a = 1) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;

const column = (mel: Uint8Array, c: number) => Array.from({ length: ROWS }, (_, r) => mel[r * COLS + c] / 255);
const rowMeans = (mel: Uint8Array) =>
  Array.from({ length: ROWS }, (_, r) => {
    let sum = 0;
    for (let c = 0; c < COLS; c += 1) sum += mel[r * COLS + c];
    return sum / COLS / 255;
  });
const frameEnergy = (mel: Uint8Array) =>
  Array.from({ length: COLS }, (_, c) => {
    let sum = 0;
    for (let r = 0; r < ROWS; r += 1) sum += mel[r * COLS + c];
    return sum / ROWS / 255;
  });

function ProbeCanvas({ kind, sample, classical, metal }: { kind: number; sample: Sample; classical?: Sample; metal?: Sample }) {
  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number) => {
      const pad = 18;
      const iw = w - pad * 2;
      const ih = h - pad * 2;
      ctx.lineJoin = "round";
      if (kind === 0) {
        const values = column(sample.mel, 40);
        const bw = iw / ROWS;
        values.forEach((v, r) => {
          const bh = Math.max(1, Math.pow(v, 1.4) * ih);
          ctx.fillStyle = rgb(mix(v), 0.35 + 0.65 * v);
          ctx.fillRect(pad + r * bw + bw * 0.12, pad + ih - bh, Math.max(1, bw * 0.76), bh);
        });
      } else if (kind === 1) {
        const e = frameEnergy(sample.mel);
        const lo = Math.min(...e);
        const hi = Math.max(...e);
        const y = (v: number) => pad + ih - ((v - lo) / (hi - lo || 1)) * ih * 0.92;
        ctx.beginPath();
        e.forEach((v, c) => ctx.lineTo(pad + (c / (COLS - 1)) * iw, y(v)));
        ctx.strokeStyle = rgb([143, 120, 255]);
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.lineTo(pad + iw, pad + ih);
        ctx.lineTo(pad, pad + ih);
        const fill = ctx.createLinearGradient(0, pad, 0, pad + ih);
        fill.addColorStop(0, "rgba(143,120,255,0.35)");
        fill.addColorStop(1, "rgba(143,120,255,0)");
        ctx.fillStyle = fill;
        ctx.fill();
      } else if (kind === 2 && classical && metal) {
        const series: [Sample, string, string][] = [
          [classical, "classical", rgb([143, 120, 255])],
          [metal, "metal", rgb([255, 180, 90])],
        ];
        const all = series.map(([s]) => rowMeans(s.mel));
        const lo = Math.min(...all.flat());
        const hi = Math.max(...all.flat());
        series.forEach(([, name, colour], i) => {
          ctx.beginPath();
          all[i].forEach((v, r) => ctx.lineTo(pad + (r / (ROWS - 1)) * iw, pad + ih - ((v - lo) / (hi - lo || 1)) * ih * 0.9));
          ctx.strokeStyle = colour;
          ctx.lineWidth = 2.4;
          ctx.stroke();
          ctx.fillStyle = colour;
          ctx.font = "600 11px ui-monospace, Menlo, Consolas, monospace";
          ctx.fillText(name.toUpperCase(), pad + 6, pad + 14 + i * 18);
        });
      } else if (kind === 3) {
        const { min, max } = sample.view.waveform;
        const peak = Math.max(...max, ...min.map(Math.abs), 1e-6);
        const mid = pad + ih / 2;
        ctx.strokeStyle = rgb([143, 120, 255], 0.9);
        ctx.lineWidth = Math.max(1, iw / min.length - 0.6);
        ctx.beginPath();
        min.forEach((lo, i) => {
          const x = pad + ((i + 0.5) / min.length) * iw;
          ctx.moveTo(x, mid - (max[i] / peak) * (ih / 2));
          ctx.lineTo(x, mid - (lo / peak) * (ih / 2));
        });
        ctx.stroke();
      }
    },
    [kind, sample, classical, metal],
  );
  return <ProgressCanvas draw={draw} className="pb-canvas" label={`Real data: ${PROPERTIES[kind].probe}`} />;
}

function ProblemStage() {
  const { status, sample, examples } = useFilmData();
  const step = useStep(BREAKS);
  const active = Math.min(Math.max(step, 0), PROPERTIES.length - 1);
  const classical = examples.find((e) => e.view.actual === "classical");
  const metal = examples.find((e) => e.view.actual === "metal");

  return (
    <div className="pb f-pad">
      <div className="pb-left">
        <div className="rv" style={at(0.0, 0.08)}>
          <p className="f-kicker">
            <b>02</b>The problem
          </p>
          <h2 className="f-big pb-title">
            Music is not
            <br />a label.
          </h2>
        </div>

        <ul className="pb-words rv" style={at(0.1, 0.18, [0.78, 0.84], { "--dy-out": "-24px" })}>
          {PROPERTIES.map((p, i) => (
            <li key={p.word} data-on={step >= 0 && i === active}>
              {p.word}
              <small>{p.note}</small>
            </li>
          ))}
        </ul>

        <div className="pb-question rv" style={at(0.84, 0.92, [0.97, 1])}>
          <p className="f-mid">
            How do you turn what people <span className="f-accent">hear</span> into something a neural network can{" "}
            <span className="f-accent">read</span>?
          </p>
        </div>
      </div>

      <figure className="pb-right rv" style={at(0.12, 0.2, [0.78, 0.84])}>
        <div className="pb-probe" key={active}>
          {!sample ? (
            <NeedsEval status={status} what="The real spectrogram probes need the Colab export." />
          ) : active === 4 ? (
            <Spectrogram data={sample.mel} rows={ROWS} cols={COLS} ariaLabel="Real Mel spectrogram: harmonics show as stacked horizontal ridges" className="pb-spec" />
          ) : (
            <ProbeCanvas kind={active} sample={sample} classical={classical} metal={metal} />
          )}
        </div>
        <figcaption className="f-tag pb-cap">
          {sample ? `${active === 2 ? "classical and metal test songs" : sample.view.file} · ${PROPERTIES[active].probe}` : "real test song"}
        </figcaption>
      </figure>
    </div>
  );
}

export default function Problem() {
  return (
    <Chapter id="problem" label="The problem" vh={340} mobileVh={300}>
      <ProblemStage />
    </Chapter>
  );
}

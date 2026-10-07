import { useCallback, useMemo } from "react";
import { model, notebook } from "../data";
import { int } from "../lib/format";
import { Spectrogram } from "../viz/charts";
import { NeedsEval, useFilmData } from "./data";
import { Chapter, ProgressCanvas, at, easeInOut, lerp, seg, useStep } from "./engine";

const ROWS = model.input_shape[0];
const COLS = model.input_shape[1];
const cfg = notebook.config.shared;
const SAMPLE_RATE = Number(cfg.sample_rate.value);
const SEGMENT_SECONDS = notebook.dataset.segment_seconds;
const SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS;
const CLIP_SECONDS = 30;

const STEPS = [
  { key: "Audio", title: "Audio", tag: `${int(SAMPLE_RATE)} Hz · mono` },
  { key: "Waveform", title: "Waveform", tag: `${SEGMENT_SECONDS} s window = ${int(SEGMENT_SAMPLES)} samples` },
  { key: "Spectrogram", title: "Spectrogram", tag: `STFT ${String(cfg.n_fft.value)} · hop ${String(cfg.hop_length.value)} · ${String(cfg.n_mels.value)} Mel bands` },
  { key: "MFCC", title: "MFCC", tag: "20 coefficients × 130 frames" },
  { key: "Model input", title: "Model input", tag: `(${ROWS}, ${COLS}, 1) · float in [0, 1]` },
] as const;
const BREAKS = [0, 0.26, 0.4, 0.6, 0.76] as const;

const warm = (v: number) => {
  const t = Math.min(1, Math.max(0, v));
  return `rgb(${Math.round(lerp(36, 255, t))},${Math.round(lerp(22, 196, Math.pow(t, 1.3)))},${Math.round(lerp(88, 130, t))})`;
};

/** Lens: the small patch of the spectrogram that is shown as raw numbers in the last step. */
const LENS_SIZE = { rows: 6, cols: 9 };

/** The patch with the most contrast, so the numbers on screen visibly differ from each other. */
function findLens(mel: Uint8Array) {
  let best = { r0: 0, c0: 0, score: -1 };
  for (let r0 = 0; r0 <= ROWS - LENS_SIZE.rows; r0 += 2) {
    for (let c0 = 0; c0 <= COLS - LENS_SIZE.cols; c0 += 2) {
      let sum = 0;
      let sumSq = 0;
      for (let r = 0; r < LENS_SIZE.rows; r += 1) {
        for (let c = 0; c < LENS_SIZE.cols; c += 1) {
          const v = mel[(r0 + r) * COLS + c0 + c];
          sum += v;
          sumSq += v * v;
        }
      }
      const n = LENS_SIZE.rows * LENS_SIZE.cols;
      const score = sumSq / n - (sum / n) ** 2;
      if (score > best.score) best = { r0, c0, score };
    }
  }
  return { r0: best.r0, c0: best.c0, ...LENS_SIZE };
}

function WaveCanvas({ min, max }: { min: number[]; max: number[] }) {
  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number, p: number) => {
      const n = min.length;
      const segmentBins = Math.round(n * (SEGMENT_SECONDS / CLIP_SECONDS));
      const peak = Math.max(...max, ...min.map(Math.abs), 1e-6);
      const reveal = seg(p, 0.02, 0.2);
      const zoom = easeInOut(seg(p, 0.26, 0.38));
      const visible = lerp(n, segmentBins, zoom); // bins that span the full width
      const lit = seg(p, 0.2, 0.26);
      const mid = h * 0.46;
      const amp = h * 0.4;
      const bar = Math.max(1.2, (w / visible) * 0.62);
      for (let i = 0; i < Math.min(Math.ceil(visible), n); i += 1) {
        if (zoom < 0.01 && i / n > reveal) break;
        const x = ((i + 0.5) / visible) * w;
        const inside = i < segmentBins;
        const strength = inside ? lerp(0.55, 1, lit) : lerp(0.55, 0.16, lit);
        ctx.fillStyle = inside && lit > 0 ? `rgba(255,196,130,${strength})` : `rgba(143,120,255,${strength})`;
        const top = mid - (max[i] / peak) * amp;
        const bottom = mid - (min[i] / peak) * amp;
        ctx.fillRect(x - bar / 2, top, bar, Math.max(1.5, bottom - top));
      }
      if (reveal < 1 && zoom < 0.01) {
        const x = reveal * w;
        const glow = ctx.createLinearGradient(x - 24, 0, x, 0);
        glow.addColorStop(0, "rgba(255,255,255,0)");
        glow.addColorStop(1, "rgba(255,255,255,0.35)");
        ctx.fillStyle = glow;
        ctx.fillRect(x - 24, 0, 24, h * 0.92);
      }
      ctx.font = "500 11px ui-monospace, Menlo, Consolas, monospace";
      ctx.fillStyle = "rgba(160,155,187,0.9)";
      ctx.textAlign = "left";
      ctx.fillText("0 s", 4, h - 6);
      ctx.textAlign = "right";
      ctx.fillText(zoom > 0.5 ? `${SEGMENT_SECONDS} s` : `${CLIP_SECONDS} s`, w - 4, h - 6);
      if (lit > 0 && zoom < 0.5) {
        const edge = (segmentBins / visible) * w;
        ctx.strokeStyle = `rgba(255,196,130,${0.9 * lit})`;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(1, 4, edge, h * 0.86);
        ctx.setLineDash([]);
        ctx.fillStyle = `rgba(255,196,130,${lit})`;
        ctx.textAlign = "left";
        ctx.fillText(`first ${SEGMENT_SECONDS} s → one training example`, 8, 20);
      }
    },
    [min, max],
  );
  return <ProgressCanvas draw={draw} label="Waveform envelope of a real test song" />;
}

function SeeingStage() {
  const { status, sample } = useFilmData();
  const step = Math.max(0, useStep(BREAKS));

  const lens = useMemo(() => (sample ? findLens(sample.mel) : { r0: 0, c0: 0, ...LENS_SIZE }), [sample]);
  const grid = useMemo(() => {
    if (!sample) return [];
    return Array.from({ length: lens.rows }, (_, i) => {
      const r = lens.r0 + lens.rows - 1 - i; // top of the grid is the highest band, like the picture
      return Array.from({ length: lens.cols }, (_, j) => sample.mel[r * COLS + lens.c0 + j] / 255);
    });
  }, [sample, lens]);

  const samples = sample ? Math.round(sample.view.duration * SAMPLE_RATE) : 0;
  const copy = [
    <>
      A {sample ? `${Math.round(sample.view.duration)}-second` : "30-second"} clip is ≈ <b>{int(samples)}</b> numbers: one air-pressure reading every 1/
      {int(SAMPLE_RATE)} of a second. Raw, that is far too long and far too unstructured to learn from.
    </>,
    <>
      Cut into {SEGMENT_SECONDS}-second windows. Each window becomes one training example, so a single song gives the network{" "}
      <b>{notebook.dataset.segments_per_song}</b> looks at the same music.
    </>,
    <>
      A short-time Fourier transform asks, instant by instant, <b>how much energy sits at each frequency</b>. Grouped on the Mel scale, hearing&apos;s own spacing,
      the sound becomes a picture: time across, pitch up.
    </>,
    <>
      MFCCs squeeze each column into 20 numbers that summarise timbre. They power this project&apos;s feature-space analysis later; the CNN itself reads the Mel
      picture, not the MFCCs.
    </>,
    <>
      Scaled from decibels into 0–1, the picture is a{" "}
      <b>
        {ROWS} × {COLS} grid of numbers
      </b>
      . The lit patch shows its real values. That grid, and nothing else, is what enters the network.
    </>,
  ];

  return (
    <div className="se f-pad">
      <div className="rv" style={at(0, 0.04, [0.97, 1])}>
        <p className="f-kicker">
          <b>04</b>Seeing the music
        </p>
        <ol className="se-steps" aria-label="Stages">
          {STEPS.map((s, i) => (
            <li key={s.key} data-on={i === step} data-done={i < step}>
              {s.key}
            </li>
          ))}
        </ol>
      </div>

      <div className="se-screen rv" style={at(0, 0.04, [0.97, 1])}>
        {!sample ? (
          <NeedsEval status={status} what="The audio-to-spectrogram transformation uses a real test song from the Colab export." />
        ) : (
          <>
            <div className="se-a">
              <div className="se-wave rv" style={at(-0.02, -0.01, [0.4, 0.46], { "--dy-out": "0px" })}>
                <WaveCanvas min={sample.view.waveform.min} max={sample.view.waveform.max} />
              </div>
              <div className="se-spec rv" style={at(0.4, 0.44, undefined, { "--dy": "0px" })}>
                <div className="se-clip">
                  <Spectrogram
                    data={sample.mel}
                    rows={ROWS}
                    cols={COLS}
                    ariaLabel={`Mel spectrogram of ${sample.view.file}, the exact model input`}
                    className="se-img"
                  />
                </div>
                <i className="se-scan" aria-hidden="true" />
                <i
                  className="se-lens rv"
                  aria-hidden="true"
                  style={{
                    ...at(0.76, 0.8, undefined, { "--dy": "0px" }),
                    left: `${(lens.c0 / COLS) * 100}%`,
                    width: `${(lens.cols / COLS) * 100}%`,
                    top: `${(1 - (lens.r0 + lens.rows) / ROWS) * 100}%`,
                    height: `${(lens.rows / ROWS) * 100}%`,
                  }}
                />
              </div>
            </div>

            <figure className="se-b rv" style={at(0.6, 0.66, [0.74, 0.8], { "--dy": "16px" })}>
              <Spectrogram
                data={sample.mfcc}
                rows={sample.view.mfcc.shape[0]}
                cols={sample.view.mfcc.shape[1]}
                ariaLabel="MFCC of the same segment"
                className="se-img"
              />
              <figcaption className="f-tag">MFCC · 20 × {COLS} · used for analysis, not by the CNN</figcaption>
            </figure>

            <div
              className="se-grid rv"
              style={at(0.78, 0.84, undefined, { "--dy": "0px" })}
              role="img"
              aria-label="Raw model input values for a 6 by 9 patch of the spectrogram"
            >
              <p className="f-tag">
                {lens.rows} × {lens.cols} of {ROWS} × {COLS} · real values
              </p>
              <div className="se-cells" style={{ gridTemplateColumns: `repeat(${lens.cols}, 1fr)` }}>
                {grid.flat().map((v, i) => (
                  <span key={i} style={{ background: warm(v), color: v > 0.55 ? "#150f2a" : "#f4f2fc" }}>
                    {v.toFixed(2)}
                  </span>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      <div className="se-copy">
        {STEPS.map((s, i) => (
          <div key={s.key} className="se-step" data-on={i === step}>
            <h2 className="f-big">
              {s.title}
              <span className="f-accent">.</span>
            </h2>
            <p className="f-copy">{copy[i]}</p>
            <p className="f-tag se-tagline">{s.tag}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Seeing() {
  return (
    <Chapter id="seeing" label="Audio to model input" vh={480} mobileVh={420}>
      <SeeingStage />
    </Chapter>
  );
}

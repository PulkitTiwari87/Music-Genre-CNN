import { useEffect, useMemo, useRef, useState } from "react";
import { model, notebook } from "../data";
import { int, pct } from "../lib/format";
import { usePrefersReducedMotion } from "../viz/hooks";
import { useFilmData } from "./data";
import { Chapter, at, lerp, useStep } from "./engine";

const FRAMES_PER_SECOND = Number(notebook.config.shared.sample_rate.value) / Number(notebook.config.shared.hop_length.value);
const MEL_ROWS = model.input_shape[0];
const MEL_COLS = model.input_shape[1];

/** Frequency "lines" driven by a real 3-second Mel segment, played back in real time. */
function SpectrumField({ mel, level }: { mel: Uint8Array | null; level: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const levelRef = useRef(level);
  const melRef = useRef(mel);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    levelRef.current = level;
    melRef.current = mel;
  }, [level, mel]);

  useEffect(() => {
    const node = canvas.current;
    const ctx = node?.getContext("2d");
    if (!node || !ctx || typeof ResizeObserver === "undefined") return;
    let w = 0;
    let h = 0;
    let dpr = 1;
    let frame = 0;
    let shown = reduced ? levelRef.current : 0.04;
    let real = 0;
    const measure = () => {
      const rect = node.getBoundingClientRect();
      dpr = Math.min(2, window.devicePixelRatio || 1);
      w = rect.width;
      h = rect.height;
      node.width = Math.max(1, Math.round(w * dpr));
      node.height = Math.max(1, Math.round(h * dpr));
    };
    const draw = (now: number) => {
      const t = reduced ? 1.2 : now / 1000;
      shown += (levelRef.current - shown) * (reduced ? 1 : 0.035);
      const data = melRef.current;
      real += ((data ? 1 : 0) - real) * (reduced ? 1 : 0.03);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const bars = w < 640 ? 52 : 116;
      const centre = h * 0.86;
      const reach = h * 0.44;
      const width = Math.max(1.5, (w / bars) * 0.34);
      const column = (t * FRAMES_PER_SECOND) % MEL_COLS;
      const c0 = Math.floor(column);
      const c1 = (c0 + 1) % MEL_COLS;
      const frac = column - c0;
      for (let i = 0; i < bars; i += 1) {
        const f = i / (bars - 1);
        const synthetic = 0.55 * Math.exp(-f * 2.2) * (0.62 + 0.38 * Math.sin(t * 2.4 + i * 0.37)) + 0.05;
        let v = synthetic;
        if (data) {
          const row = Math.round(f * (MEL_ROWS - 1));
          const sample = lerp(data[row * MEL_COLS + c0], data[row * MEL_COLS + c1], frac) / 255;
          v = lerp(synthetic, Math.pow(sample, 2.1), real);
        }
        const height = 1.5 + v * reach * shown;
        const x = ((i + 0.5) / bars) * w;
        const warm = Math.min(1, v * 1.5);
        const r = Math.round(lerp(143, 255, warm));
        const g = Math.round(lerp(120, 196, warm));
        const b = Math.round(lerp(255, 130, warm));
        ctx.fillStyle = `rgba(${r},${g},${b},${0.35 + 0.65 * Math.min(1, v * 1.6)})`;
        ctx.fillRect(x - width / 2, centre - height, width, height);
        ctx.fillStyle = `rgba(${r},${g},${b},0.16)`;
        ctx.fillRect(x - width / 2, centre + 3, width, height * 0.55);
      }
      // Fade the tips so tall bars can sit behind the headline without hurting legibility.
      const fade = ctx.createLinearGradient(0, centre - reach, 0, centre - reach * 0.25);
      fade.addColorStop(0, "rgba(0,0,0,0.92)");
      fade.addColorStop(1, "rgba(0,0,0,0)");
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = fade;
      ctx.fillRect(0, centre - reach - 4, w, reach * 0.75 + 4);
      ctx.globalCompositeOperation = "source-over";
      if (!reduced) frame = requestAnimationFrame(draw);
    };
    measure();
    const observer = new ResizeObserver(() => {
      measure();
      if (reduced) draw(0);
    });
    observer.observe(node);
    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [reduced]);

  return <canvas ref={canvas} className="f-canvas op-spectrum" aria-hidden="true" />;
}

const SCROLL_STEPS = [0.04, 0.14] as const;

export default function Opening() {
  return (
    <Chapter id="open" label="Opening" vh={230} mobileVh={200} className="op">
      <OpeningStage />
    </Chapter>
  );
}

function OpeningStage() {
  const { sample } = useFilmData();
  const reduced = usePrefersReducedMotion();
  const [timed, setTimed] = useState(0);
  const scrolled = useStep(SCROLL_STEPS) + 1; // 0 before any scroll, 1 and 2 as the visitor scrolls on

  useEffect(() => {
    if (reduced) return;
    const timers = [window.setTimeout(() => setTimed(1), 2300), window.setTimeout(() => setTimed(2), 4800)];
    return () => timers.forEach(window.clearTimeout);
  }, [reduced]);

  const frame = Math.max(reduced ? 0 : timed, scrolled);
  const r = notebook.results.v3;
  const ticker = useMemo(
    () =>
      [
        `${int(notebook.dataset.total_songs)} songs`,
        `${notebook.dataset.classes.length} genres`,
        `${int(notebook.dataset.sample_rate)} Hz`,
        `${notebook.dataset.segment_seconds} s windows`,
        `${notebook.config.shared.n_mels.value} mel bands`,
        `${notebook.dataset.segment_shape.slice(0, 2).join(" × ")} input`,
        `${int(model.parameters.total)} parameters`,
        `${pct(r.song_accuracy, 1)} song accuracy`,
      ].join("  ·  "),
    [r.song_accuracy],
  );

  return (
    <>
      <div className="rv f-fill" style={at(-0.02, -0.01, [0.8, 1], { "--dy-out": "0px" })}>
        <SpectrumField mel={sample?.mel ?? null} level={frame === 0 ? 0.04 : frame === 1 ? 0.5 : 1} />
      </div>

      <div className="rv f-fill" style={at(-0.02, -0.01, [0.55, 0.92], { "--dy-out": "70px" })}>
        <p className="op-label f-tag" data-hide={frame >= 2}>
          Music genre classification · a case study
        </p>

        <div className="op-frame" data-state={frame === 0 ? "on" : "past"}>
          <p className="f-mega">
            Music is
            <br />
            <span className="f-accent">structure.</span>
          </p>
        </div>

        <div className="op-frame" data-state={frame < 1 ? "next" : frame === 1 ? "on" : "past"}>
          <p className="f-mega">
            Can a machine
            <br />
            learn <span className="f-accent">it?</span>
          </p>
        </div>

        <div className="op-frame op-title" data-state={frame < 2 ? "next" : "on"}>
          <h1 className="op-h1">
            Music genre
            <br />
            classification
          </h1>
          <ul className="op-chips" aria-label="Technology">
            <li>GTZAN</li>
            <li>Mel spectrogram</li>
            <li>CNN</li>
          </ul>
          <p className="op-tease">
            <b>{pct(r.song_accuracy, 1)}</b> of unseen songs placed in the right genre
            <span> · </span>
            {int(notebook.dataset.total_songs)} songs, {notebook.dataset.classes.length} genres
          </p>
        </div>

        <div className="op-scroll f-tag" data-show={frame >= 2} aria-hidden="true">
          <i />
          scroll
        </div>
      </div>

      <div className="op-ticker" aria-hidden="true">
        <div>
          <span>{ticker}  ·  </span>
          <span>{ticker}  ·  </span>
        </div>
      </div>
    </>
  );
}

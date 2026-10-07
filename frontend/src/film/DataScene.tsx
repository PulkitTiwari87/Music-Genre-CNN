import { useCallback, useMemo } from "react";
import { notebook } from "../data";
import { genreColor, genreLabel, int } from "../lib/format";
import { Chapter, ProgressCanvas, ScrollNumber, at, easeInOut, lerp, rng, seg } from "./engine";

const d = notebook.dataset;
const PER_ROW = 10;

interface Dot {
  genre: number;
  /** 0 train, 1 validation, 2 test */
  split: 0 | 1 | 2;
  /** lattice cell inside its genre cluster */
  col: number;
  row: number;
  sx: number;
  sy: number;
  delay: number;
}

const hexToRgb = (hex: string): [number, number, number] => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

/** One dot per song. Within a cluster the lattice is ordered train, validation, test so the split reads as bands. */
function buildDots(): Dot[] {
  const random = rng(7);
  const per = d.songs_per_class_per_split;
  const dots: Dot[] = [];
  d.classes.forEach((genre, g) => {
    const total = d.songs_per_class[genre];
    for (let k = 0; k < total; k += 1) {
      const split: 0 | 1 | 2 = k < per.train ? 0 : k < per.train + per.val ? 1 : 2;
      dots.push({ genre: g, split, col: k % PER_ROW, row: Math.floor(k / PER_ROW), sx: random(), sy: random(), delay: random() });
    }
  });
  return dots;
}

function layout(w: number, h: number) {
  const wide = w >= 860;
  const x0 = wide ? w * 0.38 : w * 0.05;
  const x1 = w * 0.95;
  const y0 = wide ? h * 0.17 : h * 0.37;
  const y1 = h * 0.9;
  const cols = wide ? 5 : 2;
  const rows = wide ? 2 : 5;
  const gap = 2.4;
  const rowCells = Math.ceil(d.songs_per_class[d.classes[0]] / PER_ROW);
  const s = Math.min((x1 - x0) / (cols * PER_ROW + (cols - 1) * gap), (y1 - y0) / (rows * (rowCells + 2.2) + (rows - 1) * gap));
  const blockW = cols * PER_ROW * s + (cols - 1) * gap * s;
  return { x0: x0 + (x1 - x0 - blockW) / 2, y0, cols, s, gap, rowCells };
}

function DataStage() {
  const dots = useMemo(() => buildDots(), []);
  const colours = useMemo(() => d.classes.map((g) => hexToRgb(genreColor(g))), []);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number, p: number) => {
      const L = layout(w, h);
      const gather = seg(p, 0.12, 0.4);
      const tint = seg(p, 0.26, 0.42);
      const split = easeInOut(seg(p, 0.5, 0.62));
      const labels = seg(p, 0.38, 0.5);
      const fadeOut = 1 - seg(p, 0.95, 1);
      const r = Math.max(1.4, L.s * 0.34);
      dots.forEach((dot) => {
        const cx = dot.genre % L.cols;
        const cy = Math.floor(dot.genre / L.cols);
        const tx = L.x0 + (cx * (PER_ROW + L.gap) + dot.col + 0.5) * L.s;
        const ty = L.y0 + (cy * (L.rowCells + 2.2 + L.gap) + dot.row + 0.5) * L.s;
        const t = easeInOut(Math.min(1, Math.max(0, gather * 1.45 - dot.delay * 0.45)));
        const x = lerp(dot.sx * w, tx, t);
        const y = lerp(dot.sy * h, ty, t);
        const [cr, cg, cb] = colours[dot.genre];
        const red = lerp(150, cr, tint);
        const green = lerp(146, cg, tint);
        const blue = lerp(180, cb, tint);
        const base = lerp(0.35, 0.8, gather);
        const alpha = lerp(base, dot.split === 0 ? 0.3 : dot.split === 1 ? 0.62 : 1, split) * fadeOut;
        if (dot.split === 2 && split > 0.05) {
          ctx.fillStyle = `rgba(${red | 0},${green | 0},${blue | 0},${0.16 * split * fadeOut})`;
          ctx.beginPath();
          ctx.arc(x, y, r * 2.1, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = `rgba(${red | 0},${green | 0},${blue | 0},${alpha})`;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      });
      if (labels > 0) {
        ctx.font = `600 ${Math.max(9, Math.min(12, L.s * 1.15))}px ui-monospace, Menlo, Consolas, monospace`;
        ctx.textAlign = "left";
        d.classes.forEach((genre, g) => {
          const cx = g % L.cols;
          const cy = Math.floor(g / L.cols);
          const x = L.x0 + cx * (PER_ROW + L.gap) * L.s;
          const y = L.y0 + (cy * (L.rowCells + 2.2 + L.gap) + L.rowCells + 1.5) * L.s;
          ctx.fillStyle = `rgba(244,242,252,${0.85 * labels * fadeOut})`;
          ctx.fillText(genreLabel(genre).toUpperCase(), x, y);
          ctx.fillStyle = `rgba(160,155,187,${0.8 * labels * fadeOut})`;
          ctx.textAlign = "right";
          ctx.fillText(String(d.songs_per_class[genre]), x + PER_ROW * L.s, y);
          ctx.textAlign = "left";
        });
      }
    },
    [dots, colours],
  );

  const segmentsTotal = d.segments.train + d.segments.val + d.segments.test;
  const per = d.songs_per_class_per_split;

  return (
    <>
      <div className="f-fill">
        <ProgressCanvas
          draw={draw}
          label={`${int(d.total_songs)} songs, one dot each, grouped into ${d.classes.length} genres and split ${d.songs.train} train, ${d.songs.val} validation, ${d.songs.test} test`}
        />
      </div>

      <div className="da-copy f-pad">
        <div className="rv" style={at(0, 0.06, [0.3, 0.38])}>
          <p className="f-kicker">
            <b>03</b>The data
          </p>
          <p className="f-mega da-num">
            <ScrollNumber value={d.total_songs} format={(v) => int(Math.round(v))} a={0.02} b={0.2} />
            <span className="f-accent"> songs.</span>
          </p>
          <p className="f-copy">
            GTZAN: thirty-second clips of real recordings, each one a single dot here. Scattered, they are just files.
          </p>
        </div>

        <div className="rv da-abs" style={at(0.36, 0.44, [0.5, 0.56])}>
          <p className="f-kicker">
            <b>03</b>Ten genres
          </p>
          <p className="f-big">
            {d.classes.length} genres.
            <br />
            <span className="f-accent">{d.songs_per_class[d.classes[0]]} each.</span>
          </p>
          <p className="f-copy">Perfectly balanced, so the model can&apos;t score well by favouring one genre.</p>
        </div>

        <div className="rv da-abs" style={at(0.54, 0.62, [0.74, 0.8])}>
          <p className="f-kicker">
            <b>03</b>The split
          </p>
          <p className="f-big">
            Split by <span className="f-accent">song.</span>
          </p>
          <p className="f-copy">
            <b>{per.train}</b> train · <b>{per.val}</b> validation · <b>{per.test}</b> test songs per genre, stratified. The split happens{" "}
            <b>before</b> slicing, so no piece of a test song is ever seen in training. The bright dots are the {d.songs.test} test songs the model never saw.
          </p>
        </div>

        <div className="rv da-abs" style={at(0.8, 0.88, [0.95, 1])}>
          <p className="f-kicker">
            <b>03</b>The samples
          </p>
          <p className="f-big">
            <ScrollNumber value={segmentsTotal} format={(v) => int(Math.round(v))} a={0.8} b={0.92} />
            <span className="f-accent"> slices.</span>
          </p>
          <p className="f-copy">
            Each song is cut into {d.segment_seconds}-second windows: {int(d.segments.train)} train, {int(d.segments.val)} validation and{" "}
            {int(d.segments.test)} test. That is what the network actually trains on.
          </p>
        </div>
      </div>
    </>
  );
}

export default function DataScene() {
  return (
    <Chapter id="data" label="The data" vh={380} mobileVh={340}>
      <DataStage />
    </Chapter>
  );
}

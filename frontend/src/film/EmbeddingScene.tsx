import { useCallback, useMemo } from "react";
import type { EvalData } from "../data/types";
import { fixed, genreColor, genreLabel, int } from "../lib/format";
import { NeedsEval, useFilmData } from "./data";
import { Chapter, ProgressCanvas, ScrollNumber, at, easeInOut, lerp, seg } from "./engine";

interface Pt {
  sx: number;
  sy: number;
  dx: number;
  dy: number;
  genre: number;
  /** a CNN point with no MFCC twin in its genre: it fades in during the blend */
  extra: boolean;
}

/** Fit a point cloud into the unit square, centred. */
function normalise(xs: number[], ys: number[]) {
  const min = (a: number[]) => Math.min(...a);
  const max = (a: number[]) => Math.max(...a);
  const mx = (min(xs) + max(xs)) / 2;
  const my = (min(ys) + max(ys)) / 2;
  const scale = 2 / Math.max(max(xs) - min(xs), max(ys) - min(ys));
  return xs.map((x, i) => [(x - mx) * scale, (ys[i] - my) * scale] as const);
}

const median = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];

function build(data: EvalData) {
  const a = data.embeddings.mfcc_tsne;
  const b = data.embeddings.model_tsne;
  const from = normalise(a.x, a.y);
  const to = normalise(b.x, b.y);
  const bySource: number[][] = data.classes.map(() => []);
  a.genre.forEach((g, i) => bySource[g].push(i));
  const seen: number[] = data.classes.map(() => 0);
  const points: Pt[] = b.genre.map((g, i) => {
    const pool = bySource[g];
    const k = seen[g]++;
    const src = from[pool[k % pool.length]];
    return { sx: src[0], sy: src[1], dx: to[i][0], dy: to[i][1], genre: g, extra: k >= pool.length };
  });
  const centroids = data.classes.map((_, g) => {
    const mine = points.filter((p) => p.genre === g);
    return { x: median(mine.map((p) => p.dx)), y: median(mine.map((p) => p.dy)) };
  });
  return { points, centroids };
}

function hexToRgb(hex: string) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function Cloud({ data }: { data: EvalData }) {
  const { points, centroids } = useMemo(() => build(data), [data]);
  const colours = useMemo(() => data.classes.map((g) => hexToRgb(genreColor(g))), [data]);

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number, p: number) => {
      const wide = w >= 860;
      const x0 = wide ? w * 0.36 : w * 0.04;
      const x1 = w * 0.96;
      const y0 = wide ? h * 0.1 : h * 0.4;
      const y1 = h * 0.88;
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      const half = (Math.min(x1 - x0, y1 - y0) / 2) * 0.96;
      const t = easeInOut(seg(p, 0.26, 0.7));
      const fade = 1 - seg(p, 0.96, 1);
      const r = w < 640 ? 1.9 : 2.5;
      points.forEach((pt) => {
        const x = cx + lerp(pt.sx, pt.dx, t) * half;
        const y = cy + lerp(pt.sy, pt.dy, t) * half;
        const [cr, cg, cb] = colours[pt.genre];
        const alpha = (pt.extra ? seg(t, 0.15, 0.7) : 1) * 0.82 * fade;
        ctx.fillStyle = `rgba(${cr},${cg},${cb},${alpha})`;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      });
      const labels = seg(p, 0.8, 0.92) * fade;
      if (labels > 0) {
        ctx.font = "700 11px ui-monospace, Menlo, Consolas, monospace";
        ctx.textAlign = "center";
        centroids.forEach((c, g) => {
          const x = cx + c.x * half;
          const y = cy + c.y * half;
          ctx.fillStyle = `rgba(5,4,11,${0.55 * labels})`;
          ctx.fillText(genreLabel(data.classes[g]).toUpperCase(), x + 1, y + 1);
          ctx.fillStyle = `rgba(244,242,252,${0.95 * labels})`;
          ctx.fillText(genreLabel(data.classes[g]).toUpperCase(), x, y);
        });
      }
    },
    [points, centroids, colours, data.classes],
  );
  return <ProgressCanvas draw={draw} label="Two real t-SNE plots: MFCC feature space blending into the CNN's learned 128-dimensional space, coloured by genre" />;
}

function EmbeddingStage() {
  const { status, data } = useFilmData();
  const sil = data?.embeddings.silhouette;
  return (
    <>
      <div className="f-fill">
        {data ? (
          <Cloud data={data} />
        ) : (
          <div className="em-na f-pad">
            <NeedsEval status={status} what="The feature-space blend uses the t-SNE embeddings from the Colab export." />
          </div>
        )}
      </div>

      <div className="da-copy f-pad">
        <div className="rv" style={at(0, 0.06, [0.36, 0.44])}>
          <p className="f-kicker">
            <b>07</b>What it learned
          </p>
          <p className="f-big">
            Before:
            <br />
            <span className="f-warm">a blur.</span>
          </p>
          <p className="f-copy">
            Every point is a piece of music, coloured by genre. Here described by MFCC, the classic hand-made audio features: genres sit on top of each other.
          </p>
          {sil && (
            <p className="f-tag">
              silhouette <ScrollNumber value={sil.mfcc_features} format={(v) => fixed(v, 3)} a={0.08} b={0.24} className="em-sil" /> · ≈ 0 means no separation
            </p>
          )}
        </div>

        <div className="rv da-abs" style={at(0.72, 0.8, [0.97, 1])}>
          <p className="f-kicker">
            <b>07</b>What it learned
          </p>
          <p className="f-big">
            After:
            <br />
            <span className="f-accent">structure.</span>
          </p>
          <p className="f-copy">
            The network&apos;s own 128-number summary of each test segment ({data ? int(data.embeddings.model_tsne.x.length) : "1,499"} of them). Nobody told it what
            a cluster is; it was only asked to name the genre.
          </p>
          {sil && (
            <p className="f-tag">
              silhouette <ScrollNumber value={sil.cnn_embedding} format={(v) => fixed(v, 3)} a={0.74} b={0.9} className="em-sil" /> · structured, but the edges still
              overlap
            </p>
          )}
        </div>

        <p className="f-note em-foot rv" style={at(0.1, 0.16, [0.97, 1])}>
          Two real t-SNE plots of different samples: 999 song-level MFCC vectors, and 1,499 test segments in the CNN&apos;s 128-D layer. The blend pairs points by
          genre only, so no single point is tracked from one plot to the other.
        </p>
      </div>
    </>
  );
}

export default function EmbeddingScene() {
  return (
    <Chapter id="learned" label="What it learned" vh={380} mobileVh={340}>
      <EmbeddingStage />
    </Chapter>
  );
}

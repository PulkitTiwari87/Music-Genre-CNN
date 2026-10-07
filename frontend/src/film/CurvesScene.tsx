import { useCallback, useMemo, useRef } from "react";
import type { EvalData } from "../data/types";
import { fixed, genreColor, genreLabel, pct } from "../lib/format";
import { NeedsEval, useFilmData } from "./data";
import { Chapter, ScrollNumber, at, easeInOut, seg, useProgress } from "./engine";

interface Curve {
  x: number[];
  y: number[];
  score: number;
}

interface Spec {
  id: string;
  index: string;
  kicker: string;
  title: [string, string];
  copy: string;
  xLabel: string;
  yLabel: string;
  scoreName: string;
  microName: string;
  headlineName: string;
  baseline: "diagonal" | { y: number };
  perClass: (d: EvalData) => Record<string, Curve | null>;
  micro: (d: EvalData) => Curve;
  headline: (d: EvalData) => number;
  readout: (x: number, y: number) => string;
}

const ROC: Spec = {
  id: "roc",
  index: "10",
  kicker: "ROC-AUC",
  title: ["Can it tell", "them apart?"],
  copy: "Slide the decision threshold from strict to lenient. Strict: few false alarms, but songs get missed. Lenient: nothing is missed, but false alarms pile up. The ROC curve records every setting at once.",
  xLabel: "false alarms: share of other-genre songs wrongly accepted",
  yLabel: "songs caught",
  scoreName: "AUC",
  microName: "all genres pooled",
  headlineName: "macro ROC-AUC",
  baseline: "diagonal",
  perClass: (d) =>
    Object.fromEntries(Object.entries(d.song.roc.per_class).map(([g, c]) => [g, c ? { x: c.fpr, y: c.tpr, score: c.auc } : null])),
  micro: (d) => ({ x: d.song.roc.micro.fpr, y: d.song.roc.micro.tpr, score: d.song.roc.micro.auc }),
  headline: (d) => d.song.roc.macro_auc,
  readout: (x, y) => `false alarms ${pct(x, 0)} · songs caught ${pct(y, 0)}`,
};

const PR: Spec = {
  id: "pr",
  index: "11",
  kicker: "PR-AUC",
  title: ["When it speaks up,", "is it right?"],
  copy: "Precision: of the songs it calls jazz, how many are jazz. Recall: of all the jazz songs, how many it found. Ask it to find more, and precision usually drops. The curve is that trade-off.",
  xLabel: "recall: share of a genre's songs found",
  yLabel: "precision",
  scoreName: "AP",
  microName: "all genres pooled",
  headlineName: "macro PR-AUC (average precision)",
  baseline: { y: 0.1 },
  perClass: (d) =>
    Object.fromEntries(Object.entries(d.song.pr.per_class).map(([g, c]) => [g, c ? { x: c.recall, y: c.precision, score: c.ap } : null])),
  micro: (d) => ({ x: d.song.pr.micro.recall, y: d.song.pr.micro.precision, score: d.song.pr.micro.ap }),
  headline: (d) => d.song.pr.macro_ap,
  readout: (x, y) => `recall ${pct(x, 0)} · precision ${pct(y, 0)}`,
};

const W = 560;
const H = 540;
const PAD = { l: 48, r: 14, t: 14, b: 44 };
const IW = W - PAD.l - PAD.r;
const IH = H - PAD.t - PAD.b;
const px = (x: number) => PAD.l + x * IW;
const py = (y: number) => PAD.t + (1 - y) * IH;

const path = (c: { x: number[]; y: number[] }) => c.x.map((x, i) => `${i ? "L" : "M"}${px(x).toFixed(1)},${py(c.y[i]).toFixed(1)}`).join("");

/** y of the curve point whose x is nearest to `x`. */
function valueAt(c: { x: number[]; y: number[] }, x: number) {
  let best = 0;
  for (let i = 1; i < c.x.length; i += 1) if (Math.abs(c.x[i] - x) < Math.abs(c.x[best] - x)) best = i;
  return c.y[best];
}

const tick = (t: number) => (t === 0 ? "0" : t === 1 ? "1" : String(t));

function CurveStage({ spec, data }: { spec: Spec; data: EvalData }) {
  const clip = useRef<SVGRectElement>(null);
  const dot = useRef<SVGCircleElement>(null);
  const readout = useRef<HTMLParagraphElement>(null);
  const clipId = `cv-${spec.id}`;

  const { per, micro, ranked } = useMemo(() => {
    const per = spec.perClass(data);
    const ranked = data.classes
      .filter((g) => per[g])
      .map((g) => ({ genre: g, score: per[g]!.score }))
      .sort((a, b) => b.score - a.score);
    return { per, micro: spec.micro(data), ranked };
  }, [spec, data]);

  const sweep = useCallback(
    (p: number) => {
      const s = easeInOut(seg(p, 0.12, 0.64));
      clip.current?.setAttribute("width", String(PAD.l + s * IW + 1));
      const y = valueAt(micro, s);
      if (dot.current) {
        dot.current.setAttribute("cx", px(s).toFixed(1));
        dot.current.setAttribute("cy", py(y).toFixed(1));
        dot.current.style.opacity = p > 0.1 && p < 0.98 ? "1" : "0";
      }
      if (readout.current) readout.current.textContent = spec.readout(s, y);
    },
    [micro, spec],
  );
  useProgress(sweep);

  const best = ranked[0];
  const worst = ranked[ranked.length - 1];
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="cv f-pad">
      <div className="cv-left">
        <div className="rv" style={at(0, 0.05, [0.97, 1])}>
          <p className="f-kicker">
            <b>{spec.index}</b>
            {spec.kicker}
          </p>
          <h2 className="f-big cv-title">
            {spec.title[0]}
            <br />
            <span className="f-accent">{spec.title[1]}</span>
          </h2>
        </div>
        <p className="f-copy rv" style={at(0.04, 0.1, [0.97, 1])}>
          {spec.copy}
        </p>
        <p ref={readout} className="cv-readout f-num rv" style={at(0.1, 0.14, [0.66, 0.7], { "--dy-out": "0px" })} aria-hidden="true" />

        <div className="cv-score rv" style={at(0.66, 0.72, [0.97, 1])}>
          <p className="f-tag">{spec.headlineName}</p>
          <p className="cv-big f-num">
            <ScrollNumber value={spec.headline(data)} format={(v) => fixed(v, 3)} a={0.66} b={0.8} />
          </p>
          <p className="f-copy">
            Cleanest: <b>{genreLabel(best.genre)}</b> {fixed(best.score, 3)}. Hardest: <b>{genreLabel(worst.genre)}</b> {fixed(worst.score, 3)}.
          </p>
        </div>
      </div>

      <figure className="cv-plot rv" style={at(0.04, 0.1, [0.97, 1])}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${spec.kicker} curves at song level: one curve per genre and a pooled curve. ${spec.headlineName} ${fixed(spec.headline(data), 3)}`}>
          <defs>
            <clipPath id={clipId}>
              <rect ref={clip} x={0} y={0} width={0} height={H} />
            </clipPath>
          </defs>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={px(0)} x2={px(1)} y1={py(t)} y2={py(t)} className="tr-grid" />
              <line x1={px(t)} x2={px(t)} y1={py(0)} y2={py(1)} className="tr-grid" />
              <text x={px(0) - 8} y={py(t) + 3} textAnchor="end" className="tr-axis">
                {tick(t)}
              </text>
              <text x={px(t)} y={py(0) + 16} textAnchor="middle" className="tr-axis">
                {tick(t)}
              </text>
            </g>
          ))}
          {spec.baseline === "diagonal" ? (
            <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} className="cv-base" />
          ) : (
            <line x1={px(0)} y1={py(spec.baseline.y)} x2={px(1)} y2={py(spec.baseline.y)} className="cv-base" />
          )}
          <text x={px(1) - 4} y={spec.baseline === "diagonal" ? py(0.5) + 34 : py(spec.baseline.y) - 6} textAnchor="end" className="tr-axis">
            {spec.baseline === "diagonal" ? "guessing: 0.5" : "guessing: 0.10"}
          </text>
          <text x={px(0.5)} y={H - 6} textAnchor="middle" className="tr-axis">
            {spec.xLabel}
          </text>
          <text transform={`translate(12 ${py(0.5)}) rotate(-90)`} textAnchor="middle" className="tr-axis">
            {spec.yLabel}
          </text>
          <g clipPath={`url(#${clipId})`}>
            {ranked.map((r) => (
              <path key={r.genre} d={path(per[r.genre]!)} fill="none" stroke={genreColor(r.genre)} strokeWidth={1.5} opacity={0.62} strokeLinejoin="round" />
            ))}
            <path d={path(micro)} fill="none" stroke="#fcfdbf" strokeWidth={3.2} strokeLinejoin="round" strokeLinecap="round" />
          </g>
          <circle ref={dot} r={6} fill="#fcfdbf" stroke="#05040b" strokeWidth={2} style={{ opacity: 0 }} />
        </svg>
      </figure>

      <ol className="cv-legend rv" style={at(0.74, 0.82, [0.97, 1])} aria-label={`${spec.scoreName} by genre`}>
        {ranked.map((r) => (
          <li key={r.genre}>
            <i style={{ background: genreColor(r.genre) }} />
            <span>{genreLabel(r.genre)}</span>
            <b className="f-num">{fixed(r.score, 3)}</b>
          </li>
        ))}
        <li className="cv-pooled">
          <i style={{ background: "#fcfdbf" }} />
          <span>{spec.microName}</span>
          <b className="f-num">{fixed(micro.score, 3)}</b>
        </li>
      </ol>
    </div>
  );
}

function CurvesChapter({ spec }: { spec: Spec }) {
  const { status, data } = useFilmData();
  return (
    <Chapter id={spec.id} label={spec.kicker} vh={400} mobileVh={360}>
      {data ? (
        <CurveStage spec={spec} data={data} />
      ) : (
        <div className="f-pad">
          <NeedsEval status={status} what={`The ${spec.kicker} scene draws the model's real curves.`} />
        </div>
      )}
    </Chapter>
  );
}

export const RocScene = () => <CurvesChapter spec={ROC} />;
export const PrScene = () => <CurvesChapter spec={PR} />;

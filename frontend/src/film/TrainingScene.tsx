import { useCallback, useImperativeHandle, useMemo, useRef, type CSSProperties, type Ref } from "react";
import { notebook } from "../data";
import type { EpochRecord } from "../data/types";
import { summarizeFit } from "../lib/analysis";
import { fixed, int, pct } from "../lib/format";
import { Chapter, at, easeInOut, seg, useProgress } from "./engine";

const run = notebook.training.v3;
const epochs = run.epochs;
const N = epochs.length;

const W = 600;
const H = 230;
const PAD = { l: 38, r: 10, t: 12, b: 24 };
const IW = W - PAD.l - PAD.r;
const IH = H - PAD.t - PAD.b;

const xOf = (epoch: number) => PAD.l + ((epoch - 1) / (N - 1)) * IW;

const at1 = (key: keyof EpochRecord, e: number) => {
  const f = Math.min(N - 1, Math.max(0, e - 1));
  const i = Math.floor(f);
  const a = epochs[i][key] as number;
  const b = epochs[Math.min(N - 1, i + 1)][key] as number;
  return a + (b - a) * (f - i);
};

interface ChartHandle {
  update: (epoch: number) => void;
}

interface Line {
  key: keyof EpochRecord;
  name: string;
  color: string;
}

function LiveChart({
  title,
  lines,
  domain,
  format,
  ref,
  marks,
}: {
  title: string;
  lines: Line[];
  domain: [number, number];
  format: (v: number) => string;
  ref: Ref<ChartHandle>;
  marks?: { epoch: number }[];
}) {
  const id = useMemo(() => `tr-${title.replace(/\W+/g, "")}`, [title]);
  const clip = useRef<SVGRectElement>(null);
  const dots = useRef<(SVGCircleElement | null)[]>([]);
  const star = useRef<SVGGElement>(null);
  const y = (v: number) => PAD.t + IH - ((v - domain[0]) / (domain[1] - domain[0])) * IH;
  const paths = lines.map((l) => epochs.map((e, i) => `${i ? "L" : "M"}${xOf(e.epoch).toFixed(1)},${y(e[l.key] as number).toFixed(1)}`).join(""));
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => domain[0] + t * (domain[1] - domain[0]));
  const best = run.best_val_accuracy_epoch;

  useImperativeHandle(ref, () => ({
    update(e) {
      clip.current?.setAttribute("width", String(Math.max(0, xOf(e) - PAD.l + 1)));
      lines.forEach((l, i) => {
        const dot = dots.current[i];
        if (!dot) return;
        dot.setAttribute("cx", xOf(e).toFixed(1));
        dot.setAttribute("cy", Math.max(PAD.t, Math.min(PAD.t + IH, y(at1(l.key, e)))).toFixed(1));
        dot.style.opacity = e > 1.001 ? "1" : "0";
      });
      if (star.current) star.current.style.opacity = best !== null && e >= best ? "1" : "0";
    },
  }));

  return (
    <figure className="tr-chart">
      <figcaption className="f-tag">
        {title}
        {lines.map((l) => (
          <span key={l.name} style={{ color: l.color }}>
            {" "}
            ● {l.name}
          </span>
        ))}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} by epoch, training and validation`} preserveAspectRatio="xMidYMid meet">
        <defs>
          <clipPath id={id}>
            <rect ref={clip} x={PAD.l} y={0} width={0} height={H} />
          </clipPath>
        </defs>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} className="tr-grid" />
            <text x={PAD.l - 6} y={y(t) + 3} textAnchor="end" className="tr-axis">
              {format(t)}
            </text>
          </g>
        ))}
        {[1, 10, 20, N].map((e) => (
          <text key={e} x={xOf(e)} y={H - 6} textAnchor="middle" className="tr-axis">
            {e}
          </text>
        ))}
        {marks?.map((m) => (
          <line key={m.epoch} x1={xOf(m.epoch)} x2={xOf(m.epoch)} y1={PAD.t} y2={PAD.t + IH} className="tr-mark" />
        ))}
        <g clipPath={`url(#${id})`}>
          {lines.map((l, i) => (
            <path key={l.name} d={paths[i]} fill="none" stroke={l.color} strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round" />
          ))}
        </g>
        {lines.map((l, i) => (
          <circle
            key={l.name}
            ref={(el) => {
              dots.current[i] = el;
            }}
            r={4.5}
            fill={l.color}
            style={{ opacity: 0 }}
          />
        ))}
        {best !== null && lines.some((l) => l.key === "val_accuracy") && (
          <g ref={star} style={{ opacity: 0 }}>
            <circle cx={xOf(best)} cy={y(epochs[best - 1].val_accuracy)} r={9} fill="none" stroke="#fcfdbf" strokeWidth={1.5} />
            <text x={xOf(best)} y={y(epochs[best - 1].val_accuracy) + 24} textAnchor="middle" className="tr-axis tr-best">
              best · epoch {best}
            </text>
          </g>
        )}
      </svg>
    </figure>
  );
}

const lrText = (lr: number | null) => (lr === null ? "n/a" : lr >= 0.001 ? lr.toFixed(3) : lr.toExponential(1));

function TrainingStage() {
  const acc = useRef<ChartHandle>(null);
  const loss = useRef<ChartHandle>(null);
  const epochEl = useRef<HTMLSpanElement>(null);
  const readout = useRef<Record<string, HTMLElement | null>>({});
  const noteEl = useRef<HTMLParagraphElement>(null);

  const fit = useMemo(() => summarizeFit(run), []);
  const lossMax = useMemo(() => Math.ceil(Math.max(...epochs.slice(1).map((e) => Math.max(e.val_loss, e.loss))) * 4) / 4, []);
  const notes = useMemo(() => {
    const first = epochs[0];
    const list: { epoch: number; text: string }[] = [
      {
        epoch: 1,
        text: `Epoch 1: it is guessing. Validation accuracy ${pct(first.val_accuracy, 1)}; with ten genres, chance is 10%. (Its validation loss, ${fixed(first.val_loss, 2)}, is off this chart.)`,
      },
      ...run.learning_rate_reductions.slice(0, 1).map((r) => ({
        epoch: r.epoch,
        text: `Epoch ${r.epoch}: validation loss stalled, so the learning rate was cut to ${lrText(r.to)}. It is cut ${run.learning_rate_reductions.length} times in all.`,
      })),
    ];
    if (run.best_val_accuracy_epoch !== null) {
      list.push({
        epoch: run.best_val_accuracy_epoch,
        text: `Epoch ${run.best_val_accuracy_epoch}: best validation accuracy, ${pct(run.best_val_accuracy, 1)}. This checkpoint is the model that was evaluated and deployed.`,
      });
    }
    const stop = run.early_stopped_at_epoch ?? N;
    list.push({
      epoch: stop,
      text: `Epoch ${stop}: early stopping. Training accuracy ${pct(fit.finalTrainAcc, 1)} against validation ${pct(fit.finalValAcc, 1)}: ${Math.round(fit.gap * 100)} points apart. It is memorising its ${notebook.dataset.songs.train} training songs.`,
    });
    return list;
  }, [fit]);

  const play = useCallback(
    (p: number) => {
      const e = 1 + (N - 1) * easeInOut(seg(p, 0.08, 0.78));
      acc.current?.update(e);
      loss.current?.update(e);
      const shown = Math.round(e);
      if (epochEl.current) epochEl.current.textContent = String(shown).padStart(2, "0");
      const r = epochs[Math.min(N - 1, shown - 1)];
      const set = (k: string, v: string) => {
        const el = readout.current[k];
        if (el) el.textContent = v;
      };
      set("acc", pct(r.accuracy, 1));
      set("val", pct(r.val_accuracy, 1));
      set("loss", fixed(r.loss, 3));
      set("vloss", fixed(r.val_loss, 3));
      set("lr", lrText(r.learning_rate));
      const note = [...notes].reverse().find((n) => n.epoch <= shown);
      if (noteEl.current) noteEl.current.textContent = note ? note.text : "";
    },
    [notes],
  );
  useProgress(play);

  const stat = (key: string, label: string) => (
    <div>
      <dt className="f-tag">{label}</dt>
      <dd
        ref={(el) => {
          readout.current[key] = el;
        }}
        className="f-num"
      />
    </div>
  );

  const ladder = (["v1", "v2", "v3"] as const).map((k) => ({
    key: k,
    accuracy: notebook.results[k].segment_accuracy,
    params: notebook.models[k].total,
    note: { v1: "plain CNN, Flatten + big dense layer", v2: "smaller model, L2 and dropout", v3: "global pooling, SpecAugment, LR schedule, best checkpoint" }[k],
  }));

  return (
    <div className="tr f-pad">
      <div className="tr-left">
        <div className="rv" style={at(0, 0.05, [0.97, 1])}>
          <p className="f-kicker">
            <b>06</b>Training
          </p>
          <h2 className="f-big tr-title">
            It learns fast,
            <br />
            then starts <span className="f-accent">memorising.</span>
          </h2>
        </div>

        <div className="tr-swap">
        <div className="tr-now rv" style={at(0.08, 0.14, [0.86, 0.9], { "--dy-out": "-18px" })}>
          <p className="tr-epoch">
            <span className="f-tag">epoch </span>
            <span ref={epochEl} className="f-num" />
            <span className="f-tag"> / {N}</span>
          </p>
          <dl className="tr-stats">
            {stat("acc", "train acc")}
            {stat("val", "val acc")}
            {stat("loss", "train loss")}
            {stat("vloss", "val loss")}
            {stat("lr", "learning rate")}
          </dl>
          <p ref={noteEl} className="f-copy tr-note" aria-live="off" />
        </div>

        <div className="tr-ladder rv" style={at(0.89, 0.94, [0.985, 1])}>
          <p className="f-kicker">Three attempts, same split</p>
          <ol>
            {ladder.map((l) => (
              <li key={l.key}>
                <span className="f-tag">{l.key.toUpperCase()}</span>
                <i style={{ "--w": l.accuracy } as CSSProperties} />
                <b className="f-num">{pct(l.accuracy, 1)}</b>
                <em>
                  {int(l.params)} params · {l.note}
                </em>
              </li>
            ))}
          </ol>
          <p className="f-note">Segment-level test accuracy of each version. The best of the three, V3, is the one used everywhere else in this film.</p>
        </div>
        </div>
      </div>

      <div className="tr-right rv" style={at(0.06, 0.12, [0.97, 1])}>
        <LiveChart
          ref={acc}
          title="Accuracy"
          domain={[0, 1]}
          format={(v) => `${Math.round(v * 100)}%`}
          lines={[
            { key: "accuracy", name: "train", color: "#8f78ff" },
            { key: "val_accuracy", name: "validation", color: "#ffb45a" },
          ]}
          marks={run.learning_rate_reductions}
        />
        <LiveChart
          ref={loss}
          title="Loss"
          domain={[0, lossMax]}
          format={(v) => v.toFixed(1)}
          lines={[
            { key: "loss", name: "train", color: "#8f78ff" },
            { key: "val_loss", name: "validation", color: "#ffb45a" },
          ]}
          marks={run.learning_rate_reductions}
        />
        <p className="f-note">Faint lines: learning-rate reductions. Recorded epoch by epoch in the notebook (cell {run.notebook_cell}).</p>
      </div>
    </div>
  );
}

export default function TrainingScene() {
  return (
    <Chapter id="training" label="Training" vh={440} mobileVh={400}>
      <TrainingStage />
    </Chapter>
  );
}

import { useMemo, type ReactNode } from "react";
import { CLASSES, notebook } from "../data";
import { statsFromMatrix } from "../lib/analysis";
import { fixed, pct } from "../lib/format";
import { NeedsEval, useFilmData } from "./data";
import { Chapter, ScrollNumber, at, useStep } from "./engine";

const SLIDES = 4;
const BREAKS = [0, 0.25, 0.5, 0.75] as const;
const NAMES = ["Accuracy", "F1 score", "ROC-AUC", "PR-AUC"] as const;

function Slide({
  index,
  label,
  value,
  format,
  unit,
  children,
}: {
  index: number;
  label: string;
  value: number | null;
  format: (v: number) => string;
  unit?: string;
  children: ReactNode;
}) {
  const start = index / SLIDES;
  const last = index === SLIDES - 1;
  const shown = index === 0 ? 0 : start;
  return (
    <div className="rs-slide rv" style={at(shown, shown + 0.05, [last ? 0.97 : start + 0.2, last ? 1 : start + 0.25], { "--dy-out": "-40px" })}>
      <p className="f-kicker">
        <b>09</b>The results · {index + 1} of {SLIDES}
      </p>
      <h2 className="rs-label">{label}</h2>
      <p className="rs-num f-num">
        {value === null ? (
          <span className="f-dim">n/a</span>
        ) : (
          <>
            <ScrollNumber value={value} format={format} a={start + 0.02} b={start + 0.12} />
            {unit && <small>{unit}</small>}
          </>
        )}
      </p>
      <div className="rs-ctx">{children}</div>
    </div>
  );
}

function ResultsStage() {
  const { status, data } = useFilmData();
  const step = Math.max(0, useStep(BREAKS));
  const r = notebook.results.v3;
  const songF1 = useMemo(() => {
    const stats = statsFromMatrix(notebook.confusion.v3_song.matrix, CLASSES);
    return CLASSES.reduce((sum, g) => sum + stats[g].f1, 0) / CLASSES.length;
  }, []);
  const correct = Math.round(r.song_accuracy * notebook.dataset.songs.test);

  return (
    <div className="rs f-pad">
      <ol className="rs-nav" aria-label="Metrics">
        {NAMES.map((n, i) => (
          <li key={n} data-on={i === step} data-done={i < step}>
            {n}
          </li>
        ))}
      </ol>

      <Slide index={0} label="Accuracy" value={r.song_accuracy * 100} format={(v) => v.toFixed(1)} unit="%">
        <p className="f-copy">
          <b>
            {correct} of {notebook.dataset.songs.test}
          </b>{" "}
          held-out songs placed in the right genre. Chance would be 10%.
        </p>
        <p className="f-tag">per 3-second slice: {pct(r.segment_accuracy, 1)}</p>
      </Slide>

      <Slide index={1} label="F1 score" value={songF1} format={(v) => fixed(v, 3)}>
        <p className="f-copy">
          Precision and recall in one number, <b>averaged so every genre counts equally</b>, so a strong genre can&apos;t hide a weak one.
        </p>
        <p className="f-tag">per 3-second slice: {fixed(notebook.reports.v3_segment.macro.f1, 3)}</p>
      </Slide>

      <Slide index={2} label="ROC-AUC" value={data ? data.song.roc.macro_auc : null} format={(v) => fixed(v, 3)}>
        {data ? (
          <>
            <p className="f-copy">
              Ignore the cut-off. Pick a song of a genre and one that isn&apos;t: how often does the model score the right one higher?{" "}
              <b>0.5 is a coin flip, 1.0 is perfect.</b>
            </p>
            <p className="f-tag">per 3-second slice: {fixed(data.segment.roc.macro_auc, 3)} · one genre against the rest, averaged</p>
          </>
        ) : (
          <NeedsEval status={status} what="ROC-AUC needs the model's probabilities." />
        )}
      </Slide>

      <Slide index={3} label="PR-AUC" value={data ? data.song.pr.macro_ap : null} format={(v) => fixed(v, 3)}>
        {data ? (
          <>
            <p className="f-copy">
              When the model says &quot;this is jazz&quot;, how often is it right, and how much of the jazz did it find? The area under that trade-off.{" "}
              <b>A model that guesses scores 0.10.</b>
            </p>
            <p className="f-tag">per 3-second slice: {fixed(data.segment.pr.macro_ap, 3)} · average precision, macro</p>
            {data.split_check.exact === false && (
              <p className="f-note">
                The Colab re-run predicted {Math.abs(data.split_check.delta_segments ?? 0)} test segments differently from the original run; ROC and PR come from the re-run.
              </p>
            )}
          </>
        ) : (
          <NeedsEval status={status} what="PR-AUC needs the model's probabilities." />
        )}
      </Slide>
    </div>
  );
}

export default function ResultsScene() {
  return (
    <Chapter id="results" label="The results" vh={460} mobileVh={400}>
      <ResultsStage />
    </Chapter>
  );
}

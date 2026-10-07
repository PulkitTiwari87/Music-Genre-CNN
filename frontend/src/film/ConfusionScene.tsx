import { useMemo, type CSSProperties } from "react";
import { notebook } from "../data";
import { genreLabel } from "../lib/format";
import { Chapter, at } from "./engine";

const song = notebook.confusion.v3_song;
const segment = notebook.confusion.v3_segment;
const labels = song.labels;

const sum = (v: number[]) => v.reduce((a, b) => a + b, 0);

function analyse() {
  const { matrix } = song;
  const total = sum(matrix.map(sum));
  const correct = matrix.reduce((acc, row, i) => acc + row[i], 0);
  const byTrue = matrix
    .map((row, i) => ({ index: i, genre: labels[i], errors: sum(row) - row[i], total: sum(row) }))
    .sort((a, b) => b.errors - a.errors);
  const focus = byTrue[0];
  const row = matrix[focus.index];
  const strayed = row
    .map((count, j) => ({ to: labels[j], count, j }))
    .filter((c) => c.j !== focus.index && c.count > 0)
    .sort((a, b) => b.count - a.count);
  const clean = byTrue.filter((e) => e.errors === 0).map((e) => e.genre);

  // Largest confusion at slice level, counting both directions of a genre pair.
  let pair = { a: "", b: "", count: 0 };
  segment.matrix.forEach((r, i) =>
    r.forEach((_, j) => {
      if (j > i) {
        const count = segment.matrix[i][j] + segment.matrix[j][i];
        if (count > pair.count) pair = { a: labels[i], b: labels[j], count };
      }
    }),
  );
  return { matrix, total, correct, errors: total - correct, focus, strayed, clean, pair, second: byTrue[1] };
}

const short = (g: string) => genreLabel(g).slice(0, 3);

function ConfusionStage() {
  const a = useMemo(() => analyse(), []);
  const list = (items: string[]) => (items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`);

  return (
    <>
      <div className="cm-wrap" style={{ "--fy": `${((a.focus.index + 1.5) / (labels.length + 1)) * 100}%` } as CSSProperties} role="img" aria-label={`Song-level confusion matrix, ${a.correct} of ${a.total} test songs on the diagonal`}>
        <div className="cm-grid">
          <span className="cm-corner f-tag">true ↓ · predicted →</span>
          {labels.map((g) => (
            <span key={`c-${g}`} className="cm-axis cm-col">
              <span className="cm-full">{genreLabel(g)}</span>
              <span className="cm-short">{short(g)}</span>
            </span>
          ))}
          {a.matrix.map((row, i) => (
            <div key={labels[i]} className="cm-row">
              <span className="cm-axis cm-rowlabel">
                <span className="cm-full">{genreLabel(labels[i])}</span>
                <span className="cm-short">{short(labels[i])}</span>
              </span>
              {row.map((count, j) => {
                const t = count / (sum(row) || 1);
                const diag = i === j;
                const reveal = diag ? 0.05 + i * 0.016 : 0.36 + ((i * labels.length + j) / (labels.length * labels.length)) * 0.14;
                return (
                  <span
                    key={j}
                    className={`cm-cell${diag ? " diag" : count ? " miss" : ""}`}
                    style={
                      {
                        "--a": reveal,
                        "--dim": i === a.focus.index ? 0 : 0.78,
                        "--t": diag ? 0.22 + 0.78 * t : 0.3 + 0.7 * Math.min(1, t * 3),
                      } as CSSProperties
                    }
                  >
                    {count || ""}
                  </span>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="da-copy cm-copy f-pad">
        <div className="rv" style={at(0, 0.06, [0.3, 0.36])}>
          <p className="f-kicker">
            <b>12</b>Where it is right
          </p>
          <p className="f-big">
            {a.correct} of {a.total}.
          </p>
          <p className="f-copy">
            Each row is a genre&apos;s test songs; each column, what the model said. The bright diagonal is the model being right: a song landing in its own genre.
          </p>
        </div>

        <div className="rv da-abs" style={at(0.34, 0.4, [0.56, 0.62])}>
          <p className="f-kicker">
            <b>12</b>Where it is wrong
          </p>
          <p className="f-big">
            {a.errors} misses.
            <br />
            <span className="f-warm">Not random.</span>
          </p>
          <p className="f-copy">
            <b>{genreLabel(a.focus.genre)}</b> accounts for {a.focus.errors} of them, <b>{genreLabel(a.second.genre)}</b> for {a.second.errors}.{" "}
            {a.clean.length > 0 && (
              <>
                {list(a.clean.map(genreLabel))} {a.clean.length > 1 ? "have" : "has"} none.
              </>
            )}
          </p>
        </div>

        <div className="rv da-abs" style={at(0.62, 0.68, [0.94, 0.98])}>
          <p className="f-kicker">
            <b>12</b>The hardest genre
          </p>
          <p className="f-big">
            {genreLabel(a.focus.genre)} <span className="f-warm">gets lost.</span>
          </p>
          <p className="f-copy">
            Of {a.focus.total} {genreLabel(a.focus.genre).toLowerCase()} songs, {a.focus.total - a.focus.errors} were right. The rest went to{" "}
            {list(a.strayed.map((s) => `${genreLabel(s.to)} (${s.count})`))}. At the level of 3-second slices the biggest single confusion is{" "}
            <b>
              {genreLabel(a.pair.a)} ↔ {genreLabel(a.pair.b)}
            </b>
            : {a.pair.count} slices, both directions.
          </p>
        </div>
      </div>
    </>
  );
}

export default function ConfusionScene() {
  return (
    <Chapter id="confusion" label="Confusion matrix" vh={400} mobileVh={360}>
      <ConfusionStage />
    </Chapter>
  );
}

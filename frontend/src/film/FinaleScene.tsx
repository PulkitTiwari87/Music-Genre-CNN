import { model, notebook } from "../data";
import { pct } from "../lib/format";
import { Link } from "../router";
import { Chapter, ScrollNumber, at } from "./engine";

const CHAIN = ["Music", "Signal", "Features", "Learning", "Prediction"] as const;

/** Every name below is imported or declared somewhere in this repository. */
const STACK = ["TensorFlow", "Keras", "librosa", "scikit-learn", "FastAPI", "React", "TypeScript", "Vite"] as const;

function FinaleStage() {
  const r = notebook.results.v3;
  const total = notebook.dataset.songs.test;
  const wrong = total - Math.round(r.song_accuracy * total);
  return (
    <>
      <div className="fi-layer f-pad rv" style={at(0, 0.05, [0.2, 0.26])}>
        <h2 className="f-mega fi-q">
          Can a machine
          <br />
          learn the <span className="f-accent">structure</span>
          <br />
          of music?
        </h2>
      </div>

      <div className="fi-layer f-pad rv" style={at(0.26, 0.32, [0.46, 0.52])}>
        <p className="f-kicker">The answer, on songs it had never heard</p>
        <p className="fi-num f-num">
          <ScrollNumber value={r.song_accuracy * 100} format={(v) => v.toFixed(1)} a={0.28} b={0.4} />
          <small>%</small>
        </p>
        <p className="f-copy">
          named the right genre. Not perfectly: {wrong} of {total} were wrong, and the film showed where. But the structure was there to be learned.
        </p>
      </div>

      <div className="fi-layer f-pad fi-chain-wrap">
        <ol className="fi-chain" aria-label="Music, signal, features, learning, prediction">
          {CHAIN.map((w, i) => (
            <li key={w} className="rv" style={at(0.5 + i * 0.05, 0.55 + i * 0.05, [0.78, 0.84], { "--dy": "20px", "--dy-out": "-20px" })}>
              {i > 0 && <span aria-hidden="true">→</span>}
              <b className="f-big">{w}</b>
            </li>
          ))}
        </ol>
      </div>

      <div className="fi-layer f-pad fi-final rv" style={at(0.84, 0.92, undefined, { "--dy": "16px" })}>
        <h2 className="fi-title">
          Music genre
          <br />
          classification
        </h2>
        <ul className="fi-stack" aria-label="Technology stack">
          {STACK.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
        <p className="f-tag">
          CNN V3 · {pct(r.song_accuracy, 1)} on {total} unseen songs · Keras {model.framework.saved_with_keras}
        </p>
        <p className="fi-links">
          <Link to="/">Try it on your own song</Link>
          <Link to="/model">Model details</Link>
        </p>
      </div>
    </>
  );
}

export default function FinaleScene() {
  return (
    <Chapter id="finale" label="Finale" vh={420} mobileVh={380}>
      <FinaleStage />
    </Chapter>
  );
}

import { useEffect, useRef, useState, type DragEvent } from "react";
import { ACCEPTED_EXTENSIONS, predictGenre, validateFile, type Prediction } from "./api";

const GENRE_LABELS: Record<string, string> = { hiphop: "Hip-Hop" };

const genreLabel = (genre: string) =>
  GENRE_LABELS[genre] ?? genre.charAt(0).toUpperCase() + genre.slice(1);

const formatSize = (bytes: number) =>
  bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`;

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

type State =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; result: Prediction }
  | { status: "error"; message: string };

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<State>({ status: "idle" });
  const [dragging, setDragging] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  function choose(selected: File | undefined) {
    if (!selected) return;
    abortRef.current?.abort();
    const problem = validateFile(selected);
    if (problem) {
      setFile(null);
      setState({ status: "error", message: problem });
      return;
    }
    setFile(selected);
    setState({ status: "idle" });
  }

  async function analyze() {
    if (!file) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setState({ status: "loading" });
    try {
      const result = await predictGenre(file, controller.signal);
      setState({ status: "done", result });
    } catch (error) {
      if (controller.signal.aborted) return;
      setState({
        status: "error",
        message: error instanceof Error ? error.message : "Something went wrong.",
      });
    }
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files[0]);
  }

  const loading = state.status === "loading";

  return (
    <main className="page">
      <header className="hero">
        <p className="eyebrow">Deep learning · GTZAN · 10 genres</p>
        <h1>Music Genre CNN</h1>
        <p className="lede">
          Drop in a song. A convolutional network reads its Mel spectrogram, listens to every
          3-second slice, and votes on the genre.
        </p>
      </header>

      <section className="panel" aria-label="Upload">
        <label
          className={`dropzone${dragging ? " dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            type="file"
            aria-label="Audio file"
            accept={ACCEPTED_EXTENSIONS.join(",")}
            onChange={(e) => {
              choose(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          <span className="dropzone-icon" aria-hidden="true">
            <svg viewBox="0 0 32 32" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
              <path d="M7 20v-8M12 24V8M17 20V12M22 24v-6M27 22V16" />
            </svg>
          </span>
          {file ? (
            <span className="dropzone-text">
              <strong>{file.name}</strong>
              <small>{formatSize(file.size)} · click or drop to replace</small>
            </span>
          ) : (
            <span className="dropzone-text">
              <strong>Choose an audio file or drop it here</strong>
              <small>{ACCEPTED_EXTENSIONS.join("  ")} · up to 25 MB · at least 3 seconds</small>
            </span>
          )}
        </label>

        <button className="primary" type="button" onClick={analyze} disabled={!file || loading}>
          {loading ? (
            <>
              <span className="spinner" aria-hidden="true" /> Analyzing…
            </>
          ) : (
            "Predict genre"
          )}
        </button>
      </section>

      <div aria-live="polite">
        {state.status === "error" && (
          <p className="error" role="alert">
            {state.message}
          </p>
        )}
        {state.status === "done" && <Result result={state.result} />}
      </div>

      <footer className="footer">
        CNN V3 with SpecAugment · 83.3% segment accuracy and 90.0% song accuracy on GTZAN&apos;s
        held-out test set. Songs far from GTZAN&apos;s style may be classified less reliably.
      </footer>
    </main>
  );
}

function Result({ result }: { result: Prediction }) {
  const ranked = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <section className="panel result" aria-label="Prediction">
      <p className="eyebrow">Predicted genre</p>
      <h2>{genreLabel(result.genre)}</h2>
      <p className="confidence">
        {percent(result.confidence)} confidence · averaged over {result.segments} segments (
        {result.duration_seconds.toFixed(1)} s analysed)
      </p>
      <ul className="bars">
        {ranked.map(([genre, probability], index) => (
          <li key={genre} className={index === 0 ? "top" : undefined}>
            <span className="bar-label">{genreLabel(genre)}</span>
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${Math.max(probability * 100, 0.5)}%` }} />
            </span>
            <span className="bar-value">{percent(probability)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

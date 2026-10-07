import { useMemo, type CSSProperties } from "react";
import { CLASSES, model, notebook } from "../data";
import { genreColor, genreLabel, int, pct } from "../lib/format";
import { Spectrogram, Waveform } from "../viz/charts";
import { NeedsEval, useFilmData } from "./data";
import { Chapter, at } from "./engine";

function PredictionStage() {
  const { status, sample, examples, sampleIndex, setSampleIndex } = useFilmData();

  const ranked = useMemo(() => {
    if (!sample) return [];
    return CLASSES.map((g, i) => ({ genre: g, p: sample.view.probabilities[i] })).sort((a, b) => b.p - a.p);
  }, [sample]);

  return (
    <div className="pr f-pad">
      <div className="pr-head rv" style={at(0, 0.05, [0.97, 1])}>
        <p className="f-kicker">
          <b>08</b>The prediction
        </p>
        <h2 className="f-big pr-title">
          A song it has
          <br />
          <span className="f-accent">never heard.</span>
        </h2>
      </div>

      {!sample ? (
        <div className="pr-na">
          <NeedsEval status={status} what="The prediction scene uses real held-out test songs from the Colab export." />
        </div>
      ) : (
        <>
          <figure className="pr-card rv" style={at(0.08, 0.16, [0.97, 1])}>
            <Spectrogram data={sample.mel} rows={model.input_shape[0]} cols={model.input_shape[1]} ariaLabel={`Mel spectrogram of ${sample.view.file}`} className="pr-mel" />
            <div className="pr-wave">
              <Waveform min={sample.view.waveform.min} max={sample.view.waveform.max} ariaLabel={`Waveform of ${sample.view.file}`} />
            </div>
            <figcaption>
              <b>{sample.view.file}</b>
              <span className="f-tag">held-out test song · never seen in training</span>
            </figcaption>
          </figure>

          <div className="pr-model rv" style={at(0.22, 0.3, [0.97, 1], { "--dy": "0px" })} aria-hidden="true">
            <i className="pr-arrow" />
            <div className="pr-cnn">
              <b>CNN</b>
              <span>{int(model.parameters.total)} parameters</span>
              <span>
                {notebook.dataset.segments_per_song} slices × {notebook.dataset.segment_seconds} s, probabilities averaged
              </span>
            </div>
            <i className="pr-arrow" />
          </div>

          <div className="pr-out rv" style={at(0.32, 0.4, [0.97, 1])}>
            <ol className="pr-bars" key={sample.view.file} aria-label={`Genre probabilities for ${sample.view.file}`}>
              {ranked.map((r, i) => (
                <li key={r.genre} data-top={i === 0} style={{ "--w": r.p, "--c": genreColor(r.genre) } as CSSProperties}>
                  <span>{genreLabel(r.genre)}</span>
                  <i />
                  <em>{pct(r.p, 1)}</em>
                </li>
              ))}
            </ol>
            <div className="pr-verdict rv" style={at(0.62, 0.7, [0.97, 1], { "--dy": "10px" })}>
              <p className="pr-word" style={{ color: genreColor(sample.view.predicted) }}>
                {genreLabel(sample.view.predicted)}
              </p>
              <p className="f-tag">
                {pct(sample.view.confidence, 1)} sure · true label {genreLabel(sample.view.actual)} {sample.view.predicted === sample.view.actual ? "✓" : "✗"}
              </p>
            </div>
          </div>

          <div className="pr-pick rv" style={at(0.74, 0.82, [0.97, 1])}>
            <p className="f-note">The most confident correct test song of each genre. Pick one and watch the same network answer; its mistakes come later.</p>
            <div className="pr-chips" role="group" aria-label="Choose a test song">
              {examples.map((e, i) => (
                <button key={e.view.file} type="button" aria-pressed={i === sampleIndex} onClick={() => setSampleIndex(i)} style={{ "--c": genreColor(e.view.actual) } as CSSProperties}>
                  {genreLabel(e.view.actual)}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default function PredictionScene() {
  return (
    <Chapter id="prediction" label="The prediction" vh={340} mobileVh={320}>
      <PredictionStage />
    </Chapter>
  );
}

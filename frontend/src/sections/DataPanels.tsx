import { useMemo, useState } from "react";
import { CLASSES, notebook } from "../data";
import type { EvalData } from "../data/types";
import { histogram } from "../lib/analysis";
import { fixed, genreColor, genreLabel, int, pct } from "../lib/format";
import { Columns, Heatmap, HBars, Scatter } from "../viz/charts";
import { Counter, EvalGate, Source, Stat } from "../viz/ui";

/* --------------------------------------------------------------- dataset overview */
export function DatasetOverview() {
  const d = notebook.dataset;
  const counts = CLASSES.map((c) => d.songs_per_class[c]);
  const min = Math.min(...counts);
  const max = Math.max(...counts);
  const balanced = min === max;
  const totalSegments = d.segments.train + d.segments.val + d.segments.test;
  return (
    <div className="panel-block">
      <dl className="stats">
        <Stat label="Songs">
          <Counter value={d.total_songs} format={(v) => int(Math.round(v))} />
        </Stat>
        <Stat label="Genres">
          <Counter value={CLASSES.length} />
        </Stat>
        <Stat label="3-second segments" hint={`${d.segments_per_song} per 30 s song`}>
          <Counter value={totalSegments} format={(v) => int(Math.round(v))} />
        </Stat>
        <Stat label="Sample rate" hint="audio resampled on load">
          <Counter value={d.sample_rate} format={(v) => `${int(Math.round(v))} Hz`} />
        </Stat>
        <Stat label="Clip length" hint={`${d.example_clip.file}: ${int(d.example_clip.samples)} samples`}>
          <Counter value={d.example_clip.seconds} format={(v) => `${v.toFixed(1)} s`} />
        </Stat>
        <Stat label="Model input" hint="mel bands × frames × channel">
          {d.segment_shape.join(" × ")}
        </Stat>
      </dl>

      <div className="two-up">
        <div>
          <h4>Songs per genre</h4>
          <HBars
            ariaLabel="Number of songs per genre"
            max={max}
            format={int}
            items={CLASSES.map((c) => ({ label: genreLabel(c), value: d.songs_per_class[c], color: genreColor(c) }))}
          />
          <p className="finding">
            {balanced
              ? `The classes are perfectly balanced: every genre has exactly ${min} songs, so plain accuracy is not distorted by class imbalance.`
              : `Class sizes range from ${min} to ${max} songs.`}
          </p>
        </div>
        <div>
          <h4>Stratified split (by song)</h4>
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th>Split</th>
                  <th>Songs</th>
                  <th>per genre</th>
                  <th>Segments</th>
                </tr>
              </thead>
              <tbody>
                {(["train", "val", "test"] as const).map((s) => (
                  <tr key={s}>
                    <td>{s === "val" ? "validation" : s}</td>
                    <td>{d.songs[s]}</td>
                    <td>{d.songs_per_class_per_split[s]}</td>
                    <td>{int(d.segments[s])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">
            Split by {d.split_unit}.{" "}
            {d.corrupt_files.length > 0 && (
              <>
                Unreadable file skipped during feature extraction: <code>{d.corrupt_files.join(", ")}</code> (in the training split).
              </>
            )}
          </p>
        </div>
      </div>
      <Source>
        notebook outputs (cells 8, 16–22, 36–40); dataset: {d.source}. File format: {d.file_format}.
      </Source>
    </div>
  );
}

/* ------------------------------------------------------------------ histograms */
const FEATURES = {
  duration: { label: "Duration (s)", digits: 2, bins: 20 },
  rms: { label: "RMS loudness", digits: 3, bins: 24 },
  peak: { label: "Peak amplitude", digits: 2, bins: 20 },
  centroid: { label: "Spectral centroid (Hz)", digits: 0, bins: 24 },
  zcr: { label: "Zero-crossing rate", digits: 3, bins: 24 },
} as const;
type FeatureKey = keyof typeof FEATURES;

function HistogramPanel({ data }: { data: EvalData }) {
  const [feature, setFeature] = useState<FeatureKey>("centroid");
  const [genre, setGenre] = useState("all");
  const stats = data.dataset_stats;
  const spec = FEATURES[feature];
  const values = stats[feature];
  const { lo, hi, width } = useMemo(() => histogram(values, spec.bins), [values, spec.bins]);
  const edges = Array.from({ length: spec.bins + 1 }, (_, i) => lo + i * width);
  const series = useMemo(() => {
    const pick = (predicate: (i: number) => boolean) =>
      histogram(values.filter((_, i) => predicate(i)), spec.bins, [lo, hi]).counts;
    if (genre === "all") {
      return [
        { name: "train", color: "#5b9cff", values: pick((i) => stats.split[i] === "train") },
        { name: "validation", color: "#f5a623", values: pick((i) => stats.split[i] === "val") },
        { name: "test", color: "#22c55e", values: pick((i) => stats.split[i] === "test") },
      ];
    }
    const index = data.classes.indexOf(genre);
    return [
      { name: genreLabel(genre), color: genreColor(genre), values: pick((i) => stats.genre[i] === index) },
      { name: "other genres", color: "#475569", values: pick((i) => stats.genre[i] !== index) },
    ];
  }, [genre, values, stats, spec.bins, lo, hi, data.classes]);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return (
    <div className="panel-block">
      <div className="controls">
        <label>
          Feature{" "}
          <select value={feature} onChange={(e) => setFeature(e.target.value as FeatureKey)}>
            {(Object.keys(FEATURES) as FeatureKey[]).map((k) => (
              <option key={k} value={k}>
                {FEATURES[k].label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Colour by{" "}
          <select value={genre} onChange={(e) => setGenre(e.target.value)}>
            <option value="all">split</option>
            {data.classes.map((c) => (
              <option key={c} value={c}>
                {genreLabel(c)} vs rest
              </option>
            ))}
          </select>
        </label>
      </div>
      <Columns
        series={series}
        edges={edges}
        xLabel={spec.label}
        formatEdge={(v) => v.toFixed(spec.digits)}
        ariaLabel={`Histogram of ${spec.label} over ${stats.n} songs`}
        height={200}
      />
      <p className="hint">
        {stats.n} songs analysed; mean {mean.toFixed(spec.digits)}, range {lo.toFixed(spec.digits)}–{hi.toFixed(spec.digits)}.
        {stats.failed.length > 0 && ` ${stats.failed.length} unreadable file(s) excluded: ${stats.failed.map((f) => f.file).join(", ")}.`}
      </p>
      <Source>analytics/eval.json: statistics of the real audio files.</Source>
    </div>
  );
}

export function HistogramSection() {
  return (
    <EvalGate what="Feature histograms need the audio itself, which only the Colab notebook (with the dataset) can read.">
      {(data) => <HistogramPanel data={data} />}
    </EvalGate>
  );
}

/* ------------------------------------------------------------ PCA / t-SNE / UMAP */
const SPACES = [
  { id: "mfcc_pca", label: "MFCC · PCA", kind: "mfcc" },
  { id: "mfcc_tsne", label: "MFCC · t-SNE", kind: "mfcc" },
  { id: "mfcc_umap", label: "MFCC · UMAP", kind: "mfcc" },
  { id: "model_pca", label: "CNN features · PCA", kind: "model" },
  { id: "model_tsne", label: "CNN features · t-SNE", kind: "model" },
] as const;

function EmbeddingPanel({ data }: { data: EvalData }) {
  const available = SPACES.filter((s) => data.embeddings[s.id]);
  const [space, setSpace] = useState<(typeof SPACES)[number]["id"]>(available[0].id);
  const spec = SPACES.find((s) => s.id === space)!;
  const embedding = data.embeddings[space]!;
  const points = useMemo(
    () =>
      embedding.x.map((x, i) => ({
        x,
        y: embedding.y[i],
        group: embedding.genre[i],
        ring: embedding.predicted ? embedding.predicted[i] !== embedding.genre[i] : false,
      })),
    [embedding],
  );
  const variance = embedding.explained_variance;
  const sil = data.embeddings.silhouette;
  return (
    <div className="panel-block">
      <div className="chips" role="group" aria-label="Feature space">
        {available.map((s) => (
          <button key={s.id} type="button" aria-pressed={space === s.id} onClick={() => setSpace(s.id)}>
            {s.label}
          </button>
        ))}
      </div>
      <Scatter
        points={points}
        groups={data.classes.map(genreLabel)}
        colors={data.classes.map(genreColor)}
        xLabel={space.endsWith("pca") ? `PC 1${variance ? ` (${pct(variance[0], 0)} of variance)` : ""}` : "dimension 1"}
        yLabel={space.endsWith("pca") ? `PC 2${variance ? ` (${pct(variance[1], 0)})` : ""}` : "dimension 2"}
        ariaLabel={`${spec.label} projection coloured by genre`}
        describe={(i) =>
          `${genreLabel(data.classes[points[i].group])}${
            embedding.predicted ? ` · predicted ${genreLabel(data.classes[embedding.predicted[i]])}` : ""
          }`
        }
      />
      <p className="hint">
        {spec.kind === "mfcc"
          ? `Each point is one of ${points.length} songs described by ${data.embeddings.mfcc_feature_description}. MFCCs are used here only to explore separability; the CNN itself reads Mel spectrograms.`
          : `Each point is a 3-second test segment, described by the CNN's own 128-d layer before the output. White rings mark segments the model got wrong.`}
      </p>
      <p className="finding">
        Silhouette of genre labels (−1 overlapping … 1 separated): MFCC features {fixed(sil.mfcc_features, 3)}, CNN embedding{" "}
        {fixed(sil.cnn_embedding, 3)}.
      </p>
      <Source>analytics/eval.json: PCA/t-SNE computed in Colab with a fixed random seed.</Source>
    </div>
  );
}

export function EmbeddingSection() {
  return (
    <EvalGate what="PCA / t-SNE need the audio features of the whole dataset and the model's internal activations.">
      {(data) => <EmbeddingPanel data={data} />}
    </EvalGate>
  );
}

/* -------------------------------------------------------------------- heatmaps */
function CorrelationPanel({ data }: { data: EvalData }) {
  const corr = data.mfcc_correlation;
  const means = data.mfcc_class_means;
  const short = corr.labels.map((l) => l.replace("MFCC ", ""));
  const maxMean = Math.max(...means.matrix.flat().map(Math.abs));
  return (
    <div className="panel-block two-up">
      <div>
        <h4>Correlation between MFCC coefficients</h4>
        <Heatmap
          rowLabels={short}
          colLabels={short}
          matrix={corr.matrix}
          palette="diverging"
          max={1}
          format={(v) => v.toFixed(1).replace("0.", ".").replace("-.", "−.")}
          rowTitle="MFCC"
          colTitle="MFCC"
          ariaLabel="Correlation matrix of the 20 MFCC means"
          describe={(r, c) => `${corr.labels[r]} vs ${corr.labels[c]}: correlation ${fixed(corr.matrix[r][c], 2)}`}
        />
      </div>
      <div>
        <h4>Average MFCC profile per genre</h4>
        <Heatmap
          rowLabels={means.classes.map(genreLabel)}
          colLabels={short}
          matrix={means.matrix}
          palette="diverging"
          max={maxMean}
          format={(v) => v.toFixed(1)}
          rowTitle="genre"
          colTitle="MFCC"
          ariaLabel="Mean MFCC per genre, z-scored"
          describe={(r, c) => `${genreLabel(means.classes[r])}, MFCC ${c + 1}: z-score ${fixed(means.matrix[r][c], 2)}`}
        />
        <p className="hint">{means.note}</p>
      </div>
      <Source>analytics/eval.json: computed from the 20 per-song MFCC means of the real audio.</Source>
    </div>
  );
}

export function CorrelationSection() {
  return (
    <EvalGate what="Feature correlation needs the audio features of the whole dataset.">
      {(data) => <CorrelationPanel data={data} />}
    </EvalGate>
  );
}

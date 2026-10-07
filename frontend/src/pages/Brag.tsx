import { useEffect, useMemo, useState, type ReactNode } from "react";
import "../lab.css";
import { CLASSES, model, notebook } from "../data";
import { rankBy, statsFromMatrix } from "../lib/analysis";
import { fixed, genreColor, genreLabel, int, pct } from "../lib/format";
import { Link } from "../router";
import { ArchitectureDiagram } from "../sections/Architecture";
import { CorrelationSection, DatasetOverview, EmbeddingSection, HistogramSection } from "../sections/DataPanels";
import {
  ConfidenceSection,
  ConfusionPanel,
  ErrorsSection,
  MetricCards,
  PerClassPanel,
  PrSection,
  RocExplainer,
  RocSection,
} from "../sections/Evaluation";
import { ClassExplorer, Lessons, Reproduce } from "../sections/Narrative";
import { PipelineStages, SignalSection } from "../sections/Pipeline";
import { TrainingPanel } from "../sections/Training";
import { HBars } from "../viz/charts";
import { Counter, Reveal, SectionHead } from "../viz/ui";

const SCENES = [
  { id: "hook", label: "The result" },
  { id: "problem", label: "The problem" },
  { id: "data", label: "The data" },
  { id: "explore", label: "Understanding the data" },
  { id: "features", label: "Feature extraction" },
  { id: "model", label: "The model" },
  { id: "training", label: "Training" },
  { id: "evaluation", label: "Evaluation" },
  { id: "curves", label: "ROC and PR" },
  { id: "failures", label: "Where it fails" },
  { id: "learned", label: "What we learned" },
  { id: "takeaway", label: "Takeaway" },
  { id: "generate", label: "How it was generated" },
] as const;

function useActiveScene() {
  const [active, setActive] = useState<string>(SCENES[0].id);
  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActive(visible.target.id.replace("scene-", ""));
      },
      { rootMargin: "-30% 0px -55% 0px", threshold: [0, 0.2, 0.5] },
    );
    SCENES.forEach((s) => {
      const node = document.getElementById(`scene-${s.id}`);
      if (node) observer.observe(node);
    });
    return () => observer.disconnect();
  }, []);
  return active;
}

function Scene({
  id,
  title,
  eyebrow,
  lede,
  children,
}: {
  id: (typeof SCENES)[number]["id"];
  title: ReactNode;
  eyebrow?: string;
  lede?: ReactNode;
  children?: ReactNode;
}) {
  const index = String(SCENES.findIndex((s) => s.id === id) + 1).padStart(2, "0");
  return (
    <section className="scene" id={`scene-${id}`} aria-labelledby={id}>
      <Reveal>
        <SectionHead id={id} index={index} eyebrow={eyebrow ?? SCENES.find((s) => s.id === id)?.label} title={title} lede={lede} />
      </Reveal>
      {children}
    </section>
  );
}

/** Most-confused genre pairs, counting both directions of the V3 segment confusion matrix. */
function confusedPairs() {
  const { matrix, labels } = notebook.confusion.v3_segment;
  const pairs: { label: string; value: number; note: string }[] = [];
  for (let i = 0; i < labels.length; i += 1) {
    for (let j = i + 1; j < labels.length; j += 1) {
      pairs.push({
        label: `${genreLabel(labels[i])} ↔ ${genreLabel(labels[j])}`,
        value: matrix[i][j] + matrix[j][i],
        note: `${matrix[i][j]} + ${matrix[j][i]}`,
      });
    }
  }
  return pairs.sort((a, b) => b.value - a.value).slice(0, 6);
}

const TABS = [
  { id: "hist", label: "Histograms", node: <HistogramSection /> },
  { id: "space", label: "PCA · t-SNE", node: <EmbeddingSection /> },
  { id: "corr", label: "Heatmaps", node: <CorrelationSection /> },
  { id: "signal", label: "Waveform · spectrogram · MFCC", node: <SignalSection /> },
] as const;

function ExploreTabs() {
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("hist");
  return (
    <div className="panel-block">
      <div className="chips" role="tablist" aria-label="Data views">
        {TABS.map((t) => (
          <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      <div role="tabpanel">{TABS.find((t) => t.id === tab)?.node}</div>
    </div>
  );
}

export default function Brag() {
  const active = useActiveScene();
  const r = notebook.results.v3;
  const songStats = useMemo(() => statsFromMatrix(notebook.confusion.v3_song.matrix, CLASSES), []);
  const recallOrder = rankBy(songStats, "recall");
  const correctSongs = Math.round(r.song_accuracy * notebook.dataset.songs.test);
  const pairs = useMemo(() => confusedPairs(), []);
  const segStats = notebook.reports.v3_segment;
  const hardest = rankBy(segStats.per_class, "f1").slice(-2).reverse();

  useEffect(() => {
    const hash = window.location.hash.slice(1);
    if (hash) window.setTimeout(() => document.getElementById(hash)?.scrollIntoView(), 50);
  }, []);

  return (
    <main className="lab brag" id="main">
      <nav className="rail" aria-label="Scenes">
        <ol>
          {SCENES.map((s) => (
            <li key={s.id}>
              <a href={`#scene-${s.id}`} aria-current={active === s.id ? "true" : undefined}>
                <span>{s.label}</span>
              </a>
            </li>
          ))}
        </ol>
      </nav>

      <div className="scenes">
        {/* 01 · HOOK ------------------------------------------------------------ */}
        <section className="scene hook" id="scene-hook" aria-labelledby="hook">
          <p className="scene-eyebrow">
            <span className="scene-index">01</span>Music genre classification · GTZAN · CNN V3
          </p>
          <h1 id="hook">
            <span className="mega">
              <Counter value={r.song_accuracy * 100} format={(v) => `${v.toFixed(1)}%`} />
            </span>
            <span className="mega-sub">of unseen songs placed in the right genre</span>
          </h1>
          <p className="hook-lede">
            A convolutional network reads Mel spectrograms of 3-second slices, votes across all the slices of a song, and picks one of{" "}
            {CLASSES.length} genres. {correctSongs} of {notebook.dataset.songs.test} held-out test songs were right, from a model of only{" "}
            {int(model.parameters.total)} parameters.
          </p>
          <dl className="stats hook-stats">
            <div className="stat">
              <dt>per 3-second slice</dt>
              <dd>
                <Counter value={r.segment_accuracy * 100} format={(v) => `${v.toFixed(2)}%`} />
              </dd>
            </div>
            <div className="stat">
              <dt>songs · genres</dt>
              <dd>
                {int(notebook.dataset.total_songs)} · {CLASSES.length}
              </dd>
            </div>
            <div className="stat">
              <dt>model size</dt>
              <dd>{(model.file_size_bytes / 1024 / 1024).toFixed(1)} MB</dd>
            </div>
            <div className="stat">
              <dt>30 s song, dev CPU</dt>
              <dd>{Math.round(model.latency.full_30s_song.median_ms)} ms</dd>
            </div>
          </dl>
          <div className="hook-visual">
            <h2 className="visual-title">Share of each genre&apos;s test songs identified correctly</h2>
            <HBars
              ariaLabel="Song-level recall by genre"
              max={1}
              format={(v) => pct(v, 0)}
              items={recallOrder.map((g) => ({
                label: genreLabel(g),
                value: songStats[g].recall,
                color: genreColor(g),
                note: `${notebook.confusion.v3_song.matrix[CLASSES.indexOf(g)][CLASSES.indexOf(g)]}/${songStats[g].support}`,
              }))}
            />
          </div>
          <p className="cta">
            <Link to="/" className="button primary-link">
              Try it on your own song
            </Link>
            <Link to="/model" className="button">
              Model details
            </Link>
            <a href="#scene-curves" className="button">
              Jump to ROC / PR
            </a>
          </p>
        </section>

        {/* 02 · PROBLEM -------------------------------------------------------- */}
        <Scene
          id="problem"
          title="Genres blur, and a 3-second slice is not a song."
          lede={`Genre is a human label stuck on overlapping sounds. The same network that identifies ${pct(r.song_accuracy, 0)} of whole songs gets ${pct(r.segment_accuracy, 0)} of isolated slices right, because a short slice of one genre often sounds like its neighbour. Even a plain CNN baseline (V1) managed only ${pct(notebook.results.v1.segment_accuracy, 1)} per slice.`}
        >
          <Reveal className="two-up">
            <div>
              <h3 className="visual-title">Most confused genre pairs (test segments, both directions)</h3>
              <HBars
                ariaLabel="Most confused genre pairs"
                format={(v) => String(v)}
                items={pairs.map((p) => ({ label: p.label, value: p.value, note: p.note, color: "#f59e0b" }))}
              />
              <p className="hint">Notes show each direction&apos;s count. Source: CNN V3 segment confusion matrix (notebook cell 155).</p>
            </div>
            <div className="big-pair">
              <p>
                <span className="big-num">{pct(r.segment_accuracy, 1)}</span> per slice
              </p>
              <p className="arrow" aria-hidden="true">
                ↓ average the slices of a song ↓
              </p>
              <p>
                <span className="big-num accent">{pct(r.song_accuracy, 1)}</span> per song
              </p>
            </div>
          </Reveal>
        </Scene>

        {/* 03 · DATA ----------------------------------------------------------- */}
        <Scene
          id="data"
          title={`${int(notebook.dataset.total_songs)} songs, ten genres, perfectly balanced.`}
          lede="GTZAN is a small research benchmark: 100 thirty-second clips per genre. The split is made by song, so no slice of a test song was ever seen in training."
        >
          <DatasetOverview />
        </Scene>

        {/* 04 · EXPLORE -------------------------------------------------------- */}
        <Scene
          id="explore"
          title="What the data looks like."
          lede="Distributions, feature space and the actual signals. Everything here is computed from the real audio by the Colab export; nothing is drawn by hand."
        >
          <ExploreTabs />
        </Scene>

        {/* 05 · FEATURES ------------------------------------------------------- */}
        <Scene
          id="features"
          title="From a waveform to what the network sees."
          lede="Audio becomes a 128 × 130 picture of energy over pitch and time. This is the exact preprocessing in the notebook (and in the API), step by step; click a stage."
        >
          <PipelineStages />
          <Reveal>
            <p className="note">
              <b>About MFCCs:</b> they are used on this page only to explore the data. The CNN is trained on the Mel spectrogram itself, not on MFCCs.
            </p>
          </Reveal>
        </Scene>

        {/* 06 · MODEL ---------------------------------------------------------- */}
        <Scene
          id="model"
          title="Four convolution blocks, one pooling step, one decision."
          lede={`Read from the saved model file: ${model.layers.length} layers, ${int(model.parameters.total)} parameters. Watch a segment shrink from 128 × 130 to 8 × 8 feature maps and collapse into ten probabilities.`}
        >
          <ArchitectureDiagram />
        </Scene>

        {/* 07 · TRAINING ------------------------------------------------------- */}
        <Scene
          id="training"
          title="It learns fast, then starts memorising."
          lede="Recorded epoch by epoch in the notebook. Switch between the three model versions; the markers show which epoch's weights were actually evaluated."
        >
          <TrainingPanel />
        </Scene>

        {/* 08 · EVALUATION ----------------------------------------------------- */}
        <Scene
          id="evaluation"
          title="How good is it, honestly?"
          lede="Held-out test split: 150 songs the network never saw. Song level is the product metric; segment level is the network metric."
        >
          <MetricCards />
          <ConfusionPanel />
          <PerClassPanel />
        </Scene>

        {/* 09 · ROC / PR ------------------------------------------------------- */}
        <Scene
          id="curves"
          title="ROC and precision-recall: ranking quality, not just right/wrong."
          lede="Accuracy depends on one threshold. These curves use the model's probabilities across every threshold, one genre against the rest."
        >
          <Reveal>
            <RocExplainer />
          </Reveal>
          <RocSection />
          <PrSection />
        </Scene>

        {/* 10 · FAILURES ------------------------------------------------------- */}
        <Scene
          id="failures"
          title="Where it fails."
          lede={`${genreLabel(hardest[0])} and ${genreLabel(hardest[1])} are the hardest genres at segment level. Here are the mistakes, how sure the model was, and what the spectrograms show.`}
        >
          <ClassExplorer />
          <ConfidenceSection />
          <ErrorsSection />
        </Scene>

        {/* 11 · LEARNED -------------------------------------------------------- */}
        <Scene id="learned" title="What we learned." lede="Every claim below is computed from the numbers above.">
          <Reveal>
            <Lessons />
          </Reveal>
        </Scene>

        {/* 12 · TAKEAWAY ------------------------------------------------------- */}
        <section className="scene takeaway" id="scene-takeaway" aria-labelledby="takeaway">
          <Reveal>
            <SectionHead id="takeaway" index="12" eyebrow="Takeaway" title="In four lines." />
          </Reveal>
          <div className="four">
            <Reveal as="figure">
              <h3>What we built</h3>
              <p>A {CLASSES.length}-genre classifier, served as a web app: upload a song, get a genre with probabilities.</p>
            </Reveal>
            <Reveal as="figure" delay={90}>
              <h3>How it works</h3>
              <p>3-second slices → 128 × 130 Mel image → 4 conv blocks → softmax → average the slices of the song.</p>
            </Reveal>
            <Reveal as="figure" delay={180}>
              <h3>What it achieved</h3>
              <p>
                {pct(r.song_accuracy, 1)} song accuracy ({correctSongs}/{notebook.dataset.songs.test}), {pct(r.segment_accuracy, 1)} per slice, macro F1{" "}
                {fixed(segStats.macro.f1, 3)}.
              </p>
            </Reveal>
            <Reveal as="figure" delay={270}>
              <h3>What we learned</h3>
              <p>Voting beats slicing; pop and rock stay hard; one split and one run cannot give error bars.</p>
            </Reveal>
          </div>
          <p className="cta">
            <Link to="/model" className="button primary-link">
              Full model details
            </Link>
            <Link to="/" className="button">
              Try the live demo
            </Link>
          </p>
        </section>

        {/* HOW GENERATED ------------------------------------------------------- */}
        <section className="scene" id="scene-generate" aria-labelledby="generate-title">
          <SectionHead
            id="generate-title"
            index="13"
            eyebrow="Reproducibility"
            title="How these numbers were generated."
            lede="Every figure comes from a file that a script produced. Here is how to regenerate them."
          />
          <Reproduce />
        </section>
      </div>
    </main>
  );
}

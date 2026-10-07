import { useEffect, type ReactNode } from "react";
import "../lab.css";
import { Link } from "../router";
import { ArchitectureDiagram, ModelOverview } from "../sections/Architecture";
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
import { ClassExplorer, DecisionList, Reproduce } from "../sections/Narrative";
import { PipelineStages } from "../sections/Pipeline";
import { ComparisonPanel, ConfigTable, TrainingPanel } from "../sections/Training";
import { Reveal, SectionHead } from "../viz/ui";

const SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "architecture", label: "Architecture" },
  { id: "pipeline", label: "Pipeline" },
  { id: "config", label: "Training setup" },
  { id: "training", label: "Training curves" },
  { id: "evaluation", label: "Evaluation" },
  { id: "curves", label: "ROC and PR" },
  { id: "errors", label: "Errors and confidence" },
  { id: "compare", label: "Model comparison" },
  { id: "decisions", label: "Why built this way" },
  { id: "explorer", label: "Class explorer" },
  { id: "generate", label: "Reproducibility" },
] as const;

function Block({ id, title, lede, children }: { id: (typeof SECTIONS)[number]["id"]; title: string; lede?: ReactNode; children: ReactNode }) {
  return (
    <section className="scene compact" id={`m-${id}`} aria-labelledby={`h-${id}`}>
      <Reveal>
        <SectionHead id={`h-${id}`} eyebrow={SECTIONS.find((s) => s.id === id)?.label} title={title} lede={lede} />
      </Reveal>
      {children}
    </section>
  );
}

export default function Model() {
  useEffect(() => {
    const hash = window.location.hash.slice(1);
    if (hash) window.setTimeout(() => document.getElementById(hash)?.scrollIntoView(), 50);
  }, []);

  return (
    <main className="lab model-page" id="main">
      <header className="model-hero">
        <p className="scene-eyebrow">Technical reference</p>
        <h1>CNN V3: the complete model explanation</h1>
        <p className="hook-lede">
          Architecture, training, evaluation and the reasoning behind each choice, all drawn from the notebook, the saved model file and the Colab export. See the{" "}
          <Link to="/brag">case study</Link> for the story, or <Link to="/">try the model live</Link>.
        </p>
        <nav className="subnav" aria-label="Sections">
          {SECTIONS.map((s) => (
            <a key={s.id} href={`#m-${s.id}`}>
              {s.label}
            </a>
          ))}
        </nav>
      </header>

      <div className="scenes">
        <Block id="overview" title="At a glance">
          <ModelOverview />
        </Block>
        <Block id="architecture" title="Architecture, layer by layer" lede="Read directly from the saved .keras file.">
          <ArchitectureDiagram />
        </Block>
        <Block id="pipeline" title="From audio to prediction" lede="Every stage exists in the notebook and the API; cell numbers are the evidence.">
          <PipelineStages initial={null} />
        </Block>
        <Block id="config" title="Training configuration" lede="Only values found in the notebook source.">
          <ConfigTable />
        </Block>
        <Block id="training" title="Training analytics" lede="Learning curves for all three model versions, with convergence and fit read off the recorded numbers.">
          <TrainingPanel />
        </Block>
        <Block id="evaluation" title="Evaluation dashboard" lede="Held-out test split of 150 songs (1,499 segments).">
          <MetricCards />
          <ConfusionPanel />
          <PerClassPanel />
        </Block>
        <Block id="curves" title="ROC and precision-recall curves" lede="One-vs-rest curves from the model's predicted probabilities.">
          <Reveal>
            <RocExplainer />
          </Reveal>
          <RocSection />
          <PrSection />
        </Block>
        <Block id="errors" title="Error analysis and confidence">
          <ConfidenceSection />
          <ErrorsSection />
        </Block>
        <Block id="compare" title="Model comparison: V1 → V2 → V3" lede="The three experiments that exist in the notebook.">
          <ComparisonPanel />
        </Block>
        <Block id="decisions" title="Why we built it this way" lede="Each answer cites the notebook cell that implements it.">
          <DecisionList />
        </Block>
        <Block id="explorer" title="Class explorer" lede="Pick a genre to see its metrics, where its songs go, and (with the export) its curves and test songs.">
          <ClassExplorer />
        </Block>
        <Block id="generate" title="How these numbers were generated">
          <Reproduce />
        </Block>
      </div>
    </main>
  );
}

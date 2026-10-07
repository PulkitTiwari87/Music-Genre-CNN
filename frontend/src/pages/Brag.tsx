import { Suspense, lazy, useEffect, useState } from "react";
import "../film/film.css";
import ConfusionScene from "../film/ConfusionScene";
import { PrScene, RocScene } from "../film/CurvesScene";
import DataScene from "../film/DataScene";
import { FilmDataProvider } from "../film/data";
import EmbeddingScene from "../film/EmbeddingScene";
import EngineeringScene from "../film/EngineeringScene";
import FailuresScene from "../film/FailuresScene";
import FinaleScene from "../film/FinaleScene";
import { Hud } from "../film/Hud";
import Journey from "../film/Journey";
import Opening from "../film/Opening";
import PredictionScene from "../film/PredictionScene";
import Problem from "../film/Problem";
import ResultsScene from "../film/ResultsScene";
import Seeing from "../film/Seeing";
import TrainingScene from "../film/TrainingScene";
import { usePrefersReducedMotion } from "../viz/hooks";

const ReproducePanel = lazy(() => import("../film/Reproduce"));

const CHAPTERS = [
  { id: "open", label: "Opening" },
  { id: "problem", label: "The problem" },
  { id: "data", label: "The data" },
  { id: "seeing", label: "Seeing the music" },
  { id: "model", label: "The model" },
  { id: "training", label: "Training" },
  { id: "learned", label: "What it learned" },
  { id: "prediction", label: "The prediction" },
  { id: "results", label: "The results" },
  { id: "roc", label: "ROC-AUC" },
  { id: "pr", label: "PR-AUC" },
  { id: "confusion", label: "Confusion matrix" },
  { id: "failures", label: "Where it struggles" },
  { id: "engineering", label: "The engineering" },
  { id: "finale", label: "Finale" },
] as const;

/** After the film: the existing reproducibility panel, opened on demand (and by the /brag#generate links). */
function AfterCredits() {
  const [open, setOpen] = useState(() => window.location.hash === "#generate");
  useEffect(() => {
    if (window.location.hash === "#generate") window.setTimeout(() => document.getElementById("reproduce")?.scrollIntoView?.(), 120);
  }, []);
  return (
    <details id="reproduce" className="film-end" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>How these numbers were generated</summary>
      {open && (
        <div className="film-end-body">
          <Suspense fallback={<p className="f-note">Loading…</p>}>
            <ReproducePanel />
          </Suspense>
        </div>
      )}
    </details>
  );
}

export default function Brag() {
  const reduced = usePrefersReducedMotion();
  return (
    <FilmDataProvider>
      <main className="film" id="main" data-reduced={reduced}>
        <Hud chapters={CHAPTERS} />
        <Opening />
        <Problem />
        <DataScene />
        <Seeing />
        <Journey />
        <TrainingScene />
        <EmbeddingScene />
        <PredictionScene />
        <ResultsScene />
        <RocScene />
        <PrScene />
        <ConfusionScene />
        <FailuresScene />
        <EngineeringScene />
        <FinaleScene />
        <AfterCredits />
      </main>
    </FilmDataProvider>
  );
}

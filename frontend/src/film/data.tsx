import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import type { EvalData, SongView } from "../data/types";
import { useEval } from "../data/useEval";
import { decodeBytes } from "../lib/format";

/** One real test song from the Colab export with its images decoded once. */
export interface Sample {
  view: SongView;
  mel: Uint8Array;
  mfcc: Uint8Array;
}

export type FilmStatus = "loading" | "missing" | "invalid" | "synthetic" | "ready";

interface FilmData {
  status: FilmStatus;
  /** The real export, only when it loaded, validated and is not flagged synthetic. */
  data: EvalData | null;
  examples: Sample[];
  errors: Sample[];
  /** Index into `examples` shared by every scene that follows "the sample". */
  sampleIndex: number;
  setSampleIndex: (index: number) => void;
  sample: Sample | null;
}

const FilmContext = createContext<FilmData | null>(null);

const toSample = (view: SongView): Sample => ({ view, mel: decodeBytes(view.mel.data), mfcc: decodeBytes(view.mfcc.data) });

export function FilmDataProvider({ children }: { children: ReactNode }) {
  const state = useEval();
  const data = state.status === "ready" && !state.data.provenance.synthetic ? state.data : null;
  const status: FilmStatus =
    state.status === "ready" ? (state.data.provenance.synthetic ? "synthetic" : "ready") : state.status;

  const examples = useMemo(() => (data ? data.examples.map(toSample) : []), [data]);
  const errors = useMemo(() => (data ? data.errors.map(toSample) : []), [data]);
  // The least confident correct example is the most interesting to read: its runner-up genres are visible.
  const defaultIndex = useMemo(
    () => examples.reduce((best, s, i) => (s.view.confidence < examples[best].view.confidence ? i : best), 0),
    [examples],
  );
  const [picked, setPicked] = useState<number | null>(null);
  const sampleIndex = picked ?? defaultIndex;

  const value = useMemo<FilmData>(
    () => ({ status, data, examples, errors, sampleIndex, setSampleIndex: setPicked, sample: examples[sampleIndex] ?? null }),
    [status, data, examples, errors, sampleIndex],
  );
  return <FilmContext.Provider value={value}>{children}</FilmContext.Provider>;
}

export function useFilmData(): FilmData {
  const value = useContext(FilmContext);
  if (!value) throw new Error("useFilmData must be used inside <FilmDataProvider>");
  return value;
}

/** Honest placeholder for a scene that needs the Colab export (never filled with invented data). */
export function NeedsEval({ status, what }: { status: FilmStatus; what: string }) {
  const why =
    status === "loading"
      ? "Loading the evaluation export…"
      : status === "synthetic"
        ? "The loaded export is flagged synthetic, so it is not used here."
        : "Not available from current experiment: this needs analytics/eval.json from the Colab export.";
  return (
    <div className="f-na" role="note">
      <p className="f-tag">{status === "loading" ? "loading" : "data unavailable"}</p>
      <p>
        {what} {why}
      </p>
    </div>
  );
}

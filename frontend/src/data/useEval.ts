import { useEffect, useState } from "react";
import { EVAL_URL } from "./index";
import type { EvalData } from "./types";

export type EvalState =
  | { status: "loading" }
  | { status: "missing" }
  | { status: "invalid"; reason: string }
  | { status: "ready"; data: EvalData };

let cached: Promise<EvalState> | null = null;

/** Structural check: refuse a file this UI cannot render rather than crash half-way. */
export function validateEval(value: unknown): string | null {
  const data = value as Partial<EvalData> | null;
  if (!data || typeof data !== "object") return "not a JSON object";
  if (data.schema_version !== 1) return `unsupported schema_version ${String(data.schema_version)}`;
  for (const key of ["classes", "segment", "song", "embeddings", "dataset_stats", "examples", "errors"] as const) {
    if (!(key in data)) return `missing "${key}"`;
  }
  if (data.classes?.length !== 10) return "expected 10 classes";
  return null;
}

async function load(): Promise<EvalState> {
  let response: Response;
  try {
    // "no-cache" = always revalidate, so a replaced or removed export is never served stale.
    response = await fetch(EVAL_URL, { cache: "no-cache" });
  } catch {
    return { status: "missing" };
  }
  if (!response.ok) return { status: "missing" };
  let parsed: unknown;
  try {
    parsed = await response.json(); // a dev server answering with index.html is treated as missing
  } catch {
    return { status: "missing" };
  }
  const problem = validateEval(parsed);
  return problem ? { status: "invalid", reason: problem } : { status: "ready", data: parsed as EvalData };
}

/** Load the optional evaluation export once per page load. */
export function useEval(): EvalState {
  const [state, setState] = useState<EvalState>({ status: "loading" });
  useEffect(() => {
    let live = true;
    cached ??= load();
    cached.then((result) => live && setState(result));
    return () => {
      live = false;
    };
  }, []);
  return state;
}

/** Test helper: forget the cached request. */
export function resetEvalCache() {
  cached = null;
}

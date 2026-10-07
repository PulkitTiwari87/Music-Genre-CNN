import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { CLASSES, model, notebook } from "./data";
import { resetEvalCache, validateEval } from "./data/useEval";
import { statsFromMatrix, summarizeFit, topConfusions } from "./lib/analysis";
import { buildNodes } from "./sections/Architecture";
import { MetricCards, RocSection } from "./sections/Evaluation";

const fetchMock = vi.fn();

function respondWith(body: unknown | null) {
  fetchMock.mockImplementation(async () =>
    body === null ? new Response("not found", { status: 404 }) : new Response(JSON.stringify(body), { status: 200 }),
  );
}

beforeEach(() => {
  resetEvalCache();
  vi.stubGlobal("fetch", fetchMock);
  respondWith(null);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
  window.history.pushState({}, "", "/");
});

/** Minimal valid eval.json: only what the ROC panel and metric cards read. */
function evalFixture(synthetic: boolean) {
  const curve = (auc: number) => ({ fpr: [0, 0.2, 1], tpr: [0, 0.8, 1], auc, positives: 15 });
  const pr = (ap: number) => ({ recall: [0, 0.8, 1], precision: [1, 0.9, 0.1], ap, positives: 15, baseline: 0.1 });
  const only = <T,>(a: T, b: T) => Object.fromEntries(CLASSES.map((c, i) => [c, i === 0 ? a : i === 1 ? b : null]));
  const block = {
    n: 150,
    accuracy: 0.9,
    per_class: {},
    macro: { precision: 0.9, recall: 0.9, f1: 0.9 },
    weighted: { precision: 0.9, recall: 0.9, f1: 0.9 },
    confusion: [],
    roc: { per_class: only(curve(0.91), curve(0.95)), micro: curve(0.93), macro: curve(0.93), macro_auc: 0.93 },
    pr: {
      per_class: only(pr(0.8), pr(0.85)),
      micro: { recall: [0, 1], precision: [1, 0.1], ap: 0.8 },
      macro_ap: 0.82,
    },
    confidence: { edges: [0, 1], correct: [1], wrong: [0], mean_correct: 1, mean_wrong: null, reliability: [], ece: 0 },
    high_confidence_errors: [],
  };
  return {
    schema_version: 1,
    provenance: { generator: "test", generated_at: "now", synthetic, source: "unit test", versions: {} },
    split_check: { segment_accuracy: 0.9, expected: null, n_test_segments: 1, n_test_songs: 1 },
    classes: [...CLASSES],
    segment: block,
    song: block,
    dataset_stats: {},
    embeddings: {},
    examples: [],
    errors: [],
  };
}

describe("routing", () => {
  it("shows the live demo at / and links to the two new pages", () => {
    render(<App />);
    expect(screen.getByRole("heading", { level: 1, name: "Music Genre CNN" })).toBeInTheDocument();
    const nav = screen.getByRole("navigation", { name: "Main" });
    expect(within(nav).getByRole("link", { name: "Case study" })).toHaveAttribute("href", "/brag");
    expect(within(nav).getByRole("link", { name: "Model" })).toHaveAttribute("href", "/model");
  });

  it("navigates client-side to the case study and updates the title", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(within(screen.getByRole("navigation", { name: "Main" })).getByRole("link", { name: "Case study" }));
    expect(await screen.findByText("of unseen songs placed in the right genre")).toBeInTheDocument();
    expect(window.location.pathname).toBe("/brag");
    expect(document.title).toContain("Case study");
  });

  it("renders the model page at /model with the real architecture", async () => {
    window.history.pushState({}, "", "/model");
    render(<App />);
    expect(await screen.findByRole("heading", { level: 1, name: /complete model explanation/i })).toBeInTheDocument();
    const table = (await screen.findAllByRole("table")).find((t) => t.classList.contains("layers"))!;
    expect(within(table).getAllByRole("row")).toHaveLength(model.layers.length + 2); // header + footer
    expect(within(table).getByText("423,946")).toBeInTheDocument();
  });
});

describe("case-study numbers come from the notebook, not from the page", () => {
  it("shows the recorded accuracies on /brag", async () => {
    window.history.pushState({}, "", "/brag");
    render(<App />);
    const hero = await screen.findByRole("heading", { level: 1 });
    expect(hero).toHaveTextContent("90.0%"); // song_accuracy 0.9
    expect(await screen.findByText("135 of 150", { exact: false })).toBeInTheDocument();
  });

  it("says Not available instead of inventing ROC curves when eval.json is absent", async () => {
    render(<RocSection />);
    expect(await screen.findByText("Not available from current experiment")).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /ROC curves/i })).not.toBeInTheDocument();
  });
});

describe("eval.json handling", () => {
  it("draws ROC curves and AUC from a valid export, and flags synthetic data", async () => {
    respondWith(evalFixture(true));
    render(<RocSection />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/SYNTHETIC TEST DATA/);
    expect(await screen.findByText("macro ROC-AUC")).toBeInTheDocument();
    expect(screen.getByText("0.930")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /One-vs-rest ROC curves at song level/ })).toBeInTheDocument();
  });

  it("lets the user switch the inspected class", async () => {
    respondWith(evalFixture(false));
    const user = userEvent.setup();
    render(<RocSection />);
    await user.click(await screen.findByRole("button", { name: "Classical" }));
    expect(screen.getByRole("button", { name: "Classical" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/micro AUC/)).toHaveTextContent(/Classical\s*0\.950/);
  });

  it("refuses an unusable file with a reason instead of crashing", async () => {
    respondWith({ schema_version: 99 });
    render(<RocSection />);
    expect(await screen.findByText(/cannot be used/i)).toHaveTextContent("unsupported schema_version 99");
  });

  it("does not use ROC-AUC from a synthetic file in the headline metrics", async () => {
    respondWith(evalFixture(true));
    render(<MetricCards />);
    expect(await screen.findByText(/flagged synthetic/i)).toBeInTheDocument();
    expect(screen.queryByText("0.930")).not.toBeInTheDocument();
  });

  it("always revalidates the export so a replaced or removed file is never served stale", async () => {
    render(<RocSection />);
    await screen.findByText("Not available from current experiment");
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("analytics/eval.json"), { cache: "no-cache" });
  });

  it("validates structure", () => {
    expect(validateEval(null)).toBe("not a JSON object");
    expect(validateEval({ schema_version: 1 })).toMatch(/missing/);
    expect(validateEval(evalFixture(false))).toBeNull();
  });
});

describe("analysis helpers agree with the notebook", () => {
  it("recomputes the notebook's V3 classification report from the transcribed confusion matrix", () => {
    const { matrix, labels } = notebook.confusion.v3_segment;
    const derived = statsFromMatrix(matrix, labels);
    for (const label of labels) {
      const reported = notebook.reports.v3_segment.per_class[label];
      expect(derived[label].support).toBe(reported.support);
      expect(derived[label].precision).toBeCloseTo(reported.precision, 3);
      expect(derived[label].recall).toBeCloseTo(reported.recall, 3);
      expect(derived[label].f1).toBeCloseTo(reported.f1, 3);
    }
  });

  it("finds the largest confusions and orders them", () => {
    const top = topConfusions(notebook.confusion.v3_segment.matrix, CLASSES, 3);
    expect(top[0]).toMatchObject({ from: "pop", to: "rock", count: 15 });
    expect(top.map((t) => t.count)).toEqual([...top.map((t) => t.count)].sort((a, b) => b - a));
  });

  it("reads over-fitting off the recorded V3 curves", () => {
    const fit = summarizeFit(notebook.training.v3);
    expect(fit.gap).toBeGreaterThan(0.1);
    expect(fit.verdict).toMatch(/overfit/);
    expect(fit.minValLoss.epoch).toBe(notebook.training.v3.best_val_loss_epoch);
  });

  it("groups the real layers into input, 4 conv blocks, pooling, dense and output", () => {
    const nodes = buildNodes(model.layers, model.input_shape);
    expect(nodes.map((n) => n.title)).toEqual([
      "Input", "SpecAugment", "Conv block 1", "Conv block 2", "Conv block 3", "Conv block 4",
      "Global average pooling", "Dense 128", "Output",
    ]);
    expect(nodes.reduce((sum, n) => sum + n.params, 0)).toBe(model.parameters.total);
  });
});

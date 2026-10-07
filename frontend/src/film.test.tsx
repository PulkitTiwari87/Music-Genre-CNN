import { render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import realEval from "../public/analytics/eval.json";
import App from "./App";
import { notebook } from "./data";
import { resetEvalCache } from "./data/useEval";

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
  window.history.pushState({}, "", "/brag");
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
  window.history.pushState({}, "", "/");
});

const CHAPTER_IDS = [
  "open",
  "problem",
  "data",
  "seeing",
  "model",
  "training",
  "learned",
  "prediction",
  "results",
  "roc",
  "pr",
  "confusion",
  "failures",
  "engineering",
  "finale",
];

describe("/brag film structure", () => {
  it("opens on the project title, naming only technology that is in the repository", async () => {
    render(<App />);
    expect(await screen.findByRole("heading", { level: 1, name: /music genre\s*classification/i })).toBeInTheDocument();
    const tech = within(screen.getByRole("list", { name: "Technology" }));
    expect(tech.getByText("GTZAN")).toBeInTheDocument();
    expect(tech.getByText("Mel spectrogram")).toBeInTheDocument();
    expect(tech.getByText("CNN")).toBeInTheDocument();
    // The repository has no wavelet features and no mood classifier, so the film must not claim them.
    expect(document.body.textContent).not.toMatch(/wavelet|\bmood\b/i);
  });

  it("has every chapter as a landmark and in the chapter index", async () => {
    render(<App />);
    const index = await screen.findByRole("navigation", { name: "Chapters" });
    const links = within(index).getAllByRole("link");
    expect(links).toHaveLength(CHAPTER_IDS.length);
    CHAPTER_IDS.forEach((id, i) => {
      expect(links[i]).toHaveAttribute("href", `#ch-${id}`);
      expect(document.getElementById(`ch-${id}`)).toBeInTheDocument();
    });
  });

  it("marks the film as reduced-motion when the visitor asks for it", async () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query.includes("reduce"),
      media: query,
      addEventListener() {},
      removeEventListener() {},
    }));
    render(<App />);
    const main = await screen.findByRole("main");
    expect(main).toHaveAttribute("data-reduced", "true");
  });
});

describe("/brag numbers come from the notebook", () => {
  it("states the recorded song accuracy and test-set size", async () => {
    render(<App />);
    expect(await screen.findByText("90.0%")).toBeInTheDocument();
    expect((await screen.findAllByText("135 of 150", { exact: false })).length).toBeGreaterThan(0);
  });

  it("derives the confusion story from the confusion matrix rather than typing it in", async () => {
    render(<App />);
    const matrix = notebook.confusion.v3_song.matrix;
    const errors = matrix.reduce((sum, row, i) => sum + row.reduce((a, b) => a + b, 0) - row[i], 0);
    expect(await screen.findByText(new RegExp(`${errors} misses`))).toBeInTheDocument();
    // pop is the true genre of the most song-level errors in the recorded matrix
    expect(
      await screen.findByText((_, el) => el?.tagName === "P" && /Pop accounts for 5 of them/.test(el.textContent ?? "")),
    ).toBeInTheDocument();
  });
});

describe("/brag without the Colab export", () => {
  it("says so instead of inventing ROC, PR or spectrogram data", async () => {
    render(<App />);
    const notes = await screen.findAllByText(/Not available from current experiment/);
    expect(notes.length).toBeGreaterThan(0);
    expect(screen.queryByRole("img", { name: /ROC-AUC curves/i })).not.toBeInTheDocument();
    expect(screen.getAllByText("n/a").length).toBeGreaterThanOrEqual(2); // ROC-AUC and PR-AUC slides
  });
});

describe("/brag with the real Colab export", () => {
  it("shows the recorded macro ROC-AUC and PR-AUC", async () => {
    respondWith(realEval);
    render(<App />);
    const song = realEval.song;
    expect((await screen.findAllByText(song.roc.macro_auc.toFixed(3))).length).toBeGreaterThan(0);
    expect((await screen.findAllByText(song.pr.macro_ap.toFixed(3))).length).toBeGreaterThan(0);
  });

  it("computes the 'why' evidence from the class-average spectrograms", async () => {
    respondWith(realEval);
    render(<App />);
    expect(await screen.findByText(/most similar of 45/)).toBeInTheDocument();
    expect(await screen.findByText(/similarity of averages does not predict confusion/)).toBeInTheDocument();
  });

  it("offers every misclassified song as a selectable dot", async () => {
    respondWith(realEval);
    render(<App />);
    const group = await screen.findByRole("group", { name: "Misclassified songs by confidence" });
    expect(within(group).getAllByRole("button")).toHaveLength(realEval.errors.length);
  });
});

describe("/brag#generate", () => {
  it("opens the reproducibility panel that the Not-available notes link to", async () => {
    window.history.pushState({}, "", "/brag#generate");
    render(<App />);
    expect(await screen.findByText(/Three generated files feed this page/)).toBeInTheDocument();
  });
});

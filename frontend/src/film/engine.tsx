import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { usePrefersReducedMotion } from "../viz/hooks";

/* ------------------------------------------------------------------ math */
export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
/** Progress of `p` through the window [a, b], clamped to 0..1. */
export const seg = (p: number, a: number, b: number) => clamp01((p - a) / (b - a));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);
export const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

/* ----------------------------------------------------- scroll progress store */
type Listener = (p: number) => void;

export interface ProgressStore {
  get(): number;
  set(p: number): void;
  subscribe(fn: Listener): () => void;
}

function createStore(): ProgressStore {
  let value = 0;
  const listeners = new Set<Listener>();
  return {
    get: () => value,
    set(p) {
      value = p;
      listeners.forEach((fn) => fn(p));
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}

interface Entry {
  id: string;
  root: HTMLElement;
  stage: HTMLElement;
  store: ProgressStore;
  quantize: boolean;
}

const entries = new Set<Entry>();
const activeListeners = new Set<(id: string) => void>();
let activeId = "";
let frame = 0;

function update() {
  frame = 0;
  const probe = window.innerHeight * 0.5;
  let nextActive = activeId;
  entries.forEach((entry) => {
    const rect = entry.root.getBoundingClientRect();
    const span = rect.height - entry.stage.offsetHeight;
    let p = span > 0 ? clamp01(-rect.top / span) : 0;
    if (entry.quantize) p = Math.round(p * 12) / 12; // reduced motion: discrete steps instead of continuous movement
    if (p !== entry.store.get()) {
      entry.root.style.setProperty("--p", p.toFixed(4));
      entry.store.set(p);
    }
    if (rect.top <= probe && rect.bottom > probe) nextActive = entry.id;
  });
  if (nextActive !== activeId) {
    activeId = nextActive;
    activeListeners.forEach((fn) => fn(activeId));
  }
}

function schedule() {
  if (!frame) frame = requestAnimationFrame(update);
}

function register(entry: Entry) {
  if (entries.size === 0) {
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
  }
  entries.add(entry);
  schedule();
  return () => {
    entries.delete(entry);
    if (entries.size === 0) {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    }
  };
}

/** Id of the chapter currently crossing the middle of the viewport. */
export function useActiveChapter(): string {
  const [id, setId] = useState(activeId);
  useEffect(() => {
    activeListeners.add(setId);
    return () => {
      activeListeners.delete(setId);
    };
  }, []);
  return id;
}

/* ---------------------------------------------------------------- chapters */
interface ChapterContextValue {
  store: ProgressStore;
  reduced: boolean;
}

const ChapterContext = createContext<ChapterContextValue | null>(null);

export function useChapter(): ChapterContextValue {
  const value = useContext(ChapterContext);
  if (!value) throw new Error("useChapter must be used inside <Chapter>");
  return value;
}

/**
 * A pinned scene. The outer section is `vh` viewport-heights tall; the inner stage sticks to the screen
 * while the visitor scrolls through it, and scroll position becomes progress 0..1.
 * Progress reaches text and layout through the `--p` CSS variable, and reaches canvases/SVG through
 * `useProgress`, so scrolling never re-renders React.
 */
export function Chapter({
  id,
  label,
  vh = 300,
  mobileVh,
  className = "",
  children,
}: {
  id: string;
  label: string;
  vh?: number;
  mobileVh?: number;
  className?: string;
  children: ReactNode;
}) {
  const root = useRef<HTMLElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const reduced = usePrefersReducedMotion();
  const [store] = useState(createStore);

  useEffect(() => {
    if (!root.current || !stage.current) return;
    return register({ id, root: root.current, stage: stage.current, store, quantize: reduced });
  }, [id, store, reduced]);

  const style = { "--h": vh, "--hm": mobileVh ?? Math.round(vh * 0.8) } as CSSProperties;
  return (
    <section ref={root} id={`ch-${id}`} className={`ch ${className}`.trim()} style={style} aria-label={label}>
      <ChapterContext.Provider value={{ store, reduced }}>
        <div ref={stage} className="stage">
          {children}
        </div>
      </ChapterContext.Provider>
    </section>
  );
}

/** Call `fn(p)` immediately and on every progress change. `fn` must be memoised (useCallback). */
export function useProgress(fn: (p: number) => void) {
  const { store } = useChapter();
  useEffect(() => {
    fn(store.get());
    return store.subscribe(fn);
  }, [store, fn]);
}

/** Index of the last break that progress has reached (-1 before the first). Re-renders only when it changes. */
export function useStep(breaks: readonly number[]): number {
  const { store } = useChapter();
  const index = useCallback((p: number) => breaks.reduce((acc, b, i) => (p >= b ? i : acc), -1), [breaks]);
  const [step, setStep] = useState(() => index(store.get()));
  // `breaks` are module-level constants, so the initial state above is already correct.
  useEffect(() => store.subscribe((p) => setStep(index(p))), [store, index]);
  return step;
}

/** CSS custom properties for the `.rv` class: fade/slide in over [a, b], optionally out over [c, d]. */
export const at = (a: number, b: number, out?: [number, number], extra: Record<string, string | number> = {}): CSSProperties =>
  ({ "--a": a, "--b": b, ...(out ? { "--c": out[0], "--d": out[1] } : {}), ...extra }) as CSSProperties;

/** A number that counts up with scroll between progress `a` and `b`. The final value stays in the DOM for assistive tech. */
export function ScrollNumber({
  value,
  format = (v) => v.toFixed(0),
  from = 0,
  a = 0,
  b = 0.2,
  className,
}: {
  value: number;
  format?: (v: number) => string;
  from?: number;
  a?: number;
  b?: number;
  className?: string;
}) {
  const node = useRef<HTMLSpanElement>(null);
  const draw = useCallback(
    (p: number) => {
      if (node.current) node.current.textContent = format(lerp(from, value, easeOut(seg(p, a, b))));
    },
    [value, format, from, a, b],
  );
  useProgress(draw);
  return (
    <span className={className}>
      <span ref={node} aria-hidden="true">
        {format(from)}
      </span>
      <span className="sr-only">{format(value)}</span>
    </span>
  );
}

/* ------------------------------------------------------------------ canvas */
/**
 * A canvas sized to its container (device-pixel aware) that redraws on scroll progress, on resize and
 * whenever `draw` changes (new data). `draw` must be memoised.
 */
export function ProgressCanvas({
  draw,
  className = "",
  label,
  maxDpr = 2,
}: {
  draw: (ctx: CanvasRenderingContext2D, width: number, height: number, p: number) => void;
  className?: string;
  label?: string;
  maxDpr?: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const size = useRef({ w: 0, h: 0, dpr: 1 });
  const last = useRef(0);
  const paint = useCallback(
    (p: number) => {
      last.current = p;
      const node = canvas.current;
      const ctx = node?.getContext("2d");
      const { w, h, dpr } = size.current;
      if (!node || !ctx || !w || !h) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      draw(ctx, w, h, p);
    },
    [draw],
  );
  useEffect(() => {
    const node = canvas.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const rect = node.getBoundingClientRect();
      const dpr = Math.min(maxDpr, window.devicePixelRatio || 1);
      size.current = { w: rect.width, h: rect.height, dpr };
      node.width = Math.max(1, Math.round(rect.width * dpr));
      node.height = Math.max(1, Math.round(rect.height * dpr));
      paint(last.current);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [paint, maxDpr]);
  useProgress(paint);
  return <canvas ref={canvas} className={`f-canvas ${className}`.trim()} role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true} />;
}

/** Seeded pseudo-random numbers so "scattered" layouts are identical on every visit. */
export function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

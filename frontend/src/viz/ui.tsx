import { useEffect, useState, type CSSProperties, type ReactNode } from "react";
import { NOT_AVAILABLE } from "../data";
import type { EvalData } from "../data/types";
import { useEval } from "../data/useEval";
import { useInView, usePrefersReducedMotion } from "./hooks";

/** Fade/translate in when scrolled into view (instant for reduced-motion users). */
export function Reveal({
  children,
  delay = 0,
  className = "",
  as: Tag = "div",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
  as?: "div" | "section" | "li" | "figure";
}) {
  const [ref, seen] = useInView<HTMLElement>(0.15);
  const style = { "--delay": `${delay}ms` } as CSSProperties;
  return (
    <Tag ref={ref as never} className={`reveal${seen ? " in" : ""} ${className}`.trim()} style={style}>
      {children}
    </Tag>
  );
}

/** Number that counts up the first time it is visible. The final value is always in the DOM for assistive tech. */
export function Counter({
  value,
  format = (v: number) => v.toFixed(0),
  duration = 1400,
}: {
  value: number;
  format?: (value: number) => string;
  duration?: number;
}) {
  const [ref, seen] = useInView<HTMLSpanElement>(0.3);
  const reduced = usePrefersReducedMotion();
  const [animated, setShown] = useState(0);
  const shown = reduced ? value : animated;
  useEffect(() => {
    if (!seen || reduced) return;
    let frame = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      setShown(value * (1 - Math.pow(1 - t, 3)));
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [seen, reduced, value, duration]);
  /* `shown` is derived above, so reduced-motion users never run the animation at all. */
  return (
    <span ref={ref} className="counter">
      <span aria-hidden="true">{format(shown)}</span>
      <span className="sr-only">{format(value)}</span>
    </span>
  );
}

export function Source({ children }: { children: ReactNode }) {
  return <p className="source">Source: {children}</p>;
}

/** Placeholder for anything that needs data the current experiment does not contain. */
export function NotAvailable({ what, children }: { what: string; children?: ReactNode }) {
  return (
    <div className="na" role="note">
      <p className="na-title">{NOT_AVAILABLE}</p>
      <p>{what}</p>
      {children}
      <p className="na-how">
        To generate it, run the export cell described in{" "}
        <a href="/brag#generate">How these numbers were generated</a>.
      </p>
    </div>
  );
}

/** Renders `children(data)` once analytics/eval.json is available; otherwise an honest placeholder. */
export function EvalGate({
  what,
  children,
  banner = true,
}: {
  what: string;
  children: (data: EvalData) => ReactNode;
  banner?: boolean;
}) {
  const state = useEval();
  if (state.status === "loading") return <div className="skeleton" aria-busy="true" aria-label="Loading analytics" />;
  if (state.status === "missing") return <NotAvailable what={what} />;
  if (state.status === "invalid") {
    return (
      <NotAvailable what={what}>
        <p>
          <code>eval.json</code> was found but cannot be used ({state.reason}).
        </p>
      </NotAvailable>
    );
  }
  return (
    <>
      {banner && state.data.provenance.synthetic && (
        <p className="synthetic" role="alert">
          SYNTHETIC TEST DATA: this file was generated from synthetic audio and is not a result of this project.
        </p>
      )}
      {children(state.data)}
    </>
  );
}

export function SectionHead({
  id,
  index,
  eyebrow,
  title,
  lede,
}: {
  id: string;
  index?: string;
  eyebrow?: string;
  title: ReactNode;
  lede?: ReactNode;
}) {
  return (
    <header className="scene-head" id={id}>
      {(index || eyebrow) && (
        <p className="scene-eyebrow">
          {index && <span className="scene-index">{index}</span>}
          {eyebrow}
        </p>
      )}
      <h2>{title}</h2>
      {lede && <p className="scene-lede">{lede}</p>}
    </header>
  );
}

export function Stat({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd>{children}</dd>
      {hint && <p className="stat-hint">{hint}</p>}
    </div>
  );
}

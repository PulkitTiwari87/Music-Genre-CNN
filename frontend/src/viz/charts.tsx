import { useEffect, useId, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { useInView, useWidth } from "./hooks";

/* ----------------------------------------------------------------- scales */
const linear = (d0: number, d1: number, r0: number, r1: number) => (v: number) =>
  r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

export function niceTicks(lo: number, hi: number, target = 5): number[] {
  const span = hi - lo;
  if (span <= 0) return [lo];
  const raw = span / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / magnitude;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * magnitude;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

export function Legend({
  items,
}: {
  items: { name: string; color: string; dashed?: boolean; faint?: boolean }[];
}) {
  return (
    <ul className="legend">
      {items.map((item) => (
        <li key={item.name} className={item.faint ? "faint" : undefined}>
          <i style={{ background: item.dashed ? "transparent" : item.color, borderColor: item.color }} className={item.dashed ? "dashed" : ""} />
          {item.name}
        </li>
      ))}
    </ul>
  );
}

/* -------------------------------------------------------------- line chart */
export interface Series {
  name: string;
  color: string;
  points: [number, number][];
  dashed?: boolean;
  width?: number;
  faint?: boolean;
}

interface LineChartProps {
  series: Series[];
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel: string;
  yLabel: string;
  ariaLabel: string;
  description?: string;
  height?: number;
  diagonal?: boolean;
  markers?: { x: number; label: string }[];
  hover?: boolean;
  xTicks?: number[];
  yTicks?: number[];
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  legend?: boolean;
}

const M = { l: 50, r: 14, t: 12, b: 40 };

export function LineChart({
  series,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  ariaLabel,
  description,
  height = 320,
  diagonal = false,
  markers = [],
  hover = false,
  xTicks,
  yTicks,
  formatX = (v) => String(v),
  formatY = (v) => String(v),
  legend = true,
}: LineChartProps) {
  const [wrap, width] = useWidth<HTMLDivElement>();
  const [viewRef, seen] = useInView<SVGSVGElement>(0.2);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const innerW = Math.max(60, width - M.l - M.r);
  const innerH = height - M.t - M.b;
  const sx = linear(xDomain[0], xDomain[1], 0, innerW);
  const sy = linear(yDomain[0], yDomain[1], innerH, 0);
  const xt = xTicks ?? niceTicks(xDomain[0], xDomain[1], Math.max(3, Math.floor(innerW / 90)));
  const yt = yTicks ?? niceTicks(yDomain[0], yDomain[1], 5);
  const descId = useId();

  const nearest = (s: Series, x: number) =>
    s.points.reduce((best, p) => (Math.abs(p[0] - x) < Math.abs(best[0] - x) ? p : best), s.points[0]);
  const snapped = hoverX === null || !series[0] ? null : nearest(series[0], hoverX)[0];

  function onMove(event: React.PointerEvent<SVGRectElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const t = (event.clientX - box.left) / box.width;
    setHoverX(xDomain[0] + t * (xDomain[1] - xDomain[0]));
  }

  return (
    <figure className="chart" ref={wrap}>
      <svg
        ref={viewRef}
        className={`linechart${seen ? " seen" : ""}`}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={ariaLabel}
        aria-describedby={description ? descId : undefined}
      >
        {description && <desc id={descId}>{description}</desc>}
        <g transform={`translate(${M.l},${M.t})`}>
          {yt.map((v) => (
            <g key={`y${v}`}>
              <line className="grid" x1={0} x2={innerW} y1={sy(v)} y2={sy(v)} />
              <text className="tick" x={-8} y={sy(v)} textAnchor="end" dominantBaseline="middle">
                {formatY(v)}
              </text>
            </g>
          ))}
          {xt.map((v) => (
            <text key={`x${v}`} className="tick" x={sx(v)} y={innerH + 18} textAnchor="middle">
              {formatX(v)}
            </text>
          ))}
          <line className="axis" x1={0} x2={innerW} y1={innerH} y2={innerH} />
          <line className="axis" x1={0} x2={0} y1={0} y2={innerH} />
          <text className="axis-label" x={innerW / 2} y={innerH + 36} textAnchor="middle">
            {xLabel}
          </text>
          <text className="axis-label" transform={`translate(-38,${innerH / 2}) rotate(-90)`} textAnchor="middle">
            {yLabel}
          </text>
          {diagonal && (
            <line className="diagonal" x1={sx(xDomain[0])} y1={sy(yDomain[0])} x2={sx(xDomain[1])} y2={sy(yDomain[1])} />
          )}
          {markers.map((m, index) => (
            <g key={m.label}>
              <line className="marker" x1={sx(m.x)} x2={sx(m.x)} y1={0} y2={innerH} />
              <text
                className="marker-label"
                x={sx(m.x) + (sx(m.x) > innerW * 0.5 ? -4 : 4)}
                y={10 + index * 13}
                textAnchor={sx(m.x) > innerW * 0.5 ? "end" : "start"}
              >
                {m.label}
              </text>
            </g>
          ))}
          {series.map((s) => (
            <path
              key={s.name}
              className={`line${s.dashed ? " dashed" : ""}${s.faint ? " faint" : ""}`}
              d={s.points.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(2)},${sy(y).toFixed(2)}`).join("")}
              stroke={s.color}
              strokeWidth={s.width ?? 2.2}
              fill="none"
              pathLength={1}
            />
          ))}
          {hover && snapped !== null && (
            <g>
              <line className="crosshair" x1={sx(snapped)} x2={sx(snapped)} y1={0} y2={innerH} />
              {series.map((s) => {
                const p = nearest(s, snapped);
                return <circle key={s.name} cx={sx(p[0])} cy={sy(p[1])} r={3.5} fill={s.color} />;
              })}
            </g>
          )}
          {hover && (
            <rect
              x={0}
              y={0}
              width={innerW}
              height={innerH}
              fill="transparent"
              onPointerMove={onMove}
              onPointerLeave={() => setHoverX(null)}
            />
          )}
        </g>
      </svg>
      {hover && snapped !== null && (
        <div
          className="tooltip"
          style={{ left: Math.min(width - 170, Math.max(0, M.l + sx(snapped) + 10)), top: M.t + 4 } as CSSProperties}
        >
          <strong>
            {xLabel} {formatX(snapped)}
          </strong>
          {series.map((s) => (
            <span key={s.name}>
              <i style={{ background: s.color }} />
              {s.name}: {formatY(nearest(s, snapped)[1])}
            </span>
          ))}
        </div>
      )}
      {legend && <Legend items={series.map((s) => ({ name: s.name, color: s.color, dashed: s.dashed, faint: s.faint }))} />}
    </figure>
  );
}

/* ------------------------------------------------------------ column chart */
export function Columns({
  series,
  edges,
  ariaLabel,
  xLabel,
  yLabel = "count",
  height = 170,
  formatEdge = (v) => v.toFixed(2),
}: {
  series: { name: string; color: string; values: number[] }[];
  edges: number[];
  ariaLabel: string;
  xLabel: string;
  yLabel?: string;
  height?: number;
  formatEdge?: (v: number) => string;
}) {
  const [ref, seen] = useInView<HTMLDivElement>(0.2);
  const bins = series[0].values.length;
  const totals = Array.from({ length: bins }, (_, i) => series.reduce((sum, s) => sum + s.values[i], 0));
  const max = Math.max(...totals, 1);
  return (
    <figure className="chart">
      <div ref={ref} className={`columns${seen ? " seen" : ""}`} style={{ height }} role="img" aria-label={ariaLabel}>
        <span className="columns-max">{max}</span>
        {totals.map((total, i) => (
          <div
            key={i}
            className="column"
            title={`${formatEdge(edges[i])} to ${formatEdge(edges[i + 1])}: ${series.map((s) => `${s.name} ${s.values[i]}`).join(", ")}`}
          >
            <div className="column-stack" style={{ height: `${(total / max) * 100}%` }}>
              {series.map((s) => (
                <span key={s.name} style={{ flexGrow: s.values[i], background: s.color }} />
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="columns-axis">
        <span>{formatEdge(edges[0])}</span>
        <span>
          {xLabel} (y: {yLabel})
        </span>
        <span>{formatEdge(edges[edges.length - 1])}</span>
      </div>
      {series.length > 1 && <Legend items={series.map((s) => ({ name: s.name, color: s.color }))} />}
    </figure>
  );
}

/* -------------------------------------------------------------- bar chart */
export interface BarItem {
  label: string;
  value: number;
  color?: string;
  note?: string;
  muted?: boolean;
}

export function HBars({
  items,
  max,
  format,
  ariaLabel,
}: {
  items: BarItem[];
  max?: number;
  format: (v: number) => string;
  ariaLabel: string;
}) {
  const [ref, seen] = useInView<HTMLUListElement>(0.2);
  const top = max ?? Math.max(...items.map((i) => i.value), 1);
  return (
    <ul ref={ref} className={`hbars${seen ? " seen" : ""}`} aria-label={ariaLabel}>
      {items.map((item, i) => (
        <li key={item.label} className={item.muted ? "muted" : undefined} style={{ "--i": i } as CSSProperties}>
          <span className="hbar-label">{item.label}</span>
          <span className="hbar-track">
            <span
              className="hbar-fill"
              style={{ width: `${Math.min(100, (item.value / top) * 100)}%`, background: item.color }}
            />
          </span>
          <span className="hbar-value">
            {format(item.value)}
            {item.note && <small>{item.note}</small>}
          </span>
        </li>
      ))}
    </ul>
  );
}

/* ----------------------------------------------------------------- scatter */
export interface ScatterPoint {
  x: number;
  y: number;
  group: number;
  ring?: boolean;
}

export function Scatter({
  points,
  groups,
  colors,
  xLabel,
  yLabel,
  ariaLabel,
  describe,
  height = 380,
}: {
  points: ScatterPoint[];
  groups: string[];
  colors: string[];
  xLabel: string;
  yLabel: string;
  ariaLabel: string;
  describe: (index: number) => string;
  height?: number;
}) {
  const [wrap, width] = useWidth<HTMLDivElement>();
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [active, setActive] = useState<number | null>(null);
  const bounds = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const pad = (lo: number, hi: number) => [lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05] as const;
    return { x: pad(Math.min(...xs), Math.max(...xs)), y: pad(Math.min(...ys), Math.max(...ys)) };
  }, [points]);
  const innerW = width - 16;
  const innerH = height - 34;
  const sx = linear(bounds.x[0], bounds.x[1], 8, innerW + 8);
  const sy = linear(bounds.y[0], bounds.y[1], innerH, 4);

  function toggle(group: number) {
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(group)) next.delete(group);
      else next.add(group);
      return next;
    });
  }

  function onMove(event: React.PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - box.left;
    const py = event.clientY - box.top;
    let best = -1;
    let bestDistance = 14 * 14;
    points.forEach((p, i) => {
      if (hidden.has(p.group)) return;
      const d = (sx(p.x) - px) ** 2 + (sy(p.y) - py) ** 2;
      if (d < bestDistance) {
        bestDistance = d;
        best = i;
      }
    });
    setActive(best >= 0 ? best : null);
  }

  return (
    <figure className="chart" ref={wrap}>
      <svg
        className="scatter"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={ariaLabel}
        onPointerMove={onMove}
        onPointerLeave={() => setActive(null)}
      >
        <rect className="plot-bg" x={0} y={0} width={width} height={innerH + 4} rx={6} />
        {points.map((p, i) =>
          hidden.has(p.group) ? null : (
            <circle
              key={i}
              cx={sx(p.x)}
              cy={sy(p.y)}
              r={active === i ? 6 : 3.4}
              fill={colors[p.group]}
              fillOpacity={active === null || active === i ? 0.82 : 0.35}
              stroke={p.ring ? "#fff" : "none"}
              strokeWidth={p.ring ? 1.2 : 0}
            />
          ),
        )}
        <text className="axis-label" x={width / 2} y={height - 6} textAnchor="middle">
          {xLabel}
        </text>
        <text className="axis-label" transform={`translate(10,${innerH / 2}) rotate(-90)`} textAnchor="middle">
          {yLabel}
        </text>
      </svg>
      {active !== null && (
        <div
          className="tooltip"
          style={{ left: Math.min(width - 190, Math.max(0, sx(points[active].x) + 12)), top: Math.max(0, sy(points[active].y) - 10) }}
        >
          {describe(active)}
        </div>
      )}
      <ul className="legend interactive">
        {groups.map((name, i) => (
          <li key={name}>
            <button type="button" aria-pressed={!hidden.has(i)} onClick={() => toggle(i)} className={hidden.has(i) ? "off" : ""}>
              <i style={{ background: colors[i] }} />
              {name}
            </button>
          </li>
        ))}
      </ul>
    </figure>
  );
}

/* ----------------------------------------------------------------- heatmap */
const mix = (a: number[], b: number[], t: number) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const LOW = [20, 26, 46];
const HIGH = [91, 156, 255];
const NEG = [239, 68, 68];

export function Heatmap({
  rowLabels,
  colLabels,
  matrix,
  format,
  palette = "sequential",
  max,
  ariaLabel,
  rowTitle,
  colTitle,
  describe,
  selectedRow,
  onRowSelect,
}: {
  rowLabels: string[];
  colLabels: string[];
  matrix: number[][];
  format: (v: number) => string;
  palette?: "sequential" | "diverging";
  max?: number;
  ariaLabel: string;
  rowTitle: string;
  colTitle: string;
  describe: (row: number, col: number) => string;
  selectedRow?: number | null;
  onRowSelect?: (row: number) => void;
}) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const top = max ?? Math.max(...matrix.flat().map(Math.abs), 1e-9);
  const colour = (v: number) => {
    if (palette === "diverging") {
      const t = Math.min(1, Math.abs(v) / top);
      return { background: `rgb(${mix(LOW, v < 0 ? NEG : HIGH, t).join(",")})`, dark: t > 0.6 };
    }
    const t = Math.min(1, Math.max(0, v / top)) ** 0.8;
    return { background: `rgb(${mix(LOW, HIGH, t).join(",")})`, dark: t > 0.55 };
  };
  return (
    <div className="heatmap-wrap">
      <table className="heatmap" aria-label={ariaLabel} style={{ "--cols": colLabels.length } as CSSProperties}>
        <thead>
          <tr>
            <th className="corner" scope="col">
              <span className="sr-only">{rowTitle} / {colTitle}</span>
            </th>
            {colLabels.map((label, c) => (
              <th key={label} scope="col" className={hover?.[1] === c ? "hot" : ""}>
                <span>{label}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, r) => (
            <tr key={rowLabels[r]} className={selectedRow === r ? "selected" : ""}>
              <th scope="row" className={hover?.[0] === r ? "hot" : ""}>
                {onRowSelect ? (
                  <button type="button" onClick={() => onRowSelect(r)} aria-pressed={selectedRow === r}>
                    {rowLabels[r]}
                  </button>
                ) : (
                  rowLabels[r]
                )}
              </th>
              {row.map((value, c) => {
                const shade = colour(value);
                return (
                  <td
                    key={colLabels[c]}
                    className={`${shade.dark ? "dark" : ""}${hover && (hover[0] === r || hover[1] === c) ? " line" : ""}${r === c ? " diag" : ""}`}
                    style={{ background: shade.background }}
                    title={describe(r, c)}
                    aria-label={describe(r, c)}
                    onPointerEnter={() => setHover([r, c])}
                    onPointerLeave={() => setHover(null)}
                  >
                    {Math.abs(value) < 1e-9 ? <span className="zero">·</span> : format(value)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="heatmap-axes">
        <span>rows: {rowTitle}</span>
        <span>columns: {colTitle}</span>
      </p>
      <p className="heatmap-hover" aria-live="polite">
        {hover ? describe(hover[0], hover[1]) : "Hover a cell for details."}
      </p>
    </div>
  );
}

/* ---------------------------------------------------------------- waveform */
export function Waveform({ min, max, ariaLabel }: { min: number[]; max: number[]; ariaLabel: string }) {
  const peak = Math.max(...max, ...min.map(Math.abs), 1e-6);
  const mid = 40;
  const top = max.map((v, i) => `${i ? "L" : "M"}${i},${(mid - (v / peak) * mid).toFixed(1)}`).join("");
  const bottom = [...min]
    .map((v, i) => `L${i},${(mid - (v / peak) * mid).toFixed(1)}`)
    .reverse()
    .join("");
  return (
    <svg className="waveform" viewBox={`0 0 ${max.length} 80`} preserveAspectRatio="none" role="img" aria-label={ariaLabel}>
      <path d={`${top}${bottom}Z`} />
    </svg>
  );
}

/* ------------------------------------------------------------- spectrogram */
const MAGMA: [number, number[]][] = [
  [0, [0, 0, 4]],
  [0.25, [59, 15, 112]],
  [0.5, [140, 41, 129]],
  [0.75, [222, 73, 104]],
  [0.9, [254, 159, 109]],
  [1, [252, 253, 191]],
];
const LUT = Array.from({ length: 256 }, (_, i) => {
  const t = i / 255;
  const hi = MAGMA.findIndex(([stop]) => stop >= t);
  const [t1, c1] = MAGMA[Math.max(1, hi)];
  const [t0, c0] = MAGMA[Math.max(1, hi) - 1];
  return mix(c0, c1, (t - t0) / (t1 - t0 || 1));
});

export function Spectrogram({
  data,
  rows,
  cols,
  ariaLabel,
  className = "",
}: {
  data: Uint8Array;
  rows: number;
  cols: number;
  ariaLabel: string;
  className?: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const node = canvas.current;
    const context = node?.getContext("2d");
    if (!node || !context) return; // jsdom has no canvas
    const image = context.createImageData(cols, rows);
    for (let r = 0; r < rows; r += 1) {
      const outRow = rows - 1 - r; // low frequencies at the bottom
      for (let c = 0; c < cols; c += 1) {
        const [red, green, blue] = LUT[data[r * cols + c]];
        const at = (outRow * cols + c) * 4;
        image.data[at] = red;
        image.data[at + 1] = green;
        image.data[at + 2] = blue;
        image.data[at + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [data, rows, cols]);
  return <canvas ref={canvas} width={cols} height={rows} className={`spectrogram ${className}`} role="img" aria-label={ariaLabel} />;
}

export function Caption({ children }: { children: ReactNode }) {
  return <figcaption className="caption">{children}</figcaption>;
}

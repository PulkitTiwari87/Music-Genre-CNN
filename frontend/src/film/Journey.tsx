import { useCallback, useEffect, useMemo, useRef, type CSSProperties } from "react";
import { CLASSES, model } from "../data";
import { genreColor, genreLabel, int, pct } from "../lib/format";
import { buildNodes } from "../sections/Architecture";
import { Spectrogram, Waveform } from "../viz/charts";
import { useFilmData } from "./data";
import { Chapter, at, clamp01, lerp, seg, useProgress } from "./engine";

type ArchNode = ReturnType<typeof buildNodes>[number];

interface TrackNode {
  key: string;
  kind: "audio" | "input" | "aug" | "conv" | "vec" | "scores" | "genre";
  title: string;
  detail: string;
  shape: string;
  explain: string;
  arch?: ArchNode;
}

const dims = (shape: number[]) => shape.join(" × ");

function buildTrack(): TrackNode[] {
  const arch = buildNodes(model.layers, model.input_shape);
  const out: TrackNode[] = [
    { key: "audio", kind: "audio", title: "Audio", detail: "30 s clip · 22,050 Hz mono", shape: "waveform", explain: "Sound as a long list of numbers." },
  ];
  arch.forEach((node, i) => {
    if (i === 0) {
      out.push({ key: "input", kind: "input", title: "Input", detail: "3 s Mel spectrogram", shape: dims(node.shape), explain: "A 128 × 130 picture: pitch up, time across.", arch: node });
    } else if (node.trainOnly) {
      out.push({ key: node.title, kind: "aug", title: node.title, detail: node.op, shape: dims(node.shape), explain: "Training only: random bands are blanked so the network cannot memorise. Skipped when predicting.", arch: node });
    } else if (node.layers.some((l) => l.type === "Conv2D")) {
      const first = node.layers[0].input_shape;
      out.push({
        key: node.title,
        kind: "conv",
        title: node.title,
        detail: node.op,
        shape: `${first[0]} × ${first[1]} → ${node.shape[0]} × ${node.shape[1]} · ${node.shape[2]} maps`,
        explain: "Filters slide over the picture looking for patterns; pooling halves the map.",
        arch: node,
      });
    } else {
      const explain =
        node.title === "Output"
          ? "Ten raw scores become probabilities (softmax)."
          : node.title.startsWith("Global")
            ? "Each feature map is averaged down to a single number."
            : "A fully connected layer combines those numbers.";
      out.push({ key: node.title, kind: "vec", title: node.title, detail: node.op, shape: dims(node.shape), explain, arch: node });
    }
  });
  out.push({ key: "scores", kind: "scores", title: "Class scores", detail: "average of the song's 10 segments", shape: `${CLASSES.length} probabilities`, explain: "Every genre gets a probability; they sum to 1." });
  out.push({ key: "genre", kind: "genre", title: "Genre", detail: "the largest probability wins", shape: "1 label", explain: "The prediction." });
  return out;
}

/** Visual size of a tensor in em: more filters = more plates, larger map = larger plate. */
function slabs(shape: number[]) {
  const [h, w, c] = shape;
  return {
    width: 1.2 + 4.8 * Math.sqrt(w / 130),
    height: 1.2 + 9 * Math.sqrt(h / 128),
    plates: Math.max(1, Math.min(8, Math.round(Math.log2(Math.max(2, c))))),
  };
}

function NodeVisual({ node, index }: { node: TrackNode; index: number }) {
  const { sample } = useFilmData();
  if (node.kind === "audio") {
    return (
      <div className="jo-vis jo-audio">
        {sample ? <Waveform min={sample.view.waveform.min} max={sample.view.waveform.max} ariaLabel="Waveform of the sample" /> : <div className="jo-skel" />}
      </div>
    );
  }
  if (node.kind === "input") {
    return (
      <div className="jo-vis jo-mel">
        {sample ? <Spectrogram data={sample.mel} rows={model.input_shape[0]} cols={model.input_shape[1]} ariaLabel="The model input: Mel spectrogram" className="jo-img" /> : <div className="jo-skel" />}
      </div>
    );
  }
  if (node.kind === "conv" && node.arch) {
    const s = slabs(node.arch.shape);
    return (
      <div className="jo-vis jo-slabs" style={{ width: `${s.width + s.plates * 0.4}em`, height: `${s.height + s.plates * 0.3}em` }} aria-hidden="true">
        {Array.from({ length: s.plates }, (_, i) => (
          <i key={i} style={{ width: `${s.width}em`, height: `${s.height}em`, left: `${i * 0.4}em`, bottom: `${i * 0.3}em`, opacity: 0.45 + (0.55 * i) / s.plates }} />
        ))}
      </div>
    );
  }
  if (node.kind === "aug") {
    return (
      <div className="jo-vis jo-aug" aria-hidden="true">
        <i style={{ top: "22%" }} />
        <i style={{ top: "58%" }} />
        <b style={{ left: "46%" }} />
      </div>
    );
  }
  if (node.kind === "vec" && node.arch) {
    const n = node.arch.shape[0];
    return <div className="jo-vis jo-vec" style={{ height: `${2 + Math.log2(Math.max(2, n)) * 1.05}em` }} aria-hidden="true" data-n={n} />;
  }
  if (node.kind === "scores") {
    const top = sample ? sample.view.probabilities.indexOf(Math.max(...sample.view.probabilities)) : -1;
    return (
      <ol className="jo-scores" data-index={index} aria-label={sample ? `Class probabilities for ${sample.view.file}` : "Class probabilities"}>
        {CLASSES.map((g, i) => (
          <li key={g} data-top={i === top} style={{ "--w": sample ? sample.view.probabilities[i] : 0, "--c": genreColor(g) } as CSSProperties}>
            <span>{genreLabel(g)}</span>
            <i />
            <em>{sample ? pct(sample.view.probabilities[i], 1) : "n/a"}</em>
          </li>
        ))}
      </ol>
    );
  }
  return <GenreCard />;
}

function GenreCard() {
  const { sample } = useFilmData();
  if (!sample) return <p className="f-copy">Needs the Colab export.</p>;
  const { predicted, actual, confidence } = sample.view;
  return (
    <div className="jo-genre">
      <p className="jo-genre-word" style={{ color: genreColor(predicted) }}>
        {genreLabel(predicted)}
      </p>
      <p className="f-tag">
        {pct(confidence, 1)} · true label {genreLabel(actual)} {predicted === actual ? "✓" : "✗"}
      </p>
    </div>
  );
}

function JourneyStage() {
  const { sample } = useFilmData();
  const track = useMemo(() => buildTrack(), []);
  const stage = useRef<HTMLDivElement>(null);
  const rail = useRef<HTMLDivElement>(null);
  const beam = useRef<HTMLDivElement>(null);
  const nodes = useRef<(HTMLElement | null)[]>([]);
  const links = useRef<(HTMLElement | null)[]>([]);
  const label = useRef<HTMLParagraphElement>(null);
  const explain = useRef<HTMLParagraphElement>(null);
  const geo = useRef({ left: [] as number[], width: [] as number[], stageW: 1, trackW: 1 });

  useEffect(() => {
    const measure = () => {
      geo.current = {
        left: nodes.current.map((n) => n?.offsetLeft ?? 0),
        width: nodes.current.map((n) => n?.offsetWidth ?? 0),
        stageW: stage.current?.offsetWidth ?? 1,
        trackW: rail.current?.scrollWidth ?? 1,
      };
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    if (rail.current) observer.observe(rail.current);
    if (stage.current) observer.observe(stage.current);
    return () => observer.disconnect();
  }, [sample]);

  const move = useCallback(
    (p: number) => {
      const g = geo.current;
      const n = g.left.length;
      if (!n || !rail.current) return;
      const first = g.left[0] + g.width[0] / 2;
      const last = g.left[n - 1] + g.width[n - 1] / 2;
      const cam = seg(p, 0.07, 0.9);
      const beamScreen = g.stageW * lerp(0.36, 0.74, seg(p, 0.74, 0.9)); // end on a frame that shows scores and genre together
      const beamTrack = lerp(first, last, cam);
      rail.current.style.transform = `translate3d(${beamScreen - beamTrack}px,0,0)`;
      if (beam.current) beam.current.style.left = `${beamScreen}px`;
      let current = 0;
      for (let i = 0; i < n; i += 1) {
        const lit = p < 0.07 ? 0 : clamp01((beamTrack - g.left[i] + g.width[i] * 0.1) / (g.width[i] * 0.8 + 1));
        nodes.current[i]?.style.setProperty("--lit", lit.toFixed(3));
        if (lit > 0.35) current = i;
        const link = links.current[i];
        if (link && i < n - 1) {
          const from = g.left[i] + g.width[i];
          const to = g.left[i + 1];
          link.style.setProperty("--lit", clamp01((beamTrack - from) / (to - from + 1)).toFixed(3));
        }
      }
      if (label.current) label.current.textContent = `${String(current + 1).padStart(2, "0")} / ${String(n).padStart(2, "0")} · ${track[current].title}`;
      if (explain.current) explain.current.textContent = track[current].explain;
    },
    [track],
  );
  useProgress(move);

  return (
    <div ref={stage} className="jo">
      <div className="jo-head f-pad">
        <div className="rv" style={at(0, 0.04, [0.97, 1])}>
          <p className="f-kicker">
            <b>05</b>The model
          </p>
          <h2 className="f-big jo-title">
            One song,
            <br />
            <span className="f-accent">end to end.</span>
          </h2>
        </div>
        <div className="jo-now rv" style={at(0.08, 0.14, [0.97, 1])}>
          <p ref={label} className="f-tag jo-label" aria-live="off" />
          <p ref={explain} className="f-copy jo-explain" />
        </div>
      </div>

      <div ref={beam} className="jo-beam rv" style={at(0.05, 0.1, [0.95, 1], { "--dy": "0px" })} aria-hidden="true" />

      <div ref={rail} className="jo-rail">
        {track.map((node, i) => (
          <div key={node.key} className="jo-seg">
            <article
              ref={(el) => {
                nodes.current[i] = el;
              }}
              className={`jo-node jo-k-${node.kind}`}
              aria-label={`${node.title}: ${node.shape}`}
            >
              <NodeVisual node={node} index={i} />
              <h3>{node.title}</h3>
              <p className="jo-detail">{node.detail}</p>
              <p className="jo-shape">{node.shape}</p>
              {node.arch && node.arch.params > 0 && <p className="jo-params">{int(node.arch.params)} parameters</p>}
            </article>
            {i < track.length - 1 && (
              <i
                ref={(el) => {
                  links.current[i] = el;
                }}
                className="jo-link"
                aria-hidden="true"
              />
            )}
          </div>
        ))}
      </div>

      <p className="f-note jo-foot rv" style={at(0.1, 0.16, [0.95, 1])}>
        Shapes, filters and parameter counts are read from the saved model file ({int(model.parameters.total)} parameters in total). Plate count and size are
        schematic.
      </p>
    </div>
  );
}

export default function Journey() {
  return (
    <Chapter id="model" label="The model, end to end" vh={600} mobileVh={520}>
      <JourneyStage />
    </Chapter>
  );
}

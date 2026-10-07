import { useEffect, useMemo, useState } from "react";
import { model, notebook } from "../data";
import type { ModelLayer } from "../data/types";
import { bytesToMB, int } from "../lib/format";
import { HBars } from "../viz/charts";
import { useInView, usePrefersReducedMotion } from "../viz/hooks";
import { Source, Stat } from "../viz/ui";

interface Node {
  title: string;
  op: string;
  shape: number[];
  layers: ModelLayer[];
  params: number;
  trainOnly?: boolean;
}

const dims = (shape: number[]) => shape.join(" × ");

/** Collapse the flat layer list into the tensors a reader follows: input → blocks → output. */
export function buildNodes(layers: ModelLayer[], inputShape: number[]): Node[] {
  const nodes: Node[] = [{ title: "Input", op: "Mel spectrogram segment", shape: inputShape, layers: [], params: 0 }];
  let group: ModelLayer[] = [];
  let convBlocks = 0;
  const flush = () => {
    if (!group.length) return;
    const last = group[group.length - 1];
    const conv = group.find((l) => l.type === "Conv2D");
    const dense = group.find((l) => l.type === "Dense");
    let title = group[0].type;
    let op = group.map((l) => l.type).join(" → ");
    if (conv) {
      convBlocks += 1;
      const pool = group.find((l) => l.type === "MaxPooling2D");
      title = `Conv block ${convBlocks}`;
      op = `Conv ${conv.kernel_size?.join("×")} · ${conv.filters} filters · ${conv.activation} → BatchNorm → MaxPool ${pool?.pool_size?.join("×")}`;
    } else if (group[0].type === "SpecAugment") {
      title = "SpecAugment";
      op = `frequency mask ≤ ${group[0].freq_mask_param}, time mask ≤ ${group[0].time_mask_param} (training only)`;
    } else if (group[0].type === "GlobalAveragePooling2D") {
      title = "Global average pooling";
      op = "average each feature map to one number";
    } else if (dense) {
      const dropout = group.find((l) => l.type === "Dropout");
      title = dense.activation === "softmax" ? "Output" : `Dense ${dense.units}`;
      op = `Dense ${dense.units} · ${dense.activation}${dropout ? ` → Dropout ${dropout.rate}` : ""}`;
    }
    nodes.push({
      title,
      op,
      shape: last.output_shape,
      layers: group,
      params: group.reduce((sum, l) => sum + l.params, 0),
      trainOnly: group[0].type === "SpecAugment",
    });
    group = [];
  };
  layers.forEach((layer, i) => {
    group.push(layer);
    const next = layers[i + 1];
    const endsGroup =
      ["MaxPooling2D", "GlobalAveragePooling2D", "Dropout", "SpecAugment"].includes(layer.type) ||
      (layer.type === "Dense" && next?.type !== "Dropout");
    if (endsGroup) flush();
  });
  flush();
  return nodes;
}

function tensorBox(shape: number[]) {
  if (shape.length === 3) {
    return { h: 22 + (shape[0] / 128) * 96, w: 14 + 7 * Math.log2(shape[2]) };
  }
  return { h: 18 + Math.log2(Math.max(shape[0], 2)) * 6, w: 14 };
}

export function ArchitectureDiagram() {
  const nodes = useMemo(() => buildNodes(model.layers, model.input_shape), []);
  const reduced = usePrefersReducedMotion();
  const [ref, seen] = useInView<HTMLDivElement>(0.3);
  const [step, setStep] = useState(-1);
  const [selected, setSelected] = useState<number | null>(null);
  const [run, setRun] = useState(0);
  // Reduced-motion users see the finished diagram at once; the timer never runs for them.
  const active = reduced ? nodes.length : step;

  useEffect(() => {
    if (!seen || reduced) return;
    const timer = window.setInterval(() => {
      setStep((current) => {
        if (current >= nodes.length) {
          window.clearInterval(timer);
          return current;
        }
        return current + 1;
      });
    }, 420);
    return () => window.clearInterval(timer);
  }, [seen, reduced, run, nodes.length]);

  function replay() {
    setStep(-1);
    setRun((n) => n + 1);
  }

  const rowsOf = selected === null ? new Set<string>() : new Set(nodes[selected].layers.map((l) => l.name));
  const totalParams = model.parameters.total;

  return (
    <div className="panel-block" ref={ref}>
      <div className="controls">
        <button type="button" className="ghost" onClick={replay}>
          ▶ Replay data flow
        </button>
        <span className="hint">Click a stage to highlight its layers in the table below.</span>
      </div>
      <div className="arch-scroll" role="group" aria-label="CNN V3 architecture, left to right">
        <ol className="arch">
          {nodes.map((node, i) => {
            const box = tensorBox(node.shape);
            return (
              <li key={`${node.title}${i}`} className={`${i <= active ? "lit" : ""}${active === i ? " now" : ""}${node.trainOnly ? " train-only" : ""}`}>
                <button type="button" onClick={() => setSelected(selected === i ? null : i)} aria-pressed={selected === i}>
                  <span className="tensor-stage">
                    <span className="tensor" style={{ height: box.h, width: box.w }} />
                  </span>
                  <strong>{node.title}</strong>
                  <code>{dims(node.shape)}</code>
                  <span className="arch-op">{node.op}</span>
                  {node.params > 0 && <small>{int(node.params)} params</small>}
                </button>
              </li>
            );
          })}
        </ol>
      </div>
      <h4>Where the {int(totalParams)} parameters live</h4>
      <HBars
        ariaLabel="Parameters per stage"
        format={(v) => `${int(v)} (${((v / totalParams) * 100).toFixed(1)}%)`}
        items={nodes
          .filter((n) => n.params > 0)
          .map((n) => ({ label: n.title, value: n.params, color: n.title.startsWith("Conv") ? "#5b9cff" : "#f5a623" }))}
      />
      <h4>Every layer</h4>
      <div className="table-scroll">
        <table className="data layers">
          <thead>
            <tr>
              <th>#</th>
              <th>Layer</th>
              <th>Input → output</th>
              <th>Filters / units</th>
              <th>Kernel</th>
              <th>Stride</th>
              <th>Padding</th>
              <th>Activation</th>
              <th>Params</th>
            </tr>
          </thead>
          <tbody>
            {model.layers.map((l, i) => (
              <tr key={l.name} className={rowsOf.has(l.name) ? "hl" : ""}>
                <td>{i + 1}</td>
                <td>{l.type}</td>
                <td>
                  {dims(l.input_shape)} → {dims(l.output_shape)}
                </td>
                <td>{l.filters ?? l.units ?? "–"}</td>
                <td>{l.kernel_size?.join("×") ?? l.pool_size?.join("×") ?? "–"}</td>
                <td>{l.strides?.join("×") ?? (l.pool_size ? l.pool_size.join("×") : "–")}</td>
                <td>{l.padding ?? "–"}</td>
                <td>{l.activation ?? (l.rate !== undefined ? `dropout ${l.rate}` : "–")}</td>
                <td>{int(l.params)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={8}>
                Total (trainable {int(model.parameters.trainable)}, non-trainable {int(model.parameters.non_trainable)})
              </td>
              <td>{int(model.parameters.total)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
      <p className="hint">
        Convolutions use L2 regularisation 1e-4 and 3×3 kernels with &quot;same&quot; padding; max-pooling halves height and width each block. The notebook&apos;s{" "}
        <code>model.summary()</code> reports the same {int(model.parameters.total)} parameters.
      </p>
      <Source>
        layers read from {model.model_file} by analytics/extract_model.py; parameter totals match notebook cell {notebook.models.v3.notebook_cell}.
      </Source>
    </div>
  );
}

export function ModelOverview() {
  const lat = model.latency;
  return (
    <div className="panel-block">
      <dl className="stats">
        <Stat label="Model">{model.name}</Stat>
        <Stat label="Type">Convolutional neural network</Stat>
        <Stat label="Task">{model.task}</Stat>
        <Stat label="Input" hint="mel bands × frames × channel">{dims(model.input_shape)}</Stat>
        <Stat label="Output" hint="softmax over genres">{dims(model.output_shape)} classes</Stat>
        <Stat label="Parameters" hint={`${int(model.parameters.trainable)} trainable · ${int(model.parameters.non_trainable)} not`}>
          {int(model.parameters.total)}
        </Stat>
        <Stat label="File size">{bytesToMB(model.file_size_bytes)}</Stat>
        <Stat label="Framework" hint={`saved on ${model.framework.saved_on.replace("@", " ")}`}>
          Keras {model.framework.saved_with_keras} (TensorFlow)
        </Stat>
        <Stat label="Inference, 1 segment" hint="median of 30 runs">{lat.one_segment.median_ms} ms</Stat>
        <Stat label="Inference, 30 s song" hint="features + 10 segments">{lat.full_30s_song.median_ms} ms</Stat>
      </dl>
      <p className="hint">Latency: {lat.measured_on}</p>
      <Source>{model.model_file} (analytics/extract_model.py); timings measured by that script.</Source>
    </div>
  );
}

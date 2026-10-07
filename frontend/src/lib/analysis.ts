import type { ClassStats, EpochRecord, TrainingRun } from "../data/types";

export interface Confusion {
  from: string;
  to: string;
  count: number;
  /** share of the "from" class that was predicted as "to" */
  rate: number;
}

/** Largest off-diagonal cells of a confusion matrix, in descending order. */
export function topConfusions(matrix: number[][], labels: string[], limit = 5): Confusion[] {
  const out: Confusion[] = [];
  matrix.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    row.forEach((count, j) => {
      if (i !== j && count > 0) out.push({ from: labels[i], to: labels[j], count, rate: count / total });
    });
  });
  return out.sort((a, b) => b.count - a.count || b.rate - a.rate).slice(0, limit);
}

/** Per-class precision/recall/F1/support recomputed from a confusion matrix. */
export function statsFromMatrix(matrix: number[][], labels: string[]): Record<string, ClassStats> {
  const result: Record<string, ClassStats> = {};
  labels.forEach((label, i) => {
    const support = matrix[i].reduce((a, b) => a + b, 0);
    const predicted = matrix.reduce((sum, row) => sum + row[i], 0);
    const tp = matrix[i][i];
    const precision = predicted ? tp / predicted : 0;
    const recall = support ? tp / support : 0;
    result[label] = { precision, recall, f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0, support };
  });
  return result;
}

export function rowNormalize(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return row.map((v) => v / total);
  });
}

export function rankBy<T extends string>(stats: Record<T, ClassStats>, key: keyof ClassStats): T[] {
  return (Object.keys(stats) as T[]).sort((a, b) => stats[b][key] - stats[a][key]);
}

export interface FitSummary {
  epochs: number;
  finalTrainAcc: number;
  finalValAcc: number;
  gap: number;
  minValLoss: EpochRecord;
  finalValLoss: number;
  verdict: string;
}

/** Read convergence / over-fitting off the recorded curves; the verdict only uses recorded numbers. */
export function summarizeFit(run: TrainingRun): FitSummary {
  const last = run.epochs[run.epochs.length - 1];
  const minValLoss = run.epochs.reduce((best, e) => (e.val_loss < best.val_loss ? e : best));
  const gap = last.accuracy - last.val_accuracy;
  const lossRise = last.val_loss - minValLoss.val_loss;
  let verdict: string;
  if (gap > 0.1) {
    verdict = "overfit: training accuracy ends far above validation accuracy";
  } else if (gap > 0.03) {
    verdict = "mild overfit: a small gap between training and validation accuracy";
  } else {
    verdict = "well fitted: training and validation accuracy stay close";
  }
  if (lossRise > 0.02 && minValLoss.epoch < last.epoch) verdict += "; validation loss turned upward after its minimum";
  return {
    epochs: run.epochs_run,
    finalTrainAcc: last.accuracy,
    finalValAcc: last.val_accuracy,
    gap,
    minValLoss,
    finalValLoss: last.val_loss,
    verdict,
  };
}

/** Histogram bin counts over `values` using `bins` equal-width bins. */
export function histogram(values: number[], bins: number, range?: [number, number]) {
  const lo = range ? range[0] : Math.min(...values);
  const hi = range ? range[1] : Math.max(...values);
  const width = (hi - lo) / bins || 1;
  const counts = new Array<number>(bins).fill(0);
  for (const v of values) {
    const index = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / width)));
    counts[index] += 1;
  }
  return { lo, hi, width, counts };
}

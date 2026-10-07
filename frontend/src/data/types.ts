/** Shapes of the generated data files. Producers: analytics/extract_notebook.py,
 *  analytics/extract_model.py (bundled) and analytics/export_analytics.py (public/analytics/eval.json). */

export interface EpochRecord {
  epoch: number;
  accuracy: number;
  loss: number;
  val_accuracy: number;
  val_loss: number;
  learning_rate: number | null;
  seconds: number;
}

export interface TrainingRun {
  notebook_cell: number;
  epochs: EpochRecord[];
  epochs_run: number;
  max_epochs: number;
  best_val_accuracy: number;
  best_val_accuracy_epoch: number | null;
  best_checkpoint_val_accuracy: number | null;
  checkpoint_epochs: number[];
  best_val_loss_epoch: number | null;
  early_stopped_at_epoch: number | null;
  learning_rate_reductions: { epoch: number; to: number }[];
  total_seconds_reported: number;
}

export interface ClassStats {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface Averages {
  precision: number;
  recall: number;
  f1: number;
}

export interface Report {
  notebook_cell: number;
  per_class: Record<string, ClassStats>;
  accuracy: number;
  total_support: number;
  macro: Averages;
  weighted: Averages;
}

export interface ConfusionRecord {
  labels: string[];
  matrix: number[][];
  total: number;
  accuracy: number;
  notebook_cell: number;
  provenance: string;
  validated_against_report: boolean;
}

export interface ConfigItem {
  value: string | number | null;
  evidence: string | { cell: number; pattern: string };
}

export interface NotebookData {
  source_notebook: string;
  environment: { tensorflow_in_colab: string | null; accelerator: string | null; notebook_cells: number };
  dataset: {
    name: string;
    source: string;
    file_format: string;
    total_songs: number;
    classes: string[];
    songs_per_class: Record<string, number>;
    songs: { train: number; val: number; test: number };
    songs_per_class_per_split: { train: number; val: number; test: number };
    segments: { train: number; val: number; test: number };
    segments_per_song: number;
    sample_rate: number;
    example_clip: { file: string; samples: number; seconds: number };
    segment_seconds: number;
    segment_shape: number[];
    corrupt_files: string[];
    split_method: string;
    split_unit: string;
  };
  training: { v1: TrainingRun; v2: TrainingRun; v3: TrainingRun };
  models: Record<
    "v1" | "v2" | "v3",
    { total: number; trainable: number; non_trainable: number; notebook_cell: number; file_mb?: number }
  >;
  results: {
    v1: { segment_accuracy: number };
    v2: { segment_accuracy: number; song_accuracy: number };
    v3: { segment_accuracy: number; song_accuracy: number; test_loss: number };
  };
  reports: { v3_segment: Report; v2_song: Report };
  confusion: Record<"v1_segment" | "v2_segment" | "v2_song" | "v3_segment" | "v3_song", ConfusionRecord>;
  config: {
    shared: Record<string, ConfigItem>;
    v1: Record<string, ConfigItem>;
    v2: Record<string, ConfigItem>;
    v3: Record<string, ConfigItem>;
  };
}

export interface ModelLayer {
  name: string;
  type: string;
  input_shape: number[];
  output_shape: number[];
  params: number;
  trainable_params: number;
  non_trainable_params: number;
  filters?: number;
  kernel_size?: number[];
  strides?: number[];
  padding?: string;
  activation?: string;
  pool_size?: number[];
  units?: number;
  rate?: number;
  l2?: number;
  freq_mask_param?: number;
  time_mask_param?: number;
}

export interface Timing {
  median_ms: number;
  min_ms: number;
  runs: number;
}

export interface ModelData {
  model_file: string;
  file_size_bytes: number;
  name: string;
  type: string;
  task: string;
  input_shape: number[];
  output_shape: number[];
  classes: number;
  parameters: { total: number; trainable: number; non_trainable: number };
  framework: {
    saved_with_keras: string;
    saved_on: string;
    measured_with_tensorflow: string;
    measured_with_keras: string;
  };
  layers: ModelLayer[];
  latency: {
    measured_on: string;
    one_segment: Timing;
    ten_segments: Timing;
    full_30s_song: Timing;
    note: string;
  };
}

/* ------------------------------------------------------------------ eval.json */
export interface RocCurve {
  fpr: number[];
  tpr: number[];
  auc: number;
  positives?: number;
}

export interface PrCurve {
  recall: number[];
  precision: number[];
  ap: number;
  positives?: number;
  baseline?: number;
}

export interface ReliabilityBin {
  lo: number;
  hi: number;
  n: number;
  accuracy: number;
  mean_confidence: number;
}

export interface SongPrediction {
  file: string;
  actual: string;
  predicted: string;
  confidence: number;
  segments: number;
  probabilities: number[];
  top3?: { genre: string; p: number }[];
}

export interface EvalBlock {
  n: number;
  accuracy: number;
  per_class: Record<string, ClassStats>;
  macro: Averages;
  weighted: Averages;
  confusion: number[][];
  roc: {
    per_class: Record<string, RocCurve | null>;
    micro: RocCurve;
    macro: RocCurve;
    macro_auc: number;
  };
  pr: {
    per_class: Record<string, PrCurve | null>;
    micro: { recall: number[]; precision: number[]; ap: number };
    macro_ap: number;
  };
  confidence: {
    edges: number[];
    correct: number[];
    wrong: number[];
    mean_correct: number | null;
    mean_wrong: number | null;
    reliability: ReliabilityBin[];
    ece: number;
  };
  high_confidence_errors: (Partial<SongPrediction> & {
    actual: string;
    predicted: string;
    confidence: number;
    file: string;
    segment_index?: number;
  })[];
  songs?: SongPrediction[];
  n_errors?: number;
}

export interface Embedding {
  x: number[];
  y: number[];
  genre: number[];
  predicted?: number[];
  explained_variance?: number[];
}

export interface ImageBlob {
  shape: number[];
  data: string;
  min?: number;
  max?: number;
}

export interface SongView {
  file: string;
  actual: string;
  predicted: string;
  confidence: number;
  duration: number;
  probabilities: number[];
  top3: { genre: string; p: number }[];
  waveform: { min: number[]; max: number[] };
  mel: ImageBlob;
  mfcc: ImageBlob;
}

export interface EvalData {
  schema_version: number;
  provenance: {
    generator: string;
    generated_at: string;
    synthetic: boolean;
    source: string;
    versions: Record<string, string>;
  };
  split_check: {
    segment_accuracy: number;
    expected: number | null;
    matches?: boolean;
    n_test_segments: number;
    n_test_songs: number;
  };
  classes: string[];
  segment: EvalBlock;
  song: EvalBlock;
  dataset_stats: {
    n: number;
    failed: { file: string; error: string }[];
    genre: number[];
    split: string[];
    duration: number[];
    rms: number[];
    peak: number[];
    centroid: number[];
    zcr: number[];
  };
  embeddings: {
    mfcc_pca: Embedding;
    mfcc_tsne: Embedding;
    mfcc_umap: Embedding | null;
    model_pca: Embedding;
    model_tsne: Embedding;
    silhouette: { mfcc_features: number; cnn_embedding: number; note: string };
    mfcc_feature_description: string;
  };
  mfcc_correlation: { matrix: number[][]; labels: string[] };
  mfcc_class_means: { matrix: number[][]; classes: string[]; note: string };
  examples: SongView[];
  errors: SongView[];
  class_mean_mel: { genre: string; data: string }[];
  mel_shape: [number, number];
  training_history?: Record<string, number[]>;
}

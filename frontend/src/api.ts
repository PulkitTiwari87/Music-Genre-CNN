export interface Prediction {
  genre: string;
  confidence: number;
  probabilities: Record<string, number>;
  segments: number;
  duration_seconds: number;
}

/** Must match MAX_UPLOAD_BYTES in src/api.py. */
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
export const ACCEPTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg"];

/** Empty = same origin (FastAPI serves the UI or Vite proxies /api). Set VITE_API_URL when hosted separately. */
const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "");

export async function predictGenre(file: File, signal?: AbortSignal): Promise<Prediction> {
  const body = new FormData();
  body.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/predict`, { method: "POST", body, signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error("Could not reach the server. Is the API running?", { cause: error });
  }

  if (!response.ok) {
    let detail: unknown;
    try {
      detail = (await response.json()).detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(
      typeof detail === "string" ? detail : `Request failed (HTTP ${response.status}).`,
    );
  }
  return (await response.json()) as Prediction;
}

/** Returns an error message if the file can't be sent, otherwise null. */
export function validateFile(file: File): string | null {
  const name = file.name.toLowerCase();
  if (!ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return `Unsupported file type. Use ${ACCEPTED_EXTENSIONS.join(", ")}.`;
  }
  if (file.size === 0) return "That file is empty.";
  if (file.size > MAX_UPLOAD_BYTES) {
    return `File is too large (max ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB).`;
  }
  return null;
}

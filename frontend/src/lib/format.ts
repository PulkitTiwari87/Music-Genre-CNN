export const GENRE_LABELS: Record<string, string> = { hiphop: "Hip-Hop" };

export const genreLabel = (genre: string) =>
  GENRE_LABELS[genre] ?? genre.charAt(0).toUpperCase() + genre.slice(1);

/** Distinct hues for the ten genres (always paired with a label or legend, never colour alone). */
export const GENRE_COLORS: Record<string, string> = {
  blues: "#5b9cff",
  classical: "#b794ff",
  country: "#f5a623",
  disco: "#ff5fa2",
  hiphop: "#a3e635",
  jazz: "#22d3c5",
  metal: "#94a3b8",
  pop: "#facc15",
  reggae: "#22c55e",
  rock: "#ef4444",
};

export const genreColor = (genre: string) => GENRE_COLORS[genre] ?? "#999999";

export const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

export const int = (value: number) => value.toLocaleString("en-US");

export const fixed = (value: number, digits = 3) => value.toFixed(digits);

export const bytesToMB = (bytes: number) => `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

/** Decode a base64 string of uint8 bytes (as written by analytics/export_analytics.py). */
export function decodeBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}

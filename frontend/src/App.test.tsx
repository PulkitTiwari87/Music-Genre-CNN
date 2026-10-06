import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { MAX_UPLOAD_BYTES } from "./api";

const prediction = {
  genre: "hiphop",
  confidence: 0.8123,
  probabilities: {
    blues: 0.01, classical: 0.0, country: 0.02, disco: 0.03, hiphop: 0.8123,
    jazz: 0.01, metal: 0.05, pop: 0.03, reggae: 0.0277, rock: 0.01,
  },
  segments: 10,
  duration_seconds: 30,
};

const audio = (name = "song.wav", size = 2048) =>
  new File([new Uint8Array(size)], name, { type: "audio/wav" });

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("keeps the predict button disabled until a file is chosen", () => {
    render(<App />);
    expect(screen.getByRole("button", { name: /predict genre/i })).toBeDisabled();
  });

  it("uploads the file and shows the ranked prediction", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify(prediction), { status: 200 }));
    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByLabelText(/audio file/i), audio());
    expect(screen.getByText("song.wav")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /predict genre/i }));

    expect(await screen.findByRole("heading", { name: "Hip-Hop" })).toBeInTheDocument();
    expect(screen.getByText(/81\.2% confidence/)).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(10);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/predict");
    expect(init.method).toBe("POST");
    expect((init.body as FormData).get("file")).toBeInstanceOf(File);
  });

  it("shows the server's error message", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ detail: "Audio is too short." }), { status: 422 }),
    );
    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByLabelText(/audio file/i), audio());
    await user.click(screen.getByRole("button", { name: /predict genre/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Audio is too short.");
  });

  it("shows a friendly message when the server is unreachable", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByLabelText(/audio file/i), audio());
    await user.click(screen.getByRole("button", { name: /predict genre/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/could not reach the server/i);
  });

  it("disables the button and shows progress while the request is in flight", async () => {
    let resolve!: (r: Response) => void;
    fetchMock.mockReturnValue(new Promise<Response>((r) => (resolve = r)));
    const user = userEvent.setup();
    render(<App />);

    await user.upload(screen.getByLabelText(/audio file/i), audio());
    await user.click(screen.getByRole("button", { name: /predict genre/i }));
    expect(screen.getByRole("button", { name: /analyzing/i })).toBeDisabled();

    resolve(new Response(JSON.stringify(prediction), { status: 200 }));
    await waitFor(() => expect(screen.getByRole("heading", { name: "Hip-Hop" })).toBeInTheDocument());
  });

  it("rejects unsupported and oversized files without calling the API", async () => {
    // applyAccept off: the file picker's accept filter doesn't exist for drag-and-drop.
    const user = userEvent.setup({ applyAccept: false });
    render(<App />);
    const input = screen.getByLabelText(/audio file/i);

    await user.upload(input, new File(["x"], "notes.txt", { type: "text/plain" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/unsupported file type/i);

    await user.upload(input, audio("big.wav", MAX_UPLOAD_BYTES + 1));
    expect(await screen.findByRole("alert")).toHaveTextContent(/too large/i);

    expect(screen.getByRole("button", { name: /predict genre/i })).toBeDisabled();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import EmotionPredictor

TARGET_SR = 16000
WINDOW_S = 3.0
HOP_S = 1.5
MIN_CHUNK_S = 1.0

app = FastAPI()
app.mount("/assets", StaticFiles(directory=Path(__file__).parent / "assets"), name="assets")
predictor = None


@app.on_event("startup")
def load_model():
    global predictor
    predictor = EmotionPredictor()


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


def make_chunks(n_samples, sr, window_s=WINDOW_S, hop_s=HOP_S, min_chunk_s=MIN_CHUNK_S):
    """Ventanas solapadas (start, end) en samples que cubren toda la señal.
    Si el audio es más corto que una ventana, devuelve un único chunk = todo el audio.
    Un resto final más corto que min_chunk_s se funde con el chunk anterior en vez de
    quedar como una ventana casi vacía."""
    window, hop, min_len = int(window_s * sr), int(hop_s * sr), int(min_chunk_s * sr)
    if n_samples <= window:
        return [(0, n_samples)]
    spans = []
    start = 0
    while start < n_samples:
        end = min(start + window, n_samples)
        if end - start >= min_len:
            spans.append((start, end))
        elif spans:
            spans[-1] = (spans[-1][0], end)
        start += hop
    return spans


def convert_to_wav(raw_bytes, suffix, out_path):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(raw_bytes)
        tmp_in_path = tmp_in.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", str(TARGET_SR), "-ac", "1", str(out_path)],
            check=True, capture_output=True,
        )
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    raw = await audio.read()
    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        full_wav = tmp / "full.wav"
        convert_to_wav(raw, suffix, full_wav)

        signal, sr = sf.read(full_wav, dtype="float32")
        if signal.ndim > 1:
            signal = signal.mean(axis=1)

        spans = make_chunks(len(signal), sr)
        chunk_paths = []
        for i, (s, e) in enumerate(spans):
            p = tmp / f"chunk_{i}.wav"
            sf.write(p, signal[s:e], sr)
            chunk_paths.append(p)

        all_paths = [str(p) for p in chunk_paths] + [str(full_wav)]
        P, classes = predictor.predict_proba(all_paths)
        P = P / P.sum(axis=1, keepdims=True)  # los pesos del ensemble no suman 1: normalizar para mostrar %
        chunk_probs, final_probs = P[:-1], P[-1]

        chunks_out = [
            {
                "start": round(s / sr, 2),
                "end": round(e / sr, 2),
                "label": classes[int(np.argmax(probs))],
                "probs": {c: float(p) for c, p in zip(classes, probs)},
            }
            for (s, e), probs in zip(spans, chunk_probs)
        ]
        final_out = {
            "label": classes[int(np.argmax(final_probs))],
            "probs": {c: float(p) for c, p in zip(classes, final_probs)},
        }

    return {"chunks": chunks_out, "final": final_out}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

"""Extracción de embeddings SSL (wav2vec 2.0 / WavLM) para transfer learning."""
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel

TARGET_SR = 16000


def _pick_device(device=None):
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_audio(path):
    """Carga un wav como mono float32 a 16 kHz (los RAVDESS están a 48 kHz)."""
    audio, sr = sf.read(path, dtype="float32")     # (samples,) o (samples, channels)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), sr, TARGET_SR
        ).numpy()
    return audio


def extract_wav2vec(metadata: pd.DataFrame,
                    model_name: str = "facebook/wav2vec2-base",
                    device: str = None) -> pd.DataFrame:
    """Embeddings SSL mean-pooled por audio.

    Devuelve un DataFrame (n, hidden_size) indexado por path (mismo orden que la entrada).
    Determinista: el modelo corre en eval() y sin gradientes (no hay dropout activo).
    Sirve para cualquier subconjunto (speech, song o todos).
    """
    device = _pick_device(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    paths = metadata["path"].tolist()
    embeddings = []
    with torch.no_grad():
        for p in paths:
            audio = _load_audio(p)
            inputs = feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            hidden = model(input_values).last_hidden_state    # (1, T, H)
            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (H,)
            embeddings.append(emb)

    arr = np.vstack(embeddings).astype("float32")
    cols = [f"wav2vec_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=pd.Index(paths, name="path"), columns=cols)


def extract_wav2vec_layers(metadata: pd.DataFrame,
                           model_name: str = "facebook/wav2vec2-base",
                           device: str = None):
    """Embeddings mean-pooled de TODAS las capas del modelo SSL.

    Devuelve (emb, paths):
    - emb: np.ndarray float32 de shape (n_audios, n_layers, hidden) con el mean-pool
      temporal de cada capa (output_hidden_states). Para wav2vec2-base: (n, 13, 768).
    - paths: lista de paths en el mismo orden que la metadata de entrada.
    Determinista en eval() + sin gradientes.
    """
    device = _pick_device(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    paths = metadata["path"].tolist()
    out = []
    with torch.no_grad():
        for p in paths:
            audio = _load_audio(p)
            inputs = feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            hidden_states = model(input_values, output_hidden_states=True).hidden_states
            # mean-pool temporal de cada capa -> (n_layers, hidden)
            layer_means = torch.stack([h.mean(dim=1).squeeze(0) for h in hidden_states], dim=0)
            out.append(layer_means.cpu().numpy())

    emb = np.stack(out, axis=0).astype("float32")   # (n, n_layers, hidden)
    return emb, paths

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
                    device: str = None,
                    layer: int = 8) -> pd.DataFrame:
    """Embeddings SSL mean-pooled por audio, de la capa `layer`.

    Devuelve un DataFrame (n, hidden_size) indexado por path (mismo orden que la entrada).
    `layer` (default 8) indexa en output_hidden_states: la capa 8 es la mejor para emoción
    según el barrido (las últimas capas están sesgadas al objetivo de pre-entrenamiento).
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
            hidden = model(input_values, output_hidden_states=True).hidden_states[layer]  # (1, T, H)
            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (H,)
            embeddings.append(emb)

    arr = np.vstack(embeddings).astype("float32")
    cols = [f"wav2vec_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=pd.Index(paths, name="path"), columns=cols)


def layers_to_dataframe(emb, paths, layers) -> pd.DataFrame:
    """Concatena las capas `layers` del array (n, n_layers, hidden) → DataFrame (n, len(layers)*hidden).

    `emb` es la salida de `extract_wav2vec_layers` (mean-pool por capa). Columnas nombradas
    wav2vec_l{capa}_{i} e índice = paths. Permite armar representaciones multi-capa (p.ej. las
    top-3 del barrido) sin re-extraer el modelo.
    """
    sel = emb[:, layers, :].reshape(emb.shape[0], -1)
    cols = [f"wav2vec_l{l}_{i}" for l in layers for i in range(emb.shape[2])]
    return pd.DataFrame(sel.astype("float32"), index=pd.Index(paths, name="path"), columns=cols)


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


def extract_wav2vec_frames(metadata: pd.DataFrame,
                           model_name: str = "facebook/wav2vec2-base",
                           device: str = None,
                           layers=(8,)):
    """Embeddings POR FRAME (sin pooling temporal) de las capas `layers`, concatenadas por frame.

    Devuelve (seqs, paths):
    - seqs: lista de arrays float32 (T_i, hidden*len(layers)), uno por audio. T_i varía con la
      duración (wav2vec2 produce ~50 frames/seg). Es la "película" completa que necesita el modelo
      de secuencia (CRNN/RNN/atención) para aprender su propio pooling temporal. Con varias capas,
      cada frame es la concatenación de esas capas (igual idea que `layers_to_dataframe`, pero
      conservando el eje temporal).
    - paths: lista de paths en el mismo orden que la metadata de entrada.
    Determinista en eval() + sin gradientes.
    """
    device = _pick_device(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    paths = metadata["path"].tolist()
    seqs = []
    with torch.no_grad():
        for p in paths:
            audio = _load_audio(p)
            inputs = feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            hs = model(input_values, output_hidden_states=True).hidden_states
            sel = torch.cat([hs[l].squeeze(0) for l in layers], dim=-1)  # (T, H*len(layers))
            seqs.append(sel.cpu().numpy().astype("float32"))
    return seqs, paths

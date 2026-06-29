"""Data augmentation de audio (pseudo-hablantes) y extracción de features aumentadas.

La augmentation crea copias deformadas de cada audio (velocidad/pitch + ruido) que suenan como
otra persona → pseudo-hablantes. Atacan el sobreajuste a los pocos actores (24). Se aplica SOLO
a train (nunca a test) para no meter leakage.
"""
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from src.features.wav2vec import _load_audio, _pick_device

TARGET_SR = 16000


def speed_perturb(a, factor):
    """Cambia velocidad y pitch resampleando (factor>1 = más rápido y agudo; <1 = lento y grave)."""
    idx = np.arange(0, len(a), factor)
    return np.interp(idx, np.arange(len(a)), a).astype("float32")


def add_noise(a, snr_db=20, rng=None):
    """Agrega ruido gaussiano a una relación señal/ruido `snr_db`."""
    rng = np.random.default_rng() if rng is None else rng
    rms = np.sqrt(np.mean(a ** 2) + 1e-9)
    return (a + rng.normal(0, rms / (10 ** (snr_db / 20)), len(a))).astype("float32")


def extract_augmented(metadata, model_name="microsoft/wavlm-large", layers=(13, 11, 10),
                      device=None, seed=42):
    """Features mean-pooled (capas `layers` concatenadas) de cada audio limpio + 3 aumentados.

    Augmentations: velocidad 0.9, velocidad 1.1, y ruido (20 dB). Devuelve (Xc, Xa, ya, ga):
    - Xc: (n, d) features de los audios limpios.
    - Xa: (3n, d) features de los aumentados.
    - ya, ga: emoción y actor de cada aumentado (alineados con Xa).
    Determinista (seed) y con el mismo orden de augmentations → las features aumentadas de
    distintos backbones quedan alineadas entre sí.
    """
    device = _pick_device(device)
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    rng = np.random.default_rng(seed)

    def emb(a):
        iv = fe(a, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            hs = model(iv, output_hidden_states=True).hidden_states
        return torch.cat([hs[l].mean(1).squeeze(0) for l in layers], dim=-1).cpu().numpy().astype("float32")

    Xc, Xa, ya, ga = [], [], [], []
    for _, row in metadata.iterrows():
        a = _load_audio(row["path"])
        Xc.append(emb(a))
        for aug in (speed_perturb(a, 0.9), speed_perturb(a, 1.1), add_noise(a, 20, rng)):
            Xa.append(emb(aug)); ya.append(row["emotion"]); ga.append(row["actor"])
    return (np.array(Xc, "float32"), np.array(Xa, "float32"), np.array(ya), np.array(ga))

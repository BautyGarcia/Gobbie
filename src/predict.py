"""Modelo final ejecutable: predice la emoción de audios nuevos con el ensemble entrenado.

Pipeline por audio: cargar wav (16 kHz mono) → extraer embeddings de los 3 backbones SSL
(WavLM-large, HuBERT-large, wav2vec2-base; top-3 capas, mean-pool) → CMVN por hablante →
correr los 4 componentes entrenados → promedio ponderado (soft-vote) → emoción.

CMVN: las estadísticas se calculan por hablante sobre VARIOS audios de esa persona. Por eso,
para máxima accuracy hay que pasar `speakers` (un id por audio, ≥2 audios por hablante). Para un
audio suelto / hablante desconocido se omite el CMVN (accuracy menor, ~como el modelo sin CMVN).
"""
import warnings

import numpy as np
import pandas as pd
import torch
import joblib
from transformers import AutoFeatureExtractor, AutoModel, logging as hf_logging

# El orden de las features se preserva; solo difieren los nombres de columna → warning inocuo.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
hf_logging.set_verbosity_error()   # silencia el reporte verboso de carga de modelos

from src import config
from src.features.wav2vec import _load_audio, _pick_device
from src.normalize import cmvn_by_speaker

BACKBONES = {"wavlm": "microsoft/wavlm-large",
             "hubert": "facebook/hubert-large-ll60k",
             "base": "facebook/wav2vec2-base"}
TARGET_SR = 16000


class EmotionPredictor:
    """Carga el ensemble final y predice emoción sobre audios nuevos."""

    def __init__(self, model_path=None, device=None):
        self.m = joblib.load(model_path or (config.PROCESSED_DIR / "final_model.pkl"))
        self.m["mlp_aug"].device = "cpu"        # el MLP se guardó en CPU
        self.device = _pick_device(device)
        self._fe, self._net = {}, {}            # backbones SSL (lazy load)

    def _backbone(self, key):
        if key not in self._net:
            name = BACKBONES[key]
            self._fe[key] = AutoFeatureExtractor.from_pretrained(name)
            self._net[key] = AutoModel.from_pretrained(name).to(self.device).eval()
        return self._fe[key], self._net[key]

    def _embed(self, key, audios, layers):
        """Embeddings mean-pooled de las capas `layers` concatenadas, un vector por audio."""
        fe, net = self._backbone(key)
        out = []
        with torch.no_grad():
            for a in audios:
                iv = fe(a, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(self.device)
                hs = net(iv, output_hidden_states=True).hidden_states
                out.append(torch.cat([hs[l].mean(1).squeeze(0) for l in layers], dim=-1).cpu().numpy())
        return pd.DataFrame(np.array(out, dtype="float32"))

    def predict_proba(self, audio_paths, speakers=None):
        """Devuelve (probabilidades (n, n_clases), clases). Con `speakers` aplica CMVN por hablante."""
        audios = [_load_audio(p) for p in audio_paths]
        L = self.m["layers"]
        Xw = self._embed("wavlm", audios, L["wavlm"])
        Xh = self._embed("hubert", audios, L["hubert"])
        Xb = self._embed("base", audios, L["base"])

        def norm(X):
            if speakers is None:
                return X                        # audio suelto: sin CMVN (degradado)
            return cmvn_by_speaker(X.reset_index(drop=True), np.asarray(speakers))

        w = self.m["weights"]
        P = (w["wavlm"]  * self.m["lr_wavlm"].predict_proba(norm(Xw)) +
             w["hubert"] * self.m["lr_hubert"].predict_proba(norm(Xh)) +
             w["base"]   * self.m["lr_base"].predict_proba(norm(Xb)) +
             w["mlp"]    * self.m["mlp_aug"].predict_proba(norm(pd.DataFrame(np.hstack([Xw.values, Xb.values])))))
        return P, self.m["lr_wavlm"].classes_

    def predict(self, audio_paths, speakers=None):
        """Devuelve la lista de emociones predichas (una por audio)."""
        P, classes = self.predict_proba(audio_paths, speakers)
        return [classes[i] for i in P.argmax(axis=1)]


def predict(audio_paths, speakers=None, model_path=None):
    """Atajo: predice emoción para una lista de audios (carga el modelo cada vez)."""
    return EmotionPredictor(model_path).predict(audio_paths, speakers)


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:]
    if not paths:
        print("uso: python -m src.predict audio1.wav [audio2.wav ...]")
        sys.exit(1)
    for p, emo in zip(paths, EmotionPredictor().predict(paths)):
        print(f"{emo:10s}  {p}")

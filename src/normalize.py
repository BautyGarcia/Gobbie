"""Normalización de features por hablante (CMVN). Agnóstico a la representación."""
import numpy as np
import pandas as pd


def cmvn_by_speaker(X: pd.DataFrame, groups) -> pd.DataFrame:
    """Z-score de cada feature dentro de cada hablante (resta media / divide std del propio actor).

    Remueve el efecto de la identidad del hablante: cada audio queda expresado como cuánto
    se desvía del 'neutral' de su propio actor. No usa labels → válido bajo GroupKFold por
    actor (cada actor está entero en train o en test) y replicable en inferencia con unas
    pocas utterances del hablante nuevo. Preserva shape, columnas e índice de X.
    """
    groups = pd.Series(np.asarray(groups), index=X.index)
    mu = X.groupby(groups).transform("mean")
    sd = X.groupby(groups).transform("std", ddof=0).replace(0, 1.0)  # std=0 → no dividir por cero
    return (X - mu) / sd


def cmvn_frames_by_speaker(seqs, groups):
    """CMVN por hablante para secuencias frame-level (versión de [[cmvn_by_speaker]]).

    `seqs` es una lista de arrays (T_i, feat). Para cada hablante calcula media/std por feature
    sobre TODOS sus frames (de todos sus audios) y z-scorea cada frame. Misma lógica leakage-safe:
    usa solo frames del propio actor, sin labels. Devuelve una lista nueva de arrays float32.
    """
    groups = np.asarray(groups)
    out = [None] * len(seqs)
    for spk in np.unique(groups):
        idx = np.where(groups == spk)[0]
        stacked = np.concatenate([np.asarray(seqs[i], dtype="float32") for i in idx], axis=0)
        mu = stacked.mean(axis=0)
        sd = stacked.std(axis=0)
        sd[sd == 0] = 1.0
        for i in idx:
            out[i] = ((np.asarray(seqs[i], dtype="float32") - mu) / sd).astype("float32")
    return out

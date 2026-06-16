"""Extracción de features eGeMAPSv02 (functionals) con OpenSMILE."""
import opensmile
import pandas as pd

def extract_egemaps(metadata: pd.DataFrame) -> pd.DataFrame:
    """Extrae los 88 functionals eGeMAPSv02 para cada audio de metadata['path'].

    Devuelve un DataFrame indexado por path (mismo orden que la metadata de entrada).
    Sirve para cualquier subconjunto (speech, song o todos).
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    paths = metadata["path"].tolist()
    feats = smile.process_files(paths)
    # process_files indexa por MultiIndex (file, start, end); aplanamos a path
    feats.index = feats.index.get_level_values("file")
    feats.index.name = "path"
    return feats.loc[paths]  # reordenar para alinear con la metadata de entrada

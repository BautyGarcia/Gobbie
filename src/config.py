from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SPEECH_DIR = DATA_DIR / "Audio_Speech_Actors_01-24"
SONG_DIR = DATA_DIR / "Audio_Song_Actors_01-24"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROJECT_ROOT / "splits"
SPLIT_PATH = SPLITS_DIR / "split.json"

RANDOM_SEED = 42

# Mapas del esquema de nombres RAVDESS (campo -> significado)
MODALITY_MAP = {"01": "full-AV", "02": "video-only", "03": "audio-only"}
VOCAL_CHANNEL_MAP = {"01": "speech", "02": "song"}
EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
INTENSITY_MAP = {"01": "normal", "02": "strong"}
STATEMENT_MAP = {"01": "kids", "02": "dogs"}


def actor_gender(actor: int) -> str:
    """Actor impar = male, par = female (convención RAVDESS)."""
    return "male" if actor % 2 == 1 else "female"


# Columnas de metadata (todo lo demás en un dataset combinado son features)
META_COLS = [
    "path", "modality", "vocal_channel", "emotion", "intensity",
    "statement", "repetition", "actor", "gender", "split",
]

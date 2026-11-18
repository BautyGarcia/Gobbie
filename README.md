<p align="center">
  <img src="demo/assets/gobbie/gobbie-happy.png" alt="Gobbie, the demo mascot" width="160">
</p>

<h1 align="center">Gobbie: Speech Emotion Recognition</h1>

<p align="center">
  Detects the emotion in a voice recording from self-supervised speech embeddings.<br>
  <b>0.890 accuracy</b> on RAVDESS with strict speaker-independent cross-validation, and
  <b>0.767</b> on a German corpus it never saw.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Hugging Face" src="https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="Cloud Run" src="https://img.shields.io/badge/Cloud%20Run-4285F4?logo=googlecloud&logoColor=white">
</p>

---

Final project for the Machine Learning course at Universidad de San Andrés. Given a speech
clip, the system predicts one of 8 emotions: `neutral`, `calm`, `happy`, `sad`, `angry`,
`fearful`, `disgust`, `surprised`. It ships with **Gobbie**, an animated mascot web app that
listens to you through the microphone and reacts to your emotion in real time. We built it for
the poster session at AI Fest 2026.

> The written report, poster and notebook are in Spanish. The code and this README are in English.

## Results

All numbers are out-of-fold predictions from **5-fold GroupKFold by actor**, so no speaker
appears in both training and evaluation.

| Step | Accuracy | Macro-F1 |
|---|---|---|
| Baseline: Random Forest on eGeMAPS (expert acoustic features) | 0.465 | 0.449 |
| Logistic regression on wav2vec2-base, layer 8 | 0.732 | 0.727 |
| + per-speaker normalization (CMVN) | 0.803 | 0.797 |
| + top-3 layers + eGeMAPS | 0.827 | 0.821 |
| WavLM-large, top-3 layers + CMVN | 0.864 | 0.863 |
| Ensemble of 3 backbones (WavLM, HuBERT, wav2vec2) | 0.882 | 0.880 |
| **Final weighted ensemble** (+ MLP with pseudo-speaker augmentation) | **0.890** | **0.888** |
| *For comparison:* end-to-end fine-tuning of WavLM-large | 0.788 | 0.782 |
| Zero-shot on **EmoDB** (German, 6 shared classes, no retraining) | 0.767 | 0.787 |

### What we learned

- **Speaker identity is the main confound.** Variation between speakers is as large as
  variation between emotions. Per-speaker CMVN was the single biggest gain (+7 points) and
  also explains most of the recovery on the German corpus.
- **Middle layers beat the last layer.** Emotion peaks in the intermediate transformer layers
  of wav2vec2, WavLM and HuBERT. The final layers specialize in the pre-training objective.
- **More capacity didn't help with ~1,400 clips.** A linear model on frozen embeddings beat an
  MLP, a CRNN over frame-level features, and full end-to-end fine-tuning of WavLM-large. The
  fine-tuning was also about 100× more expensive.
- **Diversity beats individual strength in the ensemble.** The final model mixes logistic
  regressions on three backbones with an MLP trained on augmented "pseudo-speakers" (speed
  perturbation and noise).

## How it works

```
audio (any .wav) ─► mono 16 kHz ─► WavLM-large / HuBERT-large / wav2vec2-base
                                    └─ mean-pool the best 3 transformer layers
                                         └─ CMVN per speaker
                                              └─ weighted soft-voting ensemble
                                                   (3× LogReg + MLP)  ─► emotion
```

## Quickstart

Requires **Python 3.11**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The trained model (`data/processed/final_model.pkl`) isn't versioned. To generate it, run
the *"Entrenamiento sobre todos los datos"* section of `notebooks/main.ipynb`. That requires
[RAVDESS](https://zenodo.org/record/1188976) under `data/`. The first run downloads the three
backbones from Hugging Face (a few hundred MB). A GPU (CUDA or Apple MPS) helps, but
everything runs on CPU.

### Predict from the command line

```bash
python -m src.predict path/to/audio.wav [more.wav ...]
# happy       path/to/audio.wav
```

### Predict from Python

```python
from src.predict import EmotionPredictor

predictor = EmotionPredictor()
predictor.predict(["audio.wav"])                  # -> ['happy']
proba, classes = predictor.predict_proba(["audio.wav"])

# Pass speaker IDs to enable CMVN (noticeably more accurate):
predictor.predict(["a1.wav", "a2.wav"], speakers=["spk1", "spk1"])
```

### Run the Gobbie demo

```bash
python demo/app.py      # http://127.0.0.1:8000
```

The FastAPI backend splits the recording into overlapping 3-second windows and returns the
emotion for each one, so Gobbie can react while you talk. See
[`demo/deploy/`](demo/deploy) for the Docker and Google Cloud Run setup.

## Repository layout

```
├── notebooks/main.ipynb   end-to-end work: EDA, features, models, final model, EmoDB test
├── src/
│   ├── data.py, splits.py          RAVDESS metadata and speaker-independent splits
│   ├── features/                    eGeMAPS (openSMILE), SSL embeddings, augmentation
│   ├── normalize.py                 per-speaker CMVN
│   ├── models/                      MLP, CRNN, end-to-end fine-tuning (sklearn-style API)
│   ├── evaluation.py, metrics.py    grouped cross-validation and metrics
│   ├── domain_shift.py              EmoDB cross-corpus evaluation
│   └── predict.py                   inference class and CLI
├── tests/                 pytest suite
├── demo/                  Gobbie web app (FastAPI and vanilla JS) and Cloud Run deploy
├── informe/               IEEE-style report (LaTeX and PDF, Spanish)
├── poster/                AI Fest 2026 poster
└── splits/split.json      fixed train/val split by actor (seed 42)
```

Run the tests with `pytest`.

## Authors

**Bautista Garcia** and **Tomas Gonzalez**, Universidad de San Andrés.

## Data

- [RAVDESS](https://zenodo.org/record/1188976): Livingstone & Russo (2018), CC BY-NC-SA 4.0.
- [EmoDB](http://emodb.bilderbar.info/): Burkhardt et al. (2005), Berlin Database of
  Emotional Speech.

Neither dataset is redistributed in this repository.

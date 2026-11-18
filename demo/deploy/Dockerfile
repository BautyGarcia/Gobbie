FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch==2.12.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bajar los backbones en build para que el arranque del Space sea rapido
RUN python -c "\
from transformers import AutoFeatureExtractor, AutoModel; \
[(AutoFeatureExtractor.from_pretrained(m), AutoModel.from_pretrained(m)) \
 for m in ['microsoft/wavlm-large', 'facebook/hubert-large-ll60k', 'facebook/wav2vec2-base']]"

COPY --chown=user . .

CMD ["sh", "-c", "uvicorn demo.app:app --host 0.0.0.0 --port ${PORT:-7860}"]

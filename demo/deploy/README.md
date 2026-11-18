# Deploy de la demo (Gobbie) a Cloud Run

URL en producción: https://gobbie-kqws4k6cea-uc.a.run.app
Proyecto GCP: `gobbie-demo-bg` · Servicio: `gobbie` · Región: `us-central1`

## Cómo redesplegar

El build necesita una carpeta con esta estructura (el Dockerfile asume rutas relativas a ella):

```
build/
├── Dockerfile          (este)
├── requirements.txt    (este)
├── demo/               app.py, index.html, assets/
├── src/                __init__, config, predict, normalize,
│                       features/{__init__,wav2vec}, models/{__init__,mlp}
└── data/processed/final_model.pkl
```

Armarla y correr:

```bash
gcloud config set builds/timeout 3600
gcloud run deploy gobbie --source . --project gobbie-demo-bg --region us-central1 \
  --memory 8Gi --cpu 4 --concurrency 4 --timeout 300 --cpu-boost \
  --max-instances 2 --allow-unauthenticated
```

El build tarda ~20-30 min (baja los 3 backbones y los deja en la imagen).

## Notas

- Escala a cero: no cuesta nada sin tráfico; el primer request tras inactividad
  tarda ~1 min (arranque en frío + carga del modelo).
- Las versiones de `requirements.txt` están fijadas a las del venv local para
  que el pickle de sklearn cargue igual — no actualizarlas sin regenerar el pickle.
- El mismo Dockerfile sirve para Hugging Face Spaces (puerto 7860 por defecto),
  pero HF ya no ofrece Docker Spaces gratis.

# Plan B de deploy: contenedor reproducible de la app (sin datos; solo artefactos agregados).
#   docker build -t tpi-eaymr-app .
#   docker run --rm -p 8501:8501 tpi-eaymr-app          -> http://localhost:8501
#   (copiloto opcional) docker run --rm -p 8501:8501 -e ANTHROPIC_API_KEY=... tpi-eaymr-app
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/tpi

# Dependencias mínimas de la app, fijadas (mismas versiones con que se serializó el modelo)
COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

COPY .streamlit/ .streamlit/
COPY app/ app/
COPY models/artefactos_app.json models/rf_app.joblib models/

RUN useradd --create-home --uid 1000 tpi && chown -R tpi /srv/tpi
USER tpi

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8501/_stcore/health').status != 200)"

# Render, Railway y similares inyectan PORT; localmente queda 8501
CMD ["sh", "-c", "streamlit run app/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]

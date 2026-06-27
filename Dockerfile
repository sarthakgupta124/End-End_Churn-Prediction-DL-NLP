FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"


COPY src/       ./src/
COPY setup.py   .
COPY app.py     .

# ── Inference artifacts ────────────────────────────────────────────────────────
# Copy ONLY the 4 files that predict_pipeline.py actually loads at runtime.
# train.csv (152 MB), test.csv (65 MB), and ingested_data.csv (6 MB) are
# training artefacts — they are excluded by .dockerignore and not copied here.
COPY artifacts/model.h5                 ./artifacts/model.h5
COPY artifacts/preprocessor.pkl        ./artifacts/preprocessor.pkl
COPY artifacts/scaler.pkl              ./artifacts/scaler.pkl
COPY artifacts/mini_word_dictionary.pkl ./artifacts/mini_word_dictionary.pkl

# ── Runtime configuration ──────────────────────────────────────────────────────
# PYTHONUNBUFFERED    → print / log lines appear immediately in docker logs
# PYTHONDONTWRITEBYTECODE → no .pyc files written inside the container
# STREAMLIT_* vars    → headless mode (no browser popup) + no telemetry pings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Port ───────────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# 0.0.0.0 is required so the port is reachable from outside the container.
# Do NOT use 127.0.0.1 here — the app will start but be unreachable.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
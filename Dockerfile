FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System-level build/runtime tools required by Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/start.sh

# Railway exposes Streamlit externally; FastAPI remains internal.
EXPOSE 8501 8000

# Persist ChromaDB and dataset artifacts.
VOLUME ["/app/data"]

CMD ["/app/start.sh"]

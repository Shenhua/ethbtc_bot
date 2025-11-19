# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Paris

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user (adjust UID/GID via build args if you want)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app

WORKDIR /app

# Install deps first for better caching
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt

# Entrypoint + app
COPY entrypoint.sh /entrypoint.sh
# Strip CRLF if any, and ensure executable
RUN sed -i 's/\r$//' /entrypoint.sh && chmod 0755 /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY . /app

# Data dir for state/logs persisted via volume
RUN mkdir -p /data && chown -R app:app /app /data

USER app

ENTRYPOINT ["/entrypoint.sh"]
# Optional default command (can be overridden by compose)
CMD ["python","/app/live_executor.py","--params","configs/selected_params_15m_final.json","--symbol","ETHBTC","--mode","testnet"]
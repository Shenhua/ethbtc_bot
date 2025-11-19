# Dockerfile for ETHBTC bot v3
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default ports used by live_executor.py
ENV METRICS_PORT=9109
ENV STATUS_PORT=9110

EXPOSE 9109 9110

# Make sure the script is executable inside /app
RUN chmod +x /app/entrypoint.sh

# Use the script from /app (since WORKDIR is /app)
ENTRYPOINT ["./entrypoint.sh"]

# Safe default: dry mode on testnet-style config.
# You will usually override this with docker-compose `command:`.
CMD ["python", "/app/live_executor.py", "--params", "configs/prod_dynamic_15m_016btc.json", "--symbol", "ETHBTC", "--mode", "dry"]
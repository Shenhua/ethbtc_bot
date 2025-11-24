# Dockerfile for ETHBTC bot v4
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
# MAGIC FIX: Allows scripts in tools/ to import from core/
ENV PYTHONPATH="${PYTHONPATH}:/app"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire structure
COPY . .

ENV METRICS_PORT=9109
ENV STATUS_PORT=9110

EXPOSE 9109 9110

RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

# Command stays the same (live_executor is still in root)
CMD ["python", "/app/live_executor.py", "--params", "configs/prod_dynamic_15m_016btc.json", "--symbol", "ETHBTC", "--mode", "testnet"]
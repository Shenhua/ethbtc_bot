# Dockerfile for ETHBTC bot v5
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy Code
COPY . .

# CLEANUP: Remove any local __pycache__ that might have been copied
RUN find . -type d -name "__pycache__" -exec rm -rf {} +

ENV METRICS_PORT=9109
ENV STATUS_PORT=9110

EXPOSE 9109 9110

RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/usr/bin/tini", "--", "./entrypoint.sh"]

CMD ["python", "live_executor.py", "--params", "configs/prod_meta_live.json", "--symbol", "ETHBTC", "--mode", "testnet"]
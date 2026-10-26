# AI Gateway — self-hosted multi-provider LLM router
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    GATEWAY_DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY web ./web

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health',timeout=4).status==200 else 1)"

CMD ["python3", "app.py"]

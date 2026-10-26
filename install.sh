#!/usr/bin/env bash
# AI Gateway — one-command install (Docker required)
set -e
DIR="${GATEWAY_DIR:-$HOME/ai-gateway}"
PORT="${PORT:-7860}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install it first: https://docs.docker.com/engine/install/" >&2
  exit 1
fi

mkdir -p "$DIR" && cd "$DIR"

if [ ! -f app.py ]; then
  echo "==> fetching AI Gateway"
  curl -fsSL "https://raw.githubusercontent.com/deepuyrf-blip/ai-gateway/main/app.py" -o app.py
  mkdir -p web
  curl -fsSL "https://raw.githubusercontent.com/deepuyrf-blip/ai-gateway/main/web/index.html" -o web/index.html
  curl -fsSL "https://raw.githubusercontent.com/deepuyrf-blip/ai-gateway/main/requirements.txt" -o requirements.txt
  curl -fsSL "https://raw.githubusercontent.com/deepuyrf-blip/ai-gateway/main/Dockerfile" -o Dockerfile
fi

echo "==> building"
docker build -t ai-gateway:local .

echo "==> starting on port $PORT"
docker rm -f ai-gateway >/dev/null 2>&1 || true
docker run -d --name ai-gateway --restart unless-stopped \
  -p "$PORT:7860" -e PORT=7860 -e GATEWAY_DATA_DIR=/data \
  -v "$DIR/data:/data" ai-gateway:local

echo
echo "  AI Gateway is running."
echo "  Dashboard : http://localhost:$PORT"
echo "  Setup code: docker logs ai-gateway | grep -A1 'SETUP CODE'"
echo

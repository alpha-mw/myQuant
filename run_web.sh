#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

ensure_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "缺少命令: $command_name"
    exit 1
  fi
}

check_port() {
  local port="$1"
  local label="$2"

  if ! command -v lsof >/dev/null 2>&1; then
    return
  fi

  local occupied
  occupied="$(lsof -iTCP:"$port" -sTCP:LISTEN -n -P 2>/dev/null || true)"
  if [[ -n "$occupied" ]]; then
    echo "$label 端口 $port 已被占用，请先关闭旧进程："
    echo "$occupied"
    exit 1
  fi
}

cleanup() {
  trap - INT TERM EXIT

  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi

  wait || true
}

ensure_command "$PYTHON_BIN"
ensure_command npm

check_port "$BACKEND_PORT" "后端"
check_port "$FRONTEND_PORT" "前端"

trap cleanup INT TERM EXIT

echo "启动后端: http://$BACKEND_HOST:$BACKEND_PORT"
"$PYTHON_BIN" -m uvicorn web.main:app \
  --host "$BACKEND_HOST" \
  --port "$BACKEND_PORT" \
  --reload \
  --reload-dir "$ROOT_DIR/web" &
BACKEND_PID=$!

echo "启动前端: http://$FRONTEND_HOST:$FRONTEND_PORT"
(
  cd "$ROOT_DIR/frontend"
  npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

echo "前端页面: http://$FRONTEND_HOST:$FRONTEND_PORT/"
echo "后端接口: http://$BACKEND_HOST:$BACKEND_PORT/api/v1/health"
echo "按 Ctrl+C 可同时关闭前后端。"

wait

#!/usr/bin/env bash
# Hermes 投研助手 一键启动
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "首次运行：创建虚拟环境并安装依赖（akshare 较大，约几分钟）…"
  python3 -m venv "$DIR/.venv"
  PY="$DIR/.venv/bin/python"
  "$PY" -m pip install --upgrade pip >/dev/null
  "$PY" -m pip install -r "$DIR/requirements.txt"
fi

echo ""
echo "  Hermes 投研助手  →  http://127.0.0.1:8000"
echo "  （没有 Claude API Key 也能跑：自动进入演示模式，数据依旧真实）"
echo ""
exec "$PY" -m uvicorn app:app --app-dir "$DIR" --host 127.0.0.1 --port 8000

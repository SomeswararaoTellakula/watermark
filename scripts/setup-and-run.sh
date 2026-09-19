#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACK_DIR="$ROOT_DIR/DiffMark-main/webapp"
FRONT_DIR="$ROOT_DIR/diffmark-frontend"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source ./.venv/bin/activate
python3 -m pip install --upgrade pip wheel setuptools

REQ=""
if [ -f "$BACK_DIR/requirements.txt" ]; then
  REQ="$BACK_DIR/requirements.txt"
elif [ -f "$ROOT_DIR/DiffMark-main/requirements.txt" ]; then
  REQ="$ROOT_DIR/DiffMark-main/requirements.txt"
fi
if [ -n "$REQ" ]; then
  python3 -m pip install -r "$REQ"
fi

if [ -f "$BACK_DIR/run_server.sh" ]; then
  nohup bash "$BACK_DIR/run_server.sh" 5055 > "$ROOT_DIR/backend.log" 2>&1 &
fi

if [ -d "$FRONT_DIR" ]; then
  cd "$FRONT_DIR"
  if have_cmd npm; then
    if [ ! -d node_modules ]; then
      npm install
    fi
    npm run build
    npm run start
  else
    echo "npm not found. Install Node.js (LTS) to run the frontend:"
    echo "  brew install node    # macOS with Homebrew"
    echo "or"
    echo "  nvm install --lts    # after installing nvm"
    exit 1
  fi
fi


#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONT_DIR="$ROOT_DIR/diffmark-frontend"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

echo "== DiffMark Dev Check =="
echo "Root: $ROOT_DIR"

if ! have_cmd node || ! have_cmd npm; then
  echo "ERROR: Node.js and/or npm not found on PATH."
  echo ""
  echo "Install Node LTS (v18+) and re-run this script."
  echo "macOS with Homebrew:"
  echo "  brew install node"
  echo ""
  echo "After installation, verify:"
  echo "  node -v && npm -v"
  exit 1
fi

echo "Node: $(node -v)"
echo "npm:  $(npm -v)"

if [ ! -d "$FRONT_DIR" ]; then
  echo "ERROR: Frontend directory not found: $FRONT_DIR"
  exit 1
fi

cd "$FRONT_DIR"
if [ ! -d node_modules ]; then
  echo "Installing frontend dependencies..."
  npm install
fi

echo "Starting Next.js on http://localhost:3000 ..."
echo "Tip: Press Ctrl+C to stop."
npm run start


#!/usr/bin/env bash
# Wrapper to run the Flask app on a specified port.
cd "$(dirname "$0")/.." || exit 1
# Run app on port provided as first arg or default to 9000
PORT=${1:-9000}
python3 -c "from webapp.app import app; app.run(host='0.0.0.0', port=${PORT})"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "${SCRIPT_DIR}")"
VENV_PYTHON="${PROJECT_DIR}/venv/bin/python"
ARM_API_URL="${INFERENCE_ARM_API_URL:-http://127.0.0.1:5001}"
ARM_NAME="${INFERENCE_ARM_NAME:-left}"

if pgrep -f "${VENV_PYTHON} -m inference_service" >/dev/null 2>&1; then
    pkill -TERM -f "${VENV_PYTHON} -m inference_service"
    echo "Inference client stopped."
else
    echo "Inference client is not running."
fi

# If a blocking moveL is still active in main.py, stop it as well.
curl --fail --silent --show-error \
    -X POST \
    -H 'Content-Type: application/json' \
    -d "{\"arm_name\":\"${ARM_NAME}\"}" \
    "${ARM_API_URL}/api/arm/stop_arm" \
    >/dev/null 2>&1 || true

echo "Stop command sent to arm '${ARM_NAME}'. The .215 server was not stopped."

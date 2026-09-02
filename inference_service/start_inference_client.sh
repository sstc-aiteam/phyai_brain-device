#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "${SCRIPT_DIR}")"
VENV_PYTHON="${PROJECT_DIR}/venv/bin/python"
LOG_FILE="${PROJECT_DIR}/inference_service.log"
ARM_API_URL="${INFERENCE_ARM_API_URL:-http://127.0.0.1:5001}"
CAMERA_NAME="${INFERENCE_CAMERA_NAME:-left}"
IMAGE_URL="${INFERENCE_IMAGE_URL:-${ARM_API_URL}/api/vision/camera_rgb?camera_name=${CAMERA_NAME}}"

if [[ -n "${INFERENCE_EXECUTE_ACTIONS:-}" ]]; then
    EXECUTE_ACTIONS="${INFERENCE_EXECUTE_ACTIONS}"
else
    echo "Select inference client mode:"
    echo "  1) Inference only (arm will not move) [default]"
    echo "  2) Inference and execute arm motions"
    read -r -p "Enter 1 or 2: " MODE_SELECTION
    case "${MODE_SELECTION:-1}" in
        1)
            EXECUTE_ACTIONS="false"
            ;;
        2)
            echo "WARNING: The left arm will execute model-generated moveL commands."
            read -r -p "Type MOVE to confirm: " MOVE_CONFIRMATION
            if [[ "${MOVE_CONFIRMATION}" != "MOVE" ]]; then
                echo "Arm execution was not confirmed; inference will not start."
                exit 1
            fi
            EXECUTE_ACTIONS="true"
            ;;
        *)
            echo "ERROR: please enter 1 or 2." >&2
            exit 1
            ;;
    esac
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "ERROR: venv Python not found: ${VENV_PYTHON}" >&2
    exit 1
fi

if pgrep -f "${VENV_PYTHON} -m inference_service" >/dev/null 2>&1; then
    echo "Inference client is already running."
    exit 0
fi

if ! curl --fail --silent --show-error \
    "${ARM_API_URL}/api/camera/get_camera_status?camera_name=${CAMERA_NAME}" \
    >/dev/null; then
    echo "ERROR: main.py API is not available at ${ARM_API_URL}." >&2
    echo "Start main.py before running this script." >&2
    exit 1
fi

echo "Starting camera '${CAMERA_NAME}'..."
CAMERA_RESPONSE="$(curl --fail --silent --show-error \
    -X POST \
    -H 'Content-Type: application/json' \
    -d "{\"camera_name\":\"${CAMERA_NAME}\"}" \
    "${ARM_API_URL}/api/camera/start_camera")"

if [[ "${CAMERA_RESPONSE}" != *'"result":true'* ]]; then
    echo "ERROR: camera failed to start: ${CAMERA_RESPONSE}" >&2
    exit 1
fi

CAMERA_TEST_FILE="$(mktemp /tmp/phyai-inference-camera.XXXXXX.jpg)"
trap 'rm -f "${CAMERA_TEST_FILE}"' EXIT
curl --fail --silent --show-error "${IMAGE_URL}" --output "${CAMERA_TEST_FILE}"
if [[ ! -s "${CAMERA_TEST_FILE}" ]]; then
    echo "ERROR: camera returned an empty image." >&2
    exit 1
fi
echo "Camera is ready."

cd "${PROJECT_DIR}"
nohup env \
    INFERENCE_HOST="${INFERENCE_HOST:-192.168.50.215}" \
    INFERENCE_PORT="${INFERENCE_PORT:-5555}" \
    INFERENCE_ROBOT_IP="${INFERENCE_ROBOT_IP:-192.168.50.76}" \
    INFERENCE_IMAGE_URL="${IMAGE_URL}" \
    INFERENCE_EXECUTE_ACTIONS="${EXECUTE_ACTIONS}" \
    INFERENCE_MOTION_SPEED="${INFERENCE_MOTION_SPEED:-0.02}" \
    INFERENCE_MOTION_ACCELERATION="${INFERENCE_MOTION_ACCELERATION:-0.05}" \
    INFERENCE_MAX_TRANSLATION_DELTA="${INFERENCE_MAX_TRANSLATION_DELTA:-0.025}" \
    INFERENCE_MAX_ROTATION_DELTA="${INFERENCE_MAX_ROTATION_DELTA:-0.05}" \
    INFERENCE_INSTRUCTION="${INFERENCE_INSTRUCTION:-open drawer}" \
    INFERENCE_TIMEOUT="${INFERENCE_TIMEOUT:-30}" \
    "${VENV_PYTHON}" -m inference_service \
    > "${LOG_FILE}" 2>&1 &

CLIENT_PID=$!
sleep 1

if ! kill -0 "${CLIENT_PID}" 2>/dev/null; then
    echo "ERROR: inference client exited during startup." >&2
    tail -n 30 "${LOG_FILE}" >&2 || true
    exit 1
fi

echo "Inference client started (PID ${CLIENT_PID})."
echo "Arm execution: ${EXECUTE_ACTIONS}"
echo "Log: ${LOG_FILE}"

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from services import vision_service

# In-process VLM ownership boundary:
# - services/vision_service.py owns camera capture, YOLO, filtering, depth,
#   coordinates and the latest synchronized scene snapshot.
# - This module consumes that state directly inside the TaiROS 5000 process.
# - It does not start a camera, run YOLO, expose another web service,
#   or require a bridge process.

VLM_SESSION_ID = str(
    os.environ.get("VLM_SESSION_ID") or "d405-main"
).strip() or "d405-main"
VLM_NARRATOR_HZ = max(
    0.2,
    float(os.environ.get("VLM_NARRATOR_HZ") or 2.0),
)
VLM_NARRATOR_JPEG_QUALITY = max(
    50,
    min(95, int(os.environ.get("VLM_NARRATOR_JPEG_QUALITY") or 85)),
)

_NARRATOR_RUNTIME_LOCK = threading.Lock()
_NARRATOR_RUNTIME_STOP = threading.Event()
_NARRATOR_RUNTIME_THREAD: threading.Thread | None = None
_NARRATOR_LAST_FRAME_KEY = ""
_NARRATOR_LAST_ERROR = ""
_NARRATOR_LAST_RESULT_AT = 0.0

QWEN_BASE_URL = str(
    os.environ.get("NARRATOR_QWEN_BASE_URL") or "http://127.0.0.1:11434/v1"
).rstrip("/")
QWEN_MODEL = str(
    os.environ.get("NARRATOR_QWEN_MODEL")
    or "qwen3-vl:2b-instruct-q4_K_M"
).strip()
QWEN_TEMPERATURE = float(os.environ.get("NARRATOR_TEMPERATURE") or 0.35)
QWEN_TOP_P = float(os.environ.get("NARRATOR_TOP_P") or 0.80)
QWEN_TIMEOUT_SEC = float(os.environ.get("NARRATOR_QWEN_TIMEOUT_SEC") or 45)
OLLAMA_NATIVE_URL = str(
    os.environ.get("NARRATOR_OLLAMA_NATIVE_URL") or "http://127.0.0.1:11434"
).rstrip("/")
QWEN_KEEP_ALIVE = str(
    os.environ.get("NARRATOR_QWEN_KEEP_ALIVE") or "-1"
).strip()
QWEN_KEEPALIVE_ENABLED = str(
    os.environ.get("NARRATOR_QWEN_KEEPALIVE_ENABLED") or "1"
).strip().lower() in {"1", "true", "yes", "on"}
QWEN_KEEPALIVE_INTERVAL_SEC = max(
    30.0,
    float(os.environ.get("NARRATOR_QWEN_KEEPALIVE_INTERVAL_SEC") or 120.0),
)

# Fast VLM mode. These values only reduce the expensive visual-language request;
# YOLO detections, tracking, depth ordering, and pairwise safety logic are unchanged.
VLM_MAX_OBJECTS = max(
    1,
    int(os.environ.get("NARRATOR_VLM_MAX_OBJECTS") or 4),
)
VLM_OBJECT_IMAGE_EDGE = max(
    160,
    int(os.environ.get("NARRATOR_VLM_OBJECT_IMAGE_EDGE") or 160),
)
VLM_OBJECT_MAX_TOKENS = max(
    64,
    int(os.environ.get("NARRATOR_VLM_OBJECT_MAX_TOKENS") or 140),
)
VLM_GENERAL_IMAGE_EDGE = max(
    256,
    int(os.environ.get("NARRATOR_VLM_GENERAL_IMAGE_EDGE") or 512),
)
VLM_GENERAL_MAX_TOKENS = max(
    64,
    int(os.environ.get("NARRATOR_VLM_GENERAL_MAX_TOKENS") or 96),
)



# NARRATOR_CONTACT_SHEET_STREAMING_VLM
# One contact sheet contains close-up crops plus one whole-scene view. Qwen sees
# one image and streams one OBJ line at a time, followed by one SCENE line.
VLM_COMBINED_MAX_OBJECTS = max(
    1,
    min(
        6,
        int(os.environ.get("NARRATOR_VLM_COMBINED_MAX_OBJECTS") or 4),
    ),
)
VLM_COMBINED_IMAGE_EDGE = max(
    320,
    int(os.environ.get("NARRATOR_VLM_COMBINED_IMAGE_EDGE") or 384),
)
VLM_COMBINED_MAX_TOKENS = max(
    96,
    int(os.environ.get("NARRATOR_VLM_COMBINED_MAX_TOKENS") or 180),
)
VLM_CONTACT_SHEET_CELL_EDGE = max(
    128,
    int(os.environ.get("NARRATOR_VLM_CONTACT_CELL_EDGE") or 176),
)
VLM_FIRST_DETAIL_TARGET_SEC = max(
    3.0,
    float(os.environ.get("NARRATOR_VLM_FIRST_DETAIL_TARGET_SEC") or 10.0),
)
VLM_EMPTY_SCENE_TEXT = str(
    os.environ.get("NARRATOR_VLM_EMPTY_SCENE_TEXT")
    or "目前椅面上未見可辨識物品。"
).strip()


# Optional assisted exhibition mode.
#
# live:
#   Use the real Qwen3-VL pipeline.
# assisted:
#   Keep the real camera, YOLO detections, stable IDs, positions and placement
#   rules, but provide green/purple narration from a deterministic JSON bank.
NARRATOR_OUTPUT_MODE = str(
    os.environ.get("NARRATOR_OUTPUT_MODE") or "auto"
).strip().lower()
if NARRATOR_OUTPUT_MODE not in {"auto", "live", "assisted"}:
    raise RuntimeError(
        "NARRATOR_OUTPUT_MODE must be 'auto', 'live' or 'assisted', "
        f"got {NARRATOR_OUTPUT_MODE!r}"
    )

# In auto mode the active browser page chooses the runtime output mode.
# index.html sends a live heartbeat; demo.html sends an assisted heartbeat.
# If all demo heartbeats disappear, the service safely falls back to live.
UI_MODE_HEARTBEAT_TTL_SEC = max(
    2.5,
    float(os.environ.get("NARRATOR_UI_MODE_HEARTBEAT_TTL_SEC") or 4.5),
)
UI_MODE_WATCH_INTERVAL_SEC = max(
    0.25,
    float(os.environ.get("NARRATOR_UI_MODE_WATCH_INTERVAL_SEC") or 0.5),
)
ASSISTED_TRACK_HOLD_SEC = max(
    0.5,
    float(os.environ.get("NARRATOR_ASSISTED_TRACK_HOLD_SEC") or 1.5),
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DESCRIPTION_PATH = Path(
    os.environ.get("NARRATOR_DEMO_DESCRIPTION_PATH")
    or PROJECT_ROOT / "scene_narrator" / "demo_description_bank.json"
).expanduser().resolve()
DEMO_SEED = str(
    os.environ.get("NARRATOR_DEMO_SEED") or "official_visit_default"
).strip()
ASSISTED_OUTPUT_GAP_SEC = max(
    0.0,
    float(
        os.environ.get("NARRATOR_ASSISTED_OUTPUT_GAP_SEC")
        or 0.7
    ),
)
DEMO_DESCRIPTION_BANK: dict[str, Any] = {}

MIN_DETECTION_CONF = float(os.environ.get("NARRATOR_MIN_CONF") or 0.35)
MAX_NARRATION_LINES = int(os.environ.get("NARRATOR_MAX_LINES") or 5)
SEMANTIC_DEDUP_WINDOW_SEC = float(os.environ.get("NARRATOR_SEMANTIC_DEDUP_SEC") or 20.0)
MAX_LINES_PER_TYPE = int(os.environ.get("NARRATOR_MAX_LINES_PER_TYPE") or 2)
SCENE_CHANGE_MIN_SEC = float(os.environ.get("NARRATOR_MIN_INTERVAL_SEC") or 2.2)
TRACK_IOU_THRESHOLD = float(os.environ.get("NARRATOR_TRACK_IOU") or 0.35)
TRACK_TTL_SEC = float(os.environ.get("NARRATOR_TRACK_TTL_SEC") or 8.0)
TRACK_BOX_SMOOTH_ALPHA = float(
    os.environ.get("NARRATOR_TRACK_BOX_SMOOTH_ALPHA") or 0.38
)
MIN_TRACK_HITS_FOR_NARRATION = int(
    os.environ.get("NARRATOR_MIN_TRACK_HITS") or 2
)
# Keep recently seen tracks in the semantic scene state even when YOLO misses
# one or two frames. This state hold is for narration/Qwen scheduling only;
# the browser overlay still controls its own shorter visual fade-out.
SCENE_TRACK_HOLD_SEC = float(
    os.environ.get("NARRATOR_SCENE_TRACK_HOLD_SEC") or 8.0
)
# A previously announced class may be announced again only after it has been
# continuously absent for this long. Short detector dropouts must stay silent.
PRESENCE_REANNOUNCE_ABSENCE_SEC = float(
    os.environ.get("NARRATOR_PRESENCE_REANNOUNCE_ABSENCE_SEC") or 30.0
)
RELATIONS_ENABLED = str(
    os.environ.get("NARRATOR_RELATIONS_ENABLED") or "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Optional occlusion/grasp-risk alert. This is strictly an add-on to the existing
# YOLO and narration flow. Two already-detected objects are sent to Qwen only when
# their boxes overlap or remain extremely close. Qwen confirms visible occlusion
# or grasp-region interference; physical contact is NOT required for a safety
# alert. The normal boxes, narration and missed-box audit remain unchanged.
PAIRWISE_PICKUP_ENABLED = str(
    os.environ.get("NARRATOR_PAIRWISE_PICKUP_ENABLED") or "0"
).strip().lower() in {"1", "true", "yes", "on"}
PAIRWISE_PICKUP_DELAY_SEC = float(
    os.environ.get("NARRATOR_PAIRWISE_PICKUP_DELAY_SEC") or 2.5
)
PAIRWISE_PICKUP_RETRY_SEC = float(
    os.environ.get("NARRATOR_PAIRWISE_PICKUP_RETRY_SEC") or 15.0
)
PAIRWISE_PICKUP_MIN_CONF = float(
    os.environ.get("NARRATOR_PAIRWISE_PICKUP_MIN_CONF") or 0.65
)
# Directional ordering is more safety-critical than merely detecting a visible
# interference.  A pair may therefore trigger an alert at 0.65 while the
# before/after order remains unconfirmed unless depth or an explicit, internally
# consistent visual direction is available.
PAIRWISE_DIRECTION_MIN_CONF = float(
    os.environ.get("NARRATOR_PAIRWISE_DIRECTION_MIN_CONF") or 0.78
)
PAIRWISE_DEPTH_FRONT_MIN_DELTA_M = float(
    os.environ.get("NARRATOR_PAIRWISE_DEPTH_FRONT_MIN_DELTA_M") or 0.025
)
PAIRWISE_ALLOW_VLM_ONLY_ORDER = str(
    os.environ.get("NARRATOR_PAIRWISE_ALLOW_VLM_ONLY_ORDER") or "0"
).strip().lower() in {"1", "true", "yes", "on"}
PAIRWISE_PICKUP_MAX_PAIRS = int(
    os.environ.get("NARRATOR_PAIRWISE_PICKUP_MAX_PAIRS") or 1
)
PAIRWISE_IMAGE_EDGE = max(
    256,
    int(os.environ.get("NARRATOR_PAIRWISE_IMAGE_EDGE") or 512),
)
PAIRWISE_CONTACT_IMAGE_EDGE = max(
    224,
    int(os.environ.get("NARRATOR_PAIRWISE_CONTACT_IMAGE_EDGE") or 384),
)
PAIRWISE_MAX_TOKENS = max(
    160,
    int(os.environ.get("NARRATOR_PAIRWISE_MAX_TOKENS") or 420),
)
PAIRWISE_OVERLAP_MIN_SMALLER_COVERAGE = float(
    os.environ.get("NARRATOR_PAIRWISE_OVERLAP_MIN_SMALLER_COVERAGE") or 0.03
)
PAIRWISE_OVERLAP_MIN_IOU = float(
    os.environ.get("NARRATOR_PAIRWISE_OVERLAP_MIN_IOU") or 0.005
)
# Candidate generation is intentionally more permissive than alert generation.
# Two YOLO boxes may only touch at the visible boundary when one object hides the
# lower/side portion of the other.  Such pairs are sent to VLM only after the
# geometry remains suspicious across multiple frames.  VLM still owns the final
# physical-overlap decision, so nearby objects do not directly create alerts.
PAIRWISE_NEAR_GAP_PX = float(
    os.environ.get("NARRATOR_PAIRWISE_NEAR_GAP_PX") or 10.0
)
PAIRWISE_EXPAND_RATIO = float(
    os.environ.get("NARRATOR_PAIRWISE_EXPAND_RATIO") or 0.045
)
PAIRWISE_EXPAND_MAX_PX = float(
    os.environ.get("NARRATOR_PAIRWISE_EXPAND_MAX_PX") or 14.0
)
PAIRWISE_REQUIRED_FRAMES = int(
    os.environ.get("NARRATOR_PAIRWISE_REQUIRED_FRAMES") or 2
)
PAIRWISE_EVIDENCE_TTL_SEC = float(
    os.environ.get("NARRATOR_PAIRWISE_EVIDENCE_TTL_SEC") or 5.0
)
# Use recent YOLO tracks for overlap review so the backend sees the same short-
# lived boxes that are still visible on the live canvas during detector dropouts.
PAIRWISE_TRACK_HOLD_SEC = float(
    os.environ.get("NARRATOR_PAIRWISE_TRACK_HOLD_SEC") or 3.0
)
# Qwen can be inconsistent on the same crop. Retry a small, bounded number of
# times while the pair remains suspicious, but latch the first confirmed alert.
# A later negative answer must never erase an already confirmed safety warning.
PAIRWISE_MAX_REVIEW_ATTEMPTS = int(
    os.environ.get("NARRATOR_PAIRWISE_MAX_REVIEW_ATTEMPTS") or 3
)
PAIRWISE_NEGATIVE_COOLDOWN_SEC = float(
    os.environ.get("NARRATOR_PAIRWISE_NEGATIVE_COOLDOWN_SEC") or 45.0
)
PAIRWISE_DISAPPEAR_CLEAR_SEC = float(
    os.environ.get("NARRATOR_PAIRWISE_DISAPPEAR_CLEAR_SEC") or 2.5
)

IDLE_NARRATION_INTERVAL_SEC = float(
    os.environ.get("NARRATOR_IDLE_INTERVAL_SEC") or 5.5
)
GENERAL_IDLE_AFTER_SEC = float(
    os.environ.get("NARRATOR_GENERAL_IDLE_AFTER_SEC") or 1.0
)
# Retained for backward-compatible diagnostics only. Normal green/purple VLM
# narration is event-triggered and is never restarted by a periodic timer.
GENERAL_IDLE_INTERVAL_SEC = float(
    os.environ.get("NARRATOR_GENERAL_IDLE_INTERVAL_SEC") or 0.0
)
# Even if green and purple results become ready close together, keep a small
# visible gap so the UI reads as a progressive explanation instead of a burst.
GREEN_PURPLE_MIN_GAP_SEC = max(
    0.0,
    float(os.environ.get("NARRATOR_GREEN_PURPLE_MIN_GAP_SEC") or 1.0),
)

# YOLO-priority scheduling. Qwen waits until the detected scene has remained
# stable for a few seconds, and only one Qwen request may use the local model at
# a time. This prevents stale scene-change workers, unboxed audits, and idle
# comments from piling up while YOLO is trying to keep the boxes current.
QWEN_ENRICH_DELAY_SEC = float(
    os.environ.get("NARRATOR_QWEN_ENRICH_DELAY_SEC") or 1.0
)
QWEN_RETRY_INTERVAL_SEC = float(
    os.environ.get("NARRATOR_QWEN_RETRY_INTERVAL_SEC") or 2.0
)
SCENE_CHANGE_CONFIRM_FRAMES = max(
    1,
    int(os.environ.get("NARRATOR_SCENE_CHANGE_CONFIRM_FRAMES") or 2),
)
UNBOXED_AUDIT_ENABLED = str(
    os.environ.get("NARRATOR_UNBOXED_AUDIT_ENABLED") or "0"
).strip().lower() in {"1", "true", "yes", "on"}
UNBOXED_AUDIT_DELAY_SEC = float(
    os.environ.get("NARRATOR_UNBOXED_AUDIT_DELAY_SEC") or 12.0
)
UNBOXED_AUDIT_INTERVAL_SEC = float(
    os.environ.get("NARRATOR_UNBOXED_AUDIT_INTERVAL_SEC") or 30.0
)
GENERAL_IDLE_ENABLED = str(
    os.environ.get("NARRATOR_GENERAL_IDLE_ENABLED") or "1"
).strip().lower() in {"1", "true", "yes", "on"}

# llama.cpp is commonly started with -np 1. Keep the client side serialized too,
# and drop stale/extra jobs instead of letting background threads queue forever.
_QWEN_WORK_LOCK = threading.Lock()
_QWEN_KEEPALIVE_STOP = threading.Event()
_QWEN_KEEPALIVE_THREAD: threading.Thread | None = None

_UI_MODE_LOCK = threading.Lock()
_UI_MODE_CLIENTS: dict[str, dict[str, Any]] = {}
_UI_MODE_WATCHDOG_STOP = threading.Event()
_UI_MODE_WATCHDOG_THREAD: threading.Thread | None = None
_RUNTIME_OUTPUT_MODE = (
    NARRATOR_OUTPUT_MODE
    if NARRATOR_OUTPUT_MODE in {"live", "assisted"}
    else "live"
)


def runtime_output_mode() -> str:
    """Return the effective green/purple output mode for this process."""
    if NARRATOR_OUTPUT_MODE in {"live", "assisted"}:
        return NARRATOR_OUTPUT_MODE
    with _UI_MODE_LOCK:
        return _RUNTIME_OUTPUT_MODE


def assisted_output_enabled() -> bool:
    return runtime_output_mode() == "assisted"


def _prune_ui_mode_clients_locked(now_value: float) -> None:
    expired = [
        client_id
        for client_id, state in _UI_MODE_CLIENTS.items()
        if now_value - float(state.get("last_seen") or 0.0)
        > UI_MODE_HEARTBEAT_TTL_SEC
    ]
    for client_id in expired:
        _UI_MODE_CLIENTS.pop(client_id, None)


def _compute_auto_output_mode_locked(now_value: float) -> str:
    _prune_ui_mode_clients_locked(now_value)
    # Demo wins while at least one demo page is alive. This prevents a
    # background index tab from switching the exhibition back to live.
    if any(
        state.get("mode") == "assisted"
        for state in _UI_MODE_CLIENTS.values()
    ):
        return "assisted"
    return "live"


def _update_effective_output_mode_locked(
    now_value: float,
) -> tuple[str, str, bool]:
    global _RUNTIME_OUTPUT_MODE
    previous = _RUNTIME_OUTPUT_MODE
    current = (
        NARRATOR_OUTPUT_MODE
        if NARRATOR_OUTPUT_MODE in {"live", "assisted"}
        else _compute_auto_output_mode_locked(now_value)
    )
    _RUNTIME_OUTPUT_MODE = current
    return previous, current, previous != current


def _ollama_keep_alive_value() -> int | str:
    try:
        return int(QWEN_KEEP_ALIVE)
    except (TypeError, ValueError):
        return QWEN_KEEP_ALIVE


def pin_qwen_model() -> dict:
    """Load Qwen into Ollama and keep it resident without running vision inference."""
    payload = {
        "model": QWEN_MODEL,
        "prompt": "",
        "stream": False,
        "keep_alive": _ollama_keep_alive_value(),
    }
    req = urllib.request.Request(
        OLLAMA_NATIVE_URL + "/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=max(120.0, QWEN_TIMEOUT_SEC)) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw or "{}")


def _qwen_keepalive_worker() -> None:
    while not _QWEN_KEEPALIVE_STOP.wait(QWEN_KEEPALIVE_INTERVAL_SEC):
        # Assisted narration never needs Qwen. Do not refresh the shared
        # Ollama model while a demo page owns the runtime mode.
        if assisted_output_enabled():
            continue
        # Never queue a residency request in front of an active object or
        # environment VLM job. The next cycle will refresh residency instead.
        if _QWEN_WORK_LOCK.locked():
            continue
        try:
            result = pin_qwen_model()
            load_sec = float(result.get("load_duration") or 0) / 1_000_000_000
            print(
                f"[qwen-resident] refreshed model={QWEN_MODEL} "
                f"keep_alive={QWEN_KEEP_ALIVE} load={load_sec:.3f}s",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[qwen-resident][WARN] {type(exc).__name__}: {exc}",
                flush=True,
            )


def _pin_qwen_once_at_startup() -> None:
    """Warm the model immediately instead of waiting for the first 120 s cycle."""
    if assisted_output_enabled():
        return
    if not _QWEN_WORK_LOCK.acquire(blocking=False):
        return
    try:
        started = time.perf_counter()
        result = pin_qwen_model()
        load_sec = float(result.get("load_duration") or 0) / 1_000_000_000
        print(
            "[qwen-resident] startup warmup "
            f"model={QWEN_MODEL} load={load_sec:.3f}s "
            f"elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[qwen-resident][WARN] startup {type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        _QWEN_WORK_LOCK.release()



def canonical(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("_", " ").replace("-", " "))


def clamp(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


def extract_box(item: dict) -> list[float] | None:
    for key in ("bbox_xyxy", "xyxy", "bbox", "box", "original_bbox"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                x1, y1, x2, y2 = [float(x) for x in value[:4]]
            except Exception:
                continue
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]
        if isinstance(value, dict):
            for keys in (
                ("x1", "y1", "x2", "y2"),
                ("left", "top", "right", "bottom"),
                ("xmin", "ymin", "xmax", "ymax"),
            ):
                if all(k in value for k in keys):
                    try:
                        x1, y1, x2, y2 = [float(value[k]) for k in keys]
                    except Exception:
                        continue
                    if x2 > x1 and y2 > y1:
                        return [x1, y1, x2, y2]
    return None


def extract_label(item: dict) -> str:
    for key in ("class_name", "label", "name", "class", "category", "object_class"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return canonical(value)
    return "unknown"


def extract_conf(item: dict) -> float:
    for key in ("confidence", "score", "conf", "probability"):
        try:
            if item.get(key) is not None:
                return float(item[key])
        except Exception:
            pass
    return 0.0


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1.0, aa + ab - inter)


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def containment_ratio(inner: list[float], outer: list[float]) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(1.0, box_area(inner))


@dataclass
class Track:
    track_id: int
    label: str
    box: list[float]
    confidence: float
    last_seen: float
    hits: int = 1
    # Preserve D405 / external-detector metadata across stable-ID tracking.
    # Without this, pairwise review loses depth and cannot safely resolve which
    # object is actually closer to the camera.
    metadata: dict = field(default_factory=dict)


@dataclass
class SessionState:
    next_track_id: int = 1
    tracks: list[Track] = field(default_factory=list)
    last_signature: str = ""
    last_narrated_at: float = 0.0
    last_result: dict = field(default_factory=dict)
    qwen_pending: bool = False
    qwen_signature: str = ""
    qwen_error: str = ""
    qwen_streaming: bool = False
    qwen_stream_started_at: float = 0.0
    qwen_first_result_at: float = 0.0
    qwen_last_partial_at: float = 0.0
    qwen_expected_object_ids: list[int] = field(default_factory=list)
    qwen_received_object_ids: list[int] = field(default_factory=list)
    qwen_stream_scene_summary: str = ""
    scene_revision: int = 0
    scene_snapshot: list[dict] = field(default_factory=list)
    qwen_job_token: int = 0
    object_description_cache: dict[int, dict] = field(default_factory=dict)
    unboxed_pending: bool = False
    unboxed_job_token: int = 0
    unboxed_result: dict | None = None
    unboxed_checked_at: float = 0.0
    pairwise_pending: bool = False
    pairwise_job_token: int = 0
    pairwise_checked_at: float = 0.0
    pairwise_signature: str = ""
    pairwise_fingerprint: str = ""
    pairwise_result: dict = field(default_factory=dict)
    pairwise_ready_line: dict | None = None
    pairwise_error: str = ""
    pairwise_attempt_count: int = 0
    pairwise_negative_until: float = 0.0
    pairwise_latched: bool = False
    pairwise_last_candidate_seen_at: float = 0.0
    # Short-lived per-pair evidence used to reject one-frame bbox jitter.  This
    # lives only in memory and creates no files.
    pairwise_candidate_evidence: dict[str, dict] = field(default_factory=dict)
    idle_cycle_index: int = 0
    last_idle_emit_at: float = 0.0
    last_output_at: float = 0.0
    general_idle_pending: bool = False
    general_idle_ready: str = ""
    general_idle_job_token: int = 0
    recent_general_idle: list[str] = field(default_factory=list)
    last_presence_key: str = ""
    announced_presence_counts: dict[str, int] = field(default_factory=dict)
    presence_last_seen_at: dict[str, float] = field(default_factory=dict)
    last_general_idle_emit_at: float = 0.0
    last_object_emit_at: float = 0.0
    recent_semantic_lines: list[tuple[float, str, str]] = field(default_factory=list)
    last_scene_change_at: float = 0.0
    last_qwen_attempt_at: float = 0.0
    last_enriched_signature: str = ""
    last_general_signature: str = ""
    pending_scene_signature: str = ""
    pending_scene_snapshot: list[dict] = field(default_factory=list)
    pending_scene_confirm_count: int = 0
    reobserve_requested: bool = False
    last_scene_event_reason: str = ""
    # TaiROS integration state. Idle VLM work is suppressed while a user
    # task is active; cached narration and pairwise safety evidence remain
    # available to the text-only task planner.
    task_active: bool = False
    task_id: str = ""
    task_started_at: float = 0.0
    latest_api_result: dict = field(default_factory=dict)


_SESSIONS: dict[str, SessionState] = {}
_SESSION_LOCK = threading.Lock()


def get_session(session_id: str) -> SessionState:
    with _SESSION_LOCK:
        return _SESSIONS.setdefault(session_id, SessionState())


def reset_qwen_stream_state(
    session: SessionState,
) -> None:
    """Reset only the in-memory progressive-output diagnostics."""
    session.qwen_streaming = False
    session.qwen_stream_started_at = 0.0
    session.qwen_first_result_at = 0.0
    session.qwen_last_partial_at = 0.0
    session.qwen_expected_object_ids = []
    session.qwen_received_object_ids = []
    session.qwen_stream_scene_summary = ""


def _reset_session_narration_state(
    session: SessionState,
    *,
    reason: str,
    now_value: float | None = None,
) -> None:
    """Invalidate narration jobs without clearing YOLO tracks."""
    now_value = float(now_value if now_value is not None else time.time())

    session.reobserve_requested = True
    session.last_scene_event_reason = str(reason or "reobserve")
    session.last_scene_change_at = now_value

    session.qwen_job_token += 1
    session.qwen_pending = False
    session.qwen_signature = ""
    reset_qwen_stream_state(session)

    session.general_idle_job_token += 1
    session.general_idle_pending = False
    session.general_idle_ready = ""

    session.unboxed_job_token += 1
    session.unboxed_pending = False
    session.unboxed_result = None

    session.pairwise_job_token += 1
    session.pairwise_pending = False
    session.pairwise_ready_line = None
    session.pairwise_result = {}

    session.last_result = {}
    session.object_description_cache.clear()
    session.last_enriched_signature = ""
    session.last_general_signature = ""
    session.recent_general_idle.clear()
    session.recent_semantic_lines.clear()
    session.latest_api_result = {}

    session.last_qwen_attempt_at = 0.0
    session.last_narrated_at = 0.0
    session.last_object_emit_at = 0.0
    session.last_general_idle_emit_at = 0.0
    session.last_idle_emit_at = 0.0
    session.last_output_at = 0.0
    session.idle_cycle_index = 0
    session.qwen_error = ""


def _handle_runtime_output_mode_change(
    previous: str,
    current: str,
    *,
    reason: str,
) -> None:
    if previous == current:
        return

    now_value = time.time()
    with _SESSION_LOCK:
        for session in _SESSIONS.values():
            _reset_session_narration_state(
                session,
                reason=(
                    f"output_mode:{previous}->{current}:{reason}"
                ),
                now_value=now_value,
            )

    print(
        "[narrator-mode] runtime switch "
        f"{previous} -> {current} reason={reason}",
        flush=True,
    )

    # When returning from demo to live, warm Qwen immediately rather than
    # waiting for the next keepalive interval.
    if current == "live" and QWEN_KEEPALIVE_ENABLED:
        threading.Thread(
            target=_pin_qwen_once_at_startup,
            name="scene-narrator-qwen-mode-warmup",
            daemon=True,
        ).start()


def register_ui_mode_heartbeat(
    *,
    client_id: str,
    requested_mode: str,
) -> dict[str, Any]:
    normalized_mode = str(requested_mode or "").strip().lower()
    normalized_client_id = str(client_id or "").strip()
    if normalized_mode not in {"live", "assisted"}:
        raise ValueError(f"Unsupported UI output mode: {requested_mode!r}")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", normalized_client_id):
        raise ValueError("Invalid UI mode client ID")

    now_value = time.time()
    with _UI_MODE_LOCK:
        _UI_MODE_CLIENTS[normalized_client_id] = {
            "mode": normalized_mode,
            "last_seen": now_value,
        }
        previous, current, changed = _update_effective_output_mode_locked(
            now_value
        )
        active_clients = len(_UI_MODE_CLIENTS)

    if changed:
        _handle_runtime_output_mode_change(
            previous,
            current,
            reason=f"heartbeat:{normalized_mode}:{normalized_client_id}",
        )

    return {
        "requested_mode": normalized_mode,
        "effective_mode": current,
        "mode_changed": changed,
        "client_id": normalized_client_id,
        "active_clients": active_clients,
    }


def release_ui_mode_client(client_id: str) -> dict[str, Any]:
    normalized_client_id = str(client_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", normalized_client_id):
        raise ValueError("Invalid UI mode client ID")

    now_value = time.time()
    with _UI_MODE_LOCK:
        removed = _UI_MODE_CLIENTS.pop(normalized_client_id, None) is not None
        previous, current, changed = _update_effective_output_mode_locked(
            now_value
        )
        active_clients = len(_UI_MODE_CLIENTS)

    if changed:
        _handle_runtime_output_mode_change(
            previous,
            current,
            reason=f"release:{normalized_client_id}",
        )

    return {
        "effective_mode": current,
        "mode_changed": changed,
        "client_id": normalized_client_id,
        "client_removed": removed,
        "active_clients": active_clients,
    }


def _ui_mode_watchdog_worker() -> None:
    while not _UI_MODE_WATCHDOG_STOP.wait(UI_MODE_WATCH_INTERVAL_SEC):
        now_value = time.time()
        with _UI_MODE_LOCK:
            previous, current, changed = _update_effective_output_mode_locked(
                now_value
            )
        if changed:
            _handle_runtime_output_mode_change(
                previous,
                current,
                reason="heartbeat_expired",
            )


def suppress_duplicate_detections(
    detections: list[dict],
    *,
    iou_threshold: float = 0.42,
    center_threshold: float = 0.32,
) -> list[dict]:
    """Remove duplicate same-class boxes before they create new track IDs."""
    kept: list[dict] = []

    for det in sorted(
        detections,
        key=lambda row: float(row.get("confidence") or 0),
        reverse=True,
    ):
        duplicate = False
        det_box = det["bbox"]
        det_cx, det_cy = center(det_box)
        det_scale = max(
            1.0,
            math.sqrt(max(1.0, box_area(det_box))),
        )

        for existing in kept:
            if canonical(existing.get("label")) != canonical(det.get("label")):
                continue

            existing_box = existing["bbox"]
            overlap = iou(det_box, existing_box)
            ex_cx, ex_cy = center(existing_box)
            ex_scale = max(
                1.0,
                math.sqrt(max(1.0, box_area(existing_box))),
            )
            normalized_center_distance = (
                math.hypot(det_cx - ex_cx, det_cy - ex_cy)
                / max(det_scale, ex_scale)
            )

            if (
                overlap >= iou_threshold
                or normalized_center_distance <= center_threshold
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


def assign_stable_ids(session: SessionState, detections: list[dict]) -> list[dict]:
    detections = suppress_duplicate_detections(detections)
    now = time.time()
    session.tracks = [t for t in session.tracks if now - t.last_seen <= TRACK_TTL_SEC]
    unmatched_tracks = set(range(len(session.tracks)))
    assigned: list[dict] = []

    for det in sorted(detections, key=lambda x: x.get("confidence", 0), reverse=True):
        best_idx = None
        best_score = 0.0
        det_cx, det_cy = center(det["bbox"])
        det_w = max(1.0, det["bbox"][2] - det["bbox"][0])
        det_h = max(1.0, det["bbox"][3] - det["bbox"][1])

        for idx in unmatched_tracks:
            tr = session.tracks[idx]
            if tr.label != det["label"]:
                continue

            overlap_score = iou(tr.box, det["bbox"])
            tr_cx, tr_cy = center(tr.box)
            tr_w = max(1.0, tr.box[2] - tr.box[0])
            tr_h = max(1.0, tr.box[3] - tr.box[1])
            scale = max(1.0, (det_w + det_h + tr_w + tr_h) / 4.0)
            center_score = max(
                0.0,
                1.0 - math.hypot(det_cx - tr_cx, det_cy - tr_cy) / (scale * 2.4),
            )

            # IoU is best for stable boxes; center similarity preserves the same
            # ID when partial occlusion or detector jitter changes box size.
            score = max(overlap_score, center_score * 0.82)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= min(TRACK_IOU_THRESHOLD, 0.30):
            tr = session.tracks[best_idx]
            alpha = clamp(TRACK_BOX_SMOOTH_ALPHA, 0.05, 1.0)
            tr.box = [
                float(old) * (1.0 - alpha) + float(new) * alpha
                for old, new in zip(tr.box, det["bbox"])
            ]
            tr.confidence = det["confidence"]
            tr.last_seen = now
            tr.hits += 1
            tr.metadata = {
                key: value
                for key, value in det.items()
                if key not in {"id", "label", "bbox", "confidence", "track_hits"}
            }
            unmatched_tracks.remove(best_idx)
            track_id = tr.track_id
        else:
            track_id = session.next_track_id
            session.next_track_id += 1
            session.tracks.append(
                Track(
                    track_id=track_id,
                    label=det["label"],
                    box=det["bbox"],
                    confidence=det["confidence"],
                    last_seen=now,
                    hits=1,
                    metadata={
                        key: value
                        for key, value in det.items()
                        if key not in {"id", "label", "bbox", "confidence", "track_hits"}
                    },
                )
            )

        matched_track = next(
            (t for t in session.tracks if t.track_id == track_id),
            None,
        )
        assigned.append({
            **(matched_track.metadata if matched_track else {}),
            **det,
            "bbox": list(matched_track.box) if matched_track else list(det["bbox"]),
            "id": track_id,
            "track_hits": int(matched_track.hits if matched_track else 1),
        })

    return sorted(assigned, key=lambda x: x["id"])


def held_scene_objects(
    session: SessionState,
    *,
    width: int,
    height: int,
    now_value: float | None = None,
) -> list[dict]:
    """Return a dropout-tolerant object state for narration scheduling.

    YOLO boxes may disappear for a frame even though the physical object did
    not move.  The UI already fades old boxes visually; this function provides
    the same hysteresis to scene signatures, blue-event gating, and Qwen jobs.
    """
    now_value = float(now_value if now_value is not None else time.time())
    hold_sec = (
        ASSISTED_TRACK_HOLD_SEC
        if assisted_output_enabled()
        else SCENE_TRACK_HOLD_SEC
    )
    candidates = []
    for track in session.tracks:
        age = now_value - float(track.last_seen)
        if age > hold_sec:
            continue
        if int(track.hits) < MIN_TRACK_HITS_FOR_NARRATION:
            continue
        candidates.append({
            "id": int(track.track_id),
            "label": str(track.label),
            "bbox": [float(v) for v in track.box],
            "confidence": float(track.confidence),
            "track_hits": int(track.hits),
            "frame_width": int(width),
            "frame_height": int(height),
            "_last_seen": float(track.last_seen),
        })

    # If matching briefly creates a new track ID for the same physical object,
    # retain only the most recently seen overlapping/nearby track. Otherwise the
    # semantic count would jump from 1 -> 2 -> 1 and produce another blue line.
    kept: list[dict] = []
    for row in sorted(
        candidates,
        key=lambda x: (float(x.get("_last_seen") or 0), float(x.get("confidence") or 0)),
        reverse=True,
    ):
        duplicate = False
        rcx, rcy = center(row["bbox"])
        rscale = max(1.0, math.sqrt(max(1.0, box_area(row["bbox"]))))
        for existing in kept:
            if canonical(existing.get("label")) != canonical(row.get("label")):
                continue
            ecx, ecy = center(existing["bbox"])
            escale = max(1.0, math.sqrt(max(1.0, box_area(existing["bbox"]))))
            center_distance = math.hypot(rcx - ecx, rcy - ecy) / max(rscale, escale)
            if iou(row["bbox"], existing["bbox"]) >= 0.30 or center_distance <= 0.38:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)

    for row in kept:
        row.pop("_last_seen", None)
    return sorted(kept, key=lambda x: (canonical(x.get("label")), int(x.get("id") or 0)))


def recent_pairwise_objects(
    session: SessionState,
    *,
    width: int,
    height: int,
    now_value: float,
) -> list[dict]:
    """Return recent YOLO tracks for dropout-tolerant overlap review.

    The hold time is intentionally much shorter than SCENE_TRACK_HOLD_SEC: it
    bridges one or two missed detections without allowing stale overlap alerts.
    """
    rows: list[dict] = []
    for track in session.tracks:
        age = float(now_value) - float(track.last_seen)
        if age > max(0.2, PAIRWISE_TRACK_HOLD_SEC):
            continue
        if int(track.hits) < MIN_TRACK_HITS_FOR_NARRATION:
            continue
        rows.append({
            **dict(track.metadata or {}),
            "id": int(track.track_id),
            "label": str(track.label),
            "bbox": [float(v) for v in track.box],
            "confidence": float(track.confidence),
            "track_hits": int(track.hits),
            "frame_width": int(width),
            "frame_height": int(height),
            "last_seen_age_sec": round(max(0.0, age), 3),
        })
    return sorted(rows, key=lambda x: (canonical(x.get("label")), int(x.get("id") or 0)))


_CONTAINER_LIKE_LABELS = {
    "box", "carton", "container", "plastic container", "tray", "bowl",
    "cup", "bag", "bin", "cabinet", "drawer", "basket", "package",
    "bedside table", "bedside cabinet",
}


def plausible_inside_relation(subject: dict, outer: dict) -> bool:
    outer_label = canonical(outer.get("label"))
    subject_label = canonical(subject.get("label"))

    # Bottles, towels, bandages, saline packs, etc. are not treated as containers
    # merely because a large YOLO box geometrically surrounds another box.
    if outer_label not in _CONTAINER_LIKE_LABELS:
        return False
    if subject_label == outer_label:
        return False

    outer_area = box_area(outer["bbox"])
    subject_area = box_area(subject["bbox"])
    if outer_area <= 0 or subject_area <= 0:
        return False

    # The outer region must be meaningfully larger than the inner object.
    return outer_area >= subject_area * 1.45


def relation_candidates(objects: list[dict], image_width: int, image_height: int) -> list[dict]:
    rows: list[dict] = []
    diagonal = max(1.0, math.hypot(image_width, image_height))

    for i, a in enumerate(objects):
        acx, acy = center(a["bbox"])
        aw = max(1.0, a["bbox"][2] - a["bbox"][0])
        ah = max(1.0, a["bbox"][3] - a["bbox"][1])

        for j, b in enumerate(objects):
            if i == j:
                continue
            bcx, bcy = center(b["bbox"])
            bw = max(1.0, b["bbox"][2] - b["bbox"][0])
            bh = max(1.0, b["bbox"][3] - b["bbox"][1])
            distance = math.hypot(acx - bcx, acy - bcy)
            normalized = distance / diagonal
            pair_scale = max(1.0, (aw + ah + bw + bh) / 4.0)

            if (
                containment_ratio(a["bbox"], b["bbox"]) >= 0.88
                and plausible_inside_relation(a, b)
            ):
                rows.append({
                    "subject_id": a["id"],
                    "relation": "inside",
                    "object_id": b["id"],
                    "confidence": 0.94,
                })
                continue

            overlap = iou(a["bbox"], b["bbox"])
            if overlap >= 0.18:
                rows.append({
                    "subject_id": a["id"],
                    "relation": "overlapping",
                    "object_id": b["id"],
                    "confidence": round(clamp(0.6 + overlap, 0, 0.98), 3),
                })

            if distance <= pair_scale * 1.35 or normalized <= 0.13:
                rows.append({
                    "subject_id": a["id"],
                    "relation": "near",
                    "object_id": b["id"],
                    "confidence": round(clamp(1.0 - normalized * 2.4, 0.55, 0.97), 3),
                })

            dx = bcx - acx
            dy = bcy - acy
            if abs(dx) >= max(aw, bw) * 0.45:
                rows.append({
                    "subject_id": a["id"],
                    "relation": "left_of" if dx > 0 else "right_of",
                    "object_id": b["id"],
                    "confidence": round(clamp(abs(dx) / image_width + 0.55, 0.55, 0.95), 3),
                })
            if abs(dy) >= max(ah, bh) * 0.45:
                rows.append({
                    "subject_id": a["id"],
                    "relation": "above" if dy > 0 else "below",
                    "object_id": b["id"],
                    "confidence": round(clamp(abs(dy) / image_height + 0.55, 0.55, 0.95), 3),
                })

    seen = set()
    unique = []
    for row in sorted(rows, key=lambda r: r["confidence"], reverse=True):
        key = (row["subject_id"], row["relation"], row["object_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) >= 18:
            break
    return unique


def _intersection_area(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_edge_gap(a: list[float], b: list[float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def _expanded_box(box: list[float], pad: float, width: int, height: int) -> list[float]:
    return [
        max(0.0, float(box[0]) - pad),
        max(0.0, float(box[1]) - pad),
        min(float(width), float(box[2]) + pad),
        min(float(height), float(box[3]) + pad),
    ]



def _finite_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def object_camera_depth_m(obj: dict) -> float | None:
    """Return the best available camera-forward distance in metres."""
    for key in ("distance_m", "depth_m", "camera_depth_m", "z_m"):
        number = _finite_positive_float(obj.get(key))
        if number is not None:
            return number
    camera_xyz = obj.get("camera_xyz") or obj.get("camera_point")
    if isinstance(camera_xyz, (list, tuple)) and len(camera_xyz) >= 3:
        number = _finite_positive_float(camera_xyz[2])
        if number is not None:
            return number
    return None


def _relation_front_id(relation: str, a_id: int, b_id: int) -> int | None:
    relation = canonical(relation or "").replace(" ", "_")
    if relation in {"a_in_front_of_b", "a_blocks_b", "a_on_b", "a_covers_b"}:
        return a_id
    if relation in {"b_in_front_of_a", "b_blocks_a", "b_on_a", "b_covers_a"}:
        return b_id
    return None


def _front_token_id(value: Any, a_id: int, b_id: int) -> int | None:
    token = canonical(value or "").replace(" ", "_")
    if token in {"a", "object_a", "物件a", str(a_id)}:
        return a_id
    if token in {"b", "object_b", "物件b", str(b_id)}:
        return b_id
    return None

def overlapping_pair_candidates(
    objects: list[dict],
    image_width: int,
    image_height: int,
) -> list[dict]:
    """Return suspicious pairs among objects already boxed by YOLO.

    The prefilter accepts either:
      1. meaningful raw bbox intersection; or
      2. a tiny visible edge gap that disappears after conservative dilation.

    Case (2) is important when the front object hides the part of the rear object
    that YOLO would otherwise include in its bbox.  It is only a request for VLM
    review; it never becomes an alert without visual confirmation.
    """
    rows: list[dict] = []
    frame_scale = max(0.5, max(float(image_width), float(image_height)) / 1280.0)
    base_gap_px = max(3.0, PAIRWISE_NEAR_GAP_PX * frame_scale)
    expand_cap_px = max(4.0, PAIRWISE_EXPAND_MAX_PX * frame_scale)

    for i in range(len(objects)):
        a = objects[i]
        abox = extract_box(a)
        if abox is None:
            continue
        for j in range(i + 1, len(objects)):
            b = objects[j]
            bbox = extract_box(b)
            if bbox is None:
                continue

            area_a = max(1.0, box_area(abox))
            area_b = max(1.0, box_area(bbox))
            inter = _intersection_area(abox, bbox)
            smaller_coverage = inter / min(area_a, area_b)
            overlap_iou = iou(abox, bbox)
            edge_gap = _box_edge_gap(abox, bbox)

            short_a = max(1.0, min(abox[2] - abox[0], abox[3] - abox[1]))
            short_b = max(1.0, min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            adaptive_pad = min(
                expand_cap_px,
                max(3.0, min(short_a, short_b) * PAIRWISE_EXPAND_RATIO),
            )
            expanded_a = _expanded_box(abox, adaptive_pad, image_width, image_height)
            expanded_b = _expanded_box(bbox, adaptive_pad, image_width, image_height)
            expanded_inter = _intersection_area(expanded_a, expanded_b)
            near_gap_limit = max(base_gap_px, adaptive_pad * 1.15)

            raw_overlap = inter > 0.0 and (
                smaller_coverage >= PAIRWISE_OVERLAP_MIN_SMALLER_COVERAGE
                or overlap_iou >= PAIRWISE_OVERLAP_MIN_IOU
            )
            edge_contact = (
                edge_gap <= near_gap_limit
                and expanded_inter > 0.0
            )
            if not raw_overlap and not edge_contact:
                continue

            trigger_type = "box_overlap" if raw_overlap else "near_edge_contact"
            # Raw overlap receives the strongest rank. Near-edge candidates are
            # ranked by how close their visible boundaries are.
            proximity_score = max(0.0, 1.0 - edge_gap / max(1.0, near_gap_limit))
            score = (
                smaller_coverage * 2.2
                + overlap_iou * 1.4
                + proximity_score * (0.55 if raw_overlap else 0.38)
            )
            id_a, id_b = int(a["id"]), int(b["id"])
            depth_a = object_camera_depth_m(a)
            depth_b = object_camera_depth_m(b)
            depth_front_id = None
            depth_delta_m = None
            if depth_a is not None and depth_b is not None:
                depth_delta_m = abs(depth_a - depth_b)
                if depth_delta_m >= PAIRWISE_DEPTH_FRONT_MIN_DELTA_M:
                    depth_front_id = id_a if depth_a < depth_b else id_b
            rows.append({
                "pair_key": f"{min(id_a, id_b)}:{max(id_a, id_b)}",
                "object_a_id": id_a,
                "object_b_id": id_b,
                "object_a_label": canonical(a.get("label")),
                "object_b_label": canonical(b.get("label")),
                "object_a_name_zh": label_zh(a.get("label")),
                "object_b_name_zh": label_zh(b.get("label")),
                "object_a_bbox": [round(float(v), 2) for v in abox],
                "object_b_bbox": [round(float(v), 2) for v in bbox],
                "geometry_trigger_type": trigger_type,
                "bbox_iou": round(overlap_iou, 4),
                "smaller_box_coverage": round(smaller_coverage, 4),
                "intersection_area_px": round(inter, 2),
                "edge_gap_px": round(edge_gap, 2),
                "expanded_intersection_area_px": round(expanded_inter, 2),
                "candidate_score": round(score, 4),
                "object_a_depth_m": round(depth_a, 4) if depth_a is not None else None,
                "object_b_depth_m": round(depth_b, 4) if depth_b is not None else None,
                "depth_delta_m": round(depth_delta_m, 4) if depth_delta_m is not None else None,
                "depth_front_id": depth_front_id,
            })

    rows.sort(
        key=lambda row: (
            1 if row.get("geometry_trigger_type") == "box_overlap" else 0,
            float(row.get("candidate_score") or 0.0),
            -float(row.get("edge_gap_px") or 0.0),
        ),
        reverse=True,
    )
    return rows[:max(1, PAIRWISE_PICKUP_MAX_PAIRS)]


def stabilize_pairwise_candidates(
    session: SessionState,
    candidates: list[dict],
    *,
    now_value: float,
) -> list[dict]:
    """Require suspicious geometry to persist before spending a Qwen call.

    One missing/intersecting frame is treated as bbox jitter. A short gap is
    tolerated so a detector wobble does not reset a genuine contact pair.
    """
    current_by_key = {str(row.get("pair_key")): row for row in candidates}
    accepted: list[dict] = []
    with _SESSION_LOCK:
        evidence = session.pairwise_candidate_evidence
        for key, row in current_by_key.items():
            prev = evidence.get(key) or {}
            last_seen = float(prev.get("last_seen") or 0.0)
            streak = int(prev.get("streak") or 0)
            if now_value - last_seen <= PAIRWISE_EVIDENCE_TTL_SEC:
                streak += 1
            else:
                streak = 1
            evidence[key] = {
                "streak": streak,
                "last_seen": now_value,
                "trigger_type": row.get("geometry_trigger_type"),
            }
            required = max(1, PAIRWISE_REQUIRED_FRAMES)
            if streak >= required:
                stable = dict(row)
                stable["evidence_frames"] = streak
                accepted.append(stable)
                if streak == required:
                    print(
                        "[overlap-prefilter] accepted "
                        f"pair={key} trigger={row.get('geometry_trigger_type')} "
                        f"gap={row.get('edge_gap_px')}px frames={streak}",
                        flush=True,
                    )

        stale_keys = [
            key
            for key, value in evidence.items()
            if now_value - float(value.get("last_seen") or 0.0)
                > PAIRWISE_EVIDENCE_TTL_SEC
        ]
        for key in stale_keys:
            evidence.pop(key, None)

    accepted.sort(
        key=lambda row: (
            int(row.get("evidence_frames") or 0),
            1 if row.get("geometry_trigger_type") == "box_overlap" else 0,
            float(row.get("candidate_score") or 0.0),
        ),
        reverse=True,
    )
    return accepted[:max(1, PAIRWISE_PICKUP_MAX_PAIRS)]

def pairwise_candidate_fingerprint(candidates: list[dict]) -> str:
    # Track IDs and class labels are stable enough for one scene. Do not include
    # live bbox metrics here: detector jitter would otherwise retrigger Qwen.
    return "|".join(
        sorted(
            f"{row.get('pair_key')}:{row.get('object_a_label')}:{row.get('object_b_label')}"
            for row in candidates
        )
    )


def crop_pair_for_qwen(
    image: Image.Image,
    candidate: dict,
    *,
    pad_ratio: float = 0.22,
) -> Image.Image:
    boxes = [candidate["object_a_bbox"], candidate["object_b_bbox"]]
    x1 = min(float(box[0]) for box in boxes)
    y1 = min(float(box[1]) for box in boxes)
    x2 = max(float(box[2]) for box in boxes)
    y2 = max(float(box[3]) for box in boxes)
    union_w = max(1.0, x2 - x1)
    union_h = max(1.0, y2 - y1)
    pad = max(18.0, max(union_w, union_h) * pad_ratio)
    left = max(0, int(math.floor(x1 - pad)))
    top = max(0, int(math.floor(y1 - pad)))
    right = min(image.width, int(math.ceil(x2 + pad)))
    bottom = min(image.height, int(math.ceil(y2 + pad)))
    crop = image.crop((left, top, max(left + 1, right), max(top + 1, bottom))).convert("RGB")

    draw = ImageDraw.Draw(crop)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            22,
        )
    except Exception:
        font = ImageFont.load_default()

    palette = [((80, 235, 255), "A"), ((255, 185, 80), "B")]
    for index, key in enumerate(("object_a_bbox", "object_b_bbox")):
        box = candidate[key]
        local = [
            int(round(float(box[0]) - left)),
            int(round(float(box[1]) - top)),
            int(round(float(box[2]) - left)),
            int(round(float(box[3]) - top)),
        ]
        color, token = palette[index]
        draw.rectangle(local, outline=color, width=5)
        object_id_key = "object_a_id" if index == 0 else "object_b_id"
        text = f"{token} #{candidate[object_id_key]}"
        tx, ty = max(2, local[0]), max(2, local[1] - 28)
        text_box = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(text_box, fill=(5, 15, 22))
        draw.text((tx, ty), text, fill=color, font=font)
    return crop


def crop_pair_clean_for_qwen(
    image: Image.Image,
    candidate: dict,
    *,
    pad_ratio: float = 0.24,
) -> Image.Image:
    """Unannotated context crop so box strokes do not hide the occlusion cue."""
    boxes = [candidate["object_a_bbox"], candidate["object_b_bbox"]]
    x1 = min(float(box[0]) for box in boxes)
    y1 = min(float(box[1]) for box in boxes)
    x2 = max(float(box[2]) for box in boxes)
    y2 = max(float(box[3]) for box in boxes)
    union_w = max(1.0, x2 - x1)
    union_h = max(1.0, y2 - y1)
    pad = max(20.0, max(union_w, union_h) * pad_ratio)
    left = max(0, int(math.floor(x1 - pad)))
    top = max(0, int(math.floor(y1 - pad)))
    right = min(image.width, int(math.ceil(x2 + pad)))
    bottom = min(image.height, int(math.ceil(y2 + pad)))
    return image.crop((left, top, max(left + 1, right), max(top + 1, bottom))).convert("RGB")


def crop_pair_contact_zone_for_qwen(
    image: Image.Image,
    candidate: dict,
) -> Image.Image:
    """Tight clean crop around the overlap / nearest-boundary region."""
    a = [float(v) for v in candidate["object_a_bbox"]]
    b = [float(v) for v in candidate["object_b_bbox"]]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 > ix1 and iy2 > iy1:
        cx, cy = (ix1 + ix2) * 0.5, (iy1 + iy2) * 0.5
        base_w, base_h = max(24.0, ix2 - ix1), max(24.0, iy2 - iy1)
    else:
        # Midpoint between the nearest visible box boundaries.
        ax = min(max((b[0] + b[2]) * 0.5, a[0]), a[2])
        ay = min(max((b[1] + b[3]) * 0.5, a[1]), a[3])
        bx = min(max((a[0] + a[2]) * 0.5, b[0]), b[2])
        by = min(max((a[1] + a[3]) * 0.5, b[1]), b[3])
        cx, cy = (ax + bx) * 0.5, (ay + by) * 0.5
        base_w, base_h = max(30.0, abs(ax - bx)), max(30.0, abs(ay - by))
    object_scale = max(
        40.0,
        min(
            max(a[2] - a[0], a[3] - a[1]),
            max(b[2] - b[0], b[3] - b[1]),
        ),
    )
    half_w = max(base_w * 1.8, object_scale * 0.48)
    half_h = max(base_h * 1.8, object_scale * 0.48)
    left = max(0, int(math.floor(cx - half_w)))
    top = max(0, int(math.floor(cy - half_h)))
    right = min(image.width, int(math.ceil(cx + half_w)))
    bottom = min(image.height, int(math.ceil(cy + half_h)))
    return image.crop((left, top, max(left + 1, right), max(top + 1, bottom))).convert("RGB")


PAIRWISE_PICKUP_SYSTEM_PROMPT = """你是機器人夾取前的雙物件「可見遮擋與夾取干涉」稽核員。
每組候選都由兩個已被 YOLO 分別框出的物件構成：青色框 A、橘色框 B。
系統會依序提供：乾淨裁切圖、A/B 標記圖、兩物件交界區放大圖。

重要定義：
- 警示目標不是只找物理接觸或上下堆疊。
- 只要一個物件位於另一物件前方，遮住其可見輪廓、抓取面、邊緣、把手或夾爪接近路徑，即屬於夾取風險。
- 即使兩物件沒有直接接觸，只要前方物件造成可見遮擋或夾取干涉，也要 overlap_possible=true。
- 「前方」是更靠近相機的一方，不是畫面位置較低、框較大、或標記字母較前的一方。
- 看 T 字交界時：輪廓連續跨過交界的一方通常在前；輪廓在另一物件邊界處被截斷的一方通常在後。
- 若候選資料提供 depth_front_id，代表 D405 深度顯示該物件更靠近相機；除非深度明顯取樣失敗，方向判斷應與它一致。

判斷規則：
1. 若 A 的可見輪廓在 B 邊界處中斷、消失，或 B 擋住 A 的主要夾取區，輸出 relation=b_in_front_of_a 或 b_blocks_a，通常先處理 B。
2. 若 B 的可見輪廓在 A 邊界處中斷、消失，或 A 擋住 B 的主要夾取區，輸出 relation=a_in_front_of_b 或 a_blocks_b，通常先處理 A。
3. 明確上下堆疊、覆蓋或容器內外，也應輸出 true，並給出安全先後順序。
4. 只有真正並排、留有清楚間隙、框線抖動、印刷圖案相交，或無法看出任何輪廓遮擋時，才輸出 false。
5. bbox 幾何只能協助定位，不能單獨決定結果；必須查看乾淨圖與交界放大圖。
6. 必須另外填 front_object 與 blocked_object；兩者無法可靠判定時填 uncertain，不可猜測。
7. before_id 應是需要先移開的前方／阻擋物件；若方向仍不可靠，before_id/after_id=null。
8. confidence 表示你對「會影響安全夾取」的信心；direction_confidence 才表示你對前後方向與先後順序的信心。
9. 有清楚 T 字交界、輪廓截斷或可靠深度時，direction_confidence 可為 0.75～0.95；只有框接近但方向模糊時應低於 0.70。
10. 使用繁體中文，reason 最多 36 個中文字，並明確寫出哪個輪廓被哪個物件截斷。
11. 嚴格輸出 JSON，不加 Markdown。

輸出格式：
{
  "pair_results": [
    {
      "pair_key": "1:2",
      "object_a_id": 1,
      "object_b_id": 2,
      "overlap_possible": true,
      "relation": "a_in_front_of_b | b_in_front_of_a | a_blocks_b | b_blocks_a | a_on_b | b_on_a | a_covers_b | b_covers_a | a_inside_b | b_inside_a | adjacent_only | no_visible_occlusion | uncertain",
      "front_object": "A | B | uncertain",
      "blocked_object": "A | B | none | uncertain",
      "before_id": 2,
      "after_id": 1,
      "confidence": 0.86,
      "direction_confidence": 0.88,
      "reason": "A 的下緣在 B 上邊界處中斷，B 位於前方"
    }
  ]
}
"""


def call_qwen_pairwise_pickup(
    image: Image.Image,
    candidates: list[dict],
) -> dict:
    content: list[dict] = [{
        "type": "text",
        "text": json.dumps({
            "candidate_pairs": [
                {
                    key: row.get(key)
                    for key in (
                        "pair_key",
                        "object_a_id",
                        "object_b_id",
                        "object_a_name_zh",
                        "object_b_name_zh",
                        "geometry_trigger_type",
                        "bbox_iou",
                        "smaller_box_coverage",
                        "intersection_area_px",
                        "edge_gap_px",
                        "expanded_intersection_area_px",
                        "evidence_frames",
                        "object_a_depth_m",
                        "object_b_depth_m",
                        "depth_delta_m",
                        "depth_front_id",
                    )
                }
                for row in candidates
            ],
            "instruction": "依序判斷每張 A/B 裁切圖。幾何資料只用來說明為何送審，不能當成實體重疊證據；只有看見實體遮擋時才產生警示，並在可靠時給出夾取先後。",
        }, ensure_ascii=False),
    }]
    for row in candidates:
        content.append({
            "type": "text",
            "text": (
                f"pair_key={row['pair_key']}；"
                f"A=#{row['object_a_id']} {row['object_a_name_zh']}；"
                f"B=#{row['object_b_id']} {row['object_b_name_zh']}"
            ),
        })
        content.append({
            "type": "text",
            "text": "圖 1：乾淨裁切，先觀察實際輪廓與前後遮擋。",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_data_url(crop_pair_clean_for_qwen(image, row), max_edge=PAIRWISE_IMAGE_EDGE),
            },
        })
        content.append({
            "type": "text",
            "text": "圖 2：A/B 框僅用於對應物件身分。",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_data_url(crop_pair_for_qwen(image, row), max_edge=PAIRWISE_IMAGE_EDGE),
            },
        })
        content.append({
            "type": "text",
            "text": "圖 3：交界區放大，檢查 T 字交界、輪廓截斷與夾取區遮擋。",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_data_url(crop_pair_contact_zone_for_qwen(image, row), max_edge=PAIRWISE_CONTACT_IMAGE_EDGE),
            },
        })

    payload = {
        "model": QWEN_MODEL,
        "temperature": 0.10,
        "top_p": 0.75,
        "max_tokens": PAIRWISE_MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": PAIRWISE_PICKUP_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    }
    req = urllib.request.Request(
        QWEN_BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(req, timeout=max(QWEN_TIMEOUT_SEC, 90.0)) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    return _parse_qwen_json_content(data["choices"][0]["message"]["content"])


def clean_pairwise_pickup_result(
    raw: dict,
    candidates: list[dict],
) -> dict:
    by_key = {str(row["pair_key"]): row for row in candidates}
    accepted: list[dict] = []
    reviewed: list[dict] = []

    rows = raw.get("pair_results") if isinstance(raw, dict) else []
    if not isinstance(rows, list):
        rows = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        pair_key = str(item.get("pair_key") or "")
        source = by_key.get(pair_key)
        if source is None:
            continue
        valid_ids = {int(source["object_a_id"]), int(source["object_b_id"])}
        try:
            before_id = int(item["before_id"]) if item.get("before_id") is not None else None
            after_id = int(item["after_id"]) if item.get("after_id") is not None else None
        except Exception:
            before_id = after_id = None
        try:
            confidence = clamp(float(item.get("confidence") or 0.0), 0.0, 1.0)
        except Exception:
            confidence = 0.0
        try:
            direction_confidence = clamp(
                float(item.get("direction_confidence") or 0.0), 0.0, 1.0
            )
        except Exception:
            direction_confidence = 0.0
        relation = canonical(item.get("relation") or "uncertain").replace(" ", "_")
        raw_overlap = item.get("overlap_possible", False)
        if isinstance(raw_overlap, str):
            overlap_possible = raw_overlap.strip().lower() in {
                "1", "true", "yes", "y", "是", "有", "成立"
            }
        else:
            overlap_possible = bool(raw_overlap)

        a_id = int(source["object_a_id"])
        b_id = int(source["object_b_id"])
        relation_front_id = _relation_front_id(relation, a_id, b_id)
        explicit_front_id = _front_token_id(item.get("front_object"), a_id, b_id)
        explicit_blocked_id = _front_token_id(item.get("blocked_object"), a_id, b_id)
        depth_front_id = source.get("depth_front_id")
        try:
            depth_front_id = int(depth_front_id) if depth_front_id is not None else None
        except Exception:
            depth_front_id = None

        order_source = "unconfirmed"
        direction_conflict = False

        # D405 depth is the strongest available front/back cue. Smaller camera
        # distance means closer to the camera. It overrides an inverted VLM
        # direction and is logged explicitly for auditability.
        if depth_front_id in valid_ids:
            rear_id = b_id if depth_front_id == a_id else a_id
            if relation_front_id is not None and relation_front_id != depth_front_id:
                direction_conflict = True
                print(
                    "[pairwise-pickup][direction-conflict] "
                    f"pair={pair_key} vlm_front={relation_front_id} "
                    f"depth_front={depth_front_id} "
                    f"depth_delta={source.get('depth_delta_m')}",
                    flush=True,
                )
            before_id, after_id = depth_front_id, rear_id
            relation = (
                "a_in_front_of_b" if depth_front_id == a_id
                else "b_in_front_of_a"
            )
            order_source = "d405_depth"
            item["front_object"] = "A" if depth_front_id == a_id else "B"
            item["blocked_object"] = "B" if depth_front_id == a_id else "A"
        else:
            # VLM-only ordering is accepted only when three independent fields
            # agree: relation, front_object and blocked_object. This prevents a
            # single A/B inversion from becoming an executable sequence.
            expected_blocked = (
                b_id if explicit_front_id == a_id
                else a_id if explicit_front_id == b_id
                else None
            )
            direction_consistent = (
                relation_front_id in valid_ids
                and explicit_front_id == relation_front_id
                and explicit_blocked_id == expected_blocked
            )
            if (
                PAIRWISE_ALLOW_VLM_ONLY_ORDER
                and direction_consistent
                and direction_confidence >= PAIRWISE_DIRECTION_MIN_CONF
            ):
                before_id = explicit_front_id
                after_id = expected_blocked
                order_source = "vlm_consensus"
            else:
                before_id = after_id = None
                if overlap_possible:
                    print(
                        "[pairwise-pickup][order-rejected] "
                        f"pair={pair_key} relation={relation} "
                        f"front={item.get('front_object')} "
                        f"blocked={item.get('blocked_object')} "
                        f"direction_conf={direction_confidence:.3f}",
                        flush=True,
                    )

        reason = localize_narration_text(item.get("reason"))[:90]
        if direction_conflict:
            reason = "D405 深度與視覺方向判斷衝突，已依深度修正安全順序"
        elif order_source == "unconfirmed" and overlap_possible:
            reason = "交界處存在可見遮擋或夾取干涉，但前後方向尚未可靠確認"

        row = {
            "pair_key": pair_key,
            "object_a_id": int(source["object_a_id"]),
            "object_b_id": int(source["object_b_id"]),
            "object_a_name_zh": source["object_a_name_zh"],
            "object_b_name_zh": source["object_b_name_zh"],
            "overlap_possible": overlap_possible,
            "relation": relation,
            "before_id": before_id,
            "after_id": after_id,
            "confidence": round(confidence, 3),
            "direction_confidence": round(direction_confidence, 3),
            "front_object": item.get("front_object"),
            "blocked_object": item.get("blocked_object"),
            "order_source": order_source,
            "direction_conflict": direction_conflict,
            "reason": reason,
            "geometry_trigger": {
                "trigger_type": source.get("geometry_trigger_type"),
                "bbox_iou": source["bbox_iou"],
                "smaller_box_coverage": source["smaller_box_coverage"],
                "intersection_area_px": source["intersection_area_px"],
                "edge_gap_px": source.get("edge_gap_px"),
                "evidence_frames": source.get("evidence_frames"),
            },
        }
        reviewed.append(row)

        confirmed_overlap = (
            overlap_possible
            and confidence >= PAIRWISE_PICKUP_MIN_CONF
            and relation not in {
                "adjacent_only",
                "no_visible_occlusion",
                "touching_or_adjacent",
                "perspective_overlap",
                "uncertain",
            }
        )
        valid_order = (
            confirmed_overlap
            and before_id in valid_ids
            and after_id in valid_ids
            and before_id != after_id
            and order_source in {"d405_depth", "vlm_consensus"}
        )
        if confirmed_overlap:
            row["order_confirmed"] = bool(valid_order)
            accepted.append(row)

    return {
        "source": "qwen_overlap_alert",
        "reviewed_pairs": reviewed,
        "overlap_alerts": accepted,
        # Compatibility field for a future executor. UI behavior is alert-only.
        "pickup_order": [row for row in accepted if row.get("order_confirmed")],
    }


def pairwise_result_line(result: dict) -> dict | None:
    alerts = result.get("overlap_alerts") if isinstance(result, dict) else []
    if not isinstance(alerts, list) or not alerts:
        return None
    row = max(alerts, key=lambda x: float(x.get("confidence") or 0.0))
    names = {
        int(row["object_a_id"]): row["object_a_name_zh"],
        int(row["object_b_id"]): row["object_b_name_zh"],
    }
    object_a_name = names.get(int(row["object_a_id"]), "物件 A")
    object_b_name = names.get(int(row["object_b_id"]), "物件 B")
    reason = str(row.get("reason") or "可見重疊可能影響夾取").strip()
    reason = re.sub(r"(?<![A-Za-z])A(?![A-Za-z])", f"「{object_a_name}」", reason)
    reason = re.sub(r"(?<![A-Za-z])B(?![A-Za-z])", f"「{object_b_name}」", reason)
    reason = re.sub(r"」\s+", "」", reason)
    reason = re.sub(r"\s+「", "「", reason)
    if reason.endswith("。"):
        reason = reason[:-1]

    before_id = row.get("before_id")
    after_id = row.get("after_id")
    if row.get("order_confirmed") and before_id is not None and after_id is not None:
        before_id = int(before_id)
        after_id = int(after_id)
        before_name = names.get(before_id, f"物件 {before_id}")
        after_name = names.get(after_id, f"物件 {after_id}")
        text = (
            f"夾取風險：「{object_a_name}」與「{object_b_name}」存在可見遮擋或夾取干涉。"
            f"建議先處理「{before_name}」，再處理「{after_name}」。{reason}。"
        )
    else:
        text = (
            f"重疊風險：「{object_a_name}」與「{object_b_name}」存在可見遮擋或夾取干涉，"
            f"但目前無法可靠判定先後；夾取前請重新觀察。{reason}。"
        )

    return {
        "type": "overlap_alert",
        "text": text,
        "before_id": before_id,
        "after_id": after_id,
        "confidence": row.get("confidence"),
        "pair_key": row.get("pair_key"),
    }

def start_pairwise_pickup_audit(
    session: SessionState,
    *,
    signature: str,
    fingerprint: str,
    image: Image.Image,
    candidates: list[dict],
) -> bool:
    if not PAIRWISE_PICKUP_ENABLED or not candidates or not fingerprint:
        return False
    now = time.time()
    with _SESSION_LOCK:
        if session.pairwise_pending:
            return False
        if now - session.last_scene_change_at < PAIRWISE_PICKUP_DELAY_SEC:
            return False
        if session.pairwise_latched and session.pairwise_fingerprint == fingerprint:
            return False
        if session.pairwise_fingerprint != fingerprint:
            session.pairwise_result = {}
            session.pairwise_ready_line = None
            session.pairwise_error = ""
            session.pairwise_attempt_count = 0
            session.pairwise_negative_until = 0.0
            session.pairwise_latched = False
            session.pairwise_fingerprint = fingerprint
        if session.pairwise_attempt_count >= max(1, PAIRWISE_MAX_REVIEW_ATTEMPTS):
            if now < session.pairwise_negative_until:
                return False
            session.pairwise_attempt_count = 0
        if now < session.pairwise_negative_until:
            return False
        if now - session.pairwise_checked_at < PAIRWISE_PICKUP_RETRY_SEC:
            return False

    if not _QWEN_WORK_LOCK.acquire(blocking=False):
        return False

    with _SESSION_LOCK:
        if session.pairwise_pending or session.pairwise_fingerprint != fingerprint:
            _QWEN_WORK_LOCK.release()
            return False
        session.pairwise_job_token += 1
        token = session.pairwise_job_token
        session.pairwise_pending = True
        session.pairwise_checked_at = now
        session.pairwise_signature = signature
        session.pairwise_attempt_count += 1
        attempt = session.pairwise_attempt_count
        session.pairwise_error = ""

    image_copy = image.copy()
    candidate_copy = json.loads(json.dumps(candidates))

    def worker() -> None:
        try:
            raw = call_qwen_pairwise_pickup(image_copy, candidate_copy)
            cleaned = clean_pairwise_pickup_result(raw, candidate_copy)
            alerts = cleaned.get("overlap_alerts") or []
            ready = pairwise_result_line(cleaned)
            with _SESSION_LOCK:
                if (
                    session.pairwise_job_token == token
                    and session.pairwise_fingerprint == fingerprint
                ):
                    if alerts:
                        session.pairwise_result = cleaned
                        session.pairwise_ready_line = ready
                        session.pairwise_latched = True
                        session.pairwise_negative_until = 0.0
                    elif not session.pairwise_latched:
                        session.pairwise_result = cleaned
                        session.pairwise_ready_line = None
                        if attempt >= max(1, PAIRWISE_MAX_REVIEW_ATTEMPTS):
                            session.pairwise_negative_until = (
                                time.time() + PAIRWISE_NEGATIVE_COOLDOWN_SEC
                            )
                        else:
                            session.pairwise_negative_until = (
                                time.time() + PAIRWISE_PICKUP_RETRY_SEC
                            )
                    session.pairwise_error = ""
            next_state = (
                "latched"
                if alerts
                else (
                    f"cooldown={PAIRWISE_NEGATIVE_COOLDOWN_SEC:.0f}s"
                    if attempt >= max(1, PAIRWISE_MAX_REVIEW_ATTEMPTS)
                    else f"retry={PAIRWISE_PICKUP_RETRY_SEC:.0f}s"
                )
            )
            reviewed_rows = cleaned.get("reviewed_pairs") or []
            for review in reviewed_rows:
                print(
                    "[pairwise-pickup][review] "
                    f"pair={review.get('pair_key')} "
                    f"overlap={review.get('overlap_possible')} "
                    f"relation={review.get('relation')} "
                    f"confidence={review.get('confidence')} "
                    f"before={review.get('before_id')} after={review.get('after_id')} "
                    f"reason={review.get('reason')}",
                    flush=True,
                )
            print(
                "[pairwise-pickup] "
                f"fingerprint={fingerprint} "
                f"attempt={attempt}/{max(1, PAIRWISE_MAX_REVIEW_ATTEMPTS)} "
                f"pairs={len(cleaned.get('reviewed_pairs') or [])} "
                f"alerts={len(alerts)} state={next_state}",
                flush=True,
            )
        except Exception as exc:
            with _SESSION_LOCK:
                if session.pairwise_job_token == token:
                    session.pairwise_error = f"{type(exc).__name__}: {exc}"
                    session.pairwise_negative_until = (
                        time.time() + PAIRWISE_PICKUP_RETRY_SEC
                    )
            print(f"[pairwise-pickup][WARN] {type(exc).__name__}: {exc}", flush=True)
        finally:
            with _SESSION_LOCK:
                if session.pairwise_job_token == token:
                    session.pairwise_pending = False
            _QWEN_WORK_LOCK.release()

    threading.Thread(
        target=worker,
        name=f"scene-narrator-pairwise-{token}",
        daemon=True,
    ).start()
    return True

def scene_signature(objects: list[dict], width: int, height: int) -> str:
    """ID-independent semantic signature.

    Track IDs are implementation details and may change after a detector miss.
    A scene with one cotton swab on the right must keep the same signature even
    if its internal track ID changes from 4 to 9.
    """
    rows = []
    for obj in objects:
        cx, cy = center(obj["bbox"])
        rows.append([
            canonical(obj.get("label")),
            round(cx / max(1, width), 1),
            round(cy / max(1, height), 1),
        ])
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def scene_snapshot(objects: list[dict], width: int, height: int) -> list[dict]:
    """Normalized ID-independent state used for semantic change detection."""
    rows = []
    frame_area = max(1.0, float(width * height))
    for obj in objects:
        box = [float(v) for v in obj["bbox"]]
        cx, cy = center(box)
        rows.append({
            "label": canonical(obj.get("label")),
            "bbox": box,
            "cx": cx / max(1.0, float(width)),
            "cy": cy / max(1.0, float(height)),
            "area_ratio": box_area(box) / frame_area,
        })
    return sorted(rows, key=lambda row: (row["label"], row["cx"], row["cy"]))


def _group_snapshot_by_label(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows or []:
        grouped.setdefault(str(row.get("label") or "object"), []).append(row)
    for label in grouped:
        grouped[label].sort(key=lambda x: (float(x.get("cx") or 0), float(x.get("cy") or 0)))
    return grouped


def scene_state_changed(
    previous: list[dict],
    current: list[dict],
    *,
    width: int,
    height: int,
) -> tuple[bool, list[str]]:
    """Detect real semantic changes without depending on volatile track IDs."""
    reasons: list[str] = []
    prev_groups = _group_snapshot_by_label(previous)
    curr_groups = _group_snapshot_by_label(current)

    all_labels = sorted(set(prev_groups) | set(curr_groups))
    for label in all_labels:
        old_rows = list(prev_groups.get(label) or [])
        new_rows = list(curr_groups.get(label) or [])
        if len(old_rows) != len(new_rows):
            reasons.append(f"count_changed:{label}:{len(old_rows)}->{len(new_rows)}")
            continue

        # Greedy nearest matching within each class.  This survives track-ID
        # regeneration and two same-class objects swapping detector IDs.
        unmatched = set(range(len(new_rows)))
        for old in old_rows:
            if not unmatched:
                break
            best_index = min(
                unmatched,
                key=lambda idx: math.hypot(
                    float(new_rows[idx]["cx"]) - float(old["cx"]),
                    float(new_rows[idx]["cy"]) - float(old["cy"]),
                ),
            )
            new = new_rows[best_index]
            unmatched.remove(best_index)

            overlap = iou(old["bbox"], new["bbox"])
            center_shift = math.hypot(
                float(new["cx"]) - float(old["cx"]),
                float(new["cy"]) - float(old["cy"]),
            )
            old_area = max(1e-6, float(old["area_ratio"]))
            new_area = max(1e-6, float(new["area_ratio"]))
            area_change = max(old_area, new_area) / min(old_area, new_area)

            # Only meaningful movement invalidates Qwen. Minor hand-held camera
            # motion and bbox breathing should not continuously restart it.
            if center_shift >= 0.12 or overlap < 0.12:
                reasons.append(f"moved:{label}")
            if area_change >= 2.5:
                reasons.append(f"size_changed:{label}")

    return bool(reasons), reasons



def draw_numbered_frame(image: Image.Image, objects: list[dict]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        small = font

    for obj in objects:
        x1, y1, x2, y2 = [int(round(v)) for v in obj["bbox"]]
        draw.rectangle([x1, y1, x2, y2], outline=(80, 235, 255), width=5)
        # Keep stable track IDs internal. The exhibition frame shows only the
        # cyan detection outline so ID changes are invisible to the audience.
    return output


def image_data_url(image: Image.Image, max_edge: int = 512) -> str:
    img = image.copy()
    resampling = getattr(Image, "Resampling", Image)
    img.thumbnail((max_edge, max_edge), resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


SYSTEM_PROMPT = """你是半即時現場觀察助理。
你會收到單一物件的裁切圖，以及偵測器提供的候選類別。

候選類別只用來核對，可能正確、可能不完整，也可能錯誤。
請以裁切圖為主要依據，但不要因為只看到包裝外觀就輕易否定候選類別。

判斷原則：
- 包裝盒、塑膠袋、藥品包裝、醫療用品包裝，只是外觀或包裝形式，
  不足以否定「防水繃帶、針筒、棉花棒、生理食鹽水」等候選類別。
- 只有看見明確屬於另一物體類型的特徵，例如玩偶、杯子、鑰匙、食物、遙控器，
  才能判定候選類別明確衝突。
- 看不清楚或只能看到包裝時，判定為 compatible 或 uncertain，不得判定 conflict。

規則：
1. 使用繁體中文。
2. 不要以「我看到」「我注意到」「仔細看」「畫面中」「畫面裡」開頭。
3. 不要報位置、座標、信心值、編號或 ID。
4. text 只描述實際可見的名稱、顏色、材質、外觀、包裝或狀態。
5. label_relation 只能是：
   - "match"：明確相符
   - "compatible"：外觀與候選類別相容，但無法完全確認
   - "uncertain"：資訊不足
   - "conflict"：明確是另一種物體
6. 只有 label_relation="conflict" 時才填 observed_name。
7. 如果只是盒裝、袋裝或藥品包裝，通常應為 compatible，不是 conflict。
8. 每句最多 30 個中文字。
9. 嚴格輸出 JSON，不加 Markdown。

輸出格式：
{
  "object_descriptions": [
    {
      "object_id": 1,
      "label_relation": "compatible",
      "observed_name": "",
      "text": "藍白相間的藥品包裝盒，表面有模糊文字與圖案。"
    }
  ]
}
"""


UNBOXED_AUDIT_PROMPT = """你是醫療用品現場畫面的「漏框候選提出員」。
你會收到四張互相重疊的分區畫面，每張圖前面會標明 view_id。
青色方框代表已被 YOLO 偵測；不要重複提出青色框已覆蓋的物件。

這是第一階段的高召回搜尋：第二階段會逐一裁切並嚴格否決錯誤候選。
因此只要未框物件具有合理的類別可能性，就應提出候選；不要因品牌文字看不清楚而直接漏掉。
但仍不得把純背景、衣物或印刷圖片當成實體醫療用品。

逐一檢查以下九類，不得只找到最顯眼的一項就停止：
- ac_remotecontrol：手持式遙控器，可見按鍵、控制面板或細長控制器外形。
- bottle_alcohol_spray：瓶身加噴頭、壓頭或扳機噴嘴。
- cotton_swab：細長棒狀，至少一端疑似有棉頭。
- cotton_swabs_pp：疑似棉花棒零售包裝。
- disposable_mask：扁平口罩本體，可能可見摺線、耳掛或綁帶。
- gauze_pp：疑似紗布或敷料醫療包裝。
- saline：疑似生理食鹽水瓶、袋或包裝；可利用瓶身圖、藍白包裝與可讀文字線索。
- syringe_nipro：疑似針筒、推桿、針帽，或細長的 NIPRO 針筒包裝。
- waterproof_bandages_ppb：疑似 PPB／Nexcare 防水繃帶包裝。

規則：
1. 每個 view 都完整掃描；最多提出 12 個候選。
2. 同一實體若在重疊 view 重複出現，可以重複提出，後端會合併。
3. 不要回報青色框已覆蓋的物品。
4. 不要回報桌面、螢幕、手、陰影、一般衣物、布料、標籤文字、印刷圖片或背景設備。
5. bbox_xyxy_norm_in_view 是相對於該 view 的 [左,上,右,下]，數值 0 到 1。
6. visual_evidence 寫具體外形線索；不確定處可明確寫「待第二階段確認」。
7. 第一階段 confidence 只代表值得複核的可能性；0.35 以上即可列入。
8. 每個候選必須附上來源 view_id。
9. 嚴格輸出 JSON，不加 Markdown。

輸出格式：
{
  "missing_objects": [
    {
      "view_id": "upper_left",
      "class_name": "cotton_swab",
      "confidence": 0.58,
      "location": "中央偏右",
      "bbox_xyxy_norm_in_view": [0.61, 0.34, 0.72, 0.48],
      "visual_evidence": "可見細長棒身，端部疑似白色棉頭，待複核"
    }
  ]
}

若完全沒有合理候選：
{"missing_objects":[]}
"""

UNBOXED_VERIFY_PROMPT = """你是第二階段的醫療用品漏框複核員。
你會收到若干候選物件的裁切圖，每張圖前都有 candidate_index 與 proposed_class。
第一階段可能把一般物品硬套成醫療類別；你必須嚴格否決錯誤候選。

核心規則：
- 只根據裁切圖中實際物件判斷，不相信第一階段名稱。
- 不是該類別就 confirmed=false；不要改猜成另一個允許類別。
- 背景、包裝印刷圖案、布料、衣物或看不清楚的物品一律否決。
- disposable_mask 只有在看得到扁平口罩本體，以及耳掛／綁帶或明確口罩摺線時才能確認。
  條紋布、毛巾、襪子、手套、布袋、玩偶與衣物全部不是口罩。
- cotton_swab 必須是細長棒狀，至少一端可見白色棉頭；一般細棒、筷子或電線不得確認。
- bottle_alcohol_spray 必須可見瓶身及噴頭、壓頭或扳機噴嘴。
- saline 可由生理食鹽水字樣、saline 字樣、藍白醫療包裝或明確食鹽水瓶身圖共同確認；一般飲料瓶不得確認。
- syringe_nipro 必須可見針筒結構，或細長密封包裝上有 NIPRO／針筒用途線索。
- cotton_swabs_pp、gauze_pp、waterproof_bandages_ppb 等包裝類必須有足以區分類別的外觀、品牌或用途線索；一般紙盒不得確認。
- 不確定就 confirmed=false。

嚴格輸出 JSON：
{
  "verified_objects": [
    {
      "candidate_index": 1,
      "confirmed": false,
      "confidence": 0.96,
      "reason": "這是棕白條紋布料，沒有口罩本體或耳掛"
    }
  ]
}
"""


UNBOXED_CLASS_ZH = {
    "ac_remotecontrol": "冷氣遙控器",
    "bottle_alcohol_spray": "酒精噴瓶",
    "cotton_swab": "棉花棒",
    "cotton_swabs_pp": "棉花棒包裝",
    "disposable_mask": "一次性口罩",
    "gauze_pp": "紗布包裝",
    "saline": "生理食鹽水",
    "syringe_nipro": "NIPRO 針筒",
    "waterproof_bandages_ppb": "PPB 防水繃帶",
}


def _parse_qwen_json_content(content: Any) -> dict:
    if isinstance(content, list):
        content = "".join(
            str(x.get("text") or "")
            for x in content
            if isinstance(x, dict)
        )
    match = re.search(r"\{.*\}", str(content or ""), flags=re.S)
    parsed = json.loads(match.group(0) if match else str(content or ""))
    return parsed if isinstance(parsed, dict) else {}


def _normalized_candidate_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in value]
    except Exception:
        return None
    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
        return None
    x1, y1, x2, y2 = [clamp(x, 0.0, 1.0) for x in (x1, y1, x2, y2)]
    if x2 - x1 < 0.015 or y2 - y1 < 0.015:
        return None
    return [x1, y1, x2, y2]


def _crop_normalized_box(
    image: Image.Image,
    bbox_xyxy_norm: list[float],
    *,
    padding_ratio: float = 0.18,
) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox_xyxy_norm
    px1, py1, px2, py2 = x1 * width, y1 * height, x2 * width, y2 * height
    bw = max(2.0, px2 - px1)
    bh = max(2.0, py2 - py1)
    px1 = max(0, int(math.floor(px1 - bw * padding_ratio)))
    py1 = max(0, int(math.floor(py1 - bh * padding_ratio)))
    px2 = min(width, int(math.ceil(px2 + bw * padding_ratio)))
    py2 = min(height, int(math.ceil(py2 + bh * padding_ratio)))
    return image.crop((px1, py1, max(px1 + 1, px2), max(py1 + 1, py2))).convert("RGB")


def _unboxed_audit_views(image: Image.Image) -> list[tuple[str, tuple[float, float, float, float], Image.Image]]:
    """Create four overlapping views so small objects remain visible to Qwen."""
    width, height = image.size
    regions = [
        ("upper_left", (0.00, 0.00, 0.60, 0.62)),
        ("upper_right", (0.40, 0.00, 1.00, 0.62)),
        ("lower_left", (0.00, 0.38, 0.60, 1.00)),
        ("lower_right", (0.40, 0.38, 1.00, 1.00)),
    ]
    views = []
    for view_id, region in regions:
        x1, y1, x2, y2 = region
        crop = image.crop((
            int(round(x1 * width)),
            int(round(y1 * height)),
            int(round(x2 * width)),
            int(round(y2 * height)),
        )).convert("RGB")
        views.append((view_id, region, crop))
    return views


def _map_view_box_to_full(
    box: list[float],
    region: tuple[float, float, float, float],
) -> list[float]:
    rx1, ry1, rx2, ry2 = region
    x1, y1, x2, y2 = box
    return [
        rx1 + x1 * (rx2 - rx1),
        ry1 + y1 * (ry2 - ry1),
        rx1 + x2 * (rx2 - rx1),
        ry1 + y2 * (ry2 - ry1),
    ]


def _dedupe_unboxed_proposals(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for row in sorted(
        rows,
        key=lambda x: float(x.get("confidence") or 0.0),
        reverse=True,
    ):
        box = _normalized_candidate_box(row.get("bbox_xyxy_norm"))
        if box is None:
            continue
        class_name = str(row.get("class_name") or "")
        duplicate = False
        for old in kept:
            if str(old.get("class_name") or "") != class_name:
                continue
            old_box = _normalized_candidate_box(old.get("bbox_xyxy_norm"))
            if old_box is None:
                continue
            if iou(box, old_box) >= 0.28:
                duplicate = True
                break
            cx, cy = center(box)
            ox, oy = center(old_box)
            scale = max(0.03, math.sqrt(max(1e-9, box_area(box))), math.sqrt(max(1e-9, box_area(old_box))))
            if math.hypot(cx - ox, cy - oy) <= scale * 0.42:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
        if len(kept) >= 12:
            break
    return kept



def _normalized_existing_box(
    obj: dict,
    image_size: tuple[int, int],
) -> list[float] | None:
    box = extract_box(obj)
    if box is None:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = box
    return [
        clamp(x1 / width, 0.0, 1.0),
        clamp(y1 / height, 0.0, 1.0),
        clamp(x2 / width, 0.0, 1.0),
        clamp(y2 / height, 0.0, 1.0),
    ]


def _intersection_area(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _point_inside_expanded_box(
    point: tuple[float, float],
    box: list[float],
    padding_ratio: float = 0.12,
) -> bool:
    cx, cy = point
    bw = max(0.0, box[2] - box[0])
    bh = max(0.0, box[3] - box[1])
    return (
        box[0] - bw * padding_ratio <= cx <= box[2] + bw * padding_ratio
        and box[1] - bh * padding_ratio <= cy <= box[3] + bh * padding_ratio
    )


def _candidate_matches_existing_detection(
    candidate: dict,
    existing_objects: list[dict],
    image_size: tuple[int, int],
) -> tuple[bool, str]:
    """Deterministically reject a VLM 'missing' candidate already covered by YOLO.

    The VLM is only a proposer. It may ignore the cyan outline or return a loose
    candidate box. Geometry is therefore the source of truth for deciding
    whether an item is already boxed.
    """
    candidate_box = _normalized_candidate_box(candidate.get("bbox_xyxy_norm"))
    if candidate_box is None:
        return False, "candidate_has_no_box"

    candidate_class = canonical(candidate.get("class_name") or "").replace(" ", "_")
    candidate_center = center(candidate_box)
    candidate_area = max(1e-9, box_area(candidate_box))

    for obj in existing_objects:
        if not isinstance(obj, dict):
            continue
        existing_box = _normalized_existing_box(obj, image_size)
        if existing_box is None:
            continue

        existing_class = canonical(obj.get("label") or "").replace(" ", "_")
        existing_area = max(1e-9, box_area(existing_box))
        inter = _intersection_area(candidate_box, existing_box)
        candidate_coverage = inter / candidate_area
        existing_coverage = inter / existing_area
        same_class = bool(candidate_class and candidate_class == existing_class)
        center_covered = _point_inside_expanded_box(candidate_center, existing_box)

        # Same-class candidates are rejected even with fairly loose VLM boxes.
        if same_class and (
            center_covered
            or candidate_coverage >= 0.10
            or existing_coverage >= 0.10
            or iou(candidate_box, existing_box) >= 0.05
        ):
            return True, (
                f"same_class={existing_class} candidate_cov={candidate_coverage:.3f} "
                f"existing_cov={existing_coverage:.3f}"
            )

        # Even if the proposed class is wrong, a candidate mostly located inside
        # any existing detection is not an unboxed object.
        if center_covered and candidate_coverage >= 0.35:
            return True, (
                f"covered_by={existing_class} candidate_cov={candidate_coverage:.3f}"
            )
        if candidate_coverage >= 0.60 or existing_coverage >= 0.82:
            return True, (
                f"strong_overlap={existing_class} candidate_cov={candidate_coverage:.3f} "
                f"existing_cov={existing_coverage:.3f}"
            )

    return False, "no_existing_match"


def filter_already_boxed_unboxed_candidates(
    payload: dict | None,
    existing_objects: list[dict],
    image_size: tuple[int, int],
    *,
    stage: str,
) -> dict:
    """Remove stale/duplicate missing-object claims using current YOLO geometry."""
    if not isinstance(payload, dict):
        return {"missing_objects": []}
    rows = payload.get("missing_objects")
    if not isinstance(rows, list):
        return {"missing_objects": []}

    kept: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        matched, reason = _candidate_matches_existing_detection(
            row,
            existing_objects,
            image_size,
        )
        if matched:
            print(
                "[unboxed-audit][already-boxed] "
                f"stage={stage} class={row.get('class_name')} reason={reason}",
                flush=True,
            )
            continue
        kept.append(row)

    out = dict(payload)
    out["missing_objects"] = kept
    return out


def call_qwen_unboxed_audit(annotated: Image.Image) -> dict:
    views = _unboxed_audit_views(annotated)
    content: list[dict] = [{
        "type": "text",
        "text": (
            "請依序掃描四張分區畫面。每張圖的 bbox 都必須使用該 view 的相對座標，"
            "並在結果中填入正確 view_id。不要在找到第一項後停止。"
        ),
    }]
    region_by_id: dict[str, tuple[float, float, float, float]] = {}
    for view_id, region, crop in views:
        region_by_id[view_id] = region
        content.append({
            "type": "text",
            "text": f"view_id={view_id}; 這是完整畫面的重疊分區。",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url(crop, max_edge=800)},
        })

    payload = {
        "model": QWEN_MODEL,
        "temperature": 0.05,
        "top_p": 0.65,
        "max_tokens": 980,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": UNBOXED_AUDIT_PROMPT},
            {"role": "user", "content": content},
        ],
    }
    req = urllib.request.Request(
        QWEN_BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local",
        },
    )
    with urllib.request.urlopen(req, timeout=max(QWEN_TIMEOUT_SEC, 120.0)) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    parsed = _parse_qwen_json_content(data["choices"][0]["message"]["content"])

    mapped: list[dict] = []
    rows = parsed.get("missing_objects") if isinstance(parsed, dict) else []
    if not isinstance(rows, list):
        rows = []
    for item in rows[:20]:
        if not isinstance(item, dict):
            continue
        class_name = canonical(item.get("class_name") or "").replace(" ", "_")
        if class_name not in UNBOXED_CLASS_ZH:
            continue
        view_id = str(item.get("view_id") or "").strip()
        region = region_by_id.get(view_id)
        if region is None:
            continue
        local_box = _normalized_candidate_box(
            item.get("bbox_xyxy_norm_in_view") or item.get("bbox_xyxy_norm")
        )
        if local_box is None:
            continue
        try:
            confidence = clamp(float(item.get("confidence") or 0.0), 0.0, 1.0)
        except Exception:
            confidence = 0.0
        if confidence < 0.35:
            continue
        row = dict(item)
        row["class_name"] = class_name
        row["source_view_id"] = view_id
        row["bbox_xyxy_norm"] = _map_view_box_to_full(local_box, region)
        row["confidence"] = confidence
        mapped.append(row)

    return {"missing_objects": _dedupe_unboxed_proposals(mapped)}


def call_qwen_unboxed_verifier(
    annotated: Image.Image,
    proposed: dict,
) -> dict:
    rows = proposed.get("missing_objects") if isinstance(proposed, dict) else []
    if not isinstance(rows, list):
        return {"verified_objects": []}

    candidates: list[dict] = []
    content: list[dict] = [{
        "type": "text",
        "text": "請逐一複核下面的候選裁切圖。每個 candidate_index 都必須回傳一筆結果。",
    }]

    for item in rows[:8]:
        if not isinstance(item, dict):
            continue
        raw_class = str(item.get("class_name") or "").strip()
        class_name = canonical(raw_class).replace(" ", "_")
        if class_name not in UNBOXED_CLASS_ZH:
            continue
        bbox = _normalized_candidate_box(item.get("bbox_xyxy_norm"))
        if bbox is None:
            continue
        candidate_index = len(candidates) + 1
        candidates.append({
            "candidate_index": candidate_index,
            "class_name": class_name,
            "bbox_xyxy_norm": bbox,
            "source": item,
        })
        crop = _crop_normalized_box(annotated, bbox)
        content.append({
            "type": "text",
            "text": (
                f"candidate_index={candidate_index}; "
                f"proposed_class={class_name}; "
                f"中文候選={UNBOXED_CLASS_ZH[class_name]}"
            ),
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url(crop, max_edge=512)},
        })

    if not candidates:
        return {"verified_objects": [], "candidate_map": []}

    payload = {
        "model": QWEN_MODEL,
        "temperature": 0.0,
        "top_p": 0.35,
        "max_tokens": 520,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": UNBOXED_VERIFY_PROMPT},
            {"role": "user", "content": content},
        ],
    }
    req = urllib.request.Request(
        QWEN_BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local",
        },
    )
    with urllib.request.urlopen(req, timeout=QWEN_TIMEOUT_SEC) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    parsed = _parse_qwen_json_content(data["choices"][0]["message"]["content"])
    parsed["candidate_map"] = candidates
    return parsed


def merge_verified_unboxed_candidates(proposed: dict, verified: dict) -> dict:
    candidates = verified.get("candidate_map") if isinstance(verified, dict) else []
    verdicts = verified.get("verified_objects") if isinstance(verified, dict) else []
    if not isinstance(candidates, list) or not isinstance(verdicts, list):
        return {"missing_objects": []}

    verdict_by_index: dict[int, dict] = {}
    for row in verdicts:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("candidate_index"))
        except Exception:
            continue
        verdict_by_index[idx] = row

    accepted: list[dict] = []
    for candidate in candidates:
        idx = int(candidate["candidate_index"])
        verdict = verdict_by_index.get(idx, {})
        confirmed = verdict.get("confirmed") is True
        try:
            verify_conf = clamp(float(verdict.get("confidence") or 0.0), 0.0, 1.0)
        except Exception:
            verify_conf = 0.0
        source = dict(candidate.get("source") or {})
        try:
            proposal_conf = clamp(float(source.get("confidence") or 0.0), 0.0, 1.0)
        except Exception:
            proposal_conf = 0.0

        # Stage 1 is intentionally high-recall; Stage 2 owns the strict precision gate.
        if not confirmed or proposal_conf < 0.35 or verify_conf < 0.78:
            print(
                "[unboxed-audit][reject] "
                f"class={candidate['class_name']} proposal={proposal_conf:.2f} "
                f"verify={verify_conf:.2f} reason={verdict.get('reason')}",
                flush=True,
            )
            continue

        source["class_name"] = candidate["class_name"]
        source["bbox_xyxy_norm"] = candidate["bbox_xyxy_norm"]
        source["verification_confidence"] = verify_conf
        source["verification_reason"] = str(verdict.get("reason") or "").strip()
        source["confidence"] = min(proposal_conf, verify_conf)
        accepted.append(source)

    return {"missing_objects": accepted}


def clean_unboxed_observation(raw: dict | None) -> dict | None:
    """Convert verified multi-object Qwen audit into one compact yellow UI row."""
    if not isinstance(raw, dict):
        return None

    rows = raw.get("missing_objects")
    if not isinstance(rows, list):
        return None

    cleaned_rows: list[dict] = []
    seen_classes: set[str] = set()
    zh_to_class = {value: key for key, value in UNBOXED_CLASS_ZH.items()}

    for item in rows[:12]:
        if not isinstance(item, dict):
            continue

        raw_class = str(
            item.get("class_name")
            or item.get("possible_label")
            or item.get("label")
            or ""
        ).strip()
        class_name = canonical(raw_class).replace(" ", "_")
        if raw_class in zh_to_class:
            class_name = zh_to_class[raw_class]
        if class_name not in UNBOXED_CLASS_ZH or class_name in seen_classes:
            continue

        try:
            confidence = clamp(float(item.get("confidence") or 0.0), 0.0, 1.0)
        except Exception:
            confidence = 0.0
        if confidence < 0.65:
            continue

        location = re.sub(r"\s+", "", str(item.get("location") or "").strip())[:12]
        display_label = UNBOXED_CLASS_ZH[class_name]
        cleaned_rows.append({
            "class_name": class_name,
            "possible_label": display_label,
            "confidence": confidence,
            "location": location,
            "bbox_xyxy_norm": item.get("bbox_xyxy_norm"),
            "visual_evidence": str(item.get("visual_evidence") or "").strip(),
            "verification_reason": str(item.get("verification_reason") or "").strip(),
        })
        seen_classes.add(class_name)

        if len(cleaned_rows) >= 8:
            break

    if not cleaned_rows:
        return None

    parts = []
    for item in cleaned_rows:
        label = item["possible_label"]
        location = item.get("location") or ""
        parts.append(f"{label}（{location}）" if location else label)

    text = "漏框提醒：尚未框選「" + "、".join(parts) + "」。"
    return {
        "possible_label": "、".join(x["possible_label"] for x in cleaned_rows),
        "missing_objects": cleaned_rows,
        "count": len(cleaned_rows),
        "confidence": max(x["confidence"] for x in cleaned_rows),
        "text": text,
    }


def start_unboxed_audit(
    session: SessionState,
    *,
    signature: str,
    annotated: Image.Image,
    existing_objects: list[dict],
) -> bool:
    if not UNBOXED_AUDIT_ENABLED:
        return False

    now = time.time()
    with _SESSION_LOCK:
        if session.unboxed_pending:
            return False
        if session.last_signature != signature:
            return False
        if now - session.last_scene_change_at < UNBOXED_AUDIT_DELAY_SEC:
            return False
        if now - session.unboxed_checked_at < UNBOXED_AUDIT_INTERVAL_SEC:
            return False

    if not _QWEN_WORK_LOCK.acquire(blocking=False):
        return False

    with _SESSION_LOCK:
        if session.unboxed_pending or session.last_signature != signature:
            _QWEN_WORK_LOCK.release()
            return False
        session.unboxed_job_token += 1
        job_token = session.unboxed_job_token
        session.unboxed_pending = True
        session.unboxed_checked_at = now

    image_copy = annotated.copy()
    image_size = image_copy.size
    existing_snapshot = [dict(x) for x in existing_objects if isinstance(x, dict)]

    def worker() -> None:
        try:
            proposed = call_qwen_unboxed_audit(image_copy)
            proposed = filter_already_boxed_unboxed_candidates(
                proposed,
                existing_snapshot,
                image_size,
                stage="proposal",
            )
            verified = call_qwen_unboxed_verifier(image_copy, proposed)
            raw = merge_verified_unboxed_candidates(proposed, verified)
            raw = filter_already_boxed_unboxed_candidates(
                raw,
                existing_snapshot,
                image_size,
                stage="verified",
            )
            cleaned = clean_unboxed_observation(raw)
            with _SESSION_LOCK:
                if (
                    session.last_signature == signature
                    and session.unboxed_job_token == job_token
                ):
                    session.unboxed_result = cleaned
            if cleaned:
                labels = [
                    x.get("possible_label")
                    for x in cleaned.get("missing_objects", [])
                ]
                print(
                    f"[unboxed-audit] missing={labels}",
                    flush=True,
                )
            else:
                print("[unboxed-audit] no clear missing object", flush=True)
        except Exception as exc:
            print(f"[unboxed-audit][WARN] {type(exc).__name__}: {exc}", flush=True)
        finally:
            with _SESSION_LOCK:
                if session.unboxed_job_token == job_token:
                    session.unboxed_pending = False
            _QWEN_WORK_LOCK.release()

    threading.Thread(
        target=worker,
        name=f"scene-narrator-unboxed-{job_token}",
        daemon=True,
    ).start()
    return True


def crop_object_for_qwen(
    image: Image.Image,
    bbox: list[float],
    *,
    padding_ratio: float = 0.22,
) -> Image.Image:
    """Crop one object so appearance details stay tied to that object."""
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = max(12.0, bw * padding_ratio)
    pad_y = max(12.0, bh * padding_ratio)
    return image.crop((
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )).convert("RGB")


def _vlm_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size,
        )
    except Exception:
        return ImageFont.load_default()


def _paste_contain(
    canvas: Image.Image,
    source: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    width = max(1, right - left)
    height = max(1, bottom - top)
    image = source.convert("RGB").copy()
    resampling = getattr(Image, "Resampling", Image)
    image.thumbnail((width, height), resampling.LANCZOS)
    x = left + (width - image.width) // 2
    y = top + (height - image.height) // 2
    canvas.paste(image, (x, y))


def build_vlm_contact_sheet(
    image: Image.Image,
    objects: list[dict],
) -> Image.Image:
    """Build one image containing object close-ups and one clean scene view."""
    selected = list(objects)[:VLM_COMBINED_MAX_OBJECTS]
    count = len(selected)
    columns = 1 if count <= 1 else 2
    rows = max(1, math.ceil(max(1, count) / columns))

    base_cell = int(VLM_CONTACT_SHEET_CELL_EDGE)
    cell_width = (
        max(224, int(base_cell * 1.45))
        if columns == 2
        else max(320, int(base_cell * 2.15))
    )
    cell_height = max(156, base_cell)
    scene_height = max(144, int(base_cell * 0.90))
    canvas_width = cell_width * columns
    canvas_height = cell_height * rows + scene_height

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        (8, 17, 27),
    )
    draw = ImageDraw.Draw(canvas)
    label_font = _vlm_font(22)
    small_font = _vlm_font(18)

    for index, obj in enumerate(selected):
        row = index // columns
        column = index % columns
        left = column * cell_width
        top = row * cell_height
        right = left + cell_width
        bottom = top + cell_height

        draw.rectangle(
            [left + 2, top + 2, right - 3, bottom - 3],
            outline=(55, 215, 240),
            width=3,
        )
        draw.rectangle(
            [left + 3, top + 3, right - 4, top + 33],
            fill=(5, 24, 35),
        )

        object_id = int(obj.get("id") or 0)
        label = f"#{object_id}"
        draw.text(
            (left + 12, top + 7),
            label,
            fill=(235, 252, 255),
            font=label_font,
        )

        bbox = obj.get("bbox") or extract_box(obj)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            crop = crop_object_for_qwen(
                image,
                [float(value) for value in bbox[:4]],
                padding_ratio=0.30,
            )
            _paste_contain(
                canvas,
                crop,
                (left + 8, top + 39, right - 8, bottom - 8),
            )

    scene_top = cell_height * rows
    draw.rectangle(
        [2, scene_top + 2, canvas_width - 3, canvas_height - 3],
        outline=(180, 130, 250),
        width=3,
    )
    draw.rectangle(
        [3, scene_top + 3, canvas_width - 4, scene_top + 31],
        fill=(25, 17, 42),
    )
    draw.text(
        (12, scene_top + 6),
        "SCENE",
        fill=(235, 220, 255),
        font=small_font,
    )
    _paste_contain(
        canvas,
        image.convert("RGB"),
        (
            8,
            scene_top + 36,
            canvas_width - 8,
            canvas_height - 8,
        ),
    )
    return canvas


def _parse_vlm_protocol_line(line: str) -> tuple[str, int | None, str] | None:
    cleaned = str(line or "").strip()
    cleaned = cleaned.strip("`").strip()
    cleaned = re.sub(r"^[\-*•]\s*", "", cleaned)
    if not cleaned:
        return None

    object_match = re.match(
        r"^OBJ\s*[|｜]\s*(\d+)\s*[|｜]\s*(.+)$",
        cleaned,
        flags=re.I,
    )
    if object_match:
        return (
            "object",
            int(object_match.group(1)),
            object_match.group(2).strip(),
        )

    scene_match = re.match(
        r"^SCENE\s*[|｜]\s*(.+)$",
        cleaned,
        flags=re.I,
    )
    if scene_match:
        return ("scene", None, scene_match.group(1).strip())

    return None


def normalize_narration_sentence(value: Any) -> str:
    """Normalize malformed sentence endings such as `；。` or `，。`."""
    text_value = re.sub(
        r"\s+",
        "",
        str(value or ""),
    ).strip()

    if not text_value:
        return ""

    # Qwen occasionally emits a separator before the final punctuation,
    # for example `；。`. Remove only the redundant separator at sentence end.
    text_value = re.sub(
        r"[；;，,、]+(?=[。！？!?]+$)",
        "",
        text_value,
    )
    text_value = re.sub(
        r"[；;，,、]+$",
        "",
        text_value,
    )
    text_value = re.sub(
        r"([。！？!?])\1+$",
        r"\1",
        text_value,
    )

    if not text_value:
        return ""

    if text_value[-1] not in "。！？!?":
        text_value += "。"

    return text_value


def call_qwen(
    annotated: Image.Image,
    objects: list[dict],
    relations: list[dict],
    *,
    on_object: Any = None,
    on_scene: Any = None,
) -> dict:
    """Stream one contact-sheet request for green rows and the purple summary."""
    _ = relations
    selected_objects = sorted(
        objects,
        key=lambda item: int(item.get("id") or 0),
    )[:VLM_COMBINED_MAX_OBJECTS]

    compact_objects = [
        {
            "object_id": int(obj["id"]),
            "detector_candidate_label": label_zh(obj.get("label")),
            "original_detector_label": str(obj.get("label") or ""),
            "confidence": round(float(obj.get("confidence") or 0.0), 3),
        }
        for obj in selected_objects
        if obj.get("id") is not None
    ]
    required_ids = [row["object_id"] for row in compact_objects]

    if not required_ids:
        return {
            "object_descriptions": [],
            "scene_summary": VLM_EMPTY_SCENE_TEXT,
        }

    contact_sheet = build_vlm_contact_sheet(
        annotated,
        selected_objects,
    )

    request_context = {
        "required_objects": compact_objects,
        "required_object_ids": required_ids,
        "contact_sheet_layout": (
            "Each #ID tile is a close-up of one object. "
            "The SCENE tile is the clean whole camera view."
        ),
    }

    system_prompt = (
        "你是即時物件外觀描述器。輸入是一張近照聯絡表："
        "每個 #ID 區塊是一個 YOLO 物件近照，SCENE 區塊是完整椅面。"
        "依 required_object_ids 的順序逐行輸出，不得跳過、合併或改編號。"
        "每個物件必須描述至少兩項可見特徵，"
        "例如顏色、形狀、材質、包裝或表面特徵。"
        "外觀短句不得只重複候選類別名稱，最多22個中文字。"
        "最後用 SCENE 行概括椅面上是否有物品，最多30個中文字。"
        "不要規劃動作，不要輸出座標、信心值、JSON、Markdown或解釋。"
        "格式必須嚴格為：每行 OBJ|物件ID|外觀短句，"
        "最後一行為 SCENE|場景短句。"
        "每行結尾直接換行，不要加分號。"
    )

    user_prompt = (
        json.dumps(request_context, ensure_ascii=False)
        + "\n請立即從第一個 OBJ 行開始輸出。"
    )

    payload = {
        "model": QWEN_MODEL,
        "temperature": 0.0,
        "top_p": 0.55,
        "max_tokens": VLM_COMBINED_MAX_TOKENS,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url(
                                contact_sheet,
                                max_edge=VLM_COMBINED_IMAGE_EDGE,
                            ),
                        },
                    },
                ],
            },
        ],
    }

    request = urllib.request.Request(
        QWEN_BASE_URL + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local",
            "Accept": "text/event-stream",
        },
    )

    rows_by_id: dict[int, dict] = {}
    scene_summary = ""
    protocol_buffer = ""
    full_content = ""
    non_sse_lines: list[str] = []
    saw_sse = False

    def accept_protocol_line(line: str) -> None:
        nonlocal scene_summary
        parsed = _parse_vlm_protocol_line(line)
        if parsed is None:
            return
        kind, object_id, text_value = parsed
        text_value = normalize_narration_sentence(
            localize_narration_text(text_value)
        )

        if not text_value:
            return

        if kind == "object" and object_id in required_ids:
            row = {
                "object_id": int(object_id),
                "text": text_value[:80],
                "confidence": 0.7,
            }
            rows_by_id[int(object_id)] = row
            if callable(on_object):
                on_object(dict(row))
        elif kind == "scene":
            scene_summary = text_value[:100]
            if callable(on_scene):
                on_scene(scene_summary)

    def feed_content(content: str) -> None:
        nonlocal protocol_buffer, full_content
        if not content:
            return
        full_content += content
        protocol_buffer += content.replace("\r\n", "\n")
        while "\n" in protocol_buffer:
            line, protocol_buffer = protocol_buffer.split("\n", 1)
            accept_protocol_line(line)

    with urllib.request.urlopen(
        request,
        timeout=max(QWEN_TIMEOUT_SEC, 75.0),
    ) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("data:"):
                saw_sse = True
                payload_text = line[5:].strip()
                if payload_text == "[DONE]":
                    break
                try:
                    event = json.loads(payload_text)
                except Exception:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content is None:
                    content = (choice.get("message") or {}).get("content")
                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text") or "")
                        for item in content
                        if isinstance(item, dict)
                    )
                feed_content(str(content or ""))
            else:
                non_sse_lines.append(line)

    if not saw_sse and non_sse_lines:
        raw = "".join(non_sse_lines)
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict)
            )
        feed_content(str(content or ""))

    accept_protocol_line(protocol_buffer)

    # Compatibility fallback when a model ignores the line protocol and emits
    # the older JSON contract.
    if len(rows_by_id) < len(required_ids) and "{" in full_content:
        try:
            match = re.search(r"\{.*\}", full_content, flags=re.S)
            parsed_json = json.loads(match.group(0)) if match else {}
        except Exception:
            parsed_json = {}
        for raw_row in parsed_json.get("object_descriptions") or []:
            if not isinstance(raw_row, dict):
                continue
            try:
                object_id = int(raw_row.get("object_id"))
            except Exception:
                continue
            if object_id not in required_ids or object_id in rows_by_id:
                continue
            text_value = str(
                raw_row.get("text")
                or raw_row.get("description")
                or ""
            ).strip()
            if text_value:
                accept_protocol_line(f"OBJ|{object_id}|{text_value}")
        if not scene_summary:
            raw_summary = str(
                parsed_json.get("scene_summary")
                or parsed_json.get("summary")
                or ""
            ).strip()
            if raw_summary:
                accept_protocol_line(f"SCENE|{raw_summary}")

    label_by_id = {
        row["object_id"]: row["detector_candidate_label"]
        for row in compact_objects
    }
    for object_id in required_ids:
        if object_id in rows_by_id:
            continue
        rows_by_id[object_id] = {
            "object_id": object_id,
            "text": (
                f"{label_by_id.get(object_id, '此物件')}"
                "的外觀細節目前無法可靠判定。"
            ),
            "confidence": 0.3,
        }

    object_names = [
        row["detector_candidate_label"]
        for row in compact_objects
    ]
    if (
        not scene_summary
        or "未見可辨識物品" in scene_summary
        or "沒有可辨識物品" in scene_summary
    ):
        scene_summary = (
            "椅面上可見"
            + "、".join(object_names[:4])
            + "等物品。"
        )

    return {
        "object_descriptions": [
            rows_by_id[object_id]
            for object_id in required_ids
        ],
        "scene_summary": scene_summary,
    }


def conversational_object_text(
    text_value: Any,
    *,
    object_id: int,
    object_label: str,
    sequence_index: int,
    label_relation: Any = None,
    observed_name: Any = "",
) -> str:
    """Use the VLM observation directly and correct a clearly wrong detector label."""
    text = localize_narration_text(text_value)
    text = re.sub(r"\s+", "", text).strip()
    if not text:
        return ""

    # Remove internal IDs and unwanted conversational starters.
    text = re.sub(r"#\s*\d+", "", text)
    text = re.sub(r"編號\s*\d+\s*的?", "", text)
    text = re.sub(r"\b(?:ID|id)\s*[:：#]?\s*\d+\b", "", text)

    prefix_patterns = (
        r"^(?:我看到|我注意到|我先注意到)",
        r"^(?:仔細看|再看一下|看起來|接著看)[，,：:]?",
        r"^(?:接著我注意到|另外我注意到|我也注意到)[，,：:]?",
        r"^(?:畫面裡還有|另一個清楚可見的是)",
        r"^(?:另外可以看到|另外|此外|再者)[，,：:]?",
    )
    changed = True
    while changed:
        changed = False
        for pattern in prefix_patterns:
            cleaned = re.sub(pattern, "", text).strip()
            if cleaned != text:
                text = cleaned
                changed = True

    detector_label = label_zh(object_label)
    observed = localize_narration_text(observed_name)
    observed = re.sub(r"\s+", "", observed).strip("，,。:： ")

    # Remove an echoed detector label from the start of the VLM description.
    detector_variants = {
        detector_label,
        str(object_label or "").strip(),
        f"一個{detector_label}",
        f"這個{detector_label}",
    }
    for variant in sorted(detector_variants, key=len, reverse=True):
        if variant:
            text = re.sub(
                rf"^{re.escape(variant)}(?:是|為)?[，,：:]?",
                "",
                text,
                count=1,
            ).strip()

    if not text:
        text = observed
    if not text:
        return ""

    text = normalize_narration_sentence(text)
    if not text:
        return ""

    relation = str(label_relation or "").strip().lower()

    # Only contradict YOLO for an explicit, semantically clear conflict.
    # Generic packaging descriptions are compatible with medical-item labels.
    generic_packaging_terms = (
        "包裝", "包裝盒", "藥品盒", "藥品包裝", "醫療用品包裝",
        "塑膠袋", "袋裝", "盒裝", "紙盒", "外盒",
    )
    specific_conflict_terms = (
        "玩偶", "杯子", "鑰匙", "食物", "餅乾", "遙控器",
        "手機", "剪刀", "筆", "瓶蓋", "公仔", "娃娃",
    )

    observed_is_generic_packaging = any(
        term in (observed + text) for term in generic_packaging_terms
    )
    observed_has_specific_conflict = any(
        term in (observed + text) for term in specific_conflict_terms
    )

    should_correct = (
        relation == "conflict"
        and bool(observed)
        and not observed_is_generic_packaging
        and observed_has_specific_conflict
    )

    if should_correct:
        sentence = (
            f"更正：剛才的「{detector_label}」標籤不準確，"
            f"實際看起來是{text}"
        )
    else:
        sentence = text

    sentence = sentence.replace("，，", "，").replace("。。", "。")
    sentence = normalize_narration_sentence(sentence)
    return sentence[:140]


def distinguish_duplicate_label(
    object_id: int,
    object_label: str,
    input_objects: list[dict],
) -> str:
    """Distinguish same-class detections by horizontal position."""
    label_key = canonical(object_label)
    peers = [
        item for item in input_objects
        if isinstance(item, dict) and canonical(item.get("label")) == label_key
    ]
    if len(peers) <= 1:
        return object_label

    def center_x(item: dict) -> float:
        box = item.get("bbox") or item.get("box") or [0, 0, 0, 0]
        try:
            return (float(box[0]) + float(box[2])) / 2.0
        except Exception:
            return 0.0

    peers = sorted(peers, key=center_x)
    peer_ids = [int(item.get("id", -1)) for item in peers]
    try:
        rank = peer_ids.index(int(object_id))
    except ValueError:
        return object_label

    if len(peers) == 2:
        qualifier = "左側的" if rank == 0 else "右側的"
    elif rank == 0:
        qualifier = "左側的"
    elif rank == len(peers) - 1:
        qualifier = "右側的"
    else:
        qualifier = "中央的"
    return qualifier + object_label


def clean_narration(parsed: dict, valid_ids: set[int]) -> dict:
    scene_summary = localize_narration_text(parsed.get("scene_summary"))[:90]
    if re.search(r"(辨識到|共有|總共|共)\s*\d+\s*個物件", scene_summary):
        scene_summary = ""
    object_rows = []
    for row in parsed.get("object_descriptions") or []:
        if not isinstance(row, dict):
            continue
        try:
            object_id = int(row.get("object_id"))
        except Exception:
            continue
        if object_id not in valid_ids:
            continue
        input_objects = [
            item for item in parsed.get("_input_objects", [])
            if isinstance(item, dict)
        ]
        object_label = next(
            (item.get("label") for item in input_objects
             if int(item.get("id", -1)) == object_id),
            "",
        )
        object_label = distinguish_duplicate_label(
            object_id, object_label, input_objects
        )
        text = conversational_object_text(
            row.get("text"),
            object_id=object_id,
            object_label=object_label,
            sequence_index=len(object_rows),
            label_relation=row.get("label_relation"),
            observed_name=row.get("observed_name"),
        )
        if not text:
            continue
        object_rows.append({
            "object_id": object_id,
            "text": text,
            "confidence": clamp(float(row.get("confidence") or 0.6), 0, 1),
        })

    relation_rows = []
    for row in parsed.get("relationship_descriptions") or []:
        if not isinstance(row, dict):
            continue
        try:
            subject_id = int(row.get("subject_id"))
            object_id = int(row.get("object_id"))
        except Exception:
            continue
        if subject_id not in valid_ids or object_id not in valid_ids:
            continue
        text = localize_narration_text(row.get("text"))
        if not text:
            continue
        relation_rows.append({
            "subject_id": subject_id,
            "object_id": object_id,
            "relation": canonical(row.get("relation")),
            "text": text[:100],
            "confidence": clamp(float(row.get("confidence") or 0.6), 0, 1),
        })

    return {
        "scene_summary": scene_summary,
        "object_descriptions": object_rows[:VLM_COMBINED_MAX_OBJECTS],
        "relationship_descriptions": relation_rows[:4],
        "unboxed_observation": None,
    }


_LABELS_ZH = {
    "ac remotecontrol": "冷氣遙控器",
    "bottle alcohol spray": "酒精噴瓶",
    "chair surface": "椅面",
    "cotton swab": "棉花棒",
    "cotton swabs pp": "棉花棒包裝",
    "disposable mask": "一次性口罩",
    "gauze pp": "紗布包裝",
    "saline": "生理食鹽水",
    "syringe nipro": "NIPRO 針筒",
    "waterproof bandages ppb": "PPB 防水繃帶",
    "bandages": "防水繃帶",
    "waterproof bandages": "防水繃帶",
    "towels": "毛巾",
    "saline bag": "生理食鹽水袋",
    "saline bottle": "生理食鹽水瓶",
    "nipro syringe": "NIPRO 針筒",
    "ppb waterproof bandage": "PPB 防水繃帶",
    "towel": "毛巾",
    "alcohol spray": "酒精噴瓶",
    "alcohol spray bottle": "酒精噴瓶",
    "alcohol bottle": "酒精瓶",
    "spray bottle": "噴瓶",
    "bandage": "防水繃帶",
    "waterproof bandage": "防水繃帶",
    "saline": "生理食鹽水",
    "saline solution": "生理食鹽水",
    "normal saline": "生理食鹽水",
    "cotton swab": "棉花棒",
    "cotton bud": "棉花棒",
    "gauze": "紗布",
    "mask": "口罩",
    "face mask": "口罩",
    "remote control": "遙控器",
    "ear thermometer": "耳溫槍",
    "weighing scale": "電子磅秤",
    "privacy curtain": "病房窗簾",
    "curtain": "病房窗簾",
    "bedside table": "床頭櫃",
    "bedside cabinet": "床頭櫃",
    "hospital bed": "病床",
    "bed": "病床",
    "telephone": "電話座機",
    "phone": "電話座機",
    "landline telephone": "電話座機",
    "syringe": "針筒",
    "syringe nipro": "NIPRO 針筒",
    "waterproof bandages ppb": "PPB 防水繃帶",
    "waterproof bandage ppb": "PPB 防水繃帶",
    "chair surface": "椅面",
    "cotton swab": "棉花棒",
    "cotton swabs": "棉花棒",
    "bottle alcohol spray": "酒精噴瓶",
    "alcohol spray bottle": "酒精噴瓶",
    "plate": "盤子",
    "bowl": "碗",
    "spoon": "湯匙",
    "fork": "叉子",
    "knife": "餐刀",
    "chopsticks": "筷子",
    "straw": "吸管",
    "plastic cup": "塑膠杯",
    "paper cup": "紙杯",
    "bottle": "瓶子",
    "can": "鐵鋁罐",
    "chip bag": "零食包裝袋",
    "wasted food": "食物殘餘",
    "food residue": "食物殘渣",
    "earphone": "藍芽耳機",
    "box": "箱子",
    "fallen box": "倒落箱子",
    "dog": "犬隻",
    "unknown": "未知物件",
}


def label_zh(value: Any) -> str:
    raw = str(value or "").strip()
    key = canonical(raw)
    translated = _LABELS_ZH.get(key)
    if translated:
        return translated

    # Unknown YOLO classes are not shown as raw English in the UI.
    if re.search(r"[A-Za-z]", raw):
        return "偵測物件"
    return raw or "物件"



def load_demo_description_bank(
    path: Path = DEMO_DESCRIPTION_PATH,
) -> dict[str, Any]:
    """Load and validate the assisted-mode description bank."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到展示描述庫：{path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"展示描述庫 JSON 格式錯誤：{path} "
            f"line={exc.lineno} column={exc.colno}"
        ) from exc

    if not isinstance(config, dict):
        raise RuntimeError("展示描述庫根節點必須是 JSON object")

    objects = config.get("objects")
    summaries = config.get("scene_summaries")
    if not isinstance(objects, dict) or not objects:
        raise RuntimeError("展示描述庫必須包含非空 objects")
    if not isinstance(summaries, list) or not summaries:
        raise RuntimeError("展示描述庫必須包含非空 scene_summaries")

    normalized_objects: dict[str, dict[str, Any]] = {}
    for raw_class_name, raw_config in objects.items():
        class_name = str(raw_class_name or "").strip().lower()
        if not class_name or not isinstance(raw_config, dict):
            raise RuntimeError(f"objects.{raw_class_name} 格式錯誤")

        raw_descriptions = raw_config.get("descriptions")
        if not isinstance(raw_descriptions, list) or not raw_descriptions:
            raise RuntimeError(
                f"objects.{class_name}.descriptions 必須是非空陣列"
            )

        descriptions: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        for index, raw_entry in enumerate(raw_descriptions, start=1):
            if isinstance(raw_entry, str):
                entry = {
                    "id": f"{class_name}_{index:02d}",
                    "text": raw_entry,
                    "tags": ["generic"],
                }
            elif isinstance(raw_entry, dict):
                entry = dict(raw_entry)
            else:
                raise RuntimeError(
                    f"objects.{class_name}.descriptions[{index - 1}] 格式錯誤"
                )

            text_value = str(entry.get("text") or "").strip()
            if not text_value:
                raise RuntimeError(
                    f"objects.{class_name}.descriptions[{index - 1}] 缺少 text"
                )
            if text_value in seen_texts:
                raise RuntimeError(
                    f"objects.{class_name} 包含重複描述：{text_value}"
                )
            seen_texts.add(text_value)

            tags = entry.get("tags")
            if not isinstance(tags, list):
                tags = []
            required_classes = entry.get(
                "requires_visible_classes"
            )
            if not isinstance(required_classes, list):
                required_classes = []

            descriptions.append({
                "id": str(
                    entry.get("id")
                    or f"{class_name}_{index:02d}"
                ).strip(),
                "text": text_value,
                "tags": [
                    str(tag).strip().lower()
                    for tag in tags
                    if str(tag).strip()
                ],
                "requires_visible_classes": [
                    str(value).strip().lower()
                    for value in required_classes
                    if str(value).strip()
                ],
                "requires_other_visible": bool(
                    entry.get("requires_other_visible", False)
                ),
            })

        normalized_objects[class_name] = {
            "label_zh": str(
                raw_config.get("label_zh")
                or label_zh(class_name)
            ).strip(),
            "descriptions": descriptions,
        }

    normalized_summaries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(summaries, start=1):
        if isinstance(raw_entry, str):
            entry = {
                "id": f"scene_{index:02d}",
                "text": raw_entry,
                "tags": ["generic"],
            }
        elif isinstance(raw_entry, dict):
            entry = dict(raw_entry)
        else:
            raise RuntimeError(
                f"scene_summaries[{index - 1}] 格式錯誤"
            )

        text_value = str(entry.get("text") or "").strip()
        if not text_value:
            raise RuntimeError(
                f"scene_summaries[{index - 1}] 缺少 text"
            )
        tags = entry.get("tags")
        if not isinstance(tags, list):
            tags = []
        normalized_summaries.append({
            "id": str(
                entry.get("id")
                or f"scene_{index:02d}"
            ).strip(),
            "text": text_value,
            "tags": [
                str(tag).strip().lower()
                for tag in tags
                if str(tag).strip()
            ],
        })

    out = dict(config)
    out["objects"] = normalized_objects
    out["scene_summaries"] = normalized_summaries
    return out


def _demo_class_key(value: Any) -> str:
    return canonical(value).replace(" ", "_")


def _demo_object_tags(obj: dict) -> set[str]:
    """Create assisted-mode tags in the real-world orientation.

    coordinate_mapping=raw_top_is_real_right means the camera frame was not
    rotated before inference:
      raw top    -> real right
      raw bottom -> real left
      raw left   -> real upper
      raw right  -> real lower
    """
    tags: set[str] = set()
    box = extract_box(obj)
    if box is None:
        return {"generic", "medium", "center", "middle"}

    raw_width = max(1.0, float(box[2]) - float(box[0]))
    raw_height = max(1.0, float(box[3]) - float(box[1]))

    frame_width = max(
        1.0,
        float(obj.get("frame_width") or max(float(box[2]), 1.0)),
    )
    frame_height = max(
        1.0,
        float(obj.get("frame_height") or max(float(box[3]), 1.0)),
    )

    raw_center_x_ratio = (
        (float(box[0]) + float(box[2])) / 2.0
    ) / frame_width
    raw_center_y_ratio = (
        (float(box[1]) + float(box[3])) / 2.0
    ) / frame_height

    coordinate_mapping = str(
        DEMO_DESCRIPTION_BANK.get("coordinate_mapping")
        or "identity"
    ).strip().lower()

    if coordinate_mapping == "raw_top_is_real_right":
        # Rotate raw coordinates clockwise into the real-world orientation.
        real_width = raw_height
        real_height = raw_width
        horizontal_ratio = 1.0 - raw_center_y_ratio
        vertical_ratio = raw_center_x_ratio
    else:
        real_width = raw_width
        real_height = raw_height
        horizontal_ratio = raw_center_x_ratio
        vertical_ratio = raw_center_y_ratio

    aspect = real_width / max(1.0, real_height)
    if aspect >= 1.30:
        tags.add("wide")
    elif aspect <= 0.77:
        tags.add("tall")
    else:
        tags.add("compact")

    area_ratio = (
        raw_width * raw_height
    ) / max(1.0, frame_width * frame_height)
    if area_ratio < 0.025:
        tags.add("small")
    elif area_ratio < 0.12:
        tags.add("medium")
    else:
        tags.add("large")

    if horizontal_ratio < 0.34:
        tags.add("left")
    elif horizontal_ratio > 0.66:
        tags.add("right")
    else:
        tags.add("center")

    if vertical_ratio < 0.34:
        tags.add("upper")
    elif vertical_ratio > 0.66:
        tags.add("lower")
    else:
        tags.add("middle")

    return tags


def _demo_choose_entry(
    entries: list[dict[str, Any]],
    *,
    tags: set[str],
    selection_key: str,
    visible_classes: set[str] | None = None,
    current_class: str = "",
) -> dict[str, Any]:
    """Choose only descriptions supported by the current YOLO scene.

    A description may declare:
    - requires_visible_classes: every listed YOLO class must currently exist.
    - requires_other_visible: at least one class other than the current object
      must currently exist.

    Entries without these fields remain normal appearance-only candidates.
    """
    if not entries:
        raise RuntimeError("展示描述候選為空")

    normalized_visible = {
        _demo_class_key(value)
        for value in (visible_classes or set())
        if str(value or "").strip()
    }
    normalized_current = _demo_class_key(current_class)

    eligible_entries: list[dict[str, Any]] = []
    for entry in entries:
        required = {
            _demo_class_key(value)
            for value in (
                entry.get("requires_visible_classes") or []
            )
            if str(value or "").strip()
        }
        if not required.issubset(normalized_visible):
            continue

        if entry.get("requires_other_visible"):
            other_visible = normalized_visible - {
                normalized_current
            }
            if not other_visible:
                continue

        eligible_entries.append(entry)

    # Normally appearance-only entries guarantee a fallback. Keep this guard so
    # a malformed description bank cannot stop the 5000 service.
    if not eligible_entries:
        eligible_entries = [
            entry
            for entry in entries
            if not entry.get("requires_visible_classes")
            and not entry.get("requires_other_visible")
        ] or list(entries)

    scored: list[tuple[int, dict[str, Any]]] = []
    for entry in eligible_entries:
        entry_tags = {
            str(tag).strip().lower()
            for tag in (entry.get("tags") or [])
            if str(tag).strip()
        }
        score = len(tags & entry_tags)
        if "generic" in entry_tags:
            score += 0
        scored.append((score, entry))

    best_score = max(score for score, _ in scored)
    candidates = [
        entry
        for score, entry in scored
        if score == best_score
    ]
    digest = hashlib.sha256(
        f"{DEMO_SEED}:{selection_key}".encode("utf-8")
    ).hexdigest()
    index = int(digest[:12], 16) % len(candidates)
    return dict(candidates[index])


def _demo_join_chinese(items: list[str]) -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]}與{values[1]}"
    return "、".join(values[:-1]) + f"與{values[-1]}"


def _demo_scene_context(
    objects: list[dict],
) -> dict[str, str]:
    """Summarize distribution and spacing from real YOLO geometry."""
    regions: set[str] = set()
    boxes: list[list[float]] = []

    for obj in objects:
        tags = _demo_object_tags(obj)
        if "left" in tags:
            regions.add("左側")
        elif "right" in tags:
            regions.add("右側")
        else:
            regions.add("中央")

        box = extract_box(obj)
        if box is not None:
            boxes.append([float(value) for value in box[:4]])

    region_order = ["左側", "中央", "右側"]
    ordered_regions = [
        region for region in region_order
        if region in regions
    ]

    if len(ordered_regions) >= 3:
        distribution = "物品大致分布於椅面左側、中央與右側"
    elif len(ordered_regions) == 2:
        distribution = (
            "物品主要分布於椅面"
            + _demo_join_chinese(ordered_regions)
        )
    elif ordered_regions:
        distribution = f"物品集中於椅面{ordered_regions[0]}"
    else:
        distribution = "物品分布狀態目前無法可靠判定"

    overlap_pairs = 0
    near_pairs = 0

    for first_index in range(len(boxes)):
        for second_index in range(first_index + 1, len(boxes)):
            first = boxes[first_index]
            second = boxes[second_index]

            if (
                _intersection_area(first, second) > 0
                or iou(first, second) >= 0.02
            ):
                overlap_pairs += 1
                continue

            first_short = max(
                1.0,
                min(
                    first[2] - first[0],
                    first[3] - first[1],
                ),
            )
            second_short = max(
                1.0,
                min(
                    second[2] - second[0],
                    second[3] - second[1],
                ),
            )
            near_limit = max(
                8.0,
                min(first_short, second_short) * 0.16,
            )
            if _box_edge_gap(first, second) <= near_limit:
                near_pairs += 1

    if overlap_pairs > 0:
        spacing = (
            "部分物件存在局部重疊，"
            "但主要輪廓仍可分辨"
        )
    elif near_pairs > 0:
        spacing = (
            "部分物件彼此接近，"
            "主要輪廓仍保持清楚"
        )
    else:
        spacing = (
            "物件之間保有間隔，"
            "沒有形成明顯堆疊"
        )

    assessment = (
        "整體屬於可依規則逐項分類整理的臨時置物場景"
    )

    return {
        "distribution": distribution,
        "spacing": spacing,
        "assessment": assessment,
    }


def build_assisted_demo_narration(
    objects: list[dict],
    *,
    scene_revision: int,
    signature: str,
) -> dict[str, Any]:
    """Build deterministic green and purple text from real YOLO objects."""
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    visible_classes = {
        _demo_class_key(
            obj.get("label")
            or obj.get("class_name")
            or obj.get("name")
        )
        for obj in objects
        if isinstance(obj, dict)
    }

    for obj in sorted(
        objects,
        key=lambda item: int(item.get("id") or 0),
    )[:VLM_COMBINED_MAX_OBJECTS]:
        try:
            object_id = int(obj.get("id"))
        except Exception:
            continue

        class_name = _demo_class_key(
            obj.get("label")
            or obj.get("class_name")
            or obj.get("name")
        )
        object_config = (
            DEMO_DESCRIPTION_BANK.get("objects") or {}
        ).get(class_name)
        tags = _demo_object_tags(obj)

        if isinstance(object_config, dict):
            entry = _demo_choose_entry(
                list(object_config.get("descriptions") or []),
                tags=tags,
                selection_key=(
                    f"object:{scene_revision}:{signature}:"
                    f"{object_id}:{class_name}:{','.join(sorted(tags))}:"
                    f"{','.join(sorted(visible_classes))}"
                ),
                visible_classes=visible_classes,
                current_class=class_name,
            )
            label = str(
                object_config.get("label_zh")
                or label_zh(class_name)
            ).strip()
            text_value = str(entry.get("text") or "").strip()
            description_id = str(entry.get("id") or "")
        else:
            label = label_zh(class_name)
            text_value = f"{label}外觀輪廓清楚，系統已完成辨識。"
            description_id = "fallback"

        text_value = normalize_narration_sentence(text_value)

        rows.append({
            "object_id": object_id,
            "text": text_value,
            "confidence": 0.85,
            "source": "demo_description_bank",
            "description_id": description_id,
            "selection_tags": sorted(tags),
        })
        labels.append(label)

    unique_labels = list(dict.fromkeys(labels))
    objects_text = "、".join(unique_labels[:3]) or "已標記物品"
    count = len(rows)
    scene_tag = (
        "single"
        if count == 1
        else "few"
        if count <= 3
        else "many"
    )
    scene_context = _demo_scene_context(objects)
    scene_entry = _demo_choose_entry(
        list(DEMO_DESCRIPTION_BANK.get("scene_summaries") or []),
        tags={scene_tag},
        selection_key=(
            f"scene:{scene_revision}:{signature}:"
            f"{count}:{objects_text}:"
            f"{scene_context['distribution']}:"
            f"{scene_context['spacing']}"
        ),
    )
    scene_summary = str(scene_entry.get("text") or "").format(
        objects=objects_text,
        count=count,
        distribution=scene_context["distribution"],
        spacing=scene_context["spacing"],
        assessment=scene_context["assessment"],
    ).strip()
    scene_summary = normalize_narration_sentence(scene_summary)

    return {
        "object_descriptions": rows,
        "scene_summary": scene_summary,
        "scene_summary_id": str(scene_entry.get("id") or ""),
    }


def publish_assisted_demo_output(
    session: SessionState,
    *,
    signature: str,
    objects: list[dict],
) -> bool:
    """Publish assisted rows progressively without calling Ollama."""
    started_at = time.time()
    object_copy = json.loads(json.dumps(
        sorted(
            objects,
            key=lambda item: int(item.get("id") or 0),
        )[:VLM_COMBINED_MAX_OBJECTS]
    ))
    expected_ids = [
        int(obj["id"])
        for obj in object_copy
        if obj.get("id") is not None
    ]
    if not expected_ids:
        return False

    with _SESSION_LOCK:
        if (
            session.last_signature != signature
            or session.last_enriched_signature == signature
            or session.qwen_pending
        ):
            return False

        scene_revision = int(session.scene_revision)
        session.qwen_job_token += 1
        job_token = int(session.qwen_job_token)

        session.qwen_signature = signature
        session.qwen_pending = True
        session.qwen_error = ""
        session.qwen_streaming = True
        session.qwen_stream_started_at = started_at
        session.qwen_first_result_at = 0.0
        session.qwen_last_partial_at = 0.0
        session.qwen_expected_object_ids = list(expected_ids)
        session.qwen_received_object_ids = []
        session.qwen_stream_scene_summary = ""
        session.last_qwen_attempt_at = started_at
        session.last_result = {
            "scene_summary": "",
            "object_descriptions": [],
            "relationship_descriptions": [],
            "unboxed_observation": None,
            "source": "assisted_demo_description_bank",
        }

        expected_id_set = set(expected_ids)
        session.object_description_cache = {
            object_id: row
            for object_id, row in session.object_description_cache.items()
            if object_id not in expected_id_set
        }

        latest = session.latest_api_result
        if (
            isinstance(latest, dict)
            and int(latest.get("scene_revision") or -1) == scene_revision
        ):
            narration = dict(latest.get("narration") or {})
            narration["object_descriptions"] = []
            narration["source"] = "assisted_demo_description_bank"
            latest["narration"] = narration
            latest["qwen_streaming"] = True
            latest["qwen_pending"] = True
            latest["qwen_expected_object_count"] = len(expected_ids)
            latest["qwen_received_object_count"] = 0
            latest["qwen_expected_object_ids"] = list(expected_ids)
            latest["qwen_received_object_ids"] = []
            latest["qwen_first_result_sec"] = None
            latest["object_stage_complete"] = False
            latest["environment_stage_complete"] = False
            latest["output_mode"] = runtime_output_mode()
            latest["demo_output"] = True
            latest["demo_mode"] = "assisted"

    generated = build_assisted_demo_narration(
        object_copy,
        scene_revision=scene_revision,
        signature=signature,
    )
    rows = list(generated.get("object_descriptions") or [])
    scene_summary = str(generated.get("scene_summary") or "").strip()

    def job_is_current() -> bool:
        return bool(
            session.last_signature == signature
            and int(session.scene_revision) == scene_revision
            and int(session.qwen_job_token) == job_token
            and assisted_output_enabled()
        )

    def worker() -> None:
        completed_rows: list[dict] = []
        stale = False

        try:
            for row in rows:
                if ASSISTED_OUTPUT_GAP_SEC > 0:
                    time.sleep(ASSISTED_OUTPUT_GAP_SEC)

                with _SESSION_LOCK:
                    if not job_is_current():
                        stale = True
                        return

                    now_partial = time.time()
                    object_id = int(row["object_id"])
                    row_copy = dict(row)
                    completed_rows.append(row_copy)
                    session.object_description_cache[object_id] = row_copy
                    session.qwen_received_object_ids = [
                        int(item["object_id"])
                        for item in completed_rows
                    ]
                    session.qwen_last_partial_at = now_partial

                    if session.qwen_first_result_at <= 0:
                        session.qwen_first_result_at = now_partial

                    session.last_object_emit_at = now_partial
                    session.last_result = {
                        "scene_summary": "",
                        "object_descriptions": [
                            dict(item) for item in completed_rows
                        ],
                        "relationship_descriptions": [],
                        "unboxed_observation": None,
                        "source": "assisted_demo_description_bank",
                    }

                    latest = session.latest_api_result
                    if (
                        isinstance(latest, dict)
                        and int(latest.get("scene_revision") or -1)
                            == scene_revision
                    ):
                        narration = dict(latest.get("narration") or {})
                        narration["object_descriptions"] = [
                            dict(item) for item in completed_rows
                        ]
                        narration["source"] = (
                            "assisted_demo_description_bank"
                        )
                        latest["narration"] = narration
                        latest["qwen_streaming"] = True
                        latest["qwen_pending"] = True
                        latest["qwen_expected_object_count"] = len(
                            expected_ids
                        )
                        latest["qwen_received_object_count"] = len(
                            completed_rows
                        )
                        latest["qwen_received_object_ids"] = list(
                            session.qwen_received_object_ids
                        )
                        latest["qwen_first_result_sec"] = round(
                            session.qwen_first_result_at
                            - session.qwen_stream_started_at,
                            3,
                        )
                        latest["object_stage_complete"] = (
                            len(completed_rows) >= len(expected_ids)
                        )
                        latest["environment_stage_complete"] = False

                print(
                    "[assisted-demo][object] "
                    f"revision={scene_revision} "
                    f"id={row.get('object_id')} "
                    f"description={row.get('description_id')} "
                    f"elapsed={time.time() - started_at:.3f}s",
                    flush=True,
                )

            # Keep a final pause between the last green card and purple summary.
            if ASSISTED_OUTPUT_GAP_SEC > 0:
                time.sleep(ASSISTED_OUTPUT_GAP_SEC)

            with _SESSION_LOCK:
                if not job_is_current():
                    stale = True
                    return

                completed_at = time.time()
                final_rows = [dict(item) for item in completed_rows]

                session.last_result = {
                    "scene_summary": "",
                    "object_descriptions": final_rows,
                    "relationship_descriptions": [],
                    "unboxed_observation": None,
                    "source": "assisted_demo_description_bank",
                }
                session.last_enriched_signature = signature
                session.last_general_signature = signature
                session.general_idle_ready = scene_summary
                session.qwen_stream_scene_summary = scene_summary
                session.qwen_received_object_ids = list(expected_ids)
                session.last_narrated_at = completed_at
                session.qwen_error = ""
                session.qwen_pending = False
                session.qwen_streaming = False

                latest = session.latest_api_result
                if (
                    isinstance(latest, dict)
                    and int(latest.get("scene_revision") or -1)
                        == scene_revision
                ):
                    narration = dict(latest.get("narration") or {})
                    narration["object_descriptions"] = final_rows
                    narration["source"] = (
                        "assisted_demo_description_bank"
                    )
                    latest["narration"] = narration
                    latest["qwen_streaming"] = False
                    latest["qwen_pending"] = False
                    latest["qwen_expected_object_count"] = len(expected_ids)
                    latest["qwen_received_object_count"] = len(final_rows)
                    latest["qwen_expected_object_ids"] = list(expected_ids)
                    latest["qwen_received_object_ids"] = list(expected_ids)
                    latest["object_stage_complete"] = True
                    latest["environment_stage_complete"] = True
                    latest["output_mode"] = runtime_output_mode()
                    latest["demo_output"] = True
                    latest["demo_mode"] = "assisted"

            print(
                "[assisted-demo] complete "
                f"revision={scene_revision} objects={len(completed_rows)} "
                f"scene_summary={generated.get('scene_summary_id')} "
                f"total={time.time() - started_at:.3f}s",
                flush=True,
            )
        except Exception as exc:
            with _SESSION_LOCK:
                if int(session.qwen_job_token) == job_token:
                    session.qwen_error = f"{type(exc).__name__}: {exc}"
            print(
                "[assisted-demo][WARN] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            with _SESSION_LOCK:
                if (
                    not stale
                    and int(session.qwen_job_token) == job_token
                ):
                    session.qwen_pending = False
                    session.qwen_streaming = False

    threading.Thread(
        target=worker,
        name=f"scene-narrator-assisted-{job_token}",
        daemon=True,
    ).start()
    return True


if NARRATOR_OUTPUT_MODE in {"auto", "assisted"}:
    DEMO_DESCRIPTION_BANK = load_demo_description_bank()


def sanitize_detector_class_text(value: Any) -> str:
    """Translate any detector-class fragments that leaked into narration."""
    text_value = str(value or "").strip()
    if not text_value:
        return ""

    for english, chinese in sorted(
        _LABELS_ZH.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text_value = re.sub(
            rf"(?<![A-Za-z]){re.escape(english)}(?![A-Za-z])",
            chinese,
            text_value,
            flags=re.IGNORECASE,
        )

    return text_value


def localize_narration_text(value: Any) -> str:
    text_value = str(value or "").strip()
    # Replace longer English labels first to avoid partial replacements.
    for english, chinese in sorted(
        _LABELS_ZH.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text_value = re.sub(
            rf"(?<![A-Za-z]){re.escape(english)}(?![A-Za-z])",
            chinese,
            text_value,
            flags=re.IGNORECASE,
        )
    return sanitize_detector_class_text(text_value)


_RELATION_ZH = {
    "near": "與{object}相距很近",
    "left_of": "位於{object}左側",
    "right_of": "位於{object}右側",
    "above": "位於{object}上方",
    "below": "位於{object}下方",
    "inside": "位於{object}內部",
    "overlapping": "與{object}部分重疊",
}


def object_ref(object_id: int, objects_by_id: dict[int, dict]) -> str:
    obj = objects_by_id.get(object_id) or {}
    return label_zh(obj.get("label") or "物件")


_LAST_FAST_PATTERN_INDEX: dict[str, int] = {}

def _choose_fast_pattern(kind: str, patterns: tuple[str, ...]) -> str:
    previous = _LAST_FAST_PATTERN_INDEX.get(kind)
    choices = [(i, p) for i, p in enumerate(patterns) if i != previous]
    if not choices:
        choices = list(enumerate(patterns))
    index, pattern = random.choice(choices)
    _LAST_FAST_PATTERN_INDEX[kind] = index
    return pattern

PLACEMENT_RULE_PATH = Path(
    os.environ.get("TAIROS_PLACEMENT_RULES_PATH")
    or Path(__file__).resolve().parents[1] / "agent" / "rules.json"
).expanduser().resolve()


def load_placement_rules(
    path: Path = PLACEMENT_RULE_PATH,
) -> dict[str, dict[str, str]]:
    """讀取 agent/rules.json；只建立整理建議，不執行任何動作。"""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到整理規則：{path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"整理規則 JSON 格式錯誤：{path} "
            f"line={exc.lineno} column={exc.colno}"
        ) from exc

    destinations = config.get("destinations")
    objects = config.get("objects")
    if not isinstance(destinations, dict) or not isinstance(objects, dict):
        raise RuntimeError(
            "rule.json 必須包含 destinations 與 objects 兩個物件"
        )

    rules: dict[str, dict[str, str]] = {}
    for class_name, object_rule in objects.items():
        if not isinstance(object_rule, dict):
            raise RuntimeError(f"objects.{class_name} 必須是物件")

        destination = str(
            object_rule.get("destination") or ""
        ).strip()
        destination_rule = destinations.get(destination)
        if not isinstance(destination_rule, dict):
            raise RuntimeError(
                f"objects.{class_name}.destination={destination!r} 不存在"
            )

        object_label = str(
            object_rule.get("object_label") or ""
        ).strip()
        destination_label = str(
            destination_rule.get("label_zh") or ""
        ).strip()
        suggested_action = str(
            destination_rule.get("suggested_action") or ""
        ).strip()

        if not object_label or not destination_label or not suggested_action:
            raise RuntimeError(
                f"objects.{class_name} 或 destinations.{destination} 缺少必要欄位"
            )

        rules[str(class_name).strip().lower()] = {
            "object_label": object_label,
            "destination": destination,
            "destination_label": destination_label,
            "suggested_action": suggested_action,
        }

    return rules


PLACEMENT_RULES = load_placement_rules()


def _placement_object_class(object_data: dict) -> str:
    return str(
        object_data.get("class_name")
        or object_data.get("label")
        or object_data.get("name")
        or ""
    ).strip().lower()


def _placement_object_id(object_data: dict):
    for key in ("object_id", "track_id", "id", "instance_id"):
        if object_data.get(key) is not None:
            return object_data[key]
    return None


def build_placement_recommendations(
    objects: list[dict],
) -> list[dict]:
    """產生 UI／下游 metadata，不呼叫 Agent Tool 或機器手臂。"""
    recommendations = []

    for object_data in objects:
        class_name = _placement_object_class(object_data)
        rule = PLACEMENT_RULES.get(class_name)
        if rule is None:
            continue

        destination = rule["destination"]
        destination_label = rule["destination_label"]
        object_label = rule["object_label"]
        text = (
            f"{object_label}建議丟入垃圾桶。"
            if destination == "trash_can"
            else f"{object_label}建議放入{destination_label}。"
        )

        recommendations.append({
            "object_id": _placement_object_id(object_data),
            "class_name": class_name,
            "object_label": object_label,
            "destination": destination,
            "destination_label": destination_label,
            "suggested_action": rule["suggested_action"],
            "robot_xyz": object_data.get("robot_xyz"),
            "yaw_deg": object_data.get("yaw_deg"),
            "decision_source": "fixed_placement_rule",
            "text": text,
        })

    return recommendations


def build_placement_summary(
    recommendations: list[dict],
) -> str:
    grouped: dict[str, dict] = {}

    for item in recommendations:
        destination = str(item.get("destination") or "")
        label = str(item.get("object_label") or "").strip()
        if not destination or not label:
            continue

        group = grouped.setdefault(destination, {
            "destination_label": str(
                item.get("destination_label") or ""
            ),
            "counts": {},
            "order": [],
        })
        if label not in group["counts"]:
            group["counts"][label] = 0
            group["order"].append(label)
        group["counts"][label] += 1

    sentences = []
    for destination, group in grouped.items():
        labels = [
            f"{group['counts'][label]}個{label}"
            if group["counts"][label] > 1
            else label
            for label in group["order"]
        ]
        object_text = (
            labels[0]
            if len(labels) == 1
            else "、".join(labels[:-1]) + "與" + labels[-1]
        )
        if not object_text:
            continue

        sentences.append(
            f"{object_text}建議丟入垃圾桶"
            if destination == "trash_can"
            else f"{object_text}建議放入{group['destination_label']}"
        )

    return "；".join(sentences) + "。" if sentences else ""


def build_object_preview_descriptions(
    objects: list[dict],
) -> list[dict]:
    """Immediate green placeholders shown before the first VLM token arrives."""
    rows = []
    for obj in sorted(
        objects,
        key=lambda item: int(item.get("id") or 0),
    )[:VLM_COMBINED_MAX_OBJECTS]:
        try:
            object_id = int(obj.get("id"))
        except Exception:
            continue
        rows.append({
            "object_id": object_id,
            "class_name": str(obj.get("label") or ""),
            "object_label": label_zh(obj.get("label")),
            "text": "外觀分析中…",
            "provisional": True,
        })
    return rows


def complete_empty_scene_without_vlm(
    session: SessionState,
    *,
    signature: str,
) -> None:
    """Finish an empty semantic scene immediately without spending a VLM call."""
    with _SESSION_LOCK:
        if session.last_signature != signature:
            return
        if (
            session.last_enriched_signature == signature
            and session.last_general_signature == signature
        ):
            return
        if session.qwen_pending:
            session.qwen_job_token += 1
            session.qwen_pending = False

        session.last_result = {
            "scene_summary": "",
            "object_descriptions": [],
            "relationship_descriptions": [],
            "unboxed_observation": None,
        }
        session.object_description_cache.clear()
        session.last_enriched_signature = signature
        session.last_general_signature = signature
        session.general_idle_ready = VLM_EMPTY_SCENE_TEXT
        session.qwen_streaming = False
        session.qwen_stream_started_at = 0.0
        session.qwen_first_result_at = 0.0
        session.qwen_last_partial_at = 0.0
        session.qwen_expected_object_ids = []
        session.qwen_received_object_ids = []
        session.qwen_stream_scene_summary = VLM_EMPTY_SCENE_TEXT
        session.qwen_error = ""


def build_fast_narration(
    objects: list[dict],
    relations: list[dict],
    *,
    presence_object_ids: set[int] | None = None,
) -> dict:
    """Generate concise event narration without waiting for Qwen.

    Blue text is an event acknowledgement, not a continuous paraphraser.  The
    caller passes only newly appeared/increased classes through object IDs.
    """
    objects_by_id = {int(obj["id"]): obj for obj in objects}
    summary = ""
    presence_rows = []
    presence_source = objects
    if presence_object_ids is not None:
        presence_source = [
            obj for obj in objects
            if int(obj.get("id") or -1) in presence_object_ids
        ]

    for obj in presence_source[:2]:
        object_id = int(obj["id"])
        qualified_label = distinguish_duplicate_label(
            object_id, str(obj.get("label") or "物件"), objects
        )
        object_name = label_zh(qualified_label)
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        if cx < 0.36 * max(1.0, float(obj.get("frame_width") or 1280)):
            position = "畫面左側"
        elif cx > 0.64 * max(1.0, float(obj.get("frame_width") or 1280)):
            position = "畫面右側"
        else:
            position = "畫面中央"

        presence_rows.append({
            "object_id": object_id,
            "text": f"偵測到{object_name}，位於{position}。",
            "confidence": float(obj.get("confidence") or 0.7),
        })

    relation_rows = []
    used_pairs = set()
    relation_priority = {
        "inside": 6,
        "overlapping": 5,
        "near": 4,
        "above": 3,
        "below": 3,
        "left_of": 2,
        "right_of": 2,
    }
    for row in sorted(
        relations,
        key=lambda r: (
            relation_priority.get(str(r.get("relation")), 0),
            float(r.get("confidence") or 0),
        ),
        reverse=True,
    ):
        try:
            subject_id = int(row["subject_id"])
            object_id = int(row["object_id"])
        except Exception:
            continue
        relation = str(row.get("relation") or "")
        if relation not in _RELATION_ZH:
            continue

        # Avoid describing the same pair twice in opposite directions.
        pair_key = tuple(sorted((subject_id, object_id)))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        subject = object_ref(subject_id, objects_by_id)
        target = object_ref(object_id, objects_by_id)
        phrase = _RELATION_ZH[relation].format(object=target)
        relation_text = f"{subject}{phrase}。"
        # Do not force a transition word here. The first visible narration must
        # never begin with 「另外」 when there is no preceding observation.
        relation_rows.append({
            "subject_id": subject_id,
            "object_id": object_id,
            "relation": relation,
            "text": relation_text,
            "confidence": float(row.get("confidence") or 0.7),
        })
        if len(relation_rows) >= 1:
            break

    return {
        "scene_summary": summary,
        "presence_descriptions": presence_rows,
        "object_descriptions": [],
        "relationship_descriptions": relation_rows,
        "source": "fast_presence_and_geometry",
    }


def build_idle_narration(
    session: SessionState,
    objects: list[dict],
    enriched: dict,
) -> dict | None:
    """Emit one fresh, non-repetitive observation while the scene is static."""
    now = time.time()
    if now - session.last_idle_emit_at < IDLE_NARRATION_INTERVAL_SEC:
        return None

    object_rows = list(enriched.get("object_descriptions") or [])
    if not object_rows:
        return None

    # Make only one supplementary pass per unchanged scene.
    if session.idle_cycle_index >= len(object_rows):
        return None
    row = object_rows[session.idle_cycle_index]
    session.idle_cycle_index += 1
    session.last_idle_emit_at = now

    text_value = str(row.get("text") or "").strip()
    if not text_value:
        return None

    # Most idle lines start directly; only a minority receive a light transition.
    prefixes = ("", "", "", "再補充一點，", "另外可以看到，")
    prefix = random.choice(prefixes)

    cleaned = re.sub(
        r"^(我看到|我注意到|仔細看|畫面裡還有|接著我注意到|再看一下|換個角度看|還可以注意到|仔細觀察)[，,：:]?",
        "",
        text_value,
    ).strip()

    if not cleaned:
        return None

    return {
        "type": "idle",
        "text": prefix + cleaned,
        "object_id": row.get("object_id"),
    }


def merge_narration(fast: dict, enriched: dict | None) -> dict:
    if not enriched:
        return fast

    merged = {
        "scene_summary": (
            str(enriched.get("scene_summary") or "").strip()
            or fast.get("scene_summary")
            or ""
        ),
        "presence_descriptions": list(fast.get("presence_descriptions") or [])[:3],
        "object_descriptions": list(enriched.get("object_descriptions") or [])[:VLM_COMBINED_MAX_OBJECTS],
        "relationship_descriptions": [],
        "unboxed_observation": enriched.get("unboxed_observation"),
        "source": "presence_then_object_then_geometry",
    }

    seen_text = set()
    for row in list(fast.get("relationship_descriptions") or []) + list(
        enriched.get("relationship_descriptions") or []
    ):
        text_value = str(row.get("text") or "").strip()
        if not text_value or text_value in seen_text:
            continue
        seen_text.add(text_value)
        merged["relationship_descriptions"].append(row)
        if len(merged["relationship_descriptions"]) >= 1:
            break
    return merged


def call_qwen_general_scene_comment(
    image: Image.Image,
    objects: list[dict],
    recent_comments: list[str],
) -> str:
    """
    產生紫色層的整體場景描述。

    設計目標：
    - 描述紅色陪病椅及椅子上、椅子附近可見的物品。
    - 能辨認畫面下方是機器人自己的夾爪或機械結構。
    - 允許使用「我看到」、「我注意到」等第一人稱說法。
    - 可描述顏色、形狀、位置與可能的材質。
    - 不確定時使用保留語氣，不強迫模型完全不命名。
    - 可以描述多個物件，也允許和先前輸出重複。
    - 忽略偵測框、中心點、距離文字和其他介面標註。
    """

    # 保留既有函式介面；紫色描述仍只根據原始完整影像判斷，
    # 不把 YOLO 類別、數量、座標或 objects metadata 傳給 VLM。
    _ = objects
    _ = recent_comments

    prompt = {
    "scene_context": (
        "這是機器人第一人稱相機視角，主要朝向紅色陪病椅。"
        "畫面下方可能看見機器人自己的黑色夾爪。"
    ),
    "instruction": (
        "請用繁體中文簡短描述目前最重要的可見內容。"
        "優先說明紅色陪病椅、椅子上最明顯的物品，"
        "以及畫面下方是否看見自己的夾爪。"
        "最多描述兩種物品，不要逐項列舉。"
        "除非材質非常明顯，否則不要描述材質。"
        "不要描述氣氛、質感、設計感、整潔程度或無關背景。"
        "不確定物品名稱時，可使用『可能是』或『看起來像』。"
        "忽略偵測框、中心點、線條、文字與數字。"
        "不要提到座標、距離、信心值、標籤或物件編號。"
        "只輸出一句自然純文字，最多45個中文字。"
    ),
}

    payload = {
        "model": QWEN_MODEL,
        # 紫色描述需要自然度，但不要過度自由而產生幻覺。
        "temperature": max(0.45, QWEN_TEMPERATURE),
        "top_p": 0.9,
        "max_tokens": VLM_GENERAL_MAX_TOKENS,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是機器人的視覺觀察模組。"
                    "使用自然的繁體中文描述目前看見的場景。"
                    "區分紅色陪病椅、椅子上的物品，以及畫面下方機器人自己的夾爪。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            prompt,
                            ensure_ascii=False,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url(
                                image,
                                max_edge=VLM_GENERAL_IMAGE_EDGE,
                            )
                        },
                    },
                ],
            },
        ],
    }

    req = urllib.request.Request(
        QWEN_BASE_URL + "/chat/completions",
        data=json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local",
        },
    )

    with urllib.request.urlopen(
        req,
        timeout=QWEN_TIMEOUT_SEC,
    ) as response:
        raw = response.read().decode("utf-8")

    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]

    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        )

    comment = localize_narration_text(str(content or ""))

    # 清除模型偶爾加入的引號、Markdown 或多餘空白。
    comment = re.sub(
        r"^[「『\"'`]+|[」』\"'`]+$",
        "",
        comment,
    ).strip()

    comment = re.sub(
        r"^(?:紫色描述|場景描述|觀察結果)[：:\s]*",
        "",
        comment,
    ).strip()

    comment = re.sub(
        r"^[\-•*]+\s*",
        "",
        comment,
    ).strip()

    # 保留自然中文標點，但移除換行與連續空白。
    comment = re.sub(
        r"\s+",
        "",
        comment,
    )

    # 若模型提到介面疊圖，視為不合格輸出。
    overlay_terms = (
        "YOLO",
        "yolo",
        "偵測器",
        "偵測框",
        "偵測結果",
        "框選",
        "方框",
        "邊框",
        "輪廓框",
        "綠色邊框",
        "青色邊框",
        "橘色邊框",
        "辨識標籤",
        "中心點",
        "十字標記",
        "信心值",
        "物件編號",
        "center:",
        "distance:",
        "yaw_deg:",
    )

    if any(term in comment for term in overlay_terms):
        raise RuntimeError(
            f"Qwen mentioned detector overlay: {comment}"
        )

    if not comment:
        raise RuntimeError(
            "Qwen returned an empty general scene comment"
        )

    comment = normalize_narration_sentence(comment)

    # 最多保留約 160 字，並儘量在完整句子處截斷。
    max_chars = 60

    if len(comment) > max_chars:
        clipped = comment[:max_chars]

        sentence_end = max(
            clipped.rfind("。"),
            clipped.rfind("！"),
            clipped.rfind("？"),
        )

        if sentence_end >= 40:
            comment = clipped[:sentence_end + 1]
        else:
            comment = clipped.rstrip("，、；：")
            if comment[-1] not in "。！？":
                comment += "。"

    return comment


def start_general_idle_comment(
    session: SessionState,
    *,
    signature: str,
    image: Image.Image,
    objects: list[dict],
) -> bool:
    if not GENERAL_IDLE_ENABLED:
        return False

    with _SESSION_LOCK:
        if session.general_idle_pending or session.general_idle_ready:
            return False
        if session.last_signature != signature:
            return False

    if not _QWEN_WORK_LOCK.acquire(blocking=False):
        return False

    with _SESSION_LOCK:
        if session.general_idle_pending or session.general_idle_ready:
            _QWEN_WORK_LOCK.release()
            return False
        session.general_idle_job_token += 1
        token = session.general_idle_job_token
        session.general_idle_pending = True
        recent = list(session.recent_general_idle)

    image_copy = image.copy()
    object_copy = json.loads(json.dumps(objects))

    def worker() -> None:
        try:
            comment = call_qwen_general_scene_comment(image_copy, object_copy, recent)
            comment_core = semantic_line_core(comment)
            duplicate = any(
                semantic_lines_similar(comment_core, semantic_line_core(old_comment))
                for old_comment in recent
                if str(old_comment or "").strip()
            )
            with _SESSION_LOCK:
                if session.last_signature == signature and session.general_idle_job_token == token:
                    # A deduplicated answer still completes this scene's purple
                    # stage; otherwise pairwise would be starved forever.
                    session.last_general_signature = signature
                    if duplicate:
                        print(f"[general-idle][dedup] skipped={comment}", flush=True)
                    else:
                        session.general_idle_ready = comment
        except Exception as exc:
            with _SESSION_LOCK:
                if session.general_idle_job_token == token:
                    session.qwen_error = f"general_idle_{type(exc).__name__}: {exc}"
        finally:
            with _SESSION_LOCK:
                if session.general_idle_job_token == token:
                    session.general_idle_pending = False
            _QWEN_WORK_LOCK.release()

    threading.Thread(
        target=worker,
        name=f"scene-narrator-general-idle-{token}",
        daemon=True,
    ).start()
    return True


def start_qwen_enrichment(
    session: SessionState,
    *,
    signature: str,
    image: Image.Image,
    objects: list[dict],
    relations: list[dict],
    force: bool = False,
) -> bool:
    """Start one contact-sheet streaming job without blocking YOLO requests."""
    now = time.time()
    assisted_mode = assisted_output_enabled()
    with _SESSION_LOCK:
        if session.last_signature != signature:
            return False
        if session.qwen_pending:
            return False
        if session.last_enriched_signature == signature:
            return False
        if not objects:
            return False

        # Assisted output is ready immediately after the scene has passed the
        # existing stable-object and scene-change gates.
        if not assisted_mode:
            if not force and now - session.last_scene_change_at < QWEN_ENRICH_DELAY_SEC:
                return False
            if now - session.last_qwen_attempt_at < QWEN_RETRY_INTERVAL_SEC:
                return False

    if assisted_mode:
        return publish_assisted_demo_output(
            session,
            signature=signature,
            objects=objects,
        )

    if not _QWEN_WORK_LOCK.acquire(blocking=False):
        return False

    object_copy = json.loads(json.dumps(
        sorted(
            objects,
            key=lambda item: int(item.get("id") or 0),
        )[:VLM_COMBINED_MAX_OBJECTS]
    ))
    expected_ids = [
        int(obj["id"])
        for obj in object_copy
        if obj.get("id") is not None
    ]

    with _SESSION_LOCK:
        if session.last_signature != signature or session.qwen_pending:
            _QWEN_WORK_LOCK.release()
            return False
        session.qwen_job_token += 1
        job_token = session.qwen_job_token
        session.qwen_pending = True
        session.qwen_signature = signature
        session.qwen_error = ""
        session.last_qwen_attempt_at = now
        session.qwen_streaming = True
        session.qwen_stream_started_at = now
        session.qwen_first_result_at = 0.0
        session.qwen_last_partial_at = 0.0
        session.qwen_expected_object_ids = list(expected_ids)
        session.qwen_received_object_ids = []
        session.qwen_stream_scene_summary = ""
        session.last_result = {}
        session.object_description_cache = {
            object_id: row
            for object_id, row in session.object_description_cache.items()
            if object_id not in set(expected_ids)
        }

    image_copy = image.copy()
    relation_copy = json.loads(json.dumps(relations))

    def publish_partial_object(raw_row: dict) -> None:
        raw = {
            "object_descriptions": [dict(raw_row)],
            "_input_objects": object_copy,
        }
        cleaned = clean_narration(
            raw,
            {int(obj["id"]) for obj in object_copy},
        )
        rows = cleaned.get("object_descriptions") or []
        if not rows:
            return
        row = dict(rows[0])
        try:
            object_id = int(row.get("object_id"))
        except Exception:
            return

        with _SESSION_LOCK:
            if (
                session.last_signature != signature
                or session.qwen_job_token != job_token
            ):
                return

            now_partial = time.time()
            session.object_description_cache[object_id] = row
            received_ids = sorted(
                object_id_value
                for object_id_value in expected_ids
                if object_id_value in session.object_description_cache
            )
            session.qwen_received_object_ids = received_ids
            session.qwen_last_partial_at = now_partial
            if session.qwen_first_result_at <= 0:
                session.qwen_first_result_at = now_partial
                session.last_object_emit_at = now_partial
                print(
                    "[qwen-stream] first-object "
                    f"id={object_id} "
                    f"elapsed={now_partial - session.qwen_stream_started_at:.3f}s",
                    flush=True,
                )

            partial_rows = [
                dict(session.object_description_cache[expected_id])
                for expected_id in expected_ids
                if expected_id in session.object_description_cache
            ]
            session.last_result = {
                "scene_summary": "",
                "object_descriptions": partial_rows,
                "relationship_descriptions": [],
                "unboxed_observation": None,
            }

            latest = session.latest_api_result
            if (
                isinstance(latest, dict)
                and int(latest.get("scene_revision") or -1)
                    == int(session.scene_revision)
            ):
                narration = dict(latest.get("narration") or {})
                narration["object_descriptions"] = partial_rows
                latest["narration"] = narration
                latest["qwen_streaming"] = True
                latest["qwen_pending"] = True
                latest["qwen_expected_object_count"] = len(expected_ids)
                latest["qwen_received_object_count"] = len(received_ids)
                latest["qwen_received_object_ids"] = list(received_ids)
                latest["qwen_first_result_sec"] = round(
                    session.qwen_first_result_at
                    - session.qwen_stream_started_at,
                    3,
                )

    def publish_scene_summary(summary: str) -> None:
        with _SESSION_LOCK:
            if (
                session.last_signature == signature
                and session.qwen_job_token == job_token
            ):
                session.qwen_stream_scene_summary = str(summary or "").strip()

    def worker() -> None:
        try:
            qwen_result = call_qwen(
                image_copy,
                object_copy,
                relation_copy,
                on_object=publish_partial_object,
                on_scene=publish_scene_summary,
            )
            qwen_result["_input_objects"] = object_copy
            enriched = clean_narration(
                qwen_result,
                {int(obj["id"]) for obj in object_copy},
            )

            scene_summary = str(
                enriched.get("scene_summary")
                or qwen_result.get("scene_summary")
                or session.qwen_stream_scene_summary
                or ""
            ).strip()
            if not scene_summary:
                scene_summary = (
                    "椅面上可見"
                    + "、".join(
                        label_zh(obj.get("label"))
                        for obj in object_copy[:4]
                    )
                    + "等物品。"
                )
            enriched["scene_summary"] = ""

            completed_at = time.time()
            with _SESSION_LOCK:
                if (
                    session.last_signature == signature
                    and session.qwen_job_token == job_token
                ):
                    for row in enriched.get("object_descriptions") or []:
                        try:
                            object_id = int(row.get("object_id"))
                        except Exception:
                            continue
                        session.object_description_cache[object_id] = dict(row)

                    final_rows = [
                        dict(session.object_description_cache[expected_id])
                        for expected_id in expected_ids
                        if expected_id in session.object_description_cache
                    ]
                    enriched["object_descriptions"] = final_rows

                    session.last_result = enriched
                    session.last_enriched_signature = signature
                    session.last_general_signature = signature
                    session.general_idle_ready = scene_summary
                    session.qwen_stream_scene_summary = scene_summary
                    session.qwen_received_object_ids = [
                        int(row["object_id"])
                        for row in final_rows
                    ]
                    if final_rows and session.last_object_emit_at <= 0:
                        session.last_object_emit_at = completed_at
                    session.last_narrated_at = completed_at
                    session.qwen_error = ""

                    latest = session.latest_api_result
                    if (
                        isinstance(latest, dict)
                        and int(latest.get("scene_revision") or -1)
                            == int(session.scene_revision)
                    ):
                        narration = dict(latest.get("narration") or {})
                        narration["object_descriptions"] = final_rows
                        latest["narration"] = narration
                        latest["qwen_streaming"] = False
                        latest["qwen_pending"] = False
                        latest["qwen_expected_object_count"] = len(expected_ids)
                        latest["qwen_received_object_count"] = len(final_rows)
                        latest["qwen_received_object_ids"] = [
                            int(row["object_id"])
                            for row in final_rows
                        ]
                        latest["object_stage_complete"] = True
                        latest["environment_stage_complete"] = True

            first_sec = (
                session.qwen_first_result_at - session.qwen_stream_started_at
                if session.qwen_first_result_at > 0
                else -1.0
            )
            print(
                "[qwen-stream] complete "
                f"objects={len(enriched.get('object_descriptions') or [])}"
                f"/{len(expected_ids)} first={first_sec:.3f}s "
                f"total={completed_at - now:.3f}s",
                flush=True,
            )
        except Exception as exc:
            with _SESSION_LOCK:
                if session.qwen_job_token == job_token:
                    session.qwen_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[qwen-stream][WARN] {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            with _SESSION_LOCK:
                if session.qwen_job_token == job_token:
                    session.qwen_pending = False
                    session.qwen_streaming = False
            _QWEN_WORK_LOCK.release()

    threading.Thread(
        target=worker,
        name=f"scene-narrator-qwen-stream-{job_token}",
        daemon=True,
    ).start()
    return True


def semantic_line_core(value: Any) -> str:
    """Normalize narration for duplicate detection across track-ID changes."""
    text_value = sanitize_detector_class_text(
        localize_narration_text(value)
    )
    text_value = re.sub(
        r"^(?:更正：剛才的「[^」]+」標籤不準確，實際看起來是|"
        r"更正[:：]|目前[，,]|從畫面來看[，,]|另外可以看到[，,]|"
        r"再補充一點[，,]|可以看到)",
        "",
        text_value,
    )
    text_value = re.sub(
        r"(?:畫面|視野)(?:左側|右側|中央|中間|上方|下方)",
        "",
        text_value,
    )
    text_value = re.sub(r"(?:左側|右側|中央|中間|上方|下方)的?", "", text_value)
    text_value = re.sub(r"[「」『』，。！？、：:；;\s\-_/]", "", text_value)
    return text_value.lower().strip()


def semantic_lines_similar(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True

    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= 8 and shorter in longer:
        return True

    def bigrams(value: str) -> set[str]:
        return {value[i:i + 2] for i in range(max(0, len(value) - 1))}

    a = bigrams(left)
    b = bigrams(right)
    if not a or not b:
        return False

    overlap = len(a & b) / max(1, min(len(a), len(b)))
    return overlap >= 0.78


def filter_balanced_narration_lines(
    session: SessionState,
    lines: list[dict],
    now_value: float,
) -> list[dict]:
    """Keep a useful five-line feed without repetitive filler."""
    priority = {
        "correction": 60,
        "overlap_alert": 58,
        "object": 55,
        "unboxed": 45,
        "relation": 40,
        "presence": 30,
        "summary": 25,
        "idle": 15,
    }

    with _SESSION_LOCK:
        session.recent_semantic_lines = [
            row for row in session.recent_semantic_lines
            if now_value - row[0] <= SEMANTIC_DEDUP_WINDOW_SEC
        ]
        recent = list(session.recent_semantic_lines)

    candidates = list(enumerate(lines))
    candidates.sort(
        key=lambda item: (
            priority.get(str(item[1].get("type") or ""), 20),
            item[0],
        ),
        reverse=True,
    )

    selected_indexes: set[int] = set()
    type_counts: dict[str, int] = {}
    newly_seen: list[tuple[float, str, str]] = []

    for original_index, line in candidates:
        line_type = str(line.get("type") or "summary")
        text_value = str(line.get("text") or "").strip()
        core = semantic_line_core(text_value)
        if not core:
            continue

        # Do not let a weaker blue presence line suppress a later green detail.
        comparable_recent = [
            row for row in recent
            if not (
                line_type in {"object", "correction"}
                and row[2] == "presence"
            )
        ]

        if any(
            semantic_lines_similar(core, old_core)
            for _, old_core, _ in comparable_recent
        ):
            continue
        if any(
            semantic_lines_similar(core, old_core)
            for _, old_core, _ in newly_seen
        ):
            continue

        type_limit = 1 if line_type == "idle" else MAX_LINES_PER_TYPE
        if type_counts.get(line_type, 0) >= type_limit:
            continue

        selected_indexes.add(original_index)
        type_counts[line_type] = type_counts.get(line_type, 0) + 1
        newly_seen.append((now_value, core, line_type))

        if len(selected_indexes) >= MAX_NARRATION_LINES:
            break

    accepted = [
        line
        for index, line in enumerate(lines)
        if index in selected_indexes
    ]

    with _SESSION_LOCK:
        session.recent_semantic_lines.extend(newly_seen)
        session.recent_semantic_lines = session.recent_semantic_lines[-40:]

    return accepted[-MAX_NARRATION_LINES:]


def presence_event_object_ids(
    session: SessionState,
    objects: list[dict],
    *,
    now_value: float,
) -> set[int]:
    """Return object IDs that deserve a new blue event line.

    The same class/count is not rephrased every polling cycle. A class becomes
    eligible again only after a long confirmed absence.
    """
    grouped: dict[str, list[dict]] = {}
    for obj in objects:
        grouped.setdefault(canonical(obj.get("label") or "object"), []).append(obj)

    with _SESSION_LOCK:
        # Expire old announced state only after a genuine long absence.
        for label in list(session.announced_presence_counts):
            last_seen = float(session.presence_last_seen_at.get(label) or 0.0)
            if label not in grouped and now_value - last_seen >= PRESENCE_REANNOUNCE_ABSENCE_SEC:
                session.announced_presence_counts.pop(label, None)
                session.presence_last_seen_at.pop(label, None)

        event_ids: set[int] = set()
        for label, rows in grouped.items():
            rows = sorted(rows, key=lambda x: int(x.get("id") or 0))
            session.presence_last_seen_at[label] = now_value
            previous_count = int(session.announced_presence_counts.get(label) or 0)
            current_count = len(rows)
            if current_count > previous_count:
                for row in rows[previous_count:current_count]:
                    event_ids.add(int(row["id"]))
            session.announced_presence_counts[label] = max(previous_count, current_count)

        return event_ids



def _normalize_external_detections(raw_items: Any, *, width: int, height: int) -> list[dict]:
    """Normalize colleague YOLO/D405 output while preserving every 3-D field."""
    if isinstance(raw_items, dict):
        for key in ("objects", "detections", "detected_items", "items"):
            if isinstance(raw_items.get(key), list):
                raw_items = raw_items[key]
                break
    if not isinstance(raw_items, list):
        raise ValueError("detections_json must contain a list of detections")

    detections: list[dict] = []
    for source_index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        box = extract_box(item)
        conf = extract_conf(item)
        if not box or conf < MIN_DETECTION_CONF:
            continue
        row = dict(item)
        row.update({
            "label": extract_label(item),
            "bbox": box,
            "confidence": conf,
            "frame_width": int(width),
            "frame_height": int(height),
            "source_detection_id": item.get(
                "source_detection_id", item.get("id", source_index)
            ),
        })
        detections.append(row)
    return detections


def _analyze_with_detections(
    *,
    pil: Image.Image,
    detections: list[dict],
    session_id: str,
    force_narration: bool,
    started: float,
    yolo_inference_ms: Any = None,
    task_active: bool | None = None,
    allow_task_vlm: bool = False,
    source_meta: dict | None = None,
) -> dict:
    session = get_session(session_id)
    if task_active is not None:
        with _SESSION_LOCK:
            was_active = bool(session.task_active)
            session.task_active = bool(task_active)
            if session.task_active and not was_active:
                session.task_started_at = time.time()
                # Invalidate queued low-priority jobs. Running HTTP calls cannot
                # be killed, but stale tokens prevent their results committing.
                session.qwen_job_token += 1
                session.qwen_pending = False
                reset_qwen_stream_state(session)
                session.general_idle_job_token += 1
                session.general_idle_pending = False
                session.general_idle_ready = ""
                session.unboxed_job_token += 1
                session.unboxed_pending = False
            elif not session.task_active and was_active:
                session.task_id = ""
                session.task_started_at = 0.0
                session.reobserve_requested = True
                session.last_scene_change_at = time.time()
                session.last_result = {}
                session.object_description_cache.clear()
                session.last_enriched_signature = ""
                session.last_general_signature = ""
                session.general_idle_ready = ""
                session.last_object_emit_at = 0.0
                reset_qwen_stream_state(session)
    with _SESSION_LOCK:
        task_mode_active = bool(session.task_active)

    objects = assign_stable_ids(session, detections)

    # UI labels must never expose raw detector English.  Keep the canonical
    # detector class internally, but provide a separate Traditional-Chinese
    # display label for the live canvas overlay.
    for obj in objects:
        obj["display_name_zh"] = label_zh(obj.get("label"))

    stable_objects = [
        obj
        for obj in objects
        if int(obj.get("track_hits") or 0) >= MIN_TRACK_HITS_FOR_NARRATION
    ]

    relations = (
        relation_candidates(stable_objects, pil.width, pil.height)
        if RELATIONS_ENABLED
        else []
    )
    now = time.time()
    # Narration/Qwen state uses held tracks so a single dropped YOLO frame does
    # not become a fake disappear/reappear event.
    state_objects = held_scene_objects(
        session,
        width=pil.width,
        height=pil.height,
        now_value=now,
    )
    signature = scene_signature(state_objects, pil.width, pil.height)
    current_snapshot = scene_snapshot(state_objects, pil.width, pil.height)

    with _SESSION_LOCK:
        if session.last_scene_change_at <= 0:
            session.last_scene_change_at = now

        raw_scene_changed, raw_change_reasons = scene_state_changed(
            session.scene_snapshot,
            current_snapshot,
            width=pil.width,
            height=pil.height,
        )
        if not session.scene_snapshot and current_snapshot:
            raw_scene_changed = True
            raw_change_reasons = ["initial_scene"]

        forced_event = bool(force_narration or session.reobserve_requested)
        scene_change_pending = False
        scene_changed = False
        scene_change_reasons: list[str] = []

        if forced_event:
            scene_changed = True
            scene_change_reasons = [
                "manual_refresh" if force_narration else "task_completed"
            ]
        elif raw_scene_changed:
            scene_change_pending = True
            if session.pending_scene_signature == signature:
                session.pending_scene_confirm_count += 1
            else:
                session.pending_scene_signature = signature
                session.pending_scene_snapshot = current_snapshot
                session.pending_scene_confirm_count = 1
            if session.pending_scene_confirm_count >= SCENE_CHANGE_CONFIRM_FRAMES:
                scene_changed = True
                scene_change_pending = False
                scene_change_reasons = list(raw_change_reasons)
        else:
            session.pending_scene_signature = ""
            session.pending_scene_snapshot = []
            session.pending_scene_confirm_count = 0
            # Keep the confirmed state fresh when the scene is unchanged.
            session.last_signature = signature
            session.scene_snapshot = current_snapshot

        if scene_changed:
            session.scene_revision += 1
            session.last_signature = signature
            session.scene_snapshot = current_snapshot
            session.pending_scene_signature = ""
            session.pending_scene_snapshot = []
            session.pending_scene_confirm_count = 0
            session.reobserve_requested = False
            session.last_scene_event_reason = ",".join(scene_change_reasons)
            session.last_result = {}
            session.unboxed_result = None
            # Pairwise overlap state is intentionally NOT reset by unrelated
            # whole-scene changes. A new toy, tissue, or small bbox motion must
            # not erase a confirmed warning for the same two detected objects.
            session.idle_cycle_index = 0
            session.last_idle_emit_at = 0.0
            session.last_output_at = now
            session.general_idle_ready = ""
            session.general_idle_job_token += 1
            session.general_idle_pending = False
            session.recent_general_idle.clear()
            current_ids = {int(row["id"]) for row in state_objects}
            session.object_description_cache = {
                object_id: row
                for object_id, row in session.object_description_cache.items()
                if object_id in current_ids
            }
            session.last_narrated_at = 0.0
            session.last_object_emit_at = 0.0
            session.last_scene_change_at = now
            session.last_qwen_attempt_at = 0.0
            session.last_enriched_signature = ""
            session.last_general_signature = ""

            # Invalidate any previous Qwen job. Old workers may finish, but their
            # tokens prevent stale text from committing to the new scene.
            session.qwen_job_token += 1
            session.qwen_pending = False
            reset_qwen_stream_state(session)

        scene_revision = session.scene_revision
        scene_confirm_count = int(session.pending_scene_confirm_count)

    blue_event_ids = presence_event_object_ids(
        session,
        state_objects,
        now_value=now,
    )
    state_label_counts: dict[str, int] = {}
    for obj in state_objects:
        label_key = canonical(obj.get("label") or "object")
        state_label_counts[label_key] = state_label_counts.get(label_key, 0) + 1
    presence_key = "|".join(
        f"{label}:{count}" for label, count in sorted(state_label_counts.items())
    )
    presence_changed = bool(blue_event_ids)
    fast_narration = build_fast_narration(
        state_objects,
        relations,
        presence_object_ids=None if force_narration else blue_event_ids,
    )

    if not (force_narration or blue_event_ids):
        fast_narration["presence_descriptions"] = []

    if not scene_changed and not force_narration:
        fast_narration["relationship_descriptions"] = []

    # Qwen is deliberately delayed until YOLO reports a stable scene. The
    # current response always returns the boxes first; text arrives in a later
    # polling response. Only create the annotated Qwen image when a job is due.
    qwen_started = False
    unboxed_started = False
    pairwise_started = False
    pairwise_objects = recent_pairwise_objects(
        session,
        width=pil.width,
        height=pil.height,
        now_value=now,
    ) if PAIRWISE_PICKUP_ENABLED else []
    raw_pairwise_candidates = overlapping_pair_candidates(
        pairwise_objects,
        pil.width,
        pil.height,
    ) if PAIRWISE_PICKUP_ENABLED else []
    pairwise_candidates = stabilize_pairwise_candidates(
        session,
        raw_pairwise_candidates,
        now_value=now,
    ) if PAIRWISE_PICKUP_ENABLED else []
    pairwise_fingerprint = pairwise_candidate_fingerprint(pairwise_candidates)

    with _SESSION_LOCK:
        now_pairwise = time.time()
        if pairwise_candidates:
            session.pairwise_last_candidate_seen_at = now_pairwise
            if (
                session.pairwise_fingerprint
                and session.pairwise_fingerprint != pairwise_fingerprint
            ):
                session.pairwise_result = {}
                session.pairwise_ready_line = None
                session.pairwise_error = ""
                session.pairwise_attempt_count = 0
                session.pairwise_negative_until = 0.0
                session.pairwise_latched = False
                session.pairwise_job_token += 1
                session.pairwise_pending = False
            session.pairwise_fingerprint = pairwise_fingerprint
        elif (
            session.pairwise_last_candidate_seen_at > 0
            and now_pairwise - session.pairwise_last_candidate_seen_at
                >= PAIRWISE_DISAPPEAR_CLEAR_SEC
        ):
            session.pairwise_result = {}
            session.pairwise_ready_line = None
            session.pairwise_signature = ""
            session.pairwise_fingerprint = ""
            session.pairwise_error = ""
            session.pairwise_attempt_count = 0
            session.pairwise_negative_until = 0.0
            session.pairwise_latched = False
            session.pairwise_last_candidate_seen_at = 0.0
            session.pairwise_candidate_evidence.clear()
            session.pairwise_job_token += 1
            session.pairwise_pending = False

        pairwise_complete = bool(
            pairwise_candidates
            and session.pairwise_fingerprint == pairwise_fingerprint
            and session.pairwise_latched
            and (session.pairwise_result.get("overlap_alerts") or [])
        )
        pairwise_retry_due = bool(
            time.time() >= session.pairwise_negative_until
            and time.time() - session.pairwise_checked_at
                >= PAIRWISE_PICKUP_RETRY_SEC
        )
        pairwise_attempts_available = bool(
            session.pairwise_attempt_count
                < max(1, PAIRWISE_MAX_REVIEW_ATTEMPTS)
            or time.time() >= session.pairwise_negative_until
        )

    pairwise_needs_review = bool(
        pairwise_candidates
        and not pairwise_complete
        and pairwise_retry_due
        and pairwise_attempts_available
    )

    # Empty semantic scenes do not need a VLM call. Wait until the held-track
    # window expires, then publish a deterministic purple sentence immediately.
    if (
        not objects
        and not state_objects
        and not task_mode_active
        and not scene_change_pending
    ):
        complete_empty_scene_without_vlm(
            session,
            signature=signature,
        )

    # Event-driven VLM order:
    #   1) one contact-sheet streaming request supplies green + purple
    #   2) pairwise / unboxed work may run only after that request completes
    with _SESSION_LOCK:
        object_stage_complete = bool(
            not state_objects
            or session.last_enriched_signature == signature
        )
        environment_stage_complete = bool(
            not GENERAL_IDLE_ENABLED
            or session.last_general_signature == signature
        )
        combined_scene_complete = bool(
            session.last_enriched_signature == signature
            and session.last_general_signature == signature
        )

    if (
        state_objects
        and not task_mode_active
        and not scene_change_pending
        and not combined_scene_complete
    ):
        qwen_started = start_qwen_enrichment(
            session,
            signature=signature,
            image=pil,
            objects=state_objects,
            relations=relations,
            force=force_narration,
        )

    # Refresh completion flags after scheduling. Assisted and live narration
    # both publish asynchronously, so the current response may still be pending.
    with _SESSION_LOCK:
        object_stage_complete = bool(
            not state_objects
            or session.last_enriched_signature == signature
        )
        environment_stage_complete = bool(
            not GENERAL_IDLE_ENABLED
            or session.last_general_signature == signature
        )

    if (
        pairwise_needs_review
        and object_stage_complete
        and environment_stage_complete
        and not qwen_started
        and not task_mode_active
        and not assisted_output_enabled()
    ):
        pairwise_started = start_pairwise_pickup_audit(
            session,
            signature=signature,
            fingerprint=pairwise_fingerprint,
            image=pil,
            candidates=pairwise_candidates,
        )

    if (
        UNBOXED_AUDIT_ENABLED
        and object_stage_complete
        and environment_stage_complete
        and not pairwise_started
        and not qwen_started
        and not task_mode_active
        and not assisted_output_enabled()
    ):
        unboxed_started = start_unboxed_audit(
            session,
            signature=signature,
            annotated=draw_numbered_frame(pil, objects),
            existing_objects=state_objects,
        )

    object_preview_descriptions = build_object_preview_descriptions(
        state_objects,
    )

    with _SESSION_LOCK:
        enriched_result = (
            dict(session.last_result or {})
            if session.last_signature == signature
            else {}
        )
        cached_objects = [
            dict(session.object_description_cache[object_id])
            for object_id in sorted(session.object_description_cache)
            if object_id in {int(o["id"]) for o in state_objects}
        ]
        if cached_objects:
            enriched_result = dict(enriched_result or {})
            enriched_result["object_descriptions"] = cached_objects[:VLM_COMBINED_MAX_OBJECTS]

        qwen_pending = bool(session.qwen_pending)
        qwen_error = str(session.qwen_error or "")
        qwen_streaming = bool(session.qwen_streaming)
        qwen_expected_object_ids = list(session.qwen_expected_object_ids)
        qwen_received_object_ids = list(session.qwen_received_object_ids)
        qwen_stream_started_at = float(session.qwen_stream_started_at or 0.0)
        qwen_first_result_at = float(session.qwen_first_result_at or 0.0)
        qwen_stream_elapsed_sec = (
            max(0.0, time.time() - qwen_stream_started_at)
            if qwen_stream_started_at > 0
            else 0.0
        )
        qwen_first_result_sec = (
            max(0.0, qwen_first_result_at - qwen_stream_started_at)
            if qwen_first_result_at > 0 and qwen_stream_started_at > 0
            else None
        )
        current_unboxed = (
            dict(session.unboxed_result)
            if isinstance(session.unboxed_result, dict)
            else None
        )
        current_pairwise = (
            dict(session.pairwise_result)
            if (
                pairwise_candidates
                and isinstance(session.pairwise_result, dict)
                and session.pairwise_fingerprint == pairwise_fingerprint
            )
            else {}
        )
        pairwise_ready_line = (
            dict(session.pairwise_ready_line)
            if (
                pairwise_candidates
                and isinstance(session.pairwise_ready_line, dict)
                and session.pairwise_fingerprint == pairwise_fingerprint
            )
            else None
        )
        session.pairwise_ready_line = None
        pairwise_pending = bool(session.pairwise_pending)
        pairwise_error = str(session.pairwise_error or "")

    # A previous audit result may become stale when YOLO detects the object on a
    # later frame. Re-filter every response so an already visible cyan box can
    # never continue to appear in the yellow missing-object reminder.
    if current_unboxed:
        current_unboxed = filter_already_boxed_unboxed_candidates(
            current_unboxed,
            state_objects,
            (pil.width, pil.height),
            stage="response",
        )
        current_unboxed = clean_unboxed_observation(current_unboxed)
        if not current_unboxed:
            with _SESSION_LOCK:
                session.unboxed_result = None

    narration = merge_narration(fast_narration, enriched_result)
    if current_unboxed:
        narration["unboxed_observation"] = current_unboxed

    idle_line = None
    general_idle_line = None
    general_idle_started = False
    now_for_idle = time.time()

    # Purple narration arbitration:
    # - At most one purple line per polling cycle.
    # - A completed whole-scene comment has priority.
    # - Object-idle and general-idle share a cooldown.
    with _SESSION_LOCK:
        if session.last_output_at <= 0:
            session.last_output_at = now_for_idle

        ready_comment = str(session.general_idle_ready or "").strip()

        # Purple narration summarizes the full camera image, so it remains useful
        # even when YOLO currently recognizes only one object.
        green_gap_remaining = max(
            0.0,
            GREEN_PURPLE_MIN_GAP_SEC
            - (now_for_idle - session.last_object_emit_at),
        ) if session.last_object_emit_at > 0 else 0.0
        general_ready_allowed = (
            not task_mode_active
            and ready_comment
            and green_gap_remaining <= 0.0
        )

        if general_ready_allowed:
            general_idle_line = {"type": "idle", "text": ready_comment}
            session.general_idle_ready = ""
            session.last_output_at = now_for_idle
            session.last_idle_emit_at = now_for_idle
            session.last_general_idle_emit_at = now_for_idle
            session.recent_general_idle.append(ready_comment)
            session.recent_general_idle = session.recent_general_idle[-10:]
        # Do not replay green object descriptions as purple idle lines.
        # Purple is reserved for independently generated whole-scene comments.

        # Whole-scene purple narration is based on scene stability, not on
        # whether green object descriptions are cached. Cached green text used to
        # make this condition permanently false, so the purple overview vanished.
        scene_stable_long_enough = (
            now_for_idle - session.last_scene_change_at >= GENERAL_IDLE_AFTER_SEC
        )
        can_start_general_idle = (
            GENERAL_IDLE_ENABLED
            and not task_mode_active
            and not scene_change_pending
            and scene_stable_long_enough
            # 空場景不會執行綠色物件 Qwen，但仍可進入紫色環境描述。
            and object_stage_complete
            and session.last_general_signature != signature
            and not session.general_idle_pending
            and not session.general_idle_ready
            and not scene_changed
            and not qwen_pending
            and idle_line is None
            and general_idle_line is None
        )
    if can_start_general_idle:
        general_idle_started = start_general_idle_comment(
            session,
            signature=signature,
            image=pil,
            objects=(stable_objects or state_objects),
        )

    lines = []
    if str(narration.get("scene_summary") or "").strip():
        lines.append({"type": "summary", "text": narration["scene_summary"]})
    # Exhibition narration is object-first: appearance/state descriptions are
    # more informative and change less often than geometric relations.
    for row in narration.get("presence_descriptions") or []:
        lines.append({
            "type": "presence",
            "text": row["text"],
            "object_id": row.get("object_id"),
        })

    for row in narration.get("object_descriptions") or []:
        lines.append({
            "type": "object",
            "text": row["text"],
            "object_id": row.get("object_id"),
        })

    if pairwise_ready_line:
        lines.append(pairwise_ready_line)

    unboxed = narration.get("unboxed_observation")
    if isinstance(unboxed, dict) and unboxed.get("text"):
        lines.append({
            "type": "unboxed",
            "text": unboxed["text"],
            "possible_label": unboxed.get("possible_label"),
        })

    for row in narration.get("relationship_descriptions") or []:
        lines.append({
            "type": "relation",
            "text": row["text"],
            "subject_id": row.get("subject_id"),
            "object_id": row.get("object_id"),
            "relation": row.get("relation"),
        })

    # Never emit two purple lines in one response.
    if general_idle_line:
        lines.append(general_idle_line)
    elif idle_line:
        lines.append(idle_line)
    if lines:
        with _SESSION_LOCK:
            session.last_output_at = time.time()

    for line in lines:
        line["text"] = sanitize_detector_class_text(line.get("text"))

    lines = filter_balanced_narration_lines(
        session,
        lines,
        time.time(),
    )

    # 紫色整體描述原本只存在一個 response，下一個 bridge frame 就會消失。
    # 同一 scene signature 完成後，持續把最後一筆紫色文字放進 state，
    # 直到場景改變時 recent_general_idle 被清除。
    with _SESSION_LOCK:
        persistent_general_text = (
            str(session.recent_general_idle[-1] or "").strip()
            if (
                session.last_general_signature == signature
                and session.recent_general_idle
            )
            else ""
        )

    if (
        persistent_general_text
        and not any(
            str(line.get("type") or "") in {
                "idle", "summary", "relation", "unboxed"
            }
            for line in lines
        )
    ):
        lines.append({
            "type": "idle",
            "text": persistent_general_text,
            "persistent": True,
        })

    if any(str(line.get("type") or "") == "object" for line in lines):
        with _SESSION_LOCK:
            if session.last_object_emit_at <= 0:
                session.last_object_emit_at = time.time()

    with _SESSION_LOCK:
        green_gap_remaining = max(
            0.0,
            GREEN_PURPLE_MIN_GAP_SEC
            - (time.time() - session.last_object_emit_at),
        ) if session.last_object_emit_at > 0 else 0.0
        
        placement_recommendations = (
        build_placement_recommendations(
            stable_objects,
        )
    )

        placement_summary = (
            build_placement_summary(
                placement_recommendations,
            )
        )


    result = {
        "ok": True,
        "session_id": session_id,
        "object_count": len(objects),
        "stable_object_count": len(stable_objects),
        "objects": objects,
        "stable_objects": stable_objects,
        "semantic_objects": state_objects,
        "object_preview_descriptions": object_preview_descriptions,
        "placement_recommendations": (
            placement_recommendations
        ),
        "placement_summary": (
            placement_summary
        ),

        "relations": relations,
        "pairwise_recent_track_count": len(pairwise_objects),
        "pairwise_recent_tracks": pairwise_objects,
        "pairwise_candidates": pairwise_candidates,
        "pairwise_raw_candidate_count": len(raw_pairwise_candidates),
        "overlap_alerts": current_pairwise.get("overlap_alerts", []),
        "pickup_order": current_pairwise.get("pickup_order", []),
        "pickup_dependencies": current_pairwise.get("pickup_order", []),
        "pairwise_reviewed_pairs": current_pairwise.get("reviewed_pairs", []),
        "narration": narration,
        "lines": lines,
        "narration_updated": bool(enriched_result),
        "configured_output_mode": NARRATOR_OUTPUT_MODE,
        "output_mode": runtime_output_mode(),
        "demo_output": assisted_output_enabled(),
        "demo_mode": "assisted" if assisted_output_enabled() else "",
        "description_source": (
            "demo_description_bank"
            if assisted_output_enabled()
            else "qwen3_vl"
        ),
        "scene_changed": scene_changed,
        "scene_revision": scene_revision,
        "scene_change_reasons": scene_change_reasons,
        "scene_change_pending": bool(scene_change_pending),
        "scene_confirm_count": scene_confirm_count,
        "scene_confirm_required": SCENE_CHANGE_CONFIRM_FRAMES,
        "scene_event_reason": str(session.last_scene_event_reason or ""),
        "object_stage_complete": bool(object_stage_complete),
        "environment_stage_complete": bool(environment_stage_complete),
        "event_trigger_only": True,
        "presence_changed": presence_changed,
        "presence_key": presence_key,
        "idle_line_emitted": bool(idle_line),
        "general_idle_line_emitted": bool(general_idle_line),
        "qwen_started": qwen_started,
        "pairwise_started": pairwise_started,
        "pairwise_pending": pairwise_pending,
        "pairwise_error": pairwise_error,
        "pairwise_latched": bool(session.pairwise_latched),
        "pairwise_attempt_count": int(session.pairwise_attempt_count),
        "unboxed_started": unboxed_started,
        "qwen_pending": qwen_pending,
        "qwen_error": qwen_error,
        "qwen_streaming": qwen_streaming,
        "qwen_expected_object_count": len(qwen_expected_object_ids),
        "qwen_received_object_count": len(qwen_received_object_ids),
        "qwen_expected_object_ids": qwen_expected_object_ids,
        "qwen_received_object_ids": qwen_received_object_ids,
        "qwen_stream_elapsed_sec": round(qwen_stream_elapsed_sec, 3),
        "qwen_first_result_sec": (
            round(qwen_first_result_sec, 3)
            if qwen_first_result_sec is not None
            else None
        ),
        "qwen_first_detail_target_sec": VLM_FIRST_DETAIL_TARGET_SEC,
        "general_idle_started": general_idle_started,
        "general_idle_pending": bool(session.general_idle_pending),
        "purple_allowed": bool(GENERAL_IDLE_ENABLED),
        "green_purple_min_gap_sec": GREEN_PURPLE_MIN_GAP_SEC,
        "green_purple_gap_remaining_sec": round(green_gap_remaining, 3),
        "semantic_state_object_count": len(state_objects),
        "annotated_image": "",
        "frame_width": pil.width,
        "frame_height": pil.height,
        "yolo_inference_ms": yolo_inference_ms,
        "source_meta": dict(source_meta or {}),
        "frame_id": (source_meta or {}).get("frame_id"),
        "vision_latest_time": (source_meta or {}).get("latest_time"),
        "coordinate_mode": (source_meta or {}).get("coordinate_mode"),
        "qwen_engine_busy": _QWEN_WORK_LOCK.locked(),
        "task_active": task_mode_active,
        "vlm_suppressed": bool(task_mode_active and not allow_task_vlm),
        "allow_task_vlm": bool(allow_task_vlm),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    with _SESSION_LOCK:
        session.latest_api_result = json.loads(json.dumps(result, ensure_ascii=False))
    return result

def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _frame_key(payload: dict[str, Any]) -> str:
    for key in ("frame_id", "latest_time", "captured_at"):
        value = payload.get(key)
        if value is not None:
            return f"{key}:{value}"
    objects = payload.get("objects") or payload.get("detections") or []
    return json.dumps(objects, ensure_ascii=False, sort_keys=True, default=str)


def _pil_from_snapshot(snapshot: dict[str, Any]) -> Image.Image | None:
    """Convert a vision_service snapshot to one RGB PIL image.

    Preferred input is color_frame from get_latest_scene_snapshot(). The
    compatibility fallback also accepts jpeg_bytes. No network or second camera
    runtime is used.
    """
    frame = snapshot.get("color_frame")
    if isinstance(frame, Image.Image):
        return frame.convert("RGB").copy()

    raw = snapshot.get("jpeg_bytes")
    if isinstance(raw, (bytes, bytearray)) and raw:
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")

    if frame is None:
        return None

    # vision_service stores OpenCV BGR ndarray frames. Reuse its encoder so
    # this module does not duplicate camera/OpenCV format handling.
    encoder = getattr(vision_service, "encode_jpeg", None)
    if not callable(encoder):
        raise RuntimeError(
            "vision_service snapshot contains color_frame but encode_jpeg() "
            "is unavailable"
        )
    raw = encoder(frame, quality=VLM_NARRATOR_JPEG_QUALITY)
    if not raw:
        raise RuntimeError("vision_service failed to encode latest color frame")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _read_vision_snapshot() -> dict[str, Any] | None:
    """Read the newest TaiROS vision state directly from vision_service.

    Preferred future contract:
        vision_service.get_latest_scene_snapshot(include_robot_xyz=False)

    Until that helper is added, this falls back to the two existing public
    functions. The fallback remains in-process, but the preferred helper is
    atomic and guarantees image/detection alignment.
    """
    atomic_getter = getattr(vision_service, "get_latest_scene_snapshot", None)
    if callable(atomic_getter):
        try:
            snapshot = atomic_getter(include_robot_xyz=False)
        except TypeError:
            snapshot = atomic_getter()
        if snapshot is None:
            return None
        if not isinstance(snapshot, dict):
            raise RuntimeError("vision_service scene snapshot is not a dict")
        if snapshot.get("status") not in {None, "success"}:
            raise RuntimeError(
                str(snapshot.get("message") or "vision snapshot failed")
            )
        return snapshot

    detections_result = vision_service.get_latest_detections(
        include_robot_xyz=False,
    )
    if not isinstance(detections_result, dict):
        raise RuntimeError("vision_service detections result is not a dict")
    if detections_result.get("status") not in {None, "success"}:
        raise RuntimeError(
            str(detections_result.get("message") or "vision detections failed")
        )

    jpeg_bytes, jpeg_info = vision_service.get_latest_color_jpeg(
        quality=VLM_NARRATOR_JPEG_QUALITY,
    )
    if not jpeg_bytes:
        message = (
            (jpeg_info or {}).get("message")
            if isinstance(jpeg_info, dict)
            else "no latest color frame"
        )
        raise RuntimeError(str(message or "no latest color frame"))

    return {
        "jpeg_bytes": jpeg_bytes,
        "objects": list(detections_result.get("objects") or []),
        "latest_time": detections_result.get("latest_time"),
        "coordinate_mode": detections_result.get("coordinate_mode"),
        "has_synced_tcp_pose": detections_result.get("has_synced_tcp_pose"),
        "coordinate_success_count": detections_result.get(
            "coordinate_success_count"
        ),
        "source": "tairos_vision_service_fallback",
    }


def process_snapshot(
    snapshot: dict[str, Any],
    *,
    session_id: str = VLM_SESSION_ID,
    force_narration: bool = False,
    allow_task_vlm: bool = False,
) -> dict[str, Any]:
    """Process one in-memory vision_service snapshot."""
    started = time.perf_counter()
    pil = _pil_from_snapshot(snapshot)
    if pil is None:
        raise RuntimeError("vision_service has no latest color frame")

    detections = _normalize_external_detections(
        snapshot,
        width=pil.width,
        height=pil.height,
    )
    source_meta = {
        key: snapshot.get(key)
        for key in (
            "frame_id",
            "latest_time",
            "captured_at",
            "coordinate_mode",
            "camera_mode",
            "has_synced_tcp_pose",
            "coordinate_success_count",
            "source",
        )
        if snapshot.get(key) is not None
    }
    source_meta["detector_source"] = "tairos_vision_service_in_process"

    return _analyze_with_detections(
        pil=pil,
        detections=detections,
        session_id=session_id,
        force_narration=force_narration,
        started=started,
        yolo_inference_ms=None,
        task_active=None,
        allow_task_vlm=allow_task_vlm,
        source_meta=source_meta,
    )


def process_latest_snapshot(
    *,
    session_id: str = VLM_SESSION_ID,
    force_narration: bool = False,
    allow_task_vlm: bool = False,
) -> dict[str, Any] | None:
    snapshot = _read_vision_snapshot()
    if snapshot is None:
        return None
    return process_snapshot(
        snapshot,
        session_id=session_id,
        force_narration=force_narration,
        allow_task_vlm=allow_task_vlm,
    )


def _narrator_runtime_worker() -> None:
    global _NARRATOR_LAST_FRAME_KEY
    global _NARRATOR_LAST_ERROR
    global _NARRATOR_LAST_RESULT_AT

    interval = 1.0 / VLM_NARRATOR_HZ
    while not _NARRATOR_RUNTIME_STOP.is_set():
        started = time.monotonic()
        try:
            snapshot = _read_vision_snapshot()
            if snapshot is not None:
                key = _frame_key(snapshot)
                if key != _NARRATOR_LAST_FRAME_KEY:
                    process_snapshot(snapshot, session_id=VLM_SESSION_ID)
                    _NARRATOR_LAST_FRAME_KEY = key
                    _NARRATOR_LAST_RESULT_AT = time.time()
                    if _NARRATOR_LAST_ERROR:
                        print(
                            "[vlm-narrator] vision connection restored",
                            flush=True,
                        )
                    _NARRATOR_LAST_ERROR = ""
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            if error_text != _NARRATOR_LAST_ERROR:
                print(f"[vlm-narrator][WARN] {error_text}", flush=True)
                _NARRATOR_LAST_ERROR = error_text

        elapsed = time.monotonic() - started
        _NARRATOR_RUNTIME_STOP.wait(max(0.05, interval - elapsed))


def _start_qwen_and_mode_workers() -> None:
    global _QWEN_KEEPALIVE_THREAD
    global _UI_MODE_WATCHDOG_THREAD

    print(
        "[narrator-mode] "
        f"configured={NARRATOR_OUTPUT_MODE} "
        f"effective={runtime_output_mode()} "
        f"heartbeat_ttl={UI_MODE_HEARTBEAT_TTL_SEC:.1f}s "
        f"description_bank={DEMO_DESCRIPTION_PATH}",
        flush=True,
    )

    if NARRATOR_OUTPUT_MODE in {"auto", "assisted"}:
        object_count = len(DEMO_DESCRIPTION_BANK.get("objects") or {})
        description_count = sum(
            len((item or {}).get("descriptions") or [])
            for item in (DEMO_DESCRIPTION_BANK.get("objects") or {}).values()
            if isinstance(item, dict)
        )
        print(
            "[assisted-demo] ready "
            f"objects={object_count} descriptions={description_count} "
            f"seed={DEMO_SEED} "
            f"output_gap={ASSISTED_OUTPUT_GAP_SEC:.2f}s "
            f"track_hold={ASSISTED_TRACK_HOLD_SEC:.2f}s "
            "coordinate_mapping="
            f"{DEMO_DESCRIPTION_BANK.get('coordinate_mapping') or 'identity'}",
            flush=True,
        )

    if NARRATOR_OUTPUT_MODE == "auto":
        _UI_MODE_WATCHDOG_STOP.clear()
        if not (
            _UI_MODE_WATCHDOG_THREAD
            and _UI_MODE_WATCHDOG_THREAD.is_alive()
        ):
            _UI_MODE_WATCHDOG_THREAD = threading.Thread(
                target=_ui_mode_watchdog_worker,
                name="vlm-ui-mode-watchdog",
                daemon=True,
            )
            _UI_MODE_WATCHDOG_THREAD.start()

    if NARRATOR_OUTPUT_MODE == "assisted" or not QWEN_KEEPALIVE_ENABLED:
        return

    _QWEN_KEEPALIVE_STOP.clear()
    if not (
        _QWEN_KEEPALIVE_THREAD
        and _QWEN_KEEPALIVE_THREAD.is_alive()
    ):
        threading.Thread(
            target=_pin_qwen_once_at_startup,
            name="vlm-qwen-startup-warmup",
            daemon=True,
        ).start()
        _QWEN_KEEPALIVE_THREAD = threading.Thread(
            target=_qwen_keepalive_worker,
            name="vlm-qwen-resident",
            daemon=True,
        )
        _QWEN_KEEPALIVE_THREAD.start()


def start_runtime() -> dict[str, Any]:
    """Start the in-process narrator consumer once inside the 5000 process."""
    global _NARRATOR_RUNTIME_THREAD

    with _NARRATOR_RUNTIME_LOCK:
        if (
            _NARRATOR_RUNTIME_THREAD
            and _NARRATOR_RUNTIME_THREAD.is_alive()
        ):
            return {
                "ok": True,
                "result": True,
                "runtime_status": "already_running",
                "session_id": VLM_SESSION_ID,
            }

        _start_qwen_and_mode_workers()
        _NARRATOR_RUNTIME_STOP.clear()
        _NARRATOR_RUNTIME_THREAD = threading.Thread(
            target=_narrator_runtime_worker,
            name="tairos-vlm-narrator",
            daemon=True,
        )
        _NARRATOR_RUNTIME_THREAD.start()

    print(
        "[vlm-narrator] started "
        f"session={VLM_SESSION_ID} hz={VLM_NARRATOR_HZ:g} "
        "vision_source=services.vision_service",
        flush=True,
    )
    return {
        "ok": True,
        "result": True,
        "runtime_status": "running",
        "session_id": VLM_SESSION_ID,
        "hz": VLM_NARRATOR_HZ,
    }


def stop_runtime() -> dict[str, Any]:
    """Stop narrator-owned background workers without stopping vision_service."""
    global _NARRATOR_RUNTIME_THREAD

    _NARRATOR_RUNTIME_STOP.set()
    _QWEN_KEEPALIVE_STOP.set()
    _UI_MODE_WATCHDOG_STOP.set()

    thread = _NARRATOR_RUNTIME_THREAD
    if thread and thread.is_alive():
        thread.join(timeout=2.0)
    _NARRATOR_RUNTIME_THREAD = None

    return {
        "ok": True,
        "result": True,
        "runtime_status": "stopped",
    }


def get_health() -> dict[str, Any]:
    runtime_running = bool(
        _NARRATOR_RUNTIME_THREAD
        and _NARRATOR_RUNTIME_THREAD.is_alive()
    )
    return {
        "ok": True,
        "vision_source": "services.vision_service",
        "vision_input_mode": "in_process",
        "narrator_runtime_running": runtime_running,
        "narrator_hz": VLM_NARRATOR_HZ,
        "session_id": VLM_SESSION_ID,
        "last_result_at": _NARRATOR_LAST_RESULT_AT,
        "last_error": _NARRATOR_LAST_ERROR,
        "qwen_api": QWEN_BASE_URL,
        "qwen_model": QWEN_MODEL,
        "qwen_resident": bool(
            QWEN_KEEPALIVE_ENABLED and not assisted_output_enabled()
        ),
        "qwen_keep_alive": QWEN_KEEP_ALIVE,
        "pairwise_pickup_enabled": PAIRWISE_PICKUP_ENABLED,
        "configured_output_mode": NARRATOR_OUTPUT_MODE,
        "output_mode": runtime_output_mode(),
        "demo_output": assisted_output_enabled(),
        "demo_description_bank": str(DEMO_DESCRIPTION_PATH),
    }


def request_reobserve(
    *,
    session_id: str = VLM_SESSION_ID,
    reason: str = "main_ui_request",
) -> dict[str, Any]:
    """Handle normal reobserve requests and page-mode heartbeats."""
    reason_text = str(reason or "main_ui_request").strip()

    heartbeat_match = re.fullmatch(
        r"ui_mode_heartbeat\|(live|assisted)\|"
        r"([A-Za-z0-9_-]{1,80})",
        reason_text,
    )
    if heartbeat_match:
        mode_state = register_ui_mode_heartbeat(
            requested_mode=heartbeat_match.group(1),
            client_id=heartbeat_match.group(2),
        )
        return {
            "ok": True,
            "session_id": session_id,
            "heartbeat": True,
            **mode_state,
        }

    release_match = re.fullmatch(
        r"ui_mode_release\|([A-Za-z0-9_-]{1,80})",
        reason_text,
    )
    if release_match:
        mode_state = release_ui_mode_client(release_match.group(1))
        return {
            "ok": True,
            "session_id": session_id,
            "heartbeat": False,
            "released": True,
            **mode_state,
        }

    session = get_session(session_id)
    with _SESSION_LOCK:
        _reset_session_narration_state(
            session,
            reason=reason_text,
            now_value=time.time(),
        )

    return {
        "ok": True,
        "session_id": session_id,
        "reobserve_requested": True,
        "green_cache_cleared": True,
        "purple_cache_cleared": True,
        "reason": reason_text,
        "output_mode": runtime_output_mode(),
    }


def set_task_mode(
    *,
    active: bool,
    session_id: str = VLM_SESSION_ID,
    task_id: str = "",
) -> dict[str, Any]:
    """Pause/resume low-priority VLM work while still consuming vision state."""
    session = get_session(session_id)
    with _SESSION_LOCK:
        was_active = bool(session.task_active)
        session.task_active = bool(active)
        session.task_id = str(task_id or "") if active else ""
        session.task_started_at = time.time() if active else 0.0
        if active and not was_active:
            session.qwen_job_token += 1
            session.qwen_pending = False
            reset_qwen_stream_state(session)
            session.general_idle_job_token += 1
            session.general_idle_pending = False
            session.general_idle_ready = ""
            session.unboxed_job_token += 1
            session.unboxed_pending = False
        elif not active and was_active:
            session.reobserve_requested = True
            session.last_scene_change_at = time.time()
            session.last_result = {}
            session.object_description_cache.clear()
            session.last_enriched_signature = ""
            session.last_general_signature = ""
            session.general_idle_ready = ""
            session.last_object_emit_at = 0.0
            reset_qwen_stream_state(session)

    return {
        "ok": True,
        "session_id": session_id,
        "task_active": bool(active),
        "task_id": session.task_id,
    }


def get_state(
    session_id: str = VLM_SESSION_ID,
) -> dict[str, Any]:
    """Return the latest Scene Intelligence contract and stream progress."""
    session = get_session(session_id)
    with _SESSION_LOCK:
        result = _json_copy(session.latest_api_result or {})

        expected_ids = list(session.qwen_expected_object_ids)
        received_ids = list(session.qwen_received_object_ids)
        if result and expected_ids:
            current_ids = {
                int(obj.get("id"))
                for obj in (
                    result.get("semantic_objects")
                    or result.get("stable_objects")
                    or []
                )
                if isinstance(obj, dict) and obj.get("id") is not None
            }
            rows = [
                dict(session.object_description_cache[object_id])
                for object_id in expected_ids
                if (
                    object_id in current_ids
                    and object_id in session.object_description_cache
                )
            ]
            if rows:
                narration = dict(result.get("narration") or {})
                narration["object_descriptions"] = rows
                result["narration"] = narration
                old_lines = [
                    line
                    for line in (result.get("lines") or [])
                    if str((line or {}).get("type") or "") != "object"
                ]
                result["lines"] = old_lines + [
                    {
                        "type": "object",
                        "text": row.get("text"),
                        "object_id": row.get("object_id"),
                        "partial": bool(session.qwen_pending),
                    }
                    for row in rows
                    if str(row.get("text") or "").strip()
                ]

        started_at = float(session.qwen_stream_started_at or 0.0)
        first_at = float(session.qwen_first_result_at or 0.0)
        result["qwen_streaming"] = bool(session.qwen_streaming)
        result["qwen_pending"] = bool(session.qwen_pending)
        result["qwen_expected_object_count"] = len(expected_ids)
        result["qwen_received_object_count"] = len(received_ids)
        result["qwen_expected_object_ids"] = expected_ids
        result["qwen_received_object_ids"] = received_ids
        result["qwen_stream_elapsed_sec"] = round(
            max(0.0, time.time() - started_at)
            if started_at > 0
            else 0.0,
            3,
        )
        result["qwen_first_result_sec"] = (
            round(max(0.0, first_at - started_at), 3)
            if first_at > 0 and started_at > 0
            else None
        )
        result["qwen_first_detail_target_sec"] = VLM_FIRST_DETAIL_TARGET_SEC

        dependencies = list(
            (session.pairwise_result or {}).get("pickup_order", [])
        )
        return {
            "ok": True,
            "session_id": session_id,
            "task_active": bool(session.task_active),
            "task_id": session.task_id,
            "scene_revision": int(session.scene_revision),
            "scene": result,
            "stable_objects": list(result.get("stable_objects") or []),
            "lines": list(result.get("lines") or []),
            "pickup_dependencies": dependencies,
            "pickup_order": dependencies,
            "overlap_alerts": list(
                (session.pairwise_result or {}).get("overlap_alerts", [])
            ),
            "pairwise_reviewed_pairs": list(
                (session.pairwise_result or {}).get("reviewed_pairs", [])
            ),
            "pairwise_pending": bool(session.pairwise_pending),
            "configured_output_mode": NARRATOR_OUTPUT_MODE,
            "output_mode": runtime_output_mode(),
            "demo_output": assisted_output_enabled(),
        }

def cabinet_state_image_data_url(
    image: Image.Image,
) -> str:
    """
    將上層櫃門狀態判斷影像縮小並壓縮成 Data URL。

    此函式只供櫃門開關判斷使用，
    不影響其他 VLM 場景描述的影像品質。
    """
    img = image.convert("RGB").copy()

    resampling = getattr(
        Image,
        "Resampling",
        Image,
    )

    img.thumbnail(
        (192, 192),
        resampling.BILINEAR,
    )

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="JPEG",
        quality=40,
        optimize=False,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def check_top_cabinet_door_open_state(
    fresh_frame_timeout=1.5,
    qwen_lock_timeout=3.0,
    confidence_threshold=0.60,
):
    """
    等待一張可用的新相機畫面，使用 Qwen 判斷上層櫃門狀態。

    回傳狀態：
        open：
            明確判斷櫃門已開啟。

        closed：
            明確判斷櫃門尚未開啟。

        uncertain：
            沒有畫面、模型忙碌、解析失敗，
            或判斷信心低於門檻。

    uncertain 的處理由呼叫端決定。
    目前 task_service 會停止操作，不執行開門。
    """
    snapshot = None
    baseline_time = None
    last_candidate_time = None
    last_snapshot_error = None
    baseline_has_frame = False

    def _get_snapshot_time(value):
        if not isinstance(value, dict):
            return None

        for field in (
            "latest_time",
            "captured_at",
            "frame_id",
        ):
            raw_value = value.get(field)

            if raw_value is None:
                continue

            try:
                return float(raw_value)
            except (TypeError, ValueError):
                continue

        return None

    def _snapshot_is_valid(value):
        return (
            isinstance(value, dict)
            and value.get("status") == "success"
            and value.get("result") is True
            and value.get("color_frame") is not None
        )

    try:
        # ==================================================
        # Step 1：取得目前畫面作為時間基準
        # ==================================================
        baseline = (
            vision_service
            .get_latest_scene_snapshot(
                include_robot_xyz=False,
            )
        )

        if _snapshot_is_valid(baseline):
            baseline_has_frame = True
            baseline_time = _get_snapshot_time(
                baseline
            )

        elif isinstance(baseline, dict):
            last_snapshot_error = (
                baseline.get("message")
                or baseline.get("error")
                or baseline.get("reason")
            )

        wait_started_at = time.monotonic()

        deadline = (
            wait_started_at
            + max(
                0.1,
                float(fresh_frame_timeout),
            )
        )

        # ==================================================
        # Step 2：等待移動完成後的新畫面
        # ==================================================
        while time.monotonic() < deadline:
            candidate = (
                vision_service
                .get_latest_scene_snapshot(
                    include_robot_xyz=False,
                )
            )

            if _snapshot_is_valid(candidate):
                candidate_time = _get_snapshot_time(
                    candidate
                )

                last_candidate_time = candidate_time

                # 呼叫前沒有有效畫面時，
                # 第一張有效畫面即可使用。
                if not baseline_has_frame:
                    snapshot = candidate
                    break

                # 兩張畫面都有時間資訊時，
                # 新畫面時間必須大於基準時間。
                if (
                    baseline_time is not None
                    and candidate_time is not None
                    and candidate_time > baseline_time
                ):
                    snapshot = candidate
                    break

                # Snapshot 沒有時間資訊時，
                # 至少等待 0.3 秒再接受。
                if (
                    (
                        baseline_time is None
                        or candidate_time is None
                    )
                    and (
                        time.monotonic()
                        - wait_started_at
                    ) >= 0.3
                ):
                    snapshot = candidate
                    break

            elif isinstance(candidate, dict):
                last_snapshot_error = (
                    candidate.get("message")
                    or candidate.get("error")
                    or candidate.get("reason")
                    or last_snapshot_error
                )

            time.sleep(0.05)

        if snapshot is None:
            reason = "沒有取得新的相機畫面"

            if last_snapshot_error:
                reason += (
                    f"：{last_snapshot_error}"
                )

            return {
                "status": "success",
                "result": True,
                "state": "uncertain",
                "is_open": False,
                "confidence": 0.0,
                "reason": reason,
                "baseline_time": baseline_time,
                "latest_time": last_candidate_time,
            }

        # ==================================================
        # Step 3：將相機畫面轉換為 PIL Image
        # ==================================================
        image = _pil_from_snapshot(snapshot)

        if image is None:
            return {
                "status": "success",
                "result": True,
                "state": "uncertain",
                "is_open": False,
                "confidence": 0.0,
                "reason": "相機畫面無法轉換為影像",
                "baseline_time": baseline_time,
                "latest_time": _get_snapshot_time(
                    snapshot
                ),
            }

        # ==================================================
        # Step 4：取得 Qwen 共用鎖
        # ==================================================
        lock_acquired = (
            _QWEN_WORK_LOCK.acquire(
                timeout=max(
                    0.1,
                    float(qwen_lock_timeout),
                ),
            )
        )

        if not lock_acquired:
            return {
                "status": "success",
                "result": True,
                "state": "uncertain",
                "is_open": False,
                "confidence": 0.0,
                "reason": "VLM 目前正在執行其他工作",
                "baseline_time": baseline_time,
                "latest_time": _get_snapshot_time(
                    snapshot
                ),
            }

        try:
            # ==================================================
            # Step 5：建立 Qwen 請求
            # ==================================================
            payload = {
                "model": QWEN_MODEL,
                "temperature": 0.0,
                "top_p": 0.5,
                "max_tokens": 32,
                "stream": False,
                "response_format": {
                    "type": "json_object",
                },
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "這是固定位置的上層醫療櫃門狀態分類。"
                            "closed 的外觀："
                            "畫面中可看到完整、平整的灰色矩形門板，"
                            "黑色直立把手位於門板正面，"
                            "門板覆蓋櫃子開口，"
                            "看不到白色櫃內空間或層板。"
                            "符合此情況必須判定 closed。"

                            "open 的外觀："
                            "白色櫃內空間或白色層板明顯可見，"
                            "灰色門板已移至右側、向外展開，"
                            "不再覆蓋櫃子開口。"
                            "符合此情況才可判定 open。"
                            "不要因為看到櫃體側面、門板邊框或把手，"
                            "就判定為 open。"
                            "無法清楚看到完整門板與櫃內空間時，"
                            "判定 uncertain。"
                            "只輸出單行 JSON，不要解釋。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "只輸出單行："
                                    '{"state":"open|closed|uncertain",'
                                    '"confidence":0.0}'
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        cabinet_state_image_data_url(
                                            image
                                        )
                                    ),
                                },
                            },
                        ],
                    },
                ],
            }

            request = urllib.request.Request(
                (
                    QWEN_BASE_URL
                    + "/chat/completions"
                ),
                data=json.dumps(
                    payload,
                    ensure_ascii=False,
                ).encode("utf-8"),
                method="POST",
                headers={
                    "Content-Type": (
                        "application/json"
                    ),
                    "Authorization": (
                        "Bearer local"
                    ),
                },
            )

            # ==================================================
            # Step 6：呼叫 Qwen
            # ==================================================
            with urllib.request.urlopen(
                request,
                timeout=20.0,
            ) as response:
                raw_response = (
                    response
                    .read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )

            response_data = json.loads(
                raw_response
            )

            choices = (
                response_data.get("choices")
                or []
            )

            if not choices:
                raise RuntimeError(
                    "Qwen 回傳中沒有 choices"
                )

            content = (
                choices[0].get("message")
                or {}
            ).get("content")

            # ==================================================
            # Step 7：解析 Qwen JSON
            # ==================================================
            try:
                parsed = _parse_qwen_json_content(
                    content
                )

                if not isinstance(parsed, dict):
                    raise TypeError(
                        "Qwen JSON 根節點不是 object"
                    )

            except Exception as parse_exc:
                # Qwen 若在 reason 中途被截斷，
                # 嘗試從不完整 JSON 中救回必要欄位。
                raw_content = str(content or "")

                state_match = re.search(
                    (
                        r'"state"\s*:\s*'
                        r'"(open|closed|uncertain)"'
                    ),
                    raw_content,
                    flags=re.IGNORECASE,
                )

                confidence_match = re.search(
                    (
                        r'"confidence"\s*:\s*'
                        r'([0-9]+(?:\.[0-9]+)?)'
                    ),
                    raw_content,
                    flags=re.IGNORECASE,
                )

                if (
                    state_match is not None
                    and confidence_match is not None
                ):
                    parsed = {
                        "state": (
                            state_match
                            .group(1)
                            .lower()
                        ),
                        "confidence": float(
                            confidence_match.group(1)
                        ),
                        "reason": (
                            "說明被截斷，已恢復判斷"
                        ),
                    }

                    print(
                        "[cabinet-vlm][WARN] "
                        "Qwen JSON truncated; "
                        "state and confidence recovered. "
                        f"raw={raw_content!r}",
                        flush=True,
                    )

                else:
                    raise RuntimeError(
                        "無法解析 Qwen 回傳："
                        f"{type(parse_exc).__name__}: "
                        f"{parse_exc}；"
                        f"raw={raw_content!r}"
                    ) from parse_exc

            # ==================================================
            # Step 8：正規化 Qwen 回傳
            # ==================================================
            state = str(
                parsed.get("state")
                or "uncertain"
            ).strip().lower()

            if state not in {
                "open",
                "closed",
                "uncertain",
            }:
                state = "uncertain"

            try:
                confidence = float(
                    parsed.get("confidence")
                    or 0.0
                )
            except (TypeError, ValueError):
                confidence = 0.0

            confidence = max(
                0.0,
                min(1.0, confidence),
            )

            reason = str(
                parsed.get("reason")
                or ""
            ).strip()

            threshold = float(
                confidence_threshold
            )

            # 模型即使輸出 open 或 closed，
            # 信心不足時仍統一改為 uncertain。
            if (
                state in {"open", "closed"}
                and confidence < threshold
            ):
                original_state = state
                state = "uncertain"

                if reason:
                    reason = (
                        f"{reason}；"
                        f"原判斷為 {original_state}，"
                        f"但信心低於 {threshold:.2f}"
                    )
                else:
                    reason = (
                        f"原判斷為 {original_state}，"
                        f"但信心低於 {threshold:.2f}"
                    )

            is_open = state == "open"

            result = {
                "status": "success",
                "result": True,
                "state": state,
                "is_open": is_open,
                "confidence": confidence,
                "reason": reason,
                "baseline_time": baseline_time,
                "latest_time": _get_snapshot_time(
                    snapshot
                ),
            }

            print(
                "[cabinet-vlm] "
                f"state={state} "
                f"is_open={is_open} "
                f"confidence={confidence:.3f} "
                f"reason={reason}",
                flush=True,
            )

            return result

        finally:
            _QWEN_WORK_LOCK.release()

    except Exception as exc:
        return {
            "status": "success",
            "result": True,
            "state": "uncertain",
            "is_open": False,
            "confidence": 0.0,
            "reason": (
                "櫃門狀態判斷失敗："
                f"{type(exc).__name__}: {exc}"
            ),
            "baseline_time": baseline_time,
            "latest_time": (
                _get_snapshot_time(snapshot)
                if isinstance(snapshot, dict)
                else last_candidate_time
            ),
        }
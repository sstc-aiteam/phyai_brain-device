"""Standalone local accounts and training jobs; never calls services."""
import secrets
import tarfile
import io
import sqlite3
from pathlib import Path
import hashlib
import hmac
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from utils.response import success, error
from recording_training_replay.training.dataset import read_dataset_info
MODULE = "recording_training_replay"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "lerobot_datasets")

# Local accounts and persistent job history; cloud execution remains simulated.


def generate_password_hash(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600000).hex()
    return f"pbkdf2_sha256$600000${salt}${digest}"


def check_password_hash(encoded, password):
    algorithm, iterations, salt, expected = encoded.split("$")
    if algorithm != "pbkdf2_sha256":
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations)).hex()
    return hmac.compare_digest(actual, expected)

_training_lock = threading.RLock()
_training_sessions = {}
_training_jobs = {}
TRAINING_DB_PATH = Path(PROJECT_ROOT) / "training_uploads" / "accounts.sqlite3"


def _training_db():
    TRAINING_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(TRAINING_DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE IF NOT EXISTS users (uid TEXT PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, uid TEXT, payload TEXT)")
    return db


def register_training_user(username=None, password=None, **kwargs):
    username = str(username or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,32}", username):
        raise ValueError("帳號需為 3–32 個英文字母、數字或底線")
    if not isinstance(password, str) or not 8 <= len(password) <= 128:
        raise ValueError("密碼需為 8–128 個字元")
    password_hash = generate_password_hash(password)
    db = _training_db()
    try:
        with db:
            db.execute("INSERT INTO users VALUES (?, ?, ?)", (secrets.token_hex(16), username, password_hash))
    except sqlite3.IntegrityError:
        raise ValueError("帳號已存在") from None
    finally:
        db.close()
    return success(MODULE, "register_training_user", data={"username": username, "cloud_sync": "not_connected"})


def login_training_user(username=None, password=None, **kwargs):
    db = _training_db()
    try:
        row = db.execute("SELECT * FROM users WHERE username = ?", (str(username or "").strip().lower(),)).fetchone()
    finally:
        db.close()
    if not isinstance(password, str) or len(password) > 128 or not row or not check_password_hash(row["password_hash"], password):
        raise PermissionError("帳號或密碼錯誤")
    token = secrets.token_urlsafe(32)
    user = {"uid": row["uid"], "username": row["username"], "expires_at": time.time() + 3600}
    with _training_lock:
        for expired in [key for key, value in _training_sessions.items() if value["expires_at"] <= time.time()]:
            _training_sessions.pop(expired, None)
        _training_sessions[token] = user
    return success(MODULE, "login_training_user", data={"token": token, "user": user, "cloud_sync": "not_connected"})


def _training_user(token):
    with _training_lock:
        user = _training_sessions.get(token)
        if not user or user["expires_at"] <= time.time():
            raise PermissionError("請先登入訓練功能（登入有效一小時）")
        return dict(user)


def training_logout(token):
    with _training_lock:
        _training_sessions.pop(token, None)
    return success(MODULE, "training_logout")


def _save_training_job(job):
    db = _training_db()
    try:
        with db:
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?)", (job["job_id"], job["uid"], json.dumps(job)))
    finally:
        db.close()


def get_training_jobs(token=None, query="", job_id=None):
    user = _training_user(token)
    db = _training_db()
    try:
        rows = db.execute("SELECT payload FROM jobs WHERE uid = ? ORDER BY rowid DESC", (user["uid"],)).fetchall()
    finally:
        db.close()
    jobs = []
    for row in rows:
        job = json.loads(row["payload"])
        with _training_lock:
            active = _training_jobs.get(user["uid"], {})
            if job.get("uploading") and active.get("job_id") != job["job_id"]:
                job.update(phase="failed", uploading=False, error="服務重新啟動，工作已中斷，請重新提交")
                _save_training_job(job)
        if job_id and job["job_id"] != job_id:
            continue
        if query.lower() not in (job["dataset_path"] + " " + job["model"] + " " + job["job_id"]).lower():
            continue
        jobs.append(job)
    if job_id and not jobs:
        raise LookupError("找不到訓練工作")
    return success(MODULE, "get_training_jobs", data={"jobs": jobs, "user": user, "cloud_sync": "not_connected"})


def get_dataset_upload_status(token=None):
    jobs = get_training_jobs(token)["data"]["jobs"]
    return success(MODULE, "get_dataset_upload_status", data=jobs[0] if jobs else {
        "phase": "idle", "uploading": False, "mode": "demo"})


def _package_training_demo(uid, dataset_path, model, job_id):
    archive = Path(PROJECT_ROOT) / "training_uploads" / uid / (job_id + ".tar.gz")
    partial = archive.with_suffix(".partial")
    try:
        archive.parent.mkdir(parents=True, exist_ok=True)
        manifest = json.dumps({"mode": "demo", "user": {"uid": uid}, "model": model,
                               "job_id": job_id, "dataset": dataset_path.name,
                               "created_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False).encode()
        with tarfile.open(partial, "w:gz") as bundle:
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest)
            bundle.addfile(info, io.BytesIO(manifest))
            for item in sorted(dataset_path.rglob("*")):
                if item.is_symlink():
                    raise ValueError("dataset 不可包含符號連結")
                if item.is_file():
                    bundle.add(item, arcname=str(Path("dataset") / item.relative_to(dataset_path)), recursive=False)
        partial.replace(archive)
        with _training_lock:
            _training_jobs[uid].update(phase="simulated_training", uploading=True, current_step=0, total_steps=100,
                archive_path=str(archive), archive_bytes=archive.stat().st_size,
                training_status="simulated", uploaded=False)
            _save_training_job(_training_jobs[uid])
        for step in range(10, 101, 10):
            time.sleep(1)
            with _training_lock:
                _training_jobs[uid].update(current_step=step,
                    phase="simulated_completed" if step == 100 else "simulated_training",
                    uploading=step != 100)
                _save_training_job(_training_jobs[uid])
    except Exception as exc:
        partial.unlink(missing_ok=True)
        with _training_lock:
            _training_jobs[uid].update(phase="failed", uploading=False, error=str(exc))
            _save_training_job(_training_jobs[uid])


def start_dataset_upload(dataset_path=None, model=None, token=None, dataset_root=None, recording=False):
    user = _training_user(token)
    try:
        if model not in {"smolvla", "groot_n15"}:
            raise ValueError("請選擇 SmolVLA 或 GR00T N1.5")
        root = Path(dataset_root or DEFAULT_DATASET_DIR).resolve()
        path = Path(dataset_path or "").resolve()
        if path == root or root not in path.parents or not (path / "meta/info.json").is_file():
            raise ValueError("請選擇 lerobot_datasets 內的完整 dataset")
        if (read_dataset_info(str(path)) or {}).get("codebase_version") != "v3.0":
            raise ValueError("訓練上傳僅支援 v3 資料集")
        if not any(path.rglob("*.parquet")):
            raise ValueError("dataset 沒有 parquet 資料")
        if recording:
            raise ValueError("請先停止錄製，再打包 dataset")
        uid = user["uid"]
        with _training_lock:
            if _training_jobs.get(uid, {}).get("uploading"):
                raise ValueError("已有打包工作進行中")
            job_id = secrets.token_hex(12)
            _training_jobs[uid] = {"job_id": job_id, "uid": uid, "created_at": datetime.now(timezone.utc).isoformat(), "mode": "demo", "phase": "packaging",
                "uploading": True, "uploaded": False, "dataset_path": str(path), "model": model}
            _save_training_job(_training_jobs[uid])
            threading.Thread(target=_package_training_demo, args=(uid, path, model, job_id), daemon=True).start()
            return success(MODULE, "start_dataset_upload", data=dict(_training_jobs[uid]))
    except Exception as exc:
        return error(MODULE, "start_dataset_upload", error=exc, error_type=type(exc).__name__)


def execute(command, **kwargs):
    """Command boundary for the API facade; jobs run entirely in this module."""
    commands = {
        "register": register_training_user, "login": login_training_user,
        "logout": training_logout, "jobs": get_training_jobs,
        "status": get_dataset_upload_status, "start": start_dataset_upload,
    }
    if command not in commands:
        raise ValueError(f"Unknown training command: {command}")
    return commands[command](**kwargs)

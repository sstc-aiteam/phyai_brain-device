"""Record original state samples and compatible high-quality video in episode folders."""

import csv
import json
import os
import queue
import re
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")
FRAME_FIELDS = ("frame_index", "timestamp", *(f"state_{axis}" for axis in POSE_AXES),
                "gripper_command", *(f"joint_{index}" for index in range(1, 7)))
TIMING_FIELDS = ("sample_host_monotonic", "capture_unix_timestamp", "rtde_controller_timestamp",
                 "rtde_host_monotonic", "rtde_age_seconds")


def camera_timing_fields(name):
    return (f"camera_{name}_sensor_timestamp", f"camera_{name}_host_monotonic",
            f"camera_{name}_pairing_skew_seconds", f"camera_{name}_frame_id",
            f"camera_{name}_host_read_monotonic")


class RawWriter:
    def __init__(self, dataset_path, metadata):
        self.ffmpeg = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
        if metadata.get("camera_names") and not self.ffmpeg:
            raise RuntimeError("MP4 錄製需要 FFmpeg（包含 libx264 編碼器）")
        root = Path(dataset_path)
        root.mkdir(parents=True, exist_ok=True)
        self.episode_index = 0
        while True:
            self.path = root / f"episode_{self.episode_index:06d}"
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                self.episode_index += 1
        self.count = 0
        self.videos = {}
        self.video_frame_counts = {}
        self.start_timestamp = None
        self.last_timestamp = None
        self.camera_names = list(metadata.get("camera_names", []))
        self.fields = (*FRAME_FIELDS, *TIMING_FIELDS,
                       *(field for name in self.camera_names for field in camera_timing_fields(name)))
        self.metadata = {
            **metadata, "format": "raw", "version": 5,
            "fps": 1.0 / float(metadata["interval"]),
            "position_unit": "meter", "rotation_unit": "radian",
            "rotation_type": "axis_angle", "pose_frame": "robot_base",
            "joint_unit": "radian", "gripper_unit": "normalized_0_1",
            "gripper_zero": "fully_open", "gripper_one": "fully_closed",
            "gripper_source": "commanded",
            "timestamp_source": "host_monotonic_relative_to_first_sample",
            "timestamp_unit": "second", "timestamp_order": "strictly_increasing",
            "episode_start_host_monotonic": None,
            "timestamp_formula": "sample_host_monotonic - episode_start_host_monotonic",
            "recorded_cameras": [], "decoded_color_order": "BGR",
            "video_codec": "h264", "video_encoder": "libx264",
            "video_input_pixel_format": "bgr24", "video_pixel_format": "yuv420p", "video_crf": 10,
            "video_color_space": "bt709", "video_color_range": "limited",
            "video_lossless": False,
            "timing_columns": {
                "sample_host_monotonic": {"clock": "host_monotonic", "unit": "second",
                    "event": "sample_pairing_complete_after_camera_and_rtde_cache_read"},
                "capture_unix_timestamp": {"clock": "host_wall_clock", "unit": "second"},
                "rtde_controller_timestamp": {"clock": "robot_controller", "unit": "second"},
                "rtde_host_monotonic": {"clock": "host_monotonic", "unit": "second",
                    "event": "receive_monitor_snapshot_updated_after_rtde_reads; not recorder_cache_read_time"},
                "rtde_age_seconds": {"definition": "sample_host_monotonic - rtde_host_monotonic",
                    "unit": "second", "sign": "nonnegative; larger means older host snapshot"},
            },
            "camera_clock_sources": metadata.get("camera_clock_sources", {}),
            "visual_training_eligible": bool(metadata.get("camera_names")),
            "joint_names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"],
            "camera_specs": metadata.get("cameras", {}), "cameras": {},
            "synchronization": {"method": "software_host_monotonic",
                                "pairing_rule": "abs(camera_host_monotonic - rtde_host_monotonic)",
                                "native_clock_mapping": "not_applied; converter must calibrate before subtracting", "hardware_synchronized": False,
                                "max_host_pairing_skew_seconds": None,
                                "timing_mismatch_policy": "preserve_raw_diagnostics_for_converter",
                                "sensor_clock_calibrated": False},
            "episode_index": self.episode_index,
            "created_at": datetime.now(timezone.utc).isoformat(), "complete": False,
            "status": "recording",
            "video_encoding": "libx264 high profile, yuv420p, crf=10; lossy",
        }
        for name in self.camera_names:
            clock = self.metadata["camera_clock_sources"].get(name, {})
            self.metadata["timing_columns"].update({
                f"camera_{name}_sensor_timestamp": {"clock": "driver_native_unmapped",
                    "unit": clock.get("sensor_timestamp_unit", "unknown")},
                f"camera_{name}_host_monotonic": {"clock": "host_monotonic", "unit": "second",
                    "event": "driver_frame_receive_complete; never recorder_cache_read_time"},
                f"camera_{name}_host_read_monotonic": {"clock": "host_monotonic", "unit": "second",
                    "event": "recorder_get_frame_return; not exposure_or_arrival_time"},
                f"camera_{name}_frame_id": {"definition": "original_driver_frame_id; empty_if_unavailable"},
                f"camera_{name}_pairing_skew_seconds": {"unit": "second",
                    "definition": "abs(camera_host_monotonic - rtde_host_monotonic)",
                    "sign": "nonnegative; magnitude_only; no_lead_lag_sign"},
            })
        self._write_metadata()
        self.samples = (self.path / "frames.csv").open("w", newline="", encoding="utf-8")
        self.csv = csv.DictWriter(self.samples, fieldnames=self.fields)
        self.csv.writeheader()
        self.samples.flush()

    def _write_metadata(self):
        temporary = self.path / "episode.json.tmp"
        temporary.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path / "episode.json")

    def _write_image(self, name, image):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("MP4 影像必須是 uint8 BGR 三通道")
        if name not in self.videos:
            if not self.ffmpeg:
                raise RuntimeError("MP4 錄製需要 FFmpeg")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("camera name 必須只包含英數字、底線或連字號")
            filename = (f"episode_{self.episode_index:06d}.mp4" if not self.videos
                        else f"episode_{self.episode_index:06d}_{name}.mp4")
            height, width = image.shape[:2]
            log = (self.path / f".{name}.ffmpeg.log").open("w+b")
            process = subprocess.Popen([
                self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
                "-framerate", str(1.0 / float(self.metadata["interval"])), "-i", "pipe:0",
                "-an", "-vf", "scale=in_range=full:out_range=limited:out_color_matrix=bt709",
                "-c:v", "libx264", "-profile:v", "high", "-crf", "10", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-color_range", "tv",
                "-movflags", "+faststart", str(self.path / filename),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            self.videos[name] = (process, log, image.shape, filename)
            self.metadata["cameras"][name] = filename
            self.video_frame_counts[name] = 0
        process, _, shape, filename = self.videos[name]
        if image.shape != shape:
            raise ValueError("錄製期間相機影像尺寸不可改變")
        process.stdin.write(np.ascontiguousarray(image).tobytes())
        self.video_frame_counts[name] += 1
        return filename

    def add_frame(self, sample):
        if self.metadata["status"] != "recording":
            raise RuntimeError("episode 已完成或失敗，無法新增影格")
        try:
            self._add_frame(sample)
        except Exception as exc:
            self.mark_failed(exc)
            raise

    def _add_frame(self, sample):
        for key in ("tcp_pose", "joints"):
            values = np.asarray(sample[key], dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(f"{key} 必須包含 6 個有限數值")
        if not np.isfinite(sample["gripper"]) or not 0 <= sample["gripper"] <= 1:
            raise ValueError("gripper 必須介於 0 和 1")
        if set(sample["images"]) != set(self.camera_names):
            raise ValueError("影像相機集合必須與 episode 的錄影設定一致")
        for name, image in sample["images"].items():
            self._write_image(name, image)
        timestamp = float(sample["timestamp"])
        if not np.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("取樣時間必須有限且嚴格遞增")
        if self.start_timestamp is None:
            self.start_timestamp = timestamp
            self.metadata["episode_start_host_monotonic"] = timestamp
            self._write_metadata()
        timing = sample.get("timing", {})
        row = {"frame_index": self.count, "timestamp": timestamp - self.start_timestamp,
               "gripper_command": sample["gripper"], "sample_host_monotonic": timestamp,
               "capture_unix_timestamp": timing.get("capture_unix_timestamp"),
               "rtde_controller_timestamp": timing.get("controller_timestamp"),
               "rtde_host_monotonic": timing.get("arm_updated_monotonic")}
        updated = timing.get("arm_updated_monotonic")
        row["rtde_age_seconds"] = timestamp - float(updated) if updated is not None else None
        row.update(zip((f"state_{axis}" for axis in POSE_AXES), sample["tcp_pose"]))
        row.update(zip((f"joint_{i}" for i in range(1, 7)), sample["joints"]))
        for name in self.camera_names:
            camera = timing.get("camera_timing", {}).get(name, {})
            received = camera.get("received_monotonic")
            row[f"camera_{name}_frame_id"] = camera.get("frame_id")
            row[f"camera_{name}_host_read_monotonic"] = camera.get("read_monotonic")
            row[f"camera_{name}_sensor_timestamp"] = camera.get("sensor_timestamp")
            row[f"camera_{name}_host_monotonic"] = received
            row[f"camera_{name}_pairing_skew_seconds"] = (
                abs(float(updated) - float(received)) if updated is not None and received is not None else None)
        self.csv.writerow(row)
        self.samples.flush()
        self.last_timestamp = timestamp
        self.count += 1

    def _close_videos(self):
        failures = []
        for name, (process, log, _, _) in list(self.videos.items()):
            try:
                process.stdin.close()
                process.wait(timeout=30)
                if process.returncode:
                    log.seek(0)
                    failures.append(log.read().decode("utf-8", errors="replace"))
            except Exception as exc:
                process.kill()
                process.wait()
                failures.append(str(exc))
            finally:
                log.close()
                if process.returncode == 0:
                    (self.path / f".{name}.ffmpeg.log").unlink(missing_ok=True)
                del self.videos[name]
        if failures:
            raise RuntimeError("MP4 寫入失敗：" + "; ".join(failures))

    def _verify_counts(self):
        with (self.path / "frames.csv").open(newline="", encoding="utf-8") as source:
            csv_count = sum(1 for _ in csv.DictReader(source))
        if csv_count != self.count:
            raise RuntimeError("CSV 筆數與錄製計數不一致")
        verified = {}
        for name, filename in self.metadata["cameras"].items():
            result = subprocess.run([self.ffmpeg, "-v", "error", "-nostats", "-i", str(self.path / filename),
                "-map", "0:v:0", "-progress", "pipe:1", "-f", "null", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            counts = [int(line.split("=", 1)[1]) for line in result.stdout.splitlines() if line.startswith("frame=")]
            if result.returncode or result.stderr.strip() or not counts or counts[-1] != csv_count:
                raise RuntimeError(f"影片 {name} 解碼失敗或影格數與 CSV 不一致：{result.stderr}")
            verified[name] = counts[-1]
        self.metadata["verified_video_frame_counts"] = verified
        self.metadata["recorded_cameras"] = list(verified)

    def mark_failed(self, exc):
        self.metadata.update(status="failed", complete=False, error=str(exc),
                             sample_count=self.count)
        self._write_metadata()

    def save_episode(self):
        if self.metadata["status"] == "failed":
            raise RuntimeError("failed episode 無法標記 completed")
        if self.metadata["status"] == "completed":
            return
        try:
            if any(count != self.count for count in self.video_frame_counts.values()):
                raise RuntimeError("影片影格數與 CSV 筆數不一致")
            self.samples.flush()
            self._close_videos()
            self.samples.close()
            self._verify_counts()
            self.metadata.update(status="completed", sample_count=self.count,
                                 visual_training_eligible=bool(self.metadata["cameras"]),
                                 complete=True)
            self._write_metadata()
        except Exception as exc:
            self.mark_failed(exc)
            raise

    def finalize(self):
        try:
            self._close_videos()
            if self.metadata["status"] == "recording":
                self.mark_failed("recording closed without successful save")
        except Exception as exc:
            self.mark_failed(exc)
            raise
        finally:
            self.samples.close()


class RawRecorder:
    """Own recording state; receive commands and low-level device objects only."""

    def __init__(self):
        import threading
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.writer = None
        self.details = {"arm_name": None, "gripper_name": None, "camera_names": [],
                        "record_video": False, "output_path": None, "episode_index": None,
                        "selected_format": "raw", "dataset_format": "raw",
                        "training_target": None, "freedrive": False}
        self.error = None
        self.gripper_command = 0.0
        self.started_monotonic = None
        self.arm = None
        self.cameras = {}

    def execute(self, command, **kwargs):
        commands = {"start": self.start, "stop": self.stop, "status": self.status,
                    "set_gripper_command": self.set_gripper_command}
        if command not in commands:
            raise ValueError(f"Unknown recording command: {command}")
        return commands[command](**kwargs)

    @property
    def active(self):
        return bool(self.thread and self.thread.is_alive())

    def start(self, path, metadata, arm, cameras, gripper_command=0.0):
        import threading
        import time
        with self.lock:
            if self.thread is not None:
                raise RuntimeError("previous recording must be stopped before starting again")
            self.writer = RawWriter(path, metadata)
            self.arm, self.cameras = arm, dict(cameras)
            self.gripper_command = float(gripper_command)
            self.error = None
            self.started_monotonic = time.monotonic()
            self.details = {"output_path": str(path), "absolute_path": str(path),
                "arm_name": metadata.get("arm_name"), "gripper_name": metadata.get("gripper_name"),
                "camera_names": list(cameras), "record_video": bool(cameras),
                "dataset_format": "raw", "selected_format": "raw", "training_target": None,
                "episode_index": self.writer.episode_index, "freedrive": bool(metadata.get("freedrive"))}
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            try:
                self.thread.start()
            except Exception as exc:
                self.error = str(exc)
                self.thread = None
                self.writer.mark_failed(exc)
                self.writer.finalize()
                raise
            return {**self.details, "recording": True, "startup_seconds": 0.0}

    def _run(self):
        import threading
        import time
        pending = queue.Queue(maxsize=32)
        capture_done = threading.Event()

        def write_samples():
            try:
                while not capture_done.is_set() or not pending.empty():
                    try:
                        sample = pending.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    self.writer.add_frame(sample)
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                self.stop_event.set()

        writer_thread = threading.Thread(target=write_samples, daemon=True)
        deadline = time.monotonic()
        try:
            writer_thread.start()
            while not self.stop_event.is_set():
                sample = self.capture_sample()
                if self.stop_event.is_set():
                    break
                try:
                    pending.put_nowait(sample)
                except queue.Full as exc:
                    raise RuntimeError("raw data 寫入跟不上取樣，佇列已滿；未丟棄影格繼續錄製") from exc
                interval = float(self.writer.metadata["interval"])
                deadline += interval
                now = time.monotonic()
                # Skip expired scheduling slots without producing a burst of
                # artificial catch-up samples; preserve actual capture times.
                if deadline < now:
                    deadline += (int((now - deadline) / interval) + 1) * interval
                self.stop_event.wait(max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            with self.lock:
                if self.error is None:
                    self.error = str(exc)
            self.stop_event.set()
        finally:
            capture_done.set()
            if writer_thread.ident is not None:
                writer_thread.join()
            if self.error:
                try:
                    self.writer.mark_failed(self.error)
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("failed to persist recording failure")

    def stop(self):
        import time
        started = time.monotonic()
        with self.lock:
            thread = self.thread
        if thread is None:
            raise RuntimeError("recording is not active")
        self.stop_event.set()
        thread.join(timeout=10.0)
        if thread.is_alive():
            raise TimeoutError("recording thread 無法停止")
        with self.lock:
            try:
                if self.error:
                    raise RuntimeError(self.error)
                if self.writer.count == 0:
                    raise RuntimeError("沒有可儲存的 recording sample")
                self.writer.save_episode()
                return {**self.details, "recording": False, "sample_count": self.writer.count,
                        "stop_timings": {"total_seconds": time.monotonic() - started}}
            except Exception as exc:
                self.error = str(exc)
                self.writer.mark_failed(exc)
                raise
            finally:
                try:
                    self.writer.finalize()
                finally:
                    self.thread = None

    def status(self):
        import time
        with self.lock:
            return {**self.details, "recording": self.active,
                "session_active": self.thread is not None,
                "elapsed_seconds": time.monotonic() - self.started_monotonic if self.active else 0.0,
                "sample_count": self.writer.count if self.writer else 0, "error": self.error}

    def set_gripper_command(self, position):
        position = float(position)
        if not np.isfinite(position) or not 0 <= position <= 1:
            raise ValueError("gripper command 必須介於 0 和 1")
        with self.lock:
            self.gripper_command = position
        return {"gripper_command": position}
    def capture_sample(self):
        images, camera_timing = {}, {}
        for camera_name, camera in self.cameras.items():
            frame = camera.get_frame()
            read_completed = time.monotonic()
            received = frame.get("host_received_monotonic") if isinstance(frame, dict) else None
            image = frame.get("color_image") if isinstance(frame, dict) else None
            if image is None or image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise RuntimeError(f"camera '{camera_name}' 必須提供 640×480 uint8 color frame")
            images[camera_name] = image.copy()
            camera_timing[camera_name] = {"received_monotonic": received,
                                          "read_monotonic": read_completed, "frame_id": frame.get("frame_id"),
                                          "sensor_timestamp": frame.get("timestamp"),
                                          "sensor_timestamp_domain": "driver_native_unmapped"}
        # Read cached RTDE state after acquiring the camera frame, without touching
        # the control channel or querying the gripper over URScript.
        status = self.arm.get_recording_state()
        pose, joints = status.get("pose"), status.get("joints")
        if pose is None or joints is None:
            raise RuntimeError("手臂 pose 或 joints 無法取得")
        now = time.monotonic()
        updated = status.get("updated_monotonic")
        if updated is None or status.get("data_stale") or status.get("data_valid") is False:
            raise RuntimeError("錄製需要具有有效時間戳的最新 RTDE 狀態")
        if not np.isfinite(float(updated)) or float(updated) > now:
            raise RuntimeError("RTDE host 時間戳無效或位於未來")
        skews = {name: abs(float(updated) - timing["received_monotonic"])
                 for name, timing in camera_timing.items() if timing["received_monotonic"] is not None}
        # Preserve receive age and pairing skew as raw diagnostics. A timing
        # mismatch is for the converter to align, not a fatal recording error.
        gripper = self.gripper_command if self.details.get("gripper_name") else 0.0
        return {"timestamp": now, "tcp_pose": list(pose), "joints": list(joints),
                "gripper": max(0.0, min(1.0, gripper)), "images": images,
                "timing": {"capture_unix_timestamp": time.time(), "arm_updated_monotonic": updated,
                           "controller_timestamp": status.get("controller_timestamp"),
                           "camera_timing": camera_timing, "host_pairing_skew_seconds": skews}}



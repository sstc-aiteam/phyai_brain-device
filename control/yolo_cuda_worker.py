"""Persistent Jetson CUDA YOLO subprocess used by the Python 3.12 API."""

import argparse
import os
import pickle
import struct
import sys
import traceback


def _read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send(stream, payload):
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!Q", len(data)))
    stream.write(data)
    stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics-cuda")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA 無法使用（PyTorch CUDA build={torch.version.cuda}）"
            )
        from control.yolo_detector import YoloDetector
        detector = YoloDetector(
            model_path=args.model_path,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
        )
        _send(protocol_out, {
            "ok": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        })
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _send(protocol_out, {"ok": False, "error": str(exc)})
        return 1

    while True:
        header = _read_exact(sys.stdin.buffer, 8)
        if header is None:
            return 0
        size = struct.unpack("!Q", header)[0]
        body = _read_exact(sys.stdin.buffer, size)
        if body is None:
            return 0
        try:
            request = pickle.loads(body)
            annotated, detections = detector.predict(
                request["image"], draw=request.get("draw", True)
            )
            _send(protocol_out, {
                "ok": True,
                "annotated_frame": annotated,
                "detections": detections,
            })
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _send(protocol_out, {"ok": False, "error": str(exc)})


if __name__ == "__main__":
    raise SystemExit(main())
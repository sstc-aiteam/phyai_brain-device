from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


def _encode_numpy(value: Any) -> Any:
    # Wire-compatible with msgpack-numpy, which GR00T's MsgSerializer uses.
    if isinstance(value, np.ndarray):
        return {
            b"nd": True,
            b"type": value.dtype.str,
            b"kind": value.dtype.kind,
            b"shape": value.shape,
            b"data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        return {
            b"nd": False,
            b"type": value.dtype.str,
            b"data": value.tobytes(),
        }
    raise TypeError(f"cannot msgpack value of type {type(value).__name__}")


def _decode_numpy(value: dict[Any, Any]) -> Any:
    nd = value.get(b"nd", value.get("nd"))
    if nd is None:
        return value
    dtype = np.dtype(value.get(b"type", value.get("type")))
    data = value.get(b"data", value.get("data"))
    array = np.frombuffer(data, dtype=dtype)
    if nd:
        shape = value.get(b"shape", value.get("shape"))
        return array.reshape(shape)
    return array[0]


def pack_message(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode_numpy, use_bin_type=True)


def unpack_message(value: bytes) -> Any:
    return msgpack.unpackb(value, object_hook=_decode_numpy, raw=False)

"""Shared adapter operations for LeRobot v2 and v3.

The complete recording, replay, and training service remains in
``services.recording_training_replay_service``.  This module only centralizes
the glue shared by the two LeRobot format adapters so that v2 and v3 do not
duplicate writer finalization, recording dispatch, and recording-status code.
"""

from services import recording_training_replay_service as _service


# 完成資料集 writer 的待處理寫入。
def finalize_writer(writer, dataset_path=None):
    """Finalize a v2 or v3 dataset writer."""
    del dataset_path
    writer.finalize()


# 使用指定的資料集格式啟動錄製。
def start_format_recording(dataset_format, *args, **kwargs):
    """Start the service recorder with an explicit dataset format."""
    kwargs["dataset_format"] = dataset_format
    return _service.start_recording(*args, **kwargs)


# 停止目前進行中的錄製工作。
def stop_recording(*args, **kwargs):
    """Stop the single recording session owned by the service."""
    return _service.stop_recording(*args, **kwargs)


# 取得指定格式的錄製狀態。
def get_recording_status(dataset_format=None):
    """Read service recording status for an optional dataset format."""
    return _service.get_recording_status(dataset_format)

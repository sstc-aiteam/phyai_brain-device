"""Public service for LeRobot v3 recording, replay, and training."""
import threading
from functools import wraps

from utils.response import error, success
from recording_training_replay import startup
from recording_training_replay.recording.lerobot import executor as lerobot
from recording_training_replay.recording.task_registry import TaskRegistry
from recording_training_replay.replay import lerobot as lerobot_replay
from recording_training_replay.replay import lerobot_dataset
from recording_training_replay.storage import LEROBOT_DATASET_DIR, TASK_REGISTRY_PATH
from recording_training_replay.training import executor as training

MODULE = 'recording_training_replay'
_command_lock = threading.RLock()
_tasks = TaskRegistry()


def _command(operation):
    @wraps(operation)
    def execute(*args, **kwargs):
        with _command_lock:
            try:
                return operation(*args, **kwargs)
            except Exception as exc:
                return error(MODULE, operation.__name__, error=exc, error_type=type(exc).__name__)
    return execute


def _succeeded(result):
    return isinstance(result, dict) and result.get('status') == 'success' and result.get('result') is not False


def _require_lerobot_v3(dataset_format):
    if dataset_format not in (None, 'lerobot_v3'):
        raise ValueError('只支援 lerobot_v3；Raw Recording/Replay 已移除')


def _backend_data(result, operation):
    if not _succeeded(result):
        raise RuntimeError(f'{operation} failed: {result}')
    data = result.get('data')
    if not isinstance(data, dict):
        raise RuntimeError(f'{operation} returned invalid data')
    return data


def _recording_active():
    return bool(_backend_data(lerobot.get_recording_status(), 'get_recording_status').get('session_active'))


def _playback_active():
    return bool(_backend_data(lerobot_replay.get_robot_playback_status(), 'get_robot_playback_status').get('playing'))


def _require_idle():
    if _recording_active() or _playback_active():
        raise RuntimeError('錄製或 Replay 進行中，請先停止目前作業')


@_command
def start_recording(arm_name, gripper_name=None, output_path=None, interval=0.1, freedrive=True, record_video=True, task="robot demonstration", initial_gripper_position=0, dataset_mode="multi_task", dataset_format="lerobot_v3", dataset_name=None):
    _require_lerobot_v3(dataset_format)
    _require_idle()

    startup.update(starting=True, camera_ready=False, arm_name=arm_name, dataset_format="lerobot_v3")

    try:
        return lerobot.start_recording(arm_name=arm_name, gripper_name=gripper_name, output_path=output_path, interval=interval, freedrive=freedrive, record_video=record_video, task=task, initial_gripper_position=initial_gripper_position, dataset_mode=dataset_mode, dataset_format="lerobot_v3", dataset_name=dataset_name)
    finally:
        startup.update(starting=False)


@_command
def stop_recording(arm_name=None, gripper_name=None, dataset_format=None):
    _require_lerobot_v3(dataset_format)
    return lerobot.stop_recording(arm_name=arm_name, gripper_name=gripper_name)


@_command
def get_recording_status(dataset_format=None):
    _require_lerobot_v3(dataset_format)
    result = lerobot.get_recording_status()
    if _succeeded(result):
        result['data']['dataset_format'] = 'lerobot_v3'
    return result


@_command
def register_task(task):
    entry = _tasks.register(TASK_REGISTRY_PATH, task)
    return success(MODULE, 'register_task', data={'task': entry, 'registry_path': TASK_REGISTRY_PATH})


@_command
def get_task_registry():
    registry = _tasks.snapshot(TASK_REGISTRY_PATH)
    return success(MODULE, 'get_task_registry', data={'version': registry.get('version', 1), 'dataset_root': LEROBOT_DATASET_DIR, 'dataset_roots': {'lerobot_v3': LEROBOT_DATASET_DIR}, 'registry_path': TASK_REGISTRY_PATH, 'tasks': registry['tasks']})


@_command
def open_recording_gripper(gripper_name=None, speed=None, force=None, wait=False, timeout=None):
    return lerobot.open_recording_gripper(gripper_name=gripper_name, speed=speed, force=force, wait=wait, timeout=timeout)


@_command
def close_recording_gripper(gripper_name=None, speed=None, force=None, wait=False, timeout=None):
    return lerobot.close_recording_gripper(gripper_name=gripper_name, speed=speed, force=force, wait=wait, timeout=timeout)


@_command
def move_recording_gripper(position, speed=None, force=None, wait=False, timeout=None):
    return lerobot.move_recording_gripper(position=position, speed=speed, force=force, wait=wait, timeout=timeout)


@_command
def get_replay_catalog():
    return lerobot_dataset.get_replay_catalog()


@_command
def start_robot_playback(input_path, arm_name=None, gripper_name=None, episode_index=0, speed=None, acceleration=None, lookahead_time=0.1, gain=300, move_to_start=True, move_to_start_speed=None, move_to_start_acceleration=None, dataset_format='lerobot_v3'):
    _require_lerobot_v3(dataset_format)
    _require_idle()
    return lerobot_replay.start_robot_playback(input_path=input_path, arm_name=arm_name, gripper_name=gripper_name, episode_index=episode_index, speed=speed, acceleration=acceleration, lookahead_time=lookahead_time, gain=gain, move_to_start=move_to_start, move_to_start_speed=move_to_start_speed, move_to_start_acceleration=move_to_start_acceleration, dataset_format='lerobot_v3')


@_command
def stop_robot_playback(arm_name=None, gripper_name=None):
    return lerobot_replay.stop_robot_playback(arm_name=arm_name, gripper_name=gripper_name)


@_command
def get_robot_playback_status():
    return lerobot_replay.get_robot_playback_status()


def register_training_user(username=None, password=None, **kwargs):
    return training.execute('register', username=username, password=password, **kwargs)


def login_training_user(username=None, password=None, **kwargs):
    return training.execute('login', username=username, password=password, **kwargs)


def training_logout(token):
    return training.execute('logout', token=token)


def get_training_jobs(token=None, query='', job_id=None):
    return training.execute('jobs', token=token, query=query, job_id=job_id)


def get_dataset_upload_status(token=None):
    return training.execute('status', token=token)


@_command
def start_dataset_upload(dataset_path=None, model=None, token=None):
    return training.execute('start', dataset_path=dataset_path, model=model, token=token, dataset_root=LEROBOT_DATASET_DIR, recording=_recording_active())


def get_startup_status():
    """Progress stays readable while a recording command holds the device lock."""
    from recording_training_replay.recording.lerobot.dataset_writer import warmup_status
    return {'status': 'success', 'data': {**startup.snapshot(), 'writer': warmup_status()}}

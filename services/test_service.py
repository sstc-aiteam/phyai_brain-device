import threading

from services import record_service


LEFT_DEMO = "demo_left.json"
RIGHT_DEMO = "demo_right.json"


def _run_playback(
    input_path,
    arm_name,
    gripper_name,
):
    record_service._playback_worker(
        input_path=input_path,
        arm_name=arm_name,
        gripper_name=gripper_name,

        speed=None,
        acceleration=None,

        lookahead_time=0.1,
        gain=300,

        move_to_start=True,
        move_to_start_speed=None,
        move_to_start_acceleration=None,
    )


def run_dual_demo():
    """
    同時播放：
    left  -> demo_left.json
    right -> demo_right.json
    """

    # 清除共用 stop event，避免上一次測試留下停止狀態。
    record_service._playback_stop_event.clear()

    left_thread = threading.Thread(
        target=_run_playback,
        kwargs={
            "input_path": LEFT_DEMO,
            "arm_name": "left",
            "gripper_name": "left",
        },
        name="demo-playback-left",
        daemon=False,
    )

    right_thread = threading.Thread(
        target=_run_playback,
        kwargs={
            "input_path": RIGHT_DEMO,
            "arm_name": "right",
            "gripper_name": "right",
        },
        name="demo-playback-right",
        daemon=False,
    )

    print("[TEST] Start dual robot playback")

    left_thread.start()
    right_thread.start()

    left_thread.join()
    right_thread.join()

    print("[TEST] Dual robot playback completed")

    return True


if __name__ == "__main__":
    run_dual_demo()
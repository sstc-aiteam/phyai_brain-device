import time
from services import arm_service, gripper_service, camera_service, model_service, coordinate_service


# ============================================================
# MOVE ARM DEFAULT
# ============================================================

def move_arm_default(
    arm_name,
    speed=None,
    acceleration=None,
):

    result = arm_service.move_arm_default(
        arm_name=arm_name,
        speed=speed,
        acceleration=acceleration,
        wait=True,
    )

    if result.get("result") is not True:
        raise RuntimeError(
            f"Move arm default failed: {result}"
        )

    return result

# ============================================================
# OPEN
# ============================================================  

def open_drawer(
    arm_name,
    gripper_name,
    camera_name,
    dt=0.1,
    speed=0.1,
    acceleration=0.1,
    max_batches=100,
):

    # ============================================================
    # Action Settings
    # ============================================================

    task = "Grasp the black handle on the target drawer and pull the drawer open."

    execute_action_count = 50

    # ============================================================
    # 1. Start Camera
    # ============================================================

    camera_result = camera_service.start_camera(camera_name)

    if camera_result["result"] is not True:
        raise RuntimeError(
            f"Camera start failed: {camera_result}"
        )

    # ============================================================
    # 2. Model Inference & Action Execution
    # ============================================================

    for batch in range(max_batches):

        # --------------------------------------------------------
        # Get Current Arm Pose
        # --------------------------------------------------------

        arm_response = arm_service.get_arm_status(
            arm_name
        )

        arm = next(
            x["status"]
            for x in arm_response["data"]["arms"]
            if x["arm_name"] == arm_name
        )

        current_pose = list(
            arm["pose"]
        )

        # --------------------------------------------------------
        # Get Current Gripper State
        # --------------------------------------------------------

        grip_response = gripper_service.get_gripper_status(
            gripper_name
        )

        grip = next(
            x["status"]
            for x in grip_response["data"]["grippers"]
            if x["gripper_name"] == gripper_name
        )

        if grip["connected"] is not True:
            raise RuntimeError(
                "Gripper is not connected"
            )

        # 如果已經偵測到物體，
        # 即使實際 position < 0.5，
        # 對 SmolVLA 邏輯上仍視為 Closed / Grasping
        if grip["object_detected"] is True:

            gripper_position = 1.0

        else:

            gripper_position = float(
                grip["position"] >= 0.5
            )

        # --------------------------------------------------------
        # Model Inference
        # --------------------------------------------------------

        prediction = model_service.smolVLA.execute(
            prompt=task,
            camera_name=camera_name,
            arm_name=arm_name,
            gripper_name=gripper_name,
            robot_state=[
                *current_pose,
                gripper_position,
            ],
        )

        # SmolVLA 預測 50 筆
        # 每個 batch 只執行前 10 筆
        actions = prediction["actions"][
            :execute_action_count
        ]

        # --------------------------------------------------------
        # Delta Action -> Target TCP Pose
        # --------------------------------------------------------

        target_poses = []
        gripper_targets = []

        for action in actions:

            current_pose = (
                coordinate_service.delta_tcp_to_base_pose(
                    tcp_pose=current_pose,
                    delta_pose=action[:6],
                )
            )

            target_poses.append(
                current_pose
            )

            gripper_targets.append(
                float(
                    action[6] >= 0.5
                )
            )

        # --------------------------------------------------------
        # Execute Arm & Gripper
        # --------------------------------------------------------

        segment_start = 0

        for i, grip_target in enumerate(
            gripper_targets
        ):

            switch = (
                grip_target
                != gripper_position
            )

            is_last = (
                i
                == len(gripper_targets) - 1
            )

            if (
                not switch
                and not is_last
            ):
                continue

            # ----------------------------------------------------
            # Execute Arm Trajectory
            # ----------------------------------------------------

            segment = target_poses[
                segment_start:i + 1
            ]

            if segment:

                result = (
                    arm_service.move_arm_pose_trajectory(
                        arm_name=arm_name,
                        pose_trajectory=segment,
                        dt=dt,
                        speed=speed,
                        acceleration=acceleration,
                        wait=True,
                        move_to_start=False,
                    )
                )

                if result["result"] is not True:
                    raise RuntimeError(
                        f"Arm motion failed: {result}"
                    )

            # ----------------------------------------------------
            # Execute Gripper Action
            # ----------------------------------------------------

            if switch:

                # ========================================================
                # Close
                # ========================================================

                if grip_target >= 0.5:

                    result = (
                        gripper_service.move_gripper(
                            gripper_name=gripper_name,
                            position=1.0,
                            wait=False,
                        )
                    )

                    if result["result"] is not True:
                        raise RuntimeError(
                            f"Gripper motion failed: {result}"
                        )

                    timeout = 3.0
                    poll_interval = 0.05
                    start_time = time.time()

                    saw_moving = False
                    object_detected = None

                    # ------------------------------------------------
                    # Wait Gripper Close
                    # ------------------------------------------------

                    while True:

                        grip_response = (
                            gripper_service.get_gripper_status(
                                gripper_name
                            )
                        )

                        grip = next(
                            x["status"]
                            for x in grip_response["data"]["grippers"]
                            if x["gripper_name"] == gripper_name
                        )

                        if grip["connected"] is not True:
                            raise RuntimeError(
                                "Gripper disconnected while closing"
                            )

                        moving = grip["moving"]

                        object_detected = (
                            grip["object_detected"]
                        )

                        # 曾經看到夾爪開始移動
                        if moving is True:

                            saw_moving = True

                        # 已經偵測到物體
                        # 可直接視為成功夾到
                        if object_detected is True:

                            break

                        # 曾經開始移動，
                        # 現在停止，
                        # 代表 Close 已完成
                        if (
                            saw_moving
                            and moving is False
                        ):

                            break

                        if (
                            time.time()
                            - start_time
                            > timeout
                        ):

                            raise RuntimeError(
                                "Gripper close timeout"
                            )

                        time.sleep(
                            poll_interval
                        )

                    # ------------------------------------------------
                    # Grasp Verification
                    # ------------------------------------------------

                    if object_detected is True:

                        # 成功夾到物體
                        #
                        # 即使實際 position
                        # 只有 0.2 / 0.3 / 0.4，
                        # SmolVLA 邏輯上仍視為 closed
                        gripper_position = 1.0

                    elif object_detected is False:

                        # --------------------------------------------
                        # Empty Grasp
                        # --------------------------------------------

                        result = (
                            gripper_service.move_gripper(
                                gripper_name=gripper_name,
                                position=0.0,
                                wait=True,
                            )
                        )

                        if result["result"] is not True:
                            raise RuntimeError(
                                f"Gripper reopen failed: {result}"
                            )

                        gripper_position = 0.0

                        # 放棄目前剩餘 actions
                        #
                        # 回到下一個 batch：
                        # 1. 重新取得 Camera
                        # 2. 重新取得 Robot State
                        # 3. SmolVLA 重新推論
                        break

                    else:

                        raise RuntimeError(
                            "Unable to verify gripper object detection"
                        )

                # ========================================================
                # Open
                # ========================================================

                else:

                    result = (
                        gripper_service.move_gripper(
                            gripper_name=gripper_name,
                            position=0.0,
                            wait=True,
                        )
                    )

                    if result["result"] is not True:
                        raise RuntimeError(
                            f"Gripper motion failed: {result}"
                        )

                    gripper_position = 0.0

            # ----------------------------------------------------
            # Next Segment
            # ----------------------------------------------------

            segment_start = i + 1

    # ============================================================
    # Result
    # ============================================================

    return {
        "result": True,
        "batches_executed": max_batches,
        "drawer_open_verified": False,
    }

def take_object(
    arm_name,
    gripper_name,
    x,
    y,
    z,
    speed=None,
    acceleration=None,
):

    # ============================================================
    # 1. Open Gripper
    # ============================================================

    result = gripper_service.move_gripper(
        gripper_name=gripper_name,
        position=0.0,
        wait=True,
    )

    if result.get("result") is not True:
        raise RuntimeError(
            f"Gripper open failed: {result}"
        )

    # ============================================================
    # 2. Move Gripper Tip To Target XYZ
    # ============================================================

    result = arm_service.move_gripper_xyz(
        arm_name=arm_name,
        x=x,
        y=y,
        z=z,
        speed=speed,
        acceleration=acceleration,
        wait=True,
    )

    if result.get("result") is not True:
        raise RuntimeError(
            f"Arm motion failed: {result}"
        )

    # ============================================================
    # 3. Close Gripper
    # ============================================================

    result = gripper_service.move_gripper(
        gripper_name=gripper_name,
        position=1.0,
        wait=False,
    )

    if result.get("result") is not True:
        raise RuntimeError(
            f"Gripper close failed: {result}"
        )

    # ============================================================
    # 4. Wait For Gripper Result
    # ============================================================

    timeout = 3.0
    poll_interval = 0.05
    start_time = time.time()

    saw_moving = False
    final_grip = None

    while True:

        grip_response = (
            gripper_service.get_gripper_status(
                gripper_name
            )
        )

        grip = next(
            item["status"]
            for item in grip_response["data"]["grippers"]
            if item["gripper_name"] == gripper_name
        )

        if grip["connected"] is not True:
            raise RuntimeError(
                "Gripper disconnected while closing"
            )

        moving = grip["moving"]
        object_detected = grip["object_detected"]

        final_grip = grip

        # 有看到夾爪進入移動狀態
        if moving is True:
            saw_moving = True

        # 已經偵測到物體，可以直接成功
        if object_detected is True:
            break

        # 曾經移動，現在已停止
        # 表示 close 動作已完成
        if (
            saw_moving
            and moving is False
        ):
            break

        if (
            time.time()
            - start_time
            > timeout
        ):
            raise RuntimeError(
                "Gripper close timeout"
            )

        time.sleep(
            poll_interval
        )

    # ============================================================
    # 5. Final Object Detection
    # ============================================================

    object_detected = (
        final_grip["object_detected"]
    )

    return {
        "result":
            object_detected is True,

        "object_detected":
            object_detected,

        "target_xyz": [
            float(x),
            float(y),
            float(z),
        ],

        "gripper_status":
            final_grip,
    }


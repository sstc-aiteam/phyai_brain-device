import math
from typing import Optional, List, Literal, Dict, Any

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from services import u5_service
from services import gripper_service
from services import vision_service
from services import camera_service
from services import coordinate_service
from services import dispose_service


app = FastAPI(
    title="Robot Brain API",
    description="""
    本 API 提供以下 AI 應用功能：

    AI 視覺辨識：整合 D405 相機與 YOLO 模型，取得影像、深度資訊與目標物辨識結果。
    AI 座標推算：將影像座標與深度資料轉換為機器人可使用的空間座標。
    AI 動作規劃：根據目標物位置與機器人狀態，產生可執行的移動路徑與控制指令。
    AI 任務推論：依據視覺辨識結果、目前姿態與任務流程，判斷下一步要執行的動作。
    AI 自動撿拾任務：結合辨識、座標轉換、手臂控制與夾爪控制，完成目標物抓取與丟棄流程。
    模組化泛化應用：將視覺、推論、規劃、手臂控制與夾爪控制封裝成 API，方便未來替換不同模型、機械手臂或任務場景。
    """,
    version="1.0.0",
)


# ============================================================
# Models
# ============================================================

class MoveXYZRequest(BaseModel):
    x: float = Field(..., description="Target X position in meter")
    y: float = Field(..., description="Target Y position in meter")
    z: float = Field(..., description="Target Z position in meter")
    speed: Optional[float] = Field(None, description="Move speed")
    acceleration: Optional[float] = Field(None, description="Move acceleration")
    wait: bool = Field(True, description="Wait until motion is finished")


class MovePoseRequest(BaseModel):
    x: float = Field(..., description="Target X position in meter")
    y: float = Field(..., description="Target Y position in meter")
    z: float = Field(..., description="Target Z position in meter")
    rx: float = Field(..., description="Target RX rotation in radian")
    ry: float = Field(..., description="Target RY rotation in radian")
    rz: float = Field(..., description="Target RZ rotation in radian")
    speed: Optional[float] = Field(None, description="Move speed")
    acceleration: Optional[float] = Field(None, description="Move acceleration")
    wait: bool = Field(True, description="Wait until motion is finished")


class MoveJointsRequest(BaseModel):
    joints: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Target joint angles in radian, length must be 6",
    )
    speed: Optional[float] = Field(None, description="Move speed")
    acceleration: Optional[float] = Field(None, description="Move acceleration")
    wait: bool = Field(True, description="Wait until motion is finished")


class MoveJointsDegRequest(BaseModel):
    joints_deg: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Target joint angles in degree, length must be 6",
    )
    speed: Optional[float] = Field(None, description="Move speed")
    acceleration: Optional[float] = Field(None, description="Move acceleration")
    wait: bool = Field(True, description="Wait until motion is finished")


class DirectionRequest(BaseModel):
    direction: Literal[
        "x+", "x-",
        "y+", "y-",
        "z+", "z-",
        "rx+", "rx-",
        "ry+", "ry-",
        "rz+", "rz-",
    ] = Field(..., description="Movement, rotation, or jog direction")


class PixelDepthRequest(BaseModel):
    pixel_x: float = Field(..., description="Pixel x / u")
    pixel_y: float = Field(..., description="Pixel y / v")
    depth_m: float = Field(..., description="Depth in meter")


class CameraXYZRequest(BaseModel):
    x: float = Field(..., description="Camera X in meter")
    y: float = Field(..., description="Camera Y in meter")
    z: float = Field(..., description="Camera Z in meter")


class RobotXYZRequest(BaseModel):
    x: float = Field(..., description="Robot base X in meter")
    y: float = Field(..., description="Robot base Y in meter")
    z: float = Field(..., description="Robot base Z in meter")


class RobotXYZToPoseRequest(BaseModel):
    robot_xyz: RobotXYZRequest
    rx: Optional[float] = Field(None, description="TCP RX rotation in radian")
    ry: Optional[float] = Field(None, description="TCP RY rotation in radian")
    rz: Optional[float] = Field(None, description="TCP RZ rotation in radian")
    z_offset: float = Field(0.0, description="Z offset in meter")


class DetectionRequest(BaseModel):
    detection: Dict[str, Any]


class DetectionToPoseRequest(BaseModel):
    detection: Dict[str, Any]
    rx: Optional[float] = Field(None, description="TCP RX rotation in radian")
    ry: Optional[float] = Field(None, description="TCP RY rotation in radian")
    rz: Optional[float] = Field(None, description="TCP RZ rotation in radian")
    z_offset: float = Field(0.0, description="Z offset in meter")


class DisposeRequest(BaseModel):
    trash_pose: List[float] = Field(
        ...,
        description="Trash pose. Supports [x,y,z] or [x,y,z,rx,ry,rz]",
    )
    bin_pose: List[float] = Field(
        ...,
        description="Bin pose. Supports [x,y,z] or [x,y,z,rx,ry,rz]",
    )


# ============================================================
# Helper
# ============================================================

def _call_with_optional_motion_params(func, req):
    kwargs = req.model_dump()

    if kwargs.get("speed") is None:
        kwargs.pop("speed", None)

    if kwargs.get("acceleration") is None:
        kwargs.pop("acceleration", None)

    return func(**kwargs)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():
    return {
        "status": "success",
        "module": "api",
        "message": "UR5 Robot Brain API is running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


# ============================================================
# UR5 Status / Read
# ============================================================

@app.get("/api/ur5/status")
def get_ur5_status():
    return u5_service.get_status()


@app.get("/api/ur5/pose")
def get_ur5_pose():
    return u5_service.get_pose()


@app.get("/api/ur5/joints")
def get_ur5_joints():
    return u5_service.get_joints()


@app.get("/api/ur5/teach/pose")
def teach_pose():
    return u5_service.teach_pose()


@app.get("/api/ur5/teach/joints")
def teach_joints():
    return u5_service.teach_joints()


# ============================================================
# UR5 Motion
# ============================================================

@app.post("/api/ur5/home")
def go_home():
    return u5_service.go_home()


@app.post("/api/ur5/move_xyz")
def move_xyz(req: MoveXYZRequest):
    return _call_with_optional_motion_params(
        u5_service.move_xyz,
        req,
    )


@app.post("/api/ur5/move_pose")
def move_pose(req: MovePoseRequest):
    return _call_with_optional_motion_params(
        u5_service.move_pose,
        req,
    )


@app.post("/api/ur5/move_joints")
def move_joints(req: MoveJointsRequest):
    return _call_with_optional_motion_params(
        u5_service.move_joints,
        req,
    )


@app.post("/api/ur5/move_joints_deg")
def move_joints_deg(req: MoveJointsDegRequest):
    joints_rad = [math.radians(j) for j in req.joints_deg]

    return u5_service.move_joints(
        joints=joints_rad,
        speed=req.speed if req.speed is not None else None,
        acceleration=req.acceleration if req.acceleration is not None else None,
        wait=req.wait,
    )


@app.post("/api/ur5/stop")
def stop_ur5():
    return u5_service.stop()


# ============================================================
# UR5 Step / Rotate / Jog
# ============================================================

@app.post("/api/ur5/move_step")
def move_step(req: DirectionRequest):
    return u5_service.move_step(req.direction)


@app.post("/api/ur5/rotate_step")
def rotate_step(req: DirectionRequest):
    return u5_service.rotate_step(req.direction)


@app.post("/api/ur5/jog/start")
def jog_start(req: DirectionRequest):
    return u5_service.jog_start(req.direction)


@app.post("/api/ur5/jog/stop")
def jog_stop():
    return u5_service.jog_stop()


# ============================================================
# Gripper
# ============================================================

@app.post("/api/gripper/activate")
def gripper_activate():
    return gripper_service.activate()


@app.post("/api/gripper/grip")
def gripper_grip():
    return gripper_service.grip()


@app.post("/api/gripper/release")
def gripper_release():
    return gripper_service.release()


@app.post("/api/gripper/stop")
def gripper_stop():
    return gripper_service.stop()


@app.get("/api/gripper/check")
def gripper_check():
    return gripper_service.check()


# ============================================================
# Camera
# ============================================================

@app.post("/api/camera/start")
def camera_start():
    return camera_service.start_camera()


@app.post("/api/camera/stop")
def camera_stop():
    return camera_service.stop_camera()


@app.get("/api/camera/status")
def camera_status():
    return camera_service.get_camera_status()


@app.get("/api/camera/check")
def camera_check():
    return camera_service.check_camera()


@app.get("/api/camera/images")
def camera_images():
    return camera_service.get_images()


@app.get("/api/camera/frames")
def camera_frames():
    return camera_service.get_frames()


# ============================================================
# Vision Runtime / YOLO
# ============================================================

@app.post("/api/vision/runtime/start")
def vision_runtime_start(
    fps: int = Query(6, ge=1, le=30),
):
    return vision_service.start_runtime(fps=fps)


@app.post("/api/vision/runtime/stop")
def vision_runtime_stop():
    return vision_service.stop_runtime()


@app.get("/api/vision/runtime/status")
def vision_runtime_status():
    return vision_service.runtime_status()


@app.get("/api/vision/detections")
def get_latest_detections():
    return vision_service.get_latest_detections()


@app.get("/api/vision/objects")
def get_detected_objects():
    return vision_service.get_detected_objects()


@app.get("/api/vision/object/best")
def get_best_object(
    class_name: Optional[str] = None,
    class_id: Optional[int] = None,
    require_distance: bool = True,
    sort_by: Literal["confidence", "nearest", "center"] = "confidence",
):
    return vision_service.get_best_object(
        class_name=class_name,
        class_id=class_id,
        require_distance=require_distance,
        sort_by=sort_by,
    )


@app.get("/api/vision/object/nearest")
def get_nearest_object(
    class_name: Optional[str] = None,
    class_id: Optional[int] = None,
):
    return vision_service.get_nearest_object(
        class_name=class_name,
        class_id=class_id,
    )


@app.get("/api/vision/object/center")
def get_center_object(
    class_name: Optional[str] = None,
    class_id: Optional[int] = None,
):
    return vision_service.get_center_object(
        class_name=class_name,
        class_id=class_id,
    )


@app.get("/api/vision/object/highest_confidence")
def get_highest_confidence_object(
    class_name: Optional[str] = None,
    class_id: Optional[int] = None,
):
    return vision_service.get_highest_confidence_object(
        class_name=class_name,
        class_id=class_id,
    )


@app.get("/api/vision/object/has")
def has_object(
    class_name: Optional[str] = None,
    class_id: Optional[int] = None,
):
    return vision_service.has_object(
        class_name=class_name,
        class_id=class_id,
    )


@app.get("/api/vision/center_distance")
def get_center_distance():
    return vision_service.get_center_distance()


@app.get("/api/vision/check")
def vision_check():
    return vision_service.check()


@app.get("/api/vision/yolo.jpg")
def latest_yolo_jpeg(
    quality: int = Query(70, ge=1, le=100),
):
    jpeg_bytes, info = vision_service.get_latest_yolo_jpeg(
        quality=quality,
    )

    if jpeg_bytes is None:
        return info

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
    )


@app.get("/api/vision/yolo_stream")
def latest_yolo_stream(
    fps: int = Query(8, ge=1, le=30),
    quality: int = Query(70, ge=1, le=100),
):
    return StreamingResponse(
        vision_service.latest_yolo_stream_generator(
            fps=fps,
            quality=quality,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# Coordinate Conversion
# ============================================================

@app.get("/api/coordinate/check")
def coordinate_check():
    return coordinate_service.check_config()


@app.post("/api/coordinate/pixel_depth_to_camera_xyz")
def pixel_depth_to_camera_xyz(req: PixelDepthRequest):
    return coordinate_service.pixel_depth_to_camera_xyz(
        pixel_x=req.pixel_x,
        pixel_y=req.pixel_y,
        depth_m=req.depth_m,
    )


@app.post("/api/coordinate/camera_xyz_to_robot_xyz")
def camera_xyz_to_robot_xyz(req: CameraXYZRequest):
    return coordinate_service.camera_xyz_to_robot_xyz(
        camera_xyz={
            "x": req.x,
            "y": req.y,
            "z": req.z,
        },
    )


@app.post("/api/coordinate/pixel_depth_to_robot_xyz")
def pixel_depth_to_robot_xyz(req: PixelDepthRequest):
    return coordinate_service.pixel_depth_to_robot_xyz(
        pixel_x=req.pixel_x,
        pixel_y=req.pixel_y,
        depth_m=req.depth_m,
    )


@app.post("/api/coordinate/detection_to_camera_xyz")
def detection_to_camera_xyz(req: DetectionRequest):
    return coordinate_service.detection_to_camera_xyz(
        detection=req.detection,
    )


@app.post("/api/coordinate/detection_to_robot_xyz")
def detection_to_robot_xyz(req: DetectionRequest):
    return coordinate_service.detection_to_robot_xyz(
        detection=req.detection,
    )


@app.post("/api/coordinate/robot_xyz_to_pose")
def robot_xyz_to_pose(req: RobotXYZToPoseRequest):
    return coordinate_service.robot_xyz_to_pose(
        robot_xyz=req.robot_xyz.model_dump(),
        rx=req.rx,
        ry=req.ry,
        rz=req.rz,
        z_offset=req.z_offset,
    )


@app.post("/api/coordinate/detection_to_robot_pose")
def detection_to_robot_pose(req: DetectionToPoseRequest):
    return coordinate_service.detection_to_robot_pose(
        detection=req.detection,
        rx=req.rx,
        ry=req.ry,
        rz=req.rz,
        z_offset=req.z_offset,
    )


# ============================================================
# Dispose Task
# ============================================================

@app.post("/api/task/dispose")
def dispose(req: DisposeRequest):
    return dispose_service.dispose(
        trash_pose=req.trash_pose,
        bin_pose=req.bin_pose,
    )
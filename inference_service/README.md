# Live inference service

這是一個和既有 Flask/FastAPI 程式分離的 inference client。每個 cycle：

1. 從既有 camera endpoint 取得最新 JPEG。
2. 本服務執行於 `192.168.50.99`，並連線到 `192.168.50.76` 手臂呼叫
   `getActualTCPPose()`。
3. 將 `[x,y,z,rx,ry,rz]` 拆成 `state.eef_position` 與 `state.eef_rotation`。
4. 以 ZeroMQ + msgpack 將 RGB image、TCP pose 與 gripper state 傳到
   `192.168.50.215:5555`。
5. 驗證並輸出下一步相對位置與相對旋轉。

## 對 215 的 request contract

使用 GR00T `BaseInferenceServer` 的 ZeroMQ REQ/REP contract：

- endpoint: `get_action`
- data: JPEG bytes `image`、list `tcp_pose`、float `gripper_position`、
  `instruction`、`sequence_id`。215 server 收到後才將 JPEG 解碼成 RGB NumPy array。

回應的 action 支援模型使用的 delta keys：

```json
{
  "action": {
    "action.eef_position_delta": [0.01, 0.0, -0.01],
    "action.eef_rotation_delta": [0.0, 0.02, 0.0]
  }
}
```

## 執行

先啟動原本 brain-device camera API，然後：

```bash
python -m inference_service
```

可用的環境變數：

- `INFERENCE_ROBOT_IP`（手臂 IP，預設 `192.168.50.76`）
- `INFERENCE_IMAGE_URL`（預設 left camera 的 `/api/vision/camera_rgb`）
- `INFERENCE_CAMERA_NAME`（預設 `left`；服務會先確認相機已啟動，未啟動時只提醒一次）
- `INFERENCE_HOST`（預設 `192.168.50.215`）
- `INFERENCE_PORT`（預設 `5555`）
- `INFERENCE_HZ`（預設 `10`）
- `INFERENCE_TIMEOUT`（預設 `5` 秒）
- `INFERENCE_GRIPPER_POSITION`（預設 `0.0`）
- `INFERENCE_INSTRUCTION`（預設 `open drawer`，與訓練 dataset task 一致）

預設收到的 action 只會輸出 JSON。確認工作區安全後，設定
`INFERENCE_EXECUTE_ACTIONS=true` 才會透過本機 arm API 執行 blocking moveL。
action 預設依訓練資料定義為 TCP/tool-frame delta，單步平移範數上限 0.025 m、
旋轉範數上限 0.05 rad。超限、手臂狀態過期、Emergency Stop 或 Protective Stop
都會拒絕該步。

執行相關設定：

- `INFERENCE_EXECUTE_ACTIONS`（預設 `false`）
- `INFERENCE_ACTION_FRAME`（預設 `tool`，也可設 `base`）
- `INFERENCE_MAX_TRANSLATION_DELTA`（預設 `0.025` m）
- `INFERENCE_MAX_ROTATION_DELTA`（預設 `0.05` rad）
- `INFERENCE_MOTION_SPEED`（預設 `0.05` m/s）
- `INFERENCE_MOTION_ACCELERATION`（預設 `0.1` m/s²）

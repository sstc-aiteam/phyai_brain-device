routes/web.py        = Route / Controller 層

services/\*.py        = Service / 功能執行層

control/ur5.py       = Hardware Control / Driver-like 層

control/gripper.py   = Hardware Control / Driver-like 層

vision/yolo\_d405.py  = Vision Control 層

# Tairos SSTC

## 主要功能

- 機械手臂控制
  - 移動、姿態控制、關節控制
  - 夾爪開合控制
  - 目標物抓取與搬運流程
- 視覺辨識
  - 透過 YOLO 模型進行物件辨識
  - 取得 RGB / 深度影像與偵測結果
- 任務流程
  - 開啟垃圾桶、櫃門、抽屜等組合任務
  - 可與視覺與 VLM 互動完成更高層任務
- Web 與 API 介面
  - Flask 服務提供前端與控制 API
  - FastAPI 服務提供機器人相關 API

## 安裝與執行

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 設定環境

請先編輯 config.py，確認以下項目：

- ARM_IP: UR5 機械手臂 IP（目前設定為 192.168.50.76）
- GRIPPER_IP / GRIPPER_PORT: 夾爪連線設定
- YOLO_MODEL_PATH: YOLO 模型檔案路徑
- OLLAMA_BASE_URL / OLLAMA_MODEL: 若使用 AI Agent，請確認 OLLAMA 已啟動
- ONLY_ARM: 若僅需啟動手臂與夾爪功能，請設為 True

### 4. 啟動 Flask 服務

```bash
python main.py
```


## 常用入口

- 首頁: http://127.0.0.1:5000/
- 按鈕控制頁面: http://127.0.0.1:5000/control
- Demo 頁面: http://127.0.0.1:5000/demo

## 注意事項

- 若使用實體硬體，請先確認機械手臂、夾爪與相機已正確連線。
- YOLO 模型檔案需存在於 models/ 目錄，否則視覺辨識功能無法正常使用。
- 若使用 OLLAMA 代理，請先確認本機服務已啟動且模型可用。
- 部分任務流程需要實際機器人硬體與預設軌跡檔案才能執行。

## 開發說明

若要擴充功能，建議從以下層次著手：

- routes/: 新增 API 路由
- services/: 實作業務流程與狀態管理
- control/: 補上新的硬體驅動或控制邏輯
- agent/: 新增 Agent 行為與工具呼叫

## 資料夾與檔案放置規範（詳細）

這份規範的目標是讓程式碼「放對位置」，避免路由層、服務層、硬體層混在一起，導致後續難維護。

### 0) 架構分層（先理解這個）

- 入口層（main.py / api.py）
  - 只負責啟動服務、註冊路由。
  - 不放業務邏輯與硬體細節。
- 路由層（routes/）
  - 只處理 HTTP 請求、參數解析、回傳格式。
  - 不放任務流程與大量計算。
- 服務層（services/）
  - 放業務邏輯、流程編排、錯誤處理。
  - 路由層只呼叫這層。
- 驅動/模型層（control/、vision/）
  - control/ 放硬體 driver；vision/ 放模型推論實作。
  - 不放 API 路由。
- 前端與資料層（web/、door_trajectory/、models/、certs/）
  - 放頁面、靜態資源、JSON 軌跡、模型檔、憑證。

### 1) 根目錄（專案入口與全域設定）

- main.py
  - Flask 入口，註冊 Blueprint，啟動 HTTPS。
  - 新功能不要直接塞進 main.py，改去 routes/ + services/。
- api.py
  - FastAPI 入口，放 Pydantic Request Model 與 API Endpoint。
  - Endpoint 內只做轉接，實作邏輯交給 services/。
- config.py
  - 全域設定集中管理：IP、速度、加速度、模型路徑、座標矩陣、任務預設 joints、軌跡路徑。
  - 所有可調參數都應該先放這裡，不要散落在各 service。
- requirements.txt
  - 主服務依賴清單。
- README.md
  - 文件、啟動說明、開發規範。
- check_edge_ai_stack.sh
  - 系統健康檢查腳本（port、HTTP、外部服務）。
- trajectory.json
  - 通用軌跡資料樣本/測試資料，不放程式碼。
- 修改重點.txt
  - 臨時紀錄檔，可保留短期追蹤；長期建議改為正式變更紀錄。

### 2) routes/（API 邊界層）

這層原則：

- 放什麼
  - 參數解析（query/body）
  - 欄位驗證與錯誤格式化
  - 呼叫 services 並回傳 JSON
- 不放什麼
  - 複雜流程編排
  - 硬體連線細節
  - 模型推論主程式

各檔案用途：

- routes/web_routes.py: 首頁、demo、control 與靜態檔路由。
- routes/arm_routes.py: 手臂控制 API（移動、jog、任務入口）。
- routes/gripper_routes.py: 夾爪 API。
- routes/camera_routes.py: 相機啟停、畫面串流、距離與像素轉換 API。
- routes/vision_routes.py: 視覺 runtime、偵測查詢、影像輸出 API。
- routes/task_routes.py: 高階任務 API（開門後放置等複合動作）。
- routes/demo_routes.py: demo 流程 API。
- routes/agent_routes.py: Agent 指令入口。
- routes/vlm_routes.py: VLM narrator 狀態與重觀察 API。
- routes/patrol_routes.py: 巡邏工作流 API。
- routes/utils.py: 路由共用工具（json_response、request_error、body 解析）。

### 3) services/（業務邏輯核心）

這層原則：

- 放什麼
  - 業務流程
  - 多服務協作
  - 參數正規化與錯誤處理
  - 任務步驟編排
- 不放什麼
  - HTTP 物件（request/Blueprint）
  - 前端 DOM/畫面邏輯

各檔案用途：

- services/arm_service.py: 手臂動作與軌跡執行（pose/joints/jog/stop）。
- services/gripper_service.py: 夾爪控制與動作保護。
- services/camera_service.py: 相機控制、frame 讀取、距離與像素去投影。
- services/vision_service.py: YOLO runtime、偵測結果整理、目標選擇。
- services/coordinate_service.py: pixel/camera/robot/tcp 座標轉換。
- services/task_service.py: 複合任務主引擎（放置、開關門、JSON 軌跡套用）。
- services/demo_service.py: 展示流程腳本。
- services/record_service.py: 關節錄製與輸出檔管理。
- services/agent_service.py: Agent 與 arm/vision/task 的整合橋接。
- services/vlm_narrator_service.py: 即時場景敘述、VLM 推理與狀態機。
- services/runtime_state.py: 跨模組共享的輕量狀態（如 gripper busy）。

### 4) control/（硬體驅動層）

這層原則：

- 放什麼
  - 實際硬體 driver class
  - 底層連線、收送命令、硬體狀態讀取
- 不放什麼
  - API 路由
  - 任務流程判斷

各檔案用途：

- control/ur5.py: UR5 RTDE driver。
- control/robotiq.py: Robotiq socket driver。
- control/robotiq_urscript.py: URScript 版 Robotiq driver。
- control/robotiq_preamble.py: URScript 前置腳本片段。
- control/d405.py: Intel RealSense D405 driver。
- control/loader.py: 依 config 載入對應 driver，並驗證介面能力。

### 5) agent/（AI 工具調度層）

這層原則：

- 放什麼
  - Prompt 組裝
  - Tool schema/白名單
  - Tool call 驗證與執行器
  - LLM client
- 不放什麼
  - 直接操作 Flask request
  - 直接寫硬體通訊（應走 services/control）

各檔案用途：

- agent/agent.py: Prompt 建構、tool call 格式與參數驗證。
- agent/executor.py: 依序執行工具呼叫、失敗中止策略。
- agent/tools.py: 可供 Agent 呼叫的工具定義。
- agent/ollama_client.py: Ollama 封裝 client。
- agent/rules.json: 物件分類與目的地規則。

### 6) vision/（模型推論層）

- vision/yolo.py
  - YOLO 載入、推論、標記、深度資訊整合。
  - 保持模型邏輯純粹，不放任務流程。

### 7) workflows/（策略腳本層）

- workflows/patrol_workflow.py
  - 巡邏流程策略與狀態管理。
- workflows/grasp_workflow.py
  - 由 detection 產生 grasp 規劃的流程腳本。

建議：可重用的業務規則盡量收斂到 services/；workflows/ 保留為流程實驗與策略組合。

### 8) web/（前端層）

各檔案放置原則：

- web/*.html
  - 放頁面骨架與少量初始化。
  - 複雜互動邏輯放 JS modules，不要塞進大量 inline script。
- web/includes/library/CSSFile/robot2026.css
  - 全站樣式。
- web/includes/library/JSFile/robotjs2026.js
  - 前端主匯出入口。
- web/includes/library/JSFile/modules/core.js
  - 基礎工具（fetch 包裝、UI helper）。
- web/includes/library/JSFile/modules/api.js
  - 聚合模組 API（RobotAPI）。
- web/includes/library/JSFile/modules/arm.js
  - 手臂 API 呼叫封裝。
- web/includes/library/JSFile/modules/gripper.js
  - 夾爪 API 呼叫封裝。
- web/includes/library/JSFile/modules/vision.js
  - 相機/視覺 API 呼叫封裝。
- web/includes/library/JSFile/modules/agent.js
  - Agent/STT/TTS 互動邏輯。
- web/includes/library/JSFile/modules/demo.js
  - Demo API 呼叫。
- web/includes/library/JSFile/modules/vlm.js
  - VLM 前端狀態輪詢與渲染。
- web/assets/
  - 圖片與靜態資源。
- web/example/
  - 功能 API 範例頁，供測試與教學用。

### 9) scene_narrator/（邊緣場景敘述子系統）

- scene_narrator/realtime_scene_narrator.py
  - 獨立 FastAPI 敘述服務。
- scene_narrator/edge_scene_bridge.py
  - 從主系統抓取影像/偵測結果，送到 narrator。
- scene_narrator/demo_description_bank.json
  - 中文敘述語料與物件描述庫。
- scene_narrator/requirements_edge.txt
  - 子系統依賴清單。

### 10) 資料與資產目錄

- door_trajectory/
  - 任務軌跡 JSON（開/關門、抽屜分段）。
  - 只放資料，不放程式。
- models/
  - 模型權重檔（.pt）。
- certs/
  - HTTPS 憑證（server-cert.pem、server-key.pem）。
- hotfix_backups/
  - 歷史備份，不作為正式執行程式來源。

## 新增功能時，檔案要放哪裡（快速指引）

- 新增一支手臂 API
  - routes/arm_routes.py 新增 endpoint。
  - services/arm_service.py 新增對應業務函式。
  - 若需新硬體命令，補到 control/ur5.py。
- 新增一個複合任務
  - services/task_service.py 新增主流程。
  - routes/task_routes.py 新增入口。
  - 若有固定路徑，新增 door_trajectory/*.json 並在 config.py 配路徑。
- 新增 Agent 可呼叫工具
  - agent/tools.py 增加工具定義。
  - agent/agent.py 補驗證規則。
  - services/agent_service.py 補服務整合。
- 新增前端操作按鈕
  - web/includes/library/JSFile/modules/*.js 新增 API 封裝。
  - 對應頁面 html 綁定事件。
  - 視覺樣式放 CSS，不要硬寫 style 到各頁。

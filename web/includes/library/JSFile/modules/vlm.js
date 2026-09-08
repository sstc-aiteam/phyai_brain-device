import { resolveApiUrl } from "./core.js";

"use strict";

/*
 * TaiROS VLM 真實輸出＋狀態秒數合併版
 *
 * 固定顯示：
 *   藍色 YOLO 真實結果
 *   黃色放置建議真實結果
 *   綠色單物件 VLM 真實描述
 *   紫色整體場景 VLM 真實描述
 *
 * 每張卡片同時顯示：
 *   - 目前處理狀態
 *   - 該狀態持續秒數
 *   - 階段耗時／場景累計
 *
 */

const VLM_STATE_URL = resolveApiUrl("/api/vlm/state");
const VLM_REOBSERVE_URL = resolveApiUrl("/api/vlm/reobserve");

const VLM_POLL_INTERVAL_MS = 250;
const LAYER_TYPE_INTERVAL_MS = 42;
const OBJECT_TYPE_INTERVAL_MS = 46;
const OBJECT_CARD_GAP_MS = 260;
const VLM_ERROR_DISPLAY_THRESHOLD = 3;

const LAYER_ORDER = [
    "yolo",
    "placement",
    "object",
    "environment",
];

const LAYER_META = {
    yolo: {
        title: "即時偵測",
        waitingText: "正在讀取物件種類與畫面位置。",
    },
    placement: {
        title: "放置建議",
        waitingText: "等待穩定物件後產生放置位置建議。",
    },
    object: {
        title: "物件外觀",
        waitingText: "等待 VLM 回傳物件外觀描述。",
    },
    environment: {
        title: "場景概括",
        waitingText: "等待 VLM 回傳整體場景概括。",
    },
};

const CLASS_LABELS = {
    bottle_alcohol_spray: "酒精瓶",
    ac_remotecontrol: "遙控器",
    disposable_mask: "口罩",
    square_striped_towel: "條紋毛巾",
    gauze_pp : "紗布",
    waterproof_bandages_ppb: "防水繃帶",
    cotton_swabs_pp: "棉花棒包裝",
    syringe_nipro: "針筒",
    cotton_swab: "棉花棒",
    saline: "生理食鹽水",
};

let vlmInitialized = false;
let vlmPollingTimer = null;
let vlmRequestRunning = false;
let vlmFailureCount = 0;
let lastSceneRevision = null;
let sceneRevisionStartedAt = Date.now();

let reobserveGateActive = false;
let reobserveBaseRevision = null;

let latestVlmPayload = null;

let objectPlaybackSceneKey = null;
let objectPlaybackActiveKey = null;
let objectPlaybackAdvanceTimer = null;

const objectPlaybackRows = new Map();
const objectPlaybackQueue = [];
const objectPlaybackCompletedKeys = [];
const objectTypingState = new Map();
const seenObjectCardKeys = new Set();

const layerTypingState = new Map();

const layerState = Object.fromEntries(
    LAYER_ORDER.map((kind) => [kind, createInitialLayer(kind)]),
);

const layerCompletedAt = Object.fromEntries(
    LAYER_ORDER.map((kind) => [kind, null]),
);

const layerActivatedAt = Object.fromEntries(
    LAYER_ORDER.map((kind) => [kind, null]),
);

const layerStatusStartedAt = Object.fromEntries(
    LAYER_ORDER.map((kind) => [kind, Date.now()]),
);


function installStatusTestStyles()
{
    if (document.getElementById("vlmCombinedStyles"))
    {
        return;
    }

    const style = document.createElement("style");
    style.id = "vlmCombinedStyles";
    style.textContent = `
#understandingFeed.vlm-combined-feed{
    display:flex !important;
    flex-direction:column !important;
    align-items:stretch !important;
    gap:7px !important;
    min-width:0 !important;
    padding:2px 3px 8px 0 !important;
    overflow-x:hidden !important;
    overflow-y:auto !important;
}

#understandingFeed .vlm-combined-card{
    box-sizing:border-box !important;
    position:relative !important;
    display:flex !important;
    flex-direction:column !important;
    flex:0 0 auto !important;
    width:100% !important;
    min-width:0 !important;
    min-height:62px !important;
    margin:0 !important;
    padding:9px 10px 8px 11px !important;
    gap:4px !important;
    border:1px solid rgba(148,163,184,.18) !important;
    border-left:5px solid rgba(148,163,184,.55) !important;
    border-radius:10px !important;
    overflow:hidden !important;
    box-shadow:0 3px 10px rgba(0,0,0,.11) !important;
    color:#f8fafc !important;
}

#understandingFeed .vlm-combined-card.yolo{
    border-left-color:#38bdf8 !important;
    background:linear-gradient(90deg,rgba(14,116,144,.16),rgba(14,116,144,.05)) !important;
}
#understandingFeed .vlm-combined-card.placement{
    min-height:82px !important;
    border-left-color:#fbbf24 !important;
    background:linear-gradient(90deg,rgba(217,119,6,.16),rgba(217,119,6,.05)) !important;
}

#understandingFeed .vlm-combined-card.placement .vlm-combined-output{
    display:block !important;
    height:auto !important;
    overflow:visible !important;
    white-space:pre-line !important;
    -webkit-line-clamp:unset !important;
    line-clamp:unset !important;
}
#understandingFeed .vlm-combined-card.object{
    border-left-color:#4ade80 !important;
    background:linear-gradient(90deg,rgba(22,163,74,.15),rgba(22,163,74,.05)) !important;
}
#understandingFeed .vlm-combined-card.environment{
    min-height:108px !important;
    border-left-color:#c084fc !important;
    background:linear-gradient(90deg,rgba(126,34,206,.16),rgba(126,34,206,.05)) !important;
}

#understandingFeed .vlm-combined-card.environment .vlm-combined-output{
    -webkit-line-clamp:4 !important;
    line-clamp:4 !important;
}

#understandingFeed .vlm-combined-header{
    display:grid !important;
    grid-template-columns:minmax(0,1fr) auto !important;
    align-items:center !important;
    gap:7px !important;
    min-width:0 !important;
}

#understandingFeed .vlm-combined-title{
    min-width:0 !important;
    color:#f8fafc !important;
    font-size:13.5px !important;
    font-weight:900 !important;
    line-height:1.3 !important;
    white-space:nowrap !important;
    overflow:hidden !important;
    text-overflow:ellipsis !important;
}

#understandingFeed .vlm-combined-badge{
    justify-self:end !important;
    max-width:110px !important;
    min-width:0 !important;
    padding:2px 7px !important;
    border:1px solid rgba(148,163,184,.48) !important;
    border-radius:999px !important;
    background:rgba(15,23,42,.76) !important;
    color:#e2e8f0 !important;
    font-size:10.5px !important;
    font-weight:850 !important;
    line-height:1.3 !important;
    white-space:nowrap !important;
}

#understandingFeed .vlm-combined-badge.done{
    border-color:rgba(74,222,128,.78) !important;
    color:#bbf7d0 !important;
}
#understandingFeed .vlm-combined-badge.running{
    border-color:rgba(56,189,248,.78) !important;
    color:#bae6fd !important;
}
#understandingFeed .vlm-combined-badge.waiting,
#understandingFeed .vlm-combined-badge.idle{
    border-color:rgba(250,204,21,.68) !important;
    color:#fef08a !important;
}
#understandingFeed .vlm-combined-badge.warning{
    border-color:rgba(251,146,60,.78) !important;
    color:#fed7aa !important;
}
#understandingFeed .vlm-combined-badge.error{
    border-color:rgba(248,113,113,.82) !important;
    color:#fecaca !important;
}

#understandingFeed .vlm-combined-output{
    width:100% !important;
    min-height:20px !important;
    margin:0 !important;
    padding:0 !important;
    color:#f8fafc !important;
    font-size:13.5px !important;
    font-weight:550 !important;
    line-height:1.42 !important;
    white-space:pre-line !important;
    word-break:break-word !important;
    overflow-wrap:anywhere !important;
    display:-webkit-box !important;
    -webkit-box-orient:vertical !important;
    -webkit-line-clamp:2 !important;
    line-clamp:2 !important;
    overflow:hidden !important;
}

#understandingFeed .vlm-combined-card.object .vlm-combined-output{
    -webkit-line-clamp:3 !important;
    line-clamp:3 !important;
}

#understandingFeed .vlm-combined-diagnostic,
#understandingFeed .vlm-combined-footer{
    display:none !important;
}

#understandingFeed .vlm-combined-card[data-status="error"] .vlm-combined-diagnostic{
    display:block !important;
    margin-top:2px !important;
    padding:5px 7px !important;
    border:1px solid rgba(248,113,113,.22) !important;
    border-radius:7px !important;
    background:rgba(127,29,29,.12) !important;
    color:#fecaca !important;
    font-size:10px !important;
    line-height:1.35 !important;
    overflow-wrap:anywhere !important;
}

#understandingFeed .vlm-object-card{
    min-height:74px !important;
    border-left-width:5px !important;
    background:
        linear-gradient(90deg,rgba(22,163,74,.18),rgba(22,163,74,.055))
        !important;
}

#understandingFeed .vlm-object-card .vlm-combined-title{
    color:#dcfce7 !important;
    font-size:14px !important;
}

#understandingFeed .vlm-object-card .vlm-combined-output{
    display:block !important;
    min-height:22px !important;
    overflow:visible !important;
    color:#f0fdf4 !important;
    font-size:13.5px !important;
    font-weight:600 !important;
    line-height:1.48 !important;
    white-space:normal !important;
    -webkit-line-clamp:unset !important;
    line-clamp:unset !important;
}

#understandingFeed .vlm-combined-output.typing::after{
    content:"▋";
    display:inline-block;
    margin-left:2px;
    color:currentColor;
    animation:vlmCursorBlink .72s steps(1,end) infinite;
}

#understandingFeed .vlm-combined-card.yolo .vlm-combined-output.typing::after{
    color:#7dd3fc;
}
#understandingFeed .vlm-combined-card.placement .vlm-combined-output.typing::after{
    color:#fde68a;
}
#understandingFeed .vlm-combined-card.object .vlm-combined-output.typing::after{
    color:#86efac;
}
#understandingFeed .vlm-combined-card.environment .vlm-combined-output.typing::after{
    color:#d8b4fe;
}

#understandingFeed .vlm-object-card.vlm-card-enter{
    animation:vlmObjectCardEnter .38s cubic-bezier(.2,.78,.25,1) both;
}

@keyframes vlmObjectCardEnter{
    0%{
        opacity:0;
        transform:translateY(9px) scale(.975);
        box-shadow:0 0 0 rgba(74,222,128,0);
    }
    70%{
        opacity:1;
        transform:translateY(-1px) scale(1.006);
        box-shadow:0 8px 24px rgba(34,197,94,.16);
    }
    100%{
        opacity:1;
        transform:translateY(0) scale(1);
        box-shadow:0 3px 10px rgba(0,0,0,.11);
    }
}

@keyframes vlmCursorBlink{
    0%,48%{opacity:1;}
    49%,100%{opacity:0;}
}

@media (prefers-reduced-motion: reduce){
    #understandingFeed .vlm-object-card.vlm-card-enter{
        animation:none !important;
    }

    #understandingFeed .vlm-combined-output.typing::after{
        animation:none !important;
    }
}
`;

    document.head.appendChild(style);
}

function createInitialLayer(kind)
{
    return {
        kind,
        text: LAYER_META[kind].waitingText,
        status: "等待資料",
        statusTone: "waiting",
        detail: "尚未取得場景狀態。",
        updatedAt: null,
    };
}

function isVlmPollingAllowed()
{
    if (typeof window.isVlmPollingAllowed !== "function")
    {
        return true;
    }

    return window.isVlmPollingAllowed();
}

function normalizeText(value)
{
    return String(value ?? "")
        .replace(/\s+/g, " ")
        .trim();
}

function nowText()
{
    return new Intl.DateTimeFormat(
        "zh-TW",
        {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false,
        },
    ).format(new Date());
}

function elapsedSeconds(fromAt, toAt = Date.now())
{
    const startAt = Number(fromAt);
    const endAt = Number(toAt);
    if (!Number.isFinite(startAt) || !Number.isFinite(endAt))
    {
        return 0;
    }
    return Math.max(0, Math.floor((endAt - startAt) / 1000));
}

function activateLayer(kind)
{
    if (layerActivatedAt[kind] === null)
    {
        layerActivatedAt[kind] = Date.now();
    }
}

function currentStatusSeconds(kind)
{
    return elapsedSeconds(layerStatusStartedAt[kind] ?? sceneRevisionStartedAt);
}

function badgeText(kind, item)
{
    const shortStatus = {
        done: "完成",
        running: "處理中",
        waiting: "等待",
        idle: "待命",
        warning: "注意",
        error: "錯誤",
    }[item.statusTone] ?? item.status;

    const seconds = layerCompletedAt[kind] !== null
        ? elapsedSeconds(
            layerActivatedAt[kind] ?? sceneRevisionStartedAt,
            layerCompletedAt[kind],
        )
        : currentStatusSeconds(kind);

    return `${shortStatus} · ${seconds}秒`;
}

function elapsedText(kind)
{
    const now = Date.now();
    const completedAt = layerCompletedAt[kind];
    const endAt = completedAt ?? now;
    const activatedAt = layerActivatedAt[kind];
    const sceneSeconds = elapsedSeconds(sceneRevisionStartedAt, endAt);

    if (completedAt !== null)
    {
        const stageSeconds = elapsedSeconds(
            activatedAt ?? sceneRevisionStartedAt,
            completedAt,
        );
        return `階段耗時 ${stageSeconds} 秒｜場景累計 ${sceneSeconds} 秒`;
    }

    if (activatedAt === null)
    {
        return `等待啟動 ${sceneSeconds} 秒｜此狀態 ${currentStatusSeconds(kind)} 秒`;
    }

    return `階段進行 ${elapsedSeconds(activatedAt, now)} 秒｜此狀態 ${currentStatusSeconds(kind)} 秒｜場景累計 ${sceneSeconds} 秒`;
}

function entryText(entry)
{
    if (typeof entry === "string")
    {
        return normalizeText(entry);
    }

    return normalizeText(
        entry?.text
        ?? entry?.message
        ?? entry?.description
        ?? entry?.content
        ?? "",
    );
}

function ensureSentence(text)
{
    const normalized = normalizeText(text);

    if (!normalized)
    {
        return "";
    }

    if (/[。！？!?]$/.test(normalized))
    {
        return normalized;
    }

    return `${normalized}。`;
}

function classLabel(className)
{
    const key = normalizeText(className)
        .toLowerCase()
        .replace(/[\s-]+/g, "_");

    if (!key)
    {
        return "物件";
    }

    return CLASS_LABELS[key] ?? "未分類物件";
}

function objectBox(object)
{
    const arrayBox = (
        object?.bbox
        ?? object?.bbox_xyxy
        ?? object?.xyxy
    );

    if (Array.isArray(arrayBox) && arrayBox.length >= 4)
    {
        return arrayBox.slice(0, 4).map(Number);
    }

    const boxObject = object?.box;

    if (boxObject && typeof boxObject === "object")
    {
        return [
            Number(boxObject.x1 ?? boxObject.left),
            Number(boxObject.y1 ?? boxObject.top),
            Number(boxObject.x2 ?? boxObject.right),
            Number(boxObject.y2 ?? boxObject.bottom),
        ];
    }

    return null;
}

function objectCenterX(object)
{
    const centerX = Number(
        object?.center?.x
        ?? object?.center_x
        ?? object?.cx,
    );

    if (Number.isFinite(centerX))
    {
        return centerX;
    }

    const box = objectBox(object);

    if (
        Array.isArray(box)
        && Number.isFinite(box[0])
        && Number.isFinite(box[2])
    )
    {
        return (box[0] + box[2]) / 2;
    }

    return null;
}

function horizontalPosition(object, frameWidth)
{
    const centerX = objectCenterX(object);
    const width = Number(frameWidth);

    if (
        !Number.isFinite(centerX)
        || !Number.isFinite(width)
        || width <= 0
    )
    {
        return "";
    }

    const ratio = centerX / width;

    if (ratio < 0.34)
    {
        return "左側";
    }

    if (ratio > 0.66)
    {
        return "右側";
    }

    return "中央";
}

function getScene(payload)
{
    return (
        payload?.scene
        && typeof payload.scene === "object"
    )
        ? payload.scene
        : payload ?? {};
}

function sceneValue(payload, key, fallback = null)
{
    const scene = getScene(payload);

    if (scene?.[key] !== undefined && scene?.[key] !== null)
    {
        return scene[key];
    }

    if (payload?.[key] !== undefined && payload?.[key] !== null)
    {
        return payload[key];
    }

    return fallback;
}

function getStableObjects(payload)
{
    if (Array.isArray(payload?.stable_objects))
    {
        return payload.stable_objects;
    }

    if (Array.isArray(payload?.scene?.stable_objects))
    {
        return payload.scene.stable_objects;
    }

    if (Array.isArray(payload?.scene?.objects))
    {
        return payload.scene.objects;
    }

    return [];
}

function buildYoloText(payload)
{
    const objects = getStableObjects(payload);

    if (objects.length === 0)
    {
        return "";
    }

    const frameWidth = sceneValue(payload, "frame_width", null);
    const positionGroups = new Map();

    for (const object of objects)
    {
        const label = classLabel(
            object?.class_name
            ?? object?.label
            ?? object?.name,
        );

        const position = horizontalPosition(object, frameWidth);
        const positionKey = position || "__unknown__";

        if (!positionGroups.has(positionKey))
        {
            positionGroups.set(
                positionKey,
                {
                    position,
                    labels: new Map(),
                },
            );
        }

        const group = positionGroups.get(positionKey);
        group.labels.set(
            label,
            (group.labels.get(label) ?? 0) + 1,
        );
    }

    const descriptions = [];

    for (const group of positionGroups.values())
    {
        const objectTexts = [];

        for (const [label, count] of group.labels.entries())
        {
            objectTexts.push(
                count > 1
                    ? `${count}個${label}`
                    : label,
            );
        }

        const combinedObjects = objectTexts.join("、");

        if (!combinedObjects)
        {
            continue;
        }

        descriptions.push(
            group.position
                ? `${group.position}有${combinedObjects}`
                : `偵測到${combinedObjects}`,
        );
    }

    return ensureSentence(descriptions.join("；"));
}

function uniqueTexts(entries)
{
    const texts = [];
    const seen = new Set();

    for (const entry of entries)
    {
        const text = entryText(entry);

        if (!text || seen.has(text))
        {
            continue;
        }

        seen.add(text);
        texts.push(text);
    }

    return texts;
}

function getSceneLines(payload)
{
    if (Array.isArray(payload?.scene?.lines))
    {
        return payload.scene.lines;
    }

    if (Array.isArray(payload?.lines))
    {
        return payload.lines;
    }

    return [];
}

function latestLineText(payload, acceptedTypes)
{
    const accepted = new Set(acceptedTypes);
    const lines = getSceneLines(payload);

    for (let index = lines.length - 1; index >= 0; index -= 1)
    {
        const line = lines[index];
        const type = normalizeText(line?.type).toLowerCase();

        if (!accepted.has(type))
        {
            continue;
        }

        const text = entryText(line);

        if (text)
        {
            return text;
        }
    }

    return "";
}

function joinNaturalChineseList(items)
{
    const values = items
        .map((item) => normalizeText(item))
        .filter(Boolean);

    if (values.length === 0)
    {
        return "";
    }

    if (values.length === 1)
    {
        return values[0];
    }

    if (values.length === 2)
    {
        return `${values[0]}與${values[1]}`;
    }

    return `${values.slice(0, -1).join("、")}與${values.at(-1)}`;
}

function naturalPlacementClause(
    destination,
    destinationLabel,
    objectLabels,
)
{
    const objectsText = joinNaturalChineseList(objectLabels);

    if (!objectsText || !destinationLabel)
    {
        return "";
    }

    if (
        destination === "trash_can"
        || destinationLabel.includes("垃圾桶")
    )
    {
        return `將${objectsText}丟入${destinationLabel}`;
    }

    if (destinationLabel.includes("抽屜"))
    {
        return `將${objectsText}收納至${destinationLabel}`;
    }

    if (
        destinationLabel.includes("櫃")
        || destinationLabel.includes("箱")
        || destinationLabel.includes("盒")
    )
    {
        return `將${objectsText}放入${destinationLabel}`;
    }

    return `將${objectsText}放置於${destinationLabel}`;
}

function buildPlacementText(payload)
{
    const recommendations = (
        payload?.scene?.placement_recommendations
        ?? payload?.placement_recommendations
    );

    if (Array.isArray(recommendations) && recommendations.length > 0)
    {
        const groups = new Map();

        for (const item of recommendations)
        {
            const destination = normalizeText(
                item?.destination
                ?? "",
            );
            const destinationLabel = normalizeText(
                item?.destination_label
                ?? item?.destination
                ?? "",
            );
            const objectLabel = normalizeText(
                item?.object_label
                ?? item?.class_name
                ?? "",
            );

            if (!destinationLabel || !objectLabel)
            {
                continue;
            }

            const groupKey = `${destination}::${destinationLabel}`;

            if (!groups.has(groupKey))
            {
                groups.set(
                    groupKey,
                    {
                        destination,
                        destinationLabel,
                        objectLabels: [],
                    },
                );
            }

            const group = groups.get(groupKey);

            if (!group.objectLabels.includes(objectLabel))
            {
                group.objectLabels.push(objectLabel);
            }
        }

        const clauses = [];

        for (const group of groups.values())
        {
            const clause = naturalPlacementClause(
                group.destination,
                group.destinationLabel,
                group.objectLabels,
            );

            if (clause)
            {
                clauses.push(clause);
            }
        }

        if (clauses.length > 0)
        {
            return `依照目前整理推論，建議${clauses.join("；")}。`;
        }
    }

    const summary = entryText(
        payload?.scene?.placement_summary
        ?? payload?.placement_summary,
    );

    return summary ? ensureSentence(summary) : "";
}


function getObjectDescriptionRows(payload)
{
    const rows = payload
        ?.scene
        ?.narration
        ?.object_descriptions;

    return Array.isArray(rows)
        ? rows.filter((row) => row && typeof row === "object")
        : [];
}

function getObjectPreviewRows(payload)
{
    const rows = (
        payload?.scene?.object_preview_descriptions
        ?? payload?.object_preview_descriptions
    );

    return Array.isArray(rows)
        ? rows.filter((row) => row && typeof row === "object")
        : [];
}

function objectLabelById(payload)
{
    const map = new Map();
    const objects = [
        ...(Array.isArray(payload?.scene?.semantic_objects)
            ? payload.scene.semantic_objects
            : []),
        ...getStableObjects(payload),
    ];

    for (const object of objects)
    {
        const objectId = Number(
            object?.id
            ?? object?.object_id
            ?? object?.track_id,
        );

        if (!Number.isFinite(objectId))
        {
            continue;
        }

        map.set(
            objectId,
            classLabel(
                object?.class_name
                ?? object?.label
                ?? object?.name,
            ),
        );
    }

    return map;
}

function actualObjectDescriptionCount(payload)
{
    return getObjectDescriptionRows(payload)
        .filter((row) => entryText(row))
        .length;
}

function buildObjectText(payload)
{
    const actualRows = getObjectDescriptionRows(payload);
    const previewRows = getObjectPreviewRows(payload);
    const labels = objectLabelById(payload);
    const actualById = new Map();
    const freeActualTexts = [];

    for (const row of actualRows)
    {
        const text = entryText(row);
        if (!text)
        {
            continue;
        }

        const objectId = Number(row?.object_id);
        if (Number.isFinite(objectId))
        {
            actualById.set(objectId, text);
        }
        else
        {
            freeActualTexts.push(text);
        }
    }

    const orderedIds = [];
    for (const row of previewRows)
    {
        const objectId = Number(row?.object_id);
        if (Number.isFinite(objectId) && !orderedIds.includes(objectId))
        {
            orderedIds.push(objectId);
        }
    }
    for (const objectId of actualById.keys())
    {
        if (!orderedIds.includes(objectId))
        {
            orderedIds.push(objectId);
        }
    }

    const texts = [];

    for (const objectId of orderedIds)
    {
        const label = labels.get(objectId)
            ?? normalizeText(
                previewRows.find(
                    (row) => Number(row?.object_id) === objectId,
                )?.object_label,
            )
            ?? "物件";

        const actualText = actualById.get(objectId);

        if (actualText)
        {
            const normalizedActual = ensureSentence(actualText);
            texts.push(
                normalizedActual.startsWith(label)
                    ? normalizedActual
                    : `${label}：${normalizedActual}`,
            );
        }
        else
        {
            texts.push(`${label}：分析中…`);
        }
    }

    for (const freeText of freeActualTexts)
    {
        texts.push(ensureSentence(freeText));
    }

    if (texts.length > 0)
    {
        const visibleLimit = 2;
        const visible = texts.slice(0, visibleLimit);
        const hiddenCount = Math.max(0, texts.length - visible.length);

        if (hiddenCount > 0)
        {
            visible.push(`＋${hiddenCount} 項`);
        }

        return visible.join("\n");
    }

    return ensureSentence(
        latestLineText(payload, ["object"]),
    );
}

function buildEnvironmentText(payload)
{
    const lineText = latestLineText(
        payload,
        ["idle", "summary"],
    );

    if (lineText)
    {
        return ensureSentence(lineText);
    }

    const narration = payload?.scene?.narration;

    if (!narration || typeof narration !== "object")
    {
        return "";
    }

    const sceneSummary = entryText(narration.scene_summary);

    if (sceneSummary)
    {
        return ensureSentence(sceneSummary);
    }

    const relationships = Array.isArray(
        narration.relationship_descriptions,
    )
        ? uniqueTexts(narration.relationship_descriptions)
        : [];

    if (relationships.length > 0)
    {
        return ensureSentence(relationships.join("；"));
    }

    return "";
}

function booleanText(value)
{
    return value ? "是" : "否";
}

function numberValue(value, fallback = 0)
{
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function setLayerState(
    kind,
    {
        text,
        status,
        statusTone,
        detail,
        completed = false,
    },
)
{
    const previous = layerState[kind];
    const normalizedText = normalizeText(text) || LAYER_META[kind].waitingText;
    const normalizedStatus = normalizeText(status) || "等待中";
    const normalizedTone = normalizeText(statusTone) || "waiting";
    const statusChanged = (
        previous?.status !== normalizedStatus
        || previous?.statusTone !== normalizedTone
    );

    if (statusChanged)
    {
        layerStatusStartedAt[kind] = Date.now();
    }

    if (completed && layerCompletedAt[kind] === null)
    {
        activateLayer(kind);
        layerCompletedAt[kind] = Date.now();
    }

    layerState[kind] = {
        kind,
        text: normalizedText,
        status: normalizedStatus,
        statusTone: normalizedTone,
        detail: normalizeText(detail),
        updatedAt: (previous?.text !== normalizedText || statusChanged)
            ? nowText()
            : previous?.updatedAt,
    };
}

function updateAllLayerStates(payload)
{
    latestVlmPayload = payload;
    const sceneRevision = sceneValue(payload, "scene_revision", null);
    const stableObjects = getStableObjects(payload);
    const stableCount = numberValue(
        sceneValue(payload, "stable_object_count", stableObjects.length),
        stableObjects.length,
    );
    const objectCount = numberValue(
        sceneValue(payload, "object_count", stableCount),
        stableCount,
    );
    const scenePending = Boolean(
        sceneValue(payload, "scene_change_pending", false),
    );
    const confirmCount = numberValue(
        sceneValue(payload, "scene_confirm_count", 0),
    );
    const confirmRequired = Math.max(
        1,
        numberValue(
            sceneValue(payload, "scene_confirm_required", 2),
            2,
        ),
    );
    const objectDone = Boolean(
        sceneValue(payload, "object_stage_complete", false),
    );
    const environmentDone = Boolean(
        sceneValue(payload, "environment_stage_complete", false),
    );
    const qwenPending = Boolean(
        sceneValue(payload, "qwen_pending", false),
    );
    const qwenStarted = Boolean(
        sceneValue(payload, "qwen_started", false),
    );
    const qwenStreaming = Boolean(
        sceneValue(payload, "qwen_streaming", false),
    );
    const qwenExpectedCount = Math.max(
        0,
        numberValue(
            sceneValue(payload, "qwen_expected_object_count", stableCount),
            stableCount,
        ),
    );
    const qwenReceivedCount = Math.max(
        0,
        numberValue(
            sceneValue(payload, "qwen_received_object_count", 0),
            0,
        ),
    );
    const qwenStreamElapsed = Math.max(
        0,
        numberValue(
            sceneValue(payload, "qwen_stream_elapsed_sec", 0),
            0,
        ),
    );
    const qwenFirstResultSec = sceneValue(
        payload,
        "qwen_first_result_sec",
        null,
    );
    const qwenFirstDetailTarget = Math.max(
        1,
        numberValue(
            sceneValue(payload, "qwen_first_detail_target_sec", 10),
            10,
        ),
    );
    const engineBusy = Boolean(
        sceneValue(payload, "qwen_engine_busy", false),
    );
    const qwenError = normalizeText(
        sceneValue(payload, "qwen_error", ""),
    );
    const generalPending = Boolean(
        sceneValue(payload, "general_idle_pending", false),
    );
    const generalStarted = Boolean(
        sceneValue(payload, "general_idle_started", false),
    );
    const purpleAllowed = Boolean(
        sceneValue(payload, "purple_allowed", true),
    );
    const greenPurpleGap = Math.max(
        0,
        numberValue(
            sceneValue(payload, "green_purple_gap_remaining_sec", 0),
        ),
    );
    const yoloText = buildYoloText(payload);
    const placementText = buildPlacementText(payload);
    const objectText = buildObjectText(payload);
    const actualObjectCount = actualObjectDescriptionCount(payload);
    const environmentText = buildEnvironmentText(payload);

    activateLayer("yolo");
    if (stableCount > 0 && !scenePending)
    {
        activateLayer("placement");
        activateLayer("object");
    }
    if (objectDone)
    {
        activateLayer("environment");
    }

    if (stableCount > 0 && !scenePending)
    {
        setLayerState(
            "yolo",
            {
                text: yoloText,
                status: `已取得 ${stableCount} 個穩定物件`,
                statusTone: "done",
                detail: (
                    `即時物件 ${objectCount} 個｜場景版本 ${sceneRevision ?? "-"}`
                ),
                completed: true,
            },
        );
    }
    else if (objectCount > 0 || scenePending)
    {
        setLayerState(
            "yolo",
            {
                text: yoloText || "已偵測到候選物件，正在等待連續畫面確認。",
                status: `等待穩定中（${confirmCount}/${confirmRequired}）`,
                statusTone: "running",
                detail: `即時物件 ${objectCount} 個｜穩定物件 ${stableCount} 個`,
            },
        );
    }
    else
    {
        setLayerState(
            "yolo",
            {
                text: "目前尚未偵測到可用物件。",
                status: "持續偵測中",
                statusTone: "waiting",
                detail: `場景版本 ${sceneRevision ?? "-"}`,
            },
        );
    }

    const placementRecommendations = (
        payload?.scene?.placement_recommendations
        ?? payload?.placement_recommendations
    );
    const placementContractAvailable = (
        payload?.scene?.placement_summary !== undefined
        || payload?.placement_summary !== undefined
        || Array.isArray(placementRecommendations)
    );

    if (stableCount <= 0 || scenePending)
    {
        setLayerState(
            "placement",
            {
                text: "等待 YOLO 產生穩定物件後再套用放置推論。",
                status: "等待穩定物件",
                statusTone: "waiting",
                detail: `穩定物件 ${stableCount} 個`,
            },
        );
    }
    else if (placementText)
    {
        const ruleCount = Array.isArray(placementRecommendations)
            ? placementRecommendations.length
            : 0;

        setLayerState(
            "placement",
            {
                text: placementText,
                status: "放置建議已完成",
                statusTone: "done",
                detail: `規則命中 ${ruleCount} 筆｜只顯示建議，不執行工具`,
                completed: true,
            },
        );
    }
    else if (!placementContractAvailable)
    {
        setLayerState(
            "placement",
            {
                text: "後端尚未回傳 placement_summary 或 placement_recommendations。",
                status: "等待放置建議資料",
                statusTone: "running",
                detail: "請確認 8014 state 是否包含 placement 欄位。",
            },
        );
    }
    else
    {
        setLayerState(
            "placement",
            {
                text: "目前偵測物件沒有對應的固定放置規則。",
                status: "沒有適用規則",
                statusTone: "idle",
                detail: `穩定物件 ${stableCount} 個｜規則命中 0 筆`,
                completed: true,
            },
        );
    }

    if (objectDone && qwenExpectedCount === 0 && stableCount === 0)
    {
        setLayerState(
            "object",
            {
                text: "目前沒有可描述的物件。",
                status: "沒有物件",
                statusTone: "done",
                detail: "empty_scene=是｜未呼叫 VLM",
                completed: true,
            },
        );
    }
    else if (objectText && objectDone)
    {
        setLayerState(
            "object",
            {
                text: objectText,
                status: `物件描述已完成（${actualObjectCount}/${Math.max(actualObjectCount, qwenExpectedCount)}）`,
                statusTone: "done",
                detail: (
                    `object_done=${booleanText(objectDone)}｜`
                    + `first_result=${qwenFirstResultSec ?? "-"} 秒｜`
                    + `engine_busy=${booleanText(engineBusy)}`
                ),
                completed: true,
            },
        );
    }
    else if (qwenError && !qwenError.startsWith("general_idle_"))
    {
        setLayerState(
            "object",
            {
                text: objectText || qwenError,
                status: actualObjectCount > 0
                    ? `已取得 ${actualObjectCount} 筆，後續分析失敗`
                    : "物件描述失敗",
                statusTone: actualObjectCount > 0 ? "warning" : "error",
                detail: (
                    `qwen_pending=${booleanText(qwenPending)}｜`
                    + qwenError
                ),
            },
        );
    }
    else if (
        qwenPending
        || qwenStarted
        || qwenStreaming
    )
    {
        const expected = Math.max(
            qwenExpectedCount,
            stableCount,
            actualObjectCount,
        );

        if (actualObjectCount > 0)
        {
            setLayerState(
                "object",
                {
                    text: objectText,
                    status: `已完成 ${actualObjectCount}/${expected} 個，持續分析`,
                    statusTone: "running",
                    detail: (
                        `串流 ${qwenStreamElapsed.toFixed(1)} 秒｜`
                        + `首筆 ${qwenFirstResultSec ?? "-"} 秒｜`
                        + `engine_busy=${booleanText(engineBusy)}`
                    ),
                },
            );
        }
        else
        {
            const exceededTarget = (
                qwenStreamElapsed >= qwenFirstDetailTarget
            );

            setLayerState(
                "object",
                {
                    text: objectText
                        || "物件名稱已確認，正在等待第一筆詳細外觀。",
                    status: exceededTarget
                        ? "快速結果已顯示，詳細外觀仍在分析"
                        : "等待第一筆 VLM 描述",
                    statusTone: exceededTarget ? "warning" : "running",
                    detail: (
                        `已等待 ${qwenStreamElapsed.toFixed(1)} 秒｜`
                        + `目標 ${qwenFirstDetailTarget.toFixed(0)} 秒｜`
                        + `預計物件 ${expected} 個`
                    ),
                },
            );
        }
    }
    else if (stableCount <= 0 || scenePending)
    {
        setLayerState(
            "object",
            {
                text: objectText
                    || "等待 YOLO 物件穩定後開始外觀分析。",
                status: "等待場景穩定",
                statusTone: "waiting",
                detail: (
                    `穩定物件 ${stableCount} 個｜`
                    + `scene_pending=${booleanText(scenePending)}`
                ),
            },
        );
    }
    else if (objectDone)
    {
        setLayerState(
            "object",
            {
                text: objectText
                    || "物件階段已完成，但沒有有效外觀描述。",
                status: "完成但無文字",
                statusTone: "warning",
                detail: "請檢查 narration.object_descriptions。",
                completed: true,
            },
        );
    }
    else if (engineBusy)
    {
        setLayerState(
            "object",
            {
                text: objectText
                    || "已列出物件，正在等待上一個 VLM 工作結束。",
                status: "等待上一個 VLM 推理完成",
                statusTone: "running",
                detail: "engine_busy=是｜qwen_pending=否",
            },
        );
    }
    else
    {
        setLayerState(
            "object",
            {
                text: objectText
                    || "場景已穩定，等待排入物件外觀分析。",
                status: "等待推理排程",
                statusTone: "waiting",
                detail: (
                    `object_done=${booleanText(objectDone)}｜`
                    + `qwen_pending=${booleanText(qwenPending)}`
                ),
            },
        );
    }

    if (environmentText && environmentDone)
    {
        setLayerState(
            "environment",
            {
                text: environmentText,
                status: "整體場景描述已完成",
                statusTone: "done",
                detail: (
                    `environment_done=${booleanText(environmentDone)}｜`
                    + `general_pending=${booleanText(generalPending)}｜`
                    + `engine_busy=${booleanText(engineBusy)}`
                ),
                completed: true,
            },
        );
    }
    else if (!purpleAllowed)
    {
        setLayerState(
            "environment",
            {
                text: "目前設定未啟用紫色整體場景描述。",
                status: "功能未啟用",
                statusTone: "idle",
                detail: "purple_allowed=否",
            },
        );
    }
    else if (qwenError.startsWith("general_idle_"))
    {
        setLayerState(
            "environment",
            {
                text: qwenError,
                status: "場景描述失敗",
                statusTone: "error",
                detail: `general_pending=${booleanText(generalPending)}`,
            },
        );
    }
    else if (!objectDone)
    {
        setLayerState(
            "environment",
            {
                text: qwenStreaming
                    ? "同一次 VLM 正在逐筆回傳物件外觀，場景摘要會在最後顯示。"
                    : "整體場景摘要會在物件外觀分析完成後顯示。",
                status: qwenStreaming
                    ? "等待同一次 VLM 的場景摘要"
                    : "等待綠色階段",
                statusTone: qwenStreaming ? "running" : "waiting",
                detail: (
                    `object_done=${booleanText(objectDone)}｜`
                    + `已完成 ${qwenReceivedCount}/${Math.max(qwenExpectedCount, stableCount)} 個`
                ),
            },
        );
    }
    else if (generalPending || generalStarted)
    {
        setLayerState(
            "environment",
            {
                text: "完整畫面已送交 VLM，正在等待整體場景描述回覆。",
                status: "等待 VLM 回覆",
                statusTone: "running",
                detail: (
                    `general_pending=${booleanText(generalPending)}｜`
                    + `engine_busy=${booleanText(engineBusy)}`
                ),
            },
        );
    }
    else if (greenPurpleGap > 0)
    {
        setLayerState(
            "environment",
            {
                text: "綠色結果已完成，依展示設定等待後再顯示紫色結果。",
                status: `等待顯示間隔 ${greenPurpleGap.toFixed(1)} 秒`,
                statusTone: "running",
                detail: `green_purple_gap=${greenPurpleGap.toFixed(1)} 秒`,
            },
        );
    }
    else if (environmentDone)
    {
        setLayerState(
            "environment",
            {
                text: "整體場景階段已完成，但目前沒有可顯示的有效描述。",
                status: "完成但無文字",
                statusTone: "warning",
                detail: "請檢查 idle／summary line 與 recent_general_idle。",
                completed: true,
            },
        );
    }
    else if (engineBusy)
    {
        setLayerState(
            "environment",
            {
                text: "本機 VLM 引擎正處理其他工作，紫色階段等待模型空閒。",
                status: "等待模型空閒",
                statusTone: "running",
                detail: "engine_busy=是｜general_pending=否",
            },
        );
    }
    else
    {
        setLayerState(
            "environment",
            {
                text: "綠色階段已完成，等待排入整體場景 VLM 推理。",
                status: "等待推理排程",
                statusTone: "waiting",
                detail: (
                    `environment_done=${booleanText(environmentDone)}｜`
                    + `general_pending=${booleanText(generalPending)}`
                ),
            },
        );
    }
}

function createTextElement(className, textValue)
{
    const element = document.createElement("div");
    element.className = className;
    element.textContent = textValue;
    return element;
}

function cleanObjectCardText(textValue, label)
{
    let text = ensureSentence(textValue);

    if (!text)
    {
        return "";
    }

    const escapedLabel = String(label || "")
        .replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

    if (escapedLabel)
    {
        text = text.replace(
            new RegExp(`^(?:${escapedLabel})\\s*[：:]\\s*`),
            "",
        );
    }

    return text.trim();
}

function currentObjectSceneKey(payload)
{
    return String(
        sceneValue(
            payload,
            "scene_revision",
            lastSceneRevision ?? "scene",
        ),
    );
}

function resetObjectPlayback(sceneKey = null)
{
    if (objectPlaybackAdvanceTimer !== null)
    {
        window.clearTimeout(objectPlaybackAdvanceTimer);
        objectPlaybackAdvanceTimer = null;
    }

    for (const state of objectTypingState.values())
    {
        if (state?.timer)
        {
            window.clearInterval(state.timer);
        }
    }

    objectPlaybackSceneKey = sceneKey;
    objectPlaybackActiveKey = null;

    objectPlaybackRows.clear();
    objectPlaybackQueue.splice(0);
    objectPlaybackCompletedKeys.splice(0);
    objectTypingState.clear();
    seenObjectCardKeys.clear();
}

function actualObjectRowsForPlayback(payload)
{
    const labels = objectLabelById(payload);
    const rows = [];

    for (const [index, row] of getObjectDescriptionRows(payload).entries())
    {
        const rawText = entryText(row);

        if (!rawText)
        {
            continue;
        }

        const numericId = Number(row?.object_id);
        const objectId = Number.isFinite(numericId)
            ? numericId
            : `free-${index}`;

        const label = (
            Number.isFinite(numericId)
                ? labels.get(numericId)
                : ""
        ) || `物件 ${index + 1}`;

        rows.push({
            objectId,
            label,
            text: cleanObjectCardText(rawText, label),
        });
    }

    return rows.filter((row) => row.text);
}

function objectPlaybackBusy()
{
    return Boolean(
        objectPlaybackActiveKey
        || objectPlaybackQueue.length > 0,
    );
}

function updateObjectCardDom(objectKey)
{
    const state = objectTypingState.get(objectKey);

    if (!state)
    {
        return;
    }

    for (const card of document.querySelectorAll(".vlm-object-card"))
    {
        if (card.dataset.objectKey !== objectKey)
        {
            continue;
        }

        const output = card.querySelector(".vlm-combined-output");
        const badge = card.querySelector(".vlm-combined-badge");

        if (output)
        {
            output.textContent = state.visible;
            output.classList.toggle("typing", state.typing);
        }

        if (badge)
        {
            badge.textContent = state.typing ? "思考中" : "完成";
            badge.className = (
                `vlm-combined-badge ${state.typing ? "running" : "done"}`
            );
            card.dataset.status = state.typing ? "running" : "done";
        }
    }
}

function scheduleNextObjectCard()
{
    if (objectPlaybackAdvanceTimer !== null)
    {
        window.clearTimeout(objectPlaybackAdvanceTimer);
    }

    objectPlaybackAdvanceTimer = window.setTimeout(
        () =>
        {
            objectPlaybackAdvanceTimer = null;
            startNextObjectCard();
            renderUnderstanding();
        },
        OBJECT_CARD_GAP_MS,
    );
}

function startNextObjectCard()
{
    if (
        objectPlaybackActiveKey !== null
        || objectPlaybackQueue.length === 0
    )
    {
        return;
    }

    const objectKey = objectPlaybackQueue.shift();
    const row = objectPlaybackRows.get(objectKey);

    if (!row)
    {
        startNextObjectCard();
        return;
    }

    objectPlaybackActiveKey = objectKey;

    const characters = Array.from(row.text);
    const state = {
        target: row.text,
        visible: "",
        typing: true,
        timer: null,
        index: 0,
    };

    state.timer = window.setInterval(
        () =>
        {
            const current = objectTypingState.get(objectKey);

            if (
                !current
                || objectPlaybackActiveKey !== objectKey
            )
            {
                window.clearInterval(state.timer);
                return;
            }

            current.index += 1;
            current.visible = characters
                .slice(0, current.index)
                .join("");

            if (current.index >= characters.length)
            {
                current.visible = row.text;
                current.typing = false;

                window.clearInterval(current.timer);
                current.timer = null;

                if (!objectPlaybackCompletedKeys.includes(objectKey))
                {
                    objectPlaybackCompletedKeys.push(objectKey);
                }

                objectPlaybackActiveKey = null;
                updateObjectCardDom(objectKey);
                scheduleNextObjectCard();
                return;
            }

            updateObjectCardDom(objectKey);
        },
        OBJECT_TYPE_INTERVAL_MS,
    );

    objectTypingState.set(objectKey, state);
}

function syncObjectPlayback(payload)
{
    if (!payload)
    {
        return;
    }

    const sceneKey = currentObjectSceneKey(payload);

    if (objectPlaybackSceneKey !== sceneKey)
    {
        resetObjectPlayback(sceneKey);
    }

    for (const row of actualObjectRowsForPlayback(payload))
    {
        const objectKey = `${sceneKey}:${row.objectId}`;
        objectPlaybackRows.set(objectKey, row);

        const alreadyKnown = (
            objectPlaybackActiveKey === objectKey
            || objectPlaybackCompletedKeys.includes(objectKey)
            || objectPlaybackQueue.includes(objectKey)
        );

        if (!alreadyKnown)
        {
            objectPlaybackQueue.push(objectKey);
        }
    }

    startNextObjectCard();
}

function currentLayerSceneKey()
{
    return String(
        sceneValue(
            latestVlmPayload,
            "scene_revision",
            lastSceneRevision ?? "scene",
        ),
    );
}

function resetLayerTypewriters()
{
    for (const state of layerTypingState.values())
    {
        if (state?.timer)
        {
            window.clearInterval(state.timer);
        }
    }

    layerTypingState.clear();
}

function updateStandardLayerDom(layerKey)
{
    const state = layerTypingState.get(layerKey);

    if (!state)
    {
        return;
    }

    for (const card of document.querySelectorAll(".vlm-standard-layer-card"))
    {
        if (card.dataset.layerKey !== layerKey)
        {
            continue;
        }

        const output = card.querySelector(".vlm-combined-output");

        if (!output)
        {
            continue;
        }

        output.textContent = state.visible;
        output.classList.toggle("typing", state.typing);
    }
}

function ensureLayerTypewriter(kind, targetText)
{
    const target = String(targetText || "");
    const layerKey = `${currentLayerSceneKey()}:${kind}`;
    let state = layerTypingState.get(layerKey);

    if (state?.target === target)
    {
        return {
            layerKey,
            state,
        };
    }

    if (state?.timer)
    {
        window.clearInterval(state.timer);
    }

    if (!target)
    {
        state = {
            target,
            visible: "",
            typing: false,
            timer: null,
            index: 0,
        };
        layerTypingState.set(layerKey, state);

        return {
            layerKey,
            state,
        };
    }

    const characters = Array.from(target);

    state = {
        target,
        visible: "",
        typing: true,
        timer: null,
        index: 0,
    };

    state.timer = window.setInterval(
        () =>
        {
            const current = layerTypingState.get(layerKey);

            if (!current || current.target !== target)
            {
                window.clearInterval(state.timer);
                return;
            }

            current.index += 1;
            current.visible = characters
                .slice(0, current.index)
                .join("");

            if (current.index >= characters.length)
            {
                current.visible = target;
                current.typing = false;

                window.clearInterval(current.timer);
                current.timer = null;
            }

            updateStandardLayerDom(layerKey);
        },
        LAYER_TYPE_INTERVAL_MS,
    );

    layerTypingState.set(layerKey, state);

    return {
        layerKey,
        state,
    };
}

function createStandardLayerCard(kind, item)
{
    const typing = ensureLayerTypewriter(
        kind,
        item.text,
    );

    const card = document.createElement("section");
    card.className = `vlm-combined-card ${kind} vlm-standard-layer-card`;
    card.dataset.stage = kind;
    card.dataset.status = item.statusTone;
    card.dataset.layerKey = typing.layerKey;

    const header = document.createElement("div");
    header.className = "vlm-combined-header";

    const title = createTextElement(
        "vlm-combined-title",
        LAYER_META[kind].title,
    );

    const badge = createTextElement(
        `vlm-combined-badge ${item.statusTone}`,
        badgeText(kind, item),
    );

    header.append(title, badge);

    const output = createTextElement(
        `vlm-combined-output${typing.state.typing ? " typing" : ""}`,
        typing.state.visible,
    );

    const diagnostic = createTextElement(
        "vlm-combined-diagnostic",
        item.detail || "尚無診斷資料。",
    );

    card.append(header, output, diagnostic);
    return card;
}

function createOneObjectCard(objectKey)
{
    const row = objectPlaybackRows.get(objectKey);
    const state = objectTypingState.get(objectKey);

    if (!row || !state)
    {
        return null;
    }

    const wasSeen = seenObjectCardKeys.has(objectKey);

    const card = document.createElement("section");
    card.className = "vlm-combined-card object vlm-object-card";
    card.dataset.stage = "object";
    card.dataset.objectKey = objectKey;
    card.dataset.status = state.typing ? "running" : "done";

    if (!wasSeen)
    {
        card.classList.add("vlm-card-enter");
        seenObjectCardKeys.add(objectKey);
    }

    const header = document.createElement("div");
    header.className = "vlm-combined-header";

    const title = createTextElement(
        "vlm-combined-title",
        row.label,
    );

    const badge = createTextElement(
        `vlm-combined-badge ${state.typing ? "running" : "done"}`,
        state.typing ? "思考中" : "完成",
    );

    const output = createTextElement(
        `vlm-combined-output${state.typing ? " typing" : ""}`,
        state.visible,
    );

    header.append(title, badge);
    card.append(header, output);
    return card;
}

function createObjectCards(payload, fallbackItem)
{
    syncObjectPlayback(payload);

    const fragment = document.createDocumentFragment();
    const visibleKeys = [...objectPlaybackCompletedKeys];

    if (
        objectPlaybackActiveKey
        && !visibleKeys.includes(objectPlaybackActiveKey)
    )
    {
        visibleKeys.push(objectPlaybackActiveKey);
    }

    for (const objectKey of visibleKeys)
    {
        const card = createOneObjectCard(objectKey);

        if (card)
        {
            fragment.append(card);
        }
    }

    if (visibleKeys.length === 0)
    {
        const waitingItem = {
            ...fallbackItem,
            text: (
                actualObjectDescriptionCount(payload) > 0
                    ? "準備依序顯示物件外觀。"
                    : "等待第一筆物件外觀描述。"
            ),
            status: "等待物件結果",
            statusTone: "running",
        };

        fragment.append(
            createStandardLayerCard("object", waitingItem),
        );
    }

    return fragment;
}

function environmentDisplayItem(item)
{
    if (!objectPlaybackBusy())
    {
        return item;
    }

    return {
        ...item,
        text: "等待物件外觀逐項顯示完成。",
        status: "等待物件卡片",
        statusTone: "running",
    };
}

function renderUnderstanding()
{
    const feed = document.getElementById("understandingFeed");

    if (!feed)
    {
        return;
    }

    feed.replaceChildren();
    feed.classList.remove("vlm-status-test-feed");
    feed.classList.add("vlm-combined-feed");

    for (const kind of LAYER_ORDER)
    {
        const item = layerState[kind] ?? createInitialLayer(kind);

        if (kind === "object")
        {
            feed.append(
                createObjectCards(latestVlmPayload, item),
            );
            continue;
        }

        feed.append(
            createStandardLayerCard(
                kind,
                kind === "environment"
                    ? environmentDisplayItem(item)
                    : item,
            ),
        );
    }
}

function resetLayersForNewScene()
{
    const now = Date.now();
    sceneRevisionStartedAt = now;
    latestVlmPayload = null;
    resetObjectPlayback(null);
    resetLayerTypewriters();

    for (const kind of LAYER_ORDER)
    {
        layerState[kind] = createInitialLayer(kind);
        layerCompletedAt[kind] = null;
        layerActivatedAt[kind] = null;
        layerStatusStartedAt[kind] = now;
    }

    activateLayer("yolo");
}

function renderVlmError(error)
{
    const message = normalizeText(error?.message ?? error)
        || "場景理解服務暫時無法連線。";

    for (const kind of LAYER_ORDER)
    {
        setLayerState(
            kind,
            {
                text: message,
                status: "服務連線失敗",
                statusTone: "error",
                detail: `連續失敗 ${vlmFailureCount} 次｜${VLM_STATE_URL}`,
            },
        );
    }

    renderUnderstanding();
}

function renderVlmDisabled()
{
    const feed = document.getElementById("understandingFeed");

    if (!feed)
    {
        return;
    }

    const notice = document.createElement("div");
    notice.className = "understanding-empty";
    notice.textContent = "VLM 物件特性描述暫時關閉（測驗用）";

    feed.classList.remove("vlm-combined-feed");
    feed.replaceChildren(notice);

    const clearButton = document.getElementById(
        "clearUnderstandingButton",
    );
    if (clearButton)
    {
        clearButton.disabled = true;
    }
}

async function pollVlmState()
{
    if (!isVlmPollingAllowed())
    {
        return;
    }

    if (vlmRequestRunning)
    {
        return;
    }

    vlmRequestRunning = true;

    try
    {
        const response = await fetch(
            VLM_STATE_URL,
            {
                method: "GET",
                headers: {
                    Accept: "application/json",
                },
                cache: "no-store",
            },
        );

        if (!response.ok)
        {
            throw new Error(`VLM state HTTP ${response.status}`);
        }

        const payload = await response.json();

        if (payload?.ok !== true)
        {
            throw new Error(
                normalizeText(payload?.message)
                || "VLM state returned ok=false",
            );
        }

        if (payload?.narrator_enabled === false)
        {
            vlmFailureCount = 0;
            renderVlmDisabled();
            stopVlmPolling();
            return;
        }

        vlmFailureCount = 0;

        const sceneRevision = (
            payload?.scene_revision
            ?? payload?.scene?.scene_revision
            ?? null
        );

        if (reobserveGateActive)
        {
            const revisionChanged = (
                sceneRevision !== null
                && sceneRevision !== undefined
                && sceneRevision !== reobserveBaseRevision
            );

            if (!revisionChanged)
            {
                renderUnderstanding();
                return;
            }

            reobserveGateActive = false;
            reobserveBaseRevision = null;
            resetLayersForNewScene();
            lastSceneRevision = sceneRevision;
            updateAllLayerStates(payload);
            renderUnderstanding();
            return;
        }

        if (
            lastSceneRevision === null
            || sceneRevision !== lastSceneRevision
        )
        {
            resetLayersForNewScene();
        }

        lastSceneRevision = sceneRevision;
        updateAllLayerStates(payload);
        renderUnderstanding();
    }
    catch (error)
    {
        vlmFailureCount += 1;

        console.debug(
            "[vlm-combined] state unavailable:",
            error,
        );

        if (vlmFailureCount >= VLM_ERROR_DISPLAY_THRESHOLD)
        {
            renderVlmError(error);
        }
    }
    finally
    {
        vlmRequestRunning = false;
    }
}

async function clearUnderstanding()
{
    reobserveBaseRevision = lastSceneRevision;
    reobserveGateActive = true;
    resetLayersForNewScene();

    for (const kind of LAYER_ORDER)
    {
        layerState[kind].status = "要求重新觀察";
        layerState[kind].statusTone = "running";
        layerState[kind].detail = "等待後端建立新的場景版本。";
        layerStatusStartedAt[kind] = Date.now();
    }

    renderUnderstanding();

    try
    {
        await requestVlmReobserve("status_test_clear_and_reobserve");
    }
    catch (error)
    {
        reobserveGateActive = false;
        reobserveBaseRevision = null;
        console.warn("[vlm-combined] reobserve failed:", error);

        for (const kind of LAYER_ORDER)
        {
            layerState[kind].status = "重新觀察失敗";
            layerState[kind].statusTone = "error";
            layerState[kind].detail = normalizeText(error?.message ?? error);
            layerStatusStartedAt[kind] = Date.now();
        }

        renderUnderstanding();
    }

    window.setTimeout(() => void pollVlmState(), 200);
}

async function requestVlmReobserve(
    reason = "main_ui_request",
)
{
    const response = await fetch(
        VLM_REOBSERVE_URL,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
            },
            body: JSON.stringify({reason}),
        },
    );

    if (!response.ok)
    {
        throw new Error(
            `VLM reobserve HTTP ${response.status}`,
        );
    }

    return response.json();
}

function startVlmPolling()
{
    if (!isVlmPollingAllowed())
    {
        stopVlmPolling();
        return;
    }

    if (vlmPollingTimer !== null)
    {
        return;
    }

    void pollVlmState();

    vlmPollingTimer = window.setInterval(
        () => void pollVlmState(),
        VLM_POLL_INTERVAL_MS,
    );
}

function stopVlmPolling()
{
    if (vlmPollingTimer === null)
    {
        return;
    }

    window.clearInterval(vlmPollingTimer);
    vlmPollingTimer = null;
}

function initVlmUI()
{
    if (vlmInitialized)
    {
        return;
    }

    vlmInitialized = true;

    const initialize = () =>
    {
        const clearButton = document.getElementById(
            "clearUnderstandingButton",
        );

        clearButton?.addEventListener(
            "click",
            clearUnderstanding,
        );

        installStatusTestStyles();
        resetLayersForNewScene();
        renderUnderstanding();
        window.setTimeout(
            startVlmPolling,
            120,
        );
    };

    if (document.readyState === "loading")
    {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true},
        );
    }
    else
    {
        initialize();
    }
}

window.addEventListener(
    "pagehide",
    () =>
    {
        stopVlmPolling();
    },
);

window.startVlmPolling = startVlmPolling;
window.stopVlmPolling = stopVlmPolling;
window.pollVlmState = pollVlmState;

export {
    initVlmUI,
    startVlmPolling,
    stopVlmPolling,
    pollVlmState,
    clearUnderstanding,
    requestVlmReobserve,
    buildYoloText,
    buildPlacementText,
    buildObjectText,
    buildEnvironmentText,
};

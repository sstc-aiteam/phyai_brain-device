"""
AI Agent 核心邏輯。
"""

import json
import re
import time
from collections import Counter
from pathlib import Path
from textwrap import dedent

from agent.tools import get_tool, get_tools_for_prompt
from services import llm_service
from utils.object_identity import assign_request_object_ids

RULES_PATH = Path(__file__).with_name("rules.json")

#-0.0375  -0.10 13cm
#-0.4082 -0.6932 29
PICK_POINT_LIMITS = {
    "x": {
        "minimum": -0.2118,
        "maximum": 0.1375,
    },
    "y": {
        "minimum": -0.6936,
        "maximum": -0.4474,
    },
    "z": {
        "minimum": -0.29,
        "maximum": 0.00,
    },
}

PICK_POINT_TOOLS = {
    "place_object_in_trash_can",
    "place_object_in_top_cabinet",
    "place_object_in_second_drawer",
}

PLANNING_OBJECT_ID_PARAMETER = {
    "type": "string",
    "description": (
        "要夾取的物件 ID，必須原樣選自本次 objects 中的 object_id。"
    ),
}

PLANNING_PICK_TOOL_DESCRIPTIONS = {
    "place_object_in_trash_can": (
        "將指定 object_id 的物件夾取並放入垃圾桶。"
    ),
    "place_object_in_top_cabinet": (
        "將指定 object_id 的物件夾取並放入上方櫃子。"
    ),
    "place_object_in_second_drawer": (
        "將指定 object_id 的物件夾取並放入第二抽屜。"
    ),
}

PROMPT = """
你是一個實體機器人的 AI Agent。

你的工作：
1. 理解使用者的對話。
2. 若使用者詢問機器手臂狀態或辨識物件，依據提供的資訊回答，不可猜測；詢問場景物件時，
   還要依 rules.json 說出每種可見物件的推薦歸位位置與理由。
3. 若使用者要求機器人執行任務，必須從工具白名單中選擇適合的 API。
4. 一個使用者需求可以拆解成多個 API，並依實際執行順序放入 tool_calls。
5. tool_calls 中的工具會由上到下依序執行，因此順序必須正確。
6. 不可使用不存在的 API，也不可自行定義 API。
7. 不可把可以一次完成的單一步驟重複加入 tool_calls。
8. 若不需要執行任何動作，tool_calls 必須回傳空陣列 []。
   反之，只要 answer 表示「我會、現在會、準備、開始、幫你」執行實體動作，tool_calls 就不可為空。
   絕對不可口頭承諾執行，卻回傳空的 tool_calls。
9. 若必要參數不完整、沒有合適工具或機器手臂無法連線，
   tool_calls 必須回傳空陣列 []，並在 answer 中說明原因。
10. 每一個 tool_call 必須包含 function_name 與 arguments。
11. function_name 只能使用工具白名單中的名稱。
12. arguments 只能包含對應工具允許的參數，不可猜測參數。
12.1 decision_basis 必須提供簡潔、可供使用者核對的判斷依據；說明物件、目的地、場域
     處理原則，以及物件狀態是否有視覺證據。不要輸出冗長的內部逐步思考。
12.2 answer 與 decision_basis 必須使用自然的臺灣繁體中文，不可使用簡體中文；不可提及
     rules.json、object_id、API 或函式名稱，也不可描述系統曾經修正、重試或補齊模型規劃。
12.3 描述物件數量時必須使用正確量詞，例如一瓶酒精、一條毛巾、一瓶生理食鹽水、
     一片口罩；不可把所有物件一律稱為「一件」。
13. 「清理所有東西」、「整理全部物件」、「把東西全部收好」是立即執行的批次任務，
    不是要求稍後檢查環境；不可只回答將要檢查而回傳空的 tool_calls。
14. 批次整理時，必須逐一處理目前 objects 陣列中的每一個物件。每個物件各建立一個
    放置工具呼叫，arguments 只填入該物件的 object_id，並依 rules.json 的 destination
    與 suggested_action 決定目的地。
15. 同類別出現多個實體時也必須逐一處理；每個物件各自使用自己的 object_id，這不算重複步驟。
16. 批次整理時，若 detection_available 為 false、objects 為空、必要座標缺失、手臂未連線，
    或任何必要資訊不合法，tool_calls 必須為 []，並直接說明目前無法整理的具體原因，
    不可回答「我會先檢查」或假裝稍後會取得資料。

任務拆解範例：

使用者要求「打開櫃門，把酒精放進櫃子，再關閉櫃門」：

tool_calls 應依序為：
1. open_top_cabinet
2. place_object_in_top_cabinet
3. close_top_cabinet

使用者只要求「打開櫃門」：

tool_calls 只有：
1. open_top_cabinet

使用者只是在詢問目前看到什麼：

回答辨識到的物件，並依 rules.json 說明推薦歸位位置與理由；tool_calls 必須為空陣列 []。

使用者要求「幫我收口罩」，objects 中存在 disposable_mask，且使用者沒有指定目的地：

必須依 rules.json 將口罩送往 trash_can，完整產生：
1. open_trash_can
2. place_object_in_trash_can，arguments 使用該口罩的 object_id，例如 disposable_mask_1
3. close_trash_can
不可只在 answer 說會把口罩放到垃圾桶，卻回傳空的 tool_calls。

使用者要求「幫我清理所有東西 所有容器都打開了 不用關閉」，而 objects 中有酒精瓶、毛巾與生理食鹽水：

tool_calls 必須依 objects 逐一建立：
1. 酒精瓶依 rules.json 放入櫃子，使用 place_object_in_top_cabinet。
2. 毛巾依 rules.json 放入垃圾桶，使用 place_object_in_trash_can。
3. 生理食鹽水依 rules.json 放入抽屜，使用 place_object_in_second_drawer。
每個放置步驟都必須帶入該物件自己的 object_id，不可只回答將要檢查環境。

使用者要求「關閉櫃門」：

tool_calls 只有：
1. close_top_cabinet

使用者要求「打開抽屜」：

tool_calls 只有：
1. open_second_drawer

使用者要求「關閉抽屜」：

tool_calls 只有：
1. close_second_drawer

使用者要求「打開垃圾桶」：

tool_calls 只有：
1. open_trash_can

使用者要求「關閉垃圾桶」：

tool_calls 只有：
1. close_trash_can

如果使用者命令只是開或關門、抽屜或垃圾桶，這些動作不需要任何物件辨識資訊，仍然可以回傳對應的 open_* 或 close_* 工具。

物件選取規則：
- place_object_* 的 arguments 只能包含 object_id。
- object_id 必須原樣選自本次 objects，不可自行產生或猜測。
- 不可輸出 point_xyz 或 yaw_deg；程式會在模型規劃完成後，依 object_id 安全地填入視覺座標。

回傳規則：
- answer：以機器人的第一人稱簡潔回答，不需要提到 API 名稱。
- decision_basis：簡潔說明每個被處理物件的分類與目的地理由。若缺乏未拆封、已使用或污損證據，
  必須明確說「目前無法確認」，並使用「若……則……」的條件句，不可捏造狀態。
- tool_calls：依實際執行順序排列的工具陣列。
- 不需要執行動作時，tool_calls 必須為 []。
"""


class AgentError(RuntimeError):
    """
    Agent 執行錯誤。
    """


def _is_observation_only_request(user_text):
    """辨識只要求描述目前場景、沒有要求操作硬體的命令。"""
    if not isinstance(user_text, str):
        return False

    normalized = re.sub(r"[\s，,。.!！?？、]", "", user_text.lower())
    observation_phrases = (
        "看到什麼",
        "看到了什麼",
        "看到什麼物件",
        "有看到什麼",
        "有看到什麼物件",
        "看見什麼",
        "看有什麼",
        "場景有什麼",
        "畫面有什麼",
        "畫面中有什麼",
        "鏡頭有什麼",
        "鏡頭裡有什麼",
        "視野有什麼",
        "現在有什麼",
        "目前有什麼",
        "有哪些東西",
        "有哪些物件",
        "描述場景",
        "描述一下",
    )
    manipulation_terms = (
        "收",
        "整理",
        "清理",
        "處理",
        "歸位",
        "放",
        "丟",
        "拿起",
        "夾取",
        "打開",
        "開啟",
        "關閉",
        "關上",
        "移動手臂",
    )
    return (
        any(phrase in normalized for phrase in observation_phrases)
        and not any(term in normalized for term in manipulation_terms)
    )


def _chinese_count(number):
    """將常見的小數量轉成自然中文數字。"""
    numerals = {
        1: "一", 2: "兩", 3: "三", 4: "四", 5: "五",
        6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
    }
    return numerals.get(number, str(number))


def _build_observation_answer(detected_objects, recommendation_rules):
    """依本次偵測資料與 rules.json 產生場景及歸位建議。"""
    if not isinstance(detected_objects, dict):
        return "我目前無法取得場景辨識結果。"
    if detected_objects.get("detection_available") is not True:
        error = detected_objects.get("detection_error")
        if isinstance(error, str) and error.strip():
            return f"我目前看不到場景，原因：{error.strip()}。"
        return "我目前看不到場景。"

    objects = detected_objects.get("objects")
    if not isinstance(objects, list) or not objects:
        return "我目前沒有看到可辨識的物件。"

    object_rules = (
        recommendation_rules.get("objects", {})
        if isinstance(recommendation_rules, dict)
        else {}
    )
    identities = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        class_name = obj.get("class_name") or "未知物件"
        rule = object_rules.get(class_name)
        if isinstance(rule, dict):
            label = rule.get("count_label") or rule.get("object_label")
            unit = rule.get("count_unit") or "個"
        else:
            label = class_name
            unit = "個"
        destination_label = None
        reason = None
        if isinstance(rule, dict):
            destination_name = rule.get("destination")
            destinations = recommendation_rules.get("destinations", {})
            destination = (
                destinations.get(destination_name)
                if isinstance(destinations, dict)
                else None
            )
            if isinstance(destination, dict):
                destination_label = destination.get("label_zh")
            user_reason = rule.get("user_reason")
            if isinstance(user_reason, str) and user_reason.strip():
                reason = user_reason.strip().rstrip("。")

        identity = (
            str(label or class_name),
            str(unit),
            str(destination_label) if destination_label else None,
            reason,
        )
        identities.append(identity)

    if not identities:
        return "我目前沒有看到可辨識的物件。"

    counts = Counter(identities)
    descriptions = []
    for (label, unit, destination_label, reason), count in counts.items():
        description = f"{_chinese_count(count)}{unit}{label}"
        if destination_label:
            description += f"，建議歸位到{destination_label}"
            if reason:
                description += f"，因為{reason}"
        else:
            description += "；目前沒有對應的歸位規則"
        descriptions.append(description + "。")

    return "我目前看到：" + "".join(descriptions)


def _build_single_direct_action_plan(user_text):
    """依文字順序建立無物件、無參數的直接硬體操作計畫。"""
    if not isinstance(user_text, str):
        return None

    normalized = re.sub(r"[\s，,。.!！?？、]", "", user_text.lower())
    if any(
        term in normalized
        for term in (
            "放入", "放進", "放到", "丟入", "丟進", "丟到",
            "收進", "收好", "收拾", "清理", "整理", "處理", "夾取", "拿起",
        )
    ):
        return None

    definitions = (
        ("move_arm_default", "讓機器手臂回到預設位置", (
            "回到預設位置", "回預設位置", "回到初始位置", "回初始位置",
            "回到原位置", "回原位置", "回到原位", "回原位",
            "回到原始位置", "回原始位置", "手臂復位", "機器人復位",
        )),
        ("open_top_cabinet", "打開櫃門", (
            "開櫃", "打開櫃門", "櫃門打開", "打開上方櫃", "開啟櫃門",
        )),
        ("close_top_cabinet", "關閉櫃門", (
            "關櫃", "關閉櫃門", "關上櫃門", "櫃門關起來", "把櫃門關起來",
        )),
        ("open_second_drawer", "打開抽屜", (
            "打開抽屜", "開抽屜", "開啟抽屜", "抽屜打開",
        )),
        ("close_second_drawer", "關閉抽屜", (
            "關閉抽屜", "關抽屜", "關上抽屜", "抽屜關起來",
        )),
        ("open_trash_can", "打開垃圾桶", (
            "打開垃圾桶", "開垃圾桶", "開啟垃圾桶", "垃圾桶打開",
        )),
        ("close_trash_can", "關閉垃圾桶", (
            "關閉垃圾桶", "關垃圾桶", "關上垃圾桶", "垃圾桶關起來",
        )),
    )

    matches = []
    for function_name, action_text, phrases in definitions:
        positions = [normalized.find(phrase) for phrase in phrases if phrase in normalized]
        if positions:
            matches.append((min(positions), function_name, action_text))

    # 「開抽屜跟垃圾桶」只寫一次動詞，仍應對兩個具名容器操作。
    has_open_cue = any(term in normalized for term in ("開", "打開", "開啟"))
    has_close_cue = any(term in normalized for term in ("關", "關閉", "關上", "關起來"))
    if has_open_cue != has_close_cue:
        operation = "open" if has_open_cue else "close"
        shared_targets = (
            ("top_cabinet", ("櫃門", "櫃子", "櫃")),
            ("second_drawer", ("抽屜",)),
            ("trash_can", ("垃圾桶",)),
        )
        action_lookup = {
            "open_top_cabinet": "打開櫃門",
            "close_top_cabinet": "關閉櫃門",
            "open_second_drawer": "打開抽屜",
            "close_second_drawer": "關閉抽屜",
            "open_trash_can": "打開垃圾桶",
            "close_trash_can": "關閉垃圾桶",
        }
        matched_functions = {match[1] for match in matches}
        for target, nouns in shared_targets:
            positions = [normalized.find(noun) for noun in nouns if noun in normalized]
            function_name = f"{operation}_{target}"
            if positions and function_name not in matched_functions:
                matches.append(
                    (min(positions), function_name, action_lookup[function_name])
                )

    if not matches:
        return None

    matches.sort(key=lambda match: match[0])
    unique_matches = []
    seen_functions = set()
    for _, function_name, action_text in matches:
        if function_name in seen_functions:
            continue
        seen_functions.add(function_name)
        unique_matches.append((function_name, action_text))

    action_texts = [action_text for _, action_text in unique_matches]
    if len(action_texts) == 1:
        answer = f"我會{action_texts[0]}。"
    else:
        answer = "我會先" + action_texts[0] + "，再" + "，再".join(action_texts[1:]) + "。"

    return {
        "answer": answer,
        "tool_calls": [
            {"function_name": function_name, "arguments": {}}
            for function_name, _ in unique_matches
        ],
    }


def _requested_rule_destination_group(user_text):
    """辨識使用者以規則目的地描述的物件群組。"""
    if not isinstance(user_text, str):
        return None

    normalized = re.sub(r"[\s，,。.!！?？、]", "", user_text.lower())
    # 「丟垃圾」中的垃圾是物件群組；「丟到垃圾桶」中的垃圾桶則是
    # 使用者指定的目的地，不能因此把具名物件改成預設垃圾群組。
    text_without_trash_can = normalized.replace("垃圾桶", "")
    medical_terms = ("醫療物品", "醫療物件")
    drawer_group_patterns = (
        r"(?:要|會|應該|可以)?(?:進入|放入|放進|收進|收到)抽屜的",
        r"抽屜裡的(?:物品|物件|東西)",
    )
    trash_group_patterns = (
        r"(?:要|會|應該|可以)?(?:進入|進去|放入|放進|丟入|丟進|丟到)垃圾桶的",
        r"垃圾桶裡的(?:垃圾|廢棄物|物品|物件|東西)",
    )

    if (
        any(term in normalized for term in medical_terms)
        or any(re.search(pattern, normalized) for pattern in drawer_group_patterns)
    ):
        return "second_drawer"

    if any(
        term in normalized
        for term in (
            "所有垃圾", "所有廢棄物", "垃圾跟污損布料",
            "垃圾和污損布料", "髒布料", "髒毛巾", "髒抹布",
        )
    ):
        return "trash_can"

    if any(re.search(pattern, normalized) for pattern in trash_group_patterns):
        return "trash_can"

    if any(
        term in text_without_trash_can
        for term in ("垃圾", "廢棄物")
    ):
        return "trash_can"

    return None


def _build_rule_destination_group_plan(
    user_text,
    detected_objects,
    recommendation_rules,
):
    """對明確規則群組或無排除的全場整理建立完整工具清單。"""
    destination_name = _requested_rule_destination_group(user_text)
    if not _is_bulk_cleanup_request(user_text):
        return None

    normalized = re.sub(r"[\s，,。.!！?？、]", "", str(user_text).lower())
    if destination_name is None and any(
        term in normalized
        for term in (
            "除了", "除外", "只收", "只處理", "不要動",
            "別動", "留下", "保留",
        )
    ):
        # 有排除或選取條件時仍交給語言模型理解，不可擅自處理全部物件。
        return None

    if not isinstance(detected_objects, dict):
        return None
    objects = detected_objects.get("objects")
    if not isinstance(objects, list):
        return None
    if not isinstance(recommendation_rules, dict):
        return None

    object_rules = recommendation_rules.get("objects")
    destinations = recommendation_rules.get("destinations")
    if not isinstance(object_rules, dict) or not isinstance(destinations, dict):
        return None

    selected = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_rule = object_rules.get(obj.get("class_name"))
        if not isinstance(object_rule, dict):
            return None
        object_destination_name = object_rule.get("destination")
        if (
            destination_name is not None
            and object_destination_name != destination_name
        ):
            continue
        destination = destinations.get(object_destination_name)
        if not isinstance(destination, dict):
            return None
        function_name = destination.get("suggested_action")
        if function_name not in PICK_POINT_TOOLS:
            return None
        object_id = obj.get("object_id")
        if not isinstance(object_id, str) or not object_id.strip():
            return None
        selected.append(
            (obj, object_rule, object_id.strip(), function_name)
        )

    if not selected:
        destination = destinations.get(destination_name) or {}
        destination_label = destination.get("label_zh") or destination_name
        return {
            "tool_calls": [],
            "answer": f"我目前沒有看到需要放進{destination_label}的物件。",
        }

    labels = [
        str(rule.get("object_label") or obj.get("class_name") or "物件")
        for obj, rule, _object_id, _function_name in selected
    ]
    if len(labels) == 1:
        object_text = labels[0]
    else:
        object_text = "、".join(labels[:-1]) + "和" + labels[-1]

    if destination_name is None:
        public_answer = f"我會把{object_text}依照目前的收納規則分類處理。"
    else:
        destination = destinations.get(destination_name) or {}
        destination_label = destination.get("label_zh") or destination_name
        verb = "丟進" if destination_name == "trash_can" else "收進"
        public_answer = f"我會把{object_text}{verb}{destination_label}。"

    return {
        "tool_calls": [
            {
                "function_name": function_name,
                "arguments": {"object_id": object_id},
            }
            for _obj, _rule, object_id, function_name in selected
        ],
        "answer": public_answer,
    }


def _is_bulk_cleanup_request(user_text):
    """判斷使用者是否要求整理目前場景中的全部物件。"""
    normalized = re.sub(
        r"[\s，,。.!！?？、]",
        "",
        user_text.strip().lower(),
    )
    bulk_terms = ("全部", "清空")
    bulk_all_phrases = (
        "東西都",
        "物件都",
        "這些都",
        "桌上的都",
        "看到的都",
        "整個都",
        "清理椅子",
        "整理椅子",
        "椅子上的",
        "椅面上的",
    )
    category_bulk_terms = (
        "醫療物品",
        "醫療物件",
        "進入抽屜的",
        "放進抽屜的",
        "放入抽屜的",
        "收進抽屜的",
        "所有垃圾",
        "所有廢棄物",
        "垃圾跟污損布料",
        "垃圾和污損布料",
        "髒布料",
    )
    action_terms = (
        "清理",
        "整理",
        "收好",
        "收拾",
        "處理",
        "歸位",
        "收",
        "丟",
        "放",
        "收起來",
        "收一收",
        "收掉",
    )
    return (
        (
            any(term in normalized for term in bulk_terms)
            or any(phrase in normalized for phrase in bulk_all_phrases)
            or any(term in normalized for term in category_bulk_terms)
            or _requested_rule_destination_group(normalized) is not None
            or re.search(r"所有.*?(?:東西|物件)", normalized) is not None
        )
        and any(term in normalized for term in action_terms)
    )


def _check_bulk_cleanup_preconditions(
    user_text,
    robot_status,
    detected_objects,
):
    """在呼叫模型前，以確定性規則檢查批次整理的必要條件。"""
    if not _is_bulk_cleanup_request(user_text):
        return None

    if robot_status.get("status_available") is not True:
        message = robot_status.get("status_error")
        detail = message.strip() if isinstance(message, str) and message.strip() else "無法取得機器手臂狀態"
        return f"我目前感覺不到機器手臂，因此無法整理物件。原因：{detail}。"

    if robot_status.get("connected") is not True:
        return "我目前無法控制機器手臂，因此無法整理物件。"

    if detected_objects.get("detection_available") is not True:
        message = detected_objects.get("detection_error")
        detail = message.strip() if isinstance(message, str) and message.strip() else "無法取得物件辨識結果"
        return f"我目前看不到場景，因此無法整理物件。原因：{detail}。"

    objects = detected_objects.get("objects")
    if not isinstance(objects, list):
        return "物件辨識資料格式錯誤，因此目前無法整理物件。"

    if not objects:
        return "我目前看得到場景，但沒有看到任何可整理的物件。"

    return None


def _check_action_preconditions(robot_status, tool_calls):
    """任何硬體動作開始驗證前，都必須先確認手臂可控制。"""
    if not tool_calls:
        return None

    if not isinstance(robot_status, dict):
        return "我目前無法確認機器手臂狀態，因此不會執行任何動作。"

    if robot_status.get("status_available") is not True:
        message = robot_status.get("status_error")
        detail = (
            message.strip()
            if isinstance(message, str) and message.strip()
            else "無法取得機器手臂狀態"
        )
        return (
            "我目前感覺不到機器手臂，因此不會執行任何動作。"
            f"原因：{detail}。"
        )

    if robot_status.get("connected") is not True:
        return "我目前無法控制機器手臂，因此不會執行任何動作。"

    return None


def _load_recommendation_rules():
    """
    載入物件的推薦目的地與 API 規則。
    """
    try:
        with RULES_PATH.open("r", encoding="utf-8") as file:
            rules = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentError(f"無法載入推薦規則：{exc}") from exc

    if not isinstance(rules, dict):
        raise AgentError("rules.json 的最外層必須是 JSON object")

    if not isinstance(rules.get("destinations"), dict):
        raise AgentError("rules.json 缺少有效的 destinations")

    if not isinstance(rules.get("objects"), dict):
        raise AgentError("rules.json 缺少有效的 objects")

    return rules


def _with_request_object_ids(detected_objects):
    """為 Agent 本次偵測快照建立不重複、不可跨 request 使用的物件 ID。"""
    normalized = detected_objects.copy()
    objects = detected_objects.get("objects")

    if not isinstance(objects, list):
        return normalized

    normalized["objects"] = assign_request_object_ids(objects)
    return normalized


def _get_tools_for_planning():
    """產生 LLM 規劃工具；放置工具只讓模型選 object_id。"""
    tools = get_tools_for_prompt()

    for tool in tools:
        if tool.get("api_function") not in PICK_POINT_TOOLS:
            continue
        tool["parameters"] = {
            "object_id": PLANNING_OBJECT_ID_PARAMETER.copy(),
        }
        tool["required"] = ["object_id"]
        tool["description"] = (
            PLANNING_PICK_TOOL_DESCRIPTIONS[tool["api_function"]]
            + "規劃階段只選擇 objects 中的 object_id；"
            "不要輸出 point_xyz 或 yaw_deg。"
        )

    return tools


def build_prompt(user_text, robot_status, detected_objects):
    """
    根據使用者自然語言、工具白名單、機器人狀態
    與目前物件辨識結果，選擇適合的 API 工具。
    """
    if not isinstance(user_text, str):
        raise ValueError("user_text 必須是字串")

    normalized_text = user_text.strip()
    if not normalized_text:
        raise ValueError("沒有對話")

    if not isinstance(robot_status, dict):
        raise ValueError("robot_status 必須是 dict")

    if not isinstance(detected_objects, dict):
        raise ValueError("detected_objects 必須是 dict")

    detected_objects = _with_request_object_ids(detected_objects)

    precondition_error = _check_bulk_cleanup_preconditions(
        user_text=normalized_text,
        robot_status=robot_status,
        detected_objects=detected_objects,
    )

    if precondition_error is not None:
        return {
            "user_text": normalized_text,
            "answer": precondition_error,
            "tool_calls": [],
        }

    tools = _get_tools_for_planning()
    recommendation_rules = _load_recommendation_rules()

    # 純場景詢問使用確定性規則直接回答，避免模型漏掉歸位位置、理由，
    # 或因模型輸出格式異常而讓原本不需規劃工具的問題失敗。
    if _is_observation_only_request(normalized_text):
        time.sleep(3.0)
        return {
            "user_text": normalized_text,
            "answer": _build_observation_answer(
                detected_objects=detected_objects,
                recommendation_rules=recommendation_rules,
            ),
            "tool_calls": [],
        }

    # 明確、無參數的直接硬體命令不需要模型推論，避免模型擅自增加
    # 動作，或在複合句中口頭承諾後漏掉部分 tool_call。
    direct_action_plan = _build_single_direct_action_plan(normalized_text)
    if isinstance(direct_action_plan, dict):
        action_error = _check_action_preconditions(
            robot_status=robot_status,
            tool_calls=direct_action_plan["tool_calls"],
        )
        if action_error is not None:
            return {
                "user_text": normalized_text,
                "answer": action_error,
                "tool_calls": [],
            }

        validated_calls = []
        for index, tool_call in enumerate(direct_action_plan["tool_calls"]):
            validated_call, missing_required = _validate_tool_call(
                tool_call,
                index=index,
            )
            if validated_call is None:
                raise AgentError(
                    "直接動作缺少必要參數：" + "、".join(missing_required)
                )
            validated_calls.append(validated_call)
        return {
            "user_text": normalized_text,
            "answer": direct_action_plan["answer"],
            "tool_calls": validated_calls,
        }

    user_prompt = dedent(
        f"""
        機器手臂狀態：
        {json.dumps(robot_status, ensure_ascii=False, indent=2)}
        - status_available：是否成功取得機器手臂狀態。
        - 若 status_available 為 false，明確跟使用者說感覺不到手臂，不可以猜測機器手臂狀態。
        - 若 status_available 為 true，但是 connected 為 false，代表感覺得到有手臂，但無法控制，不可以假設手臂可以正常動作。
        - connected：機器手臂是否連線。
        - pose：機器手臂目前 TCP 位姿，格式為[x, y, z, rx, ry, rz]。
        - joints：機器手臂目前各關節角度，單位為 radian。

        物件辨識到的物件：
        {json.dumps(detected_objects, ensure_ascii=False, indent=2)}

        物件辨識規則：
        - detection_available：是否成功取得物件辨識結果。
        - 若 detection_available 為 false，明確跟使用者說你現在看不到這個世界，不可以假設看得到世界。
        - 若 detection_available 為 true，但是 objects 為空陣列，代表看得到世界，但是沒有看到任何物件，不可以假設有物件存在。
        - 若 detection_available 為 true 且 objects 不為空陣列，只能根據 objects 內的資料回答，不可以自行增加不存在的物件。
        - objects：目前辨識到的物件陣列。
        - object_id：本次命令內唯一的物件 ID。所有 place_object_* 必須用它選取物件。
        - request-local object_id 通常包含 class_name，例如 disposable_mask_1、saline_1。
        - 必須先以 class_name 找到使用者指定的物件，再使用同一筆資料的 object_id；不可依陣列位置猜測。
        - class_name：辨識到的物件類別名稱。
        - confidence：辨識到的物件辨識信心值。
        - robot_xyz：物件在機器人 Base 座標系的位置，單位為公尺。
        - yaw_deg：物件相對機器人 Base 座標系的擺放角度，單位為度。
        - condition、visual_description、state_tags：若不是 null，才可作為物件外觀或狀態判斷的證據。
        - 若 yaw_deg 是 N/A、n/a、na 或 null，代表物件方向沒有實質差異，使用 0.0 度。
        - 絕對不可以假裝有辨識到的物件，或是自行增加不存在的物件。

        回答中的處理判斷：
        - 執行收納或清理任務時，answer 應簡短說明選擇目的地的理由，不可只說「我會處理」。
        - 可依 rules.json 的 handling_guidance 說明一般處理原則，例如拋棄式口罩不建議重複使用。
        - 「已拆封、未拆封、已使用、未使用、污損、乾淨、滲漏」是物件當下狀態；只有 objects 的
          condition、visual_description 或 state_tags 明確提供證據時，才可直接斷言。
        - 沒有狀態證據時，必須使用條件句，例如「若鹽水未拆封且包裝完整，可收進抽屜」，
          不可直接說「這瓶鹽水未拆封」。
        - 判斷文字不得改變使用者明確指定的目的地，也不得繞過工具與安全驗證。

        批次整理規則：
        - 「所有」、「全部」、「清空」、「全部收好」表示要處理 objects 中的每一個物件。
        - 「除了 A 以外都收掉」、「A 除外」表示 A 不可產生任何放置工具，其餘符合條件的物件都要處理。
        - 使用者指定某一群物件時，只處理符合描述的物件。例如「醫療物品放到抽屜」應依 rules.json
          找出預設目的地為 second_drawer 的所有可見物件，不可漏掉，也不可加入其他物件。
        - 「醫療物品」、「醫療物件」、「要進入抽屜的物件」都明確定義為 rules.json 中
          destination=second_drawer 的所有可見物件；酒精瓶預設進櫃子，不屬於這個群組。
        - 「垃圾」、「廢棄物」、「污損布料」、「髒布料」明確定義為 rules.json 中
          destination=trash_can 的所有可見物件；不要憑常識改變 rules.json 的分類。
        - 「要進去垃圾桶的垃圾」、「要丟進垃圾桶的物件」同樣只代表 rules.json 中
          destination=trash_can 的可見物件；鹽水預設進抽屜，不屬於這個群組。
        - 「丟垃圾」、「清理垃圾」中的「垃圾」是物件群組，只能選擇 rules.json 中
          destination=trash_can 的物件；不可隨意選一個物件後再用其預設目的地處理。
        - detection_available 為 true 且 objects 不為空時，資料已經在本次 prompt 中，不需要也不可以回答稍後再檢查。
        - 為每一個符合命令的物件產生一個放置工具呼叫；同類物件有多個實體也要逐一處理。
        - 每個放置工具的 arguments 只能填入對應物件的 object_id，不可把不同物件的 ID 混用。
        - 使用者未指定目的地時，依 rules.json 決定每個物件的目的地；不能用一個目的地處理所有不同物件。

        使用者語意優先規則：
        - 必須完整理解否定、排除、改放、只處理、不要處理、保持開啟等語意，不可只抓關鍵字。
        - 否定的目的地絕對不可採用。例如「毛巾不要放垃圾桶，幫我放抽屜」只能放抽屜。
        - 使用者明確指定的目的地高於 rules.json；rules.json 只在使用者沒有指定目的地時提供預設值。
        - 使用者只是詢問「你看到什麼」、「場景有什麼」或要求描述場景時，除了描述 objects，
          還必須依 rules.json 說明每種物件的推薦歸位位置與理由；tool_calls 必須為 []。
        - 使用者同時要求先說明場景再執行時，answer 應先描述物件，並在同一次回覆產生完整 tool_calls。

        容器操作與排序規則：
        - 除非使用者明確說某容器目前開著，否則櫃子、第二抽屜與垃圾桶一律視為關閉。
        - 放置前若容器視為關閉，必須先呼叫對應 open_*；若使用者明確說已開著，不可重複打開。
        - 同一目的地的物件必須排在一起：打開容器、放完該容器的所有物件、立刻關閉容器，再處理下一處。
        - 預設完成一個容器後立即呼叫對應 close_*，不可把所有 close_* 集中排到最後，以降低碰撞風險。
        - 只有使用者明確說「不要關」、「保持開著」等要求時，才可省略對應 close_*。
        - 若命令只要求放置物件，仍須自動補齊必要的 open_*、放置工具與 close_*。

        若使用者要求的動作只涉及開或關門、抽屜或垃圾桶，這些動作不需要任何物件辨識資訊。
        在這種情況下，即使 detection_available 為 false，仍可直接回傳對應的 open_* 或 close_* 工具。

        YOLO 可以辨識的物件類別：
        - bottle_alcohol_spray：酒精瓶。
        - cotton_swab：棉花棒。
        - cotton_swabs_pp：棉花棒包裝。
        - disposable_mask：口罩。
        - gauze_pp：紗布包裝。
        - saline：生理食鹽水。
        - syringe_nipro：針筒。
        - waterproof_bandages_ppb：OK繃。
        - square_striped_towel：布。

        物件名稱理解規則：
        - 使用者說「酒精」、「酒精瓶」、「噴霧瓶」、「水瓶」時，對應 bottle_alcohol_spray。
        - 使用者說「棉花棒」、「棉籤」、「棉棒」時，對應 cotton_swab。
        - 使用者說「棉花棒包裝」時，對應 cotton_swabs_pp。
        - 使用者說「口罩」時，對應 disposable_mask。
        - 使用者說「紗布」時，對應 gauze_pp。
        - 使用者說「食鹽水」、「生理食鹽水」或「鹽水」時，對應 saline。
        - 使用者說「注射器」或「針筒」時，對應 syringe_nipro。
        - 使用者說「OK 繃」、「防水 OK 繃」或「防水繃帶」時，對應 waterproof_bandages_ppb。
        - 使用者說「抹布」、「毛巾」、「布」時，對應 square_striped_towel。
        - 這些規則只用於將使用者的物件名稱與別名對應到 class_name，不用於決定物件分類或目的地。

        物件推薦規則（載自 rules.json）：
        {json.dumps(recommendation_rules, ensure_ascii=False, indent=2)}
        - objects 的 key 是物件 class_name，object_label 是中文物件名稱，destination 是推薦目的地。
        - handling_guidance 是提供 answer 使用的處理理由與措辭限制，不是額外工具，也不是物件狀態證據。
        - destinations 定義每個目的地的中文名稱與 suggested_action。
        - 使用者的明確意圖優先於 rules.json。若使用者明確指定櫃子、抽屜或垃圾桶等目的地，必須依使用者指定的目的地選擇工具，不可被 rules.json 的推薦覆蓋。
        - 只有當使用者未指定目的地，例如要求「整理、收好、處理這個物件」時，才依照 rules.json 的 destination 與 suggested_action。
         - rules.json 是未指定目的地時的預設推薦規則，不是強制限制。
        - 若 rules.json 與其他內建分類描述矛盾，以 rules.json 為準；但使用者明確指定的目的地優先於 rules.json。
        - 實際比對辨識結果時，必須使用 objects 中的 class_name，不可猜測不存在的物件。

        可使用的 API 工具：
        {json.dumps(tools, ensure_ascii=False, indent=2)}
        - 不可以假設有不存在的 API 工具。

        請根據機器手臂狀態、物件辨識結果、可使用的 API 工具，回應使用者的需求。
        使用者需求：
        {normalized_text}
        """
    ).strip()

    messages = [
        {
            "role": "system",
            "content": PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    all_tools = get_tool()
    tool_names = list(all_tools.keys())

    response_schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
            },
            "decision_basis": {
                "type": "string",
            },
            "tool_calls": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "enum": tool_names,
                        },
                        "arguments": {
                            "type": "object",
                        },
                    },
                    "required": [
                        "function_name",
                        "arguments",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "answer",
            "decision_basis",
            "tool_calls",
        ],
        "additionalProperties": False,
    }

    parsed = None

    for attempt in range(2):
        
        message = (llm_service.chat_structured(messages=messages, response_schema=response_schema,))

        if not isinstance(message, dict):
            raise AgentError("Ollama message 必須是 dict")

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise AgentError("Ollama message 缺少有效 content")

        parsed = _check_response(
            content,
            detected_objects=detected_objects,
            user_text=normalized_text,
            recommendation_rules=recommendation_rules,
            robot_status=robot_status,
        )
        retryable = parsed.pop("_retryable", False)
        if not retryable or attempt == 1:
            break

        messages.extend([
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "上一份規劃未通過系統驗證："
                    f"{parsed['answer']}\n"
                    "請重新閱讀原始需求與 objects，從頭輸出完整 JSON。"
                    "每個具名物件必須選擇 class_name 相符的 object_id；"
                    "不同物件不可重複使用同一 object_id；"
                    "不要只修補單一步驟，必須重新輸出完整 tool_calls。"
                ),
            },
        ])

    if parsed is None:
        raise AgentError("Ollama 未產生可解析的規劃")

    return {
        "user_text": normalized_text,
        **parsed,
    }


def _check_pick_point_range(function_name, arguments):
    """
    驗證需要夾取物件的任務，其 point_xyz 是否位於允許範圍內。

    回傳：
    - None：座標合法。
    - str：座標超出範圍或格式錯誤的使用者提示。
    """
    if function_name not in PICK_POINT_TOOLS:
        return None

    point_xyz = arguments.get("point_xyz")

    if not isinstance(point_xyz, list) or len(point_xyz) != 3:
        return "物件座標格式不完整，目前無法執行夾取任務。"

    if not all(_is_number(value) for value in point_xyz):
        return "物件座標格式錯誤，目前無法執行夾取任務。"

    x, y, z = point_xyz

    out_of_range = []

    if not PICK_POINT_LIMITS["x"]["minimum"] <= x <= PICK_POINT_LIMITS["x"]["maximum"]:
        out_of_range.append(
            f"x={x}，允許範圍為 "
            f"{PICK_POINT_LIMITS['x']['minimum']} 至 "
            f"{PICK_POINT_LIMITS['x']['maximum']}"
        )

    if not PICK_POINT_LIMITS["y"]["minimum"] <= y <= PICK_POINT_LIMITS["y"]["maximum"]:
        out_of_range.append(
            f"y={y}，允許範圍為 "
            f"{PICK_POINT_LIMITS['y']['minimum']} 至 "
            f"{PICK_POINT_LIMITS['y']['maximum']}"
        )

    if not PICK_POINT_LIMITS["z"]["minimum"] <= z <= PICK_POINT_LIMITS["z"]["maximum"]:
        out_of_range.append(
            f"z={z}，允許範圍為 "
            f"{PICK_POINT_LIMITS['z']['minimum']} 至 "
            f"{PICK_POINT_LIMITS['z']['maximum']}"
        )

    if not out_of_range:
        return None

    return (
        "這個物件的位置超出機器手臂允許的夾取範圍，"
        "因此目前無法執行。超出範圍的座標為："
        + "；".join(out_of_range)
    )


def _find_detected_object_by_id(detected_objects, object_id):
    """從本次偵測快照尋找模型選取的物件。"""
    if not isinstance(detected_objects, dict):
        return None

    objects = detected_objects.get("objects")
    if not isinstance(objects, list):
        return None

    for obj in objects:
        if isinstance(obj, dict) and obj.get("object_id") == object_id:
            return obj

    return None


def _resolve_planning_tool_call(tool_call, index, detected_objects):
    """將模型的 object_id 規劃呼叫解析為 executor 使用的座標呼叫。"""
    if not isinstance(tool_call, dict):
        return tool_call, None, None

    function_name = tool_call.get("function_name")
    if function_name not in PICK_POINT_TOOLS:
        return tool_call, None, None

    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        return tool_call, None, None

    unknown_arguments = set(arguments) - {"object_id"}
    if unknown_arguments:
        return None, None, (
            f"第 {index + 1} 個步驟「{function_name}」的規劃參數無效："
            "place_object_* 只能提供 object_id，不可直接提供 point_xyz 或 yaw_deg。"
        )

    object_id = arguments.get("object_id")
    if not isinstance(object_id, str) or not object_id.strip():
        return None, None, (
            f"第 {index + 1} 個步驟「{function_name}」缺少有效的 object_id。"
        )
    object_id = object_id.strip()

    detected_object = _find_detected_object_by_id(
        detected_objects,
        object_id,
    )
    if detected_object is None:
        return None, None, (
            f"第 {index + 1} 個步驟「{function_name}」引用不存在的"
            f"物件 ID「{object_id}」。"
        )

    yaw_deg = detected_object.get("yaw_deg")
    if yaw_deg is None or (
        isinstance(yaw_deg, str)
        and yaw_deg.strip().lower() in {"n/a", "na"}
    ):
        yaw_deg = 0.0

    return (
        {
            "function_name": function_name,
            "arguments": {
                "point_xyz": detected_object.get("robot_xyz"),
                "yaw_deg": yaw_deg,
            },
        },
        object_id,
        None,
    )


def _check_bulk_plan_completeness(
    user_text,
    detected_objects,
    tool_calls,
    selected_object_ids,
    recommendation_rules,
):
    """檢查無排除條件的全場整理計畫是否真的處理完所有物件。"""
    if not isinstance(user_text, str) or not _is_bulk_cleanup_request(user_text):
        return []

    normalized = re.sub(r"[\s，,。.!！?？、]", "", user_text.lower())
    objects = (
        detected_objects.get("objects")
        if isinstance(detected_objects, dict)
        else None
    )
    if not isinstance(objects, list):
        return []

    category_destination = _requested_rule_destination_group(normalized)

    target_objects = objects
    if category_destination is not None:
        object_rules = (
            recommendation_rules.get("objects")
            if isinstance(recommendation_rules, dict)
            else None
        )
        if not isinstance(object_rules, dict):
            return ["無法讀取群組分類所需的 rules.json objects"]
        target_objects = [
            obj
            for obj in objects
            if (
                isinstance(obj, dict)
                and isinstance(object_rules.get(obj.get("class_name")), dict)
                and object_rules[obj["class_name"]].get("destination")
                == category_destination
            )
        ]
    else:
        selection_terms = (
            "除了", "除外", "只收", "只處理", "不要動",
            "別動", "留下", "保留",
        )
        if any(term in normalized for term in selection_terms):
            return []

    problems = []
    placed_object_ids = {
        object_id
        for object_id in selected_object_ids
        if object_id is not None
    }

    if target_objects and not placed_object_ids:
        problems.append("沒有任何 place_object_* 放置步驟")

    target_object_ids = {
        obj.get("object_id")
        for obj in target_objects
        if isinstance(obj, dict)
    }

    for index, obj in enumerate(target_objects):
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("object_id")
        if object_id not in placed_object_ids:
            class_name = obj.get("class_name") or "未知物件"
            problems.append(
                f"漏掉第 {index + 1} 個物件「{class_name}」"
                f"（object_id={object_id}）"
            )

    if category_destination is not None:
        for object_id in sorted(placed_object_ids - target_object_ids):
            obj = _find_detected_object_by_id(detected_objects, object_id)
            class_name = (
                obj.get("class_name")
                if isinstance(obj, dict)
                else "未知物件"
            )
            problems.append(
                f"多選了不屬於此群組的物件「{class_name}」"
                f"（object_id={object_id}）"
            )

    container_steps = {
        "trash_can": (
            "open_trash_can",
            "place_object_in_trash_can",
            "close_trash_can",
        ),
        "top_cabinet": (
            "open_top_cabinet",
            "place_object_in_top_cabinet",
            "close_top_cabinet",
        ),
        "second_drawer": (
            "open_second_drawer",
            "place_object_in_second_drawer",
            "close_second_drawer",
        ),
    }
    names = [call["function_name"] for call in tool_calls]

    for container_name, (open_name, place_name, close_name) in container_steps.items():
        if open_name not in names:
            continue
        open_index = names.index(open_name)
        later_names = names[open_index + 1:]
        if place_name not in later_names:
            problems.append(f"打開 {container_name} 後沒有放置任何物件")
        if close_name not in later_names:
            problems.append(f"打開 {container_name} 後沒有關閉容器")

    return problems


def _has_explicit_destination_override(user_text):
    """判斷使用者是否明確要求把物件送往特定容器。"""
    if not isinstance(user_text, str):
        return False
    normalized = re.sub(r"[\s，,。.!！?？、]", "", user_text.lower())
    return re.search(
        r"(?:放|丟|扔|收進|收到|塞進).{0,8}(?:櫃子|櫃內|抽屜|垃圾桶)",
        normalized,
    ) is not None


def _check_default_destinations(
    user_text,
    detected_objects,
    tool_calls,
    selected_object_ids,
    recommendation_rules,
):
    """未指定目的地時，確認模型選用的工具符合 rules.json。"""
    if _has_explicit_destination_override(user_text):
        return []
    if not isinstance(recommendation_rules, dict):
        return []

    object_rules = recommendation_rules.get("objects")
    destinations = recommendation_rules.get("destinations")
    if not isinstance(object_rules, dict) or not isinstance(destinations, dict):
        return []

    problems = []
    for index, call in enumerate(tool_calls):
        function_name = call["function_name"]
        if function_name not in PICK_POINT_TOOLS:
            continue

        object_id = selected_object_ids[index]
        obj = _find_detected_object_by_id(detected_objects, object_id)
        if not isinstance(obj, dict):
            continue
        class_name = obj.get("class_name")
        object_rule = object_rules.get(class_name)
        destination = (
            destinations.get(object_rule.get("destination"))
            if isinstance(object_rule, dict)
            else None
        )
        expected_function = (
            destination.get("suggested_action")
            if isinstance(destination, dict)
            else None
        )
        if expected_function in PICK_POINT_TOOLS and function_name != expected_function:
            problems.append(
                f"第 {index + 1} 個步驟選錯目的地：物件「{class_name}」"
                f"（{object_id}）未指定目的地時應使用「{expected_function}」，"
                f"但模型使用「{function_name}」"
            )

    return problems


def _repair_default_destinations(
    user_text,
    detected_objects,
    tool_calls,
    selected_object_ids,
    recommendation_rules,
):
    """未指定目的地時，以 rules.json 正規化模型選用的放置工具。"""
    if _has_explicit_destination_override(user_text):
        return tool_calls, False, [], []
    if not isinstance(recommendation_rules, dict):
        return tool_calls, False, [], []

    object_rules = recommendation_rules.get("objects")
    destinations = recommendation_rules.get("destinations")
    if not isinstance(object_rules, dict) or not isinstance(destinations, dict):
        return tool_calls, False, [], []

    repaired_calls = []
    destination_repaired = False
    plan_lines = []
    decision_lines = []

    for index, call in enumerate(tool_calls):
        function_name = call["function_name"]
        if function_name not in PICK_POINT_TOOLS:
            repaired_calls.append(call)
            continue

        object_id = selected_object_ids[index]
        obj = _find_detected_object_by_id(detected_objects, object_id)
        class_name = obj.get("class_name") if isinstance(obj, dict) else None
        object_rule = object_rules.get(class_name)
        destination = (
            destinations.get(object_rule.get("destination"))
            if isinstance(object_rule, dict)
            else None
        )
        expected_function = (
            destination.get("suggested_action")
            if isinstance(destination, dict)
            else None
        )

        if expected_function not in PICK_POINT_TOOLS:
            repaired_calls.append(call)
            continue

        repaired_call = {
            **call,
            "function_name": expected_function,
        }
        repaired_calls.append(repaired_call)

        object_label = object_rule.get("object_label") or class_name or object_id
        plan_lines.append(object_label)

        user_reason = object_rule.get("user_reason")
        if isinstance(user_reason, str) and user_reason.strip():
            decision_lines.append(user_reason.strip())
        else:
            destination_label = (
                destination.get("label_zh")
                or object_rule.get("destination")
            )
            decision_lines.append(f"{object_label}適合收進{destination_label}")

        if function_name != expected_function:
            destination_repaired = True

    return repaired_calls, destination_repaired, plan_lines, decision_lines


def _repair_container_sequence(user_text, tool_calls, selected_object_ids):
    """保留模型的放置決策，依安全規則重建容器 open/close 步驟。"""
    definitions = {
        "trash_can": {
            "names": ("垃圾桶", "垃圾桶蓋"),
            "open": "open_trash_can",
            "place": "place_object_in_trash_can",
            "close": "close_trash_can",
        },
        "top_cabinet": {
            "names": ("櫃子", "櫃門"),
            "open": "open_top_cabinet",
            "place": "place_object_in_top_cabinet",
            "close": "close_top_cabinet",
        },
        "second_drawer": {
            "names": ("抽屜", "抽屜門"),
            "open": "open_second_drawer",
            "place": "place_object_in_second_drawer",
            "close": "close_second_drawer",
        },
    }
    place_to_container = {
        definition["place"]: container_name
        for container_name, definition in definitions.items()
    }
    container_action_names = {
        definition[action]
        for definition in definitions.values()
        for action in ("open", "close")
    }
    if not any(call["function_name"] in place_to_container for call in tool_calls):
        return tool_calls, selected_object_ids, False

    normalized = re.sub(r"[\s，,。.!！?？、]", "", str(user_text or "").lower())
    states = {name: "closed" for name in definitions}
    if "所有容器都開著" in normalized:
        states = {name: "open" for name in definitions}
    else:
        for container_name, definition in definitions.items():
            if any(
                any(
                    phrase in normalized
                    for phrase in (
                        f"{name}開著",
                        f"{name}已開",
                        f"{name}已經開",
                    )
                )
                for name in definition["names"]
            ):
                states[container_name] = "open"

    leave_open = any(
        term in normalized
        for term in ("不要關", "不用關", "別關", "保持開", "留著開")
    )
    core_steps = [
        (call, selected_object_ids[index])
        for index, call in enumerate(tool_calls)
        if call["function_name"] not in container_action_names
    ]
    repaired_calls = []
    repaired_ids = []
    active_container = None

    def append_call(function_name, arguments=None, object_id=None):
        repaired_calls.append({
            "function_name": function_name,
            "arguments": arguments or {},
        })
        repaired_ids.append(object_id)

    def close_active_container():
        nonlocal active_container
        if active_container is None or leave_open:
            return
        if states[active_container] == "open":
            append_call(definitions[active_container]["close"])
            states[active_container] = "closed"
        active_container = None

    for call, object_id in core_steps:
        function_name = call["function_name"]
        container_name = place_to_container.get(function_name)

        if container_name is None:
            close_active_container()
            append_call(function_name, call["arguments"], object_id)
            continue

        if active_container is not None and active_container != container_name:
            close_active_container()

        if states[container_name] == "closed":
            append_call(definitions[container_name]["open"])
            states[container_name] = "open"

        append_call(function_name, call["arguments"], object_id)
        active_container = container_name

    close_active_container()
    return repaired_calls, repaired_ids, repaired_calls != tool_calls


def _check_container_sequence(user_text, tool_calls):
    """以容器狀態機檢查 open/place/close 的實際先後關係。"""
    if not any(call["function_name"] in PICK_POINT_TOOLS for call in tool_calls):
        return []

    definitions = {
        "trash_can": {
            "names": ("垃圾桶", "垃圾桶蓋"),
            "open": "open_trash_can",
            "place": "place_object_in_trash_can",
            "close": "close_trash_can",
        },
        "top_cabinet": {
            "names": ("櫃子", "櫃門"),
            "open": "open_top_cabinet",
            "place": "place_object_in_top_cabinet",
            "close": "close_top_cabinet",
        },
        "second_drawer": {
            "names": ("抽屜", "抽屜門"),
            "open": "open_second_drawer",
            "place": "place_object_in_second_drawer",
            "close": "close_second_drawer",
        },
    }
    normalized = re.sub(r"[\s，,。.!！?？、]", "", str(user_text or "").lower())
    states = {name: "closed" for name in definitions}
    placements = {name: 0 for name in definitions}

    if "所有容器都開著" in normalized:
        states = {name: "open" for name in definitions}
    else:
        for container_name, definition in definitions.items():
            if any(
                any(
                    phrase in normalized
                    for phrase in (
                        f"{name}開著",
                        f"{name}已開",
                        f"{name}已經開",
                    )
                )
                for name in definition["names"]
            ):
                states[container_name] = "open"

    leave_open = any(
        term in normalized
        for term in ("不要關", "不用關", "別關", "保持開", "留著開")
    )
    problems = []

    for index, call in enumerate(tool_calls):
        function_name = call["function_name"]
        for container_name, definition in definitions.items():
            if function_name == definition["open"]:
                if states[container_name] == "open":
                    problems.append(
                        f"第 {index + 1} 個步驟重複打開已開啟的 {container_name}"
                    )
                states[container_name] = "open"
                placements[container_name] = 0
                break
            if function_name == definition["place"]:
                if states[container_name] != "open":
                    problems.append(
                        f"第 {index + 1} 個步驟在 {container_name} 尚未打開時就放置物件"
                    )
                placements[container_name] += 1
                break
            if function_name == definition["close"]:
                if states[container_name] != "open":
                    problems.append(
                        f"第 {index + 1} 個步驟關閉尚未打開的 {container_name}"
                    )
                elif placements[container_name] == 0:
                    problems.append(
                        f"第 {index + 1} 個步驟在 {container_name} 尚未放置物件時就關閉"
                    )
                states[container_name] = "closed"
                break

    if not leave_open:
        for container_name, state in states.items():
            if state == "open":
                problems.append(f"任務結束時 {container_name} 仍保持開啟")

    return problems


def _check_response(
    content,
    detected_objects=None,
    user_text=None,
    recommendation_rules=None,
    robot_status=None,
):
    """
    解析並驗證模型回傳的工具執行計畫。
    """
    if not isinstance(content, str) or not content.strip():
        raise AgentError("模型沒有回傳有效內容")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AgentError(f"模型回傳不是有效 JSON：{exc}") from exc

    if not isinstance(data, dict):
        raise AgentError("模型回傳必須是 JSON object")

    answer = data.get("answer")
    decision_basis = data.get("decision_basis")
    tool_calls = data.get("tool_calls")

    if not isinstance(answer, str):
        raise AgentError("answer 必須是字串")

    if not isinstance(decision_basis, str):
        raise AgentError("decision_basis 必須是字串")

    if not isinstance(tool_calls, list):
        raise AgentError("tool_calls 必須是 array")

    answer = answer.strip() or "AI 無法回應"
    decision_basis = decision_basis.strip()

    if _is_observation_only_request(user_text):
        return {
            "answer": _build_observation_answer(
                detected_objects=detected_objects,
                recommendation_rules=recommendation_rules,
            ),
            "tool_calls": [],
        }

    direct_action_plan = _build_single_direct_action_plan(user_text)
    direct_action_answer = None
    if isinstance(direct_action_plan, dict):
        tool_calls = direct_action_plan["tool_calls"]
        direct_action_answer = direct_action_plan["answer"]

    group_plan = _build_rule_destination_group_plan(
        user_text=user_text,
        detected_objects=detected_objects,
        recommendation_rules=recommendation_rules,
    )
    group_plan_answer = None
    if isinstance(group_plan, dict):
        tool_calls = group_plan["tool_calls"]
        group_plan_answer = group_plan["answer"]

    if len(tool_calls) > 20:
        raise AgentError("tool_calls 數量不可超過 20")

    action_precondition_error = _check_action_preconditions(
        robot_status=robot_status,
        tool_calls=tool_calls,
    )
    if action_precondition_error is not None:
        return {
            "answer": action_precondition_error,
            "tool_calls": [],
        }

    validated_tool_calls = []
    selected_object_ids = []
    range_problems = []

    for index, tool_call in enumerate(tool_calls):
        resolved_tool_call, object_id, resolution_error = (
            _resolve_planning_tool_call(
                tool_call=tool_call,
                index=index,
                detected_objects=detected_objects,
            )
        )
        if resolution_error is not None:
            return {
                "answer": f"任務規劃無效，因此不會執行：{resolution_error}",
                "tool_calls": [],
                "_retryable": True,
            }

        validated_tool_call, missing_required = _validate_tool_call(
            tool_call=resolved_tool_call,
            index=index,
        )

        if validated_tool_call is None:
            function_name = resolved_tool_call.get("function_name")
            missing_text = "、".join(missing_required)

            return {
                "answer": (
                    f"任務的第 {index + 1} 個步驟「{function_name}」"
                    f"無法執行，缺少必要參數：{missing_text}。"
                ),
                "tool_calls": [],
                "_retryable": True,
            }

        tool = get_tool(validated_tool_call["function_name"])
        range_error = None

        if (
            isinstance(tool, dict)
            and isinstance(tool.get("parameters"), dict)
            and "point_xyz" in tool["parameters"]
        ):
            range_error = _check_pick_point_range(
                function_name=validated_tool_call["function_name"],
                arguments=validated_tool_call["arguments"],
            )

        if range_error is not None:
            detected_object = _find_detected_object_by_id(
                detected_objects,
                object_id,
            )
            object_name = (
                detected_object.get("class_name")
                if isinstance(detected_object, dict)
                else None
            )
            object_text = (
                f"，物件「{object_name}」"
                if isinstance(object_name, str) and object_name
                else ""
            )
            range_problems.append(
                f"第 {index + 1} 個步驟"
                f"「{validated_tool_call['function_name']}」"
                f"{object_text}：{range_error}"
            )

        validated_tool_calls.append(validated_tool_call)
        selected_object_ids.append(object_id)

    (
        validated_tool_calls,
        destination_repaired,
        natural_plan_lines,
        rule_decision_lines,
    ) = _repair_default_destinations(
        user_text=user_text,
        detected_objects=detected_objects,
        tool_calls=validated_tool_calls,
        selected_object_ids=selected_object_ids,
        recommendation_rules=recommendation_rules,
    )

    destination_problems = _check_default_destinations(
        user_text=user_text,
        detected_objects=detected_objects,
        tool_calls=validated_tool_calls,
        selected_object_ids=selected_object_ids,
        recommendation_rules=recommendation_rules,
    )
    if destination_problems:
        return {
            "answer": (
                "任務規劃不符合目的地規則，因此不會執行："
                + "；".join(destination_problems)
                + "。"
            ),
            "tool_calls": [],
            "_retryable": True,
        }

    (
        validated_tool_calls,
        selected_object_ids,
        container_steps_repaired,
    ) = _repair_container_sequence(
        user_text=user_text,
        tool_calls=validated_tool_calls,
        selected_object_ids=selected_object_ids,
    )

    if len(validated_tool_calls) > 20:
        return {
            "answer": "補齊容器安全步驟後，任務步驟超過 20 個，因此不會執行。",
            "tool_calls": [],
        }

    container_problems = _check_container_sequence(
        user_text=user_text,
        tool_calls=validated_tool_calls,
    )
    if container_problems:
        return {
            "answer": (
                "容器安全步驟無法完成，因此不會執行："
                + "；".join(container_problems)
                + "。"
            ),
            "tool_calls": [],
            "_retryable": True,
        }

    if range_problems:
        return {
            "answer": (
                "任務包含無法執行的夾取步驟，因此已拒絕整批任務："
                + "；".join(range_problems)
            ),
            "tool_calls": [],
        }

    completeness_problems = _check_bulk_plan_completeness(
        user_text=user_text,
        detected_objects=detected_objects,
        tool_calls=validated_tool_calls,
        selected_object_ids=selected_object_ids,
        recommendation_rules=recommendation_rules,
    )
    if completeness_problems:
        return {
            "answer": (
                "批次任務規劃不完整，因此已拒絕整批任務："
                + "；".join(completeness_problems)
                + "。"
            ),
            "tool_calls": [],
            "_retryable": True,
        }

    pick_destinations = {}

    for index, tool_call in enumerate(validated_tool_calls):
        function_name = tool_call["function_name"]

        if function_name not in PICK_POINT_TOOLS:
            continue

        object_id = selected_object_ids[index]
        previous = pick_destinations.get(object_id)

        if previous is not None and previous["function_name"] != function_name:
            return {
                "answer": (
                    "任務規劃衝突：第 "
                    f"{previous['step_number']} 個步驟「{previous['function_name']}」"
                    f"與第 {index + 1} 個步驟「{function_name}」"
                    f"要把同一個物件「{object_id}」放到不同目的地，"
                    "因此目前不會執行。"
                ),
                "tool_calls": [],
                "_retryable": True,
            }

        pick_destinations[object_id] = {
            "step_number": index + 1,
            "function_name": function_name,
        }

    if direct_action_answer is not None:
        public_answer = direct_action_answer
    elif group_plan_answer is not None:
        public_answer = group_plan_answer
    elif destination_repaired and natural_plan_lines:
        public_reasons = [
            line.rstrip("。； ")
            for line in rule_decision_lines
            if isinstance(line, str) and line.strip()
        ]
        object_summary = "跟".join(natural_plan_lines)
        public_answer = (
            "；".join(public_reasons)
            + f"。我會依序把{object_summary}分類處理。"
        )
    else:
        public_answer = answer.strip()

    return {
        "answer": public_answer,
        "tool_calls": validated_tool_calls,
    }


def _validate_tool_call(tool_call, index):
    """
    驗證單一工具呼叫。

    回傳 (validated_tool_call, missing_required)：
    - 驗證成功：(dict, [])。
    - 缺少必要參數：(None, list)。
    """
    step_number = index + 1

    if not isinstance(tool_call, dict):
        raise AgentError(
            f"第 {step_number} 個 tool_call 必須是 object"
        )

    allowed_fields = {
        "function_name",
        "arguments",
    }

    unknown_fields = set(tool_call) - allowed_fields

    if unknown_fields:
        unknown_text = "、".join(sorted(unknown_fields))
        raise AgentError(
            f"第 {step_number} 個 tool_call 包含未知欄位："
            f"{unknown_text}"
        )

    function_name = tool_call.get("function_name")
    arguments = tool_call.get("arguments")

    if not isinstance(function_name, str):
        raise AgentError(
            f"第 {step_number} 個 function_name 必須是字串"
        )

    function_name = function_name.strip()

    if not function_name:
        raise AgentError(
            f"第 {step_number} 個 function_name 不可為空字串"
        )

    if not isinstance(arguments, dict):
        raise AgentError(
            f"第 {step_number} 個 arguments 必須是 object"
        )

    # 使用副本進行正規化，避免修改模型回傳的原始資料。
    arguments = arguments.copy()

    tool = get_tool(function_name)

    if not isinstance(tool, dict):
        raise AgentError(
            f"第 {step_number} 個工具未授權：{function_name}"
        )

    required = tool.get("required", [])

    if not isinstance(required, list):
        raise AgentError(
            f"{function_name} 的 required 必須是 list"
        )

    parameters = tool.get("parameters", {})
    yaw_definition = (
        parameters.get("yaw_deg")
        if isinstance(parameters, dict)
        else None
    )
    yaw_value = arguments.get("yaw_deg")

    # 接近正方形的物件可能由視覺端回傳 yaw_deg=N/A；此時方向
    # 沒有實質差異，統一使用工具定義的預設角度 0.0。
    if (
        isinstance(yaw_definition, dict)
        and "default" in yaw_definition
        and (
            "yaw_deg" not in arguments
            or yaw_value is None
            or (
                isinstance(yaw_value, str)
                and yaw_value.strip().lower() in {"n/a", "na"}
            )
        )
    ):
        arguments["yaw_deg"] = yaw_definition["default"]

    missing_required = [
        name
        for name in required
        if (
            name not in arguments
            or arguments[name] is None
            or (
                isinstance(arguments[name], str)
                and not arguments[name].strip()
            )
        )
    ]

    if missing_required:
        return None, missing_required

    _validate_arguments(
        function_name=function_name,
        arguments=arguments,
        tool=tool,
    )

    return (
        {
            "function_name": function_name,
            "arguments": arguments,
        },
        [],
    )

def _validate_arguments(function_name, arguments, tool):
    """
    驗證模型傳入的參數名稱。
    """
    if not isinstance(function_name, str) or not function_name.strip():
        raise AgentError("function_name 必須是非空字串")

    if not isinstance(arguments, dict):
        raise AgentError(
            f"{function_name} 的 arguments 必須是 object"
        )

    if not isinstance(tool, dict):
        raise AgentError(
            f"{function_name} 的工具定義必須是 dict"
        )

    parameters = tool.get("parameters", {})

    if not isinstance(parameters, dict):
        raise AgentError(
            f"{function_name} 的工具參數定義錯誤"
        )

    if not all(
        isinstance(name, str) and name.strip()
        for name in parameters
    ):
        raise AgentError(
            f"{function_name} 的參數名稱必須是非空字串"
        )

    # =========================
    # 驗證參數名稱
    # =========================

    allowed = set(parameters)
    actual = set(arguments)
    unknown = actual - allowed

    if unknown:
        unknown_text = "、".join(
            sorted(str(name) for name in unknown)
        )

        raise AgentError(
            f"{function_name} 包含未授權參數："
            f"{unknown_text}"
        )


    # =========================
    # 驗證參數值
    # =========================

    for name, value in arguments.items():
        definition = parameters[name]

        if not isinstance(definition, dict):
            raise AgentError(
                f"{function_name}.{name} 的參數定義必須是 dict"
            )

        _validate_argument_value(
            function_name=function_name,
            parameter_name=name,
            value=value,
            definition=definition,
        )

def _validate_argument_value(function_name, parameter_name, value, definition):
    """
    驗證單一參數值是否符合工具定義。
    """
    full_name = f"{function_name}.{parameter_name}"

    expected_type = definition.get("type")

    if expected_type is None:
        raise AgentError(
            f"{full_name} 缺少 type 定義"
        )

    # 支援：
    # "type": "number"
    # "type": ["integer", "null"]
    allowed_types = (
        expected_type
        if isinstance(expected_type, list)
        else [expected_type]
    )

    if not all(isinstance(item, str) for item in allowed_types):
        raise AgentError(
            f"{full_name} 的 type 定義格式錯誤"
        )

    if not _matches_any_type(value, allowed_types):
        expected_text = " 或 ".join(allowed_types)

        raise AgentError(
            f"{full_name} 型別錯誤，"
            f"必須是 {expected_text}"
        )

    # value 是 None 且允許 null 時，後續規則不再檢查
    if value is None:
        return

    # =========================
    # enum 驗證
    # =========================

    enum_values = definition.get("enum")

    if enum_values is not None:
        if not isinstance(enum_values, list):
            raise AgentError(
                f"{full_name} 的 enum 必須是 list"
            )

        if value not in enum_values:
            allowed_text = "、".join(
                str(item) for item in enum_values
            )

            raise AgentError(
                f"{full_name} 的值不合法：{value}，"
                f"允許值為：{allowed_text}"
            )

    # =========================
    # 數值範圍
    # =========================

    if _is_number(value):
        minimum = definition.get("minimum")
        maximum = definition.get("maximum")

        if minimum is not None:
            if not _is_number(minimum):
                raise AgentError(
                    f"{full_name} 的 minimum 定義必須是數字"
                )

            if value < minimum:
                raise AgentError(
                    f"{full_name} 不可小於 {minimum}，"
                    f"目前為 {value}"
                )

        if maximum is not None:
            if not _is_number(maximum):
                raise AgentError(
                    f"{full_name} 的 maximum 定義必須是數字"
                )

            if value > maximum:
                raise AgentError(
                    f"{full_name} 不可大於 {maximum}，"
                    f"目前為 {value}"
                )

    # =========================
    # 字串規格
    # =========================

    if isinstance(value, str):
        min_length = definition.get("minLength")
        max_length = definition.get("maxLength")

        if min_length is not None and len(value) < min_length:
            raise AgentError(
                f"{full_name} 長度不可小於 {min_length}"
            )

        if max_length is not None and len(value) > max_length:
            raise AgentError(
                f"{full_name} 長度不可大於 {max_length}"
            )

    # =========================
    # 陣列規格
    # =========================

    if isinstance(value, list):
        min_items = definition.get("minItems")
        max_items = definition.get("maxItems")

        if min_items is not None and len(value) < min_items:
            raise AgentError(
                f"{full_name} 至少需要 {min_items} 個項目"
            )

        if max_items is not None and len(value) > max_items:
            raise AgentError(
                f"{full_name} 最多只能有 {max_items} 個項目"
            )

        item_definition = definition.get("items")

        if item_definition is not None:
            if not isinstance(item_definition, dict):
                raise AgentError(
                    f"{full_name}.items 必須是 dict"
                )

            for index, item in enumerate(value):
                _validate_argument_value(
                    function_name=function_name,
                    parameter_name=f"{parameter_name}[{index}]",
                    value=item,
                    definition=item_definition,
                )


def _matches_any_type(value, allowed_types):
    """
    判斷 value 是否符合任一工具參數型別。
    """
    return any(
        _matches_type(value, type_name)
        for type_name in allowed_types
    )


def _matches_type(value, type_name):
    """
    驗證單一 JSON Schema 型別。
    """
    if type_name == "null":
        return value is None

    if type_name == "boolean":
        return isinstance(value, bool)

    if type_name == "integer":
        # bool 是 int 的子類別，因此需要排除
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
        )

    if type_name == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    if type_name == "string":
        return isinstance(value, str)

    if type_name == "array":
        return isinstance(value, list)

    if type_name == "object":
        return isinstance(value, dict)

    raise AgentError(
        f"不支援的參數型別定義：{type_name}"
    )


def _is_number(value):
    """
    判斷是否為數字，但排除 bool。
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )

"""Stage 1 LLM adapter: natural language to validated atomic tool actions only."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from agent.planning.action_validation import (
    SemanticIRError,
    validate_atomic_action_output,
    validate_grounded_action_targets,
    validate_manipulation_preconditions,
    validate_semantic_action_coverage,
)
from agent.planning.structured_provider import StructuredProviderError, chat_structured
from agent.planning.structured_provider import resolve_provider


ABSTRACT_ACTION_EXTRACTION_PROMPT = """
你只負責從完整使用者命令抽出依出現順序排列的 atomic tool actions。

Stage 1 只回答「要做哪些 atomic actions」以及「每個 tool 必要的 scene entity arguments」。
不要判斷先後、同時、dependency、role、group、設備、左右手或實際 device assignment。
這些 temporal/resource 關係由 Stage 2 處理。

每個明確要求的動作恰好輸出一次；不可只讀句子開頭。
除了使用者明確要求的動作外，只能補齊 tool semantics 為了可執行所必需的最低限度前置 action；
不可補無關動作。

Tool prerequisite 規則：
- place_object(X) 表示放下目前已持有的 X，不是憑空把 X 移到目的地。
- 若 environment_context 沒有表示 X 在 task 開始時已被持有，
  則 place_object(X) 前必須先有 pick_object(X)。
- 這是 tool-level executable prerequisite，不是 Stage 2 temporal/resource inference。
- 不可因補 prerequisite 而改寫 grounded entity ID 或加入額外 domain-specific 動作。

所有 scene entity reference arguments，例如：
object_id、source_id、target_id、destination_id、payload_id、container_id，
都必須使用 environment_context 中已 grounding 或 structured scene 已存在的
exact entity ID。

禁止創造、改名、翻譯、正規化或猜測 entity ID。

非常重要：resource/device descriptions 不是 tool arguments。
例如：
- 左手 / 右手
- 一隻手 / 另一隻手
- 同一隻手 / 不同手臂
- arm / robot_arm
- device / resource
- 同一設備 / 不同設備

這些詞只描述執行資源關係，不得被轉成 object_id、source_id、target_id、
destination_id、payload_id、container_id 或任何其他 tool argument。
如果使用者說「用另一隻手打開垃圾桶」，Stage 1 仍只輸出
open_trash_can(container_id=<垃圾桶 entity ID>)。
「另一隻手」保留在原始 user request 中，由 Stage 2 判斷 resource relation。

只能輸出 tool_catalog 定義的 arguments。
不可新增 tool schema 沒有的參數。
不可因修正一個錯誤而刪掉該 tool 的 required arguments。

Collection grounding 規則：
- environment_context.grounded_targets 可能包含 grounding_type="collection_member"。
- 相同 collection_id 的 rows 代表同一個使用者集合 mention 的成員。
- 若使用者明確要求「所有／全部／每個」集合成員執行某操作，必須把該操作展開成每個 member 的 atomic actions。
- 每個展開 action 都必須使用該 member 的 exact object_id。
- 若使用者說「移到對應目的地／對應分流箱」，而 structured scene 或 grounded member 的 attributes 已提供 expected_container，destination_id 必須使用該 exact entity ID。
- 若使用者明確說某些物件「不要移動／保持不動」，不可替那些 entity 產生 pick/place/move actions。
- collection expansion 是合法的一對多 atomic action 展開，不視為 duplicated subplan。
- 若某句只是檢查／盤點背景資訊，而 tool_catalog 沒有 inspect/check tool，不可自行創造不存在的 action。

完整命令 coverage 自檢（輸出前必做，但不要輸出檢查過程）：
- 必須從頭到尾閱讀完整使用者需求，不可在完成前半段或某個 collection expansion 後提早停止。
- 依原始文字順序逐句／逐子句確認：這一段是 executable instruction、negative constraint、observation/context，還是純 resource description。
- 對每一個 executable instruction，使用 tool_catalog 判斷是否存在可執行該意圖的 tool。
- 若存在對應 tool，actions 中必須至少有一個 action 覆蓋該 executable instruction；不得因前一子句已展開成多個 actions 而漏掉後續子句。
- negative constraint 不產生 action；純 observation/context 若沒有對應 tool 也不產生 action；resource description 留給 Stage 2。
- 多個互相獨立的 executable instructions 都必須被保留，即使它們使用不同類型的 tool 或不同 execution resource。
- 最後再次確認：原始需求中每個可由 tool_catalog 執行的明確動作，都已在 actions 中出現恰好一次（collection member 展開除外）。

Examples:
1.「把垃圾桶打開」
{"actions":[{"function_name":"open_trash_can","arguments":{"container_id":"trash_can"}}]}

2.「手往上移兩公分，接著往左移兩公分」
{"actions":[
 {"function_name":"move_arm_step","arguments":{"direction":"z+","distance":0.02}},
 {"function_name":"move_arm_step","arguments":{"direction":"y+","distance":0.02}}
]}

3.「垃圾桶先打開，另一邊把手往上抬，之後再把手往後移，最後關掉垃圾桶」
輸出：
{"actions":[
 {"function_name":"open_trash_can","arguments":{"container_id":"trash_can"}},
 {"function_name":"move_arm_step","arguments":{"direction":"z+"}},
 {"function_name":"move_arm_step","arguments":{"direction":"y-"}},
 {"function_name":"close_trash_can","arguments":{"container_id":"trash_can"}}
]}
這階段不輸出它們的 relation。

4.「垃圾桶先開著，另一隻手順便往上移一下，做完再降回去，最後蓋起來」
輸出：
{"actions":[
 {"function_name":"open_trash_can","arguments":{"container_id":"trash_can"}},
 {"function_name":"move_arm_step","arguments":{"direction":"z+"}},
 {"function_name":"move_arm_step","arguments":{"direction":"z-"}},
 {"function_name":"close_trash_can","arguments":{"container_id":"trash_can"}}
]}
「另一隻手」不是 action argument。

5.「開垃圾桶的同時手往上移」
只輸出：
{"actions":[
 {"function_name":"open_trash_can","arguments":{"container_id":"trash_can"}},
 {"function_name":"move_arm_step","arguments":{"direction":"z+"}}
]}
不要輸出 parallel。

6.「最後讓剛剛移動的手回預設位置」
{"actions":[{"function_name":"move_arm_default","arguments":{}}]}

7.「先開櫃門、放入物件，最後關櫃門」
輸出三個對應 tool actions；不可因句子較長而漏掉最後動作。
"""


OPENAI_ACTION_EXTRACTION_PROMPT = """
Extract every explicit atomic tool action from the complete user request.

Stage 1 answers only:
1. which atomic tool actions are explicitly requested, and
2. the required scene-entity arguments for each selected tool.

Preserve the actions' textual occurrence order, but do not infer or output
dependencies, temporal relations, resource relations, roles, groups, devices,
arm IDs, or concrete device assignments. Stage 2 handles temporal/resource
relationships.

For each action, choose exactly one function_name from the supplied tool catalog
and provide only arguments allowed by that tool's schema. Do not add arguments
that the tool does not define, and do not remove required arguments while
repairing another error.

Include the minimum tool-semantic prerequisite actions required to make the
sequence executable. In particular, place_object(X) releases an object that is
already held. Unless environment_context says X is already held at task start,
place_object(X) requires an earlier pick_object(X). This is tool-level planning
semantics, not Stage 2 temporal/resource reasoning. Do not add unrelated or
domain-specific actions.

For all scene-entity reference arguments such as object_id, source_id,
target_id, destination_id, payload_id, and container_id, use exact entity IDs
already present in environment_context grounding or structured scene data.

Never invent, rename, translate, normalize, or guess an entity ID.

Collection grounding rules:
- environment_context.grounded_targets may contain rows with
  grounding_type="collection_member".
- Rows sharing the same collection_id are one grounded collection.
- If the user explicitly requests an operation over all/every members, expand
  it into the required atomic actions for each member.
- Use each member's exact object_id.
- If the user asks for the corresponding destination and structured scene data
  or grounded member attributes provide expected_container, use that exact
  entity ID as destination_id.
- Negative instructions such as "do not move the correct objects" must not
  generate pick/place/move actions for those entities.
- Collection expansion is a valid one-to-many expansion, not a duplicated
  subplan.
- If a clause only describes inspection/inventory context and the tool catalog
  has no inspect/check tool, do not invent an unsupported action.

Full-request coverage check (mandatory before returning; do not output the
check itself):
- Read the complete user request from beginning to end. Never stop after
  completing only an earlier clause or a collection expansion.
- In textual order, classify every clause as an executable instruction,
  negative constraint, observation/context, or resource description.
- For every executable instruction, inspect the supplied tool catalog and
  determine whether an available tool can perform that intent.
- If a matching tool exists, the actions array must contain at least one action
  covering that executable instruction. Expanding one earlier clause into many
  collection-member actions does not permit omitting a later clause.
- Negative constraints do not generate actions. Observation/context clauses
  without a matching tool do not generate actions. Resource descriptions remain
  for Stage 2.
- Preserve all independent executable instructions even when they use different
  tool types or different execution resources.
- Before returning, verify that every explicit executable instruction that is
  representable by the tool catalog is covered exactly once, except valid
  per-member collection expansion.

Resource/device descriptions are NOT action arguments. Phrases such as
"left hand", "right hand", "one hand", "the other hand", "same arm",
"different arm", "robot_arm", "device", or "resource" describe execution
resource relationships only. Never convert such phrases into object_id,
source_id, target_id, destination_id, payload_id, container_id, or any other
tool argument.

Example: if the user says "use the other hand to open the trash can",
Stage 1 should output only open_trash_can with the grounded trash-can entity ID
as container_id. The phrase "the other hand" remains in the original user
request for Stage 2.

Output only the structured actions array required by the schema. Do not omit
later actions from a long request, do not duplicate actions, and do not invent
actions or argument values. Python will perform strict tool and coverage checks.
"""


class ActionExtractionError(RuntimeError):
    """Stage 1 failed after bounded retries."""


def action_extraction_schema(tool_catalog):
    action_variants = []

    for tool in tool_catalog:
        function_name = tool["api_function"]
        parameters = tool.get("parameters", {})
        required = tool.get("required", [])

        action_variants.append({
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "enum": [function_name],
                },
                "arguments": {
                    "type": "object",
                    "properties": parameters,
                    "required": required,
                    "additionalProperties": False,
                },
            },
            "required": [
                "function_name",
                "arguments",
            ],
            "additionalProperties": False,
        })

    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "oneOf": action_variants,
                },
            },
        },
        "required": ["actions"],
        "additionalProperties": False,
    }


def _known_entity_ids(context: dict[str, Any] | None) -> list[str]:
    """Collect scene entity IDs only for retry guidance."""
    if not isinstance(context, dict):
        return []

    known = set()

    for row in context.get("grounded_targets") or []:
        if not isinstance(row, dict):
            continue
        entity_id = row.get("object_id") or row.get("id")
        if isinstance(entity_id, str) and entity_id.strip():
            known.add(entity_id.strip())

    for key in ("scene_snapshot", "fake_scene_snapshot"):
        snapshot = context.get(key)
        if not isinstance(snapshot, dict):
            continue
        for row in snapshot.get("entities") or []:
            if not isinstance(row, dict):
                continue
            entity_id = row.get("id") or row.get("object_id")
            if isinstance(entity_id, str) and entity_id.strip():
                known.add(entity_id.strip())

    for key in ("yolo", "fake_yolo"):
        yolo = context.get(key)
        if not isinstance(yolo, dict):
            continue
        for row in yolo.get("detections") or []:
            if not isinstance(row, dict):
                continue
            entity_id = row.get("object_id") or row.get("id")
            if isinstance(entity_id, str) and entity_id.strip():
                known.add(entity_id.strip())

    return sorted(known)


def _retry_instruction(
    error: Exception,
    *,
    context: dict[str, Any] | None = None,
) -> str:
    known_ids = _known_entity_ids(context)
    known_text = (
        f"目前可用的 grounded/scene entity IDs：{known_ids}。"
        if known_ids
        else ""
    )

    if isinstance(error, SemanticIRError):
        if error.code == "MISSING_EXPLICIT_ACTION":
            return (
                f"上一份 atomic actions 不完整：{error}。請重新從頭到尾閱讀完整原始需求，"
                "逐句／逐子句做 coverage 自檢：對每個可由 tool_catalog 執行的明確動作意圖，"
                "都必須輸出對應 atomic action。不要因前面某個 collection 已展開成多個 "
                "actions 就漏掉後續獨立動作。negative constraint、沒有對應 tool 的純 "
                "observation/context 不產生 action；resource/device 關係留給 Stage 2。"
                "不要輸出 relations/groups/devices，也不要新增使用者未要求的替代動作。"
                f"{known_text}"
            )

        if error.code == "DUPLICATED_SUBPLAN":
            return (
                "上一份把完整任務重複抽取；每個要求只輸出一次，"
                "請重新抽取完整 actions。"
            )

        if error.code == "UNKNOWN_ARGUMENT":
            return (
                f"上一份 action 使用了 tool schema 不允許的參數：{error}。"
                "請刪除未授權參數，但保留該 tool 的所有 required arguments。"
                "手臂、左右手、另一隻手、robot_arm、device、resource "
                "都不是 scene entity argument，不可塞進任何 tool arguments。"
                "只使用 tool_catalog 定義的參數名稱。"
                f"{known_text}"
            )

        if error.code == "MISSING_ARGUMENT":
            return (
                f"上一份 action 缺少 tool 的 required argument：{error}。"
                "請依 tool_catalog 補回缺少的 required argument；"
                "scene entity reference 必須使用 environment_context 中既有的 exact entity ID。"
                "不要因為刪除錯誤的 arm/device/resource 參數，就把正確的 "
                "object_id/container_id/source_id/target_id/destination_id/payload_id 一起刪掉。"
                f"{known_text}"
            )

        if error.code == "UNGROUNDED_OBJECT_ID":
            return (
                f"上一份使用了不存在的 scene entity ID：{error}。"
                "所有 entity reference arguments 必須逐字使用 environment_context "
                "中既有的 entity ID；禁止自行創造或改名。"
                f"{known_text}"
            )


        if error.code == "UNSATISFIED_ACTION_PRECONDITION":
            return (
                f"上一份 actions 有未滿足的 tool prerequisite：{error}。"
                "請重新輸出完整 atomic actions，補齊可執行所需的最低限度前置 action。"
                "place_object(X) 只能放下已持有的 X；若 environment_context 未表示 "
                "X 在 task 開始時已持有，則在 place_object(X) 前加入 pick_object(X)。"
                "不要改寫 grounded entity IDs，不要加入與原始任務無關的動作。"
                f"{known_text}"
            )

    return (
        f"上一份 atomic action extraction 無效：{error}。"
        "請重新從頭到尾閱讀完整需求並只輸出完整 actions。"
        "逐句確認所有可由 tool_catalog 執行的明確動作都被 coverage；"
        "不可因前一個 collection expansion 產生多個 actions 就漏掉後續子句。"
        "不要輸出 temporal/resource relations、devices 或 arm IDs；"
        "手臂/設備描述不是 tool arguments。"
        "若 grounded_targets 含 collection_member，請依原始需求逐 member 展開；"
        "若 member 有 expected_container 且需求要求對應目的地，"
        "使用該 exact entity ID。"
        f"{known_text}"
    )


def extract_atomic_actions(
    user_text: str,
    tool_catalog: list[dict[str, Any]],
    *,
    context: dict[str, Any] | None = None,
    allow_duplicated_subplan: bool = False,
    max_attempts: int = 3,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    user_prompt = dedent(f"""
        tool_catalog:
        {json.dumps(tool_catalog, ensure_ascii=False, indent=2)}

        environment_context:
        {json.dumps(context or {}, ensure_ascii=False, indent=2)}

        完整使用者需求:
        {user_text}
    """).strip()

    selected_provider = resolve_provider("stage1", provider)
    system_prompt = (
        OPENAI_ACTION_EXTRACTION_PROMPT
        if selected_provider == "openai"
        else ABSTRACT_ACTION_EXTRACTION_PROMPT
    )

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    messages = list(base_messages)
    schema = action_extraction_schema(tool_catalog)
    failures = []
    last_error = None

    for attempt in range(max_attempts):
        try:
            message = chat_structured(
                stage="stage1",
                messages=messages,
                response_schema=schema,
                provider=provider,
                model=model,
                api_key=api_key,
            )
        except StructuredProviderError as exc:
            raise ActionExtractionError(str(exc)) from exc

        content = message.get("content") if isinstance(message, dict) else None

        try:
            if not isinstance(content, str) or not content.strip():
                raise SemanticIRError(
                    "INVALID_LLM_OUTPUT",
                    "Stage 1 缺少 content",
                )

            parsed = json.loads(content)

            actions = validate_atomic_action_output(
                parsed,
                tool_catalog,
                allow_duplicated_subplan=allow_duplicated_subplan,
            )

            validate_grounded_action_targets(
                actions,
                context or {},
            )

            validate_manipulation_preconditions(
                actions,
                context or {},
            )

            validate_semantic_action_coverage(
                user_text,
                {"actions": actions},
                tool_catalog,
            )

            return actions

        except (json.JSONDecodeError, SemanticIRError) as exc:
            last_error = exc
            failures.append(
                (attempt + 1, str(exc), content)
            )
            messages = [
                *base_messages,
                {
                    "role": "user",
                    "content": _retry_instruction(
                        exc,
                        context=context,
                    ),
                },
            ]

    details = "\n\n".join(
        (
            f"===== ACTION EXTRACTION ATTEMPT {number} =====\n"
            f"validator_error: {error}\n"
            f"raw_content:\n{content}"
        )
        for number, error, content in failures
    )

    raise ActionExtractionError(
        f"Stage 1 無法抽出完整 actions：{last_error}\n\n{details}"
    )

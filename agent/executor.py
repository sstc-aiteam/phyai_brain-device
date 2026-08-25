"""
AI Agent 工具序列執行層。
"""

import logging
from threading import Lock, Thread

from agent.tools import get_tool
from services import arm_service, task_service


logger = logging.getLogger(__name__)

_EXECUTION_LOCK = Lock()


class ToolExecutionError(RuntimeError):
    """
    Agent 工具執行錯誤。
    """


TOOL_EXECUTORS = {
    "move_arm_default": arm_service.move_arm_default,
    "move_arm_step": arm_service.move_arm_step,

    "place_object_in_trash_can":
        task_service.place_object_in_trash_can,

    "place_object_in_top_cabinet":
        task_service.place_object_in_top_cabinet,

    "place_object_in_second_drawer":
        task_service.place_object_in_second_drawer,

    "open_trash_can":
        task_service.open_trash_can,

    "close_trash_can":
        task_service.close_trash_can,

    "open_top_cabinet":
        task_service.open_top_cabinet_door,

    "close_top_cabinet":
        task_service.close_top_cabinet,

    "open_top_cabinet_and_place_object":
        task_service.open_top_cabinet_door_and_place_object,

    # 對 AI 保留簡單且穩定的工具名稱；實際動作一律使用兩段式流程。
    "open_second_drawer":
        task_service.open_second_drawer_2_phase,

    "close_second_drawer":
        task_service.close_second_drawer,

    "open_second_drawer_and_place_object":
        task_service.open_second_drawer_and_place_object,
}


def _validate_tool_calls(tool_calls):
    """
    在啟動背景執行緒之前，先驗證整個工具序列。

    這樣可以避免執行到一半才發現後續工具不存在。
    """
    if not isinstance(tool_calls, list):
        raise ToolExecutionError("tool_calls 必須是 list")

    if not tool_calls:
        raise ToolExecutionError("tool_calls 不可為空")

    if len(tool_calls) > 20:
        raise ToolExecutionError("tool_calls 數量不可超過 20")

    validated_calls = []

    for index, tool_call in enumerate(tool_calls):
        step_number = index + 1

        if not isinstance(tool_call, dict):
            raise ToolExecutionError(
                f"第 {step_number} 個 tool_call 必須是 dict"
            )

        function_name = tool_call.get("function_name")
        arguments = tool_call.get("arguments")

        if not isinstance(function_name, str):
            raise ToolExecutionError(
                f"第 {step_number} 個 function_name 必須是字串"
            )

        function_name = function_name.strip()

        if not function_name:
            raise ToolExecutionError(
                f"第 {step_number} 個 function_name 不可為空"
            )

        if not isinstance(arguments, dict):
            raise ToolExecutionError(
                f"第 {step_number} 個 arguments 必須是 dict"
            )

        tool = get_tool(function_name)

        if not isinstance(tool, dict):
            raise ToolExecutionError(
                f"第 {step_number} 個工具未授權：{function_name}"
            )

        executor = TOOL_EXECUTORS.get(function_name)

        if not callable(executor):
            raise ToolExecutionError(
                f"第 {step_number} 個工具尚未設定執行函式："
                f"{function_name}"
            )

        validated_calls.append({
            "function_name": function_name,
            "arguments": arguments.copy(),
            "executor": executor,
        })

    return validated_calls


def _tool_result_failed(result):
    """
    判斷 service 回傳是否明確表示失敗。

    支援目前常見的兩種回傳：
    - status == "error"
    - result == False
    """
    if result.get("status") == "error":
        return True

    if result.get("result") is False:
        return True

    return False


def _run_tool_sequence(validated_calls):
    """
    在同一條背景執行緒中，依序執行所有工具。

    任一步驟失敗時立即停止，不執行後續步驟。
    """
    try:
        total_steps = len(validated_calls)

        logger.info(
            "Agent 工具序列開始執行：step_count=%s",
            total_steps,
        )

        for index, tool_call in enumerate(validated_calls):
            step_number = index + 1
            function_name = tool_call["function_name"]
            arguments = tool_call["arguments"]
            executor = tool_call["executor"]

            logger.info(
                "Agent 工具開始執行：step=%s/%s, "
                "function_name=%s, arguments=%s",
                step_number,
                total_steps,
                function_name,
                arguments,
            )

            try:
                result = executor(**arguments)
            except TypeError as exc:
                raise ToolExecutionError(
                    f"第 {step_number} 步工具參數錯誤："
                    f"{function_name}：{exc}"
                ) from exc
            except Exception as exc:
                raise ToolExecutionError(
                    f"第 {step_number} 步執行失敗："
                    f"{function_name}：{exc}"
                ) from exc

            if not isinstance(result, dict):
                raise ToolExecutionError(
                    f"第 {step_number} 步 {function_name} "
                    "回傳結果必須是 dict"
                )

            if _tool_result_failed(result):
                message = result.get("message")

                if not isinstance(message, str) or not message.strip():
                    message = "工具回傳執行失敗"

                raise ToolExecutionError(
                    f"第 {step_number} 步 {function_name} "
                    f"執行失敗：{message}"
                )

            logger.info(
                "Agent 工具執行完成：step=%s/%s, "
                "function_name=%s",
                step_number,
                total_steps,
                function_name,
            )

        logger.info(
            "Agent 工具序列全部完成：step_count=%s",
            total_steps,
        )

    except Exception:
        logger.exception("Agent 工具序列執行中止")

    finally:
        _EXECUTION_LOCK.release()


def execute_tools(tool_calls):
    """
    啟動背景執行緒，依序執行工具列表。

    此函式只負責啟動，不等待整個任務完成。
    """
    validated_calls = _validate_tool_calls(tool_calls)

    if not _EXECUTION_LOCK.acquire(blocking=False):
        raise ToolExecutionError(
            "目前已有 Agent 工具序列正在執行"
        )

    try:
        thread = Thread(
            target=_run_tool_sequence,
            args=(validated_calls,),
            daemon=False,
            name="agent-tool-sequence",
        )

        thread.start()

    except Exception:
        _EXECUTION_LOCK.release()
        raise

    return {
        "started": True,
        "step_count": len(validated_calls),
        "function_names": [
            tool_call["function_name"]
            for tool_call in validated_calls
        ],
    }

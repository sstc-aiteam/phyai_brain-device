from __future__ import annotations

import json
import re
from typing import Any

from services import llm_service

from .coordination_schema import CoordinationSpec
from .critical_state import build_critical_state
from .dispatch_schema import ActionIntent, DispatchDecision
from .goal_schema import GoalSet
from .tool_model import ToolCatalog
from .world_state import WorldState


SYSTEM_PROMPT = """
You are the runtime decision brain of a CLOSED-LOOP robot.

Choose exactly ONE DISPATCH for the CURRENT world state.

A dispatch contains one or more atomic actions that may start concurrently NOW.
You are NOT given candidate actions. Select tools yourself from available_tools.

IMPORTANT:
- Do NOT create a full future plan.
- Do NOT choose concrete device IDs. Device allocation happens after you.
- For entity-reference arguments, use exact entity IDs from world_state.
- For scalar/enum arguments, use only values explicitly listed by that tool.
- Read critical_state FIRST.
- Resource commitments are PER DEVICE, not a global lock.
- A committed device must continue, complete, or release its own commitment
  safely before that same device starts unrelated work.
- Another free compatible device may still perform independent work.

DECISION PRIORITY:
1. Preserve/resolve CURRENT physical commitments first.
2. Inspect repair_context BEFORE selecting new task work.
   - If active_frame.status == "READY", this is YOUR OWN previously blocked
     intent whose reported physical preconditions are now repaired. Either:
       A) continue that blocked_action now, or
       B) explicitly abandon it with a reason.
     Do not silently ignore a READY pending intent.
   - If active_frame.status == "BLOCKED", continue establishing its missing
     physical facts one atomic step at a time.
3. Inspect loop_context. If a recent semantic loop exists with no task credit,
   do not repeat the same state/action cycle unless it is required by the
   active physical repair.
4. Otherwise inspect goal_state together with task_progress_state. A physically
   satisfied goal that is NOT task-credited is still pending task work. Choose
   a pending goal whose credited dependencies are satisfied.
5. Use available_tools effects to identify an action that advances that goal
   or repairs the active blocked intent.
6. Check the chosen tool's CURRENT preconditions before answering.
7. If that action is not executable yet, choose one atomic action whose effects
   establish a missing precondition.
8. Ignore unrelated ambient facts and already-satisfied goals.
9. Trust CURRENT world_state only; do not assume an earlier action succeeded.
10. New observations do not erase existing physical commitments.

PARALLELISM:
- Prefer safe parallelism only when multiple independent actions can start NOW
  on distinct compatible devices.
- If action B needs an effect of action A, they are NOT parallel.
- Do not place sequential prerequisite steps in the same dispatch.
- max_parallel_actions is an upper bound, NOT a request to fill every slot.
- When only one useful/feasible action should start now, return exactly one.

PARTIAL-SUCCESS RECOVERY:
- A previous parallel dispatch may have only partially succeeded.
- Treat sibling results independently through CURRENT world_state.
- Never assume a failed sibling rolled back a successful sibling.
- If an entity is already held/placed/open/reported/etc. in CURRENT world_state,
  do not repeat an action whose precondition assumes the old state.
- Resolve remaining obligations from CURRENT world_state.
- If max_parallel_actions is temporarily 1, return exactly one useful atomic
  action. Do not encode multiple sequential steps into one action.

TASK FEEDBACK:
- Hard rejection means the proposal was structurally/physically/resource/safety
  invalid. A legal but poor task choice is normally EXECUTED instead.
- Read task_progress_state and last_execution_feedback after every execution.
- If last_execution_feedback.progress_class is "execution_failure", the command
  did not successfully change World. Re-plan from CURRENT world_state; do not
  treat actuator failure as proof that the intended task choice was wrong.
- If an action changed World but earned no task credit, reconsider whether that
  change actually helps the current goals.
- If a goal is physically satisfied but listed as uncredited, it was achieved
  out of required dependency order. Perform the milestone again only after its
  credited dependencies are satisfied.
- If goals regressed, repair them from CURRENT world_state.

PERSISTENT REPAIR INTENT:
- repair_context persists across World iterations. It records YOUR OWN earlier
  physically rejected action intentions and their CURRENT-vs-REQUIRED facts.
- repair_context.active_frame is the most recent blocked intent.
- A successful prerequisite action does NOT erase that parent intent.
- If active_frame.remaining_state_differences is non-empty and the blocked
  action is still relevant to the current task, continue establishing those
  facts one atomic step at a time.
- If active_frame.status is "READY", the pending action has regained all
  reported physical preconditions. Continue that SAME blocked_action now unless
  CURRENT goals/world make it genuinely irrelevant.
- If you intentionally change your mind, set:
    "pending_intent_resolution": "abandon"
  and provide a non-empty "pending_intent_abandon_reason".
- If you continue the READY intent, set:
    "pending_intent_resolution": "continue"
  and include the blocked_action in actions.
- Nested repair frames may exist. Resolve the top frame first; once its blocked
  action succeeds, the parent frame becomes active again.
- You MAY abandon a stale repair intent when CURRENT goals/world clearly make
  it irrelevant. The runtime never forces the blocked action.
- repair_context is memory, NOT a candidate-action list and NOT an answer key.

HARD-REJECTION REPAIR:
- rejected_dispatches may contain structured_feedback.state_differences.
- Each state difference tells you ONLY what fact was required versus what the
  CURRENT world had. It is not a candidate action and it does not tell you the
  answer.
- For each state difference, reason backward from the required fact:
    * identify which available tool EFFECT can establish that semantic fact;
    * check that repair tool's own CURRENT preconditions;
    * if the repair tool is not executable yet, recursively establish its
      missing preconditions one atomic step at a time.
- Examples of the reasoning pattern, without assuming any specific tool name:
    * subject.location current=A required=B
      -> choose a tool whose effect can make subject.location become B.
    * object.held_by current=None required=<device>
      -> choose a tool whose effect establishes object ownership/holding.
    * container.open_state current=closed required=open
      -> choose a tool whose effect establishes open_state=open.
- Do NOT simply retry the rejected action until its reported state differences
  have actually been repaired in CURRENT world_state.
- Do NOT invent actions that are not in available_tools.

LOOP MEMORY:
- loop_context summarizes RECENT EXECUTED actions and semantic World revisits.
- loop_context never blocks an action; it is evidence from your own behavior.
- If loop_detected=true and no task credit occurred in the cycle, do not repeat
  the same cycle. Choose a materially different action that advances a goal,
  repairs a physical prerequisite, resumes a READY pending intent, or explicitly
  abandons an obsolete pending intent.

RETRY:
- Respect goal dependencies and hard coordination constraints.
- Never repeat an identical HARD-REJECTED dispatch while world_state is unchanged.
- Prefer repairing an explicit structured state difference over guessing a new
  unrelated task action.
- Read every hard rejection reason and choose a materially different CURRENTLY
  executable dispatch.

Return JSON only:
{
  "pending_intent_resolution": "continue | abandon | null",
  "pending_intent_abandon_reason": "<required only for abandon>",
  "actions": [
    {
      "function_name": "<tool>",
      "arguments": {
        "<argument>": "<exact entity id or allowed scalar value>"
      }
    }
  ]
}

When no pending intent needs explicit resolution, omit the two
pending_intent_* fields or use null.
""".strip()


def _extract_json_object(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError(
                f"Runtime brain did not return JSON: {content!r}"
            )
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise RuntimeError("Runtime brain JSON root must be object")
    return value


class RuntimeBrain:
    def __init__(
        self,
        *,
        model: str = "qwen3-vl:8b-local",
        temperature: float = 0.15,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def decide_dispatch(
        self,
        *,
        world: WorldState,
        goals: GoalSet,
        tools: ToolCatalog,
        coordination: CoordinationSpec | None = None,
        device_context: Any = None,
        rejected_dispatches: list[dict[str, Any]] | None = None,
        task_progress_state: dict[str, Any] | None = None,
        repair_context: dict[str, Any] | None = None,
        loop_context: dict[str, Any] | None = None,
        max_parallel_actions: int = 2,
    ) -> DispatchDecision:
        coordination = coordination or CoordinationSpec()

        payload = {
            "critical_state": build_critical_state(
                world,
                goals,
                coordination,
            ),
            "world_state": world.snapshot(),
            "goal_state": goals.summary(world),
            "coordination_constraints": coordination.brain_view(
                world,
                goals,
            ),
            "device_context": device_context,
            "max_parallel_actions": max_parallel_actions,
            "available_tools": tools.brain_view(),
            "rejected_dispatches": rejected_dispatches or [],
            "task_progress_state": task_progress_state or {},
            "repair_context": repair_context or {},
            "loop_context": loop_context or {},
        }

        message = llm_service.chat(
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            wait=True,
            owner="runtime_brain",
        )

        content = (
            message.get("content")
            if isinstance(message, dict)
            else str(message or "")
        )
        parsed = _extract_json_object(content)

        raw_actions = parsed.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise RuntimeError(
                f"Runtime brain missing non-empty actions: {parsed!r}"
            )
        if len(raw_actions) > max_parallel_actions:
            raise RuntimeError(
                f"Runtime brain returned {len(raw_actions)} actions, "
                f"max is {max_parallel_actions}"
            )

        actions: list[ActionIntent] = []
        for raw in raw_actions:
            if not isinstance(raw, dict):
                raise RuntimeError("Each dispatch action must be object")
            function_name = raw.get("function_name")
            arguments = raw.get("arguments")
            if not isinstance(function_name, str) or not function_name:
                raise RuntimeError("action.function_name must be string")
            if not isinstance(arguments, dict):
                raise RuntimeError("action.arguments must be object")
            normalized: dict[str, str] = {}
            for key, value in arguments.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise RuntimeError(
                        "action arguments must map string -> string values"
                    )
                normalized[key] = value
            tools.get(function_name)  # unknown tool -> deterministic error
            actions.append(ActionIntent(function_name, normalized))

        resolution = parsed.get(
            "pending_intent_resolution"
        )
        if resolution is not None:
            if not isinstance(resolution, str):
                raise RuntimeError(
                    "pending_intent_resolution must be "
                    "string or null"
                )
            resolution = resolution.strip().lower()
            if resolution in {"", "null", "none"}:
                resolution = None

        abandon_reason = parsed.get(
            "pending_intent_abandon_reason"
        )
        if abandon_reason is not None and not isinstance(
            abandon_reason,
            str,
        ):
            raise RuntimeError(
                "pending_intent_abandon_reason must be "
                "string or null"
            )

        return DispatchDecision(
            tuple(actions),
            pending_intent_resolution=resolution,
            pending_intent_abandon_reason=abandon_reason,
        )

    def decide_next_action(
        self,
        *,
        world: WorldState,
        goals: GoalSet,
        tools: ToolCatalog,
        coordination: CoordinationSpec | None = None,
        device_context: Any = None,
        rejected_dispatches: list[dict[str, Any]] | None = None,
    ) -> ActionIntent:
        """Backward-compatible single-action mode for old regression tasks."""
        dispatch = self.decide_dispatch(
            world=world,
            goals=goals,
            tools=tools,
            coordination=coordination,
            device_context=device_context,
            rejected_dispatches=rejected_dispatches,
            task_progress_state=None,
            repair_context=None,
            loop_context=None,
            max_parallel_actions=1,
        )
        return dispatch.actions[0]

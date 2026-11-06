
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import tempfile
import json
from datetime import datetime, timezone
from threading import Condition, Lock, RLock, Thread
import os
from uuid import uuid4

from plan.full_task.registry import all_tools
from plan.full_task.router import expected_tool_routes, preflight_service_names
from plan.full_task.runner import run_workflow
from agent.answer_generator import answer_from_observation
from services import plan_service


_JOBS = {}
_JOBS_LOCK = RLock()
_FLIGHT_LOCK = Lock()  # agent API single-flight within this Python process


class AgentBusy(RuntimeError):
    """Another Workflow is already active."""


_MAX_JOBS = 64
_TERMINAL = frozenset(("planned", "answered", "completed", "dispatched_unverified", "stopped", "failed"))  # WORKFLOW_AUTO_DISPATCH_V1


def _now():
    return datetime.now(timezone.utc).isoformat()


_JOB_ARCHIVE_DIR = Path.home() / "robot"/ "robot_agent_logs" / "workflow"


def _save_job_file(job_id, payload):
    """Atomically write final full job JSON; no additional HTTP endpoint."""
    directory = _JOB_ARCHIVE_DIR
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    target = directory / (job_id + ".json")
    data = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    fd, temporary = tempfile.mkstemp(prefix="." + job_id + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(target)


class WorkflowJob:
    def __init__(self, *, user_text, provider, camera_name, mode, check, model):
        self.job_id = uuid4().hex
        self.user_text = user_text
        self.provider = provider
        self.camera_name = camera_name
        self.mode = mode
        self.check = check
        self.model = model
        self.created_at = _now()
        self.updated_at = self.created_at
        self.status = "queued"
        self.plan = None
        self.capability = None
        self.output = None
        self.tool_routes = None
        self.preflight = None
        self.result = None
        self.error = None
        self._condition = Condition(RLock())
        self._events = []
        self._event_seq = 0
        self._stream_done = False
        self.log_file = None
        self._step_progress = {}
        self._current_step = None
        self._completed_steps = 0
        self._emit("job_created", job_id=self.job_id, mode=self.mode,
                   user_text=self.user_text)

    def _set(self, **values):
        with self._condition:
            for field, value in values.items():
                setattr(self, field, value)
            self.updated_at = _now()
            self._condition.notify_all()

    def _emit(self, kind, **fields):
        """Buffered progress events, archived once when the Job finishes."""
        with self._condition:
            self._event_seq += 1
            event = {"seq": self._event_seq, "time": _now(),
                     "type": kind, **fields}
            self._events.append(deepcopy(event))
            # Bounded history. Normal jobs have far fewer than 512 events.
            if len(self._events) > 512:
                del self._events[:-512]
            self._condition.notify_all()

    def on_step_progress(self, event, **fields):
        """Runner observer: an error in reporting must not alter robot execution."""
        if event not in ("step_start", "step_end", "step_update"):
            return
        step_id = fields.get("step_id")
        if not isinstance(step_id, str):
            return
        with self._condition:
            summary = self._step_progress.setdefault(step_id, {
                "step_id": step_id,
                "tool_name": fields.get("tool_name"),
            })
            if event == "step_start":
                summary["status"] = "running"
                self._current_step = step_id
            else:
                summary["status"] = fields.get("status", "unknown")
                if "elapsed_seconds" in fields:
                    summary["elapsed_seconds"] = fields["elapsed_seconds"]
                if fields.get("error"):
                    summary["error"] = str(fields["error"])[:300]
                if event == "step_end":
                    if self._current_step == step_id:
                        self._current_step = None
                    self._completed_steps = sum(
                        item.get("status") not in ("running", None)
                        for item in self._step_progress.values()
                    )
            self.updated_at = _now()
        self._emit(event, **fields)

    def finish_stream(self):
        """Persist final result before announcing that this Job has finished.

        A disk-write error must not change the physical execution result.
        """
        path = str(_JOB_ARCHIVE_DIR / (self.job_id + ".json"))
        with self._condition:
            event = {"seq": self._event_seq + 1, "time": _now(),
                     "type": "job_finished", "status": self.status,
                     "error": deepcopy(self.error), "log_file": path}
            document = self.snapshot(detail="full")
            document["log_file"] = path
            document["events"].append(deepcopy(event))
        archive_error = None
        try:
            _save_job_file(self.job_id, document)
        except Exception as exc:
            archive_error = f"{type(exc).__name__}: {exc}"[:250]
        with self._condition:
            if archive_error is None:
                self.log_file = path
            else:
                event.pop("log_file", None)
                event["log_error"] = archive_error
            self._event_seq += 1
            event["seq"] = self._event_seq
            self._events.append(event)
            self._stream_done = True
            self._condition.notify_all()

    def iter_events(self, *, after=0, heartbeat_seconds=8.0):
        """Replay earlier events, then yield new events until task termination.

        Closing the HTTP connection only stops this generator, NOT the job.
        """
        cursor = after
        while True:
            with self._condition:
                ready = [deepcopy(e) for e in self._events if e["seq"] > cursor]
                done = self._stream_done
                if not ready and not done:
                    self._condition.wait(timeout=heartbeat_seconds)
                    ready = [deepcopy(e) for e in self._events if e["seq"] > cursor]
                    done = self._stream_done
            if ready:
                for event in ready:
                    cursor = event["seq"]
                    yield event
            elif done:
                return
            else:
                yield {"type": "heartbeat", "time": _now(),
                       "status": self.status}

    def wait_until_finished(self, *, detail="full"):
        """Block until worker finishes; no polling, no change to execution."""
        with self._condition:
            self._condition.wait_for(lambda: self._stream_done)
            return self.snapshot(detail=detail)

    def snapshot(self, *, detail="compact"):
        """Small default response; full detail remains available for debugging."""
        if detail not in ("compact", "full"):
            raise ValueError("detail must be compact/full")
        with self._condition:
            data = {
                "job_id": self.job_id,
                "status": self.status,
                "user_text": self.user_text,
                "provider": self.provider,
                "camera_name": self.camera_name,
                "mode": self.mode,
                "check": self.check,
                "error": deepcopy(self.error),  # Explicitly requested even when null.
                "process_local": True,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
            # Omit fields that are not yet available; don't emit useless nulls.
            if self.log_file is not None:
                data["log_file"] = self.log_file
            if self.capability is not None:
                data["capability"] = deepcopy(self.capability)
            if self.output is not None:
                data["output"] = deepcopy(self.output)
            if self.plan is not None:
                data["plan"] = deepcopy(self.plan)
                progress = {
                    "total_steps": len(self.plan.get("steps", [])),
                    "finished_steps": self._completed_steps,
                    "steps": deepcopy(list(self._step_progress.values())),
                }
                if self._current_step is not None:
                    progress["current_step"] = self._current_step
                data["progress"] = progress
            if self.result is not None:
                if detail == "full":
                    data["result"] = deepcopy(self.result)
                else:
                    result = self.result
                    data["result"] = {k: deepcopy(result[k]) for k in (
                        "success", "stopped_reason", "physical_task_completed",
                        "completion_evidence", "unverified_steps", "note"
                    ) if k in result}
            if detail == "full":
                if self.tool_routes is not None:
                    data["tool_routes"] = deepcopy(self.tool_routes)
                if self.preflight is not None:
                    data["preflight"] = deepcopy(self.preflight)
                data["events"] = deepcopy(self._events)
            return data

# === AGENT_CAPABILITY_ROUTING_V1 ===
def _answer_without_tools(job):
    """Only for selector='none': language response, no tools or observations."""
    import json
    from utils.llm.structured import chat_structured

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    answer = chat_structured(
        stage="agent_general_text_reply",
        provider=job.provider,
        model=job.model,
        messages=[
            {"role": "system", "content": (
                "你是實體機器人系統的 AI Agent 軟體助手。"
                "這一輪沒有選到可執行的註冊能力，因此只能回答一般文字問題。"
                "可以說明自己是 Agent、一般概念與使用方式，但不可宣稱已觀察相機畫面、"
                "設備狀態或完成任何動作，不可編造即時資訊或未註冊能力。"
                "若問題需要即時觀測、外部資料、操作設備或缺少可靠資訊，應明確告知限制。"
                "使用者使用繁體中文時，以繁體中文簡潔回答。"
            )},
            {"role": "user", "content": job.user_text},
        ],
        response_schema=schema,
        temperature=0.0,
        max_tokens=512,
        num_ctx=4096,
        timeout=90.0,
        owner="agent_general_text_reply",
    )
    content = answer.get("content") if isinstance(answer, dict) else None
    result = json.loads(content) if isinstance(content, str) else content
    if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
        raise RuntimeError("Agent text reply schema invalid")
    text = result["answer"].strip()
    if not text:
        raise RuntimeError("Agent text reply is empty")
    return text


def _observation_preview(service_response, limit=1800):
    """Bound text-stream volume. Full raw Service output remains in GET job."""
    import json
    value = service_response.get("data", service_response)
    try:
        preview = json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                             default=str)
    except (ValueError, TypeError):
        preview = str(type(value).__name__)
    if len(preview) > limit:
        return preview[:limit] + "... [truncated; use GET /workflow/<job_id>?detail=full]"
    return preview


def _worker(job):
    try:
        job._set(status="planning")
        job._emit("status", status="planning")
        from agent.capability_selector import select_capability
        kwargs = {"provider": job.provider}
        if job.model is not None:
            kwargs["model"] = job.model
        selected = select_capability(job.user_text, **kwargs)
        if not isinstance(selected, dict):
            raise RuntimeError("Capability Selector returned invalid data")
        capability_name = selected.get("tool_name")
        if not isinstance(capability_name, str):
            raise RuntimeError("Capability Selector returned invalid tool_name")
        # AGENT_PROVIDER_BRIDGE_V1: honor request provider in registered VLM tools.
        if capability_name == "describe_scene":
            selected_arguments = selected.get("arguments")
            if not isinstance(selected_arguments, dict):
                raise RuntimeError("describe_scene capability arguments must be an object")
            selected = {**selected, "arguments": {
                **selected_arguments, "vlm_backend": job.provider,
            }}
        job._set(capability={
            "tool_name": capability_name,
            "arguments": deepcopy(selected.get("arguments", {})),
            "reason": selected.get("reason", ""),
        })
        job._emit("capability_selected", tool_name=capability_name,
                  reason=selected.get("reason", ""))

        if capability_name == "none":
            # 'none' is a deliberate no-tool decision, NOT a Planner failure.
            # Never call hardware or fabricate observations in this branch.
            job._set(status="answering")
            job._emit("status", status="answering")
            reply = _answer_without_tools(job)
            job._set(output={"answer": reply}, status="answered")
            job._emit("capability_result", tool_name="none", answer=reply)
            job._emit("status", status="answered")
            return

        if capability_name != "plan_task":
            # Service permissions, arguments and side effects are checked again
            # by the existing Executor using the authoritative Registry.
            from agent.capability_executor import execute_capability
            job._set(status="observing")
            job._emit("status", status="observing")
            executed = execute_capability(selected)
            if executed.get("executed") is not True:
                raise RuntimeError("Capability Executor did not run a registered query")
            service_response = executed["service_response"]

            # Preserve the successful observation before generating an answer.
            job._set(
                status="answering",
                output={
                    "tool_name": capability_name,
                    "service_response": service_response,
                },
            )

            job._emit(
                "capability_result",
                tool_name=capability_name,
                summary=_observation_preview(service_response),
            )

            job._emit("status", status="answering")

            try:
                answer = answer_from_observation(
                    user_text=job.user_text,
                    capability_name=capability_name,
                    service_response=service_response,
                    provider=job.provider,
                    model=job.model,
                )

            except Exception as exc:
                # Answer failure must not invalidate a successful observation.
                job._set(
                    output={
                        "tool_name": capability_name,
                        "service_response": service_response,
                        "answer": None,
                        "answer_error": {
                            "type": type(exc).__name__,
                            "message": str(exc)[:300],
                        },
                    },
                    status="answered",
                )

                job._emit(
                    "answer_error",
                    error=f"{type(exc).__name__}: {exc}"[:300],
                )

            else:
                job._set(
                    output={
                        "answer": answer,
                        "tool_name": capability_name,
                        "service_response": service_response,
                    },
                    status="answered",
                )

                job._emit("answer", text=answer)

            job._emit("status", status="answered")
            return

        if selected.get("arguments") != {}:
            raise RuntimeError("plan_task requires empty arguments")
        job._emit("planning_full_task")
        plan = plan_service.plan_full_task(job.user_text, provider=job.provider,
                              camera_name=job.camera_name, model=job.model)
        catalog = all_tools()
        routes = expected_tool_routes(plan, catalog)
        job._set(plan=plan, tool_routes=routes)
        job._emit("plan", plan_id=plan.get("plan_id"),
                  reason=plan.get("reason", ""),
                  steps=[{"step_id": step["step_id"],
                          "tool_name": step["tool_name"],
                          "arguments": deepcopy(step["arguments"])}
                         for step in plan["steps"]])
        if job.mode == "plan":
            job._set(status="planned")
            job._emit("status", status="planned")
            return
        # Existing preflight and Action Gateway checks stay intact.
        preflight = preflight_service_names(plan, catalog)
        job._set(preflight=preflight, status="running")
        job._emit("status", status="running")
        result = run_workflow(plan, mode="real",
                              progress_callback=job.on_step_progress)
        if not isinstance(result, dict) or type(result.get("success")) is not bool:
            raise RuntimeError("Workflow Runner returned an invalid report")
        final_status = ("stopped" if not result["success"] else
                        "dispatched_unverified" if result.get("completion_evidence") == "dispatch_only_unverified" else
                        "completed")
        job._set(result=result, status=final_status)
        job._emit("status", status=final_status,
                  stopped_reason=result.get("stopped_reason"),
                  completion_evidence=result.get("completion_evidence"))
    except Exception as exc:
        job._set(status="failed", error={"type": type(exc).__name__,
                                          "message": str(exc)[:500]})
        job._emit("status", status="failed", error=deepcopy(job.error))
    finally:
        job.finish_stream()
        _FLIGHT_LOCK.release()


def create_workflow(*, user_text, provider="local", camera_name="left",
                    mode="plan", check=False, model=None):
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a nonempty string")
    if len(user_text) > 4000:
        raise ValueError("user_text exceeds 4000 characters")
    if provider not in ("openai", "local"):
        raise ValueError("provider must be openai/local")
    if camera_name not in ("left", "right", "middle"):
        raise ValueError("camera_name must be left/right/middle")
    if mode not in ("plan", "real"):
        raise ValueError("mode must be plan/real")
    if type(check) is not bool:
        raise ValueError("check must be boolean")
    if check:
        raise ValueError("check=true 已停用：/workflow/<job_id>/decision 已移除")
    if model is not None and (not isinstance(model, str) or not 1 <= len(model) <= 120):
        raise ValueError("model must be a string of 1..120 characters")
    if not _FLIGHT_LOCK.acquire(blocking=False):
        raise AgentBusy("已有另一筆 Agent Workflow 正在執行或等待人工確認")
    try:
        job = _register_job(user_text.strip(), provider, camera_name, mode, check, model)
        Thread(target=_worker, args=(job,),
               name="agent-workflow-" + job.job_id[:8], daemon=True).start()
    except BaseException:
        if "job" in locals():
            with _JOBS_LOCK:
                _JOBS.pop(job.job_id, None)
        _FLIGHT_LOCK.release()
        raise
    return job.snapshot()


def _register_job(user_text, provider, camera_name, mode, check, model):
    job = WorkflowJob(user_text=user_text.strip(), provider=provider,
                      camera_name=camera_name, mode=mode, check=check, model=model)
    with _JOBS_LOCK:
        if len(_JOBS) >= _MAX_JOBS:
            for k, old in list(_JOBS.items()):
                if old.status in _TERMINAL:
                    del _JOBS[k]
                if len(_JOBS) < _MAX_JOBS:
                    break
            if len(_JOBS) >= _MAX_JOBS:
                raise RuntimeError("Workflow job storage is full")
        _JOBS[job.job_id] = job
    return job


def _lookup_job(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise LookupError("Workflow job not found in this process")
    return job


def wait_workflow(job_id, *, detail="full"):
    """Wait on an existing Job; returns the same snapshot as GET status."""
    return _lookup_job(job_id).wait_until_finished(detail=detail)


def watch_workflow(job_id, *, after=0):
    if type(after) is not int or after < 0:
        raise ValueError("after must be a non-negative integer")
    return _lookup_job(job_id).iter_events(after=after)

import json

def _workflow_format_event(evt):
    """Human-readable text lines for curl -N, not JSON or a browser frontend."""
    kind = evt.get("type")
    if kind == "capability_selected":
        return (f"[CAPABILITY] {evt.get('tool_name', '?')} "
                f"reason={evt.get('reason', '')}\n")
    if kind == "planning_full_task":
        return "[PLANNER] generating full plan...\n"
    if kind == "capability_result":
        if "answer" in evt:
            return f"[ANSWER] {evt['answer']}\n"
        return (f"[OBSERVATION] {evt.get('tool_name', '?')}\n"
                f"  {evt.get('summary', '')}\n")
    if kind == "job_created":
        return f"[JOB] {evt['job_id']}  mode={evt['mode']}\n[TASK] {evt['user_text']}\n"
    if kind == "plan":
        steps = evt.get("steps", [])
        lines = [f"[PLAN] {len(steps)} steps  plan_id={evt.get('plan_id', '')}"]
        for s in steps:
            args = json.dumps(s.get("arguments", {}), ensure_ascii=False, separators=(',', ':'))
            lines.append(f"  {s['step_id']} {s['tool_name']} {args}")
        return "\n".join(lines) + "\n"
    if kind in ("step_start", "step_end", "step_update"):
        prefix = f"[STEP {evt.get('index', '?')}/{evt.get('total', '?')}]"
        name = evt.get("tool_name", "?")
        if kind == "step_start":
            return f"{prefix} START {name}\n"
        status = str(evt.get("status", "unknown")).upper()
        elapsed = (f"  {evt['elapsed_seconds']:.2f}s"
                   if isinstance(evt.get("elapsed_seconds"), (int, float)) else "")
        error = f"  error={evt['error']}" if evt.get("error") else ""
        return f"{prefix} {status} {name}{elapsed}{error}\n"
    if kind == "status":
        status = evt.get("status", "?")
        if status == "planning":
            return "[AGENT] selecting capability...\n"
        if status == "answering":
            return "[AGENT] composing text reply...\n"
        if status == "observing":
            return "[AGENT] querying registered information Service...\n"
        if status == "answered":
            return "[RESULT] answered (no Action dispatched)\n"
        if status == "running":
            return "[RUNNER] preflight passed; starting execution\n"
        if status == "planned":
            return "[RESULT] planned (no physical action dispatched)\n"
        if status == "failed":
            error = evt.get("error") or {}
            return f"[RESULT] FAILED {error.get('type', '')}: {error.get('message', '')}\n"
        if status == "stopped":
            return f"[RESULT] STOPPED reason={evt.get('stopped_reason')}\n"
        if status == "dispatched_unverified":
            return "[RESULT] DISPATCHED_UNVERIFIED (physical completion not independently verified)\n"
        if status == "completed":
            return "[RESULT] COMPLETED\n"
        return f"[STATUS] {status}\n"
    if kind == "heartbeat":
        return f"[WAITING] status={evt.get('status')}\n"
    if kind == "job_finished":
        message = f"[END] status={evt.get('status')}\n"
        if evt.get("log_file"):
            message += f"[FILE] {evt['log_file']}\n"
        elif evt.get("log_error"):
            message += f"[FILE ERROR] {evt['log_error']}\n"
        return message
    if kind == "answer":
        return f"[ANSWER] {evt.get('text', '')}\n"
    if kind == "answer_error":
        return f"[ANSWER ERROR] {evt.get('error', '')}\n"

    return ""


# ============================================================
# Workflow API application logic
# ============================================================

def workflow_api_error(
    action: str,
    message: str,
    error_type: str = "ValueError",
) -> dict:
    return {
        "status": "error",
        "module": "agent",
        "action": action,
        "result": False,
        "data": None,
        "message": message,
        "error_type": error_type,
    }


def _workflow_api_ok(action: str, data: dict) -> dict:
    return {
        "status": "success",
        "module": "agent",
        "action": action,
        "result": True,
        "data": data,
    }


def stream_workflow_text(job_id):
    """Convert existing Workflow events into a text stream."""
    for event in watch_workflow(job_id):
        line = _workflow_format_event(event)
        if line:
            yield line


def run_workflow_request(body):
    """
    Validate and submit one Workflow.

    Returns:
        (result, status_code, is_stream)

    The Service does not construct Flask Response objects.
    """
    action = "workflow_run_stream"

    if not isinstance(body, dict):
        return (
            workflow_api_error(
                action,
                "request body 必須是 JSON object",
            ),
            400,
            False,
        )

    output = body.get("output", "steps")

    if output not in ("steps", "final"):
        return (
            workflow_api_error(
                action,
                "output 必須是 steps 或 final",
            ),
            400,
            False,
        )

    if body.get("check") is True:
        return (
            workflow_api_error(
                action,
                "check=true 已停用；此 Agent 僅提供自動執行，請使用 check=false",
            ),
            400,
            False,
        )

    allowed = {
        "user_text",
        "provider",
        "camera_name",
        "mode",
        "check",
        "model",
        "output",
    }

    extra = set(body) - allowed

    if extra:
        return (
            workflow_api_error(
                action,
                "不支援欄位：" + ", ".join(sorted(extra)),
            ),
            400,
            False,
        )

    try:
        job = create_workflow(
            user_text=body.get("user_text"),
            provider=body.get("provider", "local"),
            camera_name=body.get("camera_name", "left"),
            mode=body.get("mode", "plan"),
            check=body.get("check", False),
            model=body.get("model"),
        )

    except ValueError as exc:
        return (
            workflow_api_error(
                action,
                str(exc),
                "ValueError",
            ),
            400,
            False,
        )

    except AgentBusy as exc:
        return (
            workflow_api_error(
                action,
                str(exc),
                "AgentBusy",
            ),
            409,
            False,
        )

    except Exception as exc:
        return (
            workflow_api_error(
                action,
                f"無法建立 Workflow：{type(exc).__name__}",
                type(exc).__name__,
            ),
            500,
            False,
        )

    job_id = job["job_id"]

    if output == "steps":
        return stream_workflow_text(job_id), 200, True

    done = wait_workflow(job_id, detail="full")

    fields = (
        "job_id",
        "status",
        "user_text",
        "provider",
        "camera_name",
        "mode",
        "check",
        "capability",
        "plan",
        "progress",
        "result",
        "output",
        "error",
        "log_file",
        "created_at",
        "updated_at",
    )

    data = {
        key: done[key]
        for key in fields
        if key in done
    }

    return _workflow_api_ok(action, data), 200, False

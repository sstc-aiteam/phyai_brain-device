"""Agent Workflow Service: one-shot task jobs, planning, runner and HTTP gates.

Core logic stays in agent.workflow. No duplicate planning or Action dispatcher.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Condition, Lock, RLock, Thread
import os
import time
from uuid import uuid4

from agent.workflow.registry import all_tools
from agent.workflow.router import expected_tool_routes, preflight_service_names
from agent.workflow.planner import plan_full_task
from agent.workflow.runner import run_workflow

_JOBS = {}
_JOBS_LOCK = RLock()
_FLIGHT_LOCK = Lock()  # agent API single-flight within this Python process


class AgentBusy(RuntimeError):
    """Another Workflow is planning/running/waiting for operator input."""


def is_busy() -> bool:
    return _FLIGHT_LOCK.locked()


_MAX_JOBS = 64
_TERMINAL = frozenset(("planned", "completed", "dispatched_unverified", "stopped", "failed"))  # WORKFLOW_AUTO_DISPATCH_V1


def _now():
    return datetime.now(timezone.utc).isoformat()


def _confirmation_timeout():
    raw = os.getenv("AGENT_WORKFLOW_CONFIRM_TIMEOUT_SEC", "900")
    try:
        seconds = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("AGENT_WORKFLOW_CONFIRM_TIMEOUT_SEC must be numeric") from exc
    if not 5 <= seconds <= 86400:
        raise ValueError("AGENT_WORKFLOW_CONFIRM_TIMEOUT_SEC must be 5..86400")
    return seconds


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
        self.tool_routes = None
        self.preflight = None
        self.result = None
        self.error = None
        self.pending_gate = None
        self._decision = None
        self._condition = Condition(RLock())

    def _set(self, **values):
        with self._condition:
            for field, value in values.items():
                setattr(self, field, value)
            self.updated_at = _now()
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            if self.mode != "real":
                physical_state = "not_dispatched_by_request"
            elif self.status in ("failed", "stopped"):
                # After an error, earlier steps may already have moved hardware.
                physical_state = "unknown_after_failure"
            elif self.status == "dispatched_unverified":
                physical_state = "dispatched_without_completion_verification"
            elif self.status == "completed":
                physical_state = "completion_evidence_in_result"
            else:
                physical_state = "not_confirmed_in_progress"
            return deepcopy({
                "physical_state": physical_state,
                "job_id": self.job_id,
                "status": self.status,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "user_text": self.user_text,
                "provider": self.provider,
                "camera_name": self.camera_name,
                "mode": self.mode,
                "check": self.check,
                "plan": self.plan,
                "tool_routes": self.tool_routes,
                "preflight": self.preflight,
                "pending_gate": self.pending_gate,
                "result": self.result,
                "error": self.error,
                "process_local": True,
            })

    def operator_gate(self, *, step, arguments, previous_step, previous_arguments,
                      retry_available, mode, phase="before", previous_status=None):
        """Runs in worker thread. Wait for an explicit matching API decision.

        Deliberately disallow API retries of already dispatched physical steps.
        A retry requires a separate, reviewed recovery process.
        """
        if mode != "real" or not self.check:
            raise RuntimeError("Unexpected operator gate in unattended mode")
        gate = {
            "gate_id": uuid4().hex,
            "phase": phase,
            "step_id": step["step_id"],
            "tool_name": step["tool_name"],
            "arguments": deepcopy(arguments),
            "previous_step_id": previous_step["step_id"] if previous_step else None,
            "previous_status": previous_status,
            "allowed_decisions": ["continue", "stop"],
            "retry_available_in_cli_only": bool(retry_available),
        }
        with self._condition:
            self.pending_gate = gate
            self.status = "waiting_confirmation"
            self.updated_at = _now()
            self._condition.notify_all()
            deadline = time.monotonic() + _confirmation_timeout()
            while self._decision is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.error = "Operator confirmation timeout; no further action authorized"
                    choice = "stop"
                    break
                self._condition.wait(timeout=remaining)
            else:
                choice = self._decision
            self._decision = None
            self.pending_gate = None
            self.status = "running" if choice == "continue" else "stopping"
            self.updated_at = _now()
        return choice

    def decide(self, gate_id, decision):
        if decision not in ("continue", "stop"):
            raise ValueError("decision must be 'continue' or 'stop'")
        if not isinstance(gate_id, str) or not gate_id:
            raise ValueError("gate_id is required")
        with self._condition:
            gate = self.pending_gate
            if self.status != "waiting_confirmation" or gate is None:
                raise RuntimeError("Workflow is not waiting for confirmation")
            if gate["gate_id"] != gate_id:
                raise RuntimeError("Stale or mismatched gate_id")
            if self._decision is not None:
                raise RuntimeError("Decision already submitted")
            self._decision = decision
            self.updated_at = _now()
            self._condition.notify_all()
        return self.snapshot()


def _worker(job):
    try:
        job._set(status="planning")
        from agent.capability_selector import select_capability
        kwargs = {"provider": job.provider}
        if job.model is not None:
            kwargs["model"] = job.model
        selected = select_capability(job.user_text, **kwargs)
        if (not isinstance(selected, dict)
                or selected.get("tool_name") != "plan_task"
                or selected.get("arguments") != {}):
            raise RuntimeError("Workflow requires the plan_task capability; information queries are not part of this endpoint")
        plan = plan_full_task(job.user_text, provider=job.provider,
                              camera_name=job.camera_name, model=job.model)
        catalog = all_tools()
        routes = expected_tool_routes(plan, catalog)
        job._set(plan=plan, tool_routes=routes)
        if job.mode == "plan":
            job._set(status="planned")
            return
        # Preflight before sending *any* physical actions; runner repeats it.
        preflight = preflight_service_names(plan, catalog)
        job._set(preflight=preflight, status="running")
        result = run_workflow(plan, mode="real",
                              step_control=job.operator_gate if job.check else None)
        if not isinstance(result, dict) or type(result.get("success")) is not bool:
            raise RuntimeError("Workflow Runner returned an invalid report")
        final_status = ("stopped" if not result["success"] else
                        "dispatched_unverified" if result.get("completion_evidence") == "dispatch_only_unverified" else
                        "completed")
        job._set(result=result, status=final_status)
    except Exception as exc:
        job._set(status="failed", error={"type": type(exc).__name__,
                                          "message": str(exc)[:500]})
    finally:
        _FLIGHT_LOCK.release()


def create_workflow(*, user_text, provider="openai", camera_name="left",
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
    if check and mode != "real":
        raise ValueError("check=true requires mode=real")
    if model is not None and (not isinstance(model, str) or not 1 <= len(model) <= 120):
        raise ValueError("model must be a string of 1..120 characters")
    if check:
        _confirmation_timeout()
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


def get_workflow(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise LookupError("Workflow job not found in this process")
    return job.snapshot()


def decide_workflow(job_id, *, gate_id, decision):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise LookupError("Workflow job not found in this process")
    return job.decide(gate_id, decision)

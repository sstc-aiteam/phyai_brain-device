from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request


BASE_URL = "http://127.0.0.1:5001"


class ManualRuntime:
    """
    Manual Runtime client.

    Lifecycle:
        start()
        status()
        confirm()
        cancel()
        run()

    session_id is managed internally.
    """

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
    ):
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None

    def _request_json(
        self,
        req: urllib.request.Request,
    ) -> dict:
        try:
            with urllib.request.urlopen(
                req,
                timeout=180,
            ) as response:
                return json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.HTTPError as exc:
            body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            print("\n===== BACKEND ERROR =====")
            print("HTTP", exc.code)

            try:
                print(
                    json.dumps(
                        json.loads(body),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            except Exception:
                print(body)

            print("=========================")

            raise RuntimeError(
                f"Backend HTTP {exc.code}"
            ) from exc

    def _post(
        self,
        path: str,
        payload: dict,
    ) -> dict:
        data = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={
                "Content-Type":
                    "application/json",
            },
            method="POST",
        )

        return self._request_json(req)

    def _get(
        self,
        path: str,
        params: dict,
    ) -> dict:
        query = urllib.parse.urlencode(
            params
        )

        req = urllib.request.Request(
            f"{self.base_url}{path}?{query}",
            method="GET",
        )

        return self._request_json(req)

    def _require_session(
        self,
    ) -> None:
        if not self.session_id:
            raise RuntimeError(
                "Runtime 尚未 start"
            )

    def start(
        self,
        user_text: str,
        *,
        goal_decomposition: dict | None = None,
        coordination_extraction: dict | None = None,
        step_planning: dict | None = None,
        openai_api_key: str | None = None,
        max_iterations: int = 10,
        force_device_id: str | None = "right_arm",
    ) -> dict:
        default_model = {
            "provider": "remote",
            "model": "Qwen3-VL-8B-Instruct",
        }

        payload = {
            "user_text": user_text,
            "planning_options": {
                "goal_decomposition": (
                    goal_decomposition
                    or dict(default_model)
                ),
                "coordination_extraction": (
                    coordination_extraction
                    or dict(default_model)
                ),
                "step_planning": (
                    step_planning
                    or dict(default_model)
                ),
            },
            "max_iterations": max_iterations,
            "force_device_id": force_device_id,
        }

        if (
            isinstance(openai_api_key, str)
            and openai_api_key.strip()
        ):
            payload["openai_api_key"] = (
                openai_api_key.strip()
            )

        result = self._post(
            "/api/plan/runtime/manual/start",
            payload,
        )

        state = result.get(
            "data",
            result,
        )

        session_id = state.get(
            "session_id"
        )

        if not session_id:
            raise RuntimeError(
                "Runtime start 沒有回傳 session_id"
            )

        self.session_id = session_id

        print()
        print("Runtime opened")
        print("session_id =", session_id)

        return state

    def status(
        self,
    ) -> dict:
        self._require_session()

        result = self._get(
            "/api/plan/runtime/manual/status",
            {
                "session_id":
                    self.session_id,
            },
        )

        return result.get(
            "data",
            result,
        )

    def confirm(
        self,
    ) -> dict:
        self._require_session()

        result = self._post(
            "/api/plan/runtime/manual/confirm",
            {
                "session_id":
                    self.session_id,
            },
        )

        return result.get(
            "data",
            result,
        )

    def cancel(
        self,
    ) -> dict:
        self._require_session()

        result = self._post(
            "/api/plan/runtime/manual/cancel",
            {
                "session_id":
                    self.session_id,
            },
        )

        return result.get(
            "data",
            result,
        )

    @staticmethod
    def _show_state(
        state: dict,
    ) -> None:
        print()
        print("=" * 72)

        print(
            "STATUS :",
            state.get("status"),
        )

        print(
            "STAGE  :",
            state.get("stage"),
        )

        if state.get("message"):
            print()
            print("MESSAGE")
            print(state["message"])

        if state.get("proposed_step"):
            print()
            print(
                "PROPOSED STEP:",
                state["proposed_step"],
            )

        if state.get("proposed_state"):
            print(
                "PRE-COMMIT:",
                state["proposed_state"],
            )

        if state.get("proposed_instruction"):
            print()
            print("PROPOSED ACTION")
            print(
                state["proposed_instruction"]
            )

        if state.get("proposed_reason"):
            print()
            print("REASON")
            print(
                state["proposed_reason"]
            )

        if state.get("instruction"):
            print()
            print("MANUAL ACTION")
            print(
                state["instruction"]
            )

        if state.get("error"):
            print()
            print("ERROR")
            print(state["error"])

    def run(
        self,
        user_text: str,
        *,
        goal_decomposition: dict | None = None,
        coordination_extraction: dict | None = None,
        step_planning: dict | None = None,
        openai_api_key: str | None = None,
        max_iterations: int = 10,
        force_device_id: str | None = "right_arm",
        poll_interval: float = 0.75,
    ) -> dict:
        self.start(
            user_text,
            goal_decomposition=
                goal_decomposition,
            coordination_extraction=
                coordination_extraction,
            step_planning=
                step_planning,
            openai_api_key=
                openai_api_key,
            max_iterations=
                max_iterations,
            force_device_id=
                force_device_id,
        )

        last_display = None

        try:
            while True:
                state = self.status()

                display_key = (
                    state.get("status"),
                    state.get("stage"),
                    state.get("pending_step"),
                    state.get("proposed_step"),
                    state.get("proposed_state"),
                    state.get("proposed_instruction"),
                    state.get("instruction"),
                    state.get("message"),
                    state.get("error"),
                )

                if display_key != last_display:
                    self._show_state(
                        state
                    )
                    last_display = display_key

                if state.get(
                    "awaiting_confirmation"
                ):
                    print()

                    answer = input(
                        "[Enter] 已完成此動作 / q 取消 > "
                    ).strip().lower()

                    if answer == "q":
                        state = self.cancel()
                        self._show_state(
                            state
                        )
                        return state

                    self.confirm()
                    last_display = None
                    continue

                status = state.get(
                    "status"
                )

                if status not in {
                    "starting",
                    "running",
                    "awaiting_confirmation",
                }:
                    print()
                    print("=" * 72)
                    print(
                        "Runtime closed:",
                        status,
                    )
                    return state

                time.sleep(
                    poll_interval
                )

        except KeyboardInterrupt:
            print()
            print("Ctrl+C received.")

            try:
                state = self.cancel()
                print("Runtime cancelled.")
                return state
            except Exception:
                raise


if __name__ == "__main__":
    runtime = ManualRuntime()

    runtime.run(
        "把口罩放進垃圾桶",
        max_iterations=10,
        force_device_id="right_arm",
    )

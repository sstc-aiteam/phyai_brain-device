"""
Common Edge-side vision-skill runner.

Default behavior is LOCAL.

Only backend="remote" uses the PC VLM.
Any other backend value uses the existing local llm_service path.

Skill markdown stays on the Edge.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import OLLAMA_VLM_MODEL
from services import model_service
from agent.vlm import remote_vlm


_SKILLS_DIR = (
    Path(__file__).resolve().parent.parent
    / "skills"
)


@lru_cache(maxsize=None)
def load_skill(
    skill: str,
) -> str:
    """
    Load an Edge-side skill markdown file.

    Accepted examples:
        "describe_scene"
        "describe_scene.md"
        "agent/skills/describe_scene.md"
    """

    if not isinstance(skill, str):
        raise TypeError(
            "skill 必須是字串"
        )

    value = skill.strip()

    if not value:
        raise ValueError(
            "skill 不可為空"
        )

    candidate = Path(
        value
    ).expanduser()

    if (
        candidate.suffix == ".md"
        or "/" in value
    ):
        if not candidate.is_absolute():
            repo_root = (
                Path(__file__)
                .resolve()
                .parents[2]
            )

            candidate = (
                repo_root
                / candidate
            )

        path = candidate.resolve()

    else:
        path = (
            _SKILLS_DIR
            / f"{value}.md"
        ).resolve()

    if not path.is_file():
        raise RuntimeError(
            f"VLM skill 不存在：{path}"
        )

    content = (
        path.read_text(
            encoding="utf-8"
        )
        .strip()
    )

    # Remove optional YAML front matter.
    if content.startswith("---"):
        parts = content.split(
            "---",
            2,
        )

        if len(parts) == 3:
            content = (
                parts[2]
                .strip()
            )

    if not content:
        raise RuntimeError(
            f"VLM skill 為空：{path}"
        )

    return content


def _image_data_url(
    image_jpeg: bytes,
) -> str:
    if (
        not isinstance(
            image_jpeg,
            (bytes, bytearray),
        )
        or not image_jpeg
    ):
        raise ValueError(
            "image_jpeg 必須是非空 JPEG bytes"
        )

    encoded = (
        base64.b64encode(
            bytes(image_jpeg)
        )
        .decode(
            "ascii"
        )
    )

    return (
        "data:image/jpeg;base64,"
        f"{encoded}"
    )


def run_vision_skill(
    *,
    image_jpeg: bytes,
    skill: str,
    response_schema: dict[str, Any],
    user_prompt: str,
    backend: str | None = None,
    local_model: str | None = None,
    remote_model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    num_ctx: int = 8192,
    timeout: float = 60.0,
    keep_alive: int | str | None = -1,
    wait: bool = True,
    owner: str = "vision_skill",
) -> dict[str, Any]:
    """
    Run one image + one Edge skill + one structured schema.

    Backend policy:
        backend == "remote"
            -> PC remote VLM

        anything else, including None / "local"
            -> existing Edge local llm_service

    Return value is always a message object:
        {
            "role": "assistant",
            "content": "<JSON string>"
        }

    This intentionally matches llm_service.chat_structured().
    """

    if not isinstance(
        response_schema,
        dict,
    ):
        raise TypeError(
            "response_schema 必須是 dict"
        )

    if (
        not isinstance(
            user_prompt,
            str,
        )
        or not user_prompt.strip()
    ):
        raise ValueError(
            "user_prompt 必須是非空字串"
        )

    system_prompt = load_skill(
        skill
    )

    messages = [
        {
            "role":
                "system",

            "content":
                system_prompt,
        },

        {
            "role":
                "user",

            "content": [
                {
                    "type":
                        "text",

                    "text":
                        user_prompt.strip(),
                },

                {
                    "type":
                        "image_url",

                    "image_url": {
                        "url":
                            _image_data_url(
                                image_jpeg
                            ),
                    },
                },
            ],
        },
    ]

    use_remote = (
        isinstance(backend, str)
        and backend.strip().lower()
        == "remote"
    )

    if use_remote:
        return (
            remote_vlm
            .chat_structured(
                messages=
                    messages,

                response_schema=
                    response_schema,

                model=
                    remote_model,

                temperature=
                    temperature,

                max_tokens=
                    max_tokens,

                timeout=
                    timeout,
            )
        )

    return (
        model_service.qwen
        .chat_structured(
            messages=
                messages,

            response_schema=
                response_schema,

            model=(
                local_model
                or OLLAMA_VLM_MODEL
            ),

            temperature=
                temperature,

            max_tokens=
                max_tokens,

            num_ctx=
                num_ctx,

            timeout=
                timeout,

            keep_alive=
                keep_alive,

            wait=
                wait,

            owner=
                owner,
        )
    )

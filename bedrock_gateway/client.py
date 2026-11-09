from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import boto3
from botocore.config import Config

from .config import settings

logger = logging.getLogger("bedrock_gateway")

_client = None


def get_bedrock_client():
    global _client
    if _client is None:
        session_kwargs: dict[str, Any] = {}
        if settings.aws_profile:
            session_kwargs["profile_name"] = settings.aws_profile
        session = boto3.Session(**session_kwargs)
        _client = session.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            config=Config(retries={"max_attempts": 2, "mode": "adaptive"}),
        )
    return _client


def converse(params: dict[str, Any]) -> dict[str, Any]:
    client = get_bedrock_client()
    return client.converse(**params)


def converse_stream(params: dict[str, Any]) -> Any:
    client = get_bedrock_client()
    return client.converse_stream(**params)


async def stream_response(
    params: dict[str, Any],
    model: str,
) -> AsyncIterator[str]:
    """Yield SSE-formatted chunks from a Bedrock stream."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    try:
        response = converse_stream(params)
        stream = response.get("stream", [])
    except Exception as e:
        logger.error("Bedrock stream error: %s", e)
        error_chunk = _make_chunk(request_id, created, model, content=f"Error: {e}", finish="stop")
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    role_sent = False

    for event in stream:
        if "contentBlockStart" in event:
            block_start = event["contentBlockStart"].get("start", {})
            if "toolUse" in block_start:
                tu = block_start["toolUse"]
                tool_call = {
                    "index": event["contentBlockStart"].get("contentBlockIndex", 0),
                    "id": tu.get("toolUseId", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {"name": tu["name"], "arguments": ""},
                }
                chunk = _make_chunk(
                    request_id, created, model,
                    role="assistant" if not role_sent else None,
                    tool_calls=[tool_call],
                )
                role_sent = True
                yield f"data: {json.dumps(chunk)}\n\n"

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                chunk = _make_chunk(
                    request_id, created, model,
                    role="assistant" if not role_sent else None,
                    content=delta["text"],
                )
                role_sent = True
                yield f"data: {json.dumps(chunk)}\n\n"
            elif "toolUse" in delta:
                tool_delta = {
                    "index": event["contentBlockDelta"].get("contentBlockIndex", 0),
                    "function": {"arguments": delta["toolUse"].get("input", "")},
                }
                chunk = _make_chunk(
                    request_id, created, model,
                    tool_calls=[tool_delta],
                )
                yield f"data: {json.dumps(chunk)}\n\n"

        elif "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")
            finish_map = {
                "end_turn": "stop",
                "stop_sequence": "stop",
                "max_tokens": "length",
                "tool_use": "tool_calls",
            }
            chunk = _make_chunk(
                request_id, created, model,
                finish=finish_map.get(stop_reason, "stop"),
            )
            yield f"data: {json.dumps(chunk)}\n\n"

        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            chunk = _make_chunk(
                request_id, created, model,
                usage={
                    "prompt_tokens": usage.get("inputTokens", 0),
                    "completion_tokens": usage.get("outputTokens", 0),
                    "total_tokens": (
                        usage.get("inputTokens", 0) + usage.get("outputTokens", 0)
                    ),
                },
            )
            yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


def _make_chunk(
    request_id: str,
    created: int,
    model: str,
    role: str | None = None,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls

    choice: dict[str, Any] = {"index": 0, "delta": delta}
    if finish:
        choice["finish_reason"] = finish
    else:
        choice["finish_reason"] = None

    chunk: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    if usage:
        chunk["usage"] = usage
    return chunk

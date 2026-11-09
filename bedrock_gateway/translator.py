from __future__ import annotations

import json
import uuid
from typing import Any

from .models import ChatCompletionRequest, ChatMessage, ToolCall, FunctionCall


def _convert_content_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI content parts to Bedrock content blocks."""
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if part["type"] == "text":
            blocks.append({"text": part["text"]})
        elif part["type"] == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:"):
                media_type, _, b64_data = url.partition(";base64,")
                media_type = media_type.removeprefix("data:")
                blocks.append({
                    "image": {
                        "format": media_type.split("/")[-1].replace("jpeg", "jpeg"),
                        "source": {"bytes": b64_data},
                    }
                })
    return blocks


def _message_to_content(msg: ChatMessage) -> list[dict[str, Any]]:
    """Convert a single message's content to Bedrock content blocks."""
    if msg.content is None:
        return []
    if isinstance(msg.content, str):
        return [{"text": msg.content}] if msg.content else []
    return _convert_content_parts(msg.content)


def to_bedrock_messages(
    messages: list[ChatMessage],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Convert OpenAI messages to Bedrock system + messages format."""
    system_prompts: list[dict[str, Any]] = []
    bedrock_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            text = msg.content if isinstance(msg.content, str) else ""
            if text:
                system_prompts.append({"text": text})

        elif msg.role == "user":
            content = _message_to_content(msg)
            if content:
                bedrock_messages.append({"role": "user", "content": content})

        elif msg.role == "assistant":
            content = _message_to_content(msg)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        tool_input = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        tool_input = {}
                    content.append({
                        "toolUse": {
                            "toolUseId": tc.id,
                            "name": tc.function.name,
                            "input": tool_input,
                        }
                    })
            if content:
                bedrock_messages.append({"role": "assistant", "content": content})

        elif msg.role == "tool":
            bedrock_messages.append({
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": msg.tool_call_id or "",
                            "content": [{"text": msg.content or ""}],
                        }
                    }
                ],
            })

    return system_prompts or None, bedrock_messages


def to_bedrock_tools(request: ChatCompletionRequest) -> list[dict[str, Any]] | None:
    """Convert OpenAI tool definitions to Bedrock tool config."""
    if not request.tools:
        return None
    tools = []
    for tool in request.tools:
        spec: dict[str, Any] = {"name": tool.function.name}
        if tool.function.description:
            spec["description"] = tool.function.description
        spec["inputSchema"] = {"json": tool.function.parameters or {"type": "object"}}
        tools.append({"toolSpec": spec})
    return tools


def to_bedrock_tool_choice(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert OpenAI tool_choice to Bedrock toolChoice."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"auto": {}}
        if tool_choice == "required":
            return {"any": {}}
        if tool_choice == "none":
            return None
    elif isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            return {"tool": {"name": tool_choice["function"]["name"]}}
    return None


def build_converse_params(
    request: ChatCompletionRequest,
    bedrock_model_id: str,
    default_max_tokens: int,
) -> dict[str, Any]:
    """Build the full parameter dict for bedrock converse/converse_stream."""
    system, messages = to_bedrock_messages(request.messages)

    params: dict[str, Any] = {
        "modelId": bedrock_model_id,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": request.max_tokens or default_max_tokens,
        },
    }

    if request.temperature is not None:
        params["inferenceConfig"]["temperature"] = request.temperature
    if request.top_p is not None:
        params["inferenceConfig"]["topP"] = request.top_p
    if request.stop:
        stops = [request.stop] if isinstance(request.stop, str) else request.stop
        params["inferenceConfig"]["stopSequences"] = stops

    if system:
        params["system"] = system

    tools = to_bedrock_tools(request)
    if tools:
        tool_config: dict[str, Any] = {"tools": tools}
        tc = to_bedrock_tool_choice(request.tool_choice)
        if tc:
            tool_config["toolChoice"] = tc
        params["toolConfig"] = tool_config

    return params


def bedrock_response_to_openai(
    response: dict[str, Any],
    model: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Convert a Bedrock converse response to OpenAI format."""
    import time

    rid = request_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])
    usage_info = response.get("usage", {})
    stop_reason = response.get("stopReason", "end_turn")

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append({
                "id": tu.get("toolUseId", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": tu["name"],
                    "arguments": json.dumps(tu.get("input", {})),
                },
            })

    finish_reason_map = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }

    result_message: dict[str, Any] = {"role": "assistant"}
    text = "\n".join(text_parts) if text_parts else None
    if text is not None:
        result_message["content"] = text
    else:
        result_message["content"] = None
    if tool_calls:
        result_message["tool_calls"] = tool_calls

    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": result_message,
                "finish_reason": finish_reason_map.get(stop_reason, "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": usage_info.get("inputTokens", 0),
            "completion_tokens": usage_info.get("outputTokens", 0),
            "total_tokens": (
                usage_info.get("inputTokens", 0) + usage_info.get("outputTokens", 0)
            ),
        },
    }

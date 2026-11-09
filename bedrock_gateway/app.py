from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .client import converse, stream_response
from .config import settings
from .models import ChatCompletionRequest
from .translator import bedrock_response_to_openai, build_converse_params

logger = logging.getLogger("bedrock_gateway")

app = FastAPI(
    title="Bedrock Gateway",
    description="OpenAI-compatible API proxy for AWS Bedrock",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_aliases: dict[str, str] = {}


@app.on_event("startup")
async def _load_aliases():
    global _model_aliases
    _model_aliases = settings.load_model_aliases()
    logger.info("Loaded %d model aliases", len(_model_aliases))


def _check_auth(request: Request) -> None:
    if not settings.api_key:
        return
    auth = request.headers.get("authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _resolve_model(model: str) -> str:
    return _model_aliases.get(model, model)


@app.get("/v1/models")
async def list_models(request: Request):
    _check_auth(request)
    models = []
    for alias, bedrock_id in _model_aliases.items():
        models.append({
            "id": alias,
            "object": "model",
            "created": 0,
            "owned_by": "bedrock",
            "bedrock_model_id": bedrock_id,
        })
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    _check_auth(request)

    bedrock_model_id = _resolve_model(body.model)
    logger.info(
        "Request: model=%s -> %s, stream=%s, messages=%d",
        body.model, bedrock_model_id, body.stream, len(body.messages),
    )

    try:
        params = build_converse_params(body, bedrock_model_id, settings.default_max_tokens)
    except Exception as e:
        logger.error("Failed to build converse params: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    if body.stream:
        return StreamingResponse(
            stream_response(params, body.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response = converse(params)
    except Exception as e:
        error_msg = str(e)
        logger.error("Bedrock converse error: %s", error_msg)
        if "AccessDeniedException" in error_msg:
            raise HTTPException(status_code=403, detail=f"Bedrock access denied: {error_msg}")
        if "ValidationException" in error_msg:
            raise HTTPException(status_code=400, detail=f"Bedrock validation error: {error_msg}")
        if "ThrottlingException" in error_msg:
            raise HTTPException(status_code=429, detail=f"Bedrock rate limit: {error_msg}")
        raise HTTPException(status_code=502, detail=f"Bedrock error: {error_msg}")

    openai_response = bedrock_response_to_openai(response, body.model)
    return JSONResponse(content=openai_response)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "message": "Bedrock Gateway - OpenAI-compatible API proxy for AWS Bedrock",
        "docs": "/docs",
        "health": "/health",
    }

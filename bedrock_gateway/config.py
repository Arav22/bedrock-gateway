from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic_settings import BaseSettings

_DEFAULT_ALIASES: dict[str, str] = {
    "gpt-4o": "us.anthropic.claude-sonnet-4-6",
    "gpt-4o-mini": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "gpt-4-turbo": "us.anthropic.claude-opus-4-6-v1",
    "gpt-4": "us.anthropic.claude-opus-4-6-v1",
    "gpt-3.5-turbo": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-3-opus": "us.anthropic.claude-opus-4-6-v1",
    "claude-3-sonnet": "us.anthropic.claude-sonnet-4-6",
    "claude-3-haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
}


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    aws_region: str = "us-east-1"
    aws_profile: str | None = None
    log_level: str = "info"
    model_aliases_file: str | None = None
    default_max_tokens: int = 4096
    api_key: str | None = None

    model_config = {"env_prefix": "BEDROCK_GATEWAY_"}

    def load_model_aliases(self) -> dict[str, str]:
        aliases = dict(_DEFAULT_ALIASES)
        if self.model_aliases_file:
            path = Path(self.model_aliases_file)
            if path.exists():
                aliases.update(json.loads(path.read_text()))
        env_aliases = os.environ.get("BEDROCK_GATEWAY_MODEL_ALIASES")
        if env_aliases:
            aliases.update(json.loads(env_aliases))
        return aliases


settings = Settings()

# Bedrock Gateway

Drop-in OpenAI-compatible API proxy for **AWS Bedrock**. Use any tool that speaks the OpenAI API — Cursor, Aider, Continue, Open Interpreter, etc. — with your AWS Bedrock models and credits.

## Why?

Most AI coding tools only support OpenAI-compatible endpoints. If you have AWS credits or prefer Bedrock, you're locked out. Bedrock Gateway bridges that gap with zero code changes on the client side.

## Quick Start

```bash
# Clone
git clone https://github.com/Arav22/bedrock-gateway.git
cd bedrock-gateway

# Install
pip install -r requirements.txt

# Run (uses your default AWS credentials)
python -m bedrock_gateway
```

The server starts at `http://localhost:8000`.

### With Docker

```bash
docker compose up -d
```

## Usage

Point any OpenAI-compatible client at the gateway:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

`gpt-4` gets routed to Claude on Bedrock automatically.

### With Cursor / Continue / Aider

Set the base URL to `http://localhost:8000/v1` and use any model name from the alias list.

### With Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # or set BEDROCK_GATEWAY_API_KEY
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Model Aliases

Built-in mappings (override with env or config file):

| OpenAI Model | Bedrock Model |
|---|---|
| `gpt-4o` | `us.anthropic.claude-sonnet-4-6` |
| `gpt-4-turbo` | `us.anthropic.claude-opus-4-6-v1` |
| `gpt-4` | `us.anthropic.claude-opus-4-6-v1` |
| `gpt-3.5-turbo` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `gpt-4o-mini` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |

You can also pass Bedrock model IDs directly (e.g. `anthropic.claude-opus-4-6-v1`).

### Custom Aliases

**Via environment variable:**

```bash
export BEDROCK_GATEWAY_MODEL_ALIASES='{"my-model": "anthropic.claude-opus-4-6-v1"}'
```

**Via JSON file:**

```bash
export BEDROCK_GATEWAY_MODEL_ALIASES_FILE=./aliases.json
```

## Configuration

All settings via environment variables prefixed with `BEDROCK_GATEWAY_`:

| Variable | Default | Description |
|---|---|---|
| `BEDROCK_GATEWAY_HOST` | `0.0.0.0` | Bind address |
| `BEDROCK_GATEWAY_PORT` | `8000` | Port |
| `BEDROCK_GATEWAY_AWS_REGION` | `us-east-1` | AWS region |
| `BEDROCK_GATEWAY_AWS_PROFILE` | (none) | AWS profile name |
| `BEDROCK_GATEWAY_API_KEY` | (none) | Optional auth key for the proxy |
| `BEDROCK_GATEWAY_DEFAULT_MAX_TOKENS` | `4096` | Default max tokens |
| `BEDROCK_GATEWAY_LOG_LEVEL` | `info` | Log level |

## Features

- **Chat Completions** — Full `/v1/chat/completions` support
- **Streaming** — Server-sent events (SSE) streaming
- **Tool/Function Calling** — Translates OpenAI tool calls to Bedrock format
- **Vision** — Passes through base64 images
- **Model Aliasing** — Map any name to any Bedrock model
- **Auth** — Optional API key protection
- **Docker** — Ready to deploy

## AWS Credentials

Uses standard AWS credential chain:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS profile (`BEDROCK_GATEWAY_AWS_PROFILE`)
3. IAM role (EC2, ECS, Lambda)
4. SSO session

## License

MIT

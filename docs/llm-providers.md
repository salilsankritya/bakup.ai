# LLM Provider Configuration — bakup.ai

bakup.ai supports multiple LLM providers for AI-powered log analysis,
root-cause reasoning, and code review. This document covers setup,
provider options, and tuning parameters.

## Supported Providers

| Provider        | ID             | Requires API Key | Notes                              |
|-----------------|----------------|------------------|------------------------------------|
| **OpenAI**      | `openai`       | Yes              | GPT-4o, GPT-4.1, GPT-4.1 Mini     |
| **Anthropic**   | `anthropic`    | Yes              | Claude Sonnet 4, Opus 4, Haiku 3.5 |
| **Azure OpenAI**| `azure_openai` | Yes              | Any Azure-deployed model           |
| **Ollama**      | `ollama`       | No               | Local models, no data leaves machine|

## Available Models

### OpenAI
- `gpt-4o` (default)
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4o-mini`
- `gpt-4.1-nano`

### Anthropic
- `claude-sonnet-4-20250514` (default)
- `claude-opus-4-20250514`
- `claude-3-5-haiku-20241022`

### Azure OpenAI
- `gpt-4o` (default)
- `gpt-4.1`
- `gpt-4.1-mini`

### Ollama (Local)
- `gemma3:4b` (default — recommended for CPU-only machines)
- `llama3`
- `mistral`
- `codellama`
- Any model installed locally is auto-detected

## Configuration

### Via the UI

1. Click the gear icon in the sidebar or header
2. Select a **Provider** from the dropdown
3. Select a **Model** or type a custom model name
4. Enter your **API Key** (not needed for Ollama)
5. Optionally adjust **Advanced Settings** (temperature, max tokens, etc.)
6. Click **Test Connection** to verify
7. Click **Save & Activate**

### Via the Config File

The config is stored at `backend/model-weights/bakup_llm_config.json`:

```json
{
  "provider": "ollama",
  "model": "gemma3:4b",
  "api_key": "",
  "azure_endpoint": "",
  "azure_api_version": "2024-02-01",
  "ollama_base_url": "http://localhost:11434",
  "configured": true,
  "temperature": 0.1,
  "num_predict": 1024,
  "num_ctx": 8192,
  "timeout": 300
}
```

### Via Environment Variables

Environment variables override config file values at runtime:

| Variable                  | Default | Description                                |
|---------------------------|---------|--------------------------------------------|
| `BAKUP_LLM_TEMPERATURE`  | `0.1`   | Generation temperature (0.0–2.0)           |
| `BAKUP_LLM_MAX_TOKENS`   | `1024`  | Max tokens to generate per request         |
| `BAKUP_OLLAMA_CTX`        | `8192`  | Ollama context window size                 |
| `BAKUP_OLLAMA_TIMEOUT`    | `300`   | Ollama request timeout in seconds          |
| `BAKUP_MAX_CONTEXT_CHARS` | `8000`  | Max evidence context chars sent to LLM     |
| `BAKUP_CONFIDENCE_THRESHOLD` | `0.35` | Minimum confidence for extractive results |

## Generation Parameters

| Parameter     | Default | Range         | Description                        |
|---------------|---------|---------------|------------------------------------|
| `temperature` | `0.1`   | 0.0 – 2.0    | Lower = more deterministic         |
| `num_predict` | `1024`  | 64 – 32768   | Max tokens per response            |
| `num_ctx`     | `8192`  | 512 – 131072 | Context window (Ollama only)       |
| `timeout`     | `300`   | 10 – 1800    | Request timeout in seconds         |

## Ollama Auto-Detection

When Ollama is selected as the provider, bakup.ai automatically queries
the local Ollama server (`/api/tags`) to discover installed models. These
appear as clickable chips in the setup modal. Install new models with:

```bash
ollama pull gemma3:4b
ollama pull llama3
ollama pull mistral
```

## API Endpoints

| Method   | Path                 | Description                          |
|----------|----------------------|--------------------------------------|
| `GET`    | `/llm/status`        | Health check (poll from UI)          |
| `GET`    | `/llm/config`        | Current config (API key masked)      |
| `POST`   | `/llm/config`        | Save config + test connectivity      |
| `POST`   | `/llm/test`          | Test connectivity without saving     |
| `DELETE` | `/llm/config`        | Reset to unconfigured                |
| `GET`    | `/llm/providers`     | List all providers, models, defaults |
| `GET`    | `/llm/ollama-models` | Auto-detect installed Ollama models  |

## Architecture

Provider logic is isolated into adapter modules under
`backend/core/llm/providers/`:

```
backend/core/llm/
├── config_store.py          # Config persistence, provider/model metadata
├── llm_service.py           # Main LLM service (routing, fallback, quality gate)
├── prompt_templates.py      # System prompts for different analysis modes
├── providers/
│   ├── __init__.py           # Re-exports all provider functions
│   ├── ollama_provider.py    # Ollama adapter (urllib, no dependencies)
│   ├── openai_provider.py    # OpenAI + Azure OpenAI adapter
│   └── anthropic_provider.py # Anthropic Claude adapter (urllib)
```

Each adapter exposes three functions:
- `call(cfg, user_message, system_prompt) -> str`
- `call_with_tools(cfg, messages, tools) -> dict`
- `ping(cfg) -> None`

## Fallback Behavior

- If the LLM is **not configured**, queries return extractive results with
  a note to configure an LLM for full reasoning.
- If an LLM call **fails** (timeout, network error, bad key), the system
  falls back to extractive mode with an error message. No crashes.
- If a response is **too short or truncated**, the quality gate retries once
  with a doubled token budget.

## Debug Logging

Every LLM call logs:
- Provider and model name
- System prompt length and context length
- Response length
- Any fallback or retry activity

View in the backend console output (lines prefixed `[bakup:llm]`).

## Security

- API keys are stored locally at `model-weights/bakup_llm_config.json`
- File permissions are set to owner-read-only (0600) on POSIX systems
- API keys are **never logged**, **never returned in API responses**
  (masked to `****xxxx`), and **never transmitted** except to the
  selected provider's API endpoint
- The config file should not be committed to version control (already
  in `.gitignore`)

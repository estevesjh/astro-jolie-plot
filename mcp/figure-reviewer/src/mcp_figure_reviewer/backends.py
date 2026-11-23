"""Anthropic LLM backends for the blind figure reviewer.

Backend selection order (backend="auto"):
  1. `claude_cli`    - local Claude Code CLI (`claude -p`) run inside an isolated temp
                       directory that holds only the anonymized, metadata-stripped
                       `figure.png`. Uses your existing Claude Code login, no API key.
  2. `anthropic_api` - direct Anthropic Messages API (`ANTHROPIC_API_KEY`) via httpx.
                       Stateless: sends only the stripped PNG bytes and the prompt.

Models are Anthropic models. Override with FIGURE_REVIEW_MODEL (fast) and
FIGURE_REVIEW_DEEP_MODEL (deep), or per backend with FIGURE_REVIEW_CLAUDE_MODEL and
FIGURE_REVIEW_ANTHROPIC_MODEL.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel

from .blinding import BLIND_FILENAME, IsolatedBlindSandbox, PreparedBlindImage
from .schemas import dereference_json_schema

T = TypeVar("T", bound=BaseModel)

_CREDENTIAL_ENV_KEYS = frozenset({"ANTHROPIC_API_KEY"})


def load_credentials_dotenv(env_path: Path | None = None) -> list[str]:
    """Safely load allowlisted LLM API keys from `~/.env` via `os.environ.setdefault`.

    Never logs or returns secret values — returns only the list of loaded key names.
    Explicit environment variables always take precedence over `~/.env`.
    """
    if env_path is None and os.environ.get("FIGURE_REVIEW_SKIP_DOTENV", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return []
    path = (env_path or (Path.home() / ".env")).expanduser()
    if not path.is_file():
        return []

    preexisting = set(os.environ.keys())
    loaded: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key not in _CREDENTIAL_ENV_KEYS or key in preexisting:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        if val:
            os.environ[key] = val
            if key not in loaded:
                loaded.append(key)
    return loaded


# Auto-load `~/.env` once at startup so MCP server & CLI pick up saved API keys automatically
load_credentials_dotenv()


DEFAULT_FAST_MODEL = os.environ.get("FIGURE_REVIEW_MODEL", "claude-sonnet-5-5")
DEFAULT_DEEP_MODEL = os.environ.get("FIGURE_REVIEW_DEEP_MODEL", "claude-opus-5-5")
DEFAULT_ANTHROPIC_MODEL = os.environ.get("FIGURE_REVIEW_ANTHROPIC_MODEL", DEFAULT_FAST_MODEL)
DEFAULT_CLAUDE_CLI_MODEL = os.environ.get("FIGURE_REVIEW_CLAUDE_MODEL", DEFAULT_FAST_MODEL)

RECOMMENDED_MODELS = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
    "claude-fable-5-2",
]


def default_model_for(backend: str, speed: str = "fast") -> str:
    """Return the default Anthropic model for a backend and speed preset ('fast' or 'deep')."""
    if speed == "deep":
        return DEFAULT_DEEP_MODEL
    if backend == "anthropic_api":
        return DEFAULT_ANTHROPIC_MODEL
    return DEFAULT_CLAUDE_CLI_MODEL

@dataclass(frozen=True)
class BackendCallResult:
    """Validated structured result plus provenance and timing from an LLM backend."""

    parsed: BaseModel
    backend: str
    model: str
    blinding_level: Literal["airtight_stateless_api", "isolated_sandbox_cli"]
    duration_seconds: float


class BackendUnavailableError(RuntimeError):
    """Raised when the requested LLM backend is not configured or installed."""


def _find_binary(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for candidate in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def detect_backends() -> dict[str, Any]:
    """Inspect available Anthropic backends without exposing any secret values."""
    claude_bin = _find_binary("claude")
    anthropic_api_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))

    available_order: list[str] = []
    if claude_bin:
        available_order.append("claude_cli")
    if anthropic_api_ok:
        available_order.append("anthropic_api")

    preferred_env = os.environ.get("FIGURE_REVIEW_BACKEND", "").strip().lower()
    if preferred_env in available_order:
        active_default: str | None = preferred_env
    else:
        active_default = available_order[0] if available_order else None
    default_model_map = {
        "anthropic_api": DEFAULT_ANTHROPIC_MODEL,
        "claude_cli": DEFAULT_CLAUDE_CLI_MODEL,
    }

    return {
        "active_default_backend": active_default,
        "active_default_model": default_model_map.get(active_default) if active_default else None,
        "backends": {
            "claude_cli": {
                "available": bool(claude_bin),
                "binary": claude_bin,
                "provider": "Anthropic (Claude Code CLI)",
                "blinding_level": "isolated_sandbox_cli",
                "default_model": DEFAULT_CLAUDE_CLI_MODEL,
                "deep_model": DEFAULT_DEEP_MODEL,
                "recommended_models": RECOMMENDED_MODELS,
                "how_to_enable": "Install Claude Code and sign in (`claude`).",
            },
            "anthropic_api": {
                "available": anthropic_api_ok,
                "provider": "Anthropic (Messages API)",
                "blinding_level": "airtight_stateless_api",
                "default_model": DEFAULT_ANTHROPIC_MODEL,
                "deep_model": DEFAULT_DEEP_MODEL,
                "recommended_models": RECOMMENDED_MODELS,
                "how_to_enable": "Set ANTHROPIC_API_KEY in the environment.",
            },
        },
    }


def resolve_backend(requested: str = "auto") -> str:
    """Resolve 'auto' or validate an explicit backend name."""
    info = detect_backends()
    if requested == "auto":
        default = info["active_default_backend"]
        if not default:
            raise BackendUnavailableError(
                "No LLM backend available. Install and sign in to Claude Code (`claude`), "
                "or set ANTHROPIC_API_KEY."
            )
        return str(default)

    if requested not in info["backends"]:
        raise ValueError(
            f"Unknown backend '{requested}'. Choose from: auto, {', '.join(info['backends'].keys())}"
        )
    if not info["backends"][requested]["available"]:
        hint = info["backends"][requested]["how_to_enable"]
        raise BackendUnavailableError(f"Backend '{requested}' is not available. {hint}")
    return requested


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse JSON from raw LLM text, stripping optional markdown code fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: locate outermost {...}
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        obj = json.loads(cleaned[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Model did not return a valid JSON object. Preview: {raw_text[:400]!r}")


async def _call_anthropic_api(
    *,
    system_prompt: str,
    user_prompt: str,
    schema_model: type[T],
    prepared_image: PreparedBlindImage | None,
    model: str | None,
    timeout_seconds: int,
) -> BackendCallResult:
    """Stateless, airtight Anthropic Messages REST API call via httpx."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise BackendUnavailableError("ANTHROPIC_API_KEY is not set.")

    chosen_model = model or DEFAULT_ANTHROPIC_MODEL
    schema_json = json.dumps(dereference_json_schema(schema_model), indent=2)
    content: list[dict[str, Any]] = []
    if prepared_image is not None:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": prepared_image.mime_type,
                    "data": base64.b64encode(prepared_image.png_bytes).decode("ascii"),
                },
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                f"{user_prompt}\n\n"
                "Return ONLY a single valid JSON object conforming strictly to this JSON Schema "
                "(no markdown fences or commentary):\n"
                f"{schema_json}"
            ),
        }
    )

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": chosen_model,
                "max_tokens": 4096,
                "temperature": 0.1,
                "system": system_prompt,
                "messages": [{"role": "user", "content": content}],
            },
        )
        resp.raise_for_status()
        data = resp.json()

    elapsed = round(time.monotonic() - t0, 2)
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    raw_text = "\n".join(text_blocks)
    parsed = schema_model.model_validate(_extract_json_object(raw_text))
    return BackendCallResult(
        parsed=parsed,
        backend="anthropic_api",
        model=chosen_model,
        blinding_level="airtight_stateless_api",
        duration_seconds=elapsed,
    )


async def _call_claude_cli(
    *,
    system_prompt: str,
    user_prompt: str,
    schema_model: type[T],
    prepared_image: PreparedBlindImage | None,
    model: str | None,
    timeout_seconds: int,
) -> BackendCallResult:
    """Run local Claude Code CLI (`claude -p`) inside an isolated /tmp sandbox."""
    claude_bin = _find_binary("claude")
    if not claude_bin:
        raise BackendUnavailableError("`claude` CLI not found on PATH.")

    chosen_model = model or DEFAULT_CLAUDE_CLI_MODEL
    schema_str = json.dumps(dereference_json_schema(schema_model), separators=(",", ":"))
    t0 = time.monotonic()

    with IsolatedBlindSandbox(prepared_image, prefix="blind_claude_") as sandbox:
        assert sandbox.root is not None
        cmd = [
            claude_bin,
            "-p",
            user_prompt,
            "--append-system-prompt",
            system_prompt,
            "--model",
            chosen_model,
            "--output-format",
            "json",
            "--json-schema",
            schema_str,
            "--max-turns",
            "3" if prepared_image is not None else "1",
        ]
        if prepared_image is not None:
            cmd.extend(["--allowedTools", "Read"])
        else:
            cmd.extend(["--allowedTools", ""])

        env = os.environ.copy()
        env["TMPDIR"] = str(sandbox.root)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(sandbox.root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds + 20
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"`claude` CLI timed out after {timeout_seconds}s")

        elapsed = round(time.monotonic() - t0, 2)
        stdout = stdout_b.decode("utf-8", errors="replace").strip()
        stderr = stderr_b.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(
                f"`claude` CLI failed (exit {proc.returncode}): {(stderr or stdout)[:600]}"
            )

        envelope = _extract_json_object(stdout)
        if isinstance(envelope.get("structured_output"), dict):
            payload_obj = envelope["structured_output"]
        elif isinstance(envelope.get("result"), str):
            payload_obj = _extract_json_object(envelope["result"])
        else:
            payload_obj = envelope

        parsed = schema_model.model_validate(payload_obj)
        return BackendCallResult(
            parsed=parsed,
            backend="claude_cli",
            model=chosen_model,
            blinding_level="isolated_sandbox_cli",
            duration_seconds=elapsed,
        )


async def call_structured_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    schema_model: type[T],
    prepared_image: PreparedBlindImage | None = None,
    backend: str = "auto",
    model: str | None = None,
    speed: str = "fast",
    timeout_seconds: int = 75,
) -> BackendCallResult:
    """Invoke the selected Anthropic backend and return a schema-validated Pydantic result."""
    resolved = resolve_backend(backend)
    chosen_model = model or default_model_for(resolved, speed=speed)
    call = _call_anthropic_api if resolved == "anthropic_api" else _call_claude_cli
    return await call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_model=schema_model,
        prepared_image=prepared_image,
        model=chosen_model,
        timeout_seconds=timeout_seconds,
    )


def backend_uses_cli_sandbox(backend: str) -> bool:
    """Return True if the resolved backend reads figure.png from a sandboxed directory."""
    return resolve_backend(backend) == "claude_cli"

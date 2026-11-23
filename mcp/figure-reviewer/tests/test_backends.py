"""Unit tests for schemas, backend priority, JSON extraction, and Anthropic payload blinding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, PngImagePlugin

from mcp_figure_reviewer import backends
from mcp_figure_reviewer.blinding import prepare_blind_image
from mcp_figure_reviewer.reviewer import run_blind_review
from mcp_figure_reviewer.schemas import (
    BlindReviewPayload,
    MasterGradePayload,
    dereference_json_schema,
)


def _contains_key(obj: object, forbidden: str) -> bool:
    if isinstance(obj, dict):
        if forbidden in obj:
            return True
        return any(_contains_key(v, forbidden) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_key(item, forbidden) for item in obj)
    return False


def test_dereferenced_schemas_have_no_refs_or_defs() -> None:
    for cls in (BlindReviewPayload, MasterGradePayload):
        schema = dereference_json_schema(cls)
        assert not _contains_key(schema, "$defs")
        assert not _contains_key(schema, "$ref")
        assert schema["type"] == "object"
        assert "properties" in schema


def test_extract_json_object_handles_fences_and_surrounding_text() -> None:
    raw = 'Here is the review:\n```json\n{"a": 1, "b": "ok"}\n```\nDone.'
    assert backends._extract_json_object(raw) == {"a": 1, "b": "ok"}


def test_backend_priority_prefers_claude_cli_then_anthropic_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FIGURE_REVIEW_BACKEND", raising=False)
    with patch.object(backends, "_find_binary", return_value="/usr/local/bin/claude"):
        assert backends.resolve_backend("auto") == "claude_cli"
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        assert backends.resolve_backend("auto") == "claude_cli"
        monkeypatch.setenv("FIGURE_REVIEW_BACKEND", "anthropic_api")
        assert backends.resolve_backend("auto") == "anthropic_api"
    with patch.object(backends, "_find_binary", return_value=None):
        monkeypatch.delenv("FIGURE_REVIEW_BACKEND", raising=False)
        assert backends.resolve_backend("auto") == "anthropic_api"


def test_default_models_are_anthropic() -> None:
    assert backends.default_model_for("claude_cli", "fast").startswith("claude-")
    assert backends.default_model_for("anthropic_api", "deep").startswith("claude-")


def test_load_credentials_dotenv_allowlist_and_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FIGURE_REVIEW_SKIP_DOTENV", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("UNRELATED_SECRET", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        'export ANTHROPIC_API_KEY="fake-key-for-unit-test"\n'
        "UNRELATED_SECRET=should-not-load\n",
        encoding="utf-8",
    )
    loaded = backends.load_credentials_dotenv(env_file)
    assert loaded == ["ANTHROPIC_API_KEY"]
    assert backends.os.environ.get("ANTHROPIC_API_KEY") == "fake-key-for-unit-test"
    assert "UNRELATED_SECRET" not in backends.os.environ


@pytest.mark.asyncio
async def test_anthropic_api_receives_only_stripped_bytes_and_no_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaky = tmp_path / "SECRET_PROJECT_fig9_DESY3_CPF2.png"
    img = Image.new("RGB", (640, 480), (200, 210, 220))
    info = PngImagePlugin.PngInfo()
    info.add_text("Title", "SECRET_METADATA_TITLE")
    img.save(leaky, format="PNG", pnginfo=info)

    fake_payload = {
        "one_sentence_conclusion": "A blank light-blue test image with no plotted data.",
        "panel_encodings": [],
        "unclear_terms": [],
        "colormap_audit": {
            "has_colorbar": False,
            "observed_palette_description": "Flat pale blue canvas.",
            "colormap_role_match": "NOT_APPLICABLE",
            "stretch_and_midpoint_notes": "N/A",
            "contrast_and_accessibility": "No plotted curves.",
            "colormap_issues": [],
        },
        "visual_and_layout_issues": [],
        "confusing_elements": ["No axes or curves are plotted."],
        "standalone_readability_score": 1,
        "readability_summary": "Empty test canvas.",
    }

    captured: dict[str, object] = {}

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"content": [{"type": "text", "text": json.dumps(fake_payload)}]}

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, *, headers, json):
            captured["url"] = url
            captured["body"] = json
            return _FakeResp()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))

    with patch.object(backends.httpx, "AsyncClient", _FakeClient):
        review = await run_blind_review(
            leaky, backend="anthropic_api", model="claude-sonnet-5-5", detail="full"
        )

    assert review.blinding_certificate.backend == "anthropic_api"
    assert review.blinding_certificate.blinding_level == "airtight_stateless_api"
    assert review.blinding_certificate.original_filename_hidden is True
    assert "Title" in review.blinding_certificate.stripped_metadata_keys

    # Inspect the exact request body: no filename or metadata leaks.
    body = captured["body"]
    assert body["model"] == "claude-sonnet-5-5"
    blob = json.dumps(body)
    for forbidden in ("SECRET_PROJECT", "DESY3", "CPF2", "SECRET_METADATA_TITLE"):
        assert forbidden not in blob

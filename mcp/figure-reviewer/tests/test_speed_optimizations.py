"""Unit tests for speed optimizations #2 (0-token chat Master), #5 (fail-fast pre-gate), and #6."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastmcp import Client
from PIL import Image, PngImagePlugin

from mcp_figure_reviewer import backends
from mcp_figure_reviewer.backends import BackendCallResult
from mcp_figure_reviewer.blinding import (
    IsolatedBlindSandbox,
    image_settings_for_speed,
    inspect_figure_pixels,
    prepare_blind_image,
)
from mcp_figure_reviewer.reviewer import (
    Phase1GateBlockedError,
    run_batch_blind_review,
    run_blind_review,
    run_deterministic_grade,
    run_end_to_end_review,
    run_phase2_context_review,
)
from mcp_figure_reviewer.schemas import (
    BlindReviewPayload,
    ColorbarReading,
    ColormapAudit,
    CompactBlindReviewPayload,
    PanelEncoding,
    Phase2ReviewPayload,
    Phase2ScoreBreakdown,
    TermAuditItem,
    dereference_json_schema,
)
from mcp_figure_reviewer.server import mcp


def _clean_blind_payload(*, misaligned_cbar: bool = False, jargon: bool = False) -> BlindReviewPayload:
    colorbars = []
    if misaligned_cbar:
        colorbars.append(
            ColorbarReading(
                location="Top-right colorbar",
                colorbar_type="diverging",
                vmin_tick=-0.31,
                vmax_tick=0.97,
                neutral_center_tick=0.33,
                zero_is_at_white_center=False,
                has_extend_arrows=True,
            )
        )
    terms = [
        TermAuditItem(
            term="lambda_ob",
            location="x-axis",
            blind_guess="Observed optical richness",
            clarity_tag="CLEAR",
            reason="Standard richness notation.",
        )
    ]
    if jargon:
        terms.append(
            TermAuditItem(
                term="CPF=2",
                location="Title",
                blind_guess="Unknown flag",
                clarity_tag="OPAQUE_JARGON",
                reason="Internal solver flag.",
            )
        )
    return BlindReviewPayload(
        one_sentence_conclusion="Clusters show higher projected richness density than random sightlines.",
        panel_encodings=[
            PanelEncoding(
                panel_id="Main",
                panel_title="Real (CPF=2)" if jargon else "Cluster vs Random Richness Density",
                x_axis="Projected radius R [h^-1 Mpc]",
                y_axis="Richness density n_lambda",
                colors_and_colorbar="Soft RdBu_r diverging map" if misaligned_cbar else "Teal and marigold curves",
                markers_and_linestyles="Solid and dashed lines",
                annotations_and_legends="Clusters vs random sightlines",
            )
        ],
        unclear_terms=terms,
        colormap_audit=ColormapAudit(
            has_colorbar=misaligned_cbar,
            colorbars=colorbars,
            observed_palette_description="Soft RdBu_r" if misaligned_cbar else "Teal and marigold",
            colormap_role_match="APPROPRIATE",
            stretch_and_midpoint_notes="White at +0.33" if misaligned_cbar else "N/A",
            contrast_and_accessibility="Crisp contrast.",
            colormap_issues=[],
        ),
        visual_and_layout_issues=[],
        confusing_elements=[],
        standalone_readability_score=5 if not (jargon or misaligned_cbar) else 2,
        readability_summary="Clear publication plot." if not (jargon or misaligned_cbar) else "Issues found.",
    )


def _fake_phase2_payload() -> Phase2ReviewPayload:
    return Phase2ReviewPayload(
        phase2_verdict="PASS",
        score_breakdown=Phase2ScoreBreakdown(
            notation_hygiene_score=48,
            narrative_alignment_score=46,
            phase2_score=94,
        ),
        symbol_hygiene=[],
        orphaned_terms=[],
        narrative_guidance_assessment="Manuscript guides reader cleanly.",
        unreferenced_figure_elements=[],
        unsupported_text_claims=[],
        tex_edits=[],
        figure_label_fixes=[],
        caption_fixes=[],
        reviewer2_summary="Clean integration.",
    )


def _make_rainbow_image(path: Path, size: tuple[int, int] = (900, 700)) -> None:
    """Create a synthetic jet/rainbow image that deterministically trips rainbow_jet_suspected."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    bands = [
        (230, 20, 20),    # saturated red
        (240, 220, 15),   # saturated yellow
        (20, 220, 40),    # neon green
        (15, 220, 230),   # cyan
        (25, 45, 235),    # saturated blue
    ]
    bw = w // len(bands)
    for i, rgb in enumerate(bands):
        arr[:, i * bw : (i + 1) * bw] = rgb
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")


def _make_saturated_heatmap(path: Path, size: tuple[int, int] = (900, 700)) -> None:
    """Create a synthetic high-chroma dark-clipped heatmap tripping saturation_clipping_suspected."""
    w, h = size
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    # 6 multi-hue chromatic strips covering >30% area, with big dark maroon & dark navy clipped blocks
    strips = [
        (110, 5, 10),     # dark clipped maroon (L~0.22, S~0.91)
        (8, 22, 105),     # dark clipped navy (L~0.22, S~0.86)
        (185, 45, 45),
        (55, 125, 185),
        (230, 140, 110),
        (140, 190, 225),
    ]
    sh = h // len(strips)
    for i, rgb in enumerate(strips):
        arr[i * sh : (i + 1) * sh, 50 : w - 50] = rgb
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")


# ---------------------------------------------------------------------------
# Optimization #2: Zero-token deterministic chat Master grading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opt2_deterministic_grade_unlocks_phase2_with_zero_master_llm_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    fig = tmp_path / "desy3_richness_density_lamge45.png"
    Image.new("RGB", (900, 700), (245, 245, 245)).save(fig)
    tex = tmp_path / "main.tex"
    tex.write_text(r"\section{Results} Figure~\ref{fig:r} shows richness density.", encoding="utf-8")

    call_schemas: list[str] = []

    async def _fake_llm(*, schema_model, **kwargs):
        call_schemas.append(schema_model.__name__)
        if schema_model in (BlindReviewPayload, CompactBlindReviewPayload):
            return BackendCallResult(
                parsed=_clean_blind_payload(),
                backend="claude_cli",
                model="claude-sonnet-5-5",
                blinding_level="airtight_stateless_api",
                duration_seconds=0.2,
            )
        assert schema_model is Phase2ReviewPayload
        return BackendCallResult(
            parsed=_fake_phase2_payload(),
            backend="claude_cli",
            model="claude-sonnet-5-5",
            blinding_level="airtight_stateless_api",
            duration_seconds=0.2,
        )

    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=False),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", side_effect=_fake_llm),
    ):
        async with Client(mcp) as client:
            blind_res = await client.call_tool("blind_review_figure", {"image_path": str(fig)})
            review_id = blind_res.data["review_id"]

            # 1) Calling without MATCH attestation returns PENDING and keeps Phase 2 locked
            pending = await client.call_tool(
                "grade_blind_deterministic",
                {"blind_review": review_id},
            )
            assert pending.data["phase1_passed"] is False
            assert pending.data["conclusion_match"] == "PENDING"
            assert pending.data["grading_backend"] == "chat_master_deterministic"
            assert pending.data["duration_seconds"] < 0.2
            assert pending.data["chat_master_checklist"]

            p2_locked = await client.call_tool(
                "phase2_context_review",
                {
                    "image_path": str(fig),
                    "phase1_grade_id": pending.data["phase1_grade_id"],
                    "tex_path": str(tex),
                },
            )
            assert p2_locked.data["blocked"] is True

            # 2) Chat Master attests MATCH -> 0 LLM calls, unlocks Phase 2 immediately!
            approved = await client.call_tool(
                "grade_blind_deterministic",
                {
                    "blind_review": review_id,
                    "master_conclusion_match": "MATCH",
                    "conclusion_notes": "Matches cluster > random sightline density.",
                },
            )
            assert approved.data["phase1_passed"] is True
            assert approved.data["verdict"] == "PASS"
            assert approved.data["phase1_score"] == 100
            assert approved.data["phase1_grade_id"].startswith("p1_det_")

            p2_res = await client.call_tool(
                "phase2_context_review",
                {
                    "image_path": str(fig),
                    "phase1_grade_id": approved.data["phase1_grade_id"],
                    "tex_path": str(tex),
                },
            )
            assert p2_res.data["phase2_passed"] is True
            assert p2_res.data["final_score_summary"]["final_verdict"] == "READY_FOR_PAPER"

    # External MasterGradePayload LLM was NEVER called!
    assert call_schemas == ["CompactBlindReviewPayload", "Phase2ReviewPayload"]


@pytest.mark.asyncio
async def test_opt2_deterministic_grade_blocks_jargon_and_misaligned_colorbar_even_if_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    fig = tmp_path / "fig9_neg_B1B2.png"
    Image.new("RGB", (900, 700), (245, 245, 245)).save(fig)

    async def _fake_llm(*, schema_model, **kwargs):
        return BackendCallResult(
            parsed=_clean_blind_payload(misaligned_cbar=True, jargon=True),
            backend="claude_cli",
            model="claude-sonnet-5-5",
            blinding_level="airtight_stateless_api",
            duration_seconds=0.2,
        )

    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=False),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", side_effect=_fake_llm),
    ):
        rev = await run_blind_review(fig)

    grade = run_deterministic_grade(rev, master_conclusion_match="MATCH")
    assert grade.phase1_passed is False
    assert grade.verdict == "REDO_FIGURE"
    sources = {h.source for h in grade.deterministic_hits}
    assert "code_jargon_heuristic" in sources
    assert "colorbar_midpoint_misaligned" in sources
    assert grade.filename_audit.has_jargon is True
    assert "B1B2" in grade.filename_audit.jargon_tokens_found
    assert grade.filename_audit.suggested_filename == "fig9.png"


# ---------------------------------------------------------------------------
# Optimization #5: Fast-Fail 0-Token Pre-Gate before LLM calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opt5_fail_fast_short_circuits_low_res_rainbow_and_saturation_with_zero_llm_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))

    tiny = tmp_path / "tiny_fig.png"
    Image.new("RGB", (300, 200), (255, 255, 255)).save(tiny)

    rainbow = tmp_path / "rainbow_fig.png"
    _make_rainbow_image(rainbow)

    saturated = tmp_path / "saturated_heatmap.png"
    _make_saturated_heatmap(saturated)

    # Confirm inspect_figure_pixels surfaces fast_fail_blockers
    assert inspect_figure_pixels(tiny).fast_fail_blockers
    assert inspect_figure_pixels(rainbow).fast_fail_blockers
    assert inspect_figure_pixels(saturated).fast_fail_blockers

    mock_llm = AsyncMock(side_effect=AssertionError("LLM must NOT be invoked on fast-fail!"))
    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=False),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", mock_llm),
    ):
        for bad_fig in (tiny, rainbow, saturated):
            t0 = time.monotonic()
            rev = await run_blind_review(bad_fig, fail_fast=True)
            elapsed = time.monotonic() - t0
            assert rev.fast_failed is True
            assert rev.fast_fail_reasons
            assert rev.blinding_certificate.blinding_level == "deterministic_pregate"
            assert rev.review_id.startswith("blind_ff_")
            assert elapsed < 1.0

        # End-to-end also short-circuits cleanly with 0 LLM calls and blocks Phase 2
        tex = tmp_path / "main.tex"
        tex.write_text(r"\section{Test} text", encoding="utf-8")
        e2e = await run_end_to_end_review(
            rainbow,
            intended_conclusion="Some conclusion.",
            tex_path=tex,
            fail_fast=True,
        )
        assert e2e["fast_failed"] is True
        assert e2e["phase1_passed"] is False
        assert e2e["phase2_status"] == "BLOCKED_BY_PHASE1"
        assert e2e["verdict"] == "REDO_FIGURE"

    assert mock_llm.call_count == 0


@pytest.mark.asyncio
async def test_opt5_fail_fast_can_be_disabled_and_filename_jargon_is_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    jargon_fig = tmp_path / "fig9_neg_B1B2.png"
    Image.new("RGB", (900, 700), (240, 240, 240)).save(jargon_fig)

    mock_llm = AsyncMock(
        return_value=BackendCallResult(
            parsed=_clean_blind_payload(),
            backend="claude_cli",
            model="claude-sonnet-5-5",
            blinding_level="airtight_stateless_api",
            duration_seconds=0.1,
        )
    )
    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=False),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", mock_llm),
    ):
        # Default fail_fast does NOT block clean pixels just for filename jargon
        rev_ok = await run_blind_review(jargon_fig, fail_fast=True, fail_fast_on_filename_jargon=False)
        assert rev_ok.fast_failed is False
        assert "B1B2" in rev_ok.filename_jargon_tokens
        assert mock_llm.call_count == 1

        # Opt-in fail_fast_on_filename_jargon=True DOES short-circuit with 0 additional LLM calls
        rev_ff = await run_blind_review(jargon_fig, fail_fast=True, fail_fast_on_filename_jargon=True)
        assert rev_ff.fast_failed is True
        assert mock_llm.call_count == 1


# ---------------------------------------------------------------------------
# Optimization #6: 1200px WebP fast mode, Client singleton, thinking_budget=0, batch=5
# ---------------------------------------------------------------------------


def test_opt6_fast_vs_deep_image_settings_and_webp_blinding(tmp_path: Path) -> None:
    assert image_settings_for_speed("fast") == (150, 1200, "WEBP")
    assert image_settings_for_speed("deep") == (220, 1800, "PNG")

    big = tmp_path / "secret_DESY3_CPF2.png"
    img = Image.new("RGB", (2400, 1600), (220, 230, 240))
    info = PngImagePlugin.PngInfo()
    info.add_text("Title", "SECRET_METADATA_MUST_BE_STRIPPED")
    img.save(big, format="PNG", pnginfo=info)

    fast_prep = prepare_blind_image(big, dpi=150, max_dimension=1200, output_format="WEBP")
    assert fast_prep.width_px == 1200
    assert fast_prep.height_px == 800
    assert fast_prep.orig_width_px == 2400
    assert fast_prep.orig_height_px == 1600
    assert fast_prep.mime_type == "image/webp"
    assert fast_prep.blind_filename == "figure.webp"
    assert fast_prep.payload_format == "WEBP"
    assert b"SECRET_METADATA" not in fast_prep.png_bytes

    with IsolatedBlindSandbox(fast_prep) as sb:
        assert sb.figure_path is not None
        assert sb.figure_path.name == "figure.webp"
        assert [e.name for e in sb.root.iterdir()] == ["figure.webp"]


@pytest.mark.asyncio
async def test_opt6_batch_default_concurrency_five_and_mixed_fast_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    good = tmp_path / "good_fig.png"
    Image.new("RGB", (900, 700), (240, 240, 240)).save(good)
    bad = tmp_path / "rainbow_bad.png"
    _make_rainbow_image(bad)

    mock_llm = AsyncMock(
        return_value=BackendCallResult(
            parsed=_clean_blind_payload(),
            backend="claude_cli",
            model="claude-sonnet-5-5",
            blinding_level="airtight_stateless_api",
            duration_seconds=0.1,
        )
    )
    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=False),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", mock_llm),
    ):
        res = await run_batch_blind_review([str(good), str(bad), str(good)])

    assert res["max_concurrency"] == 5
    assert res["count"] == 3
    statuses = [r["status"] for r in res["summary"]]
    assert statuses == ["ok", "fast_failed", "ok"]
    # LLM was only called for the 2 good figures — NOT for the rainbow bad figure!
    # Note: because both good entries point to the exact same image, the 2nd hits cache if sequential
    # or both run concurrently (1 or 2 calls depending on task scheduling, never 3).
    assert 1 <= mock_llm.call_count <= 2


# ---------------------------------------------------------------------------
# Optimization #3 (Compact Triage Schema) & #4 (Content-Hash Cache)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opt3_compact_triage_schema_catches_jargon_and_misaligned_colorbar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    schema = dereference_json_schema(CompactBlindReviewPayload)
    assert schema["type"] == "object"
    assert "$ref" not in str(schema)

    fig = tmp_path / "compact_test.png"
    Image.new("RGB", (900, 700), (240, 240, 240)).save(fig)

    compact_payload = CompactBlindReviewPayload(
        one_sentence_conclusion="Positive correlation near center.",
        visible_labels_and_titles=["Real /NEGATIVE (CPF=2)", "R [h^-1 Mpc]"],
        flagged_terms=[
            TermAuditItem(
                term="/NEGATIVE",
                location="Title",
                blind_guess="Unknown flag",
                clarity_tag="OPAQUE_JARGON",
                reason="Code flag.",
            )
        ],
        colorbars=[
            ColorbarReading(
                location="Right colorbar",
                colorbar_type="diverging",
                vmin_tick=-0.31,
                vmax_tick=0.97,
                neutral_center_tick=0.33,
                zero_is_at_white_center=False,
            )
        ],
        observed_palette_description="Blue-white-red diverging heatmap",
        colormap_role_match="APPROPRIATE",
        colormap_issues=["White center sits at +0.33 instead of 0.0."],
        visual_and_layout_issues=[],
        confusing_elements=["Title contains code flags."],
        standalone_readability_score=2,
        readability_summary="Needs code jargon removed and colorbar centered at 0.",
    )

    async def _fake_llm(*, schema_model, **kwargs):
        assert schema_model is CompactBlindReviewPayload
        return BackendCallResult(
            parsed=compact_payload,
            backend="claude_cli",
            model="claude-sonnet-5-5",
            blinding_level="isolated_sandbox_cli",
            duration_seconds=0.2,
        )

    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=True),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", side_effect=_fake_llm),
    ):
        rev = await run_blind_review(fig, speed="fast", detail="auto")

    assert rev.detail_mode == "compact"
    grade = run_deterministic_grade(rev, master_conclusion_match="MATCH")
    assert grade.phase1_passed is False
    hit_terms = {h.term for h in grade.deterministic_hits}
    assert "/NEGATIVE" in hit_terms
    assert "CPF=2" in hit_terms
    assert any(h.source == "colorbar_midpoint_misaligned" for h in grade.deterministic_hits)


@pytest.mark.asyncio
async def test_opt4_content_hash_cache_returns_instant_hit_and_invalidates_on_caption_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    fig = tmp_path / "cached_plot.png"
    Image.new("RGB", (900, 700), (242, 244, 246)).save(fig)

    mock_llm = AsyncMock(
        return_value=BackendCallResult(
            parsed=_clean_blind_payload(),
            backend="claude_cli",
            model="claude-sonnet-5-5",
            blinding_level="isolated_sandbox_cli",
            duration_seconds=1.5,
        )
    )
    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=True),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", mock_llm),
    ):
        r1 = await run_blind_review(fig, caption="Draft caption v1", detail="full", use_cache=True)
        assert r1.cache_hit is False
        assert mock_llm.call_count == 1

        # Exact repeat -> instant cache hit, ZERO extra LLM calls!
        r2 = await run_blind_review(fig, caption="Draft caption v1", detail="full", use_cache=True)
        assert r2.cache_hit is True
        assert r2.blinding_certificate.duration_seconds < 0.2
        assert mock_llm.call_count == 1

        # Changing caption -> cache miss, invokes LLM cleanly!
        r3 = await run_blind_review(fig, caption="Revised caption v2", detail="full", use_cache=True)
        assert r3.cache_hit is False
        assert mock_llm.call_count == 2

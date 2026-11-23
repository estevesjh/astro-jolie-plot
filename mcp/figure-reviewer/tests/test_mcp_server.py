"""In-process FastMCP protocol tests for the figure-scientific-reviewer server."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client
from PIL import Image

from mcp_figure_reviewer.backends import BackendCallResult
from mcp_figure_reviewer.schemas import (
    BlindReviewPayload,
    ColormapAudit,
    ColormapFeedback,
    CompactBlindReviewPayload,
    FilenameAudit,
    MasterGradePayload,
    PanelEncoding,
    Phase1ScoreBreakdown,
    Phase2ReviewPayload,
    Phase2ScoreBreakdown,
    TermAuditItem,
)
from mcp_figure_reviewer.server import mcp


@pytest.mark.asyncio
async def test_mcp_lists_all_nine_tools() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {
            "check_backends",
            "inspect_figure_pixels",
            "blind_review_figure",
            "grade_blind_deterministic",
            "grade_blind_review",
            "phase2_context_review",
            "review_figure_end_to_end",
            "audit_figure_labels_vector",
            "blind_review_batch",
        }


@pytest.mark.asyncio
async def test_mcp_check_backends_and_pixel_audit(tmp_path: Path) -> None:
    fig = tmp_path / "sample.png"
    Image.new("RGB", (900, 600), (240, 240, 240)).save(fig)

    async with Client(mcp) as client:
        b_res = await client.call_tool("check_backends", {})
        assert "backends" in b_res.data
        assert "claude_cli" in b_res.data["backends"]
        assert "anthropic_api" in b_res.data["backends"]

        p_res = await client.call_tool("inspect_figure_pixels", {"image_path": str(fig)})
        assert p_res.data["width_px"] == 900
        assert p_res.data["height_px"] == 600
        assert p_res.data["low_resolution_warning"] is False


@pytest.mark.asyncio
async def test_mcp_blind_then_master_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    fig = tmp_path / "secret_fig_CPF2_B1.png"
    Image.new("RGB", (800, 600), (255, 255, 255)).save(fig)

    fake_blind = BlindReviewPayload(
        one_sentence_conclusion="Clusters show higher density than random sightlines.",
        panel_encodings=[
            PanelEncoding(
                panel_id="Main",
                panel_title="Real /NEGATIVE (CPF=2)",
                x_axis="R [h^-1 Mpc]",
                y_axis="n_lambda",
                colors_and_colorbar="Teal and coral lines",
                markers_and_linestyles="Solid and dashed lines",
                annotations_and_legends="B1 sample",
            )
        ],
        unclear_terms=[
            TermAuditItem(
                term="/NEGATIVE",
                location="Title",
                blind_guess="Cannot determine",
                clarity_tag="OPAQUE_JARGON",
                reason="Looks like an internal code flag.",
            )
        ],
        colormap_audit=ColormapAudit(
            has_colorbar=False,
            observed_palette_description="Teal and coral lines",
            colormap_role_match="APPROPRIATE",
            stretch_and_midpoint_notes="N/A",
            contrast_and_accessibility="High contrast.",
            colormap_issues=[],
        ),
        visual_and_layout_issues=[],
        confusing_elements=["Title contains code flags /NEGATIVE and CPF=2."],
        standalone_readability_score=2,
        readability_summary="Confusing internal code flags.",
    )
    fake_master = MasterGradePayload(
        verdict="REDO_FIGURE",
        conclusion_match="MATCH",
        conclusion_comparison_notes="Blind reviewer grasped the core physical trend.",
        score_breakdown=Phase1ScoreBreakdown(
            conclusion_accuracy_score=36,
            variable_clarity_score=10,
            visual_design_score=20,
            phase1_score=66,
        ),
        colormap_feedback=ColormapFeedback(
            matches_house_preferences="ALIGNED",
            palette_diagnosis="Uses teal and coral consistent with DECADE style.",
            stretch_and_midpoint_verdict="N/A (line plot).",
            recommended_colormap_or_palette="Keep current (decade2/decade3)",
            color_fixes=[],
        ),
        filename_audit=FilenameAudit(
            current_filename="secret_fig_CPF2_B1.png",
            has_jargon=True,
            jargon_tokens_found=["CPF2", "B1"],
            suggested_filename="stacked_richness_density_lamge60.png",
            reason="Replace internal CPF2 and B1 batch codes with physical description.",
        ),
        label_comparisons=[],
        notation_violations=["/NEGATIVE, CPF=2, and B1 are banned code flags."],
        caption_gaps=["Define n_lambda in the caption."],
        figure_code_fixes=["Replace '/NEGATIVE (CPF=2)' with 'Richness Excess (Free Central)'."],
        caption_fixes=["Define n_lambda and cluster richness cut explicitly."],
        master_summary="Redo figure to remove code flags /NEGATIVE, CPF=2, and B1.",
    )
    fake_phase2 = Phase2ReviewPayload(
        phase2_verdict="PASS",
        score_breakdown=Phase2ScoreBreakdown(
            notation_hygiene_score=46,
            narrative_alignment_score=44,
            phase2_score=90,
        ),
        symbol_hygiene=[],
        orphaned_terms=[],
        narrative_guidance_assessment="Text guides reader cleanly.",
        unreferenced_figure_elements=[],
        unsupported_text_claims=[],
        tex_edits=[],
        figure_label_fixes=[],
        caption_fixes=[],
        reviewer2_summary="Clean integration.",
    )

    async def _fake_llm(*, schema_model, prepared_image, **kwargs):
        if schema_model in (BlindReviewPayload, CompactBlindReviewPayload):
            assert prepared_image is not None
            return BackendCallResult(
                parsed=fake_blind,
                backend="claude_cli",
                model="claude-sonnet-5-5",
                blinding_level="airtight_stateless_api",
                duration_seconds=0.5,
            )
        if schema_model is MasterGradePayload:
            assert prepared_image is None
            return BackendCallResult(
                parsed=fake_master,
                backend="claude_cli",
                model="claude-sonnet-5-5",
                blinding_level="airtight_stateless_api",
                duration_seconds=0.4,
            )
        assert schema_model is Phase2ReviewPayload
        return BackendCallResult(
            parsed=fake_phase2,
            backend="claude_cli",
            model="claude-sonnet-5-5",
            blinding_level="airtight_stateless_api",
            duration_seconds=0.4,
        )

    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=False),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", side_effect=_fake_llm),
    ):
        async with Client(mcp) as client:
            blind_res = await client.call_tool(
                "blind_review_figure",
                {"image_path": str(fig), "domain": "astrophysics"},
            )
            review_id = blind_res.data["review_id"]
            assert review_id.startswith("blind_")
            assert blind_res.data["blinding_certificate"]["original_filename_hidden"] is True

            grade_res = await client.call_tool(
                "grade_blind_review",
                {
                    "blind_review": review_id,
                    "intended_conclusion": "Clusters have higher richness density than random sightlines.",
                },
            )
            assert grade_res.data["verdict"] == "REDO_FIGURE"
            assert grade_res.data["phase1_score"] == 66
            assert grade_res.data["phase1_passed"] is False
            hit_terms = {h["term"] for h in grade_res.data["deterministic_hits"]}
            assert "/NEGATIVE" in hit_terms
            assert "CPF=2" in hit_terms
            assert "B1" in hit_terms

            # Iteration Rule: Phase 2 MUST refuse a failed Phase 1 grade ID
            tex_file = tmp_path / "draft.tex"
            tex_file.write_text(r"\section{Test} Figure text.", encoding="utf-8")
            p2_blocked = await client.call_tool(
                "phase2_context_review",
                {
                    "image_path": str(fig),
                    "phase1_grade_id": grade_res.data["phase1_grade_id"],
                    "tex_path": str(tex_file),
                },
            )
            assert p2_blocked.data["blocked"] is True
            assert p2_blocked.data["phase2_status"] == "BLOCKED_BY_PHASE1"

            report_file = tmp_path / "e2e_report.md"
            e2e_res = await client.call_tool(
                "review_figure_end_to_end",
                {
                    "image_path": str(fig),
                    "intended_conclusion": "Clusters have higher richness density than random sightlines.",
                    "caption": "Stacked richness density around clusters.",
                    "tex_path": str(tex_file),
                    "test_with_caption": True,
                    "save_report_path": str(report_file),
                },
            )
            assert e2e_res.data["verdict"] == "REDO_FIGURE"
            assert e2e_res.data["phase2_status"] == "BLOCKED_BY_PHASE1"
            assert e2e_res.data["final_score"] == 66
            assert "blind_review_stage2_with_caption" in e2e_res.data
            assert report_file.is_file()
            assert "Scoreboard" in report_file.read_text(encoding="utf-8")

            batch_res = await client.call_tool(
                "blind_review_batch",
                {"image_paths": [str(fig), str(fig)]},
            )
            assert batch_res.data["count"] == 2
            assert all(row["status"] == "ok" for row in batch_res.data["summary"])


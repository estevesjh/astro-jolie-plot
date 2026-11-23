"""Unit tests for Phase 2 (.tex context extraction), 0-100 scoring, and the Iteration Rule gate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from mcp_figure_reviewer.backends import BackendCallResult
from mcp_figure_reviewer.reviewer import (
    Phase1GateBlockedError,
    run_deterministic_grade,
    run_end_to_end_review,
    run_phase2_context_review,
)
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
    ScientificContrastAudit,
    SymbolHygieneItem,
    TermAuditItem,
    VisualIssue,
    dereference_json_schema,
)
from mcp_figure_reviewer.tex_context import extract_tex_context


def _fake_blind_clean() -> BlindReviewPayload:
    return BlindReviewPayload(
        one_sentence_conclusion=(
            "Observed richness counts are accurately reproduced by convolving a Schechter member "
            "richness function with a two-state sightline projection kernel."
        ),
        panel_encodings=[
            PanelEncoding(
                panel_id="Panel (a)",
                panel_title="Observed richness density",
                x_axis="Observed richness lambda_ob",
                y_axis="Comoving number density n(lambda_ob, z) [h^3 Mpc^-3]",
                colors_and_colorbar="Black circles, coral solid curve, teal dashed, plum dotted",
                markers_and_linestyles="Data circles vs forward model curves",
                annotations_and_legends="Total model, 2-halo projected, 1-halo clean",
            )
        ],
        unclear_terms=[
            TermAuditItem(
                term="lambda_ob",
                location="Panel (a) x-axis",
                blind_guess="Observed optical richness",
                clarity_tag="CLEAR",
                reason="Explicitly defined in caption and standard cluster notation.",
            )
        ],
        colormap_audit=ColormapAudit(
            has_colorbar=False,
            observed_palette_description="Coral, teal, and plum curves",
            colormap_role_match="APPROPRIATE",
            stretch_and_midpoint_notes="N/A",
            contrast_and_accessibility="Crisp contrast.",
            colormap_issues=[],
        ),
        visual_and_layout_issues=[],
        confusing_elements=[],
        standalone_readability_score=5,
        readability_summary="Clear, uncluttered, self-contained figure with caption.",
    )


def _fake_master_payload(
    *, verdict: str = "PASS", score_parts: tuple[int, int, int] = (38, 32, 23)
) -> MasterGradePayload:
    c, v, d = score_parts
    return MasterGradePayload(
        verdict=verdict,  # type: ignore[arg-type]
        conclusion_match="MATCH" if verdict == "PASS" else "MISMATCH",
        conclusion_comparison_notes="Matches ground truth." if verdict == "PASS" else "Missed key claim.",
        score_breakdown=Phase1ScoreBreakdown(
            conclusion_accuracy_score=c,
            variable_clarity_score=v,
            visual_design_score=d,
            phase1_score=c + v + d,
        ),
        colormap_feedback=ColormapFeedback(
            matches_house_preferences="ALIGNED",
            palette_diagnosis="Matches DECADE palette.",
            stretch_and_midpoint_verdict="N/A",
            recommended_colormap_or_palette="Keep current",
            color_fixes=[],
        ),
        filename_audit=FilenameAudit(
            current_filename="good_fig.png",
            has_jargon=False,
            jargon_tokens_found=[],
            suggested_filename="good_fig.png",
            reason="Clean filename.",
        ),
        label_comparisons=[],
        notation_violations=[],
        caption_gaps=[],
        figure_code_fixes=[] if verdict == "PASS" else ["Fix confusing labels."],
        caption_fixes=[],
        master_summary="Phase 1 summary.",
    )


def _fake_phase2_payload(
    *, verdict: str = "PASS", notation_pts: int = 46, narrative_pts: int = 44
) -> Phase2ReviewPayload:
    return Phase2ReviewPayload(
        phase2_verdict=verdict,  # type: ignore[arg-type]
        score_breakdown=Phase2ScoreBreakdown(
            notation_hygiene_score=notation_pts,
            narrative_alignment_score=narrative_pts,
            phase2_score=notation_pts + narrative_pts,
        ),
        symbol_hygiene=[
            SymbolHygieneItem(
                symbol_in_figure=r"\lambda_{\mathrm{ob}}",
                symbol_in_tex_or_table=r"\lambda_{\mathrm{ob}}",
                status="ALIGNED",
                fix="None.",
            )
        ],
        orphaned_terms=[],
        narrative_guidance_assessment="Section 3.1 explicitly walks through Panels (a)-(c).",
        unreferenced_figure_elements=[],
        unsupported_text_claims=[],
        tex_edits=[],
        figure_label_fixes=[],
        caption_fixes=[],
        reviewer2_summary="Clean 1-to-1 notation alignment and strong narrative integration.",
    )


def test_phase2_schema_dereferences_cleanly() -> None:
    schema = dereference_json_schema(Phase2ReviewPayload)
    assert schema["type"] == "object"
    assert "$defs" not in str(schema)
    assert "$ref" not in str(schema)


def test_extract_tex_context_finds_macros_caption_and_referencing_paragraphs(tmp_path: Path) -> None:
    tex = tmp_path / "paper.tex"
    tex.write_text(
        r"""\documentclass{aastex701}
\newcommand{\lob}{\lambda_{\mathrm{ob}}}
\newcommand{\ltr}{\lambda_{\mathrm{1h}}}
\begin{document}
\section{Counting Method}
We model cluster counts without cosmological priors. Figure~\ref{fig:counting} shows the
observed richness density $n(\lob, z)$ alongside the deconvolved member density $n(\ltr, z)$.

\begin{figure*}[t]
\includegraphics{sec3_counting_method.pdf}
\caption{Observed richness density $n(\lob, z)$ and member density $n(\ltr, z)$.}
\label{fig:counting}
\end{figure*}

As illustrated in Panel (a) of Figure~\ref{fig:counting}, the 2-halo component dominates near threshold.
\end{document}
""",
        encoding="utf-8",
    )

    ctx = extract_tex_context(tex, image_path="sec3_counting_method.pdf")
    assert ctx.figure_label == "fig:counting"
    assert r"\lob = \lambda_{\mathrm{ob}}" in ctx.macros
    assert "Observed richness density" in (ctx.figure_caption_in_tex or "")
    assert len(ctx.referencing_paragraphs) == 2
    assert "Counting Method" in ctx.section_excerpt


def test_extract_tex_context_on_real_main_tex_if_present() -> None:
    real_tex = Path("/nonexistent/local-only-file")
    if not real_tex.is_file():
        return
    ctx = extract_tex_context(real_tex, figure_label="fig:counting_method")
    assert ctx.figure_label == "fig:counting_method"
    assert len(ctx.macros) >= 10
    assert "Cosmology-agnostic" in (ctx.figure_caption_in_tex or "")
    assert len(ctx.referencing_paragraphs) >= 1


@pytest.mark.asyncio
async def test_iteration_rule_blocks_phase2_when_phase1_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    fig = tmp_path / "bad_fig.png"
    Image.new("RGB", (800, 500), (255, 255, 255)).save(fig)
    tex = tmp_path / "main.tex"
    tex.write_text(r"\section{Test} See Fig.~\ref{fig:bad}.", encoding="utf-8")

    async def _fake_llm(*, schema_model, **kwargs):
        if schema_model in (BlindReviewPayload, CompactBlindReviewPayload):
            return BackendCallResult(
                parsed=_fake_blind_clean(),
                backend="claude_cli",
                model="claude-sonnet-5-5",
                blinding_level="airtight_stateless_api",
                duration_seconds=0.3,
            )
        if schema_model is MasterGradePayload:
            return BackendCallResult(
                parsed=_fake_master_payload(verdict="REDO_FIGURE", score_parts=(12, 20, 20)),
                backend="claude_cli",
                model="claude-sonnet-5-5",
                blinding_level="airtight_stateless_api",
                duration_seconds=0.3,
            )
        raise AssertionError("Phase 2 LLM must NEVER be invoked when Phase 1 fails!")

    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=False),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", side_effect=_fake_llm),
    ):
        res = await run_end_to_end_review(
            fig,
            intended_conclusion="Different truth.",
            tex_path=tex,
        )
        assert res["phase1_passed"] is False
        assert res["phase1_score"] == 52
        assert res["phase2_status"] == "BLOCKED_BY_PHASE1"
        assert res["phase2_score"] is None
        assert res["final_score"] == 52
        assert res["letter_grade"] == "F"
        assert res["final_verdict"] == "REDO_FIGURE"

        # Direct Phase 2 call with failed phase1_grade_id must raise Phase1GateBlockedError
        with pytest.raises(Phase1GateBlockedError):
            await run_phase2_context_review(
                fig,
                phase1_grade=res["phase1_grade_id"],
                tex_path=tex,
            )


@pytest.mark.asyncio
async def test_phase1_pass_unlocks_phase2_and_computes_final_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    fig = tmp_path / "good_fig.png"
    Image.new("RGB", (800, 500), (255, 255, 255)).save(fig)
    tex = tmp_path / "main.tex"
    tex.write_text(
        r"\section{Counting} Figure~\ref{fig:good} shows observed richness counts.",
        encoding="utf-8",
    )

    async def _fake_llm(*, schema_model, **kwargs):
        if schema_model in (BlindReviewPayload, CompactBlindReviewPayload):
            return BackendCallResult(
                parsed=_fake_blind_clean(),
                backend="claude_cli",
                model="claude-sonnet-5-5",
                blinding_level="airtight_stateless_api",
                duration_seconds=0.3,
            )
        if schema_model is MasterGradePayload:
            return BackendCallResult(
                parsed=_fake_master_payload(verdict="PASS", score_parts=(38, 32, 23)),
                backend="claude_cli",
                model="claude-sonnet-5-5",
                blinding_level="airtight_stateless_api",
                duration_seconds=0.3,
            )
        assert schema_model is Phase2ReviewPayload
        return BackendCallResult(
            parsed=_fake_phase2_payload(verdict="PASS", notation_pts=46, narrative_pts=45),
            backend="claude_cli",
            model="claude-sonnet-5-5",
            blinding_level="airtight_stateless_api",
            duration_seconds=0.4,
        )

    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=False),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", side_effect=_fake_llm),
    ):
        report_path = tmp_path / "final_report.md"
        res = await run_end_to_end_review(
            fig,
            intended_conclusion="Observed richness counts match Schechter + sightline kernel.",
            caption="Observed richness density n(lambda_ob, z).",
            tex_path=tex,
            save_report_path=report_path,
        )
        assert res["phase1_passed"] is True
        assert res["phase1_score"] == 93
        assert res["phase2_status"] == "PASSED"
        assert res["phase2_score"] == 91
        assert res["final_score"] == 92
        assert res["letter_grade"] == "A"
        assert res["final_verdict"] == "READY_FOR_PAPER"
        md = report_path.read_text(encoding="utf-8")
        assert "**FINAL SCORE** | **92/100 (A)**" in md
        assert "Phase 2: Reviewer 2 Manuscript Integration" in md


@pytest.mark.asyncio
async def test_contrast_visibility_and_unit_mismatch_hard_block_both_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A figure failing the 2-second Eye-Test or top/bottom unit match can NEVER get Grade A/PASS."""
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    fig = tmp_path / "collapsed_contrast.png"
    Image.new("RGB", (800, 500), (255, 255, 255)).save(fig)

    bad_blind = _fake_blind_clean().model_copy(
        update={
            "visual_and_layout_issues": [
                VisualIssue(
                    severity="major",
                    category="contrast_visibility",
                    location="Panel (b)",
                    description=(
                        "Optical-centered and halo-centered curves collapse invisibly (<0.2% apart) "
                        "in the mixed average; reader cannot see what mis-centering does by eye."
                    ),
                ),
                VisualIssue(
                    severity="major",
                    category="unit_norm_mismatch",
                    location="Panel (b) top vs bottom strip",
                    description=(
                        "Top panel plots self-normalized dimensionless lambda(R)/lambda(0) while "
                        "bottom residual plots raw unnormalized richness difference."
                    ),
                ),
            ]
        }
    )

    # 1. Deterministic 0-token gate MUST block & cap <= 74 even if conclusion_match='MATCH'
    det = run_deterministic_grade(bad_blind, master_conclusion_match="MATCH")
    assert det.phase1_passed is False
    assert det.verdict == "REDO_FIGURE"
    assert det.phase1_score <= 74
    assert det.phase1_letter_grade in ("C", "D", "F")
    assert any("2-second Eye-Test" in r for r in det.gate_block_reasons)

    # 2. LLM Master Gate MUST also enforce cap <= 74 & REDO_FIGURE even if LLM tries to give 98/100 PASS
    async def _fake_generous_llm(*, schema_model, **kwargs):
        if schema_model in (BlindReviewPayload, CompactBlindReviewPayload):
            return BackendCallResult(
                parsed=bad_blind,
                backend="claude_cli",
                model="claude-sonnet-5-5",
                blinding_level="airtight_stateless_api",
                duration_seconds=0.2,
            )
        assert schema_model is MasterGradePayload
        return BackendCallResult(
            parsed=_fake_master_payload(verdict="PASS", score_parts=(40, 35, 23)),
            backend="claude_cli",
            model="claude-sonnet-5-5",
            blinding_level="airtight_stateless_api",
            duration_seconds=0.2,
        )

    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=False),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", side_effect=_fake_generous_llm),
    ):
        res = await run_end_to_end_review(
            fig,
            intended_conclusion="Mis-centering profile contrast.",
            caption="Stacked richness profile comparison.",
        )
        assert res["phase1_passed"] is False
        assert res["phase1_score"] <= 74
        assert res["final_verdict"] == "REDO_FIGURE"
        sb = res["master_grade"]["score_breakdown"]
        assert sb["conclusion_accuracy_score"] + sb["variable_clarity_score"] + sb["visual_design_score"] == res["phase1_score"]


def test_wrong_plotted_variable_is_major_unless_strong_argument_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not plotting the actual variable being studied is a MAJOR hard-blocking defect unless well-justified."""
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    # Even if visual_and_layout_issues is empty, contrast_audit.plots_modeled_variable_directly=False MUST block!
    wrong_var_blind = _fake_blind_clean().model_copy(
        update={
            "contrast_audit": ScientificContrastAudit(
                curves_visibly_separated_by_eye=True,
                plots_modeled_variable_directly=False,
                top_bottom_units_matched="SINGLE_PANEL",
                contrast_audit_note="Plots self-normalized lambda(R)/lambda(0) instead of modeled lambda(R).",
            )
        }
    )

    # Without justification -> MAJOR hard block, capped <= 74, REDO_FIGURE
    g_blocked = run_deterministic_grade(wrong_var_blind, master_conclusion_match="MATCH")
    assert g_blocked.phase1_passed is False
    assert g_blocked.verdict == "REDO_FIGURE"
    assert g_blocked.phase1_score <= 74
    assert any("wrong_plotted_variable" in r for r in g_blocked.gate_block_reasons)

    # Short/weak excuse (< 20 chars) -> STILL blocked!
    g_weak = run_deterministic_grade(
        wrong_var_blind,
        master_conclusion_match="MATCH",
        allow_transformed_variable_reason="looks nicer",
    )
    assert g_weak.phase1_passed is False

    # Substantive difference-enhancing justification -> waives wrong_plotted_variable cleanly
    g_ok = run_deterministic_grade(
        wrong_var_blind,
        master_conclusion_match="MATCH",
        allow_transformed_variable_reason=(
            "Multiplying by R plots R*DeltaSigma(R) to flatten the power law and expose the 15% 1h/2h shift."
        ),
    )
    assert g_ok.phase1_passed is True
    assert g_ok.verdict == "PASS"
    assert any("Transformed variable accepted" in c for c in g_ok.chat_master_checklist)



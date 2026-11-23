"""Unit tests for deterministic Table of Notation parsing and code-jargon scanning."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_figure_reviewer.notation import (
    audit_filename_jargon,
    parse_notation_table,
    scan_blind_review_deterministically,
)
from mcp_figure_reviewer.schemas import (
    BlindReviewPayload,
    ColormapAudit,
    PanelEncoding,
    TermAuditItem,
)


def _sample_blind_review(
    *,
    title: str,
    annotations: str,
    unclear_terms: list[str],
) -> BlindReviewPayload:
    return BlindReviewPayload(
        one_sentence_conclusion="Test conclusion.",
        panel_encodings=[
            PanelEncoding(
                panel_id="Panel (a)",
                panel_title=title,
                x_axis="R [h^-1 Mpc]",
                y_axis="n_lambda [h^3 Mpc^-3]",
                colors_and_colorbar="Teal and marigold curves",
                markers_and_linestyles="Solid line and open circles",
                annotations_and_legends=annotations,
            )
        ],
        unclear_terms=[
            TermAuditItem(
                term=t,
                location="Panel (a)",
                blind_guess="Unknown",
                clarity_tag="OPAQUE_JARGON",
                reason="Looks like code shorthand.",
            )
            for t in unclear_terms
        ],
        colormap_audit=ColormapAudit(
            has_colorbar=False,
            observed_palette_description="Teal and marigold curves",
            colormap_role_match="APPROPRIATE",
            stretch_and_midpoint_notes="N/A",
            contrast_and_accessibility="Good contrast.",
            colormap_issues=[],
        ),
        visual_and_layout_issues=[],
        confusing_elements=[],
        standalone_readability_score=3,
        readability_summary="Needs cleanup.",
    )


def test_parse_markdown_notation_table_and_detect_banned_jargon(tmp_path: Path) -> None:
    md = tmp_path / "Table_of_Notation.md"
    md.write_text(
        """# Table of Notation

> Never write "uncensored", "floored", or `pole-clamped`.

## Symbols

| Symbol | Definition | Units | Meaning |
| :--- | :--- | :--- | :--- |
| $\\lambda_{\\mathrm{ob}}$ | $\\lambda_{\\mathrm{1h}} + \\lambda_{\\mathrm{2h}}$ | Galaxies | Observed richness |
| $R$ | $\\chi\\theta$ | $h^{-1}$ Mpc | Transverse separation |

## Plain Replacements

| Code Term (Do Not Use) | Plain Astrophysics Description |
| :--- | :--- |
| `REAL/NEGATIVE`, `/NEGATIVE`, `CPF=2` | Background-subtracted richness field; free central galaxy weight |
| `ON_NOHOST`, `var_H` | Host removed |
| `pedestal`, `zero-lag` | Random richness; at zero separation |
""",
        encoding="utf-8",
    )

    spec = parse_notation_table(md)
    assert len(spec.allowed_symbols) == 2
    assert "uncensored" in spec.banned_terms
    assert "/NEGATIVE" in spec.banned_terms
    assert "ON_NOHOST" in spec.banned_terms
    assert "pedestal" in spec.banned_terms

    jargon_review = _sample_blind_review(
        title="Real /NEGATIVE Corr (CPF=2, Sec. 3, B1)",
        annotations="ON_NOHOST (25p) vs uncensored pedestal from data.npz",
        unclear_terms=["/NEGATIVE", "CPF=2", "ON_NOHOST", "25p"],
    )
    hits = scan_blind_review_deterministically(jargon_review, notation_spec=spec)
    hit_terms = {h.term for h in hits}
    assert "/NEGATIVE" in hit_terms
    assert "CPF=2" in hit_terms
    assert "ON_NOHOST" in hit_terms
    assert "uncensored" in hit_terms
    assert "pedestal" in hit_terms
    assert any("Sec. 3" in t for t in hit_terms)
    assert "25p" in hit_terms


def test_clean_physical_labels_produce_zero_false_positives(tmp_path: Path) -> None:
    md = tmp_path / "Table_of_Notation.md"
    md.write_text(
        """| Code Term (Do Not Use) | Replacement |
| :--- | :--- |
| `/NEGATIVE`, `CPF=2`, `ON_NOHOST` | Background-subtracted richness field |
""",
        encoding="utf-8",
    )
    spec = parse_notation_table(md)

    clean_review = _sample_blind_review(
        title="Stacked Richness Excess Density (Host Removed)",
        annotations="Clusters (lambda >= 20), SNR = 26.7, DOF = 14, 0.35 <= z < 0.50",
        unclear_terms=[],
    )
    hits = scan_blind_review_deterministically(clean_review, notation_spec=spec)
    assert hits == []


def test_parse_real_desy3_notation_table_if_present() -> None:
    real_path = Path(
        "/nonexistent/local-only-file"
    )
    if not real_path.is_file():
        return
    spec = parse_notation_table(real_path)
    assert len(spec.allowed_symbols) >= 15
    assert any("NEGATIVE" in t for t in spec.banned_terms)
    assert any("ON_NOHOST" in t for t in spec.banned_terms)
    assert any("uncensored" in t for t in spec.banned_terms)


def test_audit_filename_jargon_catches_batch_and_flag_tokens() -> None:
    assert set(audit_filename_jargon("fig9_real_desy3_gp_2d_cov_neg_B1B2.png")) == {"neg", "B1B2"}
    assert audit_filename_jargon("fig8_real_desy3_single_clusters_2d_B1.png") == ["B1"]
    assert set(audit_filename_jargon("plot_CPF2_NOHOST_varH_v3.pdf")) == {"CPF2", "NOHOST", "varH", "v3"}
    assert audit_filename_jargon("fig9_desy3_gp_spatial_covariance_lamge45.png") == []


def test_vector_label_audit_catches_banned_words_and_stats_jargon_in_figure_labels() -> None:
    from mcp_figure_reviewer.reviewer import run_deterministic_grade
    from mcp_figure_reviewer.vector_audit import resolve_astro_math_bin, run_vector_label_audit

    if resolve_astro_math_bin() is None:
        return

    bad_review = _sample_blind_review(
        title="Random Richness Pedestal & Closure Check",
        annotations="Survival function S(M) via Palm calculus for mass-matched clusters",
        unclear_terms=[],
    )
    audit, hits = run_vector_label_audit(bad_review)
    assert audit["enabled"] is True
    assert audit["papers_indexed"] >= 150
    sources = {h.source for h in hits}
    terms_low = {h.term.lower() for h in hits}
    assert "vector_banned_word" in sources
    assert "vector_stats_jargon" in sources
    assert "pedestal" in terms_low
    assert "closure" in terms_low
    assert any("survival function" in t for t in terms_low)
    assert any("palm" in t for t in terms_low)

    # Confirm grade_blind_deterministic automatically blocks Phase 1 on vector-detected label jargon!
    grade = run_deterministic_grade(bad_review, master_conclusion_match="MATCH")
    assert grade.phase1_passed is False
    assert grade.verdict == "REDO_FIGURE"
    assert grade.vector_label_audit.get("enabled") is True
    assert any(h.source == "vector_banned_word" for h in grade.deterministic_hits)
    assert any(h.source == "vector_stats_jargon" for h in grade.deterministic_hits)


def test_vector_label_audit_passes_clean_astrophysics_labels() -> None:
    from mcp_figure_reviewer.vector_audit import resolve_astro_math_bin, run_vector_label_audit

    if resolve_astro_math_bin() is None:
        return

    clean_review = _sample_blind_review(
        title="Stacked Weak Lensing Excess Surface Mass Density \\Delta\\Sigma(R)",
        annotations="Optically selected galaxy clusters, R / R_\\lambda, two-halo term",
        unclear_terms=[],
    )
    audit, hits = run_vector_label_audit(clean_review)
    assert audit["enabled"] is True
    assert hits == []
    assert any(c["phrase"] in {"weak lensing", "galaxy clusters", "two-halo term"} for c in audit["core_astro_concepts_recognized"])


@pytest.mark.asyncio
async def test_parallel_expectation_first_blind_figure_review_uses_vector_papers_and_zero_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from fastmcp import Client
    from PIL import Image

    from mcp_figure_reviewer.backends import BackendCallResult
    from mcp_figure_reviewer.schemas import (
        CompactBlindReviewPayload,
        FigureReaderExpectationPayload,
        dereference_json_schema,
    )
    from mcp_figure_reviewer.server import mcp
    from mcp_figure_reviewer.vector_audit import resolve_astro_math_bin, run_vector_paper_selection

    if resolve_astro_math_bin() is None:
        return

    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))
    assert "$ref" not in str(dereference_json_schema(FigureReaderExpectationPayload))

    # 1. Vector paper selection works cleanly in-process (<50 ms)
    sel = run_vector_paper_selection(
        "Cluster mis-centering X-ray offset distribution in redMaPPer",
        methodology="Comparing optical centered fraction f_cen against X-ray centroids",
        top_k=3,
    )
    assert sel["enabled"] is True
    assert sel["papers_indexed"] >= 100
    assert any("Zhang" in p["citation"] and "2019" in p["citation"] for p in sel["anchor_papers"])

    # 2. Blind figure review with expectation_first=True runs Stage 2A (ZERO pixels!) || Stage 2B (pixels only!)
    fig = tmp_path / "summary_plot.png"
    Image.new("RGB", (900, 700), (245, 245, 245)).save(fig)

    fake_exp = FigureReaderExpectationPayload(
        reader_persona_summary="Zhang+19 & Costanzi+19 reader.",
        top_physical_questions=["Does optical f_cen match Zhang+19?"],
        expected_minimal_layout=["1x3 horizontal strip: projection, f_cen, b_eff/b_host"],
        expected_literature_benchmarks=["Zhang+19 X-ray f_cen = 0.75 +/- 0.08"],
        cut_or_move_to_table_or_appendix=["Move raw halo mass and SNR to a compact table."],
        confusion_traps_to_avoid=["Do not plot raw b_eff and b_host separately."],
    )
    fake_blind = _sample_blind_review(title="Clean Summary", annotations="Clusters", unclear_terms=[])

    seen_calls: list[tuple[str, bool]] = []

    async def _fake_llm(*, schema_model, prepared_image, **kwargs):
        seen_calls.append((schema_model.__name__, prepared_image is not None))
        if schema_model is FigureReaderExpectationPayload:
            assert prepared_image is None  # Airtight blinding: Stage 2A sees ZERO pixels!
            return BackendCallResult(
                parsed=fake_exp,
                backend="claude_cli",
                model="claude-sonnet-5-5",
                blinding_level="airtight_stateless_api",
                duration_seconds=0.1,
            )
        assert schema_model in (BlindReviewPayload, CompactBlindReviewPayload)
        assert prepared_image is not None
        return BackendCallResult(
            parsed=fake_blind,
            backend="claude_cli",
            model="claude-sonnet-5-5",
            blinding_level="isolated_sandbox_cli",
            duration_seconds=0.1,
        )

    with (
        patch("mcp_figure_reviewer.reviewer.backend_uses_cli_sandbox", return_value=False),
        patch("mcp_figure_reviewer.reviewer.call_structured_llm", side_effect=_fake_llm),
    ):
        async with Client(mcp) as client:
            res = await client.call_tool(
                "blind_review_figure",
                {
                    "image_path": str(fig),
                    "topic": "DES Y3 redMaPPer optical projection effects and cluster mis-centering",
                    "methodology": "Blind optical self-calibration predicting f_cen and halo bias boost",
                    "expectation_first": True,
                },
            )
            exp = res.data["reader_expectations"]
            assert exp["reader_persona_summary"] == "Zhang+19 & Costanzi+19 reader."
            assert exp["anchor_papers"]
            assert exp["papers_indexed"] >= 100

    assert ("FigureReaderExpectationPayload", False) in seen_calls
    assert any(has_img for name, has_img in seen_calls if name in ("BlindReviewPayload", "CompactBlindReviewPayload"))

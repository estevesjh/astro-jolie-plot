"""Strict Pydantic v2 schemas for two-phase blinded figure review, scoring, and grading."""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field


def dereference_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return a clean, self-contained JSON Schema with all $defs/$ref inlined."""
    raw = model_cls.model_json_schema()
    defs = raw.pop("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].rsplit("/", 1)[-1]
                target = copy.deepcopy(defs[ref_name])
                for k, v in node.items():
                    if k != "$ref":
                        target.setdefault(k, v)
                return _resolve(target)
            cleaned: dict[str, Any] = {}
            for k, v in node.items():
                if k == "title":
                    continue
                cleaned[k] = _resolve(v)
            return cleaned
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(raw)


def score_to_letter_grade(score: int) -> Literal["A", "B", "C", "D", "F"]:
    """Map a 0-100 score to an A-F letter grade."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


class BlindingCertificate(BaseModel):
    """Auditable proof of how physical blinding was enforced for Reviewer 1."""

    backend: str = Field(description="LLM backend identifier (claude_cli or anthropic_api).")
    model: str = Field(description="Exact model identifier invoked.")
    blinding_level: Literal[
        "airtight_stateless_api", "isolated_sandbox_cli", "deterministic_pregate"
    ] = Field(
        description=(
            "airtight_stateless_api: stateless HTTP API receiving stripped image bytes only "
            "(zero tools, zero filesystem); isolated_sandbox_cli: local CLI run in an "
            "ephemeral /tmp directory containing only anonymized, metadata-stripped figure image; "
            "deterministic_pregate: short-circuited by 0-token pixel gate before any LLM call."
        )
    )
    original_filename_hidden: bool = True
    metadata_stripped: bool = True
    stripped_metadata_keys: list[str] = Field(default_factory=list)
    caption_provided: bool = False
    project_context_provided: bool = False
    image_sha256: str
    image_width_px: int
    image_height_px: int
    source_format: str
    source_page: int | None = None
    payload_format: str = "PNG"
    payload_bytes: int = 0
    duration_seconds: float = 0.0


class PanelEncoding(BaseModel):
    """Visual decoding of a single panel strictly from the rendered pixels (+ caption if given)."""

    panel_id: str = Field(description="Panel identifier, e.g. 'Panel (a)', 'Left', or 'Main'.")
    panel_title: str = Field(description="Verbatim panel title, or '(none)' if absent.")
    x_axis: str = Field(description="Exact x-axis label, units, scale, and visible meaning.")
    y_axis: str = Field(description="Exact y-axis label, units, scale, and visible meaning.")
    colors_and_colorbar: str = Field(
        description="What colors, shading, or colorbars represent, including scale/ticks."
    )
    markers_and_linestyles: str = Field(
        description="What markers, curves, contours, bands, or reference lines represent."
    )
    annotations_and_legends: str = Field(
        description="All visible legend entries and in-panel text annotations."
    )


class TermAuditItem(BaseModel):
    """Audit of one variable, symbol, label, or abbreviation visible on the figure."""

    term: str = Field(description="Verbatim symbol, label, or abbreviation visible in the plot.")
    location: str = Field(description="Where it appears, e.g. 'Panel (b) title', 'Legend'.")
    blind_guess: str = Field(
        description="Honest physical meaning inferred solely from the image (+ caption if given)."
    )
    clarity_tag: Literal["CLEAR", "NEEDS_CAPTION", "OPAQUE_JARGON"] = Field(
        description=(
            "CLEAR: standard self-explanatory astrophysics/physics quantity; "
            "NEEDS_CAPTION: reasonable symbol requiring a crisp caption definition; "
            "OPAQUE_JARGON: code flag, batch ID, section ref, unnamed tuple, or confusing label."
        )
    )
    reason: str = Field(description="Why this label is clear, needs a caption, or is confusing.")


class VisualIssue(BaseModel):
    """A concrete visual readability, layout, or color-scale issue."""

    severity: Literal["critical", "major", "minor"]
    category: Literal[
        "overlap_clutter",
        "colorbar_stretch",
        "colormap_choice",
        "split_panel_symmetry",
        "missing_units",
        "tiny_font",
        "legend_collision",
        "ambiguous_encoding",
        "contrast_visibility",
        "wrong_plotted_variable",
        "unit_norm_mismatch",
        "other",
    ]
    location: str = Field(description="Specific panel or figure region affected.")
    description: str = Field(description="Precise description of the visual defect.")


class ColorbarReading(BaseModel):
    """Exact tick values read off a single colorbar so midpoint anchoring can be verified arithmetically."""

    location: str = Field(description="Which colorbar, e.g. 'Top-right colorbar' or 'Bottom-left colorbar'.")
    colorbar_type: Literal["diverging", "sequential", "categorical"]
    vmin_tick: float | None = Field(default=None, description="Exact numeric value printed at bottom/left end.")
    vmax_tick: float | None = Field(default=None, description="Exact numeric value printed at top/right end.")
    neutral_center_tick: float | None = Field(
        default=None,
        description=(
            "For diverging colorbars: the exact numeric value sitting at the pure-white/neutral "
            "center of the colorbar (read directly from the middle tick or interpolated visually)."
        ),
    )
    zero_is_at_white_center: bool | None = Field(
        default=None,
        description="True ONLY if 0 sits visually at the neutral white/cream center; False if 0 is colored blue/red.",
    )
    has_extend_arrows: bool = Field(default=False, description="True if triangular clipped-tail arrows appear at ends.")


class ColormapAudit(BaseModel):
    """Blind Reviewer 1 assessment of colormap choice, stretch, contrast, and palette clarity."""

    has_colorbar: bool = Field(description="True if any panel uses a continuous colorbar/heatmap.")
    colorbars: list[ColorbarReading] = Field(
        default_factory=list,
        description="One entry per visible colorbar with exact printed vmin, vmax, and neutral-center tick numbers.",
    )
    observed_palette_description: str = Field(
        description="Concise description of visible curve/fill/heatmap colors (e.g. blue-white-red diverging, coral/teal/gold lines)."
    )
    colormap_role_match: Literal["APPROPRIATE", "QUESTIONABLE", "MISMATCHED", "NOT_APPLICABLE"] = Field(
        description="Whether sequential vs diverging vs categorical color choice fits the plotted quantities."
    )
    stretch_and_midpoint_notes: str = Field(
        description=(
            "For colorbars/heatmaps: state the exact numeric value where the white/neutral center sits "
            "(e.g. +0.33 instead of 0.00), whether 0 looks wrongly colored blue/red, and whether tails use extend arrows."
        )
    )
    contrast_and_accessibility: str = Field(
        description="Comment on pale/vanishing hues, colorblind distinguishability, and curve/marker contrast."
    )
    colormap_issues: list[str] = Field(
        default_factory=list,
        description="Specific colormap, palette, midpoint-shift, stretch, or contrast problems observed.",
    )


class ScientificContrastAudit(BaseModel):
    """Mandatory check: eye-visible contrast, plotting the actual modeled variable, and top/bottom unit match."""

    curves_visibly_separated_by_eye: bool = Field(
        default=True,
        description=(
            "True if primary compared curves are visibly separated by eye in 2 seconds; False if they collapse "
            "invisibly on top of each other (mixed-average dilution) or an unrelated dataset curve hijacks contrast."
        ),
    )
    plots_modeled_variable_directly: bool = Field(
        default=True,
        description=(
            "True ONLY if the main y-axis plots the actual variable being studied/modeled directly (or a "
            "well-justified scaling that clearly enhances differences); False if it invents an ad-hoc "
            "normalization/rescaling (e.g. dividing each curve by its own peak) without a very good argument."
        ),
    )
    top_bottom_units_matched: Literal["MATCHED", "MISMATCHED", "SINGLE_PANEL"] = Field(
        default="SINGLE_PANEL",
        description=(
            "MATCHED if stacked main + residual/ratio panels share identical normalization & units; "
            "MISMATCHED if normalized/dimensionless top sits over raw physical-unit residuals; "
            "SINGLE_PANEL if no residual/ratio sub-panel."
        ),
    )
    contrast_audit_note: str = Field(
        default="",
        description="One crisp sentence (<=18 words) on eye-visible separation, modeled variable, and unit match.",
    )


class BlindReviewPayload(BaseModel):
    """Phase 1 — Reviewer 1 structured blind visual & variable extraction."""

    one_sentence_conclusion: str = Field(
        description="Main physical conclusion of the figure in ONE concise sentence."
    )
    panel_encodings: list[PanelEncoding] = Field(
        description="Panel-by-panel breakdown of axes, colors, markers, and annotations."
    )
    unclear_terms: list[TermAuditItem] = Field(
        description=(
            "Every variable, symbol, label, or abbreviation visible in the figure with its "
            "blind physical definition and clarity tag."
        )
    )
    colormap_audit: ColormapAudit = Field(
        description="Dedicated audit of colormap choice, colorbar stretch/midpoint, and palette contrast."
    )
    contrast_audit: ScientificContrastAudit = Field(
        default_factory=ScientificContrastAudit,
        description="Mandatory audit of 2-second eye-visible separation, plotting the actual modeled variable, and top/bottom unit match.",
    )
    visual_and_layout_issues: list[VisualIssue] = Field(
        description="Concrete visual/layout/colorbar/symmetry defects seen in the image."
    )
    confusing_elements: list[str] = Field(
        description="Ruthlessly honest bullets of anything confusing, contradictory, or unreadable."
    )
    standalone_readability_score: int = Field(
        ge=1,
        le=5,
        description="1 = unreadable without author; 3 = needs caption/polish; 5 = publication-ready.",
    )
    readability_summary: str = Field(
        description="Short 1-2 sentence overall assessment of standalone figure clarity."
    )


class CompactBlindReviewPayload(BaseModel):
    """Fast ~250-token triage schema for Reviewer 1 during interactive chats (`detail="compact"`).

    Keeps the 4 high-signal referee essentials — 1-sentence conclusion, flagged/jargon terms,
    exact colorbar ticks + top visual issues, and 1-5 readability — while skipping verbose
    panel-by-panel encodings and CLEAR term enumerations for 2-3x faster decoding.
    """

    one_sentence_conclusion: str = Field(
        description="Main physical conclusion of the figure in ONE concise sentence."
    )
    visible_labels_and_titles: list[str] = Field(
        default_factory=list,
        description="Short verbatim list of panel titles, axis labels, and legend labels (<=12 strings).",
    )
    flagged_terms: list[TermAuditItem] = Field(
        default_factory=list,
        description=(
            "ONLY terms tagged OPAQUE_JARGON or NEEDS_CAPTION (skip CLEAR standard symbols). "
            "Include code flags, batch IDs, undefined symbols, or section refs."
        ),
    )
    colorbars: list[ColorbarReading] = Field(
        default_factory=list,
        description="Exact printed vmin_tick, vmax_tick, and neutral_center_tick for every visible colorbar.",
    )
    observed_palette_description: str = Field(
        description="Ultra-short (<=12 words) description of visible curve/heatmap colors."
    )
    colormap_role_match: Literal["APPROPRIATE", "QUESTIONABLE", "MISMATCHED", "NOT_APPLICABLE"]
    colormap_issues: list[str] = Field(
        default_factory=list,
        description="Specific midpoint-shift, saturation clipping, rainbow, or contrast issues (if any).",
    )
    visual_and_layout_issues: list[VisualIssue] = Field(
        default_factory=list,
        description="Only actual visual/layout/overlap/colorbar/contrast/variable defects (empty if clean).",
    )
    contrast_audit: ScientificContrastAudit = Field(
        default_factory=ScientificContrastAudit,
        description="Mandatory audit of 2-second eye-visible separation, plotting the actual modeled variable, and top/bottom unit match.",
    )
    confusing_elements: list[str] = Field(
        default_factory=list,
        description="Top 1-3 concise bullets of anything confusing or unreadable (empty if clean).",
    )
    standalone_readability_score: int = Field(
        ge=1,
        le=5,
        description="1 = unreadable without author; 3 = needs caption/polish; 5 = publication-ready.",
    )
    readability_summary: str = Field(
        description="One concise sentence assessing standalone figure clarity."
    )

    def to_full_payload(self) -> BlindReviewPayload:
        """Convert compact triage output into a compatible BlindReviewPayload."""
        labels_joined = "; ".join(self.visible_labels_and_titles[:12]) if self.visible_labels_and_titles else "(compact triage mode)"
        panel_enc = (
            [
                PanelEncoding(
                    panel_id="All panels (compact triage)",
                    panel_title=labels_joined,
                    x_axis="(captured in visible_labels_and_titles)",
                    y_axis="(captured in visible_labels_and_titles)",
                    colors_and_colorbar=self.observed_palette_description,
                    markers_and_linestyles="(compact triage mode)",
                    annotations_and_legends=labels_joined,
                )
            ]
            if self.visible_labels_and_titles
            else []
        )
        stretch_note = (
            "; ".join(self.colormap_issues)
            if self.colormap_issues
            else (
                "Colorbar ticks extracted in compact triage mode."
                if self.colorbars
                else "N/A"
            )
        )
        return BlindReviewPayload(
            one_sentence_conclusion=self.one_sentence_conclusion,
            panel_encodings=panel_enc,
            unclear_terms=self.flagged_terms,
            colormap_audit=ColormapAudit(
                has_colorbar=bool(self.colorbars),
                colorbars=self.colorbars,
                observed_palette_description=self.observed_palette_description,
                colormap_role_match=self.colormap_role_match,
                stretch_and_midpoint_notes=stretch_note,
                contrast_and_accessibility=self.readability_summary,
                colormap_issues=self.colormap_issues,
            ),
            contrast_audit=self.contrast_audit,
            visual_and_layout_issues=self.visual_and_layout_issues,
            confusing_elements=self.confusing_elements,
            standalone_readability_score=self.standalone_readability_score,
            readability_summary=self.readability_summary,
        )


class ColorForensics(BaseModel):
    """Zero-token deterministic pixel color analysis."""

    dominant_chromatic_hexes: list[str] = Field(default_factory=list)
    chromatic_pixel_fraction: float = 0.0
    dead_pale_pixel_fraction: float = 0.0
    saturated_dark_extreme_fraction: float = 0.0
    likely_heatmap_or_colorbar: bool = False
    saturation_clipping_suspected: bool = False
    tab10_suspected: bool = False
    rainbow_jet_suspected: bool = False
    matched_house_palettes: list[str] = Field(default_factory=list)
    color_notes: list[str] = Field(default_factory=list)


class FigureReaderExpectationPayload(BaseModel):
    """Stage 2A Blind Pre-Figure Reader Expectation (ZERO image pixels; grounded in vector-selected papers)."""

    reader_persona_summary: str = Field(
        description="One concise sentence naming the vector-selected anchor papers and reader lens."
    )
    top_physical_questions: list[str] = Field(
        default_factory=list,
        description="Top 3-4 core physical questions these readers immediately want the figure to answer.",
    )
    expected_minimal_layout: list[str] = Field(
        default_factory=list,
        description="Minimal panel count, axes, and curves needed to maximize understanding in minimal page space (<=4 bullets).",
    )
    expected_literature_benchmarks: list[str] = Field(
        default_factory=list,
        description="Concrete benchmarks, bands, or scaling expectations from the anchor papers that readers expect to compare against.",
    )
    cut_or_move_to_table_or_appendix: list[str] = Field(
        default_factory=list,
        description="Redundant/secondary quantities that cause panel overload and belong in a table or appendix.",
    )
    confusion_traps_to_avoid: list[str] = Field(
        default_factory=list,
        description="Definition mismatches, twin-axis traps, or apples-to-apples pitfalls (<=4 bullets).",
    )


class FigureReaderExpectation(FigureReaderExpectationPayload):
    """Persisted Stage 2A Blind Pre-Figure Expectation with vector paper-selection provenance."""

    expectation_id: str
    topic: str
    methodology: str = ""
    anchor_papers: list[dict[str, Any]] = Field(default_factory=list)
    papers_indexed: int = 0
    paper_selection_ms: float = 0.0
    cache_hit: bool = False
    backend: str
    model: str
    duration_seconds: float = 0.0


class BlindFigureReview(BlindReviewPayload):
    """Complete Phase 1 Reviewer 1 output with review_id, color forensics, optional expectations, and blinding certificate."""

    review_id: str
    round_number: int = 1
    source_filename: str | None = Field(
        default=None,
        description="Recorded AFTER blind review for Master filename hygiene audit; NEVER shown to Reviewer 1.",
    )
    filename_jargon_tokens: list[str] = Field(
        default_factory=list,
        description="Opaque batch/code tokens detected deterministically in source_filename (never shown to Reviewer 1).",
    )
    fast_failed: bool = Field(
        default=False,
        description="True if 0-token pixel pre-gate short-circuited before invoking Reviewer 1 LLM.",
    )
    fast_fail_reasons: list[str] = Field(
        default_factory=list,
        description="Hard deterministic pixel/colormap defects that triggered fast-fail.",
    )
    detail_mode: Literal["compact", "full"] = Field(
        default="full",
        description="'compact': fast ~250-token triage schema; 'full': full panel-by-panel audit.",
    )
    cache_hit: bool = Field(
        default=False,
        description="True if served instantaneously (0.0s, 0 LLM tokens) from content-hash cache.",
    )
    color_forensics: ColorForensics = Field(default_factory=ColorForensics)
    reader_expectations: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional Stage 2A Pre-Figure Blind Reader Expectations grounded in vector-selected anchor papers (ZERO pixels seen).",
    )
    blinding_certificate: BlindingCertificate


class LabelComparison(BaseModel):
    """Master comparison between Reviewer 1's blind guess and project ground truth."""

    term: str
    blind_guess: str
    intended_meaning: str
    status: Literal[
        "MATCH",
        "PARTIAL",
        "MISMATCH",
        "MISSING_FROM_CAPTION",
        "CODE_JARGON",
        "INVENTED_TERM",
        "NOTATION_COLLISION",
    ]
    recommended_fix: str


class Phase1ScoreBreakdown(BaseModel):
    """Explicit 0-100 rubric breakdown for Phase 1 (Visual Clarity)."""

    conclusion_accuracy_score: int = Field(ge=0, le=40, description="0-40 pts: physical message match.")
    variable_clarity_score: int = Field(ge=0, le=35, description="0-35 pts: clear physical variables, zero jargon.")
    visual_design_score: int = Field(ge=0, le=25, description="0-25 pts: layout, overlaps, colorbar stretch.")
    phase1_score: int = Field(ge=0, le=100, description="Total Phase 1 score (sum of sub-scores, 0-100).")


class ColormapFeedback(BaseModel):
    """Master Gate critique of colormap & palette alignment with house preferences."""

    matches_house_preferences: Literal["ALIGNED", "NEEDS_TUNING", "NON_COMPLIANT"]
    palette_diagnosis: str = Field(
        description="Assessment of curve/fill/heatmap palette against the active color preferences."
    )
    stretch_and_midpoint_verdict: str = Field(
        description="Assessment of diverging midpoint anchoring (0 or median) and central percentile stretch."
    )
    recommended_colormap_or_palette: str = Field(
        description="Specific house colormap/palette recommendation (e.g. tyrian_div, pal4, vol1-164, wada164_safe) or 'Keep current'."
    )
    color_fixes: list[str] = Field(
        default_factory=list,
        description="Concrete colormap, norm, percentile-stretch, or palette edits.",
    )


class FilenameAudit(BaseModel):
    """Master audit of the figure filename itself to catch opaque batch/code suffixes (e.g. B1B2, neg)."""

    current_filename: str
    has_jargon: bool
    jargon_tokens_found: list[str] = Field(default_factory=list)
    suggested_filename: str = Field(
        description="Clean, self-explanatory physical filename keeping the same extension (or identical if already clean)."
    )
    reason: str


class MasterGradePayload(BaseModel):
    """Phase 1 Master Gate evaluation comparing Reviewer 1 vs. intended physical truth."""

    verdict: Literal["PASS", "REVISE_CAPTION", "REDO_FIGURE"] = Field(
        description=(
            "PASS: Reviewer 1 understood the physics, visuals/colorbars are clean, and labels/"
            "caption are clear; REVISE_CAPTION: figure pixels are clean, only caption needs work; "
            "REDO_FIGURE: physical conclusion mismatch, confusing visuals/colorbars, or code/"
            "invented jargon requires editing the plotting script."
        )
    )
    conclusion_match: Literal["MATCH", "PARTIAL_MISMATCH", "MISMATCH"]
    conclusion_comparison_notes: str = Field(
        description="Exact comparison between Reviewer 1's conclusion and intended physical truth."
    )
    score_breakdown: Phase1ScoreBreakdown
    colormap_feedback: ColormapFeedback = Field(
        description="Dedicated critique of colormap, stretch, saturation avoidance, and palette preferences."
    )
    filename_audit: FilenameAudit = Field(
        description="Audit of the figure filename itself, proposing a clean physical rename if code/batch jargon (e.g. B1B2, neg) appears."
    )
    label_comparisons: list[LabelComparison] = Field(
        description="Item-by-item truth check of variables, labels, and panels."
    )
    notation_violations: list[str] = Field(
        description="Code jargon, banned words, invented labels, or Table of Notation collisions."
    )
    caption_gaps: list[str] = Field(
        description="Symbols or conventions needed in the caption that are currently missing."
    )
    figure_code_fixes: list[str] = Field(
        description="Concrete plotting-script changes needed before Phase 1 can pass."
    )
    caption_fixes: list[str] = Field(
        description="Concrete caption edits or symbol definitions to add."
    )
    master_summary: str = Field(
        description="Concise 2-4 sentence executive verdict for the author."
    )


class DeterministicAuditHit(BaseModel):
    """A zero-token deterministic finding from notation, jargon, vector field audit, or colorbar-midpoint scanning."""

    term: str
    source: Literal[
        "notation_table_banned",
        "explicit_banned",
        "code_jargon_heuristic",
        "colorbar_midpoint_misaligned",
        "vector_banned_word",
        "vector_stats_jargon",
        "vector_symbol_collision",
    ]
    location: str
    detail: str


class MasterGradeResult(MasterGradePayload):
    """Persisted Phase 1 Master Gate result, controlling unlock of Phase 2."""

    conclusion_match: Literal["MATCH", "PARTIAL_MISMATCH", "MISMATCH", "PENDING"]
    phase1_grade_id: str
    review_id: str | None = None
    phase1_score: int = Field(ge=0, le=100)
    phase1_letter_grade: Literal["A", "B", "C", "D", "F"]
    phase1_passed: bool = Field(
        description="Hard gate: Phase 2 (Manuscript Integration) only unlocks when True."
    )
    gate_block_reasons: list[str] = Field(
        default_factory=list,
        description="Exact reasons Phase 1 blocked progression to Phase 2 (empty if passed).",
    )
    pass_threshold: int = 80
    style_profile: str = "esteves"
    color_forensics: ColorForensics = Field(default_factory=ColorForensics)
    deterministic_hits: list[DeterministicAuditHit] = Field(default_factory=list)
    vector_label_audit: dict[str, Any] = Field(
        default_factory=dict,
        description="Empirical 160-paper vectorized label, banned-word, stats-jargon & symbol-collision audit.",
    )
    grading_mode: Literal["llm_master", "chat_master_deterministic", "deterministic_pregate"] = (
        "llm_master"
    )
    blind_one_sentence_conclusion: str | None = Field(
        default=None,
        description="Reviewer 1's blind 1-sentence conclusion echoed for chat-Master verification.",
    )
    chat_master_checklist: list[str] = Field(
        default_factory=list,
        description="Actionable verification items for the calling chat agent acting as Master.",
    )
    grading_backend: str
    grading_model: str
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Phase 2: Context Loop (Manuscript Integration — Reviewer 2)
# ---------------------------------------------------------------------------

class SymbolHygieneItem(BaseModel):
    """1-to-1 notation alignment check for a single symbol across figure, caption, and .tex."""

    symbol_in_figure: str
    symbol_in_tex_or_table: str = Field(
        description="Matching canonical symbol in .tex / Table of Notation, or '(missing)'."
    )
    status: Literal[
        "ALIGNED",
        "MISMATCH",
        "ORPHANED_IN_FIGURE",
        "AD_HOC_VARIABLE",
        "UNIT_MISMATCH",
        "CODE_JARGON",
    ]
    fix: str = Field(description="Exact edit to figure label, caption, or .tex.")


class Phase2ScoreBreakdown(BaseModel):
    """Explicit 0-100 rubric breakdown for Phase 2 (Manuscript Integration)."""

    notation_hygiene_score: int = Field(
        ge=0, le=50, description="0-50 pts: strict 1-to-1 symbol/unit alignment with .tex & Table of Notation."
    )
    narrative_alignment_score: int = Field(
        ge=0, le=50, description="0-50 pts: .tex explicitly guides reader through figure & figure advances argument."
    )
    phase2_score: int = Field(
        ge=0, le=100, description="Total Phase 2 score (sum of sub-scores, 0-100)."
    )


class Phase2ReviewPayload(BaseModel):
    """Phase 2 — Reviewer 2 structured notation hygiene and narrative integration audit."""

    phase2_verdict: Literal["PASS", "REVISE_TEX", "REVISE_CAPTION", "REDO_FIGURE"] = Field(
        description=(
            "PASS: strict notation match + narrative smoothly guides reader through figure; "
            "REVISE_TEX: figure is clean, update .tex text to align narrative/symbols; "
            "REVISE_CAPTION: update caption definitions; "
            "REDO_FIGURE: figure labels conflict with manuscript notation and must be edited."
        )
    )
    score_breakdown: Phase2ScoreBreakdown
    symbol_hygiene: list[SymbolHygieneItem] = Field(
        description="Symbol-by-symbol verification between figure, caption, .tex, and notation table."
    )
    orphaned_terms: list[str] = Field(
        description="Terms or ad-hoc variables appearing in the figure/caption but never defined or used in .tex."
    )
    narrative_guidance_assessment: str = Field(
        description="Does the .tex text explicitly guide the reader through each panel and advance the argument?"
    )
    unreferenced_figure_elements: list[str] = Field(
        description="Panels, curves, or insets shown in the figure that the .tex prose ignores or fails to explain."
    )
    unsupported_text_claims: list[str] = Field(
        description="Claims in the .tex prose attributed to this figure that are not clearly supported by the plot."
    )
    tex_edits: list[str] = Field(description="Concrete .tex prose or macro edits.")
    figure_label_fixes: list[str] = Field(description="Concrete figure label/symbol fixes.")
    caption_fixes: list[str] = Field(description="Concrete caption fixes.")
    reviewer2_summary: str = Field(description="Concise 2-4 sentence Phase 2 integration verdict.")


class FinalScoreSummary(BaseModel):
    """Combined publication-readiness score across Phase 1 (Blind) and Phase 2 (Context)."""

    phase1_score: int = Field(ge=0, le=100)
    phase1_passed: bool
    phase2_score: int | None = Field(default=None, ge=0, le=100)
    phase2_passed: bool | None = None
    final_score: int = Field(ge=0, le=100)
    letter_grade: Literal["A", "B", "C", "D", "F"]
    final_verdict: Literal[
        "READY_FOR_PAPER",
        "REVISE_TEX",
        "REVISE_CAPTION",
        "REDO_FIGURE",
    ]
    scoring_formula: str


class Phase2ReviewResult(Phase2ReviewPayload):
    """Complete Phase 2 output with combined Final Score and gate provenance."""

    phase2_review_id: str
    phase1_grade_id: str
    phase2_score: int = Field(ge=0, le=100)
    phase2_letter_grade: Literal["A", "B", "C", "D", "F"]
    phase2_passed: bool
    gate_overridden: bool = False
    final_score_summary: FinalScoreSummary
    tex_path: str
    figure_label_matched: str | None = None
    backend: str
    model: str
    duration_seconds: float = 0.0


class PixelAuditResult(BaseModel):
    """Zero-token deterministic image & metadata pre-audit."""

    image_path: str
    source_format: str
    page: int | None = None
    width_px: int
    height_px: int
    aspect_ratio: float
    file_size_bytes: int
    image_sha256: str
    embedded_metadata_keys: list[str] = Field(
        description="Metadata keys embedded in the source file that will be stripped before blinding."
    )
    near_white_pixel_fraction: float
    near_black_pixel_fraction: float
    low_resolution_warning: bool
    warnings: list[str] = Field(default_factory=list)
    fast_fail_blockers: list[str] = Field(
        default_factory=list,
        description="Hard 0-token pixel/colormap defects that trigger fail_fast short-circuit before LLM calls.",
    )
    filename_jargon_tokens: list[str] = Field(default_factory=list)
    color_forensics: ColorForensics = Field(default_factory=ColorForensics)
    diverging_percentile_guide: dict[str, str] = Field(
        default_factory=lambda: {
            "98%": "percentile(x, 1.0) to percentile(x, 99.0)",
            "95% (default)": "percentile(x, 2.5) to percentile(x, 97.5)",
            "91%": "percentile(x, 4.5) to percentile(x, 95.5)",
            "68%": "percentile(x, 16.0) to percentile(x, 84.0)",
        }
    )

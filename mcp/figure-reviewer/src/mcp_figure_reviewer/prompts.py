"""Distinct system & user prompts for:
- Phase 1 / Reviewer 1 (Blind Visual Clarity Reviewer — figure + optional caption ONLY)
- Phase 1 / Master Gate (Truth vs. Blind Evaluator + 0-100 Phase 1 Score)
- Phase 2 / Reviewer 2 (Manuscript Context, Notation Hygiene & Narrative Alignment Reviewer)
"""

from __future__ import annotations

import json
from typing import Any

from .colormap import get_color_preferences_text
from .notation import NotationSpec
from .schemas import BlindReviewPayload, ColorForensics, DeterministicAuditHit, MasterGradeResult
from .tex_context import ExtractedTexContext
from .vector_audit import format_vector_audit_for_prompt


# ---------------------------------------------------------------------------
# Phase 1: Reviewer 1 (Blind Visual Clarity Reviewer)
# ---------------------------------------------------------------------------

def build_blind_system_prompt(domain: str = "astrophysics") -> str:
    """Distinct system prompt for Reviewer 1 (Phase 1: Blind Loop)."""
    clean_domain = (domain or "scientific").strip()
    return (
        f"You are Reviewer 1: a blind scientific reviewer. You know nothing about this specific "
        f"{clean_domain} project.\n\n"
        "Strict Phase 1 blinding rules:\n"
        "- Base EVERY statement strictly on what is explicitly visible in the attached figure image "
        "(and caption, ONLY if a caption is explicitly provided below).\n"
        "- Do NOT guess beyond what is explicitly shown.\n"
        "- Do NOT invent project context, paper sections, code meanings, or unstated definitions.\n"
        "- Be ruthlessly honest if anything is confusing, undefined, cluttered, visually broken, "
        "or reads like internal code jargon.\n"
        "- Return valid JSON matching the required schema."
    )


def build_blind_user_prompt(
    *,
    caption: str | None = None,
    extra_checks: list[str] | None = None,
    cli_image_filename: str | None = None,
) -> str:
    """Build Reviewer 1's Phase 1 blind visual-clarity prompt."""
    parts: list[str] = []
    if cli_image_filename:
        parts.append(
            f"IMPORTANT SANDBOX RULE: Inspect ONLY `{cli_image_filename}` in the current working "
            "directory using your file-reading tool. Do NOT read, list, search, or access any "
            "other file or directory anywhere on disk, and do NOT invoke any MCP tools, "
            "subagents, or skills. Be concise and output the required JSON immediately.\n"
        )

    parts.append(
        "**Phase 1 Task (Visual Clarity):** Analyze the attached figure.\n"
        "1. State the main physical conclusion of the plot in one sentence "
        "(`one_sentence_conclusion`).\n"
        "2. Identify what the axes, colors, markers, linestyles, colorbars, and annotations "
        "represent based only on the image"
        + (" and the provided caption" if caption else "")
        + ", panel by panel (`panel_encodings`).\n"
        "3. List and define ALL variables, labels, symbols, abbreviations, or terms shown in "
        "the figure (`unclear_terms`), explicitly flagging any that lack a clear physical "
        "meaning. For each, record the verbatim term, its location, your honest blind physical "
        "definition, and tag it as:\n"
        "   - `CLEAR` (self-explanatory physics/statistics meaning clear from the figure"
        + (" and caption" if caption else "")
        + "),\n"
        "   - `NEEDS_CAPTION` (reasonable symbol/quantity still missing a crisp definition), or\n"
        "   - `OPAQUE_JARGON` (undefined code flag, batch ID, internal section number, unnamed "
        "parameter tuple, cryptic abbreviation, or label lacking physical intuition).\n"
        "4. Audit the colormap, palette, and color scaling (`colormap_audit`), commenting explicitly on:\n"
        "   - For EVERY colorbar visible in the figure, fill an entry in `colormap_audit.colorbars` "
        "by reading the EXACT printed tick numbers verbatim: `vmin_tick` (bottom/left), `vmax_tick` "
        "(top/right), and `neutral_center_tick` (the EXACT number printed next to the pure-white/"
        "neutral band in the middle of a diverging colorbar — do NOT assume it is 0; if the middle "
        "tick printed beside white is `+0.33` or `+0.41`, record `0.33` or `0.41` and set "
        "`zero_is_at_white_center=false`!).\n"
        "   - Whether sequential vs. diverging vs. categorical colors fit the plotted physics (flag "
        "rainbow/jet, harsh muddy dark-navy/brick-red hues, or default clashing colors).\n"
        "   - **Saturation check:** Flag any clipped/saturated flat dark-red or dark-blue plateaus in "
        "heatmaps where internal gradients vanish because the color limits are too tight or the "
        "colormap endpoints crush to near-black.\n"
        "   - Contrast, pale/vanishing fills on white, and colorblind distinguishability.\n"
        "5. Audit visual readability, layout & scientific contrast (`visual_and_layout_issues` AND mandatory `contrast_audit`), remembering that **the goal of a scientific plot is to enhance the differences between things**:\n"
        "   - **Scientific Contrast / 2-Second Eye-Test (`contrast_visibility` + `contrast_audit.curves_visibly_separated_by_eye`):** If the figure or caption claims to show or compare a systematic/physical effect (e.g. with vs. without an effect, centered vs. mis-centered, baseline vs. upgrade), can you actually SPOT the difference by eye in 2 seconds from the main curves? Set `curves_visibly_separated_by_eye=false` and flag as `major`/`critical` (`contrast_visibility`) if primary comparison curves collapse invisibly on top of each other (e.g. mixed-average dilution) or an unrelated dataset/operator curve visually hijacks the comparison.\n"
        "   - **Plot the Actual Modeled Variable (`wrong_plotted_variable` + `contrast_audit.plots_modeled_variable_directly`):** **Not plotting the actual variable being studied/modeled directly is a `MAJOR` defect — only acceptable with a very good argument that explicitly enhances differences.** Take the modeled equation and plot it directly; set `plots_modeled_variable_directly=false` and flag `severity='major', category='wrong_plotted_variable'` if the y-axis invents an ad-hoc rescaling or self-normalization (like dividing each curve by its own peak $y(0)$) without a compelling difference-enhancing justification.\n"
        "   - **Top/Bottom Normalization & Unit Match (`unit_norm_mismatch` + `contrast_audit.top_bottom_units_matched`):** For any stacked main + residual/ratio panel, verify both panels share the EXACT same normalization and units (raw $y$ + $\\Delta y$, OR normalized $y/y_0$ + fractional ratio). Set `top_bottom_units_matched='MISMATCHED'` and flag `severity='major', category='unit_norm_mismatch'` if a dimensionless self-normalized top panel sits above raw physical-unit residuals.\n"
        "   - overlapping text, legends colliding with curves or vertical/horizontal reference lines,\n"
        "   - cramped stat boxes, excessive panel crowding, tiny or clipped labels, missing units,\n"
        "   - unexplained split-triangle / asymmetric matrix quadrants or broken colorbar stretches.\n"
        "6. List anything confusing or contradictory (`confusing_elements`) and rate standalone "
        "readability from 1 (unreadable without author) to 5 (publication-ready self-contained).\n\n"
        "Do not guess beyond what is explicitly shown. Be ruthlessly honest if something is confusing."
    )

    if caption:
        parts.append(
            "\n**Figure Caption Provided to Reviewer 1:**\n"
            f"\"\"\"\n{caption.strip()}\n\"\"\"\n"
            "Judge whether the figure + caption together stand on their own without needing the "
            "manuscript text. Still flag anything undefined, contradictory, or code-jargon."
        )
    else:
        parts.append(
            "\n**No caption is provided.** Judge the figure strictly from its rendered pixels alone."
        )

    if extra_checks:
        bullet_checks = "\n".join(f"- {c}" for c in extra_checks if c.strip())
        if bullet_checks:
            parts.append(f"\n**Additional visual/clarity checks:**\n{bullet_checks}")

    return "\n".join(parts)


def build_compact_blind_user_prompt(
    *,
    caption: str | None = None,
    extra_checks: list[str] | None = None,
    cli_image_filename: str | None = None,
) -> str:
    """Build Reviewer 1's fast ~250-token compact triage prompt (`detail="compact"`)."""
    parts: list[str] = []
    if cli_image_filename:
        parts.append(
            f"SANDBOX RULE: Read ONLY `{cli_image_filename}` in cwd using `read_file`. "
            "Do NOT touch other files, MCP tools, subagents, or skills. Return terse JSON immediately.\n"
        )

    parts.append(
        "**Phase 1 Fast Triage (keep total JSON under ~250 tokens):**\n"
        "1. `one_sentence_conclusion`: Main physical conclusion in ONE crisp sentence.\n"
        "2. `visible_labels_and_titles`: Copy <=12 short verbatim panel titles, axis labels, or legend strings.\n"
        "3. `flagged_terms`: List ONLY `OPAQUE_JARGON` (code flags, batch IDs, section refs, cryptic tags) "
        "or `NEEDS_CAPTION` (undefined symbols). SKIP all `CLEAR` standard symbols!\n"
        "4. `colorbars` + `colormap_issues`: For every colorbar, read exact printed `vmin_tick`, `vmax_tick`, "
        "and `neutral_center_tick` (exact number next to pure white; set `zero_is_at_white_center=false` if "
        "0 is tinted!). Flag rainbow/jet, off-zero white center, or saturated clipped dark plateaus.\n"
        "5. `contrast_audit` (MANDATORY) + `visual_and_layout_issues` & `confusing_elements`: Remember the goal of a plot is to "
        "**enhance differences between things** by plotting the **actual modeled variable** directly. Fill `contrast_audit` "
        "(`curves_visibly_separated_by_eye`, `plots_modeled_variable_directly`, `top_bottom_units_matched`, `contrast_audit_note`) "
        "and flag `major` defects in `visual_and_layout_issues` if: (a) primary compared curves collapse invisibly or an unrelated "
        "curve hijacks contrast (`contrast_visibility`); (b) the main y-axis does NOT plot the actual modeled variable being studied "
        "(e.g. ad-hoc self-normalization $y/y(0)$ without a very good difference-enhancing argument: `wrong_plotted_variable`, **MAJOR**); "
        "(c) stacked top/bottom panels mismatch normalization/units (`unit_norm_mismatch`, **MAJOR**); or (d) overlaps/tiny fonts/broken layouts. Keep each bullet <=15 words.\n"
        "6. `standalone_readability_score` (1-5) + 1-sentence `readability_summary`."
    )

    if caption:
        parts.append(f"\n**Caption provided:**\n\"\"\"\n{caption.strip()}\n\"\"\"")
    else:
        parts.append("\n**No caption provided** — judge strictly from rendered pixels.")

    if extra_checks:
        bullet_checks = "\n".join(f"- {c}" for c in extra_checks if c.strip())
        if bullet_checks:
            parts.append(f"\n**Extra checks:**\n{bullet_checks}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 1: Master Gate Evaluator (Truth vs. Reviewer 1 + 0-100 Score)
# ---------------------------------------------------------------------------

def build_master_system_prompt(pass_threshold: int = 80) -> str:
    """System prompt for the unblinded Master evaluating Phase 1 (Reviewer 1)."""
    return (
        "You are the Master Scientific Figure Referee evaluating Phase 1 (The Blind Loop).\n"
        "Compare Reviewer 1's physically blinded extraction against the author's ground truth, "
        "draft caption, and Table of Notation.\n\n"
        "Explicit Phase 1 Scoring Rubric (0-100 total):\n"
        "- `conclusion_accuracy_score` (0-40): Did Reviewer 1 extract the true physical takeaway?\n"
        "  * MATCH = 34-40 | PARTIAL_MISMATCH = 15-28 | MISMATCH = 0-14.\n"
        "- `variable_clarity_score` (0-35): Are all axes, symbols, and labels physically clear?\n"
        "  * Deduct heavily (-8 to -15 each) for code flags, batch IDs, internal section numbers, "
        "invented labels outside the Table of Notation, or symbol collisions.\n"
        "  * Deduct moderately (-3 to -6 each) for physics symbols undefined in the caption.\n"
        "- `visual_design_score` (0-25): Are panels uncluttered, legends clear of lines/curves, "
        "units present, split-panel symmetries sound, **claimed systematics/contrasts visibly clear by eye "
        "(clean Effect OFF vs. Effect ON contrast; no overlapping collapsed curves or hijacking unrelated curves)**, "
        "**stacked top and bottom residual/ratio panels matched in normalization and units**, and colors/colormaps "
        "aligned with the active color & colormap preferences (no rainbow/jet, no default tab10, no dead-pale fills, "
        "properly anchored diverging midpoints + central percentile stretch {98%, 95%, 91%, 68%} "
        "with extend='both')?\n"
        "  * Deduct heavily (-10 to -15) and cap `phase1_score <= 74` (cannot be Grade A/PASS) if Reviewer 1 flags "
        "`major`/`critical` `contrast_visibility` (claimed systematic is invisible by eye / diluted in mixed average "
        "/ hijacked), `wrong_plotted_variable` (**not plotting the actual variable being studied is MAJOR**, only "
        "acceptable with a very good difference-enhancing argument), or `unit_norm_mismatch` (self-normalized top panel "
        "paired with raw unnormalized bottom residual).\n"
        "- Populate `colormap_feedback` with a direct critique of the figure's palette/colormap, "
        "midpoint anchoring, AND saturation avoidance (flag clipped dark plateaus and recommend "
        "soft truncated `RdBu_r` [0.10, 0.90] or `sky_div` + `TwoSlopeNorm(vcenter=0.0)` with "
        "98% bounds / headroom when saturation occurs).\n"
        "- Populate `filename_audit`: inspect `source_filename` and `filename_jargon_tokens`. If "
        "the filename contains opaque batch/code tokens (like `B1`, `B1B2`, `neg`, `CPF2`, "
        "`nohost`, `varH`, `rk20`, `gt30` when cuts changed), set `has_jargon=true` and propose a "
        "clean, concise, self-explanatory physical `suggested_filename` keeping the same extension "
        "(e.g. `fig9_desy3_gp_spatial_covariance_lamge45.png`). Note: filename hygiene alone is a "
        "rename recommendation and does NOT require `REDO_FIGURE` if the rendered pixels are clean.\n"
        "- `phase1_score` MUST equal `conclusion_accuracy_score + variable_clarity_score + "
        "visual_design_score`.\n\n"
        "Decision & Gate Policy (Phase 1 Iteration Rule):\n"
        f"- `PASS` ONLY if `conclusion_match == 'MATCH'`, `phase1_score >= {pass_threshold}`, "
        "zero code/invented jargon or notation collisions, and zero critical/major visual defects.\n"
        "- `REVISE_CAPTION` if the rendered figure itself is clean, physical, and notation-compliant, "
        "and ONLY caption definitions or clarifications are missing.\n"
        "- `REDO_FIGURE` if Reviewer 1 misunderstood the physical conclusion, hit code/invented "
        "jargon or notation collisions, or encountered major layout/overlap/colorbar defects.\n"
        "- Never propose replacing code jargon with new invented jargon; always recommend either "
        "the canonical symbol from the Table of Notation or plain astrophysics English.\n"
        "Return valid JSON matching the required schema."
    )


def _format_compact_notation(notation_spec: NotationSpec) -> str:
    """Format a compact, high-signal Table of Notation excerpt to keep LLM latency low."""
    lines: list[str] = []
    if notation_spec.symbol_definitions:
        lines.append("Canonical symbols:")
        for sym, defn in list(notation_spec.symbol_definitions.items())[:55]:
            lines.append(f"- {sym}: {defn[:110]}")
    if notation_spec.banned_terms:
        lines.append("Banned / code-jargon terms: " + ", ".join(notation_spec.banned_terms[:40]))
    return "\n".join(lines) if lines else notation_spec.raw_excerpt[:4500]


def build_master_user_prompt(
    *,
    blind_review: BlindReviewPayload,
    intended_conclusion: str,
    intended_labels: dict[str, str] | None = None,
    caption: str | None = None,
    notation_spec: NotationSpec | None = None,
    deterministic_hits: list[DeterministicAuditHit] | None = None,
    color_forensics: ColorForensics | None = None,
    source_filename: str | None = None,
    filename_jargon_tokens: list[str] | None = None,
    style_profile: str = "esteves",
    color_preferences: str | None = None,
    caption_stage_blind_review: BlindReviewPayload | None = None,
    vector_label_audit: dict[str, Any] | None = None,
) -> str:
    """Assemble the unblinded Master Phase 1 evaluation prompt."""
    payload: dict[str, Any] = {
        "source_filename": source_filename or "(unknown)",
        "filename_jargon_tokens": filename_jargon_tokens or [],
        "intended_conclusion": intended_conclusion,
        "intended_labels": intended_labels or {},
        "draft_caption": caption or "(NO CAPTION PROVIDED)",
        "reviewer1_blind_output": blind_review.model_dump(),
        "deterministic_jargon_and_notation_hits": [
            h.model_dump() for h in (deterministic_hits or [])
        ],
        "deterministic_pixel_color_forensics": (
            color_forensics.model_dump() if color_forensics is not None else {}
        ),
    }
    if caption_stage_blind_review is not None:
        payload["reviewer1_stage2_with_caption"] = caption_stage_blind_review.model_dump()

    prompt = [
        "Evaluate Reviewer 1's Phase 1 output against the Master's ground truth and color preferences below.\n"
        "Be concise (<= 2 sentences per field, <= 12 label comparisons) and output JSON immediately.\n",
        "```json",
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        "```",
        "\n### Active Color & Colormap Preferences",
        get_color_preferences_text(style_profile=style_profile, custom_preferences=color_preferences),
    ]

    vec_block = format_vector_audit_for_prompt(vector_label_audit)
    if vec_block:
        prompt.extend(["\n### Vectorized Field Label, Jargon & Banned-Words Audit (160+ Papers)", vec_block])

    if notation_spec is not None:
        prompt.extend(
            [
                f"\n### Canonical Table of Notation (`{notation_spec.source_path}`)",
                "All figure symbols MUST follow this table; avoid introducing new terms unless strictly needed.",
                "```markdown",
                _format_compact_notation(notation_spec),
                "```",
            ]
        )

    prompt.append(
        "\nProduce the structured Master Phase 1 verdict JSON with explicit 0-100 score breakdown, "
        "truth-vs-blind comparisons, notation violations, caption gaps, and prioritized fixes."
    )
    return "\n".join(prompt)


# ---------------------------------------------------------------------------
# Phase 2: Reviewer 2 (Context, Notation Hygiene & Narrative Alignment)
# ---------------------------------------------------------------------------

def build_reviewer2_system_prompt(domain: str = "astrophysics", pass_threshold: int = 80) -> str:
    """Distinct system prompt for Reviewer 2 (Phase 2: Manuscript Context Loop)."""
    clean_domain = (domain or "astrophysics").strip()
    return (
        f"You are Reviewer 2: a ruthless {clean_domain} manuscript-integration and notation-hygiene "
        "referee conducting Phase 2 (The Context Loop).\n"
        "This figure has already passed Phase 1 standalone visual clarity. Your job is strictly "
        "to audit how the figure integrates into the `.tex` manuscript and Table of Notation.\n\n"
        "Phase 2 Audit Pillars & Scoring Rubric (0-100 total):\n"
        "1. `notation_hygiene_score` (0-50):\n"
        "   - Verify EVERY symbol, subscript, superscript, macro, and unit in the figure and "
        "caption matches the `.tex` manuscript and Table of Notation 1-to-1.\n"
        "   - Ruthlessly flag orphaned terms (symbols in the figure never defined/used in `.tex`), "
        "ad-hoc variables, symbol collisions, unit mismatches, code flags, or newly invented labels.\n"
        "2. `narrative_alignment_score` (0-50):\n"
        "   - Check whether the `.tex` prose explicitly introduces and walks the reader through "
        "every panel, curve, and feature of the figure.\n"
        "   - Check whether the figure actively advances the section's physical argument — **the goal of the plot is "
        "to enhance differences between things**: verify that the figure plots the **actual modeled variable from the "
        "`.tex` equation** directly (**not plotting the actual variable being studied is a MAJOR defect**, only acceptable "
        "with a very good difference-enhancing argument) and that any systematic/effect discussed in `.tex` is **visibly "
        "demonstrated by eye** via a clean Effect OFF vs. ON contrast with matched top/bottom normalization & units "
        "(never hidden behind collapsed mixed averages).\n"
        "   - Flag unreferenced figure panels/curves AND text claims not visibly supported by the plot.\n"
        "- `phase2_score` MUST equal `notation_hygiene_score + narrative_alignment_score`.\n\n"
        "Verdict rules:\n"
        f"- `PASS` ONLY if `phase2_score >= {pass_threshold}`, zero symbol collisions/ad-hoc "
        "variables, and the `.tex` prose clearly guides the reader through the figure.\n"
        "- `REVISE_TEX` if the figure itself is consistent, and only the `.tex` prose needs updates "
        "to guide the reader or remove orphaned/unsupported references.\n"
        "- `REVISE_CAPTION` if only the caption needs alignment with `.tex` notation.\n"
        "- `REDO_FIGURE` if labels/symbols rendered inside the figure conflict with the `.tex` "
        "manuscript or Table of Notation.\n"
        "Return valid JSON matching the required schema."
    )


def build_reviewer2_user_prompt(
    *,
    tex_context: ExtractedTexContext,
    caption: str | None = None,
    notation_spec: NotationSpec | None = None,
    phase1_grade: MasterGradeResult | None = None,
    cli_image_filename: str | None = None,
) -> str:
    """Build Reviewer 2's Phase 2 manuscript-integration user prompt."""
    parts: list[str] = []
    if cli_image_filename:
        parts.append(
            f"IMPORTANT SANDBOX RULE: Inspect ONLY `{cli_image_filename}` in the current working "
            "directory for the figure image. All required manuscript `.tex` excerpts and notation "
            "tables are embedded directly below — do NOT browse other files on disk, and do NOT "
            "invoke any MCP tools, subagents, or skills. Keep every JSON field concise (<= 2 "
            "sentences per item, <= 12 symbols in `symbol_hygiene`) and output the JSON immediately.\n"
        )

    effective_caption = caption or tex_context.figure_caption_in_tex or "(NO CAPTION PROVIDED)"
    parts.append(
        "**Phase 2 Task (Manuscript Integration):** Inspect the approved figure image together "
        "with its caption, the `.tex` manuscript excerpt, and the Table of Notation below.\n"
        "1. Perform a symbol-by-symbol notation hygiene check (`symbol_hygiene`) and list any "
        "`orphaned_terms` or ad-hoc variables.\n"
        "2. Evaluate narrative integration (`narrative_guidance_assessment`, "
        "`unreferenced_figure_elements`, `unsupported_text_claims`).\n"
        "3. Score `notation_hygiene_score` (0-50) + `narrative_alignment_score` (0-50) = "
        "`phase2_score` (0-100) and emit concrete `tex_edits`, `figure_label_fixes`, and "
        "`caption_fixes`.\n"
    )

    parts.append(f"### Figure Caption\n```\n{effective_caption}\n```")

    if phase1_grade is not None:
        parts.append(
            f"### Phase 1 Gate Summary\n"
            f"- Phase 1 Score: {phase1_grade.phase1_score}/100 ({phase1_grade.phase1_letter_grade})\n"
            f"- Phase 1 Passed: {phase1_grade.phase1_passed}\n"
            f"- Conclusion Check: {phase1_grade.conclusion_match} — "
            f"{phase1_grade.conclusion_comparison_notes}"
        )
        vec_block = format_vector_audit_for_prompt(phase1_grade.vector_label_audit)
        if vec_block:
            parts.append(f"### Vectorized Field Label & Jargon Audit (160+ Papers)\n{vec_block}")

    if tex_context.macros:
        parts.append(
            "### Manuscript Preamble Macros\n```latex\n"
            + "\n".join(tex_context.macros)
            + "\n```"
        )

    if tex_context.figure_block:
        parts.append(
            f"### Figure Environment in `{tex_context.tex_path}`\n```latex\n"
            f"{tex_context.figure_block}\n```"
        )

    if tex_context.referencing_paragraphs:
        joined = "\n\n---\n\n".join(tex_context.referencing_paragraphs[:6])
        parts.append(f"### Paragraphs Referencing This Figure\n```latex\n{joined}\n```")

    if tex_context.section_excerpt:
        parts.append(
            f"### Surrounding Manuscript Section (`{tex_context.tex_path}`)\n```latex\n"
            f"{tex_context.section_excerpt}\n```"
        )

    if notation_spec is not None:
        parts.append(
            f"### Canonical Table of Notation (`{notation_spec.source_path}`)\n```markdown\n"
            f"{_format_compact_notation(notation_spec)}\n```"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 2A: Pre-Figure Blind Reader Expectation (Vector-Selected Anchor Papers)
# ---------------------------------------------------------------------------


def build_figure_expectation_system_prompt(domain: str = "astrophysics") -> str:
    """System prompt for Stage 2A Blind Pre-Figure Reader (ZERO image pixels seen)."""
    clean_domain = (domain or "astrophysics").strip()
    return (
        f"You are an independent {clean_domain} referee deeply familiar with the vector-selected "
        "field anchor papers listed below. You have NOT seen the author's figure yet (strictly "
        "blind to all image pixels).\n\n"
        "Your mission is **minimal pages, maximum physical understanding**:\n"
        "- Identify the top 3-4 physical questions this reader immediately expects the figure to answer.\n"
        "- Prescribe the **strictly minimal** panel layout (fewest panels/curves needed; reject "
        "kitchen-sink 6-panel clutter) with exact axes and shared legends — remembering that **the goal of the plot "
        "is to enhance the differences between things**: plot the **exact modeled quantity from the equation** directly "
        "(do NOT invent ad-hoc normalizations unless they explicitly stress differences), and whenever a systematic or "
        "model effect is tested, prescribe a clean **Effect OFF vs. Effect ON** (or dialed parameter ladder) contrast "
        "holding everything else fixed, so the reader spots the difference in 2 seconds by eye.\n"
        "- Specify concrete literature benchmarks/bands from the anchor papers the reader expects "
        "to compare against.\n"
        "- State which secondary/redundant quantities should be cut or moved to a table or appendix.\n"
        "- Highlight definition mismatches or plotting confusion traps (e.g. twin y-axes, diluting a systematic "
        "in an unsplit mixed average, hijacking the contrast with an unrelated dataset/aperture offset, or pairing "
        "a self-normalized top panel with a raw unnormalized bottom residual strip).\n"
        "Keep every bullet ultra-crisp (<= 1-2 sentences) and return valid JSON immediately."
    )


def build_figure_expectation_user_prompt(
    *,
    topic: str,
    methodology: str,
    anchor_papers_block: str,
) -> str:
    return (
        "### Blind Pre-Figure Reader Expectation Task (ZERO Image Pixels Provided)\n"
        f"**Topic:** {topic.strip()}\n"
        f"**Methodology Brief:** {(methodology or '(not specified)').strip()}\n\n"
        f"### Anchor Literature Lens (Vector-Selected From Field Library)\n"
        f"{anchor_papers_block}\n\n"
        "Strictly BEFORE seeing any figure pixels, output the minimal expected figure design, "
        "benchmarks, cut list, and confusion traps as structured JSON immediately."
    )

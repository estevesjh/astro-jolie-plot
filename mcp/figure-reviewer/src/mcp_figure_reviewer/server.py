"""FastMCP stdio server for the gated two-phase scientific figure peer-review loop.

Phase 1 (Blind Loop — Reviewer 1 + Master Gate):
  `blind_review_figure` -> `grade_blind_review` (0-100 Phase 1 score + `phase1_passed` gate)

Phase 2 (Context Loop — Reviewer 2, strictly gated on Phase 1 passing):
  `phase2_context_review` (`.tex` manuscript + Table of Notation -> 0-100 Phase 2 score +
  combined 0-100 Final Score & letter grade)
"""

from __future__ import annotations

from typing import Any, Literal

from fastmcp import FastMCP

from .backends import detect_backends
from .blinding import inspect_figure_pixels as _inspect_pixels
from .reviewer import (
    Phase1GateBlockedError,
    load_blind_review,
    run_batch_blind_review,
    run_blind_review,
    run_deterministic_grade,
    run_end_to_end_review,
    run_master_grade,
    run_phase2_context_review,
)
from .vector_audit import run_vector_label_audit

mcp = FastMCP(
    name="figure-scientific-reviewer",
    instructions=(
        "Physically blinded, two-phase scientific figure peer-review server with 0-100 scoring.\n"
        "FAST CHAT WORKFLOW (recommended during interactive chats — 2x faster):\n"
        "  1. Call `blind_review_figure` (runs a <0.1s 0-token `fail_fast` pixel/colormap pre-gate "
        "     before invoking Reviewer 1 on a compact 1200px WebP in `speed='fast'`).\n"
        "  2. YOU (the chat agent) already hold the ground truth and caption — act as Master yourself "
        "     and call `grade_blind_deterministic` (0 LLM tokens, <0.05s) with "
        "     `master_conclusion_match='MATCH'|'PARTIAL_MISMATCH'|'MISMATCH'` to run all deterministic "
        "     checks (code jargon, Table of Notation banned terms, colorbar midpoint arithmetic, "
        "     saturation clipping, filename jargon) and stamp a `phase1_grade_id`.\n"
        "  3. ONLY after `phase1_passed=True`, call `phase2_context_review` with that `phase1_grade_id` "
        "     and `.tex` manuscript to verify strict notation hygiene & narrative alignment.\n"
        "STANDALONE / CI WORKFLOW: Use `grade_blind_review` or `review_figure_end_to_end` to run an "
        "independent external Master LLM. For multi-figure triage, call `blind_review_batch` "
        "(parallel `max_concurrency=5`)."
    ),
)


@mcp.tool
def check_backends() -> dict[str, Any]:
    """List available Python LLM backends, active default Anthropic model, and blinding level."""
    return detect_backends()


@mcp.tool
def inspect_figure_pixels(
    image_path: str,
    page: int = 1,
    dpi: int = 220,
) -> dict[str, Any]:
    """Zero-token deterministic pre-audit of a figure image or PDF page.

    Checks pixel dimensions, aspect ratio, file size, white/black saturation fractions,
    low-resolution warnings, and embedded PNG/PDF metadata keys that blinding will strip.
    """
    return _inspect_pixels(image_path, page=page, dpi=dpi).model_dump()


@mcp.tool
async def blind_review_figure(
    image_path: str,
    domain: str = "astrophysics",
    caption: str | None = None,
    extra_checks: list[str] | None = None,
    topic: str | None = None,
    methodology: str | None = None,
    pin_papers: list[str] | None = None,
    top_k_papers: int = 3,
    expectation_first: bool = False,
    round_number: int = 1,
    page: int = 1,
    dpi: int | None = None,
    backend: str = "auto",
    speed: str = "fast",
    detail: Literal["auto", "compact", "full"] = "auto",
    use_cache: bool = True,
    fail_fast: bool = True,
    fail_fast_on_filename_jargon: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    """Phase 1 (Reviewer 1): physically blinded visual-clarity review (+ optional parallel Expectation-First Reader).

    Speed & pre-gate optimizations:
    - `fail_fast=True` (default): runs a ~0.05s 0-token pixel/colormap pre-gate BEFORE calling
      any LLM; short-circuits immediately on blurry low-res thumbnails, rainbow/`jet`, or severe
      dark saturation clipping (pass `fail_fast=False` to force LLM review anyway).
    - `expectation_first=True` (or pass `topic` / `methodology`): runs vector anchor-paper
      retrieval (<5 ms, 0 tokens over your 160+ paper library) and executes Stage 2A (Blind
      Pre-Figure Reader Expectation — ZERO pixels seen, minimal-page layout & benchmarks) IN
      PARALLEL with Stage 2B (Blind Reviewer 1 — ONLY stripped pixels, ZERO expectations),
      adding ~0 s wall-clock latency!
    - `detail="auto"` (default): uses a compact ~250-token triage schema in `speed="fast"` (2-3x
      faster decoding) and full panel-by-panel schema in `speed="deep"`.
    - `use_cache=True` (default): SHA-256 content cache returns `0.0s` instant hits on repeats.

    Next step in chat: pass `review_id` to `grade_blind_deterministic` (0 tokens, <0.05s).
    """
    review = await run_blind_review(
        image_path,
        domain=domain,
        caption=caption,
        extra_checks=extra_checks,
        topic=topic,
        methodology=methodology,
        pin_papers=pin_papers,
        top_k_papers=top_k_papers,
        expectation_first=expectation_first,
        round_number=round_number,
        page=page,
        dpi=dpi,
        backend=backend,
        model=model,
        speed=speed,
        detail=detail,
        use_cache=use_cache,
        fail_fast=fail_fast,
        fail_fast_on_filename_jargon=fail_fast_on_filename_jargon,
    )
    return review.model_dump()


@mcp.tool
def grade_blind_deterministic(
    blind_review: str | dict[str, Any],
    master_conclusion_match: Literal["PENDING", "MATCH", "PARTIAL_MISMATCH", "MISMATCH"] = "PENDING",
    conclusion_notes: str | None = None,
    notation_table_path: str | None = None,
    banned_terms: list[str] | None = None,
    style_profile: str = "esteves",
    pass_threshold: int = 80,
    master_score_override: int | None = None,
    resolved_opaque_terms: list[str] | None = None,
    master_figure_fixes: list[str] | None = None,
    master_caption_fixes: list[str] | None = None,
    allow_transformed_variable_reason: str | None = None,
) -> dict[str, Any]:
    """Zero-token deterministic Phase 1 Master Gate (<0.05s) — preferred in interactive chats!

    Avoids spawning a redundant second LLM subprocess when the calling chat agent already holds
    the ground truth, caption, and `.tex` in context. Runs all hard deterministic audits:
      - Code jargon regexes & Table of Notation banned terms,
      - Arithmetic diverging colorbar midpoint verification (pure white anchored at 0.0),
      - Color forensics (rainbow/`jet`, dark saturation clipping, `tab10`, dead-pale fills),
      - Filename jargon hygiene audit (flags `B1B2`, `neg`, `CPF2`, `v3`, etc.),
      - Reviewer 1 `OPAQUE_JARGON` and `critical` visual defects.

    When you have compared Reviewer 1's `one_sentence_conclusion` against your intended physics,
    pass `master_conclusion_match="MATCH"` (or `"PARTIAL_MISMATCH"` / `"MISMATCH"`). If all
    deterministic gates also pass, `phase1_passed=True` and the returned `phase1_grade_id`
    (`p1_det_...`) immediately unlocks `phase2_context_review`.
    """
    result = run_deterministic_grade(
        blind_review,
        master_conclusion_match=master_conclusion_match,
        conclusion_notes=conclusion_notes,
        notation_table_path=notation_table_path,
        banned_terms=banned_terms,
        style_profile=style_profile,
        pass_threshold=pass_threshold,
        master_score_override=master_score_override,
        resolved_opaque_terms=resolved_opaque_terms,
        master_figure_fixes=master_figure_fixes,
        master_caption_fixes=master_caption_fixes,
        allow_transformed_variable_reason=allow_transformed_variable_reason,
    )
    return result.model_dump()


@mcp.tool
async def grade_blind_review(
    blind_review: str | dict[str, Any],
    intended_conclusion: str,
    intended_labels: dict[str, str] | None = None,
    caption: str | None = None,
    notation_table_path: str | None = None,
    banned_terms: list[str] | None = None,
    style_profile: str = "esteves",
    color_preferences: str | None = None,
    pass_threshold: int = 80,
    allow_transformed_variable_reason: str | None = None,
    backend: str = "auto",
    speed: str = "fast",
    model: str | None = None,
) -> dict[str, Any]:
    """Phase 1 Master Gate: compare Reviewer 1 against truth, score 0-100, and audit colormap preferences.

    Pass either the `review_id` from `blind_review_figure` or the blind review JSON dict.
    Runs a 0-token deterministic notation/jargon + color forensics scan plus Master evaluation:
      - Conclusion accuracy (0-40) + Variable clarity (0-35) + Visual & colormap design (0-25).
    `style_profile="esteves"` (default) checks the house style + Sanzo Wada palette
    and diverging percentile-stretch rules; pass `"generic"` or custom `color_preferences` to override.
    Returns `phase1_score`, `phase1_letter_grade`, `phase1_passed` (hard gate unlocking Phase 2),
    `phase1_grade_id`, `colormap_feedback`, `gate_block_reasons`, and concrete figure/caption fixes.
    """
    result = await run_master_grade(
        blind_review,
        intended_conclusion=intended_conclusion,
        intended_labels=intended_labels,
        caption=caption,
        notation_table_path=notation_table_path,
        banned_terms=banned_terms,
        style_profile=style_profile,
        color_preferences=color_preferences,
        pass_threshold=pass_threshold,
        allow_transformed_variable_reason=allow_transformed_variable_reason,
        backend=backend,
        model=model,
        speed=speed,
    )
    return result.model_dump()


@mcp.tool
async def phase2_context_review(
    image_path: str,
    phase1_grade_id: str,
    tex_path: str,
    caption: str | None = None,
    notation_table_path: str | None = None,
    figure_label: str | None = None,
    section_hint: str | None = None,
    domain: str = "astrophysics",
    pass_threshold: int = 80,
    force_override: bool = False,
    page: int = 1,
    dpi: int | None = None,
    backend: str = "auto",
    speed: str = "fast",
    model: str | None = None,
) -> dict[str, Any]:
    """Phase 2 (Reviewer 2): strict notation hygiene & narrative alignment against `.tex`.

    Enforces the Iteration Rule: REFUSES to run if `phase1_grade_id` did not pass Phase 1
    (unless `force_override=True` for debugging).

    Reviewer 2 inspects the approved figure + caption + extracted `.tex` manuscript context
    (macros, figure environment, referencing paragraphs, enclosing section) + Table of Notation:
      - Notation hygiene (0-50): 1-to-1 symbol/unit match, orphaned terms, ad-hoc variables.
      - Narrative alignment (0-50): whether `.tex` explicitly guides the reader through the plot
        and whether the figure advances the section's argument.
    Returns `phase2_score` (0-100), combined `final_score_summary` (0-100 + A-F letter grade +
    `READY_FOR_PAPER` / `REVISE_TEX` / `REVISE_CAPTION` / `REDO_FIGURE`), and exact edits.
    """
    try:
        result = await run_phase2_context_review(
            image_path,
            phase1_grade=phase1_grade_id,
            tex_path=tex_path,
            caption=caption,
            notation_table_path=notation_table_path,
            figure_label=figure_label,
            section_hint=section_hint,
            domain=domain,
            pass_threshold=pass_threshold,
            force_override=force_override,
            page=page,
            dpi=dpi,
            backend=backend,
            model=model,
            speed=speed,
        )
        return result.model_dump()
    except Phase1GateBlockedError as exc:
        return {
            "blocked": True,
            "phase2_status": "BLOCKED_BY_PHASE1",
            "error": str(exc),
        }


@mcp.tool
async def review_figure_end_to_end(
    image_path: str,
    intended_conclusion: str,
    intended_labels: dict[str, str] | None = None,
    caption: str | None = None,
    notation_table_path: str | None = None,
    banned_terms: list[str] | None = None,
    style_profile: str = "esteves",
    color_preferences: str | None = None,
    tex_path: str | None = None,
    figure_label: str | None = None,
    section_hint: str | None = None,
    domain: str = "astrophysics",
    topic: str | None = None,
    methodology: str | None = None,
    pin_papers: list[str] | None = None,
    top_k_papers: int = 3,
    expectation_first: bool = False,
    include_caption_in_phase1: bool = True,
    test_with_caption: bool = False,
    pass_threshold: int = 80,
    extra_checks: list[str] | None = None,
    page: int = 1,
    dpi: int | None = None,
    backend: str = "auto",
    speed: str = "fast",
    detail: Literal["auto", "compact", "full"] = "auto",
    use_cache: bool = True,
    fail_fast: bool = True,
    fail_fast_on_filename_jargon: bool = False,
    master_mode: Literal["llm", "deterministic"] = "llm",
    master_conclusion_match: Literal["PENDING", "MATCH", "PARTIAL_MISMATCH", "MISMATCH"] = "PENDING",
    blind_model: str | None = None,
    master_model: str | None = None,
    reviewer2_model: str | None = None,
    save_report_path: str | None = None,
) -> dict[str, Any]:
    """Gated two-phase scientific figure review pipeline with 0-100 Final Score & timings.

    1. Runs a ~0.05s 0-token `fail_fast` pixel/colormap pre-gate before any LLM call.
    2. Runs Phase 1 (Reviewer 1 blind extraction + Master Gate -> `phase1_score` 0-100 &
       `phase1_passed`). Set `master_mode="deterministic"` to skip the external Master LLM call.
    3. Enforces the Iteration Rule: ONLY if `phase1_passed=True` AND `tex_path` is provided,
       advances to Phase 2 (Reviewer 2 manuscript notation hygiene & narrative alignment ->
       `phase2_score` 0-100). If Phase 1 fails, Phase 2 is strictly blocked (`BLOCKED_BY_PHASE1`).
    4. Defaults to `speed="fast"` + `detail="auto"` (`compact` ~250-token triage + 1200px WebP +
       SHA-256 cache); pass `speed="deep"` or `detail="full"` for full panel-by-panel audit.
    """
    return await run_end_to_end_review(
        image_path,
        intended_conclusion=intended_conclusion,
        intended_labels=intended_labels,
        caption=caption,
        notation_table_path=notation_table_path,
        banned_terms=banned_terms,
        style_profile=style_profile,
        color_preferences=color_preferences,
        tex_path=tex_path,
        figure_label=figure_label,
        section_hint=section_hint,
        domain=domain,
        topic=topic,
        methodology=methodology,
        pin_papers=pin_papers,
        top_k_papers=top_k_papers,
        expectation_first=expectation_first,
        include_caption_in_phase1=include_caption_in_phase1,
        test_with_caption=test_with_caption,
        pass_threshold=pass_threshold,
        extra_checks=extra_checks,
        page=page,
        dpi=dpi,
        backend=backend,
        speed=speed,
        detail=detail,
        use_cache=use_cache,
        fail_fast=fail_fast,
        fail_fast_on_filename_jargon=fail_fast_on_filename_jargon,
        master_mode=master_mode,
        master_conclusion_match=master_conclusion_match,
        blind_model=blind_model,
        master_model=master_model,
        reviewer2_model=reviewer2_model,
        save_report_path=save_report_path,
    )


@mcp.tool
def audit_figure_labels_vector(
    blind_review: str | dict[str, Any] | None = None,
    labels: list[str] | None = None,
    caption: str | None = None,
    notation_table_path: str | None = None,
    project_root: str | None = None,
) -> dict[str, Any]:
    """Zero-token vectorized audit of figure labels, legends & titles via `astro-math-review figure-labels`.

    Checks every visible axis label, panel title, legend, and annotation against:
      1. Persistent Banned Words Registry (`~/.config/astro-math-reviewer/banned_words.yaml`:
         `pedestal`, `shelf`, `closure`, `unselected`, `zero-lag`, plus any user-added bans),
      2. Empirical 160+ paper Field Concept & Jargon Index (stats jargon like `survival function`,
         `Palm calculus`, `Fano factor`, empirical `df_papers / N`, unfamiliar acronyms),
      3. Dual-view 160+ paper Field-Notation Index (symbol consensus & near-collisions).

    Pass either a `review_id` (from `blind_review_figure`) or raw `labels=[...]` strings.
    """
    loaded = load_blind_review(blind_review) if blind_review is not None else None
    audit, hits = run_vector_label_audit(
        loaded,
        raw_labels=labels,
        caption=caption,
        notation_table_path=notation_table_path,
        project_root=project_root,
    )
    return {
        **audit,
        "blocking_hits": [h.model_dump() for h in hits],
        "clean": len(hits) == 0,
    }


@mcp.tool
async def blind_review_batch(
    image_paths: list[str],
    domain: str = "astrophysics",
    extra_checks: list[str] | None = None,
    backend: str = "auto",
    speed: str = "fast",
    detail: Literal["auto", "compact", "full"] = "auto",
    use_cache: bool = True,
    fail_fast: bool = True,
    fail_fast_on_filename_jargon: bool = False,
    model: str | None = None,
    max_concurrency: int = 5,
) -> dict[str, Any]:
    """Run independent Phase 1 Reviewer 1 blind reviews across multiple figures in parallel (default concurrency 5)."""
    return await run_batch_blind_review(
        image_paths,
        domain=domain,
        extra_checks=extra_checks,
        backend=backend,
        model=model,
        speed=speed,
        detail=detail,
        use_cache=use_cache,
        fail_fast=fail_fast,
        fail_fast_on_filename_jargon=fail_fast_on_filename_jargon,
        max_concurrency=max_concurrency,
    )


def main() -> None:
    """Entry point for the stdio MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()

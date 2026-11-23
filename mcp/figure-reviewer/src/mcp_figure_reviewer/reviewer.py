"""High-level orchestration for the gated two-phase scientific figure review workflow.

Phase 1 — The Blind Loop (Visual Clarity — Reviewer 1 + Master Gate):
  - Reviewer 1 sees ONLY metadata-stripped figure pixels (+ optional caption).
  - Master grades Reviewer 1 against intended physical truth -> Phase 1 Score (0-100)
    and hard boolean gate `phase1_passed`.

Phase 2 — The Context Loop (Manuscript Integration — Reviewer 2):
  - Strictly gated on `phase1_passed == True` (enforces the Iteration Rule).
  - Reviewer 2 sees the approved figure + caption + extracted `.tex` manuscript
    context + Table of Notation -> Phase 2 Score (0-100) and Combined Final Score (0-100).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from .backends import (
    backend_uses_cli_sandbox,
    call_structured_llm,
    default_model_for,
    resolve_backend,
)
from .blinding import (
    evaluate_fast_fail_blockers,
    image_settings_for_speed,
    prepare_blind_image,
)
from .colormap import analyze_figure_colors, audit_colorbar_midpoints_deterministically
from .notation import (
    NotationSpec,
    audit_filename_jargon,
    parse_notation_table,
    scan_blind_review_deterministically,
)
from .prompts import (
    build_blind_system_prompt,
    build_blind_user_prompt,
    build_compact_blind_user_prompt,
    build_figure_expectation_system_prompt,
    build_figure_expectation_user_prompt,
    build_master_system_prompt,
    build_master_user_prompt,
    build_reviewer2_system_prompt,
    build_reviewer2_user_prompt,
)
from .schemas import (
    BlindFigureReview,
    BlindingCertificate,
    BlindReviewPayload,
    ColormapAudit,
    ColormapFeedback,
    CompactBlindReviewPayload,
    FigureReaderExpectation,
    FigureReaderExpectationPayload,
    FilenameAudit,
    FinalScoreSummary,
    MasterGradePayload,
    MasterGradeResult,
    Phase1ScoreBreakdown,
    Phase2ReviewPayload,
    Phase2ReviewResult,
    VisualIssue,
    score_to_letter_grade,
)
from .tex_context import extract_tex_context
from .vector_audit import (
    format_anchor_papers_for_figure_expectation,
    run_vector_label_audit,
    run_vector_paper_selection,
)


def _merge_deterministic_hits(
    base_hits: list[Any],
    vector_hits: list[Any],
) -> list[Any]:
    """Merge vectorized label hits cleanly without duplicating an identical term+location."""
    merged = list(base_hits)
    seen = {(h.term.strip().lower(), h.location.strip().lower()) for h in merged}
    for vh in vector_hits:
        key = (vh.term.strip().lower(), vh.location.strip().lower())
        if key not in seen:
            seen.add(key)
            merged.append(vh)
    return merged


class Phase1GateBlockedError(PermissionError):
    """Raised when Phase 2 is invoked before Phase 1 has passed the blind clarity gate."""


def _review_store_dir() -> Path:
    base = Path(
        os.environ.get(
            "FIGURE_REVIEW_STORE_DIR",
            str(Path(tempfile.gettempdir()) / "mcp_figure_reviewer_sessions"),
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_blind_review(review: BlindFigureReview) -> Path:
    path = _review_store_dir() / f"{review.review_id}.json"
    path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
    return path


def save_phase1_grade(grade: MasterGradeResult) -> Path:
    path = _review_store_dir() / f"{grade.phase1_grade_id}.json"
    path.write_text(grade.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_blind_review(review_or_id: str | dict[str, Any] | BlindFigureReview | BlindReviewPayload) -> BlindFigureReview:
    """Load a BlindFigureReview from an object, dict, JSON string, or persisted review_id."""
    if isinstance(review_or_id, BlindFigureReview):
        return review_or_id
    if isinstance(review_or_id, BlindReviewPayload):
        return load_blind_review(review_or_id.model_dump())
    if isinstance(review_or_id, dict):
        if "review_id" in review_or_id and "blinding_certificate" in review_or_id:
            return BlindFigureReview.model_validate(review_or_id)
        payload = BlindReviewPayload.model_validate(review_or_id)
        return BlindFigureReview(
            **payload.model_dump(),
            review_id="inline",
            blinding_certificate=BlindingCertificate(
                backend="external",
                model="external",
                blinding_level="airtight_stateless_api",
                image_sha256="unknown",
                image_width_px=0,
                image_height_px=0,
                source_format="UNKNOWN",
            ),
        )

    text = review_or_id.strip()
    if text.startswith("{"):
        return load_blind_review(json.loads(text))

    candidate = Path(text)
    if candidate.is_file():
        return BlindFigureReview.model_validate_json(candidate.read_text(encoding="utf-8"))
    stored = _review_store_dir() / f"{text}.json"
    if stored.is_file():
        return BlindFigureReview.model_validate_json(stored.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Could not find blind review for id/path: {text}")


def load_phase1_grade(grade_or_id: str | dict[str, Any] | MasterGradeResult) -> MasterGradeResult:
    """Load a persisted MasterGradeResult by phase1_grade_id, path, JSON string, or dict."""
    if isinstance(grade_or_id, MasterGradeResult):
        return grade_or_id
    if isinstance(grade_or_id, dict):
        return MasterGradeResult.model_validate(grade_or_id)
    text = grade_or_id.strip()
    if text.startswith("{"):
        return MasterGradeResult.model_validate_json(text)
    candidate = Path(text)
    if candidate.is_file():
        return MasterGradeResult.model_validate_json(candidate.read_text(encoding="utf-8"))
    stored = _review_store_dir() / f"{text}.json"
    if stored.is_file():
        return MasterGradeResult.model_validate_json(stored.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Could not find Phase 1 grade for id/path: {text}")


def compute_final_score_summary(
    phase1_grade: MasterGradeResult,
    phase2_result: Phase2ReviewPayload | None = None,
    *,
    phase2_score: int | None = None,
    phase2_passed: bool | None = None,
) -> FinalScoreSummary:
    """Compute transparent 0-100 Final Score, letter grade, and publication verdict."""
    p1_score = phase1_grade.phase1_score
    p1_pass = phase1_grade.phase1_passed

    if phase2_result is None or phase2_score is None or phase2_passed is None:
        capped = min(p1_score, 69) if not p1_pass else p1_score
        if p1_pass:
            verdict = "READY_FOR_PAPER"
        elif phase1_grade.verdict == "REVISE_CAPTION":
            verdict = "REVISE_CAPTION"
        else:
            verdict = "REDO_FIGURE"
        return FinalScoreSummary(
            phase1_score=p1_score,
            phase1_passed=p1_pass,
            phase2_score=None,
            phase2_passed=None,
            final_score=capped,
            letter_grade=score_to_letter_grade(capped),
            final_verdict=verdict,  # type: ignore[arg-type]
            scoring_formula="Phase 1 only (0-100; capped <=69 if Phase 1 gate failed)",
        )

    combined = int(round(0.50 * p1_score + 0.50 * phase2_score))
    if not p1_pass or not phase2_passed:
        combined = min(combined, 74)

    if p1_pass and phase2_passed:
        final_verdict = "READY_FOR_PAPER"
    elif not p1_pass:
        final_verdict = (
            "REVISE_CAPTION" if phase1_grade.verdict == "REVISE_CAPTION" else "REDO_FIGURE"
        )
    else:
        final_verdict = (
            "READY_FOR_PAPER"
            if phase2_result.phase2_verdict == "PASS"
            else phase2_result.phase2_verdict
        )

    return FinalScoreSummary(
        phase1_score=p1_score,
        phase1_passed=p1_pass,
        phase2_score=phase2_score,
        phase2_passed=phase2_passed,
        final_score=combined,
        letter_grade=score_to_letter_grade(combined),
        final_verdict=final_verdict,  # type: ignore[arg-type]
        scoring_formula="0.50 * Phase1_Visual_Clarity + 0.50 * Phase2_Manuscript_Integration",
    )


# ---------------------------------------------------------------------------
# Phase 1: Reviewer 1 (Blind Visual Clarity) & Master Gate
# ---------------------------------------------------------------------------

def _suggest_clean_filename(filename: str, jargon_tokens: list[str]) -> str:
    """Deterministically strip opaque batch/code tokens from a filename as a starting suggestion."""
    if not jargon_tokens:
        return filename
    p = Path(filename)
    bad = {t.lower() for t in jargon_tokens}
    parts = [tok for tok in re.split(r"[_\-\s]+", p.stem) if tok and tok.lower() not in bad]
    clean_stem = "_".join(parts) if parts else "figure_clean"
    return f"{clean_stem}{p.suffix}"


def _fast_fail_issue_category(reason: str) -> Literal["colormap_choice", "colorbar_stretch", "tiny_font", "other"]:
    low = reason.lower()
    if "rainbow" in low or "jet" in low:
        return "colormap_choice"
    if "saturation" in low or "clipping" in low:
        return "colorbar_stretch"
    if "resolution" in low:
        return "tiny_font"
    return "other"


def _resolve_detail_mode(detail: str, speed: str) -> Literal["compact", "full"]:
    norm = (detail or "auto").strip().lower()
    if norm == "compact":
        return "compact"
    if norm == "full":
        return "full"
    return "compact" if (speed or "fast").strip().lower() == "fast" else "full"


def _blind_cache_path(
    *,
    image_sha256: str,
    caption: str | None,
    domain: str,
    extra_checks: list[str] | None,
    detail_mode: str,
    backend: str,
    model: str | None,
    speed: str,
) -> Path:
    try:
        resolved_backend = resolve_backend(backend)
    except Exception:
        resolved_backend = backend
    chosen_model = model or default_model_for(resolved_backend, speed=speed)
    key_payload = json.dumps(
        {
            "v": 3,
            "image_sha256": image_sha256,
            "caption": (caption or "").strip(),
            "domain": (domain or "astrophysics").strip().lower(),
            "extra_checks": [c.strip() for c in (extra_checks or []) if c.strip()],
            "detail_mode": detail_mode,
            "backend": resolved_backend,
            "model": chosen_model,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:24]
    cache_dir = _review_store_dir() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"blind_{digest}.json"


def _figure_expectation_cache_path(
    *,
    topic: str,
    methodology: str,
    anchor_ids: list[str],
    domain: str,
    backend: str,
    model: str | None,
    speed: str,
) -> Path:
    try:
        resolved_backend = resolve_backend(backend)
    except Exception:
        resolved_backend = backend
    chosen_model = model or default_model_for(resolved_backend, speed=speed)
    key_payload = json.dumps(
        {
            "v": 3,
            "kind": "fig_expect",
            "topic": topic.strip(),
            "methodology": methodology.strip(),
            "anchor_ids": anchor_ids,
            "domain": (domain or "astrophysics").strip().lower(),
            "backend": resolved_backend,
            "model": chosen_model,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:24]
    cache_dir = _review_store_dir() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"figexp_{digest}.json"


async def run_figure_reader_expectation(
    topic: str,
    *,
    methodology: str = "",
    pin_papers: list[str] | None = None,
    project_root: str | Path | None = None,
    notation_table_path: str | Path | None = None,
    top_k_papers: int = 3,
    domain: str = "astrophysics",
    backend: str = "auto",
    model: str | None = None,
    speed: str = "fast",
    use_cache: bool = True,
    timeout_seconds: int = 75,
) -> FigureReaderExpectation:
    """Stage 2A Blind Pre-Figure Reader Expectation (ZERO image pixels; vector-selected anchor papers)."""
    t0 = time.monotonic()
    sel = run_vector_paper_selection(
        topic,
        methodology,
        pin_papers=pin_papers,
        project_root=project_root,
        notation_table_path=notation_table_path,
        top_k=top_k_papers,
    )
    anchor_ids = [p["paper_id"] for p in (sel.get("anchor_papers") or [])]
    cache_file = _figure_expectation_cache_path(
        topic=topic,
        methodology=methodology,
        anchor_ids=anchor_ids,
        domain=domain,
        backend=backend,
        model=model,
        speed=speed,
    )
    if use_cache and cache_file.is_file():
        try:
            cached = FigureReaderExpectation.model_validate_json(cache_file.read_text(encoding="utf-8"))
            return cached.model_copy(
                update={
                    "expectation_id": f"fexp_c_{uuid.uuid4().hex[:10]}",
                    "cache_hit": True,
                    "duration_seconds": round(time.monotonic() - t0, 4),
                }
            )
        except Exception:
            pass

    anchor_block = format_anchor_papers_for_figure_expectation(sel)
    call = await call_structured_llm(
        system_prompt=build_figure_expectation_system_prompt(domain=domain),
        user_prompt=build_figure_expectation_user_prompt(
            topic=topic,
            methodology=methodology,
            anchor_papers_block=anchor_block,
        ),
        schema_model=FigureReaderExpectationPayload,
        prepared_image=None,
        backend=backend,
        model=model,
        speed=speed,
        timeout_seconds=timeout_seconds,
    )
    assert isinstance(call.parsed, FigureReaderExpectationPayload)

    res = FigureReaderExpectation(
        **call.parsed.model_dump(),
        expectation_id=f"fexp_{uuid.uuid4().hex[:10]}",
        topic=topic,
        methodology=methodology,
        anchor_papers=list(sel.get("anchor_papers") or []),
        papers_indexed=int(sel.get("papers_indexed") or 0),
        paper_selection_ms=float(sel.get("query_ms") or 0.0),
        cache_hit=False,
        backend=call.backend,
        model=call.model,
        duration_seconds=round(time.monotonic() - t0, 2),
    )
    if use_cache:
        cache_file.write_text(res.model_dump_json(indent=2), encoding="utf-8")
    return res


async def run_blind_review(
    image_path: str | Path,
    *,
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
    model: str | None = None,
    speed: str = "fast",
    detail: Literal["auto", "compact", "full"] = "auto",
    use_cache: bool = True,
    fail_fast: bool = True,
    fail_fast_on_filename_jargon: bool = False,
    timeout_seconds: int = 75,
) -> BlindFigureReview:
    """Phase 1 / Reviewer 1: strictly physically blinded review of figure (+ optional caption).

    Speed optimizations:
    - `fail_fast=True` (default): runs a ~0.05s 0-token pixel/color pre-gate before any LLM call.
    - `detail="auto"` (default): uses compact ~250-token triage schema in `speed="fast"` (2-3x
      faster decoding) and full panel-by-panel schema in `speed="deep"`.
    - `use_cache=True` (default): content-hash cache keyed by stripped image SHA-256 + caption
      returns `0.0s` instant hits on repeats (e.g. re-grading or tweaking truth).
    """
    t0 = time.monotonic()
    eff_detail = _resolve_detail_mode(detail, speed)
    eff_dpi, eff_max_dim, eff_fmt = image_settings_for_speed(speed, dpi=dpi)
    prepared = prepare_blind_image(
        image_path,
        page=page,
        dpi=eff_dpi,
        max_dimension=eff_max_dim,
        output_format=eff_fmt,
    )
    forensics = analyze_figure_colors(prepared.png_bytes)
    fn_jargon = audit_filename_jargon(image_path)

    if fail_fast:
        blockers = evaluate_fast_fail_blockers(
            width_px=prepared.orig_width_px or prepared.width_px,
            height_px=prepared.orig_height_px or prepared.height_px,
            color_forensics=forensics,
            filename_jargon_tokens=fn_jargon,
            fail_on_filename_jargon=fail_fast_on_filename_jargon,
        )
        if blockers:
            elapsed = round(time.monotonic() - t0, 3)
            cert = BlindingCertificate(
                backend="deterministic_pregate",
                model="0-token-pixel-audit",
                blinding_level="deterministic_pregate",
                original_filename_hidden=True,
                metadata_stripped=True,
                stripped_metadata_keys=prepared.stripped_metadata_keys,
                caption_provided=bool(caption and caption.strip()),
                project_context_provided=False,
                image_sha256=prepared.sha256,
                image_width_px=prepared.width_px,
                image_height_px=prepared.height_px,
                source_format=prepared.source_format,
                source_page=prepared.source_page,
                payload_format=prepared.payload_format,
                payload_bytes=len(prepared.png_bytes),
                duration_seconds=elapsed,
            )
            review = BlindFigureReview(
                one_sentence_conclusion=(
                    "Fast-failed before LLM review due to hard 0-token pixel/colormap blockers."
                ),
                panel_encodings=[],
                unclear_terms=[],
                colormap_audit=ColormapAudit(
                    has_colorbar=forensics.likely_heatmap_or_colorbar,
                    colorbars=[],
                    observed_palette_description=(
                        f"Dominant chromatic hexes: {', '.join(forensics.dominant_chromatic_hexes) or 'none'}."
                    ),
                    colormap_role_match=(
                        "MISMATCHED"
                        if (forensics.rainbow_jet_suspected or forensics.saturation_clipping_suspected)
                        else "NOT_APPLICABLE"
                    ),
                    stretch_and_midpoint_notes=(
                        f"Saturated dark-extreme pixel fraction: {forensics.saturated_dark_extreme_fraction:.1%}."
                    ),
                    contrast_and_accessibility="; ".join(forensics.color_notes) or "Not evaluated (fast-failed).",
                    colormap_issues=[
                        b for b in blockers if ("colormap" in b.lower() or "saturation" in b.lower())
                    ],
                ),
                visual_and_layout_issues=[
                    VisualIssue(
                        severity="critical",
                        category=_fast_fail_issue_category(b),
                        location="Whole figure",
                        description=b,
                    )
                    for b in blockers
                ],
                confusing_elements=list(blockers),
                standalone_readability_score=1,
                readability_summary=(
                    f"Short-circuited in {elapsed:.2f}s (0 LLM tokens): {'; '.join(blockers)}"
                ),
                review_id=f"blind_ff_{uuid.uuid4().hex[:10]}",
                round_number=round_number,
                source_filename=Path(image_path).name,
                filename_jargon_tokens=fn_jargon,
                fast_failed=True,
                fast_fail_reasons=blockers,
                detail_mode=eff_detail,
                cache_hit=False,
                color_forensics=forensics,
                blinding_certificate=cert,
            )
            save_blind_review(review)
            return review

    do_expect = bool(expectation_first or (topic and topic.strip()) or (methodology and methodology.strip()))
    eff_topic = (topic or "").strip() or (caption or "").strip()[:220] or f"{domain} scientific figure"

    cache_file = _blind_cache_path(
        image_sha256=prepared.sha256,
        caption=caption,
        domain=domain,
        extra_checks=extra_checks,
        detail_mode=eff_detail,
        backend=backend,
        model=model,
        speed=speed,
    )

    async def _run_vision_stage() -> tuple[BlindReviewPayload, BlindingCertificate, bool]:
        if use_cache and cache_file.is_file():
            try:
                cached = BlindFigureReview.model_validate_json(cache_file.read_text(encoding="utf-8"))
                updated_cert = cached.blinding_certificate.model_copy(
                    update={"duration_seconds": round(time.monotonic() - t0, 4)}
                )
                return BlindReviewPayload.model_validate(cached.model_dump()), updated_cert, True
            except Exception:
                pass

        uses_sandbox = backend_uses_cli_sandbox(backend)
        system_prompt = build_blind_system_prompt(domain=domain)
        if eff_detail == "compact":
            user_prompt = build_compact_blind_user_prompt(
                caption=caption,
                extra_checks=extra_checks,
                cli_image_filename=prepared.blind_filename if uses_sandbox else None,
            )
            schema_cls: type[BlindReviewPayload] | type[CompactBlindReviewPayload] = (
                CompactBlindReviewPayload
            )
        else:
            user_prompt = build_blind_user_prompt(
                caption=caption,
                extra_checks=extra_checks,
                cli_image_filename=prepared.blind_filename if uses_sandbox else None,
            )
            schema_cls = BlindReviewPayload

        call = await call_structured_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_model=schema_cls,
            prepared_image=prepared,
            backend=backend,
            model=model,
            speed=speed,
            timeout_seconds=timeout_seconds,
        )
        if isinstance(call.parsed, CompactBlindReviewPayload):
            full_payload = call.parsed.to_full_payload()
        else:
            assert isinstance(call.parsed, BlindReviewPayload)
            full_payload = call.parsed

        cert_out = BlindingCertificate(
            backend=call.backend,
            model=call.model,
            blinding_level=call.blinding_level,
            original_filename_hidden=True,
            metadata_stripped=True,
            stripped_metadata_keys=prepared.stripped_metadata_keys,
            caption_provided=bool(caption and caption.strip()),
            project_context_provided=False,
            image_sha256=prepared.sha256,
            image_width_px=prepared.width_px,
            image_height_px=prepared.height_px,
            source_format=prepared.source_format,
            source_page=prepared.source_page,
            payload_format=prepared.payload_format,
            payload_bytes=len(prepared.png_bytes),
            duration_seconds=call.duration_seconds,
        )
        return full_payload, cert_out, False

    if do_expect:
        exp_res, (full_payload, cert, was_cached) = await asyncio.gather(
            run_figure_reader_expectation(
                eff_topic,
                methodology=(methodology or "").strip(),
                pin_papers=pin_papers,
                top_k_papers=top_k_papers,
                domain=domain,
                backend=backend,
                model=model,
                speed=speed,
                use_cache=use_cache,
                timeout_seconds=timeout_seconds,
            ),
            _run_vision_stage(),
        )
        exp_dict = exp_res.model_dump()
    else:
        full_payload, cert, was_cached = await _run_vision_stage()
        exp_dict = {}

    review = BlindFigureReview(
        **full_payload.model_dump(),
        review_id=f"blind_{'c_' if was_cached else ''}{uuid.uuid4().hex[:10]}",
        round_number=round_number,
        source_filename=Path(image_path).name,
        filename_jargon_tokens=fn_jargon,
        fast_failed=False,
        fast_fail_reasons=[],
        detail_mode=eff_detail,
        cache_hit=was_cached,
        color_forensics=forensics,
        reader_expectations=exp_dict,
        blinding_certificate=cert,
    )
    save_blind_review(review)
    if use_cache and not was_cached:
        cache_file.write_text(review.model_dump_json(indent=2), encoding="utf-8")
    return review


def _collect_contrast_blockers(
    loaded: BlindFigureReview,
    *,
    allow_transformed_variable_reason: str | None = None,
) -> tuple[list[VisualIssue], str | None]:
    """Collect major/critical scientific-contrast, modeled-variable, and unit-match blockers.

    Not plotting the actual variable being studied (`wrong_plotted_variable` or
    `contrast_audit.plots_modeled_variable_directly == False`) is a MAJOR hard-blocking defect
    by default — only waived if the caller supplies a substantive justification
    (`allow_transformed_variable_reason`, >= 20 chars) explaining why the transformation
    explicitly enhances differences.
    """
    waive_reason = (allow_transformed_variable_reason or "").strip()
    has_strong_waiver = len(waive_reason) >= 20

    blockers: list[VisualIssue] = [
        v
        for v in loaded.visual_and_layout_issues
        if v.severity in ("critical", "major")
        and v.category in ("contrast_visibility", "wrong_plotted_variable", "unit_norm_mismatch")
    ]
    seen_cats = {v.category for v in blockers}
    ca = loaded.contrast_audit
    note_tail = f" ({ca.contrast_audit_note.strip()})" if ca.contrast_audit_note.strip() else ""

    if not ca.curves_visibly_separated_by_eye and "contrast_visibility" not in seen_cats:
        blockers.append(
            VisualIssue(
                severity="major",
                category="contrast_visibility",
                location="Primary comparison panel",
                description=(
                    "Primary compared curves are not visibly separated by eye in 2 seconds "
                    f"(curves collapse invisibly or contrast is hijacked){note_tail}."
                ),
            )
        )
    if not ca.plots_modeled_variable_directly and "wrong_plotted_variable" not in seen_cats:
        blockers.append(
            VisualIssue(
                severity="major",
                category="wrong_plotted_variable",
                location="Main y-axis",
                description=(
                    "Does not plot the actual modeled variable being studied directly "
                    f"(ad-hoc rescaling/normalization without a strong difference-enhancing argument){note_tail}."
                ),
            )
        )
    if ca.top_bottom_units_matched == "MISMATCHED" and "unit_norm_mismatch" not in seen_cats:
        blockers.append(
            VisualIssue(
                severity="major",
                category="unit_norm_mismatch",
                location="Stacked main vs. residual/ratio panels",
                description=(
                    "Top panel and bottom residual/ratio strip use mismatched normalization or units"
                    f"{note_tail}."
                ),
            )
        )

    waived_note: str | None = None
    if has_strong_waiver and any(v.category == "wrong_plotted_variable" for v in blockers):
        blockers = [v for v in blockers if v.category != "wrong_plotted_variable"]
        waived_note = f"Transformed variable accepted with explicit justification: {waive_reason}"

    return blockers, waived_note


def run_deterministic_grade(
    blind_review: str | dict[str, Any] | BlindFigureReview,
    *,
    master_conclusion_match: Literal["PENDING", "MATCH", "PARTIAL_MISMATCH", "MISMATCH"] = "PENDING",
    conclusion_notes: str | None = None,
    notation_table_path: str | Path | None = None,
    banned_terms: list[str] | None = None,
    style_profile: str = "esteves",
    pass_threshold: int = 80,
    master_score_override: int | None = None,
    resolved_opaque_terms: list[str] | None = None,
    master_figure_fixes: list[str] | None = None,
    master_caption_fixes: list[str] | None = None,
    allow_transformed_variable_reason: str | None = None,
) -> MasterGradeResult:
    """Zero-token deterministic Phase 1 Master Gate (<0.05s) for interactive chat workflows.

    In interactive chats, the calling agent already IS the Master (holding the ground truth,
    caption, and `.tex` in context). This function runs every deterministic check — code jargon,
    Table of Notation banned terms, colorbar midpoint arithmetic, saturation clipping, rainbow/jet,
    filename jargon, and Reviewer 1 critical issues — and persists a `MasterGradeResult`
    (`p1_det_...`) that cleanly enforces or unlocks Phase 2 without spawning a second LLM call.
    """
    t0 = time.monotonic()
    loaded = load_blind_review(blind_review)
    notation_spec: NotationSpec | None = (
        parse_notation_table(notation_table_path) if notation_table_path else None
    )

    deterministic_hits = scan_blind_review_deterministically(
        loaded,
        notation_spec=notation_spec,
        extra_banned_terms=banned_terms,
    )
    deterministic_hits.extend(audit_colorbar_midpoints_deterministically(loaded))
    vector_audit, vector_hits = run_vector_label_audit(
        loaded,
        notation_table_path=notation_table_path,
    )
    deterministic_hits = _merge_deterministic_hits(deterministic_hits, vector_hits)

    current_fn = loaded.source_filename or "figure.png"
    fn_jargon = (
        audit_filename_jargon(current_fn, notation_spec=notation_spec)
        if loaded.source_filename
        else list(loaded.filename_jargon_tokens)
    )
    suggested_fn = _suggest_clean_filename(current_fn, fn_jargon)

    forensics = loaded.color_forensics
    is_esteves = (style_profile or "esteves").strip().lower() in {
        "esteves",
        "house",
        "decade",
        "anbajagane",
        "wada",
    }

    resolved_set = {t.strip().strip("$").lower() for t in (resolved_opaque_terms or []) if t.strip()}
    raw_opaque = [t for t in loaded.unclear_terms if t.clarity_tag == "OPAQUE_JARGON"]
    opaque_terms = [t for t in raw_opaque if t.term.strip().strip("$").lower() not in resolved_set]
    downgraded_to_caption = [
        t for t in raw_opaque if t.term.strip().strip("$").lower() in resolved_set
    ]
    needs_caption_terms = [
        t for t in loaded.unclear_terms if t.clarity_tag == "NEEDS_CAPTION"
    ] + downgraded_to_caption
    critical_visual = [v for v in loaded.visual_and_layout_issues if v.severity == "critical"]
    major_visual = [v for v in loaded.visual_and_layout_issues if v.severity == "major"]
    minor_visual = [v for v in loaded.visual_and_layout_issues if v.severity == "minor"]
    contrast_or_unit_blockers, waived_var_note = _collect_contrast_blockers(
        loaded,
        allow_transformed_variable_reason=allow_transformed_variable_reason,
    )
    blocker_ids = {(v.category, v.location, v.description) for v in contrast_or_unit_blockers}
    major_non_contrast = [
        v for v in major_visual if (v.category, v.location, v.description) not in blocker_ids
    ]
    midpoint_hits = [h for h in deterministic_hits if h.source == "colorbar_midpoint_misaligned"]
    jargon_hits = [h for h in deterministic_hits if h.source != "colorbar_midpoint_misaligned"]

    # Deterministic 0-100 rubric scoring
    if loaded.fast_failed:
        c_score, v_score, d_score = 0, 15, 5
    else:
        if master_conclusion_match == "MATCH":
            if loaded.standalone_readability_score >= 5:
                c_score = 40
            elif loaded.standalone_readability_score == 4:
                c_score = 38
            else:
                c_score = 35
        elif master_conclusion_match == "PARTIAL_MISMATCH":
            c_score = 22
        elif master_conclusion_match == "MISMATCH":
            c_score = 8
        else:  # PENDING — chat Master has not yet attested MATCH/MISMATCH
            c_score = 0

        if any(v.category == "contrast_visibility" for v in contrast_or_unit_blockers):
            c_score = min(c_score, 32)

        v_deduct = (
            10 * len(opaque_terms)
            + 8 * len(jargon_hits)
            + (2 if loaded.blinding_certificate.caption_provided else 1) * len(needs_caption_terms)
        )
        v_score = max(0, min(35, 35 - v_deduct))

        d_deduct = (
            10 * len(critical_visual)
            + 12 * len([v for v in contrast_or_unit_blockers if v.severity == "major"])
            + 5 * len(major_non_contrast)
            + 2 * len(minor_visual)
            + 10 * len(midpoint_hits)
            + (10 if forensics.rainbow_jet_suspected else 0)
            + (6 if forensics.saturation_clipping_suspected else 0)
            + (4 if (is_esteves and forensics.tab10_suspected and not forensics.matched_house_palettes) else 0)
            + (3 if (is_esteves and forensics.dead_pale_pixel_fraction > 0.06 and not forensics.likely_heatmap_or_colorbar) else 0)
            + (6 if loaded.colormap_audit.colormap_role_match == "MISMATCHED" else 0)
            + (3 if loaded.colormap_audit.colormap_role_match == "QUESTIONABLE" else 0)
        )
        d_score = max(0, min(25, 25 - d_deduct))

    total_score = c_score + v_score + d_score
    if contrast_or_unit_blockers and not loaded.fast_failed and total_score > 74:
        # Cap total_score <= 74 so a figure failing the 2-second Eye-Test or top/bottom unit match can NEVER get Grade A/PASS
        excess = total_score - 74
        cut_d = min(d_score, excess)
        d_score -= cut_d
        excess -= cut_d
        cut_c = min(c_score, excess)
        c_score -= cut_c
        excess -= cut_c
        v_score = max(0, v_score - excess)
        total_score = c_score + v_score + d_score
    if master_score_override is not None and not loaded.fast_failed:
        total_score = max(0, min(100, int(master_score_override)))
        # Keep breakdown summing cleanly to override
        delta = total_score - (c_score + v_score + d_score)
        c_score = max(0, min(40, c_score + delta))
        rem = total_score - (c_score + v_score + d_score)
        v_score = max(0, min(35, v_score + rem))
        rem2 = total_score - (c_score + v_score + d_score)
        d_score = max(0, min(25, d_score + rem2))
        total_score = c_score + v_score + d_score

    # Collect deterministic fixes & violations
    notation_violations: list[str] = [f"{h.term} ({h.location}): {h.detail}" for h in jargon_hits]
    for ot in opaque_terms:
        notation_violations.append(f"OPAQUE_JARGON `{ot.term}` ({ot.location}): {ot.reason}")

    color_fixes: list[str] = list(loaded.colormap_audit.colormap_issues)
    for mh in midpoint_hits:
        color_fixes.append(f"{mh.location}: {mh.detail}")
    if forensics.rainbow_jet_suspected:
        color_fixes.append(
            "Replace rainbow/jet colormap with soft truncated RdBu_r [0.10, 0.90], decade_sky, or rocket_r."
        )
    if forensics.saturation_clipping_suspected:
        color_fixes.append(
            f"Dark saturated extremes cover {forensics.saturated_dark_extreme_fraction:.1%} of pixels: "
            "slice RdBu_r to [0.10, 0.90] and widen bounds (98% + headroom)."
        )
    if is_esteves and forensics.tab10_suspected and not forensics.matched_house_palettes:
        color_fixes.append(
            "Replace default matplotlib tab10 hues with curated decade2/3/4/5, classic3, or Sanzo Wada palette."
        )
    if is_esteves and forensics.dead_pale_pixel_fraction > 0.06 and not forensics.likely_heatmap_or_colorbar:
        color_fixes.append(
            "Avoid chalky low-chroma pale fills (L>0.72, S<0.50); use saturated #219ebc stepfilled or skyblue/red."
        )

    figure_code_fixes: list[str] = list(master_figure_fixes or [])
    if loaded.fast_failed:
        figure_code_fixes.extend(loaded.fast_fail_reasons)
    figure_code_fixes.extend(notation_violations)
    figure_code_fixes.extend(color_fixes)
    for vi in critical_visual + major_visual + contrast_or_unit_blockers:
        figure_code_fixes.append(f"[{vi.severity.upper()} / {vi.category}] ({vi.location}): {vi.description}")
    # Deduplicate preserving order
    figure_code_fixes = list(dict.fromkeys(figure_code_fixes))

    caption_gaps = [f"Define `{t.term}` ({t.location}): {t.reason}" for t in needs_caption_terms]
    caption_fixes = list(dict.fromkeys(list(master_caption_fixes or []) + caption_gaps))

    # Hard Phase 1 Gate rules (identical rigor to external Master LLM gate)
    block_reasons: list[str] = []
    if loaded.fast_failed:
        block_reasons.extend(loaded.fast_fail_reasons)
    if master_conclusion_match == "PENDING":
        block_reasons.append(
            "Awaiting chat Master verification of Reviewer 1's conclusion "
            "(compare `blind_one_sentence_conclusion` against ground truth and re-call with "
            "`master_conclusion_match='MATCH'` to unlock Phase 2)."
        )
    elif master_conclusion_match != "MATCH":
        block_reasons.append(
            f"Physical conclusion is {master_conclusion_match} (requires MATCH)."
        )
    if deterministic_hits:
        terms = sorted({h.term for h in deterministic_hits})
        block_reasons.append(
            f"Deterministic jargon/banned-term/midpoint scanner flagged: {', '.join(terms[:8])}."
        )
    if opaque_terms:
        block_reasons.append(
            f"Reviewer 1 flagged {len(opaque_terms)} OPAQUE_JARGON label(s): "
            f"{', '.join(t.term for t in opaque_terms[:6])}."
        )
    if critical_visual and not loaded.fast_failed:
        block_reasons.append(
            f"Reviewer 1 flagged {len(critical_visual)} critical visual defect(s)."
        )
    if contrast_or_unit_blockers and not loaded.fast_failed:
        cats = sorted({v.category for v in contrast_or_unit_blockers})
        block_reasons.append(
            f"Reviewer 1 flagged {len(contrast_or_unit_blockers)} scientific contrast / modeled-variable / top-bottom unit defect(s) ({', '.join(cats)}) — failed 2-second Eye-Test."
        )
    if forensics.rainbow_jet_suspected and not loaded.fast_failed:
        block_reasons.append("Deterministic color forensics detected rainbow/jet colormap.")
    if forensics.saturation_clipping_suspected and not loaded.fast_failed:
        block_reasons.append(
            f"Deterministic color forensics detected dark saturation clipping "
            f"({forensics.saturated_dark_extreme_fraction:.1%} of pixels)."
        )
    if loaded.colormap_audit.colormap_role_match == "MISMATCHED" and not loaded.fast_failed:
        block_reasons.append("Reviewer 1 flagged colormap_role_match=MISMATCHED.")
    if total_score < pass_threshold and master_conclusion_match != "PENDING":
        block_reasons.append(
            f"Phase 1 score {total_score}/100 is below threshold {pass_threshold}/100."
        )

    passed = len(block_reasons) == 0
    if passed:
        verdict: Literal["PASS", "REVISE_CAPTION", "REDO_FIGURE"] = "PASS"
    elif (
        not loaded.fast_failed
        and not deterministic_hits
        and not opaque_terms
        and not critical_visual
        and not contrast_or_unit_blockers
        and not forensics.rainbow_jet_suspected
        and not forensics.saturation_clipping_suspected
        and master_conclusion_match == "MATCH"
        and caption_fixes
    ):
        verdict = "REVISE_CAPTION"
    else:
        verdict = "REDO_FIGURE"

    # Colormap compliance summary
    if forensics.rainbow_jet_suspected or forensics.saturation_clipping_suspected or midpoint_hits:
        cm_compliance: Literal["ALIGNED", "NEEDS_TUNING", "NON_COMPLIANT"] = "NON_COMPLIANT"
    elif color_fixes:
        cm_compliance = "NEEDS_TUNING"
    else:
        cm_compliance = "ALIGNED"

    recommended_cmap = (
        "Soft RdBu_r [0.10, 0.90] + TwoSlopeNorm(vcenter=0.0) or decade_sky"
        if (forensics.rainbow_jet_suspected or forensics.saturation_clipping_suspected or midpoint_hits)
        else ("decade2/3/4/5 or Sanzo Wada (vol1-164 / vol1-2)" if (is_esteves and forensics.tab10_suspected) else "Keep current")
    )

    checklist: list[str] = []
    if master_conclusion_match == "PENDING":
        checklist.append(
            f"Verify Reviewer 1 conclusion matches your intended physics: \"{loaded.one_sentence_conclusion}\""
        )
        checklist.append(
            "Verify every symbol in `unclear_terms` matches your intended meaning and `Table_of_Notation`."
        )
    if fn_jargon:
        checklist.append(
            f"Filename hygiene: rename `{current_fn}` -> `{suggested_fn}` (removes {fn_jargon})."
        )
    if waived_var_note:
        checklist.append(waived_var_note)
    if loaded.reader_expectations:
        exp_anchors = ", ".join(p.get("citation", "") for p in (loaded.reader_expectations.get("anchor_papers") or [])[:3])
        exp_layout = "; ".join((loaded.reader_expectations.get("expected_minimal_layout") or [])[:2])
        checklist.append(
            f"Expectation-First Check ({exp_anchors}): compare figure against expected minimal layout ({exp_layout})."
        )
    if not checklist:
        checklist.append(
            "All deterministic checks clean and chat Master attested MATCH — ready to proceed to Phase 2."
            if passed
            else "Resolve the listed `gate_block_reasons` and re-run Phase 1 before Phase 2."
        )

    grading_mode: Literal["chat_master_deterministic", "deterministic_pregate"] = (
        "deterministic_pregate" if loaded.fast_failed else "chat_master_deterministic"
    )
    elapsed = round(time.monotonic() - t0, 4)

    result = MasterGradeResult(
        verdict=verdict,
        conclusion_match=master_conclusion_match,
        conclusion_comparison_notes=(
            conclusion_notes
            or (
                "Fast-failed before LLM review due to deterministic 0-token pixel blockers."
                if loaded.fast_failed
                else (
                    "Chat Master attested MATCH against intended physical truth."
                    if master_conclusion_match == "MATCH"
                    else (
                        "Pending chat Master comparison of Reviewer 1's blind conclusion against ground truth."
                        if master_conclusion_match == "PENDING"
                        else f"Chat Master marked conclusion {master_conclusion_match}."
                    )
                )
            )
        ),
        score_breakdown=Phase1ScoreBreakdown(
            conclusion_accuracy_score=c_score,
            variable_clarity_score=v_score,
            visual_design_score=d_score,
            phase1_score=total_score,
        ),
        colormap_feedback=ColormapFeedback(
            matches_house_preferences=cm_compliance,
            palette_diagnosis=(
                f"Dominant hexes: {forensics.dominant_chromatic_hexes}; "
                f"house matches: {forensics.matched_house_palettes or ['none']}; "
                f"tab10={forensics.tab10_suspected}, rainbow={forensics.rainbow_jet_suspected}, "
                f"sat_clip={forensics.saturation_clipping_suspected}."
            ),
            stretch_and_midpoint_verdict=(
                "; ".join(h.detail for h in midpoint_hits)
                if midpoint_hits
                else loaded.colormap_audit.stretch_and_midpoint_notes
            ),
            recommended_colormap_or_palette=recommended_cmap,
            color_fixes=color_fixes,
        ),
        filename_audit=FilenameAudit(
            current_filename=current_fn,
            has_jargon=bool(fn_jargon),
            jargon_tokens_found=fn_jargon,
            suggested_filename=suggested_fn,
            reason=(
                f"Replace opaque batch/code token(s) {fn_jargon} with clear physical descriptors."
                if fn_jargon
                else "Filename is clean and descriptive."
            ),
        ),
        label_comparisons=[],
        notation_violations=notation_violations,
        caption_gaps=caption_gaps,
        figure_code_fixes=figure_code_fixes,
        caption_fixes=caption_fixes,
        master_summary=(
            f"0-token deterministic Master Gate ({grading_mode}): {verdict} "
            f"({total_score}/100, passed={passed}). "
            + (
                f"Blocked by: {'; '.join(block_reasons)}"
                if block_reasons
                else "Zero deterministic jargon, midpoint, or pixel defects; unlocked for Phase 2."
            )
        ),
        phase1_grade_id=f"p1_det_{uuid.uuid4().hex[:10]}",
        review_id=loaded.review_id,
        phase1_score=total_score,
        phase1_letter_grade=score_to_letter_grade(total_score),
        phase1_passed=passed,
        gate_block_reasons=block_reasons,
        pass_threshold=pass_threshold,
        style_profile=style_profile,
        color_forensics=forensics,
        deterministic_hits=deterministic_hits,
        vector_label_audit=vector_audit,
        grading_mode=grading_mode,
        blind_one_sentence_conclusion=loaded.one_sentence_conclusion,
        chat_master_checklist=checklist,
        grading_backend=grading_mode,
        grading_model="0-token-deterministic",
        duration_seconds=elapsed,
    )
    save_phase1_grade(result)
    return result


async def run_master_grade(
    blind_review: str | dict[str, Any] | BlindFigureReview,
    *,
    intended_conclusion: str,
    intended_labels: dict[str, str] | None = None,
    caption: str | None = None,
    notation_table_path: str | Path | None = None,
    banned_terms: list[str] | None = None,
    style_profile: str = "esteves",
    color_preferences: str | None = None,
    caption_stage_blind_review: BlindFigureReview | None = None,
    pass_threshold: int = 80,
    allow_transformed_variable_reason: str | None = None,
    backend: str = "auto",
    model: str | None = None,
    speed: str = "fast",
    timeout_seconds: int = 75,
) -> MasterGradeResult:
    """Phase 1 Master Gate: compare Reviewer 1 against truth, score 0-100, enforce pass gate."""
    loaded = load_blind_review(blind_review)
    if loaded.fast_failed:
        return run_deterministic_grade(
            loaded,
            master_conclusion_match="MISMATCH",
            conclusion_notes="Fast-failed before LLM review due to 0-token pixel/colormap blockers.",
            notation_table_path=notation_table_path,
            banned_terms=banned_terms,
            style_profile=style_profile,
            pass_threshold=pass_threshold,
        )
    notation_spec: NotationSpec | None = (
        parse_notation_table(notation_table_path) if notation_table_path else None
    )
    deterministic_hits = scan_blind_review_deterministically(
        loaded,
        notation_spec=notation_spec,
        extra_banned_terms=banned_terms,
    )
    deterministic_hits.extend(audit_colorbar_midpoints_deterministically(loaded))
    vector_audit, vector_hits = run_vector_label_audit(
        loaded,
        caption=caption,
        notation_table_path=notation_table_path,
    )
    deterministic_hits = _merge_deterministic_hits(deterministic_hits, vector_hits)
    fn_jargon = (
        audit_filename_jargon(loaded.source_filename, notation_spec=notation_spec)
        if loaded.source_filename
        else []
    )

    system_prompt = build_master_system_prompt(pass_threshold=pass_threshold)
    user_prompt = build_master_user_prompt(
        blind_review=loaded,
        intended_conclusion=intended_conclusion,
        intended_labels=intended_labels,
        caption=caption,
        notation_spec=notation_spec,
        deterministic_hits=deterministic_hits,
        color_forensics=loaded.color_forensics,
        source_filename=loaded.source_filename,
        filename_jargon_tokens=fn_jargon,
        style_profile=style_profile,
        color_preferences=color_preferences,
        caption_stage_blind_review=caption_stage_blind_review,
        vector_label_audit=vector_audit,
    )

    call = await call_structured_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_model=MasterGradePayload,
        prepared_image=None,
        backend=backend,
        model=model,
        speed=speed,
        timeout_seconds=timeout_seconds,
    )
    assert isinstance(call.parsed, MasterGradePayload)

    # Recompute Phase 1 score deterministically from sub-scores to guarantee arithmetic integrity
    sb = call.parsed.score_breakdown
    contrast_or_unit_blockers, _waived_note = _collect_contrast_blockers(
        loaded,
        allow_transformed_variable_reason=allow_transformed_variable_reason,
    )
    recomputed_score = max(
        0,
        min(100, sb.conclusion_accuracy_score + sb.variable_clarity_score + sb.visual_design_score),
    )
    if contrast_or_unit_blockers and recomputed_score > 74:
        excess = recomputed_score - 74
        cut_d = min(sb.visual_design_score, excess)
        sb.visual_design_score -= cut_d
        excess -= cut_d
        cut_c = min(sb.conclusion_accuracy_score, excess)
        sb.conclusion_accuracy_score -= cut_c
        excess -= cut_c
        sb.variable_clarity_score = max(0, sb.variable_clarity_score - excess)
        recomputed_score = (
            sb.conclusion_accuracy_score + sb.variable_clarity_score + sb.visual_design_score
        )
    sb.phase1_score = recomputed_score

    # Deterministic Phase 1 hard-gate enforcement
    block_reasons: list[str] = []
    if call.parsed.verdict != "PASS":
        block_reasons.append(f"Master verdict is {call.parsed.verdict} (requires PASS).")
    if call.parsed.conclusion_match != "MATCH":
        block_reasons.append(
            f"Physical conclusion is {call.parsed.conclusion_match} (requires MATCH)."
        )
    if recomputed_score < pass_threshold:
        block_reasons.append(
            f"Phase 1 score {recomputed_score}/100 is below threshold {pass_threshold}/100."
        )
    if deterministic_hits:
        terms = sorted({h.term for h in deterministic_hits})
        block_reasons.append(
            f"Deterministic jargon/banned-term scanner flagged: {', '.join(terms[:8])}."
        )
    critical_visual = [
        v for v in loaded.visual_and_layout_issues if v.severity == "critical"
    ]
    if critical_visual:
        block_reasons.append(
            f"Reviewer 1 flagged {len(critical_visual)} critical visual defect(s)."
        )
    if contrast_or_unit_blockers:
        cats = sorted({v.category for v in contrast_or_unit_blockers})
        block_reasons.append(
            f"Reviewer 1 flagged {len(contrast_or_unit_blockers)} scientific contrast / modeled-variable / top-bottom unit defect(s) ({', '.join(cats)}) — failed 2-second Eye-Test."
        )

    passed = len(block_reasons) == 0
    enforced_verdict = call.parsed.verdict
    if not passed and enforced_verdict == "PASS":
        enforced_verdict = "REDO_FIGURE"

    payload_dict = call.parsed.model_dump()
    payload_dict["verdict"] = enforced_verdict
    payload_dict["score_breakdown"]["phase1_score"] = recomputed_score

    result = MasterGradeResult(
        **payload_dict,
        phase1_grade_id=f"p1_{uuid.uuid4().hex[:10]}",
        review_id=loaded.review_id,
        phase1_score=recomputed_score,
        phase1_letter_grade=score_to_letter_grade(recomputed_score),
        phase1_passed=passed,
        gate_block_reasons=block_reasons,
        pass_threshold=pass_threshold,
        style_profile=style_profile,
        color_forensics=loaded.color_forensics,
        deterministic_hits=deterministic_hits,
        vector_label_audit=vector_audit,
        grading_mode="llm_master",
        blind_one_sentence_conclusion=loaded.one_sentence_conclusion,
        grading_backend=call.backend,
        grading_model=call.model,
        duration_seconds=call.duration_seconds,
    )
    save_phase1_grade(result)
    return result


# ---------------------------------------------------------------------------
# Phase 2: Reviewer 2 (Context Loop — Manuscript Integration)
# ---------------------------------------------------------------------------

async def run_phase2_context_review(
    image_path: str | Path,
    *,
    phase1_grade: str | dict[str, Any] | MasterGradeResult,
    tex_path: str | Path,
    caption: str | None = None,
    notation_table_path: str | Path | None = None,
    figure_label: str | None = None,
    section_hint: str | None = None,
    domain: str = "astrophysics",
    pass_threshold: int = 80,
    force_override: bool = False,
    page: int = 1,
    dpi: int | None = None,
    backend: str = "auto",
    model: str | None = None,
    speed: str = "fast",
    timeout_seconds: int = 75,
) -> Phase2ReviewResult:
    """Phase 2 / Reviewer 2: notation hygiene & narrative alignment (strictly gated on Phase 1)."""
    p1 = load_phase1_grade(phase1_grade)
    if not p1.phase1_passed and not force_override:
        reasons = "; ".join(p1.gate_block_reasons) or f"verdict={p1.verdict}"
        raise Phase1GateBlockedError(
            f"Iteration Rule blocked Phase 2: Phase 1 grade `{p1.phase1_grade_id}` did NOT pass "
            f"(score {p1.phase1_score}/100, {reasons}). Edit the plot or caption and re-run "
            "Phase 1 until it passes before integrating into the manuscript."
        )

    eff_dpi, eff_max_dim, eff_fmt = image_settings_for_speed(speed, dpi=dpi)
    prepared = prepare_blind_image(
        image_path,
        page=page,
        dpi=eff_dpi,
        max_dimension=eff_max_dim,
        output_format=eff_fmt,
    )
    tex_ctx = extract_tex_context(
        tex_path,
        image_path=image_path,
        figure_label=figure_label,
        section_hint=section_hint,
    )
    notation_spec: NotationSpec | None = (
        parse_notation_table(notation_table_path) if notation_table_path else None
    )
    uses_sandbox = backend_uses_cli_sandbox(backend)

    system_prompt = build_reviewer2_system_prompt(domain=domain, pass_threshold=pass_threshold)
    user_prompt = build_reviewer2_user_prompt(
        tex_context=tex_ctx,
        caption=caption,
        notation_spec=notation_spec,
        phase1_grade=p1,
        cli_image_filename=prepared.blind_filename if uses_sandbox else None,
    )

    call = await call_structured_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_model=Phase2ReviewPayload,
        prepared_image=prepared,
        backend=backend,
        model=model,
        speed=speed,
        timeout_seconds=timeout_seconds,
    )
    assert isinstance(call.parsed, Phase2ReviewPayload)

    sb = call.parsed.score_breakdown
    p2_score = max(0, min(100, sb.notation_hygiene_score + sb.narrative_alignment_score))
    sb.phase2_score = p2_score

    blocking_hygiene = [
        item
        for item in call.parsed.symbol_hygiene
        if item.status in {"MISMATCH", "CODE_JARGON", "UNIT_MISMATCH", "AD_HOC_VARIABLE"}
    ]
    p2_passed = (
        call.parsed.phase2_verdict == "PASS"
        and p2_score >= pass_threshold
        and len(blocking_hygiene) == 0
    )
    enforced_p2_verdict = call.parsed.phase2_verdict
    if not p2_passed and enforced_p2_verdict == "PASS":
        enforced_p2_verdict = "REDO_FIGURE" if blocking_hygiene else "REVISE_TEX"

    payload_dict = call.parsed.model_dump()
    payload_dict["phase2_verdict"] = enforced_p2_verdict
    payload_dict["score_breakdown"]["phase2_score"] = p2_score

    final_summary = compute_final_score_summary(
        p1,
        call.parsed,
        phase2_score=p2_score,
        phase2_passed=p2_passed,
    )

    return Phase2ReviewResult(
        **payload_dict,
        phase2_review_id=f"p2_{uuid.uuid4().hex[:10]}",
        phase1_grade_id=p1.phase1_grade_id,
        phase2_score=p2_score,
        phase2_letter_grade=score_to_letter_grade(p2_score),
        phase2_passed=p2_passed,
        gate_overridden=(not p1.phase1_passed and force_override),
        final_score_summary=final_summary,
        tex_path=tex_ctx.tex_path,
        figure_label_matched=tex_ctx.figure_label,
        backend=call.backend,
        model=call.model,
        duration_seconds=call.duration_seconds,
    )


# ---------------------------------------------------------------------------
# Reporting & End-to-End Orchestration
# ---------------------------------------------------------------------------

def format_markdown_report(
    *,
    figure_label: str,
    blind_review: BlindFigureReview,
    master_grade: MasterGradeResult | None = None,
    caption_stage_review: BlindFigureReview | None = None,
    phase2_result: Phase2ReviewResult | None = None,
) -> str:
    """Render a clean, scored Markdown peer-review report across Phase 1 & Phase 2."""
    cert = blind_review.blinding_certificate
    lines: list[str] = [f"# Scientific Figure Peer Review: `{figure_label}`", ""]

    if master_grade is not None:
        summary = (
            phase2_result.final_score_summary
            if phase2_result is not None
            else compute_final_score_summary(master_grade)
        )
        lines.extend(
            [
                "## Scoreboard",
                "| Stage | Score | Gate / Status |",
                "| :--- | :--- | :--- |",
                f"| **Phase 1 — Visual Clarity (Reviewer 1)** | "
                f"**{master_grade.phase1_score}/100 ({master_grade.phase1_letter_grade})** "
                f"(Conclusion {master_grade.score_breakdown.conclusion_accuracy_score}/40, "
                f"Variables {master_grade.score_breakdown.variable_clarity_score}/35, "
                f"Visuals {master_grade.score_breakdown.visual_design_score}/25) | "
                f"{'PASS' if master_grade.phase1_passed else f'BLOCKED ({master_grade.verdict})'} |",
            ]
        )
        if phase2_result is not None:
            lines.append(
                f"| **Phase 2 — Manuscript Integration (Reviewer 2)** | "
                f"**{phase2_result.phase2_score}/100 ({phase2_result.phase2_letter_grade})** "
                f"(Notation {phase2_result.score_breakdown.notation_hygiene_score}/50, "
                f"Narrative {phase2_result.score_breakdown.narrative_alignment_score}/50) | "
                f"{'PASS' if phase2_result.phase2_passed else phase2_result.phase2_verdict} |"
            )
        else:
            p2_status = (
                "Not run (no `.tex` provided)"
                if master_grade.phase1_passed
                else "BLOCKED by Iteration Rule (Phase 1 must pass first)"
            )
            lines.append(f"| **Phase 2 — Manuscript Integration (Reviewer 2)** | — | {p2_status} |")
        lines.extend(
            [
                f"| **FINAL SCORE** | **{summary.final_score}/100 ({summary.letter_grade})** | "
                f"**{summary.final_verdict}** |",
                "",
            ]
        )

    lines.extend(
        [
            "## Blinding Certificate (Reviewer 1)",
            f"- **Review ID:** `{blind_review.review_id}` (Round {blind_review.round_number})",
            f"- **Backend / Model:** `{cert.backend}` / `{cert.model}` ({cert.blinding_level})",
            f"- **Original Filename Hidden:** `{cert.original_filename_hidden}` | "
            f"**Metadata Stripped:** `{cert.metadata_stripped}` "
            f"(stripped keys: `{cert.stripped_metadata_keys or ['none']}`)",
            f"- **Caption Provided:** `{cert.caption_provided}` | "
            f"**Project Context Provided:** `{cert.project_context_provided}`",
            f"- **Image:** `{cert.source_format}` -> `{cert.payload_format}` "
            f"{cert.image_width_px}x{cert.image_height_px} px "
            f"({cert.payload_bytes / 1024:.0f} KB, sha256: `{cert.image_sha256[:12]}...`, "
            f"latency: {cert.duration_seconds:.1f}s)",
        ]
    )
    if blind_review.reader_expectations:
        exp = blind_review.reader_expectations
        anchors = ", ".join(p.get("citation", "") for p in (exp.get("anchor_papers") or [])) or "field general"
        lines.extend(
            [
                "",
                "## Pre-Figure Blind Reader Expectations (Vector-Selected Anchor Papers, Zero Pixels)",
                f"- **Anchor Papers ({exp.get('papers_indexed', 0)} indexed, {exp.get('paper_selection_ms', 0)} ms):** {anchors}",
                f"- **Persona:** {exp.get('reader_persona_summary', '')}",
                "- **Expected Minimal Layout:** " + "; ".join(exp.get("expected_minimal_layout") or ["(none)"]),
                "- **Expected Benchmarks:** " + "; ".join(exp.get("expected_literature_benchmarks") or ["(none)"]),
                "- **Cut / Move to Table or Appendix:** " + "; ".join(exp.get("cut_or_move_to_table_or_appendix") or ["(none)"]),
            ]
        )
    lines.extend(
        [
            "",
            "## Phase 1: Reviewer 1 Blind Assessment (Visual Clarity)",
            f"- **One-Sentence Physical Conclusion:** {blind_review.one_sentence_conclusion}",
            f"- **Standalone Readability Rating:** **{blind_review.standalone_readability_score}/5** — "
            f"{blind_review.readability_summary}",
            "",
            "### Visual Encodings (Axes, Colors, Markers)",
        ]
    )
    for p in blind_review.panel_encodings:
        lines.extend(
            [
                f"#### {p.panel_id}: {p.panel_title}",
                f"- **X-axis:** {p.x_axis}",
                f"- **Y-axis:** {p.y_axis}",
                f"- **Colors / Colorbar:** {p.colors_and_colorbar}",
                f"- **Markers / Linestyles:** {p.markers_and_linestyles}",
                f"- **Annotations / Legends:** {p.annotations_and_legends}",
                "",
            ]
        )

    lines.append("### Variable & Label Definitions (Blind Audit)")
    if blind_review.unclear_terms:
        lines.extend(
            [
                "| Term | Location | Tag | Blind Definition | Reason |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for item in blind_review.unclear_terms:
            lines.append(
                f"| `{item.term}` | {item.location} | **{item.clarity_tag}** | "
                f"{item.blind_guess} | {item.reason} |"
            )
    else:
        lines.append("- None flagged.")

    lines.extend(
        [
            "",
            "### Colormap & Palette Audit (Reviewer 1 Blind + 0-Token Color Forensics)",
            f"- **Observed Palette:** {blind_review.colormap_audit.observed_palette_description} "
            f"(Role match: **{blind_review.colormap_audit.colormap_role_match}**)",
            f"- **Colorbar Stretch & Midpoint:** {blind_review.colormap_audit.stretch_and_midpoint_notes}",
            f"- **Contrast & Accessibility:** {blind_review.colormap_audit.contrast_and_accessibility}",
            f"- **Dominant Hexes (0-token):** `{blind_review.color_forensics.dominant_chromatic_hexes}` | "
            f"House matches: `{blind_review.color_forensics.matched_house_palettes or ['none']}` | "
            f"tab10={blind_review.color_forensics.tab10_suspected}, "
            f"rainbow/jet={blind_review.color_forensics.rainbow_jet_suspected}",
        ]
    )
    for cm_issue in blind_review.colormap_audit.colormap_issues:
        lines.append(f"- **Colormap Issue:** {cm_issue}")

    lines.extend(["", "### Visual & Layout Issues"])
    if blind_review.visual_and_layout_issues:
        for issue in blind_review.visual_and_layout_issues:
            lines.append(
                f"- **[{issue.severity.upper()} / {issue.category}]** ({issue.location}): "
                f"{issue.description}"
            )
    else:
        lines.append("- None flagged.")

    lines.extend(["", "### Confusing Elements"])
    for item in blind_review.confusing_elements or ["None flagged."]:
        lines.append(f"- {item}")

    if caption_stage_review is not None:
        lines.extend(
            [
                "",
                "### Reviewer 1 Follow-Up With Draft Caption",
                f"- **Conclusion With Caption:** {caption_stage_review.one_sentence_conclusion}",
                f"- **Readability With Caption:** **{caption_stage_review.standalone_readability_score}/5** — "
                f"{caption_stage_review.readability_summary}",
            ]
        )

    if master_grade is not None:
        lines.extend(
            [
                "",
                f"## Phase 1 Master Gate: **{master_grade.verdict}** "
                f"({master_grade.phase1_score}/100, `{master_grade.phase1_grade_id}`)",
                f"- **Conclusion Match:** `{master_grade.conclusion_match}` — "
                f"{master_grade.conclusion_comparison_notes}",
                f"- **Executive Summary:** {master_grade.master_summary}",
            ]
        )
        if master_grade.gate_block_reasons:
            lines.append("- **Gate Block Reasons:**")
            for reason in master_grade.gate_block_reasons:
                lines.append(f"  - {reason}")

        lines.extend(["", "### Deterministic Jargon & Notation Hits (0-Token Check)"])
        if master_grade.deterministic_hits:
            for hit in master_grade.deterministic_hits:
                lines.append(
                    f"- **`{hit.term}`** (`{hit.source}` at {hit.location}): {hit.detail}"
                )
        else:
            lines.append("- None detected deterministically.")

        va = master_grade.vector_label_audit
        if va and va.get("enabled"):
            lines.extend(
                [
                    "",
                    f"### Vectorized Field Label & Jargon Audit ({va.get('papers_indexed', 0)} Papers, 0 Tokens)",
                    f"- **Command:** `{va.get('vector_command', 'astro-math-review figure-labels')}` "
                    f"({va.get('labels_audited', 0)} labels, {va.get('duration_ms', 0)} ms)",
                ]
            )
            for b in (va.get("banned_words_detected") or [])[:8]:
                lines.append(
                    f"- **Banned Word [{b.get('severity')}]** ({b['location']}): `{b['matched']}` "
                    f"(df={b.get('field_df_papers', 0)}/{va.get('papers_indexed', 0)}) -> {b.get('replacement', '')}"
                )
            for s in (va.get("stats_jargon_detected") or [])[:8]:
                lines.append(
                    f"- **Stats Jargon** ({s['location']}): `{s['matched']}` "
                    f"(df={s.get('field_df_papers', 0)}/{va.get('papers_indexed', 0)}) -> {s.get('astro_replacement', '')}"
                )
            for c in (va.get("symbol_collisions") or [])[:6]:
                lines.append(
                    f"- **Symbol Near-Collision** ({c['location']}): `{c['symbol']}` — "
                    f"{(c.get('collision_warnings') or [''])[0]}"
                )

        lines.extend(["", "### Label Truth-vs-Blind Table"])
        if master_grade.label_comparisons:
            lines.extend(
                [
                    "| Term | Blind Guess | Ground Truth | Status | Recommended Fix |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for row in master_grade.label_comparisons:
                lines.append(
                    f"| `{row.term}` | {row.blind_guess} | {row.intended_meaning} | "
                    f"**{row.status}** | {row.recommended_fix} |"
                )
        else:
            lines.append("- No label discrepancies.")

        lines.extend(
            [
                "",
                f"### Colormap & House Style Feedback (`{master_grade.style_profile}` profile)",
                f"- **Preference Compliance:** **{master_grade.colormap_feedback.matches_house_preferences}**",
                f"- **Palette Diagnosis:** {master_grade.colormap_feedback.palette_diagnosis}",
                f"- **Stretch & Midpoint:** {master_grade.colormap_feedback.stretch_and_midpoint_verdict}",
                f"- **Recommended Colormap/Palette:** `{master_grade.colormap_feedback.recommended_colormap_or_palette}`",
            ]
        )
        for cfix in master_grade.colormap_feedback.color_fixes:
            lines.append(f"- **Color Fix:** {cfix}")

        fa = master_grade.filename_audit
        lines.extend(
            [
                "",
                "### Filename Hygiene Audit",
                f"- **Current:** `{fa.current_filename}` | **Has Jargon:** `{fa.has_jargon}` "
                f"(tokens: `{fa.jargon_tokens_found or ['none']}`)",
                f"- **Suggested Filename:** `{fa.suggested_filename}` — {fa.reason}",
            ]
        )

        lines.extend(["", "### Required Figure Code Fixes (Phase 1)"])
        for fix in master_grade.figure_code_fixes or ["None."]:
            lines.append(f"- {fix}")

        lines.extend(["", "### Required Caption Fixes (Phase 1)"])
        for fix in master_grade.caption_fixes or ["None."]:
            lines.append(f"- {fix}")

    if phase2_result is not None:
        lines.extend(
            [
                "",
                f"## Phase 2: Reviewer 2 Manuscript Integration — **{phase2_result.phase2_verdict}** "
                f"({phase2_result.phase2_score}/100, `{phase2_result.phase2_review_id}`)",
                f"- **Manuscript:** `{phase2_result.tex_path}` "
                f"(figure label: `{phase2_result.figure_label_matched or 'auto-section'}`)",
                f"- **Narrative Guidance:** {phase2_result.narrative_guidance_assessment}",
                f"- **Summary:** {phase2_result.reviewer2_summary}",
                "",
                "### Symbol Hygiene (Figure vs. `.tex` & Table of Notation)",
            ]
        )
        if phase2_result.symbol_hygiene:
            lines.extend(
                [
                    "| Symbol in Figure | Canonical `.tex` / Table | Status | Fix |",
                    "| :--- | :--- | :--- | :--- |",
                ]
            )
            for sym in phase2_result.symbol_hygiene:
                lines.append(
                    f"| `{sym.symbol_in_figure}` | `{sym.symbol_in_tex_or_table}` | "
                    f"**{sym.status}** | {sym.fix} |"
                )
        else:
            lines.append("- No symbol hygiene discrepancies flagged.")

        lines.extend(["", "### Orphaned Terms & Unreferenced Elements"])
        lines.append(
            "- **Orphaned / ad-hoc terms:** "
            + (", ".join(f"`{t}`" for t in phase2_result.orphaned_terms) or "None.")
        )
        lines.append(
            "- **Unreferenced figure elements:** "
            + ("; ".join(phase2_result.unreferenced_figure_elements) or "None.")
        )
        lines.append(
            "- **Unsupported `.tex` claims:** "
            + ("; ".join(phase2_result.unsupported_text_claims) or "None.")
        )

        lines.extend(["", "### Required `.tex` Edits (Phase 2)"])
        for edit in phase2_result.tex_edits or ["None."]:
            lines.append(f"- {edit}")

        lines.extend(["", "### Required Figure / Caption Edits (Phase 2)"])
        for edit in (phase2_result.figure_label_fixes + phase2_result.caption_fixes) or ["None."]:
            lines.append(f"- {edit}")

    return "\n".join(lines) + "\n"


async def run_end_to_end_review(
    image_path: str | Path,
    *,
    intended_conclusion: str,
    intended_labels: dict[str, str] | None = None,
    caption: str | None = None,
    notation_table_path: str | Path | None = None,
    banned_terms: list[str] | None = None,
    style_profile: str = "esteves",
    color_preferences: str | None = None,
    tex_path: str | Path | None = None,
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
    save_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the gated two-phase review pipeline (fast by default, 0-token fail-fast pre-gate)."""
    wall_t0 = time.monotonic()
    stage1_caption = caption if (include_caption_in_phase1 and not test_with_caption) else None

    if test_with_caption and caption and caption.strip():
        stage1, stage2_caption_review = await asyncio.gather(
            run_blind_review(
                image_path,
                domain=domain,
                caption=None,
                extra_checks=extra_checks,
                topic=topic,
                methodology=methodology,
                pin_papers=pin_papers,
                top_k_papers=top_k_papers,
                expectation_first=expectation_first,
                page=page,
                dpi=dpi,
                backend=backend,
                model=blind_model,
                speed=speed,
                detail=detail,
                use_cache=use_cache,
                fail_fast=fail_fast,
                fail_fast_on_filename_jargon=fail_fast_on_filename_jargon,
            ),
            run_blind_review(
                image_path,
                domain=domain,
                caption=caption,
                extra_checks=extra_checks,
                page=page,
                dpi=dpi,
                backend=backend,
                model=blind_model,
                speed=speed,
                detail=detail,
                use_cache=use_cache,
                fail_fast=fail_fast,
                fail_fast_on_filename_jargon=fail_fast_on_filename_jargon,
            ),
        )
        if stage1.fast_failed:
            stage2_caption_review = None
    else:
        stage1 = await run_blind_review(
            image_path,
            domain=domain,
            caption=stage1_caption,
            extra_checks=extra_checks,
            topic=topic,
            methodology=methodology,
            pin_papers=pin_papers,
            top_k_papers=top_k_papers,
            expectation_first=expectation_first,
            page=page,
            dpi=dpi,
            backend=backend,
            model=blind_model,
            speed=speed,
            detail=detail,
            use_cache=use_cache,
            fail_fast=fail_fast,
            fail_fast_on_filename_jargon=fail_fast_on_filename_jargon,
        )
        stage2_caption_review = None

    if stage1.fast_failed or master_mode == "deterministic":
        master = run_deterministic_grade(
            stage1,
            master_conclusion_match="MISMATCH" if stage1.fast_failed else master_conclusion_match,
            notation_table_path=notation_table_path,
            banned_terms=banned_terms,
            style_profile=style_profile,
            pass_threshold=pass_threshold,
        )
    else:
        master = await run_master_grade(
            stage1,
            intended_conclusion=intended_conclusion,
            intended_labels=intended_labels,
            caption=caption,
            notation_table_path=notation_table_path,
            banned_terms=banned_terms,
            style_profile=style_profile,
            color_preferences=color_preferences,
            caption_stage_blind_review=stage2_caption_review,
            pass_threshold=pass_threshold,
            backend=backend,
            model=master_model or blind_model,
            speed=speed,
        )

    # Hard Iteration Rule: Only run Phase 2 if Phase 1 passed AND tex_path was supplied
    phase2: Phase2ReviewResult | None = None
    phase2_status = "NOT_REQUESTED"
    if tex_path is not None:
        if master.phase1_passed:
            phase2 = await run_phase2_context_review(
                image_path,
                phase1_grade=master,
                tex_path=tex_path,
                caption=caption,
                notation_table_path=notation_table_path,
                figure_label=figure_label,
                section_hint=section_hint,
                domain=domain,
                pass_threshold=pass_threshold,
                page=page,
                dpi=dpi,
                backend=backend,
                model=reviewer2_model or blind_model,
                speed=speed,
            )
            phase2_status = "PASSED" if phase2.phase2_passed else phase2.phase2_verdict
        else:
            phase2_status = "BLOCKED_BY_PHASE1"

    total_wall_seconds = round(time.monotonic() - wall_t0, 2)
    timings = {
        "phase1_reviewer1_s": stage1.blinding_certificate.duration_seconds,
        "phase1_master_gate_s": master.duration_seconds,
        "phase2_reviewer2_s": phase2.duration_seconds if phase2 is not None else 0.0,
        "total_wall_seconds": total_wall_seconds,
    }

    final_summary = (
        phase2.final_score_summary
        if phase2 is not None
        else compute_final_score_summary(master)
    )

    fig_name = Path(image_path).name
    report_md = format_markdown_report(
        figure_label=fig_name,
        blind_review=stage1,
        master_grade=master,
        caption_stage_review=stage2_caption_review,
        phase2_result=phase2,
    )
    saved_to: str | None = None
    if save_report_path:
        out_path = Path(save_report_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_md, encoding="utf-8")
        saved_to = str(out_path)

    result: dict[str, Any] = {
        "verdict": master.verdict,
        "final_verdict": final_summary.final_verdict,
        "final_score": final_summary.final_score,
        "letter_grade": final_summary.letter_grade,
        "phase1_score": master.phase1_score,
        "phase1_passed": master.phase1_passed,
        "phase1_grade_id": master.phase1_grade_id,
        "phase2_status": phase2_status,
        "phase2_score": phase2.phase2_score if phase2 is not None else None,
        "fast_failed": stage1.fast_failed,
        "fast_fail_reasons": stage1.fast_fail_reasons,
        "timings_seconds": timings,
        "standalone_readability_score": stage1.standalone_readability_score,
        "conclusion_match": master.conclusion_match,
        "final_score_summary": final_summary.model_dump(),
        "blind_review_stage1": stage1.model_dump(),
        "master_grade": master.model_dump(),
        "report_markdown": report_md,
    }
    if stage2_caption_review is not None:
        result["blind_review_stage2_with_caption"] = stage2_caption_review.model_dump()
    if phase2 is not None:
        result["phase2_review"] = phase2.model_dump()
    if saved_to is not None:
        result["saved_report_path"] = saved_to
    return result


async def run_batch_blind_review(
    image_paths: list[str],
    *,
    domain: str = "astrophysics",
    extra_checks: list[str] | None = None,
    backend: str = "auto",
    model: str | None = None,
    speed: str = "fast",
    detail: Literal["auto", "compact", "full"] = "auto",
    use_cache: bool = True,
    fail_fast: bool = True,
    fail_fast_on_filename_jargon: bool = False,
    max_concurrency: int = 5,
) -> dict[str, Any]:
    """Run independent Phase 1 Reviewer 1 blind reviews across multiple figures in parallel."""
    sem = asyncio.Semaphore(max(1, min(max_concurrency, 8)))

    async def _one(path: str) -> dict[str, Any]:
        async with sem:
            try:
                rev = await run_blind_review(
                    path,
                    domain=domain,
                    caption=None,
                    extra_checks=extra_checks,
                    backend=backend,
                    model=model,
                    speed=speed,
                    detail=detail,
                    use_cache=use_cache,
                    fail_fast=fail_fast,
                    fail_fast_on_filename_jargon=fail_fast_on_filename_jargon,
                )
                opaque = [t.term for t in rev.unclear_terms if t.clarity_tag == "OPAQUE_JARGON"]
                critical_major = [
                    i.description
                    for i in rev.visual_and_layout_issues
                    if i.severity in {"critical", "major"}
                ]
                return {
                    "image_path": path,
                    "status": "fast_failed" if rev.fast_failed else "ok",
                    "review_id": rev.review_id,
                    "fast_failed": rev.fast_failed,
                    "fast_fail_reasons": rev.fast_fail_reasons,
                    "standalone_readability_score": rev.standalone_readability_score,
                    "one_sentence_conclusion": rev.one_sentence_conclusion,
                    "opaque_jargon_terms": opaque,
                    "critical_or_major_visual_issues": critical_major,
                    "blind_review": rev.model_dump(),
                }
            except Exception as exc:
                return {"image_path": path, "status": "error", "error": str(exc)}

    results = await asyncio.gather(*(_one(p) for p in image_paths))
    return {
        "count": len(results),
        "max_concurrency": max(1, min(max_concurrency, 8)),
        "summary": [
            {
                "image_path": r["image_path"],
                "status": r["status"],
                "review_id": r.get("review_id"),
                "fast_failed": r.get("fast_failed", False),
                "fast_fail_reasons": r.get("fast_fail_reasons", []),
                "score": r.get("standalone_readability_score"),
                "opaque_jargon_count": len(r.get("opaque_jargon_terms", [])),
                "major_visual_issue_count": len(r.get("critical_or_major_visual_issues", [])),
                "one_sentence_conclusion": r.get("one_sentence_conclusion"),
                "error": r.get("error"),
            }
            for r in results
        ],
        "reviews": results,
    }

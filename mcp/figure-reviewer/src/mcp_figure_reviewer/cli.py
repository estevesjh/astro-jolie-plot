"""Standalone CLI for the gated two-phase Master -> Reviewer 1 -> Reviewer 2 workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .backends import detect_backends
from .blinding import inspect_figure_pixels
from .reviewer import (
    Phase1GateBlockedError,
    format_markdown_report,
    load_phase1_grade,
    run_batch_blind_review,
    run_blind_review,
    run_deterministic_grade,
    run_end_to_end_review,
    run_master_grade,
    run_phase2_context_review,
)


def _read_caption(caption: str | None, caption_file: str | None) -> str | None:
    if caption_file:
        return Path(caption_file).expanduser().read_text(encoding="utf-8").strip()
    return caption


def _run_interactive_loop(args: argparse.Namespace) -> int:
    """Interactive terminal Master loop: iterate Phase 1 until it passes, then unlock Phase 2."""
    round_num = 1
    caption_text = _read_caption(args.caption, args.caption_file)

    while True:
        print(f"\n{'=' * 72}")
        print(f"PHASE 1 (BLIND LOOP — ROUND {round_num}): Invoking Reviewer 1...")
        print(f"{'=' * 72}")
        rev1 = asyncio.run(
            run_blind_review(
                args.image,
                domain=args.domain,
                caption=caption_text if not args.pixels_only else None,
                round_number=round_num,
                page=args.page,
                backend=args.backend,
                speed=args.speed,
                model=args.model,
            )
        )
        print(f"Reviewer 1 ({rev1.blinding_certificate.model}): {rev1.one_sentence_conclusion}")
        print(f"Standalone Readability: {rev1.standalone_readability_score}/5")

        print("\nEvaluating Reviewer 1 against ground truth (Master Gate)...")
        grade1 = asyncio.run(
            run_master_grade(
                rev1,
                intended_conclusion=args.truth,
                caption=caption_text,
                notation_table_path=args.notation,
                style_profile=args.style_profile,
                color_preferences=args.color_prefs,
                pass_threshold=args.threshold,
                backend=args.backend,
                speed=args.speed,
                model=args.model,
            )
        )
        print(
            f"\nPHASE 1 RESULT: {grade1.verdict} | "
            f"Score: {grade1.phase1_score}/100 ({grade1.phase1_letter_grade}) | "
            f"Gate Passed: {grade1.phase1_passed} (id={grade1.phase1_grade_id})"
        )
        print(
            f"  Breakdown: Conclusion {grade1.score_breakdown.conclusion_accuracy_score}/40 | "
            f"Variables {grade1.score_breakdown.variable_clarity_score}/35 | "
            f"Visuals {grade1.score_breakdown.visual_design_score}/25"
        )
        if grade1.gate_block_reasons:
            print("  Block Reasons:")
            for r in grade1.gate_block_reasons:
                print(f"    - {r}")
        if grade1.figure_code_fixes:
            print("  Figure Fixes Needed:")
            for f in grade1.figure_code_fixes:
                print(f"    - {f}")
        cm = grade1.colormap_feedback
        print(
            f"  Colormap ({grade1.style_profile}): {cm.matches_house_preferences} | "
            f"Recommended: {cm.recommended_colormap_or_palette}"
        )
        for cf in cm.color_fixes:
            print(f"    - {cf}")
        if grade1.caption_fixes:
            print("  Caption Fixes Needed:")
            for f in grade1.caption_fixes:
                print(f"    - {f}")

        if grade1.phase1_passed:
            print("\nPhase 1 PASSED! Unlocking Phase 2 (Manuscript Integration — Reviewer 2)...")
            break

        if round_num >= args.max_rounds:
            print(f"\nStopped after max_rounds={args.max_rounds} with Phase 1 still blocked.")
            return 2

        try:
            ans = input(
                "\nEdit your plot/caption, then press [Enter] to re-run Phase 1 "
                "(or type 'q' to stop): "
            ).strip()
        except EOFError:
            return 2
        if ans.lower() in {"q", "quit", "exit"}:
            return 2
        caption_text = _read_caption(args.caption, args.caption_file)
        round_num += 1

    # Phase 2 unlocked!
    if not args.tex:
        report = format_markdown_report(
            figure_label=Path(args.image).name,
            blind_review=rev1,
            master_grade=grade1,
        )
        if args.out:
            Path(args.out).write_text(report, encoding="utf-8")
            print(f"Saved report to {args.out}")
        print(f"\nFINAL SCORE (Phase 1 only): {grade1.phase1_score}/100 ({grade1.phase1_letter_grade})")
        return 0

    print(f"\n{'=' * 72}")
    print("PHASE 2 (CONTEXT LOOP): Invoking Reviewer 2 against .tex manuscript...")
    print(f"{'=' * 72}")
    p2 = asyncio.run(
        run_phase2_context_review(
            args.image,
            phase1_grade=grade1,
            tex_path=args.tex,
            caption=caption_text,
            notation_table_path=args.notation,
            figure_label=args.fig_label,
            section_hint=args.section,
            domain=args.domain,
            pass_threshold=args.threshold,
            page=args.page,
            backend=args.backend,
            model=args.model,
        )
    )
    fs = p2.final_score_summary
    report = format_markdown_report(
        figure_label=Path(args.image).name,
        blind_review=rev1,
        master_grade=grade1,
        phase2_result=p2,
    )
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Saved full report to {args.out}")
    else:
        print(report)
    print(
        f"FINAL SCORE: {fs.final_score}/100 ({fs.letter_grade}) — {fs.final_verdict} "
        f"[Phase 1: {fs.phase1_score}/100 | Phase 2: {fs.phase2_score}/100]"
    )
    return 0 if fs.final_verdict == "READY_FOR_PAPER" else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="figure-review",
        description="Gated two-phase scientific figure peer review using Anthropic Claude models.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("backends", help="Check available Python LLM backends and Anthropic models.")

    p_pix = sub.add_parser("inspect", help="Deterministic 0-token pixel & metadata pre-audit.")
    p_pix.add_argument("image", help="Path to PNG/JPG/WebP/PDF figure.")
    p_pix.add_argument("--page", type=int, default=1)

    p_blind = sub.add_parser("blind", help="Phase 1 / Reviewer 1 only (blind visual extraction).")
    p_blind.add_argument("image")
    p_blind.add_argument("--domain", default="astrophysics")
    p_blind.add_argument("--caption", default=None)
    p_blind.add_argument("--caption-file", default=None)
    p_blind.add_argument("--backend", default="auto")
    p_blind.add_argument("--speed", choices=["fast", "deep"], default="fast")
    p_blind.add_argument("--detail", choices=["auto", "compact", "full"], default="auto")
    p_blind.add_argument("--no-cache", action="store_true", help="Bypass SHA-256 blind review cache.")
    p_blind.add_argument("--no-fail-fast", action="store_true", help="Disable 0-token pixel pre-gate.")
    p_blind.add_argument("--fail-fast-filename", action="store_true", help="Also fail-fast on filename jargon.")
    p_blind.add_argument("--model", default=None)
    p_blind.add_argument("--page", type=int, default=1)
    p_blind.add_argument("--json", action="store_true", dest="as_json")

    p_gfast = sub.add_parser(
        "grade-fast",
        help="Zero-token deterministic Phase 1 Master Gate (<0.05s) on an existing blind review.",
    )
    p_gfast.add_argument("review_id", help="review_id or JSON path from `blind`.")
    p_gfast.add_argument(
        "--match",
        choices=["PENDING", "MATCH", "PARTIAL_MISMATCH", "MISMATCH"],
        default="PENDING",
        help="Master attestation of Reviewer 1's physical conclusion.",
    )
    p_gfast.add_argument("--notes", default=None, help="Optional Master conclusion comparison notes.")
    p_gfast.add_argument("--notation", default=None, help="Path to Table_of_Notation.md or .tex.")
    p_gfast.add_argument("--style-profile", default="esteves")
    p_gfast.add_argument("--threshold", type=int, default=80)
    p_gfast.add_argument("--json", action="store_true", dest="as_json")

    p_batch = sub.add_parser(
        "batch",
        help="Run parallel Phase 1 blind reviews across multiple figures (default concurrency 5).",
    )
    p_batch.add_argument("images", nargs="+", help="Figure file paths.")
    p_batch.add_argument("--domain", default="astrophysics")
    p_batch.add_argument("--backend", default="auto")
    p_batch.add_argument("--speed", choices=["fast", "deep"], default="fast")
    p_batch.add_argument("--detail", choices=["auto", "compact", "full"], default="auto")
    p_batch.add_argument("--no-cache", action="store_true")
    p_batch.add_argument("--concurrency", type=int, default=5)
    p_batch.add_argument("--no-fail-fast", action="store_true")
    p_batch.add_argument("--model", default=None)
    p_batch.add_argument("--json", action="store_true", dest="as_json")

    p_p1 = sub.add_parser("phase1", help="Phase 1 (Blind Loop): Reviewer 1 + Master Gate (0-100).")
    p_p1.add_argument("image")
    p_p1.add_argument("--truth", required=True, help="Intended physical conclusion.")
    p_p1.add_argument("--caption", default=None)
    p_p1.add_argument("--caption-file", default=None)
    p_p1.add_argument("--notation", default=None, help="Path to Table_of_Notation.md or .tex.")
    p_p1.add_argument("--style-profile", default="esteves", help="Color style profile ('esteves' or 'generic').")
    p_p1.add_argument("--color-prefs", default=None, help="Extra custom color/colormap preference instructions.")
    p_p1.add_argument("--pixels-only", action="store_true", help="Hide caption from Reviewer 1.")
    p_p1.add_argument("--no-fail-fast", action="store_true", help="Disable 0-token pixel pre-gate.")
    p_p1.add_argument("--threshold", type=int, default=80)
    p_p1.add_argument("--domain", default="astrophysics")
    p_p1.add_argument("--backend", default="auto")
    p_p1.add_argument("--speed", choices=["fast", "deep"], default="fast")
    p_p1.add_argument("--detail", choices=["auto", "compact", "full"], default="auto")
    p_p1.add_argument("--no-cache", action="store_true")
    p_p1.add_argument("--model", default=None)
    p_p1.add_argument("--page", type=int, default=1)
    p_p1.add_argument("--out", default=None)
    p_p1.add_argument("--json", action="store_true", dest="as_json")

    p_p2 = sub.add_parser(
        "phase2",
        help="Phase 2 (Context Loop): Reviewer 2 notation hygiene & narrative check (gated on Phase 1).",
    )
    p_p2.add_argument("image")
    p_p2.add_argument("--phase1-id", required=True, help="Passing phase1_grade_id from Phase 1.")
    p_p2.add_argument("--tex", required=True, help="Path to .tex manuscript.")
    p_p2.add_argument("--caption", default=None)
    p_p2.add_argument("--caption-file", default=None)
    p_p2.add_argument("--notation", default=None)
    p_p2.add_argument("--fig-label", default=None, help="LaTeX \\label{fig:...} to locate.")
    p_p2.add_argument("--section", default=None, help="Optional section title/number hint.")
    p_p2.add_argument("--threshold", type=int, default=80)
    p_p2.add_argument("--force-override", action="store_true")
    p_p2.add_argument("--domain", default="astrophysics")
    p_p2.add_argument("--backend", default="auto")
    p_p2.add_argument("--speed", choices=["fast", "deep"], default="fast")
    p_p2.add_argument("--model", default=None)
    p_p2.add_argument("--page", type=int, default=1)
    p_p2.add_argument("--json", action="store_true", dest="as_json")

    p_e2e = sub.add_parser(
        "review",
        help="Run Phase 1 -> (if passed & --tex given) Phase 2 end-to-end with Final Score (0-100).",
    )
    p_e2e.add_argument("image")
    p_e2e.add_argument("--truth", required=True)
    p_e2e.add_argument("--caption", default=None)
    p_e2e.add_argument("--caption-file", default=None)
    p_e2e.add_argument("--notation", default=None)
    p_e2e.add_argument("--style-profile", default="esteves")
    p_e2e.add_argument("--color-prefs", default=None)
    p_e2e.add_argument("--tex", default=None, help="Optional .tex manuscript for Phase 2.")
    p_e2e.add_argument("--fig-label", default=None)
    p_e2e.add_argument("--section", default=None)
    p_e2e.add_argument("--pixels-only", action="store_true")
    p_e2e.add_argument("--test-with-caption", action="store_true")
    p_e2e.add_argument("--no-fail-fast", action="store_true")
    p_e2e.add_argument("--threshold", type=int, default=80)
    p_e2e.add_argument("--domain", default="astrophysics")
    p_e2e.add_argument("--backend", default="auto")
    p_e2e.add_argument("--speed", choices=["fast", "deep"], default="fast")
    p_e2e.add_argument("--detail", choices=["auto", "compact", "full"], default="auto")
    p_e2e.add_argument("--no-cache", action="store_true")
    p_e2e.add_argument("--model", default=None)
    p_e2e.add_argument("--out", default=None)
    p_e2e.add_argument("--json", action="store_true", dest="as_json")

    p_loop = sub.add_parser(
        "loop",
        help="Interactive Master loop: repeat Phase 1 after manual tweaks until pass, then run Phase 2.",
    )
    p_loop.add_argument("image")
    p_loop.add_argument("--truth", required=True)
    p_loop.add_argument("--tex", default=None)
    p_loop.add_argument("--caption", default=None)
    p_loop.add_argument("--caption-file", default=None)
    p_loop.add_argument("--notation", default=None)
    p_loop.add_argument("--style-profile", default="esteves")
    p_loop.add_argument("--color-prefs", default=None)
    p_loop.add_argument("--fig-label", default=None)
    p_loop.add_argument("--section", default=None)
    p_loop.add_argument("--pixels-only", action="store_true")
    p_loop.add_argument("--threshold", type=int, default=80)
    p_loop.add_argument("--max-rounds", type=int, default=5)
    p_loop.add_argument("--domain", default="astrophysics")
    p_loop.add_argument("--backend", default="auto")
    p_loop.add_argument("--speed", choices=["fast", "deep"], default="fast")
    p_loop.add_argument("--model", default=None)
    p_loop.add_argument("--page", type=int, default=1)
    p_loop.add_argument("--out", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "backends":
        print(json.dumps(detect_backends(), indent=2))
        return 0

    if args.cmd == "inspect":
        print(inspect_figure_pixels(args.image, page=args.page).model_dump_json(indent=2))
        return 0

    if args.cmd == "blind":
        cap = _read_caption(args.caption, args.caption_file)
        rev = asyncio.run(
            run_blind_review(
                args.image,
                domain=args.domain,
                caption=cap,
                page=args.page,
                backend=args.backend,
                speed=args.speed,
                detail=args.detail,
                use_cache=not args.no_cache,
                fail_fast=not args.no_fail_fast,
                fail_fast_on_filename_jargon=args.fail_fast_filename,
                model=args.model,
            )
        )
        if args.as_json:
            print(rev.model_dump_json(indent=2))
        else:
            print(format_markdown_report(figure_label=Path(args.image).name, blind_review=rev))
        return 2 if rev.fast_failed else 0

    if args.cmd == "grade-fast":
        grade = run_deterministic_grade(
            args.review_id,
            master_conclusion_match=args.match,
            conclusion_notes=args.notes,
            notation_table_path=args.notation,
            style_profile=args.style_profile,
            pass_threshold=args.threshold,
        )
        if args.as_json:
            print(grade.model_dump_json(indent=2))
        else:
            print(
                f"Deterministic Phase 1 Gate: {grade.verdict} | "
                f"Score: {grade.phase1_score}/100 ({grade.phase1_letter_grade}) | "
                f"Passed: {grade.phase1_passed} (id={grade.phase1_grade_id}, "
                f"{grade.duration_seconds:.3f}s)"
            )
            for reason in grade.gate_block_reasons:
                print(f"  - BLOCK: {reason}")
            for item in grade.chat_master_checklist:
                print(f"  - CHECK: {item}")
        return 0 if grade.phase1_passed else 2

    if args.cmd == "batch":
        res = asyncio.run(
            run_batch_blind_review(
                args.images,
                domain=args.domain,
                backend=args.backend,
                speed=args.speed,
                detail=args.detail,
                use_cache=not args.no_cache,
                fail_fast=not args.no_fail_fast,
                model=args.model,
                max_concurrency=args.concurrency,
            )
        )
        print(json.dumps(res if args.as_json else res["summary"], indent=2))
        return 0

    if args.cmd == "phase1":
        cap = _read_caption(args.caption, args.caption_file)
        res = asyncio.run(
            run_end_to_end_review(
                args.image,
                intended_conclusion=args.truth,
                caption=cap,
                notation_table_path=args.notation,
                style_profile=args.style_profile,
                color_preferences=args.color_prefs,
                domain=args.domain,
                include_caption_in_phase1=not args.pixels_only,
                pass_threshold=args.threshold,
                page=args.page,
                backend=args.backend,
                speed=args.speed,
                detail=args.detail,
                use_cache=not args.no_cache,
                fail_fast=not args.no_fail_fast,
                blind_model=args.model,
                master_model=args.model,
                save_report_path=args.out,
            )
        )
        print(json.dumps(res, indent=2) if args.as_json else res["report_markdown"])
        return 0 if res["phase1_passed"] else 2

    if args.cmd == "phase2":
        cap = _read_caption(args.caption, args.caption_file)
        try:
            p2 = asyncio.run(
                run_phase2_context_review(
                    args.image,
                    phase1_grade=args.phase1_id,
                    tex_path=args.tex,
                    caption=cap,
                    notation_table_path=args.notation,
                    figure_label=args.fig_label,
                    section_hint=args.section,
                    domain=args.domain,
                    pass_threshold=args.threshold,
                    force_override=args.force_override,
                    page=args.page,
                    backend=args.backend,
                    model=args.model,
                )
            )
        except Phase1GateBlockedError as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 2
        if args.as_json:
            print(p2.model_dump_json(indent=2))
        else:
            fs = p2.final_score_summary
            print(
                f"Phase 2 Verdict: {p2.phase2_verdict} ({p2.phase2_score}/100, {p2.phase2_letter_grade})\n"
                f"FINAL SCORE: {fs.final_score}/100 ({fs.letter_grade}) — {fs.final_verdict}\n\n"
                f"Summary: {p2.reviewer2_summary}"
            )
        return 0 if p2.phase2_passed else 3

    if args.cmd == "review":
        cap = _read_caption(args.caption, args.caption_file)
        res = asyncio.run(
            run_end_to_end_review(
                args.image,
                intended_conclusion=args.truth,
                caption=cap,
                notation_table_path=args.notation,
                style_profile=args.style_profile,
                color_preferences=args.color_prefs,
                tex_path=args.tex,
                figure_label=args.fig_label,
                section_hint=args.section,
                domain=args.domain,
                include_caption_in_phase1=not args.pixels_only,
                test_with_caption=args.test_with_caption,
                pass_threshold=args.threshold,
                backend=args.backend,
                speed=args.speed,
                detail=args.detail,
                use_cache=not args.no_cache,
                fail_fast=not args.no_fail_fast,
                blind_model=args.model,
                master_model=args.model,
                reviewer2_model=args.model,
                save_report_path=args.out,
            )
        )
        print(json.dumps(res, indent=2) if args.as_json else res["report_markdown"])
        return 0 if res["final_verdict"] == "READY_FOR_PAPER" else 2

    if args.cmd == "loop":
        return _run_interactive_loop(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())

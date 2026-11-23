---
name: figure-reviewer
description: Blind, two-phase peer review of a scientific figure using Anthropic Claude models (via the `figure_reviewer_mcp` server). Phase 1 lets a reader who sees only the pixels say what the figure shows, then gates on a 0-100 score. Phase 2 checks notation and narrative against the .tex manuscript. Use after plotting any figure that will go into a paper, talk or note.
---

# Blind figure review (Claude models)

Never let the author of a figure certify it. The author knows what the figure should say and
unconsciously fills in what is missing. This skill gives the figure to a reader who has only
the pixels, then compares that reader's answer with the intended message.

**The rule:** do not start Phase 2 until Phase 1 passes (`phase1_passed == True`). A figure must
survive on its own before it joins the manuscript.

## Tools

The MCP server `figure_reviewer_mcp` provides these tools (names appear as
`mcp__figure_reviewer_mcp__<tool>`; load them with ToolSearch if they are deferred).

| Tool | Use |
|---|---|
| `check_backends` | Which Anthropic backend and model will run |
| `inspect_figure_pixels` | Zero-token pixel and color forensics |
| `blind_review_figure` | Phase 1, Reviewer 1 (blind) |
| `blind_review_batch` | Phase 1 for several figures in parallel |
| `grade_blind_deterministic` | Master gate with zero model tokens, using your conclusion match |
| `audit_figure_labels_vector` | Optional label and jargon audit (needs the astro-math-reviewer companion) |
| `phase2_context_review` | Phase 2, Reviewer 2 (notation and narrative) |
| `review_figure_end_to_end` | Phase 1, then Phase 2 only if Phase 1 passes |

Backends: `claude_cli` (uses your Claude Code login) or `anthropic_api` (`ANTHROPIC_API_KEY`).
Default models: `claude-sonnet-5-5` for `speed="fast"`, `claude-opus-5-5` for `speed="deep"`.
Set `FIGURE_REVIEW_MODEL` or `FIGURE_REVIEW_DEEP_MODEL` to change them.

## Phase 1: the blind loop (Reviewer 1)

1. **Pre-gate, zero tokens.** `blind_review_figure` first checks resolution, rainbow or `jet`
   colormaps and saturation clipping. If it fails, fix the plotting script and rerun.
2. **Blind read.** Call `blind_review_figure(image_path, caption=..., domain="astrophysics")`.
   Reviewer 1 sees only the metadata-stripped image and the optional caption. It never sees the
   manuscript, the project, or the filename. It returns a one-sentence conclusion, decodes the
   axes, colors and markers, and defines every variable from the figure alone.
3. **Master gate.** You hold the intended message. Compare it with Reviewer 1's
   `one_sentence_conclusion`, then call
   `grade_blind_deterministic(blind_review=review_id, master_conclusion_match="MATCH" | "PARTIAL_MISMATCH" | "MISMATCH")`.
   This also runs the deterministic checks: code jargon, banned terms, colorbar midpoint
   arithmetic, saturation clipping and rainbow colormaps.
4. **Score (0 to 100).** Conclusion accuracy (0 to 40), variable clarity (0 to 35), visual design
   and colorbars (0 to 25). To pass, the conclusion must MATCH, the score must reach 80, no
   deterministic check may fire, and no critical visual defect may remain.
5. **If it fails,** change the figure or the caption, then repeat Phase 1. Fix the figure first.
   Do not argue with the reader.

If you cannot write the intended conclusion in one sentence, the figure is not ready for review.

## Phase 2: the context loop (Reviewer 2)

Unlocked only by a passing `phase1_grade_id` (otherwise the tool returns `BLOCKED_BY_PHASE1`).
Call `phase2_context_review(image_path, phase1_grade_id=..., tex_path="main.tex", caption=...,
notation_table_path="notation.md", figure_label="fig:...")`.

- **Notation hygiene (0 to 50).** Symbols, subscripts and units match the `.tex` macros and the
  notation table. Orphaned terms and symbol collisions are flagged.
- **Narrative alignment (0 to 50).** The text walks the reader through every panel and curve.
  Unreferenced panels and unsupported claims are flagged.

Final score is the mean of the two phases, capped at 74 if either fails. Verdicts:
`READY_FOR_PAPER`, `REVISE_TEX`, `REVISE_CAPTION`, `REDO_FIGURE`.

## What the review can and cannot tell you

- The reviewer is a language model. It stands in for a careful, unprimed reader and does not
  replace one. Show important figures to a human colleague too.
- It tests whether the figure communicates. It does not test whether the physics is right.
- Its answer is only as useful as your stated intent.

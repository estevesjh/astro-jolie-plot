"""Deterministic LaTeX manuscript context extractor for Phase 2 (Context Loop)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedTexContext:
    """Focused LaTeX manuscript slice around a figure for Reviewer 2."""

    tex_path: str
    macros: list[str] = field(default_factory=list)
    figure_label: str | None = None
    figure_caption_in_tex: str | None = None
    figure_block: str = ""
    referencing_paragraphs: list[str] = field(default_factory=list)
    section_excerpt: str = ""


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*$", "", tex, flags=re.MULTILINE)


def _extract_braced(text: str, start_idx: int) -> str:
    """Extract balanced {...} starting at text[start_idx] == '{'."""
    if start_idx >= len(text) or text[start_idx] != "{":
        return ""
    depth = 0
    out: list[str] = []
    for ch in text[start_idx:]:
        if ch == "{":
            if depth > 0:
                out.append(ch)
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def extract_tex_context(
    tex_path: str | Path,
    *,
    image_path: str | Path | None = None,
    figure_label: str | None = None,
    section_hint: str | None = None,
    max_chars: int = 6500,
) -> ExtractedTexContext:
    """Extract relevant preamble macros, figure environment, referencing paragraphs, and local section."""
    path = Path(tex_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LaTeX manuscript not found: {path}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    clean = _strip_comments(raw)

    ctx = ExtractedTexContext(tex_path=str(path), figure_label=figure_label)

    # 1. Extract all preamble macros first; we'll filter to those relevant to the figure/section below
    all_macros: list[tuple[str, str]] = []
    for m in re.finditer(
        r"\\(?:re)?newcommand\*?\{\\(\w+)\}(?:\[\d+\])?\{([^\n]{1,160})\}", clean
    ):
        all_macros.append((f"\\{m.group(1)}", m.group(2).strip()))

    # 2. Locate matching \begin{figure[*]} ... \end{figure[*]}
    image_stem = Path(image_path).stem if image_path else None
    matched_span: tuple[int, int] | None = None

    for m in re.finditer(
        r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", clean, flags=re.DOTALL
    ):
        block = m.group(0)
        lbl_m = re.search(r"\\label\{([^}]+)\}", block)
        lbl = lbl_m.group(1).strip() if lbl_m else None

        is_match = False
        if figure_label and lbl and lbl == figure_label:
            is_match = True
        elif image_stem and image_stem in block:
            is_match = True
        elif figure_label and figure_label in block:
            is_match = True

        if is_match:
            ctx.figure_block = block.strip()
            ctx.figure_label = lbl or figure_label
            cap_m = re.search(r"\\caption\s*\{", block)
            if cap_m:
                ctx.figure_caption_in_tex = _extract_braced(block, cap_m.end() - 1).strip()
            matched_span = (m.start(), m.end())
            break

    # 3. Extract all paragraphs citing \ref{figure_label}
    paragraphs = re.split(r"\n\s*\n", clean)
    if ctx.figure_label:
        ref_pat = re.compile(
            r"\\(?:auto|c|C|eq|fig)?ref\{[^}]*" + re.escape(ctx.figure_label) + r"[^}]*\}"
        )
        for para in paragraphs:
            if ref_pat.search(para) and "\\begin{figure" not in para:
                ctx.referencing_paragraphs.append(para.strip())

    # 4. Extract enclosing section (or section_hint match, or local neighborhood)
    sec_matches = list(
        re.finditer(r"\\(?:sub)?section\*?\{([^}]+)\}", clean)
    )

    chosen_start = 0
    chosen_end = min(len(clean), max_chars)

    if section_hint and sec_matches:
        hint_lo = section_hint.lower()
        for idx, sm in enumerate(sec_matches):
            if hint_lo in sm.group(1).lower() or hint_lo in sm.group(0).lower():
                chosen_start = sm.start()
                chosen_end = (
                    sec_matches[idx + 1].start()
                    if idx + 1 < len(sec_matches)
                    else len(clean)
                )
                break
    elif matched_span is not None and sec_matches:
        fig_pos = matched_span[0]
        # Find enclosing section header before the figure
        prev_idx = 0
        for idx, sm in enumerate(sec_matches):
            if sm.start() <= fig_pos:
                prev_idx = idx
            else:
                break
        chosen_start = sec_matches[prev_idx].start()
        chosen_end = (
            sec_matches[prev_idx + 1].start()
            if prev_idx + 1 < len(sec_matches)
            else len(clean)
        )
    elif ctx.referencing_paragraphs:
        first_para = ctx.referencing_paragraphs[0]
        pos = clean.find(first_para)
        if pos != -1:
            chosen_start = max(0, pos - 2500)
            chosen_end = min(len(clean), pos + max_chars - 2500)

    excerpt = clean[chosen_start:chosen_end].strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "\n... [truncated]"
    ctx.section_excerpt = excerpt

    # Keep macros that actually appear in the figure block, referencing paragraphs, or local section
    local_text = "\n".join([ctx.figure_block, *ctx.referencing_paragraphs, ctx.section_excerpt])
    used_macros = [
        f"{name} = {defn}"
        for name, defn in all_macros
        if re.search(re.escape(name) + r"(?![A-Za-z])", local_text)
    ]
    ctx.macros = used_macros[:60] if used_macros else [f"{n} = {d}" for n, d in all_macros[:40]]
    return ctx

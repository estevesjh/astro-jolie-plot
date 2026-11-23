"""Deterministic Table of Notation parser and code-jargon scanner (zero LLM tokens)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import BlindReviewPayload, DeterministicAuditHit


@dataclass
class NotationSpec:
    """Parsed canonical symbols and banned/code-jargon terms from a Table of Notation."""

    source_path: str
    allowed_symbols: list[str] = field(default_factory=list)
    symbol_definitions: dict[str, str] = field(default_factory=dict)
    banned_terms: list[str] = field(default_factory=list)
    banned_replacements: dict[str, str] = field(default_factory=dict)
    raw_excerpt: str = ""


# Standard physics/statistics uppercase acronyms that are legitimate in `X = value` labels
ALLOWED_UPPERCASE_STAT_TOKENS = {
    "SNR",
    "DOF",
    "FWHM",
    "RMS",
    "STD",
    "AIC",
    "BIC",
    "PDF",
    "CDF",
    "MAP",
    "MCMC",
    "NFW",
    "HOD",
    "LSS",
    "BCG",
    "LOS",
    "DES",
    "SPT",
    "ACT",
    "LSST",
    "CMB",
    "BAO",
}

# Domain-general code-jargon / internal-shorthand regexes that frequently leak onto plots
CODE_JARGON_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b[A-Z]{2,}_[A-Z0-9_]{2,}\b"),
        "ALL_CAPS_SNAKE_CASE identifier looks like an internal code flag or column name.",
    ),
    (
        re.compile(r"(?:^|[\s(])/[A-Z]{3,}\b"),
        "Slash-prefixed uppercase token (e.g. /NEGATIVE) looks like an IDL/CLI keyword flag.",
    ),
    (
        re.compile(r"\b([A-Z]{2,6})\s*=\s*\d+(?:\.\d+)?\b"),
        "Uppercase flag=number assignment (e.g. CPF=2) looks like a solver option, not physics.",
    ),
    (
        re.compile(r"\bSec\.?\s*\d+(?:\.\d+)*\b", re.IGNORECASE),
        "Internal manuscript section reference (e.g. Sec. 3) makes a figure non-self-contained.",
    ),
    (
        re.compile(r"\b(?:var_[A-Za-z0-9]+|rk\s*=\s*\d+|R0_ladder|B[1-9](?:[-–]B?[1-9])?)\b"),
        "Internal batch, run, or variant code (e.g. B1, rk=20, var_H) instead of physical cuts.",
    ),
    (
        re.compile(r"\b\d{1,2}p\b"),
        "Parameter-count shorthand (e.g. 25p, 31p) lacks physical meaning without spelling out the model.",
    ),
    (
        re.compile(r"\b(?:Univ\.\s*[A-Z]|det\.)", re.IGNORECASE),
        "Cryptic internal abbreviation (e.g. Univ. A, det.) should be replaced with physical wording.",
    ),
    (
        re.compile(r"\.(?:npz|npy|fits|hdf5|h5|csv|parquet|py)\b", re.IGNORECASE),
        "Data or script filename leaked into figure label.",
    ),
]


def _clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("*").strip("`").strip())


def _extract_quoted_or_backticked_terms(line: str) -> list[str]:
    terms: list[str] = []
    for m in re.findall(r"`([^`]{2,40})`", line):
        terms.append(m.strip())
    for m in re.findall(r'"([^"]{2,40})"', line):
        terms.append(m.strip())
    return terms


def parse_notation_table(path: str | Path) -> NotationSpec:
    """Parse a Markdown Table of Notation or LaTeX macro preamble into canonical + banned sets."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Notation table not found: {file_path}")

    text = file_path.read_text(encoding="utf-8", errors="replace")
    spec = NotationSpec(source_path=str(file_path))

    if file_path.suffix.lower() == ".tex":
        for m in re.finditer(r"\\(?:re)?newcommand\*?\{\\(\w+)\}(?:\[\d+\])?\{([^}]+)\}", text):
            macro = f"\\{m.group(1)}"
            spec.allowed_symbols.append(macro)
            spec.symbol_definitions[macro] = m.group(2).strip()
        spec.raw_excerpt = text[:6000]
        return spec

    # Markdown table parsing
    current_table_kind: str | None = None  # "symbol" or "banned"
    for line in text.splitlines():
        stripped = line.strip()

        # Check explicit "never write ..." / "Do not use ..." directives in prose/callouts
        lower = stripped.lower()
        if any(
            phrase in lower
            for phrase in ("never write", "do not use", "banned", "avoid inventing", "forbid")
        ):
            for term in _extract_quoted_or_backticked_terms(stripped):
                if term and term not in spec.banned_terms:
                    spec.banned_terms.append(term)

        if not stripped.startswith("|"):
            current_table_kind = None
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue

        # Skip markdown separator rows like | :--- | :--- |
        if all(re.fullmatch(r"[:\-\s]+", c) for c in cells):
            continue

        header0 = cells[0].lower()
        header_all = " ".join(c.lower() for c in cells)

        if any(k in header0 for k in ("code term", "do not use", "banned", "avoid", "deprecated", "jargon")):
            current_table_kind = "banned"
            continue
        if any(k in header0 for k in ("symbol", "notation", "macro", "quantity")):
            current_table_kind = "symbol"
            continue
        if current_table_kind is None and "meaning" in header_all:
            current_table_kind = "symbol"
            continue

        if current_table_kind == "banned":
            raw_terms = cells[0]
            replacement = _clean_cell(cells[1]) if len(cells) > 1 else ""
            # Split comma/semicolon-separated code terms in the first column
            parts = re.split(r"[,;]", raw_terms)
            for part in parts:
                cleaned = _clean_cell(part)
                # Strip markdown links/dollar signs only lightly so symbols stay searchable
                if cleaned and len(cleaned) >= 2:
                    if cleaned not in spec.banned_terms:
                        spec.banned_terms.append(cleaned)
                    if replacement:
                        spec.banned_replacements[cleaned] = replacement
        elif current_table_kind == "symbol":
            sym = _clean_cell(cells[0])
            meaning = _clean_cell(cells[-1]) if len(cells) >= 2 else ""
            if sym:
                spec.allowed_symbols.append(sym)
                spec.symbol_definitions[sym] = meaning

    # Keep a bounded raw excerpt for the Master LLM prompt (without blowing context)
    spec.raw_excerpt = text[:12000]
    return spec


def _term_matches(needle: str, haystack: str) -> bool:
    """Match a banned term cleanly against extracted figure text without silly false positives."""
    needle_clean = needle.strip().strip("$").strip("`")
    if len(needle_clean) < 2:
        return False
    escaped = re.escape(needle_clean)
    # If alphanumeric at boundaries, require word boundaries
    left = r"\b" if needle_clean[0].isalnum() else r"(?:^|[\s(\[/,:;])"
    right = r"\b" if needle_clean[-1].isalnum() else r"(?:$|[\s)\],:;])"
    flags = 0 if (needle_clean.isupper() and len(needle_clean) <= 3) else re.IGNORECASE
    return bool(re.search(f"{left}{escaped}{right}", haystack, flags=flags))


def scan_blind_review_deterministically(
    blind_review: BlindReviewPayload,
    *,
    notation_spec: NotationSpec | None = None,
    extra_banned_terms: list[str] | None = None,
) -> list[DeterministicAuditHit]:
    """Scan all text read off the figure by the blind reviewer for banned terms & code jargon."""
    corpus: list[tuple[str, str]] = []
    for p in blind_review.panel_encodings:
        corpus.extend(
            [
                (f"{p.panel_id} title", p.panel_title),
                (f"{p.panel_id} x-axis", p.x_axis),
                (f"{p.panel_id} y-axis", p.y_axis),
                (f"{p.panel_id} colors/colorbar", p.colors_and_colorbar),
                (f"{p.panel_id} markers/lines", p.markers_and_linestyles),
                (f"{p.panel_id} annotations/legends", p.annotations_and_legends),
            ]
        )
    for item in blind_review.unclear_terms:
        corpus.append((item.location, item.term))

    hits: list[DeterministicAuditHit] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(term: str, source: str, location: str, detail: str) -> None:
        key = (term.lower(), source, location.lower())
        if key not in seen:
            seen.add(key)
            hits.append(
                DeterministicAuditHit(
                    term=term,
                    source=source,  # type: ignore[arg-type]
                    location=location,
                    detail=detail,
                )
            )

    # 1. Notation-table banned terms
    if notation_spec is not None:
        for banned in notation_spec.banned_terms:
            # Skip complex range expressions like "B1..B7" or "z0/z1/z2" -> split on "/"
            sub_terms = [t.strip() for t in re.split(r"\s*/\s*", banned) if len(t.strip()) >= 2]
            if not sub_terms:
                sub_terms = [banned]
            for sub in sub_terms:
                if ".." in sub or "–" in sub:
                    continue
                for loc, text in corpus:
                    if _term_matches(sub, text):
                        repl = notation_spec.banned_replacements.get(banned, "")
                        detail = (
                            f"Banned by Table of Notation ({Path(notation_spec.source_path).name})."
                            + (f" Use instead: {repl}" if repl else "")
                        )
                        _add(sub, "notation_table_banned", loc, detail)

    # 2. Explicit user-supplied banned terms
    for banned in extra_banned_terms or []:
        for loc, text in corpus:
            if _term_matches(banned, text):
                _add(banned, "explicit_banned", loc, "Explicitly listed in banned_terms.")

    # 3. Domain-general code-jargon heuristics
    for loc, text in corpus:
        for pattern, reason in CODE_JARGON_PATTERNS:
            for match in pattern.finditer(text):
                if match.lastindex and match.group(1) in ALLOWED_UPPERCASE_STAT_TOKENS:
                    continue
                token = match.group(0).strip()
                _add(token, "code_jargon_heuristic", loc, reason)

    return hits


# Filename-level opaque batch / solver / scratch tokens (e.g. B1B2, neg, cpf2, nohost, varH, rk20)
FILENAME_JARGON_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^(?:B[1-9]){1,4}$"),
        "Internal batch code (e.g. B1, B1B2) — replace with physical richness cut (e.g. lamge45).",
    ),
    (
        re.compile(r"^(?:neg|negative|cpf[0-9]+|nohost|var[A-Za-z0-9]|rk[0-9]+|r0|z[0-9]|det|t[0-2])$", re.IGNORECASE),
        "Internal solver/flag shorthand — replace with plain physical description.",
    ),
    (
        re.compile(r"^(?:v[0-9]+|proto|tmp|test|draft|old|new|final[0-9]*)$", re.IGNORECASE),
        "Scratch/version suffix — strip before manuscript integration.",
    ),
]


def audit_filename_jargon(
    image_path: str | Path,
    *,
    notation_spec: NotationSpec | None = None,
) -> list[str]:
    """Deterministically detect opaque batch/code tokens in a figure filename (0 LLM tokens)."""
    stem = Path(image_path).stem
    tokens = [tok for tok in re.split(r"[_\-\s]+", stem) if tok]
    flagged: list[str] = []
    banned_lower = {
        b.strip().strip("`").strip("/").lower()
        for b in (notation_spec.banned_terms if notation_spec else [])
        if len(b.strip()) >= 2
    }

    for tok in tokens:
        low = tok.lower()
        if low in banned_lower:
            if tok not in flagged:
                flagged.append(tok)
            continue
        for pat, _ in FILENAME_JARGON_PATTERNS:
            if pat.fullmatch(tok):
                if tok not in flagged:
                    flagged.append(tok)
                break
    return flagged

"""Bridge to the vectorized `astro-math-review figure-labels` command (0 LLM tokens).

Audits all visible figure labels (panel titles, axes, colorbars, legends, annotations,
`unclear_terms`, and optional caption) against:
  1. Persistent Banned Words Registry (`~/.config/astro-math-reviewer/banned_words.yaml`)
  2. Empirical 160+ paper Field Concept & Jargon Index (stats jargon, prevalence, acronyms)
  3. Dual-view 160+ paper Field-Notation Index (symbol consensus & near-collisions)

Runs in a single fast subprocess invocation (`astro-math-review figure-labels --input-json - --json`)
and degrades cleanly to a no-op if the binary is unavailable or `ASTRO_MATH_VECTOR_DISABLE=1`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .schemas import BlindReviewPayload, DeterministicAuditHit

# Optional companion package (`mcp_astro_math_reviewer`). When absent, the vector label audit
# is skipped and the other deterministic checks still run. Point to a checkout with these:
#   ASTRO_MATH_REVIEW_SRC  directory that contains the `mcp_astro_math_reviewer` package
#   ASTRO_MATH_REVIEW_BIN  path to the `astro-math-review` executable
DEFAULT_ASTRO_MATH_SRC = Path(os.environ.get("ASTRO_MATH_REVIEW_SRC", "/nonexistent-astro-math-src"))
DEFAULT_ASTRO_MATH_BIN = Path(os.environ.get("ASTRO_MATH_REVIEW_BIN", "/nonexistent-astro-math-review"))
_MEM_VECTOR_AUDIT_CACHE: dict[str, dict[str, Any]] = {}


def _try_inprocess_vector_audit(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Fast in-process path (<15 ms warm) using `mcp_astro_math_reviewer.concept_index` directly."""
    if os.environ.get("ASTRO_MATH_VECTOR_SUBPROCESS", "").strip().lower() in {"1", "true", "yes"}:
        return None
    if os.environ.get("ASTRO_MATH_REVIEW_BIN", "").strip():
        return None
    try:
        if DEFAULT_ASTRO_MATH_SRC.is_dir():
            src_s = str(DEFAULT_ASTRO_MATH_SRC)
            if src_s not in sys.path:
                sys.path.append(src_s)
        from mcp_astro_math_reviewer.concept_index import audit_figure_labels_vector

        data = audit_figure_labels_vector(
            payload.get("labels") or [],
            symbols=payload.get("symbols") or [],
            caption=str(payload.get("caption") or ""),
            project_root=payload.get("project_root"),
        )
        data["enabled"] = True
        data["vector_command"] = "astro-math-review figure-labels (in-process)"
        return data
    except Exception:
        return None


def _vector_cache_key(payload: dict[str, Any], bin_path: str) -> str:
    bw_path = Path(
        os.environ.get(
            "ASTRO_MATH_BANNED_WORDS_FILE",
            str(Path.home() / ".config" / "astro-math-reviewer" / "banned_words.yaml"),
        )
    ).expanduser()
    bw_sig = f"{int(bw_path.stat().st_mtime)}:{bw_path.stat().st_size}" if bw_path.is_file() else "none"
    blob = json.dumps({"v": 1, "bin": bin_path, "bw": bw_sig, "payload": payload}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def resolve_astro_math_bin() -> str | None:
    """Locate the `astro-math-review` vectorized CLI binary (or None if disabled/missing)."""
    if os.environ.get("ASTRO_MATH_VECTOR_DISABLE", "").strip().lower() in {"1", "true", "yes"}:
        return None
    env_bin = os.environ.get("ASTRO_MATH_REVIEW_BIN", "").strip()
    if env_bin:
        p = Path(env_bin).expanduser()
        return str(p) if p.is_file() else shutil.which(env_bin)
    if DEFAULT_ASTRO_MATH_BIN.is_file():
        return str(DEFAULT_ASTRO_MATH_BIN)
    return shutil.which("astro-math-review")


def infer_project_root(
    *,
    project_root: str | Path | None = None,
    notation_table_path: str | Path | None = None,
) -> str | None:
    """Infer repository root from explicit `project_root` or `notes/Table_of_Notation.md` path."""
    if project_root:
        p = Path(project_root).expanduser().resolve()
        return str(p) if p.is_dir() else None
    if notation_table_path:
        np_path = Path(notation_table_path).expanduser().resolve()
        for parent in (np_path.parent, np_path.parent.parent):
            if (parent / "draft" / "main.tex").is_file() or (parent / "refs").is_dir() or (parent / ".banned_words.yaml").is_file():
                return str(parent)
    return None


def extract_figure_labels_payload(
    blind_review: BlindReviewPayload,
    *,
    caption: str | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Extract all visible figure labels and symbol items from Blind Reviewer 1's output."""
    labels: list[dict[str, str]] = []
    symbols: list[dict[str, str]] = []
    seen_labels: set[tuple[str, str]] = set()

    def _add_label(loc: str, txt: str) -> None:
        clean = (txt or "").strip()
        if not clean or clean in {"(none)", "(compact triage mode)", "(captured in visible_labels_and_titles)", "N/A"}:
            return
        key = (loc.lower(), clean.lower())
        if key not in seen_labels:
            seen_labels.add(key)
            labels.append({"location": loc, "text": clean})

    for p in blind_review.panel_encodings:
        _add_label(f"{p.panel_id} title", p.panel_title)
        _add_label(f"{p.panel_id} x-axis", p.x_axis)
        _add_label(f"{p.panel_id} y-axis", p.y_axis)
        _add_label(f"{p.panel_id} colorbar", p.colors_and_colorbar)
        _add_label(f"{p.panel_id} annotations/legends", p.annotations_and_legends)

    for item in blind_review.unclear_terms:
        _add_label(item.location, item.term)
        raw_t = item.term.strip().strip("$")
        if len(raw_t) >= 2 and ("\\" in raw_t or "_" in raw_t or "^" in raw_t):
            symbols.append(
                {
                    "location": item.location,
                    "symbol": raw_t,
                    "context": f"{item.blind_guess} {item.reason}".strip(),
                }
            )

    payload: dict[str, Any] = {
        "labels": labels,
        "symbols": symbols,
        "caption": (caption or "").strip(),
    }
    if project_root:
        payload["project_root"] = str(project_root)
    return payload


def run_vector_label_audit(
    blind_review: BlindReviewPayload | None = None,
    *,
    raw_labels: list[str] | None = None,
    caption: str | None = None,
    notation_table_path: str | Path | None = None,
    project_root: str | Path | None = None,
    timeout_seconds: int = 15,
) -> tuple[dict[str, Any], list[DeterministicAuditHit]]:
    """Run the vectorized `astro-math-review figure-labels` audit (0 tokens).

    Returns `(audit_dict, deterministic_hits)`:
      - `audit_dict`: full empirical report (`papers_indexed`, `banned_words_detected`,
        `stats_jargon_detected`, `symbol_collisions`, `symbol_alignments`, `subfield_astro_needs_gloss`,
        `unfamiliar_acronyms`, `duration_ms`).
      - `deterministic_hits`: hard blocking hits (`vector_banned_word`, `vector_stats_jargon`,
        `vector_symbol_collision`) ready to merge into `MasterGradeResult.deterministic_hits`.
    """
    bin_path = resolve_astro_math_bin()
    if not bin_path:
        return {"enabled": False, "reason": "astro-math-review CLI not found or disabled."}, []

    eff_root = infer_project_root(
        project_root=project_root,
        notation_table_path=notation_table_path,
    )

    if blind_review is not None:
        payload = extract_figure_labels_payload(
            blind_review,
            caption=caption,
            project_root=eff_root,
        )
    else:
        payload = {
            "labels": [
                {"location": f"Label #{i}", "text": lbl}
                for i, lbl in enumerate(raw_labels or [], start=1)
                if lbl and lbl.strip()
            ],
            "symbols": [],
            "caption": (caption or "").strip(),
        }
        if eff_root:
            payload["project_root"] = eff_root

    if not payload["labels"] and not payload["caption"]:
        return {"enabled": True, "labels_audited": 0, "banned_words_detected": [], "stats_jargon_detected": [], "symbol_collisions": []}, []

    ckey = _vector_cache_key(payload, bin_path)
    cache_dir = Path(
        os.environ.get(
            "FIGURE_REVIEW_STORE_DIR",
            str(Path(tempfile.gettempdir()) / "mcp_figure_reviewer_sessions"),
        )
    ) / "vector_label_cache"
    cache_file = cache_dir / f"vla_{ckey}.json"

    data: dict[str, Any] | None = _MEM_VECTOR_AUDIT_CACHE.get(ckey)
    if data is None and cache_file.is_file():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            _MEM_VECTOR_AUDIT_CACHE[ckey] = data
        except Exception:
            data = None

    if data is None:
        data = _try_inprocess_vector_audit(payload)

    if data is None:
        try:
            proc = subprocess.run(
                [bin_path, "figure-labels", "--input-json", "-", "--json"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return {
                    "enabled": False,
                    "reason": f"astro-math-review exited {proc.returncode}: {(proc.stderr or '')[:160]}",
                }, []
            data = json.loads(proc.stdout)
            data["enabled"] = True
            data["vector_command"] = f"{Path(bin_path).name} figure-labels"
        except Exception as exc:
            return {"enabled": False, "reason": f"vector audit error: {exc}"}, []

    _MEM_VECTOR_AUDIT_CACHE[ckey] = data
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass

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

    # 1. Hard MAJOR/FATAL banned words (e.g. `pedestal`, `shelf`, `closure`, `unselected`, custom user bans)
    for b in data.get("banned_words_detected") or []:
        if b.get("severity") not in {"FATAL", "MAJOR"}:
            continue
        rep = f" Use instead: {b['replacement']}." if b.get("replacement") else ""
        near = b.get("nearest_field_concepts") or []
        near_note = f" Nearest field concepts: {', '.join(near[:3])}." if near else ""
        detail = (
            f"Banned word `{b['word']}` ({b['category']}, df={b.get('field_df_papers', 0)}/{b.get('papers_indexed', 0)} papers): "
            f"{b.get('reason', '')}{rep}{near_note}"
        ).strip()
        _add(str(b["matched"]), "vector_banned_word", str(b["location"]), detail)

    # 2. Non-field statistics / actuarial / stat-mech jargon (e.g. `survival function`, `Palm calculus`, `Fano factor`)
    for s in data.get("stats_jargon_detected") or []:
        near = s.get("nearest_field_concepts") or []
        near_note = f" Nearest field concepts: {', '.join(near[:3])}." if near else ""
        detail = (
            f"Non-astro stats/math jargon `{s['term']}` ({s['category']}, "
            f"df={s.get('field_df_papers', 0)}/{s.get('papers_indexed', 0)} papers). "
            f"Use instead: {s.get('astro_replacement', '')}.{near_note}"
        ).strip()
        _add(str(s["matched"]), "vector_stats_jargon", str(s["location"]), detail)

    # 3. High-confidence field symbol near-collisions
    for c in data.get("symbol_collisions") or []:
        warn = (c.get("collision_warnings") or [""])[0]
        if warn:
            _add(
                str(c["symbol"]),
                "vector_symbol_collision",
                str(c["location"]),
                f"Vectorized field-notation near-collision: {warn}",
            )

    return data, hits


def format_vector_audit_for_prompt(audit: dict[str, Any] | None) -> str:
    """Format a compact summary of the vectorized 160-paper label audit for Master / Reviewer 2."""
    if not audit or not audit.get("enabled"):
        return ""
    lines: list[str] = [
        f"Vectorized Field Label & Jargon Audit ({audit.get('papers_indexed', 0)} papers, "
        f"{audit.get('concepts_indexed', 0)} concepts, {audit.get('symbol_records_indexed', 0)} symbol records):"
    ]
    for b in (audit.get("banned_words_detected") or [])[:8]:
        lines.append(
            f"- [BANNED WORD/{b.get('severity')}] ({b['location']}): `{b['matched']}` "
            f"(df={b.get('field_df_papers', 0)}/{audit.get('papers_indexed', 0)}) -> USE: {b.get('replacement', '')}"
        )
    for s in (audit.get("stats_jargon_detected") or [])[:8]:
        lines.append(
            f"- [STATS JARGON] ({s['location']}): `{s['matched']}` "
            f"(df={s.get('field_df_papers', 0)}/{audit.get('papers_indexed', 0)}) -> USE: {s.get('astro_replacement', '')}"
        )
    for c in (audit.get("symbol_collisions") or [])[:6]:
        lines.append(
            f"- [SYMBOL COLLISION] ({c['location']}): `{c['symbol']}` — {(c.get('collision_warnings') or [''])[0][:140]}"
        )
    for a in (audit.get("symbol_alignments") or [])[:8]:
        if a.get("alignment_status") != "NEAR_COLLISION":
            cons = ", ".join(a.get("consensus_field_symbols") or [])
            lines.append(
                f"- [SYMBOL {a['alignment_status']}] ({a['location']}): `{a['symbol']}` (field consensus: {cons or 'none'})"
            )
    if audit.get("unfamiliar_acronyms"):
        acrs = ", ".join(f"{x['acronym']} (df={x['field_df_papers']})" for x in audit["unfamiliar_acronyms"][:6])
        lines.append(f"- [RARE ACRONYMS ON FIGURE]: {acrs}")
    return "\n".join(lines) if len(lines) > 1 else ""


def run_vector_paper_selection(
    topic: str,
    methodology: str = "",
    *,
    pin_papers: list[str] | None = None,
    project_root: str | Path | None = None,
    notation_table_path: str | Path | None = None,
    top_k: int = 3,
    timeout_seconds: int = 12,
) -> dict[str, Any]:
    """Vectorized (<5 ms warm, 0 tokens) anchor-paper retrieval via `mcp_astro_math_reviewer.paper_index`."""
    eff_root = infer_project_root(project_root=project_root, notation_table_path=notation_table_path)

    # 1. Fast in-process path (<5 ms warm)
    if (
        os.environ.get("ASTRO_MATH_VECTOR_DISABLE", "").strip().lower() not in {"1", "true", "yes"}
        and os.environ.get("ASTRO_MATH_VECTOR_SUBPROCESS", "").strip().lower() not in {"1", "true", "yes"}
        and not os.environ.get("ASTRO_MATH_REVIEW_BIN", "").strip()
    ):
        try:
            if DEFAULT_ASTRO_MATH_SRC.is_dir():
                src_s = str(DEFAULT_ASTRO_MATH_SRC)
                if src_s not in sys.path:
                    sys.path.append(src_s)
            from mcp_astro_math_reviewer.paper_index import (
                build_or_load_paper_index,
                select_anchor_papers,
            )
            from mcp_astro_math_reviewer.vector_index import DEFAULT_NOTES_PAPERS_DIR

            eff_dirs: list[str | Path] = []
            if eff_root:
                from mcp_astro_math_reviewer.ledger import discover_project_files

                rf = discover_project_files(eff_root).get("refs_dir")
                if rf is not None:
                    eff_dirs.append(rf)
            if DEFAULT_NOTES_PAPERS_DIR.is_dir():
                eff_dirs.append(DEFAULT_NOTES_PAPERS_DIR)

            p_idx = build_or_load_paper_index(eff_dirs or None)
            out = select_anchor_papers(
                topic,
                methodology,
                pin_papers=pin_papers,
                index=p_idx,
                top_k=top_k,
            )
            out["enabled"] = True
            return out
        except Exception:
            pass

    # 2. Subprocess CLI fallback (`astro-math-review papers --json`)
    bin_path = resolve_astro_math_bin()
    if not bin_path:
        return {"enabled": False, "anchor_papers": [], "papers_indexed": 0, "query_ms": 0.0}
    cmd = [bin_path, "papers", "--topic", topic, "--top-k", str(top_k), "--json"]
    if methodology:
        cmd.extend(["--methodology", methodology])
    if pin_papers:
        cmd.append("--pin")
        cmd.extend(pin_papers)
    if eff_root:
        cmd.extend(["--project", eff_root])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            data["enabled"] = True
            return data
    except Exception:
        pass
    return {"enabled": False, "anchor_papers": [], "papers_indexed": 0, "query_ms": 0.0}


def format_anchor_papers_for_figure_expectation(selection: dict[str, Any]) -> str:
    """Format compact grounded anchor-paper cards for Stage 2A Blind Figure Expectation."""
    papers = selection.get("anchor_papers") or []
    if not papers:
        return "(No local anchor papers matched — adopt a standard cluster-cosmology ApJ/MNRAS referee lens.)"
    lines = [
        f"Vector-Selected Anchor Papers From Field Library ({selection.get('papers_indexed', 0)} papers indexed):"
    ]
    for p in papers:
        syms = ", ".join(p.get("key_symbols") or []) or "standard"
        concepts = ", ".join(p.get("key_concepts") or [])
        lines.append(
            f"- **{p['citation']}** — *{p['title']}* (sim={p['similarity']}, mode={p['selection_mode']})\n"
            f"  Key concepts: {concepts}\n"
            f"  Canonical symbols: {syms}\n"
            f"  Summary: {p.get('summary', '')[:260]}"
        )
    return "\n".join(lines)

"""Deterministic pixel color forensics & house colormap/palette preference profiles.

Zero-LLM-token analyzer (Pillow + NumPy) that extracts dominant chromatic colors from a
rendered figure, detects common colormap anti-patterns (default matplotlib tab10,
rainbow/jet maps, dead chalky low-chroma fills), and checks alignment with configurable
house style preferences (default: the house style + Sanzo Wada rules).
"""

from __future__ import annotations

import colorsys
import io
import math
from typing import Any

import numpy as np
from PIL import Image

from .schemas import BlindReviewPayload, ColorForensics, DeterministicAuditHit

# ---------------------------------------------------------------------------
# Reference Palettes & House Style Preferences (from esteves_plot.py)
# ---------------------------------------------------------------------------

TAB10_HEXES = [
    "#1f77b4",  # tab:blue
    "#ff7f0e",  # tab:orange
    "#2ca02c",  # tab:green
    "#d62728",  # tab:red
    "#9467bd",  # tab:purple
    "#8c564b",  # tab:brown
    "#e377c2",  # tab:pink
    "#bcbd22",  # tab:olive
    "#17becf",  # tab:cyan
]

HOUSE_PALETTES: dict[str, list[str]] = {
    "classic3": ["#b22222", "#000000", "#808080"],
    "hist2_skyblue_red": ["#87ceeb", "#ff0000"],
    "pal2": ["#219ebc", "#ffba0a"],
    "pal3": ["#f07167", "#219ebc", "#ffba0a"],
    "pal4": ["#f07167", "#219ebc", "#ffba0a", "#c767ac"],
    "pal5": ["#f07167", "#219ebc", "#ffba0a", "#c767ac", "#59a270"],
    "blues3": ["#c6dbef", "#6baed6", "#2171b5"],
    "getdist2": ["#e03424", "#006fed"],
    "vol1-2": ["#12354e", "#f99d1b"],
    "vol1-164": ["#dd4027", "#fcb315", "#4f4086"],
    "vol1-267": ["#006eb8", "#00978d", "#f99d1b", "#fdbf68"],
    "vol1-331": ["#1e0e3f", "#c53c69", "#4f4086", "#9a72aa"],
    "vol1-323": ["#111314", "#762c19", "#b2b73e", "#fdc57e"],
    "sky_div": ["#219ebc", "#ffffff", "#c767ac"],
    "tyrian_div": ["#12354e", "#f3efe6", "#dd4027"],
    "wada331_div": ["#4f4086", "#f3f0e7", "#c53c69"],
}

ESTEVES_COLOR_PREFERENCES = """House Color & Colormap Preferences (`esteves_plot.py` / house style + Sanzo Wada):
1. Banned defaults: NEVER use default matplotlib `tab10` colors, rainbow/`jet` colormaps, or lifeless chalky low-chroma fills (e.g. `#DCE3EC` / `lightsteelblue`, where HLS lightness > 0.72 and saturation < 0.50).
2. Curated discrete palettes (matched to role):
   - Model vs. data/mock vs. baseline (2-3 curves): `classic3` (`["firebrick", "black", "grey"]`).
   - Primary vs. secondary surveys / 2-5 category curves: `pal2` (`#219ebc` Teal, `#ffba0a` Marigold), `pal3` (`#f07167` Coral, `#219ebc`, `#ffba0a`), `pal4` (+ `#c767ac` Plum), `pal5` (+ `#59a270` Sage Green).
   - Histograms: saturated `#219ebc` `stepfilled` (alpha~0.65) + unfilled `#ffba0a` `step` outline, OR `hist2` (`skyblue` + `red` KDE).
   - Posterior contours (GetDist): `getdist2` (`#e03424` Brick Red, `#006fed` Cobalt Steel) with 68%/95%/99% nested fills vs. dashed outlines.
   - Sanzo Wada plates: `vol1-164` (`#dd4027, #fcb315, #4f4086`), `vol1-2` (`#12354e, #f99d1b`), `vol1-267`, `vol1-331`, `vol1-323`.
3. Preferred continuous colormaps:
   - **Diverging correlation matrices, residuals, and signed 2D maps (PREFERRED):** Soft **`RdBu_r`** (truncated to `[0.10, 0.90]` so endpoints stay rich and never crush into near-black maroon/navy!) OR **`sky_div`** (`#219ebc` Teal -> pure `#ffffff` white -> `#c767ac` Plum) / `coolwarm`. Avoid dark, heavy navy/brick-red diverging maps like `tyrian_div` on correlation matrices.
   - **Sequential positive-only maps / footprints:** `teal_seq`, `gold_seq`, `blues3`, `rocket_r`, `wada323`.
   - **Ordered multi-curve ladders (`N >= 5` monotonic bins, e.g. richness thresholds):** NEVER sample rainbow-like multi-hue ramps (such as `wada164_safe`, `jet`, `turbo`) for `plt.plot` lines — they produce jarring violet-navy-teal-green-yellow-orange ladders that clash with 2D heatmaps! Use a single harmonious monotonic ramp (e.g. `PuBu`/`Blues` `[0.38, 0.92]`, `rocket` `[0.18, 0.76]`, `Reds` `[0.30, 0.88]`) OR matched two-pole colors drawn directly from the companion `RdBu_r` heatmap (skipping white).
   - **Cross-panel color harmony:** When a multi-panel figure pairs a 2D heatmap with 1D curve panels, curve colors MUST harmonize with the heatmap colormap family.
   - **`scatter(..., c=z)` on white background:** `wada164_safe` (for colored scatter clouds only, NOT 1D line ladders!).
4. **MANDATORY diverging midpoint, percentile-stretch & ANTI-SATURATION rule (CRITICAL):**
   - Set `[vmin, vmax]` from central data percentiles chosen among `{98%, 95%, 91%, 68%}` (prefer `98%` `[1, 99]` or modest headroom on smooth correlation matrices so dipole cores and diagonal ridges preserve visible internal gradients rather than clipping into flat saturated dark plateaus!), rendered with `extend="both"` only when extreme outliers are truly clipped.
   - **Avoid saturation:** Flag flat clipped dark-red/dark-blue plateaus inside heatmaps! Recommend slicing `RdBu_r` to `[0.10, 0.90]` and widening bounds slightly (e.g. `98%` + 5% headroom, and flooring `vmin_use = min(vmin, -0.35 * vmax)` so tiny negative noise does not saturate dark blue).
   - **NEVER use a linear `Normalize(vmin, vmax)` on asymmetric `[vmin, vmax]` with a diverging colormap!** Always anchor pure white strictly at **`0.0`** using `TwoSlopeNorm(vmin=vmin_use, vcenter=0.0, vmax=vmax)`.
   - Shared comparison heatmaps (e.g. Cluster vs. Random) must share an identical quantitative colorbar scale and `TwoSlopeNorm(vcenter=0)`.
5. Spines & ticks (paper mode): thin neutral grey (`0.55`, 0.85 pt) 4-sided box with inward major/minor grey ticks — avoid harsh pitch-black frames clashing against bright colors."""

GENERIC_COLOR_PREFERENCES = """General Scientific Colormap & Palette Standards:
1. Avoid rainbow/`jet` colormaps; use perceptually uniform sequential maps (e.g. viridis, cividis, rocket_r) for monotonic data and balanced diverging maps (e.g. RdBu_r, coolwarm, PuOr) for signed data.
2. Anchor diverging colormaps at 0 (or explicit reference center), clip outlier tails with central percentiles + `extend="both"`, and keep side-by-side panels on identical color scales.
3. Ensure curves/markers are colorblind-friendly (distinguishable by both hue and linestyle/marker) with strong contrast against the white background."""


def get_color_preferences_text(
    style_profile: str = "esteves",
    custom_preferences: str | None = None,
) -> str:
    """Return the active color/colormap preference guide for Master Gate & Reviewer 2."""
    key = (style_profile or "esteves").strip().lower()
    if key in {"esteves", "house", "wada"}:
        base = ESTEVES_COLOR_PREFERENCES
    elif key in {"generic", "standard", "journal"}:
        base = GENERIC_COLOR_PREFERENCES
    else:
        base = ESTEVES_COLOR_PREFERENCES

    if custom_preferences and custom_preferences.strip():
        return f"{base}\n\nAdditional User Color Preferences:\n{custom_preferences.strip()}"
    return base


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    s = h.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _rgb_dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def analyze_figure_colors(png_bytes: bytes) -> ColorForensics:
    """Extract dominant chromatic hexes and detect tab10, rainbow/jet, and dead-pale fills."""
    with Image.open(io.BytesIO(png_bytes)) as img:
        thumb = img.convert("RGB")
        thumb.thumbnail((360, 360))
        arr = np.asarray(thumb, dtype=np.uint8).reshape(-1, 3)

    # Separate chromatic pixels from near-white / near-grey / near-black background & text
    r = arr[:, 0].astype(np.float32) / 255.0
    g = arr[:, 1].astype(np.float32) / 255.0
    b = arr[:, 2].astype(np.float32) / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    light = 0.5 * (cmax + cmin)
    denom = 1.0 - np.abs(2.0 * light - 1.0)
    sat = np.where(delta > 1e-6, delta / np.maximum(denom, 1e-6), 0.0)

    # Chromatic pixels: visible color saturation and not near-pure white or black
    chrom_mask = (delta > 0.07) & (light > 0.08) & (light < 0.96)
    chrom_fraction = float(np.mean(chrom_mask))

    # Dead-pale check matching esteves_plot._reject_dead_pale (L > 0.72 and 0.08 < S < 0.50)
    dead_pale_mask = (delta > 0.04) & (light > 0.72) & (light < 0.94) & (sat < 0.50)
    dead_pale_fraction = float(np.mean(dead_pale_mask))

    # Saturated/crushed dark extreme check (e.g. clipped deep maroon/navy plateaus in heatmaps)
    sat_dark_mask = (delta > 0.14) & (light < 0.24) & (sat > 0.55)
    sat_dark_fraction = float(np.mean(sat_dark_mask))

    # Quantize chromatic pixels into 16-level RGB bins to recover crisp dominant hexes
    dominant_hexes: list[str] = []
    chrom_rgbs: list[tuple[int, int, int]] = []
    if np.any(chrom_mask):
        c_arr = arr[chrom_mask]
        quant = (c_arr // 16) * 16 + 8
        packed = (
            quant[:, 0].astype(np.int32) * 65536
            + quant[:, 1].astype(np.int32) * 256
            + quant[:, 2].astype(np.int32)
        )
        vals, counts = np.unique(packed, return_counts=True)
        order = np.argsort(-counts)
        for idx in order:
            v = int(vals[idx])
            rgb = ((v >> 16) & 255, (v >> 8) & 255, v & 255)
            # Deduplicate perceptually near-identical bins (distance < 34)
            if all(_rgb_dist(rgb, prev) >= 34.0 for prev in chrom_rgbs):
                chrom_rgbs.append(rgb)
                dominant_hexes.append(_rgb_to_hex(rgb))
                if len(dominant_hexes) >= 8:
                    break

    # Detect matplotlib tab10 signature (>= 2 distinct tab10 hues within RGB distance <= 22)
    tab10_rgbs = [_hex_to_rgb(h) for h in TAB10_HEXES]
    matched_tab10: list[str] = []
    for hex_t, rgb_t in zip(TAB10_HEXES, tab10_rgbs):
        if any(_rgb_dist(c, rgb_t) <= 22.0 for c in chrom_rgbs):
            matched_tab10.append(hex_t)
    tab10_suspected = len(matched_tab10) >= 2

    # Detect rainbow / jet signature: simultaneous saturated red + neon-green + cyan + blue
    rainbow_suspected = False
    if np.sum(chrom_mask) > 500:
        hi_sat = (sat > 0.65) & (light > 0.25) & (light < 0.78)
        if np.sum(hi_sat) > 400:
            hs_arr = arr[hi_sat]
            hues = np.array(
                [colorsys.rgb_to_hsv(px[0] / 255.0, px[1] / 255.0, px[2] / 255.0)[0] for px in hs_arr[::2]]
            )
            has_red = bool(np.mean((hues < 0.05) | (hues > 0.94)) > 0.04)
            has_yellow = bool(np.mean((hues >= 0.12) & (hues <= 0.20)) > 0.04)
            has_neon_green = bool(np.mean((hues >= 0.26) & (hues <= 0.40)) > 0.05)
            has_cyan = bool(np.mean((hues >= 0.45) & (hues <= 0.56)) > 0.05)
            has_blue = bool(np.mean((hues >= 0.58) & (hues <= 0.72)) > 0.05)
            rainbow_suspected = sum([has_red, has_yellow, has_neon_green, has_cyan, has_blue]) >= 4

    # Match against the curated house palettes (excluding white/black/grey neutrals)
    matched_palettes: list[str] = []
    for pal_name, hexes in HOUSE_PALETTES.items():
        pal_chrom = [
            _hex_to_rgb(h)
            for h in hexes
            if h.lower() not in {"#ffffff", "#000000", "#808080", "#f3efe6", "#f3f0e7", "#111314"}
        ]
        if not pal_chrom:
            continue
        hits = sum(1 for prgb in pal_chrom if any(_rgb_dist(c, prgb) <= 26.0 for c in chrom_rgbs))
        needed = min(2, len(pal_chrom))
        if hits >= needed:
            matched_palettes.append(pal_name)

    # Check if there is a broad continuous color ramp (likely heatmap/colorbar)
    is_likely_heatmap = chrom_fraction > 0.22 and len(chrom_rgbs) >= 5
    sat_clip_suspected = is_likely_heatmap and sat_dark_fraction > 0.035

    notes: list[str] = []
    if sat_clip_suspected:
        notes.append(
            f"Saturated dark extremes detected ({sat_dark_fraction:.1%} of pixels are dark/clipped "
            "high-chroma tones): avoid saturation by slicing `RdBu_r` to `[0.10, 0.90]` and widening "
            "percentile bounds (e.g. 98% [1, 99] + slight headroom) so peak/dipole gradients stay visible."
        )
    if rainbow_suspected:
        notes.append(
            "Rainbow/`jet`-like full-spectrum hues detected; replace with `sky_div` "
            "(#219ebc -> #ffffff -> #c767ac), soft `RdBu_r`/`coolwarm`, `wada323`, or `wada164_safe`."
        )
    if tab10_suspected:
        notes.append(
            f"Default matplotlib `tab10`-like colors detected ({', '.join(matched_tab10)}); "
            "consider the curated `pal2/3/4/5`, `classic3`, or Sanzo Wada (`vol1-164`, `vol1-2`, `vol1-331`)."
        )
    if dead_pale_fraction > 0.06 and not is_likely_heatmap:
        notes.append(
            f"Noticeable low-chroma pale fill area ({dead_pale_fraction:.1%} of pixels, L>0.72 & S<0.50); "
            "avoid chalky grey-blue fills (use `#219ebc` stepfilled + `#ffba0a` step or `skyblue`/`red`)."
        )
    if matched_palettes:
        notes.append(f"Matches palette/colormap signature(s): {', '.join(matched_palettes[:4])}.")
    if is_likely_heatmap and "tyrian_div" in matched_palettes:
        notes.append(
            "Dark navy/brick `tyrian_div` heatmap detected: prefer cleaner `sky_div` "
            "(#219ebc -> #ffffff -> #c767ac) or soft `RdBu_r`/`coolwarm` anchored at 0 via TwoSlopeNorm(vcenter=0)."
        )
    if is_likely_heatmap:
        notes.append(
            "High chromatic coverage suggests a 2D heatmap/matrix: verify central percentile stretch "
            "({98%, 95%, 91%, 68%}, default [2.5, 97.5]) with `extend='both'` AND pure white anchored at 0.0 via `TwoSlopeNorm(vcenter=0)`."
        )

    return ColorForensics(
        dominant_chromatic_hexes=dominant_hexes,
        chromatic_pixel_fraction=round(chrom_fraction, 4),
        dead_pale_pixel_fraction=round(dead_pale_fraction, 4),
        saturated_dark_extreme_fraction=round(sat_dark_fraction, 4),
        likely_heatmap_or_colorbar=is_likely_heatmap,
        saturation_clipping_suspected=sat_clip_suspected,
        tab10_suspected=tab10_suspected,
        rainbow_jet_suspected=rainbow_suspected,
        matched_house_palettes=matched_palettes[:6],
        color_notes=notes,
    )


def audit_colorbar_midpoints_deterministically(
    blind_review: BlindReviewPayload,
) -> list[DeterministicAuditHit]:
    """Deterministically catch diverging colorbars whose white center drifted off 0."""
    hits: list[DeterministicAuditHit] = []
    for cb in blind_review.colormap_audit.colorbars:
        if cb.colorbar_type != "diverging":
            continue
        vmin = cb.vmin_tick
        vmax = cb.vmax_tick
        mid = cb.neutral_center_tick
        if vmin is None or vmax is None or vmax <= vmin:
            continue
        span = vmax - vmin
        if not (vmin < 0.0 < vmax):
            continue

        if mid is not None and abs(mid) > 0.08 * span:
            hits.append(
                DeterministicAuditHit(
                    term=f"diverging midpoint={mid:+.2f} (range [{vmin:+.2f}, {vmax:+.2f}])",
                    source="colorbar_midpoint_misaligned",
                    location=cb.location,
                    detail=(
                        f"Diverging colorbar white/neutral center sits at {mid:+.2f} instead of 0.0 "
                        f"({abs(mid)/span:.0%} of span off-zero), causing uncorrelated 0.0 cells to "
                        "render strongly colored. Anchor white at 0 with "
                        "`TwoSlopeNorm(vmin, vcenter=0.0, vmax)` and use `sky_div` or soft `RdBu_r`/`coolwarm`."
                    ),
                )
            )
            continue

        if cb.zero_is_at_white_center is False:
            hits.append(
                DeterministicAuditHit(
                    term=f"zero not neutral (range [{vmin:+.2f}, {vmax:+.2f}])",
                    source="colorbar_midpoint_misaligned",
                    location=cb.location,
                    detail=(
                        f"Zero is tinted rather than neutral white on diverging range "
                        f"[{vmin:+.2f}, {vmax:+.2f}]. Anchor white at 0 with "
                        "`TwoSlopeNorm(vmin, vcenter=0.0, vmax)`."
                    ),
                )
            )
            continue

        if mid is None and abs(vmin + vmax) / span > 0.25:
            inferred_mid = 0.5 * (vmin + vmax)
            hits.append(
                DeterministicAuditHit(
                    term=f"asymmetric diverging bounds [{vmin:+.2f}, {vmax:+.2f}]",
                    source="colorbar_midpoint_misaligned",
                    location=cb.location,
                    detail=(
                        f"Strongly asymmetric diverging bounds [{vmin:+.2f}, {vmax:+.2f}] put a "
                        f"linear midpoint at {inferred_mid:+.2f} rather than 0.0. Anchor white "
                        "cleanly at 0 with `TwoSlopeNorm(vcenter=0.0)`."
                    ),
                )
            )
    return hits

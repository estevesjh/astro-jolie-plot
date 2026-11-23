"""Unit tests for deterministic pixel color forensics and house colormap preference profiles."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw

from mcp_figure_reviewer.blinding import inspect_figure_pixels
from mcp_figure_reviewer.colormap import (
    analyze_figure_colors,
    audit_colorbar_midpoints_deterministically,
    get_color_preferences_text,
)
from mcp_figure_reviewer.schemas import BlindReviewPayload, ColorbarReading, ColormapAudit


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_detects_decade4_house_palette_cleanly() -> None:
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, color in enumerate(["#f07167", "#219ebc", "#ffba0a", "#c767ac"]):
        draw.rectangle([20, 20 + i * 60, 380, 60 + i * 60], fill=color)

    forensics = analyze_figure_colors(_png_bytes(img))
    assert "decade4" in forensics.matched_house_palettes
    assert forensics.tab10_suspected is False
    assert forensics.rainbow_jet_suspected is False


def test_detects_matplotlib_tab10_defaults() -> None:
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, color in enumerate(["#1f77b4", "#ff7f0e", "#2ca02c"]):
        draw.rectangle([20, 20 + i * 80, 380, 75 + i * 80], fill=color)

    forensics = analyze_figure_colors(_png_bytes(img))
    assert forensics.tab10_suspected is True
    assert any("tab10" in n for n in forensics.color_notes)


def test_detects_rainbow_jet_colormap() -> None:
    img = Image.new("RGB", (300, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    bands = ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000"]
    w = 300 // len(bands)
    for i, c in enumerate(bands):
        draw.rectangle([i * w, 0, (i + 1) * w, 200], fill=c)

    forensics = analyze_figure_colors(_png_bytes(img))
    assert forensics.rainbow_jet_suspected is True
    assert any("Rainbow" in n or "jet" in n for n in forensics.color_notes)


def test_detects_dead_pale_chalky_histogram_fill() -> None:
    img = Image.new("RGB", (300, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 270, 170], fill="#dce3ec")

    forensics = analyze_figure_colors(_png_bytes(img))
    assert forensics.dead_pale_pixel_fraction > 0.30
    assert any("pale fill" in n for n in forensics.color_notes)


def test_color_preferences_profile_supports_esteves_generic_and_custom() -> None:
    esteves = get_color_preferences_text("esteves")
    assert "decade_sky" in esteves
    assert "TwoSlopeNorm" in esteves
    assert "decade4" in esteves
    assert "vol1-164" in esteves

    generic = get_color_preferences_text("generic", custom_preferences="Prefer cividis for maps.")
    assert "cividis" in generic
    assert "decade_sky" not in generic


def test_deterministic_colorbar_midpoint_audit_catches_off_zero_white() -> None:
    def _review(cb: ColorbarReading) -> BlindReviewPayload:
        return BlindReviewPayload(
            one_sentence_conclusion="Correlation matrix.",
            panel_encodings=[],
            unclear_terms=[],
            colormap_audit=ColormapAudit(
                has_colorbar=True,
                colorbars=[cb],
                observed_palette_description="Blue-white-red diverging.",
                colormap_role_match="APPROPRIATE",
                stretch_and_midpoint_notes="White at +0.33.",
                contrast_and_accessibility="Good.",
                colormap_issues=[],
            ),
            visual_and_layout_issues=[],
            confusing_elements=[],
            standalone_readability_score=3,
            readability_summary="Test.",
        )

    # Off-zero linear norm (fig9 bug: vmin=-0.31, vmax=+0.97, white at +0.33) -> MUST FLAG
    bad = _review(
        ColorbarReading(
            location="Top-right colorbar",
            colorbar_type="diverging",
            vmin_tick=-0.31,
            vmax_tick=0.97,
            neutral_center_tick=0.33,
            zero_is_at_white_center=False,
            has_extend_arrows=True,
        )
    )
    hits = audit_colorbar_midpoints_deterministically(bad)
    assert len(hits) == 1
    assert hits[0].source == "colorbar_midpoint_misaligned"
    assert "+0.33" in hits[0].term

    # Properly TwoSlopeNorm-anchored at 0.0 -> ZERO hits
    good = _review(
        ColorbarReading(
            location="Top-right colorbar",
            colorbar_type="diverging",
            vmin_tick=-0.31,
            vmax_tick=0.97,
            neutral_center_tick=0.0,
            zero_is_at_white_center=True,
            has_extend_arrows=True,
        )
    )
    assert audit_colorbar_midpoints_deterministically(good) == []


def test_real_figure_color_forensics_if_present() -> None:
    fig = Path(
        "/nonexistent/local-only-file"
        "fig9_desy3_gp_spatial_covariance_lamge45.png"
    )
    if not fig.is_file():
        return
    audit = inspect_figure_pixels(fig)
    assert audit.color_forensics.likely_heatmap_or_colorbar is True
    assert len(audit.color_forensics.dominant_chromatic_hexes) >= 4
    assert audit.color_forensics.rainbow_jet_suspected is False

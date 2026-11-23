"""Optional live check against a real Anthropic backend. Skipped when none is available.

Run with:  pytest -m live
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from mcp_figure_reviewer.backends import detect_backends
from mcp_figure_reviewer.reviewer import run_blind_review

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_live_blind_review_on_simple_plot(tmp_path: Path, monkeypatch) -> None:
    if detect_backends()["active_default_backend"] is None:
        pytest.skip("No Anthropic backend (claude_cli or anthropic_api) is available.")
    monkeypatch.setenv("FIGURE_REVIEW_STORE_DIR", str(tmp_path / "store"))

    fig = tmp_path / "plot.png"
    img = Image.new("RGB", (900, 600), "white")
    d = ImageDraw.Draw(img)
    d.line([(100, 500), (800, 100)], fill=(178, 34, 34), width=4)
    d.rectangle([100, 50, 800, 500], outline=(140, 140, 140), width=2)
    d.text((420, 540), "x", fill="black")
    img.save(fig)

    review = await run_blind_review(fig, caption="A straight line.", detail="compact")
    assert review.blinding_certificate.model.startswith("claude-")
    assert review.blinding_certificate.original_filename_hidden is True

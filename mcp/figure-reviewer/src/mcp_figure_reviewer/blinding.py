"""Physical blinding engine for scientific figures.

Guarantees that the blind reviewer never receives:
- Original filenames or repository directory paths (e.g. fig9_real_desy3_...png)
- Embedded PNG text chunks (tEXt/iTXt/zTXt), EXIF, XMP, ICC creator tags, or PDF metadata
- Any neighboring project files when executed via a sandboxed CLI backend
"""

from __future__ import annotations

import hashlib
import io
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .colormap import analyze_figure_colors
from .notation import audit_filename_jargon
from .schemas import ColorForensics, PixelAuditResult

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
SUPPORTED_PDF_SUFFIXES = {".pdf"}
BLIND_FILENAME = "figure.png"

FAST_MODE_DPI = 150
FAST_MODE_MAX_DIMENSION = 1200
FAST_MODE_FORMAT = "WEBP"

DEEP_MODE_DPI = 220
DEEP_MODE_MAX_DIMENSION = 1800
DEEP_MODE_FORMAT = "PNG"


def image_settings_for_speed(
    speed: str = "fast",
    *,
    dpi: int | None = None,
    max_dimension: int | None = None,
    output_format: str | None = None,
) -> tuple[int, int, str]:
    """Return (dpi, max_dimension, output_format) tuned for 'fast' vs 'deep' review modes.

    - fast: 1200 px max dimension, 150 DPI PDF rasterization, WebP quality=90 (~4-5x smaller
      payload and ~44% fewer vision pixels than 1600 px PNG while keeping 9pt LaTeX crisp).
    - deep: 1800 px max dimension, 220 DPI PDF rasterization, lossless PNG.
    """
    is_deep = (speed or "fast").strip().lower() == "deep"
    eff_dpi = dpi if dpi is not None else (DEEP_MODE_DPI if is_deep else FAST_MODE_DPI)
    eff_max_dim = (
        max_dimension
        if max_dimension is not None
        else (DEEP_MODE_MAX_DIMENSION if is_deep else FAST_MODE_MAX_DIMENSION)
    )
    eff_fmt = (
        output_format.upper()
        if output_format is not None
        else (DEEP_MODE_FORMAT if is_deep else FAST_MODE_FORMAT)
    )
    return eff_dpi, eff_max_dim, eff_fmt


@dataclass(frozen=True)
class PreparedBlindImage:
    """Clean, metadata-stripped image payload ready for a blinded LLM call."""

    png_bytes: bytes
    sha256: str
    width_px: int
    height_px: int
    source_format: str
    source_page: int | None
    stripped_metadata_keys: list[str]
    mime_type: str = "image/png"
    blind_filename: str = BLIND_FILENAME
    payload_format: str = "PNG"
    orig_width_px: int = 0
    orig_height_px: int = 0


def _render_pdf_page(pdf_path: Path, page: int = 1, dpi: int = 220) -> tuple[Image.Image, list[str]]:
    """Render a 1-indexed PDF page to an RGB PIL Image and record PDF metadata keys."""
    import pymupdf

    if page < 1:
        raise ValueError(f"PDF page must be >= 1, got {page}")

    with pymupdf.open(str(pdf_path)) as doc:
        if page > len(doc):
            raise ValueError(
                f"PDF {pdf_path.name} has {len(doc)} page(s); requested page {page}"
            )
        meta = doc.metadata or {}
        stripped_keys = sorted(k for k, v in meta.items() if v)
        pdf_page = doc.load_page(page - 1)
        scale = dpi / 72.0
        pix = pdf_page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img, stripped_keys


def _load_raster_image(image_path: Path) -> tuple[Image.Image, str, list[str]]:
    """Load a raster image, capture its embedded metadata keys, and composite onto white RGB."""
    with Image.open(image_path) as raw:
        source_format = (raw.format or image_path.suffix.lstrip(".")).upper()
        stripped_keys = sorted(
            str(k)
            for k in raw.info.keys()
            if k not in {"dpi", "jfif", "jfif_version", "jfif_unit", "jfif_density"}
        )
        exif = raw.getexif()
        if exif and len(exif) > 0 and "exif" not in stripped_keys:
            stripped_keys.append("exif")

        # Composite transparent images cleanly onto white paper background
        if raw.mode in ("RGBA", "LA") or (raw.mode == "P" and "transparency" in raw.info):
            rgba = raw.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            rgb = raw.convert("RGB")

    return rgb, source_format, sorted(set(stripped_keys))


def prepare_blind_image(
    image_path: str | Path,
    *,
    page: int = 1,
    dpi: int = 180,
    max_dimension: int = 1600,
    output_format: str = "PNG",
) -> PreparedBlindImage:
    """Load a figure and return strictly metadata-stripped RGB image bytes.

    Rebuilds the image from raw RGB pixel bytes so zero PNG text chunks, EXIF,
    XMP, ICC profiles, or creator strings survive into the blinded payload.
    Supports lossless 'PNG' (default) as well as compact 'WEBP' / 'JPEG' (quality=90)
    for fast-mode vision inference.
    """
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Figure file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in SUPPORTED_PDF_SUFFIXES:
        rgb, stripped_keys = _render_pdf_page(path, page=page, dpi=dpi)
        source_format = "PDF"
        source_page: int | None = page
    elif suffix in SUPPORTED_IMAGE_SUFFIXES:
        rgb, source_format, stripped_keys = _load_raster_image(path)
        source_page = None
    else:
        raise ValueError(
            f"Unsupported figure format '{suffix}'. Supported: "
            f"{sorted(SUPPORTED_IMAGE_SUFFIXES | SUPPORTED_PDF_SUFFIXES)}"
        )

    orig_w, orig_h = rgb.size
    w, h = orig_w, orig_h
    # Downscale large images proportionally while preserving crisp plot text
    if max(w, h) > max_dimension:
        scale = max_dimension / float(max(w, h))
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
        w, h = rgb.size

    # Rebuild from raw pixel buffer so zero metadata dicts/chunks carry over
    clean = Image.frombytes("RGB", (w, h), rgb.tobytes())
    buf = io.BytesIO()
    fmt = (output_format or "PNG").strip().upper()
    if fmt == "WEBP":
        clean.save(buf, format="WEBP", quality=90, method=4)
        mime_type = "image/webp"
        blind_filename = "figure.webp"
        payload_format = "WEBP"
    elif fmt in {"JPG", "JPEG"}:
        clean.save(buf, format="JPEG", quality=90, optimize=True)
        mime_type = "image/jpeg"
        blind_filename = "figure.jpg"
        payload_format = "JPEG"
    else:
        clean.save(buf, format="PNG", optimize=True)
        mime_type = "image/png"
        blind_filename = BLIND_FILENAME
        payload_format = "PNG"

    payload_bytes = buf.getvalue()
    digest = hashlib.sha256(payload_bytes).hexdigest()

    return PreparedBlindImage(
        png_bytes=payload_bytes,
        sha256=digest,
        width_px=w,
        height_px=h,
        source_format=source_format,
        source_page=source_page,
        stripped_metadata_keys=stripped_keys,
        mime_type=mime_type,
        blind_filename=blind_filename,
        payload_format=payload_format,
        orig_width_px=orig_w,
        orig_height_px=orig_h,
    )


class IsolatedBlindSandbox:
    """Ephemeral /tmp directory containing ONLY the anonymized, stripped figure image."""

    def __init__(self, prepared: PreparedBlindImage | None = None, prefix: str = "blind_fig_") -> None:
        self._prepared = prepared
        self._prefix = prefix
        self.root: Path | None = None
        self.figure_path: Path | None = None

    def __enter__(self) -> IsolatedBlindSandbox:
        self.root = Path(tempfile.mkdtemp(prefix=self._prefix, dir=tempfile.gettempdir()))
        if self._prepared is not None:
            self.figure_path = self.root / self._prepared.blind_filename
            self.figure_path.write_bytes(self._prepared.png_bytes)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.root is not None and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


def evaluate_fast_fail_blockers(
    *,
    width_px: int,
    height_px: int,
    color_forensics: ColorForensics,
    filename_jargon_tokens: list[str] | None = None,
    fail_on_filename_jargon: bool = False,
) -> list[str]:
    """Identify hard 0-token pixel/colormap defects that should short-circuit LLM review."""
    blockers: list[str] = []
    # Trigger resolution fast-fail only on genuinely blurry thumbnails (not valid wide strips)
    if max(width_px, height_px) < 600 or min(width_px, height_px) < 350:
        blockers.append(
            f"Very low resolution ({width_px}x{height_px} px): regenerate at >=200 DPI before "
            "spending LLM review time."
        )
    if color_forensics.rainbow_jet_suspected:
        blockers.append(
            "Rainbow/jet-like full-spectrum colormap detected: replace with soft RdBu_r "
            "[0.10, 0.90], decade_sky, or a perceptually uniform sequential map before LLM review."
        )
    if color_forensics.saturation_clipping_suspected:
        blockers.append(
            f"Severe dark-extreme saturation clipping detected "
            f"({color_forensics.saturated_dark_extreme_fraction:.1%} of pixels): slice RdBu_r to "
            "[0.10, 0.90] and widen percentile stretch (98% + headroom) before LLM review."
        )
    if fail_on_filename_jargon and filename_jargon_tokens:
        blockers.append(
            f"Filename contains opaque batch/code tokens {filename_jargon_tokens}: rename to a "
            "clean physical filename before review."
        )
    return blockers


def inspect_figure_pixels(
    image_path: str | Path,
    *,
    page: int = 1,
    dpi: int = 220,
) -> PixelAuditResult:
    """Zero-token deterministic resolution, metadata, and clipping pre-audit."""
    path = Path(image_path).expanduser().resolve()
    prepared = prepare_blind_image(path, page=page, dpi=dpi, max_dimension=10000)

    with Image.open(io.BytesIO(prepared.png_bytes)) as img:
        # Downsample thumbnail for fast pure-Python luminance statistics
        thumb = img.copy()
        thumb.thumbnail((400, 400))
        getter = getattr(thumb, "get_flattened_data", thumb.getdata)
        pixels = list(getter())

    total = max(1, len(pixels))
    near_white = sum(1 for r, g, b in pixels if r >= 250 and g >= 250 and b >= 250) / total
    near_black = sum(1 for r, g, b in pixels if r <= 12 and g <= 12 and b <= 12) / total

    aspect = round(prepared.width_px / max(1, prepared.height_px), 3)
    low_res = min(prepared.width_px, prepared.height_px) < 600
    warnings: list[str] = []
    if low_res:
        warnings.append(
            f"Low resolution ({prepared.width_px}x{prepared.height_px} px): small tick/math labels may blur; recommend >=200 DPI."
        )
    if aspect > 4.0 or aspect < 0.25:
        warnings.append(
            f"Extreme aspect ratio ({aspect}:1): verify panel readability and font legibility at print column width."
        )
    if near_white > 0.92:
        warnings.append(
            f"Very high white-space fraction ({near_white:.1%}): figure may have excessive margins or faint curves."
        )
    if prepared.stripped_metadata_keys:
        warnings.append(
            f"Source file contains metadata keys {prepared.stripped_metadata_keys}; physical blinding will strip all of them."
        )

    forensics = analyze_figure_colors(prepared.png_bytes)
    fn_jargon = audit_filename_jargon(path)
    if forensics.rainbow_jet_suspected:
        warnings.append("Colormap warning: rainbow/jet-like full-spectrum hues detected.")
    if forensics.saturation_clipping_suspected:
        warnings.append(
            f"Saturation warning: {forensics.saturated_dark_extreme_fraction:.1%} of pixels hit dark saturated extremes; widen percentile stretch (e.g. 98%) or truncate RdBu_r to [0.10, 0.90]."
        )
    if forensics.tab10_suspected and not forensics.matched_house_palettes:
        warnings.append("Palette warning: default matplotlib tab10 colors suspected.")
    if fn_jargon:
        warnings.append(
            f"Filename contains opaque batch/code tokens {fn_jargon}; consider renaming to a clean physical filename."
        )

    fast_fail_blockers = evaluate_fast_fail_blockers(
        width_px=prepared.width_px,
        height_px=prepared.height_px,
        color_forensics=forensics,
        filename_jargon_tokens=fn_jargon,
        fail_on_filename_jargon=False,
    )

    return PixelAuditResult(
        image_path=str(path),
        source_format=prepared.source_format,
        page=prepared.source_page,
        width_px=prepared.width_px,
        height_px=prepared.height_px,
        aspect_ratio=aspect,
        file_size_bytes=path.stat().st_size,
        image_sha256=prepared.sha256,
        embedded_metadata_keys=prepared.stripped_metadata_keys,
        near_white_pixel_fraction=round(near_white, 4),
        near_black_pixel_fraction=round(near_black, 4),
        low_resolution_warning=low_res,
        warnings=warnings,
        fast_fail_blockers=fast_fail_blockers,
        filename_jargon_tokens=fn_jargon,
        color_forensics=forensics,
    )

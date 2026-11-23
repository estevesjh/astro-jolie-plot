"""Plotting toolkit following the Anbajagane et al. (2026, OJA/DECADE)
house style, paired with curated Sanzo Wada plates (vol1-164, vol1-2, vol1-267,
vol1-331, vol1-323) and density-adaptive scatter / contour / histogram helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm

# ---------------------------------------------------------------------------
# Curated Palettes
# ---------------------------------------------------------------------------

PALETTES: dict[str, list[str]] = {
    # Traditional workhorse (model / data or mock / baseline)
    "classic3": ["firebrick", "black", "grey"],
    "traditional": ["firebrick", "black", "grey"],
    "hist2": ["skyblue", "red"],                                    # Favorite sns.histplot(..., kde=True) pair
    "skyblue_red": ["skyblue", "red"],
    # Exact Anbajagane et al. (2026, DECADE_Clusters_20260831.pdf) vector hexes
    "decade2": ["#219ebc", "#ffba0a"],                              # DECADE Teal, DES Marigold
    "decade3": ["#f07167", "#219ebc", "#ffba0a"],                   # Coral, Teal, Marigold (terciles / g-r, r-i, i-z)
    "decade4": ["#f07167", "#219ebc", "#ffba0a", "#c767ac"],        # + Orchid Plum (4 richness / z bins)
    "decade5": ["#f07167", "#219ebc", "#ffba0a", "#c767ac", "#59a270"],  # + Sage Green (5th series, Fig. 2)
    "decade_blues": ["#c6dbef", "#6baed6", "#2171b5"],              # ColorBrewer 3-bin Blues (Figs. F1-F2)
    "getdist2": ["#e03424", "#006fed"],                             # GetDist lines (fills: #f3aea7, #99c5f8)
    # Sanzo Wada favorites
    "vol1-2": ["#12354e", "#f99d1b"],                               # Dark Tyrian Blue, Yellow Orange
    "vol1-164": ["#dd4027", "#fcb315", "#4f4086"],                  # Red Orange, Orange Yellow, Violet
    "vol1-267": ["#006eb8", "#00978d", "#f99d1b", "#fdbf68"],       # Blue, Benzol Green, Yellow Orange, Cream Yellow
    "vol1-331": ["#1e0e3f", "#c53c69", "#4f4086", "#9a72aa"],       # Dull Violet Black, Indian Lake, Violet, Light Mauve
    "vol1-323": ["#111314", "#762c19", "#b2b73e", "#fdc57e"],       # Black, Madder Brown, Citron Yellow, Cinnamon Buff
    "vol1-249": ["#547076", "#793327", "#d6b43e", "#c2ae93"],       # Dark Medici Blue, Hay's Russet, Olive Ocher, Ecru
    "vol2-204": ["#112f2c", "#b73f74", "#5caac7"],                  # Deep Slate Green, Rosolanc Purple, Pale King's Blue
    "idx51": ["#c27544", "#007190", "#59256a"],                     # sw.palette(3, index=51)
}

_COLORMAPS_STOPS: dict[str, tuple[str, list[str]]] = {
    "decade_sky": ("div", ["#219ebc", "#ffffff", "#c767ac"]),       # Exact Fig. 2 (p.5) colorbar gradient
    "decade_teal": ("seq", ["#ffffff", "#219ebc"]),                 # Exact Fig. 1 DECADE footprint ramp
    "decade_gold": ("seq", ["#ffffff", "#ffba0a"]),                 # Exact Fig. 1 DES footprint ramp
    "decade_blues": ("seq", ["#c6dbef", "#6baed6", "#2171b5"]),
    "wada323": ("seq", ["#111314", "#762c19", "#b2b73e", "#fdc57e"]),
    "wada164_safe": ("seq", ["#4f4086", "#12354e", "#00978d", "#fcb315", "#dd4027"]),
    "tyrian_div": ("div", ["#12354e", "#f3efe6", "#dd4027"]),
    "wada331_div": ("div", ["#4f4086", "#f3f0e7", "#c53c69"]),
}


def get_palette(
    n: int | None = None,
    name: str | None = None,
    *,
    role: str = "categories",
    family: str | None = None,
    index: int = 0,
) -> list[str]:
    """Return a curated color list (`role='comparison'` -> traditional firebrick/black/grey; `categories` -> DECADE/Wada)."""
    if name is not None:
        key = name.lower()
        if key in PALETTES:
            cols = PALETTES[key]
            return cols if n is None else (cols * ((n // len(cols)) + 1))[:n]
        try:
            import sanzo_wada as sw
            return sw.get_combination(key).hexes
        except Exception as exc:
            raise ValueError(f"Unknown palette {name!r}") from exc

    n = 3 if n is None else int(n)
    if role == "comparison" and n <= 3:
        return PALETTES["classic3"][:n]
    if n <= 2:
        return PALETTES["decade2"][:n]
    if n == 3:
        return PALETTES["decade3"]
    if n == 4:
        return PALETTES["decade4"]
    if n == 5:
        return PALETTES["decade5"]

    try:
        import sanzo_wada as sw
        for idx in range(index, len(sw.combinations)):
            cols = sw.palette(n, family=family, index=idx)
            if all(sum(int(h[i : i + 2], 16) for i in (1, 3, 5)) / 3.0 < 222 for h in cols):
                return cols
    except Exception:
        pass

    cm = get_cmap("wada164_safe")
    return [mpl.colors.to_hex(cm(v)) for v in np.linspace(0.05, 0.95, n)]


def get_cmap(name: str = "decade_sky", *, discrete: int | None = None):
    """Return a matplotlib Colormap tuned to a specific visual role."""
    key = name.lower()
    if key in _COLORMAPS_STOPS:
        _, stops = _COLORMAPS_STOPS[key]
        if discrete is not None:
            cm_cont = LinearSegmentedColormap.from_list(key, stops, N=256)
            return ListedColormap([cm_cont(v) for v in np.linspace(0, 1, discrete)], name=f"{key}_{discrete}")
        return LinearSegmentedColormap.from_list(key, stops, N=256)
    return plt.get_cmap(name)


def diverging_norm(data: np.ndarray, *, vcenter: float = 0.0, q: float = 99.0) -> TwoSlopeNorm:
    """Symmetric TwoSlopeNorm around `vcenter` clipped safely against outliers."""
    arr = np.asarray(data, float)
    finite = arr[np.isfinite(arr)]
    span = float(np.percentile(np.abs(finite - vcenter), q)) if finite.size else 1.0
    span = max(span, 1e-12)
    return TwoSlopeNorm(vmin=vcenter - span, vcenter=vcenter, vmax=vcenter + span)


# ---------------------------------------------------------------------------
# Typography, Frames & Axes Geometry (Anbajagane / OJA Default)
# ---------------------------------------------------------------------------

FRAME = "0.55"
TICK_LABEL_COLOR = "0.18"


def box_inward(
    ax: plt.Axes,
    *,
    frame_color: str = FRAME,
    label_color: str = TICK_LABEL_COLOR,
    frame_lw: float = 0.85,
    minor: bool = True,
) -> plt.Axes:
    """Apply crisp 4-sided grey box spines + inward major & minor ticks on all 4 sides."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(frame_color)
        sp.set_linewidth(frame_lw)
    if minor:
        ax.minorticks_on()
    ax.tick_params(
        axis="both", which="major", direction="in", top=True, right=True,
        length=5.0, width=frame_lw, color=frame_color, labelcolor=label_color, labelsize=9.5, pad=4,
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True,
        length=2.5, width=0.65, color=frame_color,
    )
    return ax


def style_axes(axs, **kw) -> None:
    """Apply `box_inward` to one Axes or any nested array/list of Axes."""
    for ax in np.ravel(axs):
        box_inward(ax, **kw)


def apply_style(mode: str = "daily", *, font_scale: float | None = None) -> None:
    """Configure matplotlib/seaborn globally:
      - `daily` (aliases: `talk`, `wada`, default): Seaborn `style='white', context='talk', font_scale=0.8`
        with grey (`0.55`) spines/ticks and crisp white bar edges — fast, clean exploratory style.
      - `paper` (aliases: `anbajagane`, `decade`): Serif + Computer Modern math (`cm`), grey (`0.55`)
        4-sided boxed frames with inward major & minor ticks on all 4 sides, capless thin errorbars.
    """
    key = mode.lower()
    if key in ("paper", "anbajagane", "decade"):
        plt.rcdefaults()
        mpl.rcParams.update({
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.85,
            "axes.edgecolor": FRAME,
            "axes.labelcolor": "0.15",
            "text.color": "0.15",
            "xtick.color": FRAME,
            "ytick.color": FRAME,
            "xtick.labelcolor": TICK_LABEL_COLOR,
            "ytick.labelcolor": TICK_LABEL_COLOR,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "xtick.minor.width": 0.65,
            "ytick.minor.width": 0.65,
            "patch.edgecolor": "white",
            "lines.linewidth": 1.45,
            "errorbar.capsize": 0.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 9.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 220,
        })
        return

    if key not in ("daily", "talk", "wada"):
        raise ValueError(f"Unknown style mode {mode!r}; expected 'daily'/'talk' or 'paper'/'anbajagane'.")

    try:
        import seaborn as sns
        sns.set_theme(style="white", context="talk", font_scale=0.8 if font_scale is None else font_scale)
    except ImportError:
        plt.rcdefaults()
    mpl.rcParams.update({
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": FRAME,
        "axes.labelcolor": "0.15",
        "text.color": "0.15",
        "xtick.color": FRAME,
        "ytick.color": FRAME,
        "xtick.labelcolor": TICK_LABEL_COLOR,
        "ytick.labelcolor": TICK_LABEL_COLOR,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "patch.edgecolor": "white",
        "errorbar.capsize": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 220,
        "legend.frameon": False,
    })


# ---------------------------------------------------------------------------
# Plot Primitives (Histograms, Scatter, Contours, Stacked Grids)
# ---------------------------------------------------------------------------

def scatter_style(n: int, *, role: str = "auto") -> dict:
    """Return marker kwargs tuned to N without oversized blobs or washed-out fills.

    - `open_ring` (default for comparison scatter like DECADE Fig. 3): hollow rings with crisp edges
    - `discrete` (N <= 250): crisp solid points with white rim
    - `cloud`: tiny low-alpha rasterized dots behind medians/contours
    - `c_mapped`: saturated colored points with hairline rim when mapped to `c=z`
    """
    if role == "open_ring":
        s = float(np.clip(280.0 / np.sqrt(max(n, 1)), 5.0, 20.0))
        alpha = float(np.clip(8.5 / np.sqrt(max(n, 1)), 0.25, 0.72))
        return dict(s=s, facecolors="none", alpha=alpha, linewidths=0.75, rasterized=n > 2500)
    if role == "discrete" or (role == "auto" and n <= 250):
        return dict(s=28, alpha=0.88, edgecolors="white", linewidths=0.7, rasterized=False)
    if role == "c_mapped":
        s = float(np.clip(440.0 / np.sqrt(max(n, 1)), 6.0, 24.0))
        alpha = float(np.clip(10.0 / np.sqrt(max(n, 1)), 0.38, 0.85))
        return dict(s=s, alpha=alpha, edgecolors="0.25", linewidths=0.35, rasterized=n > 5000)
    # cloud / dense auto
    if n <= 3000:
        s = float(np.clip(260.0 / np.sqrt(max(n, 1)), 5.0, 12.0))
        alpha = float(np.clip(6.5 / np.sqrt(max(n, 1)), 0.16, 0.36))
    else:
        s = float(np.clip(520.0 / np.sqrt(max(n, 1)), 2.2, 5.8))
        alpha = float(np.clip(26.0 / (max(n, 1) ** 0.56), 0.05, 0.18))
    return dict(s=s, alpha=alpha, edgecolors="none", linewidths=0.0, rasterized=n > 3000)


def plot_scatter_compare(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    label: str | None = None,
    open_rings: bool = True,
) -> None:
    """Plot comparison scatter in Anbajagane Fig. 3 style (hollow translucent rings or micro-dots)."""
    kw = scatter_style(len(x), role="open_ring" if open_rings else "cloud")
    if open_rings:
        ax.scatter(x, y, edgecolors=color, **kw)
        if label:
            ax.scatter([], [], s=24, facecolors="none", edgecolors=color, linewidths=1.0, label=label)
    else:
        ax.scatter(x, y, color=color, label=label, **kw)


def _reject_dead_pale(color: str) -> None:
    """Block lifeless chalky low-chroma histogram fills like #DCE3EC (keeps bright skyblue S=0.71)."""
    import colorsys
    r, g, b = mpl.colors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l > 0.72 and s < 0.50:
        raise ValueError(
            f"Refusing dead-pale histogram color {color!r} (L={l:.2f}, S={s:.2f}); "
            "use a saturated tone (#219ebc, skyblue/red, #ffba0a, #12354e) or crisp step outline."
        )


def plot_step_hist(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    bins,
    color: str = "#219ebc",
    filled: bool = True,
    alpha: float = 0.65,
    lw: float = 1.4,
    density: bool = True,
    label: str | None = None,
) -> None:
    """Plot rich filled primary histogram (alpha=0.65) or unfilled step comparison histogram (Figs. 3 & 7)."""
    if filled:
        _reject_dead_pale(color)
        ax.hist(data, bins=bins, density=density, histtype="stepfilled", color=color, alpha=alpha, edgecolor=color, linewidth=1.0, label=label)
    else:
        ax.hist(data, bins=bins, density=density, histtype="step", color=color, linewidth=lw, label=label)


def plot_kde_hist(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    color: str = "skyblue",
    label: str | None = None,
    stat: str = "density",
    alpha: float = 0.48,
    bins="auto",
    kde: bool = True,
    edgecolor: str = "white",
    linewidth: float = 0.75,
) -> None:
    """Seaborn histplot + KDE overlay (favorite pair: color='skyblue' vs color='red'; never black edges)."""
    import seaborn as sns
    _reject_dead_pale(color)
    sns.histplot(
        x=data, color=color, label=label, stat=stat, alpha=alpha,
        bins=bins, kde=kde, edgecolor=edgecolor, linewidth=linewidth, ax=ax,
    )


def plot_cloud_with_trend(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    bins: int | Sequence[float] = 9,
    marker: str = "o",
    label: str | None = None,
    logx: bool = False,
) -> None:
    """Plot muted background scatter + 16-84% band + crisp white-rimmed binned median."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & ((x > 0) if logx else True)
    x, y = x[ok], y[ok]

    kw = scatter_style(x.size, role="cloud")
    ax.scatter(x, y, color=color, zorder=2, **kw)

    edges = (
        np.geomspace(np.percentile(x, 2), np.percentile(x, 98), bins + 1)
        if isinstance(bins, int) and logx
        else (np.linspace(np.percentile(x, 2), np.percentile(x, 98), bins + 1) if isinstance(bins, int) else np.asarray(bins))
    )
    xc, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 5:
            xc.append(float(np.median(x[m])))
            lo.append(float(np.percentile(y[m], 16)))
            med.append(float(np.median(y[m])))
            hi.append(float(np.percentile(y[m], 84)))
    if xc:
        ax.fill_between(xc, lo, hi, color=color, alpha=0.18, lw=0, zorder=3)
        ax.plot(xc, med, f"{marker}-", color=color, lw=1.7, ms=4.8, mec="white", mew=0.8, zorder=5, label=label)


def plot_filled_contours(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    filled: bool = True,
    linestyle: str = "-",
    label: str | None = None,
    levels: Sequence[float] = (0.01, 0.05, 0.32),
) -> None:
    """Draw 68%, 95%, 99% KDE contours matching exact Anbajagane Fig. 5/6 alphas (0.35, 0.55, 0.75)."""
    import seaborn as sns
    lvls = sorted(levels)
    if filled:
        alphas = (0.35, 0.55, 0.75) if len(lvls) == 3 else np.linspace(0.35, 0.75, len(lvls))
        bounds = [*lvls, 1.0]
        for lo, hi, al in zip(bounds[:-1], bounds[1:], alphas):
            sns.kdeplot(x=x, y=y, levels=[lo, hi], fill=True, color=color, alpha=float(al), ax=ax)
    sns.kdeplot(x=x, y=y, levels=lvls, fill=False, color=color, linestyles=linestyle, linewidths=1.15, ax=ax)
    if label:
        ax.plot([], [], color=color, ls=linestyle, lw=1.4, label=label)


def save_fig(
    fig: plt.Figure,
    out_stem: str | Path,
    *,
    png: bool = True,
    pdf: bool = False,
    dpi: int = 220,
) -> list[Path]:
    """Save figure to PNG by default; write PDF only when `pdf=True`."""
    stem = Path(out_stem)
    if stem.suffix.lower() in (".png", ".pdf"):
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if png:
        p = stem.with_suffix(".png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    if pdf:
        p = stem.with_suffix(".pdf")
        fig.savefig(p, bbox_inches="tight")
        written.append(p)
    return written

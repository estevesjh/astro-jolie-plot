"""Regenerate every figure in docs/images from synthetic data.

    python examples/make_gallery.py

Needs numpy, matplotlib and seaborn. The palette strip uses `sanzo-wada` if installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "astro-jolie-plot" / "scripts"))
import esteves_plot as ep  # noqa: E402

OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 170
rng = np.random.default_rng(3)


def save(fig, name):
    ep.save_fig(fig, OUT / name, dpi=DPI)
    plt.close(fig)


# 1. Parameter ladder: effect off vs on, zero-gap stack, matched units -----------------------
ep.apply_style("paper")
r = np.logspace(-1, 1.2, 60)


def profile(f):
    base = 60 * (r / 0.5) ** -0.9 / (1 + (r / 4) ** 1.6)
    return base * (1 - 0.45 * f * np.exp(-0.5 * (np.log(r / 0.35) / 0.6) ** 2))


cols = ep.get_palette(4)
fig, (a, b) = plt.subplots(2, 1, figsize=(4.6, 5), sharex=True,
                           gridspec_kw=dict(hspace=0.0, height_ratios=[2.2, 1]))
for f, c in zip([0, 0.5, 0.75, 1.0], cols):
    a.plot(r, profile(f), color=c, lw=1.5, ls="-" if f else "--", label=rf"$f_{{\rm mis}}={f:g}$")
    b.plot(r, profile(f) - profile(0), color=c, lw=1.5, ls="-" if f else "--")
a.set_xscale("log"); a.set_yscale("log")
a.set_ylabel(r"$\Delta\Sigma\,[h M_\odot\,{\rm pc}^{-2}]$")
b.set_ylabel(r"$\Delta\Sigma-\Delta\Sigma_{f=0}$"); b.set_xlabel(r"$r_p\,[{\rm Mpc}/h]$")
a.legend(frameon=False, fontsize=9, loc="lower left")
ep.style_axes([a, b])
for x in (a, b):
    x.label_outer()
save(fig, "ladder")

# 2. Step histogram + open-ring scatter -------------------------------------------------------
ep.apply_style("paper")
fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw=dict(wspace=0.28))
x1, x2 = rng.normal(0, 1, 6000), rng.normal(0.15, 1.1, 6000)
ep.plot_step_hist(a, x1, bins=40, color="#219ebc", label="DECADE")
ep.plot_step_hist(a, x2, bins=40, color="#ffba0a", filled=False, label="DES")
a.set_xlabel(r"$\ln\lambda$"); a.set_ylabel("Density"); a.legend(frameon=False, fontsize=9)
xs = rng.normal(0, 1, 400); ys = xs + rng.normal(0, 0.35, 400)
ep.plot_scatter_compare(b, xs, ys, color="#219ebc")
lim = (-3.5, 3.5); b.plot(lim, lim, ls=":", color="0.3", lw=1); b.set_xlim(lim); b.set_ylim(lim)
b.set_xlabel(r"$\ln\lambda_{\rm DECADE}$"); b.set_ylabel(r"$\ln\lambda_{\rm DES}$")
ep.style_axes([a, b])
save(fig, "hist_scatter")

# 3. 68/95/99% contours -----------------------------------------------------------------------
ep.apply_style("paper")
fig, a = plt.subplots(figsize=(3.9, 3.7))
m = rng.multivariate_normal([0.3, 0.8], [[0.004, -0.0025], [-0.0025, 0.01]], 6000)
m2 = rng.multivariate_normal([0.31, 0.79], [[0.006, -0.003], [-0.003, 0.014]], 6000)
ep.plot_filled_contours(a, m[:, 0], m[:, 1], color="#e03424")
ep.plot_filled_contours(a, m2[:, 0], m2[:, 1], color="#006fed", filled=False, linestyle="--")
a.text(0.05, 0.93, "68%, 95%, 99%", transform=a.transAxes, fontsize=9, color="0.25")
a.set_xlabel(r"$\Omega_{\rm m}$"); a.set_ylabel(r"$S_8$")
ep.style_axes(a)
save(fig, "contours")

# 4. Daily mode, firebrick / black / grey -----------------------------------------------------
ep.apply_style("daily")
fig, a = plt.subplots(figsize=(5, 3.4))
rr = np.linspace(0.2, 10, 100)
for c, k, l in zip(["firebrick", "black", "grey"], [1, 0.8, 0.6], ["model", "data", "baseline"]):
    a.plot(rr, k * np.exp(-rr / 4) + 0.05 * k, color=c, lw=2, label=l)
a.set_xlabel("r"); a.set_ylabel("y"); a.legend(frameon=False)
sns.despine()
save(fig, "daily")

# 5. The colorbar bug a blind reader catches: white must sit at zero --------------------------
ep.apply_style("paper")
yy, xx = np.mgrid[-1:1:200j, -1:1:200j]
z = 0.97 * np.exp(-((xx - 0.3) ** 2 + (yy + 0.1) ** 2) / 0.18) - 0.31 * np.exp(-((xx + 0.5) ** 2 + (yy - 0.3) ** 2) / 0.12)
cmap = ep.get_cmap("RdBu_r")
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw=dict(wspace=0.5))
for ax, norm, title in [
    (axs[0], Normalize(vmin=-0.31, vmax=0.97), "Linear limits: white at +0.33"),
    (axs[1], TwoSlopeNorm(vmin=-0.31, vcenter=0.0, vmax=0.97), "TwoSlopeNorm: white at 0"),
]:
    im = ax.imshow(z, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax, pad=0.03).set_label("residual")
    ax.set_title(title, fontsize=10); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ep.box_inward(ax)
save(fig, "colorbar_midpoint")

# 6. Palette strip ----------------------------------------------------------------------------
ep.apply_style("daily")
rows = [("classic3", ep.get_palette(name="classic3")), ("hist2", ep.get_palette(name="hist2")),
        ("decade4", ep.get_palette(name="decade4")), ("decade5", ep.get_palette(name="decade5")),
        ("vol1-164", ep.get_palette(name="vol1-164")), ("vol1-2", ep.get_palette(name="vol1-2")),
        ("vol1-267", ep.get_palette(name="vol1-267")), ("vol1-331", ep.get_palette(name="vol1-331"))]
fig, ax = plt.subplots(figsize=(6, 0.42 * len(rows)))
for i, (name, cs) in enumerate(rows):
    for j, c in enumerate(cs):
        ax.add_patch(plt.Rectangle((j, -i), 0.92, 0.8, color=c, ec="none"))
    ax.text(-0.15, -i + 0.4, name, ha="right", va="center", fontsize=11)
ax.set_xlim(-0.2, 5.2); ax.set_ylim(-len(rows) + 0.7, 1); ax.axis("off")
save(fig, "palettes")
print("wrote", sorted(p.name for p in OUT.glob("*.png")))

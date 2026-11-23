---
name: plotting
description: A plotting design guide and toolkit following the Anbajagane et al. (2026, OJA/DECADE) publication style — serif Computer Modern typography, boxed 4-sided grey spines (0.55) with inward major/minor grey ticks, less-is-more information and layout restraint, zero-gap stacked grids, filled-vs-step and skyblue/red KDE histograms, 68/95/99% contours, open-ring or micro-dot scatters, Sanzo Wada favorites (vol1-164, vol1-2, vol1-331, vol1-267, vol1-323), and mandatory visual PNG QA.
---

# Anbajagane + Sanzo Wada Plotting Standard (`esteves_plot.py`)

Helper module installed inside this skill:
`scripts/esteves_plot.py`, next to this file (`${CLAUDE_PLUGIN_ROOT}/skills/plotting/scripts/` when installed as a plugin).
*(Needs `matplotlib`, `seaborn` and, for more than three categories, `sanzo-wada`: `pip install git+https://github.com/estevesjh/sanzo-wada-colors`.)*

---

## 0. Less Is More (Information & Layout Restraint)

- **Only strictly necessary information:** Never add an extra curve, point series, reference line, uncertainty band, inset, or callout annotation unless it is strictly needed for the figure's claim.
- **Avoid crowded multi-panel grids:** Do not default to many-column × many-row grids (`nrows >= 3` or `ncols >= 3` with dense subpanels). Prefer a **single focused panel** or compact **2–3 panel layouts** (`1×2`, `1×3`, or `2×1` stacked); split complex topics into separate clear figures unless a matrix/grid is explicitly requested.
- **Zero visual clutter:** No background gridlines, no legend/text boxes, and no decorative stats badges when the curve speaks for itself.

## 0.5. Scientific Contrast & 2-Second Eye-Test (Enhance the Differences)

> **Core Rule:** **The goal of a scientific plot is to enhance the differences between things.**

- **Plot Effect OFF vs. Effect ON (Controlled Contrast):** Whenever illustrating a systematic, physical mechanism, or model ingredient, ALWAYS show a clean ceteris-paribus contrast — **effect OFF vs. effect ON** (or a dialed parameter ladder, e.g., $f_{\mathrm{mis}} = 0, 0.50, 0.75, 1.00$) holding everything else fixed. **NEVER** bury a systematic inside an unsplit mixed sample average where most objects are unaffected or opposing effects cancel invisibly on top of each other.
- **Plot the Exact Modeled Equation (Not Plotting the Studied Variable Is a MAJOR Defect):** Take the exact physical quantity that appears in the model equation (e.g., $\lambda(R,0)$, $\Delta\Sigma(R)$, $n(\lambda_{\mathrm{ob}}, z)$) and plot it directly. **Not plotting the actual variable being studied is a MAJOR defect — only acceptable with a very good argument that explicitly enhances the differences between curves** (never invent ad-hoc rescalings or self-normalizations like dividing each curve by its own peak $\lambda(0,0)$).
- **One Physical Question per Panel (No Hijacking Curves):** Do not mix an unrelated survey, aperture, or data-vs.-mock offset onto a systematic-isolation panel where that unrelated gap dominates the reader's eye.
- **Matched Top/Bottom Units & Normalization:** Whenever a stacked residual or ratio sub-panel is paired with a top panel, **both panels MUST share the exact same modeled quantity, normalization, and units** (raw $y$ on top $\leftrightarrow$ raw $\Delta y$ below; OR normalized $y/y_0$ on top $\leftrightarrow$ fractional ratio $(y/y_0)/(y_{\mathrm{ref}}/y_{\mathrm{ref},0}) - 1$ below — **NEVER** self-normalized dimensionless top paired with raw physical-unit differences below).
- **2-Second Eye-Test:** Look at the rendered PNG without reading the caption: can a physicist immediately spot the difference between the curves in 2 seconds by eye? If the primary comparison curves visually collapse into one line, redesign the selection, quantity, or scale before handing the plot back.

---

## 1. Two Visual Modes (`daily` vs `paper`)

Pick the mode by context:

### Mode A: `daily` (`talk` — *default for notes, notebooks & diagnostics*)
- Call `ep.apply_style("daily")` (`sns.set_theme(style="white", context="talk", font_scale=0.8)`).
- Neutral grey (`0.55`) spines & ticks (or `sns.despine()` on simple standalone plots), white bar edges (`patch.edgecolor="white"`), `legend.frameon=False`.
- Ideal for quick exploratory checks, `vol1-164` / `vol1-2` / `["firebrick", "black", "grey"]` curves, and `["skyblue", "red"]` `kde=True` histograms.

### Mode B: `paper` (`anbajagane` — *use for paper figures & camera-ready panels*)
- Call `ep.apply_style("paper")` and `ep.box_inward(ax)` (or `ep.style_axes(axs)`).
- **Typography:** `font.family = "serif"`, `mathtext.fontset = "cm"`, `pdf.fonttype = 42`. Axis labels use clean math units like `$r_p\,[{\rm Mpc}/h]$`, `$\lambda\,[{\rm Richness}]$`, `$N_{\rm spec}/{\rm pix}$`.
- **Spines & Ticks (4-Sided Neutral Grey — NEVER harsh black against bright colors):**
  - Thin neutral grey border (`0.55`, `0.85` pt) on all 4 sides — **never despine**, and avoid pitch-black frames/tick marks or black histogram bar edges clashing with `skyblue`/`red`/DECADE hues.
  - **Inward major (`5.0` pt) + minor (`2.5` pt) grey ticks (`color="0.55"`) on top, bottom, left, right** with crisp charcoal tick labels (`labelcolor="0.18"`). Keep minor ticks visible on log axes too.
- **Line & Errorbar Discipline:**
  - Restrained curves (`lw=1.3–1.6`); dashed or dotted lines for comparison catalogs/models/references.
  - **Capless thin errorbars** (`errorbar.capsize=0`, `elinewidth=0.95`), with tiny horizontal jitter (~`2%`) when multi-bin points overlap.
- **Zero-Gap Stacked Panels & Legends:**
  - Stacked vertical subplots use `hspace=0.0` (sharing x-axes via `label_outer()`, avoiding colliding y-end ticks across shared boundaries).
  - Corner text inside panels (`SPT`, `z = 0.2`, `Fiducial`, `68%, 95%, 99%`, color-coded `p = 0.57` or `f_out = 0.12%`) with **no bounding boxes**, and `frameon=False` legends.

---

## 2. Color Sets (Matched to Plot Role & Curve Count)

Do **not** use Matplotlib `tab10` or arbitrary bright hexes. Use `esteves_plot.get_palette(...)`:

| Plot Role | Preset Name | Exact Colors / Hexes |
|---|---|---|
| **Model / Data or Mock / Baseline (traditional 2–3 curves)** | `classic3` / `traditional` *(or `role="comparison"`)* | `["firebrick", "black", "grey"]` |
| **Exploratory distribution pair (`histplot(..., kde=True)`)** | `hist2` / `skyblue_red` | `["skyblue", "red"]` |
| **Primary vs Secondary surveys (DECADE vs DES, contours, hists)** | `decade2` | `#219ebc` Teal, `#ffba0a` Marigold |
| **3 quantiles / terciles / colors** | `decade3` | `#f07167` Coral, `#219ebc` Teal, `#ffba0a` Marigold |
| **4-bin curves (richness, colors, $z$ bins)** | `decade4` | `#f07167` Coral, `#219ebc` Teal, `#ffba0a` Marigold, `#c767ac` Plum |
| **5-curve extended** | `decade5` | `decade4` + `#59a270` Sage Green |
| **3-bin discrete map / hist** | `decade_blues` | `#c6dbef`, `#6baed6`, `#2171b5` |
| **GetDist / posterior triangle contours** | `getdist2` | `#e03424` Brick Red (`#f3aea7`), `#006fed` Cobalt Steel (`#99c5f8`) |
| **Sanzo Wada favorites (when using Wada plates)** | `vol1-164` / `vol1-2` / `vol1-267` / `vol1-331` / `vol1-323` | `#dd4027,#fcb315,#4f4086` · `#12354e,#f99d1b` · `#006eb8,#00978d,#f99d1b,#fdbf68` · `#1e0e3f,#c53c69,#4f4086,#9a72aa` |

---

## 3. Plot Type Recipes (Scatter, Contours, Histograms, Colormaps)

- **Comparison Scatter (DECADE Fig. 3 style):** Use `ep.plot_scatter_compare(..., open_rings=True)` -> hollow rings or filled points at `alpha=0.30` so dense clouds show outlier tails without hiding the second sample, plus a thin dotted 1:1 diagonal (`ls=":", color="0.3"`). For huge clouds with trend lines, use `ep.plot_cloud_with_trend(...)` or `hexbin`.
- **Histograms (NEVER dead pale fills):** Never use lifeless chalky grey-blue fills like `#DCE3EC` or `lightsteelblue` (`plot_step_hist()` and `plot_kde_hist()` block low-chroma pale fills).
  - *DECADE comparison (Figs. 3 & 7):* saturated `#219ebc` `stepfilled` at `alpha=0.65` + unfilled `#ffba0a` `step` outline (`lw=1.4`) via `ep.plot_step_hist(...)`.
  - *Exploratory KDE histogram favorite:* `sns.histplot(..., color="skyblue", kde=True)` paired with `color="red"` via `ep.plot_kde_hist(...)` (`hist2`).
  - *Under fitted `["firebrick", "black", "grey"]` curves:* `#219ebc` (`alpha=0.55–0.65`), `vol1-2` `#12354e` (`alpha=0.45`), or crisp `step` / neutral `grey`.
- **2D Credible Contours (DECADE Figs. 5, 6, C1):** Primary = nested `68%, 95%, 99%` filled contours at `alpha=(0.35, 0.55, 0.75)`; secondary = dashed (`--`) unfilled contours via `ep.plot_filled_contours(...)`.
- **Colormaps:**
  - **Diverging correlation matrices, residuals & 2D sky density (PREFERRED):** Soft truncated **`RdBu_r`** (`[0.10, 0.90]` to avoid near-black saturated endpoints) or `decade_sky` (`#219ebc -> #ffffff -> #c767ac`). **ALWAYS anchor white at `0.0`** via `TwoSlopeNorm(vmin=min(vmin, -0.38*vmax), vcenter=0.0, vmax=vmax)` using `98%` bounds + slight headroom so peaks and dipole cores preserve internal gradients without flat saturation (never linear `Normalize` on asymmetric bounds).
  - **Single-survey footprints / sequential density:** `decade_teal` (`#ffffff -> #219ebc`), `decade_gold` (`#ffffff -> #ffba0a`), `decade_blues` (`#c6dbef -> #6baed6 -> #2171b5`), `wada323`, `rocket_r`
  - **Ordered `plt.plot` ladders (`N >= 5`) & cross-panel harmony:** Never sample `wada164_safe` for 1D `plt.plot` ladders (creates clashing violet-green-yellow-orange lines). Match the companion heatmap using `RdBu_r` pole colors (skipping white), monotonic `PuBu`/`Blues` `[0.38, 0.92]`, or unified `rocket` / `rocket_r`.
  - **`scatter(..., c=z)` on white:** `wada164_safe` (dense scatter color-mapping only, never line ladders)

---

## 4. Output & Mandatory Visual QA

1. **PNG first (`pdf=False` default):** Save `.png` at `220` dpi (`bbox_inches="tight"`); generate `.pdf` only on explicit request (`pdf=True`).
2. **Always view your rendered PNG with the Read tool** after running the script — first apply the **2-Second Eye-Test** (is the claimed systematic/contrast immediately visible by eye, with matched top/bottom normalization & units and no hijacking curves?), then check that only strictly necessary curves/points/panels appear (no crowded grids or clutter), verify grey frame ticks on all 4 sides, clean non-overlapping boundary ticks in `hspace=0.0` stacks, legend spacing, colorbar padding, and contrast. Fix and re-render before handing back.
3. **Blind review (recommended for any figure that goes in a paper):** run the `figure-reviewer` skill. A reader who sees only the pixels states what the figure shows, and you compare that with what you meant. Fix the figure until it passes.

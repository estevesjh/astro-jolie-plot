# plotting-skills

Two Claude skills for scientists who make figures:

- **`plotting`** sets the defaults: a restrained publication style, a short list of palettes, and one rule for what a plot is for.
- **`figure-reviewer`** checks the result. A model that sees only the pixels says what the figure shows, and you compare that with what you meant.

Both run on Anthropic Claude models.

## Why review a figure blind

We all check our figures. We reread the axes, fix the labels and confirm the units.

The trouble is that the person checking is the wrong instrument. You know what the figure is supposed to show, so your eye finds it even when the figure does not say it. A color with no legend reads as obvious, because you supply the meaning. A label borrowed from your code reads as clear, because you remember the variable.

Readers do not supply any of that. They give a figure a few seconds. If the message is not in the pixels, they leave with the wrong message or none, and the referee writes "unclear." You find out late, when a change is expensive.

So hand the figure to a reader who has only the pixels, and ask what it shows. Then compare the answer with what you meant. If the two differ, the figure is wrong, whatever you think of it.

### It is a blinded experiment

You would not let the experimenter who knows which sample was treated grade the result. A figure review needs the same discipline.

| In the lab | In the review |
|---|---|
| **Blind the observer.** | Reviewer 1 sees only the image, with metadata stripped, plus the caption if you give one. It never sees the manuscript, the project or the filename. |
| **Separate experimenter from grader.** | The plotting agent never certifies its own figure. A different reader looks at it, and you compare its one-sentence conclusion with your intent: `MATCH`, `PARTIAL_MISMATCH` or `MISMATCH`. |
| **Fix the pass criteria before you measure.** | The gate is written into the tool. To pass, the conclusion must match, the score must reach 80, no deterministic check may fire and no critical visual defect may remain. |
| **Calibrate with an instrument that does not drift.** | Zero-token checks give the same answer for the same figure. They flag jargon, banned labels, rainbow colormaps, saturation clipping and a misplaced colorbar midpoint. Results are cached on the stripped pixels and caption. |
| **Test the part before the system.** | Phase 2 stays locked until Phase 1 passes. The figure must survive alone before it is checked against the manuscript. |

### A bug you cannot see but a stranger can

Take a diverging colormap for a residual map. You choose limits of −0.31 to +0.97 and let matplotlib spread the colors linearly. White now sits at +0.33, not at zero.

You do not notice, because you know zero means no residual. A stranger reads the colorbar ticks and concludes that +0.33 is the null. The reviewer reads exactly those ticks, and a deterministic check on the midpoint arithmetic blocks the figure. The fix is one line: anchor white at zero with `TwoSlopeNorm`.

### A real run

This figure shows a parameter ladder: a profile for four values of one parameter, with the raw difference from the fiducial below. It was drawn with the `plotting` skill from synthetic data.

<img src="docs/images/ladder.png" alt="Stacked panels: profiles for four values of f_mis on top and the difference from f_mis = 0 below" width="420">

Reviewer 1 got only this image and the default model (`claude-sonnet-5-5`, through the `claude` CLI). It answered, in about 11 seconds:

> Increasing f_mis suppresses ΔΣ at r_p ≲ 1 Mpc/h (max deficit ≈ −43 near 0.25 Mpc/h) while leaving r_p ≳ 2 Mpc/h unchanged.

That is the message the figure was built to carry. Had the answer been different, the figure would have needed work.

### What it can and cannot tell you

- The reviewer is a language model. It stands in for a careful, unprimed reader and does not replace one. Show important figures to a human colleague too.
- It tests whether a figure communicates. It does not test whether the physics is right. A clear figure of a wrong result passes.
- Its verdict is only as useful as your stated intent. If you cannot write the intended conclusion in one sentence, the figure is not ready for any reviewer.

## Install

As a Claude Code plugin (skills and MCP server together):

```text
/plugin marketplace add estevesjh/plotting-skills
/plugin install plotting-skills@plotting-skills
```

The reviewer needs [`uv`](https://docs.astral.sh/uv/) to start its server, and either of:

- Claude Code installed and signed in (backend `claude_cli`, no API key), or
- `ANTHROPIC_API_KEY` set (backend `anthropic_api`).

For the plotting helper:

```bash
pip install matplotlib seaborn numpy
pip install git+https://github.com/estevesjh/sanzo-wada-colors   # only for 4+ categories
```

### Choosing models

| Variable | Effect | Default |
|---|---|---|
| `FIGURE_REVIEW_MODEL` | model for `speed="fast"` | `claude-sonnet-5-5` |
| `FIGURE_REVIEW_DEEP_MODEL` | model for `speed="deep"` | `claude-opus-5-5` |
| `FIGURE_REVIEW_BACKEND` | `claude_cli` or `anthropic_api` | first one available |

`claude-haiku-5-5` is a cheaper choice for quick triage.

## The plotting standard

The rule: **the goal of a scientific plot is to enhance the differences between things.**

- Show effect off against effect on, or a ladder of one parameter, with everything else fixed.
- Plot the quantity that appears in the equation, not a rescaled stand-in.
- One physical question per panel. Residual panels share the units of the panel above.
- Run the two-second eye test: can a physicist see the difference without reading the caption?
- Less is more: one panel, or two or three. No gridlines, no legend boxes.

Two modes: `daily` (seaborn `white` and `talk`, for notes) and `paper` (serif text, Computer Modern math, grey four-sided frame with inward ticks). Colors are `["firebrick", "black", "grey"]` for two or three categories and curated presets or [Sanzo Wada](https://github.com/estevesjh/sanzo-wada-colors) combinations beyond that. See [`skills/plotting/SKILL.md`](skills/plotting/SKILL.md) for every preset and recipe, and [`skills/plotting/scripts/esteves_plot.py`](skills/plotting/scripts/esteves_plot.py) for the helper.

```python
import sys; sys.path.insert(0, "skills/plotting/scripts")
import esteves_plot as ep

ep.apply_style("paper")
fig, ax = plt.subplots()
...
ep.style_axes(ax)
ep.save_fig(fig, "figure")
```

## Repository layout

```text
skills/plotting/           SKILL.md and the esteves_plot.py helper
skills/figure-reviewer/    SKILL.md for the blind review
mcp/figure-reviewer/       the MCP server (Python), Anthropic backends only
.mcp.json                  starts the server when installed as a plugin
.claude-plugin/            plugin and marketplace manifests
```

Run the server's tests with `uv run --directory mcp/figure-reviewer --with pytest --with pytest-asyncio pytest -m "not live"`.

## License

MIT

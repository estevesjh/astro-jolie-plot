# figure-reviewer MCP server

A blind, two-phase peer review for scientific figures, run by Anthropic Claude models.

The server hides everything about a figure except its pixels (metadata stripped, filename
hidden) and asks a model what the figure shows. A second phase, unlocked only after the first
passes, checks notation and narrative against your `.tex`. See the repository README for why
this design works.

## Backends

| Backend | Needs | Default model |
|---|---|---|
| `claude_cli` | Claude Code installed and signed in (`claude -p`) | `claude-sonnet-5-5` (fast), `claude-opus-5-5` (deep) |
| `anthropic_api` | `ANTHROPIC_API_KEY` | same |

`backend="auto"` prefers `claude_cli`. Environment overrides:

| Variable | Effect |
|---|---|
| `FIGURE_REVIEW_BACKEND` | `claude_cli` or `anthropic_api` |
| `FIGURE_REVIEW_MODEL` | model for `speed="fast"` |
| `FIGURE_REVIEW_DEEP_MODEL` | model for `speed="deep"` |
| `FIGURE_REVIEW_CLAUDE_MODEL`, `FIGURE_REVIEW_ANTHROPIC_MODEL` | per-backend model |

`claude-haiku-5-5` is a cheaper choice for fast triage.

## Run

```bash
uv run --directory mcp/figure-reviewer mcp-figure-reviewer     # MCP server (stdio)
uv run --directory mcp/figure-reviewer figure-review backends  # which backends are usable
```

Optional: set `ASTRO_MATH_REVIEW_SRC` or `ASTRO_MATH_REVIEW_BIN` to enable the vector label audit
from the companion astro-math-reviewer package. Without it, that one check is skipped.

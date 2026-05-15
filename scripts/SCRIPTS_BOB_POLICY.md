# `scripts/` Clean Code policy

## Scope

This document scopes how Uncle Bob's Clean Code 50-LOC guideline applies
inside `scripts/`. The library code (`src/`) and test suite (`tests/`)
follow the strict <50-LOC rule; `scripts/` is intentionally exempted as
described below.

## Exempted: figure-generation functions

Functions whose body is dominated by matplotlib axis configuration,
layout, color/style choices, and label placement are **exempted** from
the 50-LOC guideline.

Affected directories (non-exhaustive):

- `scripts/hero/`
- `scripts/comprehensive/`
- `scripts/reasoning_flow/`
- `scripts/graph_figures/`
- `scripts/attack_landscape/`
- `scripts/paper_figures/`
- `scripts/compare/`

### Rationale

1. **No visual regression coverage.** The figure scripts produce PNG/PDF
   artifacts inspected by humans. There is no automated check that
   verifies axis ticks, legend placement, color palette, or label
   positioning post-refactor.
2. **Bulk is configuration, not logic.** A 150-LOC `fig_*` function
   typically has <20 lines of data wrangling and >100 lines of
   matplotlib calls — each call configures one visual element
   (`ax.set_xlim`, `ax.spines[...]`, `ax.annotate`, etc.). Splitting
   these into helpers obscures the "what the figure looks like" intent
   without reducing logical complexity.
3. **One-shot scripts.** These functions run once per analysis pass to
   produce paper figures. They are not reused, composed, or extended.
   The 50-LOC guideline targets functions that are read, tested, and
   modified repeatedly; figure generators are write-once, regenerate-on-
   demand.
4. **Refactor risk asymmetry.** Refactoring a `fig_*` function carries
   real risk of silently breaking the figure (wrong axis, missing
   legend, swapped color mapping) with no automated detector. The
   reward — slightly shorter functions — does not justify the risk
   without visual-regression coverage.

### Conditions

- The function name MUST be a clear single-figure identifier
  (`fig_*`, `graph*`, `stat*`, etc.).
- The function MUST have one well-defined output (a saved figure or
  table).
- Data wrangling, statistics, and I/O helpers should still be extracted
  into module-level helpers under 50 LOC when feasible — the exemption
  covers the top-level figure-rendering function only.

## Not exempted

The following script categories follow the standard <50-LOC rule:

- `scripts/dataprep/` data-pipeline orchestrators (data integrity
  matters; refactor is testable via dataset checksums).
- `scripts/diagnostics/build_stats_table.py` table-building functions
  (output is structured data, comparable across runs).
- CLI dispatchers in `scripts/make/`, `scripts/cli.py`, and similar
  (interface layer, easy to test).

Current violations in these non-exempted categories will be addressed
incrementally as they are touched in regular work, rather than via a
blanket refactor pass.

## Future work

A visual-regression test harness (perceptual hash + tolerance threshold,
or pytest-mpl baselines) would unblock refactoring of the exempted
figure functions. Until that exists, the exemption stands.

## Precedent

This policy mirrors the in-line exemption pattern used for
`APGDAttack.run` (154 LOC, exempted via docstring rationale — preserving
byte-identical numeric output matching the Croce-Hein 2020 paper and
292 LOC of pinned regression fixtures).

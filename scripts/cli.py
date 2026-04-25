"""``adreason-figures`` console-script dispatcher.

Maps a subcommand to the matching ``scripts/make_*.py`` module and
re-executes it via ``runpy.run_module`` so the script's existing main()
or top-level ``if __name__ == "__main__"`` block runs unchanged.

Usage
-----
    adreason-figures <subcommand> [script-args...]

Subcommands
-----------
    hero | comprehensive | paper | attack-landscape | reasoning-flow
    graph | compare | figures
"""
from __future__ import annotations

import runpy
import sys

SUBCOMMANDS: dict[str, str] = {
    "hero":             "scripts.make_hero_figures",
    "comprehensive":    "scripts.make_comprehensive_figures",
    "paper":            "scripts.make_paper_figures",
    "attack-landscape": "scripts.make_attack_landscape",
    "reasoning-flow":   "scripts.make_reasoning_flow_figures",
    "graph":            "scripts.make_graph_figures",
    "compare":          "scripts.make_compare_figures",
    "figures":          "scripts.make_figures",
}


def _usage() -> str:
    cmds = "\n  ".join(sorted(SUBCOMMANDS))
    return (
        "usage: adreason-figures <subcommand> [script-args...]\n\n"
        "available subcommands:\n  " + cmds
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return 0 if args and args[0] in {"-h", "--help"} else 1

    sub = args[0]
    if sub not in SUBCOMMANDS:
        print(f"adreason-figures: unknown subcommand {sub!r}", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    # Forward remaining args to the target script via sys.argv.
    target_module = SUBCOMMANDS[sub]
    sys.argv = [target_module, *args[1:]]
    try:
        runpy.run_module(target_module, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared text-parsing helpers with zero in-package dependencies.

Lives at the package root so any submodule can import it without
introducing a cycle (``metrics`` → ``agents`` would be a regression).
"""

from __future__ import annotations


def find_balanced_close(text: str, start: int) -> int:
    """Return the index of the ``}`` that closes the ``{`` at ``text[start]``.

    Honours JSON string-escape rules so braces inside string literals do not
    affect nesting depth. Returns ``-1`` if no balanced close exists.
    """
    depth = 0
    in_string = False
    escaped = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1

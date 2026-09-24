"""The token-aware source census reader: source text with COMMENT/STRING/f-string-literal tokens removed.

A census over raw text flags a module's own prose, which is the false positive that teaches
people to word documents around a gate. Importable from any test directory through the root
conftest's own path.
"""
from __future__ import annotations

import tokenize
from pathlib import Path

#: FSTRING_MIDDLE is 3.12+ (PEP 701); on the 3.11 floor f-strings lex as STRING, and -1 matches no token type.
_FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", -1)


def code_text(path: Path) -> str:
    with path.open("rb") as handle:
        return "\n".join(
            tok.string for tok in tokenize.tokenize(handle.readline) if tok.type not in {
                tokenize.COMMENT, tokenize.STRING, _FSTRING_MIDDLE,
            }
        )

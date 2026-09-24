"""Encoding resolution authority: an absent declaration RAISES; there is no terminal default
(LAW-11). Imports stdlib only, below `mantis.encoding` in the DAG.
"""
from __future__ import annotations


class AbsentEncodingError(ValueError):
    """No encoding declared (LAW-11): there is no dense/v6 default."""


def reconcile_encoding(declared: str | None) -> str:
    """The declared encoding name. Raises: AbsentEncodingError (none declared)."""
    if not declared:
        raise AbsentEncodingError(
            "no encoding declared: the identity key is required (LAW-11 — there is no terminal "
            "'v6'/dense default). Declare identity.encoding explicitly."
        )
    return declared

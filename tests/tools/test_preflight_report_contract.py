"""The preflight_report contract's rc band is DERIVED from `RESERVED_CODES`, never transcribed (C-5)."""
from __future__ import annotations

import re
from pathlib import Path
from _toolpath import load_module_by_path

_REPO = Path(__file__).resolve().parents[2]
_DOC = _REPO / "docs" / "contracts" / "preflight_report.md"


def _parent_module():
    return load_module_by_path(
        "preflight_mint_parent", _REPO / "tools" / "ci_gates" / "preflight_mint_parent.py")


def test_the_doc_states_the_derived_band_and_no_stale_one() -> None:
    codes = _parent_module().RESERVED_CODES
    band = f"{min(codes)}–{max(codes)}"
    text = _DOC.read_text(encoding="utf-8")
    assert band in text, f"the contract must name the derived band {band}"
    stale = re.findall(r"\b4[0-9]–4[0-9]\b", text)
    assert set(stale) == {band}, f"a stale band literal survives in the contract: {stale}"


def test_the_doc_no_longer_pins_the_reversed_short_tier_claim() -> None:
    text = _DOC.read_text(encoding="utf-8")
    assert "can never be preflighted in the short tier" not in text
    assert "sync_lag" in text and "PreflightBurstTooShortError" in text

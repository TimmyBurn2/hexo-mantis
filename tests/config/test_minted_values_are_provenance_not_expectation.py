"""A minted value is pinned ONCE, with its provenance; everywhere else the RELATION is asserted.

AUDIT-1 F-49. run5's minted `eval.random_model_sims: 96` and `eval.sealbot_model_sims: 128` (the
latter deleted with the sealbot rung, R362(c)) were asserted as literals in four places — the regime-parity suite and its `_p2` twin, and both
`test_resolved_config_emit*` files — under docstrings that said DERIVED. Re-pointing
`production_config` at run6 would have reddened all four with "96 != N" and no line anywhere
saying 96 was run5's.

The one legitimate provenance pin survives with its grounds:
`tests/config/test_eval_config_remint.py::test_the_gate_parity_values_are_pinned`, which
asserts the numbers against every production config and cites the adjudication that chose them. Everywhere else now
asserts what the test is actually about — that the resolver hands back the shipped value.

WHAT THE AUDIT'S PIN ASKED FOR, AND WHAT IS EXECUTABLE. The audit named "a delta config with
`random_model_sims: 97` reds on the DIFFERENCE, not on '97 != 96'"; that is arm 1 below. Its
other half — "the mutated-TOML fixture with `board_size = 21` moves every repaired assertion and
leaves only the census red" — is NOT executable in this tree: `registry.toml` is compiled INTO
the extension at build time, so no test can load a mutated registry into live spec objects. The
executable equivalent is that every repaired assertion reads the spec, which is what the
repaired sites do and what arm 3 samples.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mantis.config.census import production_configs
from mantis.config.resolve.nsims import resolve_eval_model_sims
from mantis.encoding.registry import all_specs, lookup

#: Any census member: each arm needs a real mint as its base, never a particular one.
_VEHICLE = production_configs(Path(__file__).resolve().parents[2])[0]


def test_a_REMINTED_sims_value_passes_the_relation_that_a_literal_would_have_reddened(
    smoke_run_config,
):
    """Arm 1 — the audit's pin. A config that mints a DIFFERENT value is not a test failure."""
    remint = smoke_run_config(_VEHICLE, eval={"random_model_sims": 97})
    assert remint.eval.random_model_sims == 97
    value = remint.eval.random_model_sims
    assert resolve_eval_model_sims("random", value) == value, (
        "the random resolver did not hand back the re-minted value — under the literal "
        "form this test read `== 96` and would have failed here, reporting a MINT as a bug"
    )


def test_the_registry_owned_quantities_are_read_from_the_registry(smoke_run_config):
    """Arm 3 — a sample of the repaired class: geometry comes off the row, not off a literal.

    These are the quantities `crates/mantis-encoding/tests/registry_census.rs` is the ONE literal
    home for. A test that restates them is a second home that nothing reconciles, and eight
    assertions across `mantis-core` did exactly that until this packet.
    """
    from mantis import _engine

    for name in ("gnn_axis_v1", "gnn_axis_r8"):
        spec = lookup(name)
        assert _engine.Board.with_encoding_name(name).size == spec.board_size, name
        assert _engine.RegistrySpec.from_registry(name).policy_stride == spec.policy_logit_count
    cfg = smoke_run_config(_VEHICLE)
    assert cfg.identity.encoding in {s.name for s in all_specs()}


def test_the_one_provenance_pin_still_exists_and_still_names_its_grounds():
    """The counterpart to deleting three copies: the ONE pin that keeps 96/128 on the record
    must still be there. Deleting copies without this check would quietly lose the numbers."""
    from pathlib import Path

    source = (Path(__file__).resolve().parent / "test_eval_config_remint.py").read_text(
        encoding="utf-8"
    )
    assert "def test_the_gate_parity_values_are_pinned" in source, (
        "the ONE provenance pin for the minted gate sims is gone. The relation-form assertions "
        "elsewhere deliberately do NOT restate those numbers, so nothing else records them"
    )
    assert "adjudication A-3" in source, (
        "the provenance pin no longer cites its grounds, which is what makes it a provenance "
        "pin rather than a fourth copy of the literals"
    )

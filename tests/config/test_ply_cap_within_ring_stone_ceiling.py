"""R328 amendment — a configured ply cap may not exceed the HEXG ring's stone ceiling.

THE MEASURED INSTANCE. Encoding the R247 human corpus at radius 8 hit `MAX_STONES`: 88 of
8 698 games exceed 257 plies and 7 866 of 547 251 ply rows (1.4374 %) cannot be stored,
because `stones_qr` is a FIXED-WIDTH `[capacity * MAX_STONES * 2]` slot and
`push_graph_position` refuses anything wider. The architect ruled `MAX_STONES` STAYS 256 and
the corpus truncates with its loss counted — on the ground that the RUN's own games never
reach it, the ply-cap prereg row targeting ~256.

**THIS SUITE IS WHAT MAKES THAT GROUND ENFORCEABLE RATHER THAN ASSUMED.** A ground that holds
only while nobody preregs a bigger number is a hope. With the relation armed, a cap past the
ring is a config-load error at MINT, and raising the cap requires raising `MAX_STONES` first —
a mint-class change whose ring memory cost gets measured instead of assumed.
"""
from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from mantis._engine import max_stones


def test_the_ceiling_is_read_from_the_engine_and_is_not_typed_in_the_schema() -> None:
    """The relation's whole soundness rests on this: the number has ONE authority.

    `max_stones()` exists because the schema needed the ceiling and the alternative was a
    `256` typed beside a fixed-width Rust allocation — the one place a drift would be silent.
    Derived here too: the expectation is that the schema and the engine agree, not that either
    equals a literal this file also types."""
    import ast
    import inspect
    import textwrap

    from mantis.config.schema import core

    source = textwrap.dedent(
        inspect.getsource(core.RunConfig._ply_cap_within_the_rings_stone_ceiling)
    )
    fn = ast.parse(source).body[0]
    assert isinstance(fn, ast.FunctionDef)
    # The DOCSTRING is stripped before scanning. Prose explaining why the ceiling is 256 is
    # not a transcription of it — scanning the raw source made this row fire on its own
    # explanation, which would have trained the next reader to delete the explanation.
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    code = "\n".join(ast.unparse(node) for node in body)
    assert "max_stones()" in code, "the relation must call the engine getter"
    assert str(max_stones()) not in code, (
        f"the schema types the ceiling ({max_stones()}) instead of reading it; that is a "
        f"second authority over a fixed-width allocation.\n{code}"
    )


def test_a_cap_at_the_ceiling_validates(smoke_run_config) -> None:
    """The BOUNDARY, on the admitting side. A relation that refused its own ceiling would
    silently cost one ply and nothing would say so."""
    config = smoke_run_config("run5.yaml", selfplay={"max_game_moves": max_stones()})
    assert config.selfplay.max_game_moves == max_stones()


def test_a_cap_one_past_the_ceiling_REDS_AT_MINT(smoke_run_config) -> None:
    """The mutation pin, and the boundary on the refusing side."""
    with pytest.raises(ValidationError, match="exceeds the HEXG ring"):
        smoke_run_config("run5.yaml", selfplay={"max_game_moves": max_stones() + 1})


def test_the_refusal_names_both_operands_and_the_way_out(smoke_run_config) -> None:
    """A refusal that names neither number sends the reader to the wrong knob."""
    with pytest.raises(ValidationError) as excinfo:
        smoke_run_config("run5.yaml", selfplay={"max_game_moves": 4096})
    message = str(excinfo.value)
    assert "4096" in message and str(max_stones()) in message, (
        f"the refusal must name the configured cap AND the ceiling: {message!r}"
    )
    assert "MAX_STONES" in message, "the refusal must name the constant to raise"


def test_every_shipped_graph_config_is_inside_the_ceiling(smoke_run_config) -> None:
    """The census: arming this relation changes NOTHING for any config in the tree.

    A relation that refused a shipped config would be a regression, not a guard — this is the
    assertion that tells the two apart, and it is the same shape the value-pool census takes."""
    configs_dir = pathlib.Path(__file__).resolve().parents[2] / "configs"
    names = sorted(p.name for p in configs_dir.glob("*.yaml"))
    assert names, f"no shipped configs found under {configs_dir}; this census is vacuous"
    checked = 0
    for name in names:
        config = smoke_run_config(name)
        if config.identity.representation != "graph":
            continue
        checked += 1
        assert config.selfplay.max_game_moves <= max_stones(), (
            f"{name} mints max_game_moves={config.selfplay.max_game_moves} over the ceiling"
        )
    assert checked > 0, "no shipped graph config was reached; this census is vacuous"

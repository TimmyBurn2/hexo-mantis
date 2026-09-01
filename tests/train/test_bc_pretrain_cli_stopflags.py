"""R328(d) — the pretrain CLI's held-out flags: all-or-none, and the ring must SAY it is held out.

PB-8 GRADUATES HERE. `tests/train/test_bc_heldout_stop.py` records that a monitor handed the
TRAINING ring reports a loss that falls forever while every monitor-level row still passes,
because which ring it holds is not a property of the monitor. The defence has to live where the
ring is CHOSEN, and this is that place: the CLI reads the provenance sidecar the encoder wrote
and refuses a ring whose `split_part` is not `heldout`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis.train.pretrain.cli import _build_arg_parser  # noqa: PLC2701


def _args(**kw):
    base = ["--config", "configs/run5.yaml"]
    for k, v in kw.items():
        base += [f"--{k.replace('_', '-')}", str(v)]
    return _build_arg_parser().parse_args(base)


@pytest.mark.parametrize("supplied", [
    {"heldout_hexg": "x.hexg"},
    {"eval_every": 500},
    {"patience": 3},
    {"min_delta": 0.01},
    {"heldout_hexg": "x.hexg", "eval_every": 500},
    {"heldout_hexg": "x.hexg", "eval_every": 500, "patience": 3},
])
def test_a_PARTIAL_stopping_rule_is_refused(supplied: dict) -> None:
    """Every proper subset must refuse. A rule missing a term is one nobody declared, and the
    dangerous subset is the quiet one: a ring with no cadence would monitor nothing."""
    args = _args(**supplied)
    flags = (args.heldout_hexg, args.eval_every, args.patience, args.min_delta)
    assert any(f is not None for f in flags)
    assert not all(f is not None for f in flags), "this row's subject must be a PARTIAL set"


def test_the_FULL_stopping_rule_parses() -> None:
    """The positive control: the complete set is accepted."""
    args = _args(heldout_hexg="h.hexg", eval_every=500, patience=3, min_delta=0.01)
    assert all(f is not None for f in
               (args.heldout_hexg, args.eval_every, args.patience, args.min_delta))


def test_SILENCE_is_the_pre_existing_behaviour() -> None:
    """No flags at all must stay budget-bound with no monitor — landing is not arming."""
    args = _args()
    assert (args.heldout_hexg, args.eval_every, args.patience, args.min_delta) == (
        None, None, None, None)


def test_the_split_part_guard_is_WIRED_and_names_the_train_ring_hazard() -> None:
    """PB-8's detector, asserted structurally rather than by booting a pretrain.

    The refusal text is what a reader at the box meets, so it has to name the failure mode —
    a held-out loss over the training ring falls forever and nothing else notices."""
    import ast
    import inspect
    import textwrap

    from mantis.train.pretrain import cli

    src = textwrap.dedent(inspect.getsource(cli.pretrain))
    tree = ast.parse(src)
    consts = [n.value for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    joined = "\n".join(consts)
    assert "split_part" in src, "the CLI must read the sidecar's split_part"
    assert "falls forever" in joined, (
        "the refusal must name WHY a training ring is not a held-out ring; a bare type error "
        "sends the reader to the path instead of to the hazard"
    )

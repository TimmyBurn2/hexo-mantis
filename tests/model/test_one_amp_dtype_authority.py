"""AUDIT-1 F-31 — every `autocast` in `src/` names its dtype, and it comes from the authority.

THE DEFECT. `mantis.model.amp.amp_dtype_for` is the ONE dtype authority (LAW-06), honoured by
`Trainer.__init__` and `InferenceServer.__init__` (AUDIT-1 F-47 deleted the third,
`subsystems.cuda_warmup`, which had no caller at all). Four sites did
not honour it, and one of them mattered on a decision path:

  * `selfplay/inference_local.py::infer_batch` / `infer_batch_per_cluster` called
    `autocast(device_type=..., enabled=(cuda|mps))` with **no `dtype=` at all** — so the DENSE
    eval/arena forward ran at torch's device default whatever `train.amp_dtype` declared. That
    is the forward LAW-15 reads a deploy-matched promotion bar off.
  * the same module handed its graph server a literal `{"train": {"amp_dtype": "bf16"}}`.
  * `diagnostics/fusion_calibrate.py` autocast at a literal `torch.bfloat16` — right for the
    graph path by coincidence with LAW-06, which is exactly why it was invisible.

WHY A CENSUS AND NOT FOUR ROW-TESTS. The bf16 parity test pins the RESOLVER; it cannot see a
call site that never asks the resolver. That is how a forward with no dtype at all survived
beside a law about dtypes. This walks the AST instead.
"""
from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"
#: The authority itself, and the only file allowed to name a `torch.<dtype>` beside autocast.
_AUTHORITY = "model/amp.py"


def _autocast_calls() -> list[tuple[str, int, ast.Call]]:
    found: list[tuple[str, int, ast.Call]] = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = str(path.relative_to(_SRC))
        if rel == _AUTHORITY:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name) else None)
            if name == "autocast":
                found.append((rel, node.lineno, node))
    return found


def test_the_census_has_a_subject() -> None:
    """Vacuity guard. A census that reaches no `autocast` would pass forever — which is the
    failure mode of the instrument this file replaces."""
    calls = _autocast_calls()
    assert len(calls) >= 8, f"only {len(calls)} autocast call(s) found — the walk is broken"


def test_every_autocast_in_src_names_a_dtype() -> None:
    """The load-bearing row. An `autocast` with no `dtype=` runs at torch's device default,
    which is a dtype nobody declared, chosen by the hardware."""
    missing = [
        f"{rel}:{line}" for rel, line, call in _autocast_calls()
        if not any(kw.arg == "dtype" for kw in call.keywords)
    ]
    assert not missing, (
        f"autocast with no `dtype=`: {missing}. The dtype then comes from the device, not from "
        "`train.amp_dtype` — and on the eval/arena path that is the forward LAW-15 reads the "
        "promotion bar off (AUDIT-1 F-31)."
    )


def test_no_autocast_dtype_is_a_torch_literal() -> None:
    """`dtype=torch.bfloat16` at a call site is a second authority even when it agrees with
    LAW-06 today — agreement is the reason a copy survives, not a reason to keep it."""
    literals = []
    for rel, line, call in _autocast_calls():
        for kw in call.keywords:
            if kw.arg != "dtype":
                continue
            value = kw.value
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) \
                    and value.value.id == "torch":
                literals.append(f"{rel}:{line} -> torch.{value.attr}")
    assert not literals, (
        f"a torch dtype literal on an autocast: {literals}. `amp_dtype_for` owns the decision "
        "(LAW-06); a literal that is right today goes stale silently."
    )


def test_no_net_is_constructed_outside_the_one_builder() -> None:
    """F-31's second half. `tests/model/_bf16_parity.py` called `GnnNet(build_arch())` directly
    — the only direct net ctor outside `mantis.model` — so the parity net carried no `.arch`
    handle and was not the object production builds. `build_net` is the ONE authority."""
    offenders: list[str] = []
    for root in (_SRC, _SRC.parents[1] / "tests"):
        for path in sorted(root.rglob("*.py")):
            rel = str(path.relative_to(_SRC.parents[1]))
            if rel.startswith("src/mantis/model/"):
                continue  # the builder and the net definitions themselves
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id in {"GnnNet", "GnnNetV2", "HexTacToeNet"}:
                    offenders.append(f"{rel}:{node.lineno} -> {node.func.id}(...)")
    assert not offenders, (
        f"a net constructed outside `mantis.model.build_net`: {offenders}. `build_net` is what "
        "attaches the declared `.arch` handle every artifact writer reads (the "
        "arch-travels-with-the-model convention), so a direct ctor produces a net production "
        "cannot snapshot or stamp."
    )

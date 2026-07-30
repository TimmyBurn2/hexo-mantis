"""⊕ WPMAIN ORACLE — `train.device` is a CONFIG FACT, and the `--device` flag is dead
(R126 / DESIGN ADDENDUM C.1, oracles O-G1/O-G2/O-G3).

RED-at-import until IMPL lands `mantis.run.build_run_collaborators` and the
`train.device: Literal["cpu","cuda"]` schema field.

What this file exists to stop, and why the architect ruled it mint-critical:

At `b482243` the run device is a CLI-only input on both callers — `--device`, required, no
default, any torch device string. So `preflight_mint.py --config configs/run5.yaml --device
cpu` preflights a CUDA-minted run on the CPU. That is not a hypothetical: it is exactly the
wall the WPBOX burst hit (CARD-RUN5-GPU-OOM, a 16 GiB GPU OOM in GNN inference), and a
cpu-flagged preflight FALSE-CLEARS it. R126 grounds (a) names the corollary: an instrument
that can be pointed away from the failure it exists to find is not an instrument (LAW-03).

The three oracles:

- **O-G1** — the value reaches the REAL consumers. `torch.device(config.train.device)` is
  computed once in the builder and threaded into `init_trainer(...)` and `WorkerPool(...)`,
  which keep their `device` constructor parameters (collaborator threading BELOW the
  composition surface, not config-fact carriers). Registry row producer.
- **O-G2** — the SC-A pair: absent -> named raise; off-vocabulary -> named raise against a
  CLOSED `Literal`. `test_o16_all_fields_required_no_code_side_defaults` already covers
  `TrainConfig`, so the no-default half needs no edit (measured, C.1.1); what it does NOT
  give is either of these drives.
- **O-G3** — the flag cannot come back, on EITHER parser. This is the named
  equal-or-stronger successor for every `--device` census entry the R88 sweep drops
  (P-15/P-16), on the presence->ban pattern loop 1 already used for O-9's builder tokens.

Fakes: none. O-G1 drives the REAL builder (integration tier — it constructs a real net and a
real pool); O-G2/O-G3 are schema drives and source censuses.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

import mantis.run as mantis_run
from mantis.config.loader import load_config
from mantis.config.schema import RunConfig
from mantis.run import build_run_collaborators, launch_run  # RED-at-import anchor

_REPO = Path(__file__).resolve().parents[2]
_RUN_PY = _REPO / "src" / "mantis" / "run.py"
_TOOL_PY = _REPO / "tools" / "ci_gates" / "preflight_mint.py"
_CONFIGS = _REPO / "configs"

#: The closed vocabulary R126 writes, verbatim and in R126's own member order. The order is
#: deliberate and NOT reconciled with `eval.worker_device`'s `Literal["cuda","cpu"]`: member
#: order carries no validation semantics in pydantic, and reordering an untouched seam is
#: scope widening for zero behaviour (ADDENDUM C.1.1).
_DEVICE_VOCABULARY = ("cpu", "cuda")


def _dump(name: str = "smoke_gnn.yaml") -> dict:
    return load_config(_CONFIGS / name).model_dump()


def _string_constants(tree: ast.AST) -> set[str]:
    """Every string CONSTANT in code position — docstrings excluded, so prose that mentions
    a device does not trip a census about what the code hardcodes."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)):
                docstrings.add(id(body[0].value))
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings}


def _declared_options(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {arg.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}


# ══ O-G2 — the named-raise pair (SC-A, mirroring O-D4) ════════════════════════════════
def test_a_config_without_a_train_device_is_refused_by_name() -> None:
    """O-G2, arm 1 — R1/LAW-11 posture: absent is an ERROR naming the key, never a
    code-side "cpu".

    MUTATION THAT REDS IT: `device: str = "cpu"` on `TrainConfig`. A defaulted device is the
    posture-divergence hole one layer down from the flag: every config that forgets it
    silently preflights and runs on the CPU, and the GPU wall stays invisible."""
    payload = _dump()
    payload["train"].pop("device", None)
    with pytest.raises(ValidationError, match="device"):
        RunConfig.model_validate(payload)


@pytest.mark.parametrize("value", ["mps", "cuda:1", "CPU", ""])
def test_a_device_outside_the_closed_vocabulary_is_refused_naming_the_members(value: str) -> None:
    """O-G2, arm 2 — the vocabulary is CLOSED, and the refusal must teach it.

    `cuda:1` is the deliberate narrowing (ADDENDUM C.1.1): the dead flag accepted any torch
    device string, so a pinned multi-GPU index would have parsed and booted. It is now
    unrepresentable, matching `eval.worker_device`'s own closed vocabulary; widening the enum
    later is a named design act, not a config edit.

    MUTATION THAT REDS IT: declare the field as a bare `str`. Every existing config still
    validates, `test_o16` still passes (a `str` field with no default is required), and the
    first typo — `cude` — reaches `torch.device()` as a runtime error inside a boot instead
    of a validation error at load."""
    payload = _dump()
    payload["train"]["device"] = value
    with pytest.raises(ValidationError) as exc_info:
        RunConfig.model_validate(payload)
    message = str(exc_info.value)
    assert "device" in message
    for member in _DEVICE_VOCABULARY:
        assert member in message, (
            f"the refusal must render the CLOSED vocabulary so the operator can fix it "
            f"without reading the schema; {member!r} missing from {message!r}"
        )


@pytest.mark.parametrize("value", list(_DEVICE_VOCABULARY))
def test_both_vocabulary_members_validate(value: str) -> None:
    """O-G2, premise arm — neither member is accidentally unreachable (a `Literal["cpu"]`
    typo would pass every refusal assertion above)."""
    payload = _dump()
    payload["train"]["device"] = value
    assert RunConfig.model_validate(payload).train.device == value


# ══ O-G3 — no device ROUTE on either caller ═══════════════════════════════════════════
def test_neither_parser_declares_a_device_flavoured_option() -> None:
    """O-G3, arm 1 — the named successor for the dropped `--device` census entries (P-15).

    Both parsers are swept, because R126 kills the flag on BOTH callers and a ban on one
    side leaves the divergence route open on the other. The pattern is O-10's own
    (`no CLI switch may reach eval_enabled`), applied to the device fact.

    MUTATION THAT REDS IT: re-add `--device` — or `--torch-device`, or `--gpu`, which a
    literal `"--device" not in declared` check would wave through. The census is a substring
    match on the declared option strings, deliberately."""
    for path in (_RUN_PY, _TOOL_PY):
        declared = _declared_options(path)
        assert declared, f"premise: {path.name} declares CLI options at all"
        offenders = [option for option in declared if "device" in option.lower()]
        assert not offenders, (
            f"{path.relative_to(_REPO)} declares {offenders}: device is a CONFIG FACT "
            "(R126), and a flag beside it is the posture-divergence hole that false-cleared "
            "the WPBOX GPU wall"
        )
    tool_tree = ast.parse(_TOOL_PY.read_text(encoding="utf-8"))
    residue = sorted(value for value in _string_constants(tool_tree)
                     if "device" in value.lower() and (value.startswith("-") or value == "device"))
    assert not residue, (
        f"the tool still carries device CLI tokens {residue}: the `_child_argv` append and "
        "the `_require_preflight_args` row die with the flag (P-14), or the parent goes on "
        "passing an argument the child no longer accepts. Only FLAG-SHAPED literals are "
        "banned — prose and error text stay writable, because a census that makes people "
        "word their messages around it is a census that teaches the wrong lesson"
    )
    reads = [node for node in ast.walk(tool_tree)
             if isinstance(node, ast.Attribute) and node.attr == "device"
             and isinstance(node.value, ast.Name) and node.value.id == "args"]
    assert not reads, (
        "…and no `args.device` read survives: both of them (`:888`, `:913`) move into the "
        "builder as `config.train.device`"
    )


def test_no_composition_entry_point_declares_a_device_parameter() -> None:
    """O-G3, arm 2 — the parameter half. MF-1: no parameter carries a config fact, which is
    the same doctrine that deleted `eval_enabled` and `run_id` from `compose_run`.

    MUTATION THAT REDS IT: `build_run_collaborators(..., device: str = "cpu")` — a parameter
    default is a MIGRATED authority (MF-2 Attack B): every `dataclasses.fields`-style census
    stays green while a caller that omits the argument silently inherits a posture."""
    for function in (build_run_collaborators, launch_run, mantis_run.compose_run):
        parameters = list(inspect.signature(function).parameters)
        assert "device" not in parameters, (
            f"{function.__name__} declares a device parameter: the fact has exactly one "
            f"authority, the key (R126/MF-1); got {parameters}"
        )


def test_the_composition_root_hardcodes_no_device_string() -> None:
    """O-G3, arm 3 — the last route: not a flag, not a parameter, a LITERAL.

    MUTATION THAT REDS IT: `torch.device("cpu")` in the builder. The signature census sees
    nothing, the parser census sees nothing, every config still validates, and the run boots
    on the CPU whatever run5 says — the false-clear, one layer deeper. Docstrings are
    excluded from the scan on purpose: prose that DESCRIBES the vocabulary must stay
    writable (§C.1.4's amp discussion names both members)."""
    constants = _string_constants(ast.parse(_RUN_PY.read_text(encoding="utf-8")))
    offenders = constants & set(_DEVICE_VOCABULARY)
    assert not offenders, (
        f"src/mantis/run.py hardcodes {sorted(offenders)}: the device is read from "
        "`config.train.device` and from nowhere else (DESIGN §1.2 item 3)"
    )


# ══ O-G1 — the value reaches the real consumers ═══════════════════════════════════════
@pytest.mark.integration
def test_the_configs_device_reaches_the_real_trainer_and_the_real_pool(
    tmp_path, smoke_run_config
) -> None:
    """O-G1 — the registry row's named producer (R93: set the knob, observe the consumer).

    The REAL builder, the REAL `init_trainer` -> `build_net`, the REAL `WorkerPool`. Both
    consumers keep their own `device` constructor parameters, so the assertion is that the
    ONE computed `torch.device` object's value arrives at both — a builder that threads the
    config to the trainer and a literal to the pool is a run whose learner and actors sit on
    different devices, which is a class of failure that shows up as a mystery slowdown.

    MUTATION THAT REDS IT: hardcode either constructor's device. The static census above
    catches the literal spelling; this catches the value, including a transposition that
    passes the literal ban (e.g. threading `eval.worker_device` — the ADJACENT fact R126
    explicitly rules a DIFFERENT fact, so transcribing one into the other is the proxy
    inference the ruling refuses)."""
    config = smoke_run_config("smoke_gnn.yaml", train={"device": "cpu"})
    collab = build_run_collaborators(config=config, out_dir=tmp_path)
    assert collab.trainer.device == torch.device("cpu"), (
        f"the trainer must sit on the config's declared device; got {collab.trainer.device}"
    )
    assert collab.pool.device == torch.device("cpu"), (
        f"…and so must the self-play pool; got {collab.pool.device}"
    )


@pytest.mark.integration
def test_a_cuda_minted_config_never_silently_boots_on_the_cpu(tmp_path, smoke_run_config) -> None:
    """O-G1, the mutation direction CI can observe without a GPU — and the property R126
    grounds (a) actually demands.

    On a CUDA box: the declared `cuda` reaches both consumers. On a CUDA-less box (the CI
    tier, and this repo's pinned CPU torch wheel): the boot FAILS LOUD. Either way, what is
    unrepresentable is the third outcome — a cuda-declared config quietly running on the
    CPU, which is what a `--device cpu` preflight against a cuda run5 produced and what
    turned a 16 GiB OOM into a green.

    MUTATION THAT REDS IT: any fallback that coerces an unavailable device to `cpu`
    (`torch.device("cuda" if torch.cuda.is_available() else "cpu")` — the single most
    commonly written line in this class). It is invisible to every other oracle here."""
    config = smoke_run_config("smoke_gnn.yaml", train={"device": "cuda"})
    if torch.cuda.is_available():
        collab = build_run_collaborators(config=config, out_dir=tmp_path)
        assert collab.trainer.device.type == "cuda" and collab.pool.device.type == "cuda"
    else:
        with pytest.raises(Exception) as exc_info:  # noqa: B017 — torch's own class varies
            build_run_collaborators(config=config, out_dir=tmp_path)
        assert "cuda" in str(exc_info.value).lower() or "CUDA" in str(exc_info.value), (
            "the failure must name the device it could not honour, not surface as an "
            f"unrelated error; got {exc_info.value!r}"
        )

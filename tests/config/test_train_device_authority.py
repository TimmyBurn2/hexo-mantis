"""Pin `train.device` as a config fact with no CLI, parameter or literal route beside it.

A device flag lets a preflight be pointed at the CPU for a CUDA-minted run, which false-clears
the very GPU wall the preflight exists to find.
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
from mantis.run import build_run_collaborators, launch_run

_REPO = Path(__file__).resolve().parents[2]
_RUN_PY = _REPO / "src" / "mantis" / "run.py"
_TOOL_PY = _REPO / "tools" / "ci_gates" / "preflight_mint.py"
_CONFIGS = _REPO / "configs"

#: The closed vocabulary, verbatim. Member order carries no pydantic semantics, so it is
#: deliberately not reconciled with `eval.worker_device`'s opposite ordering.
_DEVICE_VOCABULARY = ("cpu", "cuda")


def _dump(name: str = "smoke_preflight_armed.yaml") -> dict:
    return load_config(_CONFIGS / name).model_dump()


def _string_constants(tree: ast.AST) -> set[str]:
    """Collect every string constant in code position, excluding docstrings."""
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


def test_a_config_without_a_train_device_is_refused_by_name() -> None:
    """Prove an absent `train.device` is an error naming the key, never a code-side "cpu".

    Killer: `device: str = "cpu"` on `TrainConfig`.
    """
    payload = _dump()
    payload["train"].pop("device", None)
    with pytest.raises(ValidationError, match="device"):
        RunConfig.model_validate(payload)


@pytest.mark.parametrize("value", ["mps", "cuda:1", "CPU", ""])
def test_a_device_outside_the_closed_vocabulary_is_refused_naming_the_members(value: str) -> None:
    """Prove an off-vocabulary device is refused with the closed vocabulary rendered.

    `cuda:1` is deliberately unrepresentable: widening the enum is a design act, not a config
    edit. Killer: declare the field as a bare `str` — every existing config still validates.
    """
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
    """Prove neither vocabulary member is accidentally unreachable."""
    payload = _dump()
    payload["train"]["device"] = value
    assert RunConfig.model_validate(payload).train.device == value


def test_neither_parser_declares_a_device_flavoured_option() -> None:
    """Prove neither parser declares a device-flavoured option.

    The census is a deliberate substring match, so `--torch-device` and `--gpu` are caught too;
    both parsers are swept because a ban on one leaves the route open on the other.
    """
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
    """Prove no composition entry point declares a device parameter.

    Killer: a defaulted `device` parameter — a migrated authority every field census misses.
    """
    for function in (build_run_collaborators, launch_run, mantis_run.compose_run):
        parameters = list(inspect.signature(function).parameters)
        assert "device" not in parameters, (
            f"{function.__name__} declares a device parameter: the fact has exactly one "
            f"authority, the key (R126/MF-1); got {parameters}"
        )


def test_the_composition_root_hardcodes_no_device_string() -> None:
    """Prove the composition root hardcodes no device string.

    Killer: `torch.device("cpu")` in the builder, which every other census here waves through.
    Docstrings are excluded so prose describing the vocabulary stays writable.
    """
    constants = _string_constants(ast.parse(_RUN_PY.read_text(encoding="utf-8")))
    offenders = constants & set(_DEVICE_VOCABULARY)
    assert not offenders, (
        f"src/mantis/run.py hardcodes {sorted(offenders)}: the device is read from "
        "`config.train.device` and from nowhere else (DESIGN §1.2 item 3)"
    )


@pytest.mark.integration
def test_the_configs_device_reaches_the_real_trainer_and_the_real_pool(
    tmp_path, smoke_run_config
) -> None:
    """Prove the config's device reaches the real trainer and the real pool.

    Killer: hardcode either constructor's device, or thread the adjacent `eval.worker_device`
    — both pass the static censuses above and split learner from actors.
    """
    config = smoke_run_config("smoke_preflight_armed.yaml", train={"device": "cpu"})
    collab = build_run_collaborators(config=config, out_dir=tmp_path)
    assert collab.trainer.device == torch.device("cpu"), (
        f"the trainer must sit on the config's declared device; got {collab.trainer.device}"
    )
    assert collab.pool.device == torch.device("cpu"), (
        f"…and so must the self-play pool; got {collab.pool.device}"
    )


@pytest.mark.integration
def test_a_cuda_minted_config_never_silently_boots_on_the_cpu(
    tmp_path, smoke_run_config, monkeypatch,
) -> None:
    """Prove a cuda-minted config either reaches cuda or fails loud, never boots on the cpu.

    Killer: any fallback coercing an unavailable device to `cpu`, invisible to every other
    oracle here.
    """
    config = smoke_run_config("smoke_preflight_armed.yaml", train={"device": "cuda"})
    if torch.cuda.is_available():
        # The allocator posture is another halt with its own oracle; supplied as a launch would.
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        collab = build_run_collaborators(config=config, out_dir=tmp_path)
        assert collab.trainer.device.type == "cuda" and collab.pool.device.type == "cuda"
    else:
        with pytest.raises(Exception) as exc_info:  # noqa: B017 — torch's own class varies
            build_run_collaborators(config=config, out_dir=tmp_path)
        assert "cuda" in str(exc_info.value).lower() or "CUDA" in str(exc_info.value), (
            "the failure must name the device it could not honour, not surface as an "
            f"unrelated error; got {exc_info.value!r}"
        )

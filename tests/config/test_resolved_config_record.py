"""R347 / CONFIG-1 — the run directory's COMPLETE resolved-config record.

WHY THIS FILE EXISTS. CONFIG-1 gave the operational constants schema defaults and took them
out of the YAML, so a shipped config no longer STATES every value its run uses. That is only
safe if the run writes its own complete account somewhere durable: without it, a schema
default revised next month silently rewrites what an old run is understood to have used, and
nothing in the tree could contradict it.

`mantis.config.emit.write_resolved_config` is that account, and `build_run_collaborators`
calls it FIRST — before the trainer, the buffer or the pool — so it exists even for a boot
that dies at a later seam. The rows below pin the three properties that make it a record
rather than a file: it is COMPLETE (every schema leaf, including the ones the shipped config
omits), it is STRICT (the bytes re-validate), and it is WRITTEN BY THE PRODUCTION BUILDER
(not only by a test calling the helper).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from mantis.config import load_config
from mantis.config.emit import RESOLVED_CONFIG_FILENAME, write_resolved_config
from mantis.config.schema import (
    ARCH_SCOPED_KEYS,
    OPERATIONAL_DEFAULT_KEYS,
    RunConfig,
    leaf_paths,
)
from mantis.eval.rounds import EVAL_CONCURRENCY_ROW
from mantis.model import ARCH_KIND_ROW
from mantis.train.warmstart import WARM_START_ROW

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"


def _leaves(node: object, prefix: str = "") -> set[str]:
    out: set[str] = set()
    for key, value in (node or {}).items():  # type: ignore[union-attr]
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= _leaves(value, f"{path}.")
        else:
            out.add(path)
    return out


#: Every key a shipped config may legally omit, from the DECLARED registries and the three
#: named optional rows — not from a second walk of `model_fields`, which would be a duplicate
#: schema walker (`tests/config/test_one_schema_leaf_walker.py` refuses those) and, worse, a
#: second answer to a question the registries already answer. That the registries and the
#: schema agree is `test_schema.py::test_o16_all_fields_required_no_code_side_defaults`'s job,
#: asserted there in both directions; this file consumes the answer.
OMITTABLE_KEYS: frozenset[str] = frozenset(
    {key for key, _grounds in OPERATIONAL_DEFAULT_KEYS}
    | {f"{key.section}.{key.field}" for key in ARCH_SCOPED_KEYS}
    | {ARCH_KIND_ROW, WARM_START_ROW, EVAL_CONCURRENCY_ROW}
)


def _under_an_omittable_key(leaf: str) -> bool:
    return leaf in OMITTABLE_KEYS or any(
        leaf.startswith(f"{key}.") for key in OMITTABLE_KEYS)


#: Distinguishes "absent" from a real `None`, which is a posture in this schema.
_ABSENT = object()


def _value_at(node: object, dotted: str) -> object:
    """The value at `dotted` in a nested mapping, or `_ABSENT` when the path does not exist."""
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _ABSENT
        node = node[part]
    return node


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_the_record_states_every_leaf_the_shipped_file_left_to_a_default(
    name: str, tmp_path: Path
) -> None:
    """The claim CONFIG-1 rests on, measured per shipped config: the record is a SUPERSET of
    the file, and the difference is exactly what the operational registry declared."""
    config = load_config(CONFIGS / name)
    written = _leaves(yaml.safe_load(
        write_resolved_config(config, tmp_path).read_text(encoding="utf-8")))
    shipped = _leaves(yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8")))
    assert shipped <= written, (
        f"{name}: the record LOST a key the shipped file states: {sorted(shipped - written)}"
    )
    # The difference may only be keys a registry DECLARES omittable.
    undeclared = {leaf for leaf in written - shipped if not _under_an_omittable_key(leaf)}
    assert not undeclared, (
        f"{name}: the record carries {sorted(undeclared)}, which the shipped file omits and "
        "no registry declares omittable — a value from nowhere"
    )


def test_the_record_is_complete_against_the_LIVE_SCHEMA_not_against_the_file(
    tmp_path: Path,
) -> None:
    """Superset-of-the-file would still pass on a record missing a key BOTH omit. The floor is
    the schema's own leaf set, derived by the same walker gate 13 uses."""
    config = load_config(CONFIGS / "run6.yaml")
    written = _leaves(yaml.safe_load(
        write_resolved_config(config, tmp_path).read_text(encoding="utf-8")))
    # A leaf inside an OPTIONAL BLOCK this config left DISARMED is absent by construction —
    # the block itself is present in the record, as an explicit null, which is the fact a
    # reader needs and is what the second assertion checks. Nothing else may be missing.
    record = yaml.safe_load(write_resolved_config(config, tmp_path).read_text(encoding="utf-8"))
    nulled = {leaf for leaf in written if _value_at(record, leaf) is None}
    missing = {leaf for leaf in leaf_paths(RunConfig)
               if leaf not in written and not any(leaf.startswith(f"{n}.") for n in nulled)}
    assert not missing, f"the record omits live schema leaves: {sorted(missing)}"
    assert nulled, (
        "run6 disarms at least one optional block, so an empty null set means the record is "
        "not carrying the disarmed postures as explicit nulls"
    )


def test_the_record_RE_VALIDATES_from_its_own_bytes(tmp_path: Path) -> None:
    """"Still strict" is a property of the FILE, not of the object it came from — so it is
    read back off disk and pushed through `RunConfig`, `extra="forbid"` and all."""
    config = load_config(CONFIGS / "run6.yaml")
    path = write_resolved_config(config, tmp_path)
    reloaded = RunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert reloaded == config, "the record does not reconstruct the config it was written from"


def test_a_record_that_would_not_validate_is_a_REFUSAL_and_not_a_written_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation self-test (LAW-07): drop the re-validation and this row goes green on a
    record nobody could load. A truncated dump must raise, not be written and forgotten."""
    config = load_config(CONFIGS / "run6.yaml")
    broken = config.model_dump()
    broken["train"].pop("lr")
    monkeypatch.setattr(type(config), "model_dump", lambda self, **_kw: broken, raising=False)
    with pytest.raises(Exception, match="lr"):
        write_resolved_config(config, tmp_path)


def test_the_PRODUCTION_builder_writes_the_record_before_any_collaborator() -> None:
    """A helper nothing calls is not a record. `build_run_collaborators` must call it, and it
    must call it BEFORE `init_trainer` — a boot that dies at the trainer is exactly the boot
    whose configuration a reader needs, and a record written afterwards would not exist."""
    tree = ast.parse((REPO_ROOT / "src" / "mantis" / "run.py").read_text(encoding="utf-8"))
    builder = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build_run_collaborators"
    )
    order = [
        node.func.id for node in ast.walk(builder)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"write_resolved_config", "init_trainer"}
    ]
    assert "write_resolved_config" in order, (
        "build_run_collaborators does not write the resolved-config record; CONFIG-1's YAML "
        "shrink then has no durable account behind it"
    )
    assert order.index("write_resolved_config") < order.index("init_trainer"), (
        f"the record must be written before the first collaborator; call order was {order}"
    )


def test_the_filename_is_one_authority() -> None:
    """The constant is what a reader greps for; a second spelling anywhere is a second file
    nobody finds."""
    assert RESOLVED_CONFIG_FILENAME.endswith(".yaml")
    run_py = (REPO_ROOT / "src" / "mantis" / "run.py").read_text(encoding="utf-8")
    assert RESOLVED_CONFIG_FILENAME not in run_py, (
        "run.py spells the record's filename itself instead of importing the constant"
    )

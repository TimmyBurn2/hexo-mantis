# R8 justify: one manifest CHECKER — loader, the producer-kind verifiers, the producer-test
# resolver and the deselection refusal are one decision procedure over one document, applied
# to every row in sequence; a verifier living apart from the loader could be handed a row
# shape the loader never produces.
"""Producer-manifest loader and checker: every gate/monitor input cites a live producer and test.

Resolution rules per producer kind:
  * `symbol`        — import the module, resolve the dotted attribute.
  * `event_literal` — the literal appears as a CODE string constant (never an identifier
    substring, never a docstring occurrence) in the named module's source.
  * `seam`          — the named attr resolves AND the row names the owing WP (`pending: <WP>`).
Every row's `producer_test` node (`<path>::<test_fn>`) must resolve to a real test function.

Import budget: stdlib + `yaml` at module scope; producer modules are imported LAZILY inside
the checker so the supervisor's package stays torch-free.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from mantis.util.yaml_io import DuplicateKeyError, parse_config_yaml

#: The manifest that ships beside this module — the ONE seam-7 instance.
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("producer_manifest.yaml")

_VALID_KINDS = ("symbol", "event_literal", "seam")


class ManifestError(RuntimeError):
    """A manifest row failed to resolve. The message ALWAYS names the row id."""

    def __init__(self, row_id: str, reason: str) -> None:
        super().__init__(f"producer_manifest row [{row_id}]: {reason}")
        self.row_id = row_id
        self.reason = reason


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Parse the manifest yaml (no resolution).

    Raises:
        ManifestError: the file is unreadable, not valid UTF-8, not well-formed YAML, carries
            a duplicate key, is not a mapping, or lacks a required top-level key.
    """
    target = Path(path)
    try:
        # `yaml.safe_load` is last-wins on duplicates; a duplicated row id would silently
        # drop the first row's producer while the manifest still parsed clean.
        raw: Any = parse_config_yaml(target)
    except (OSError, UnicodeDecodeError, DuplicateKeyError, yaml.YAMLError) as exc:
        raise ManifestError("<file>", f"unreadable manifest {target}: {exc!r}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("<file>", f"manifest {target} must be a mapping")
    doc = cast("dict[str, Any]", raw)
    for key in ("version", "channel", "gates"):
        if key not in doc:
            raise ManifestError("<file>", f"manifest {target} is missing the '{key}' key")
    return doc


def verify_manifest(path: Path | str, repo_root: Path | str) -> int:
    """Resolve EVERY row of the manifest at ``path``; return the number of rows checked.

    Raises `ManifestError` (naming the row) on the first unresolved producer or missing
    producer test. An EMPTY gate list is a failure, never a vacuous pass: a gate surface
    with zero producers is a phantom-armed abort chain waiting to happen (R4).
    """
    doc = load_manifest(path)
    root = Path(repo_root)
    gates_raw = doc.get("gates")
    if not isinstance(gates_raw, list) or not gates_raw:
        raise ManifestError("<gates>", "the manifest declares no gate rows (empty = FAIL)")
    gates = cast("list[Any]", gates_raw)
    seen: set[str] = set()
    for index, row_raw in enumerate(gates):
        if not isinstance(row_raw, dict):
            raise ManifestError(f"<row {index}>", "row must be a mapping")
        row = cast("dict[str, Any]", row_raw)
        row_id = str(row.get("id") or f"<row {index}>")
        if not row.get("id"):
            raise ManifestError(row_id, "row is missing its 'id'")
        if row_id in seen:
            raise ManifestError(row_id, "duplicate row id")
        seen.add(row_id)
        _verify_row(row_id, row, root)
    return len(gates)


def _verify_row(row_id: str, row: Mapping[str, Any], root: Path) -> None:
    producer_raw = row.get("producer")
    if not isinstance(producer_raw, dict):
        raise ManifestError(row_id, "row is missing its 'producer' mapping")
    producer = cast("dict[str, Any]", producer_raw)
    kind = str(producer.get("kind", ""))
    if kind not in _VALID_KINDS:
        raise ManifestError(row_id, f"unknown producer kind {kind!r} (expected {_VALID_KINDS})")

    if kind == "symbol":
        _verify_symbol(row_id, producer)
    elif kind == "event_literal":
        _verify_event_literal(row_id, producer)
    else:
        _verify_seam(row_id, row, producer)

    for extra_key in ("feeds_from", "also"):
        extra = row.get(extra_key, producer.get(extra_key))
        if extra is None:
            continue
        if isinstance(extra, dict):
            _verify_symbol(row_id, cast("dict[str, Any]", extra))
        else:
            _resolve_dotted(row_id, str(extra))

    _verify_producer_test(row_id, row, root)


def _verify_symbol(row_id: str, producer: Mapping[str, Any]) -> None:
    module = producer.get("module")
    symbol = producer.get("symbol")
    if not module or not symbol:
        raise ManifestError(row_id, "a symbol producer needs both 'module' and 'symbol'")
    _resolve_dotted(row_id, f"{module}.{symbol}")


def _verify_seam(row_id: str, row: Mapping[str, Any], producer: Mapping[str, Any]) -> None:
    _verify_symbol(row_id, producer)
    if "pending" not in row:
        raise ManifestError(
            row_id,
            "a seam row must declare 'pending: <WP>' — the WP that owes the concrete producer",
        )
    pending = row.get("pending")
    if not isinstance(pending, str) or not pending.strip():
        raise ManifestError(
            row_id, "'pending' must name the owning WP (a pending gate with no owner is banned)"
        )


def _string_constants(tree: ast.AST) -> set[str]:
    """Return every string constant in ``tree`` that is not a docstring.

    Docstrings are excluded structurally (first statement of a module, class or function
    body), not by heuristic: a documented producer is not a live one.
    """
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    return {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings
    }


def _verify_event_literal(row_id: str, producer: Mapping[str, Any]) -> None:
    module = producer.get("module")
    literal = producer.get("literal")
    if not module or not literal:
        raise ManifestError(row_id, "an event_literal producer needs 'module' and 'literal'")
    source = _module_source(row_id, str(module))
    try:
        tree = ast.parse(source, filename=str(module))
    except SyntaxError as exc:
        raise ManifestError(row_id, f"module {module} is unparseable: {exc!r}") from exc
    if str(literal) not in _string_constants(tree):
        raise ManifestError(
            row_id,
            f"the literal {literal!r} does not appear as a CODE string constant in {module} "
            "(an identifier substring never satisfies an event_literal row, and neither does "
            "an occurrence inside a docstring — a documented producer is not a live one)",
        )


def _verify_producer_test(row_id: str, row: Mapping[str, Any], root: Path) -> None:
    node = row.get("producer_test")
    if not node or "::" not in str(node):
        raise ManifestError(row_id, "row needs a 'producer_test' node id (<path>::<test_fn>)")
    rel, _, test_name = str(node).partition("::")
    # The node must be a COLLECTED test, not merely a function of that name in that file:
    # a `conftest.py::__init__` and a private helper both satisfy the weaker check.
    parts = Path(rel).parts
    if not parts or parts[0] != "tests" or not Path(rel).name.startswith("test_") \
            or Path(rel).suffix != ".py":
        raise ManifestError(
            row_id, f"producer_test path {rel!r} must be a collected test module "
                    "(tests/**/test_*.py)")
    if not test_name.startswith("test_"):
        raise ManifestError(
            row_id, f"producer_test node {test_name!r} must be a test function (test_*)")
    path = root / rel
    if not path.is_file():
        raise ManifestError(row_id, f"producer_test file {rel} does not exist")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ManifestError(row_id, f"producer_test file {rel} is unparseable: {exc!r}") from exc
    for candidate in ast.walk(tree):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and candidate.name == test_name:
            _refuse_deselected(row_id, rel, candidate, tree)
            return
    raise ManifestError(row_id, f"producer_test {test_name!r} not found in {rel}")


#: Markers that REMOVE a test from the default tier; a producer test that does not run is
#: the same standing as no producer test at all.
_DESELECTING_MARKERS = frozenset({"skip", "skipif", "slow", "integration"})


def _marker_names(decorators: list[ast.expr]) -> set[str]:
    """Return the `pytest.mark.<name>` names on a decorator list, called or bare."""
    names: set[str] = set()
    for node in decorators:
        target = node.func if isinstance(node, ast.Call) else node
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute) \
                and target.value.attr == "mark":
            names.add(target.attr)
    return names


def _refuse_deselected(row_id: str, rel: str, func: ast.FunctionDef | ast.AsyncFunctionDef,
                       tree: ast.AST) -> None:
    """A producer test the default tier does not collect proves nothing when it is not run."""
    found = _marker_names(func.decorator_list) & _DESELECTING_MARKERS
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) \
                else [node.value]
            found |= _marker_names(list(values)) & _DESELECTING_MARKERS
    if found:
        raise ManifestError(
            row_id,
            f"producer_test {rel}::{func.name} carries {sorted(found)} — a marker that "
            "DESELECTS it from the default tier. A producer test that does not run is the "
            "same evidence as no producer test at all (LAW-07)",
        )


def _resolve_dotted(row_id: str, target: str) -> Any:
    """Resolve ``pkg.mod.Attr.sub`` — the longest importable module prefix, then getattr."""
    parts = target.split(".")
    module = None
    consumed = 0
    for cut in range(len(parts), 0, -1):
        candidate = ".".join(parts[:cut])
        try:
            module = importlib.import_module(candidate)
        except ImportError as exc:
            # "No such module" and "module exists but fails to import" are different
            # defects; walking to a shorter prefix would report the latter as a rename.
            if _module_exists(candidate):
                raise ManifestError(
                    row_id,
                    f"module {candidate} exists but failed to import: {exc!r} — the producer "
                    "module is broken/unavailable, not renamed",
                ) from exc
            continue
        except Exception as exc:  # noqa: BLE001 — a producer module that explodes is a dead producer
            raise ManifestError(row_id, f"importing {candidate} failed: {exc!r}") from exc
        consumed = cut
        break
    if module is None:
        raise ManifestError(row_id, f"no importable module prefix of {target!r}")
    obj: Any = module
    for attr in parts[consumed:]:
        try:
            obj = getattr(obj, attr)
        except AttributeError as exc:
            # An instance counter assigned in `__init__` is a real producer that `getattr`
            # on the class cannot see — resolve it from the source instead.
            if isinstance(obj, type) and _class_assigns_attr(obj, attr):
                return None
            raise ManifestError(row_id, f"{target!r} does not resolve ({attr!r} missing)") from exc
    return obj


def _class_assigns_attr(owner: type, attr: str) -> bool:
    """True when ``owner``'s source assigns ``self.<attr>`` (an instance-attribute producer)."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(owner)))
    except (OSError, TypeError, SyntaxError):
        return False
    for node in ast.walk(tree):
        target: ast.expr | None = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        if (isinstance(target, ast.Attribute) and target.attr == attr
                and isinstance(target.value, ast.Name) and target.value.id == "self"):
            return True
    return False


def _module_exists(module: str) -> bool:
    """True when ``module`` is a real, findable module (regardless of whether it imports)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _module_source(row_id: str, module: str) -> str:
    """Return the source text of ``module`` without importing it."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError) as exc:
        raise ManifestError(row_id, f"module {module} is not importable: {exc!r}") from exc
    if spec is None or not spec.origin:
        raise ManifestError(row_id, f"module {module} has no source file")
    try:
        return Path(spec.origin).read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(row_id, f"cannot read the source of {module}: {exc!r}") from exc

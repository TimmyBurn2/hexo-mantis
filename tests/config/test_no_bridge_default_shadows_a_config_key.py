"""No bridge signature default stands behind a config key.

`build_runner_config` passes every kwarg, so the `#[pyo3(signature = ...)]` defaults
transcribed into `_engine.pyi` never fire today — but a kwarg added to the Rust ctor with a
default and forgotten in `build_runner_config` compiles and RUNS on the bridge's value.
The census is keyed on the SCHEMA rather than a hand-written knob list, so a key minted
tomorrow is covered the day it lands and a default that shadows nothing is untouched.
"""
from __future__ import annotations

import ast
from pathlib import Path

from mantis.config.schema import RunConfig, leaf_paths

_REPO = Path(__file__).resolve().parents[2]
_STUBS = (
    _REPO / "src" / "mantis" / "_engine.pyi",
    _REPO / "crates" / "mantis-bridge" / "python" / "mantis" / "_engine.pyi",
)


#: The defaults present when this census landed. NOT an allowlist that grows: the census
#: reds on anything NEW and also on a row here that no longer exists, so the register
#: cannot outlive the defaults it names. They live in the Rust `#[pyo3(signature = ...)]`
#: blocks, so removing them means rewriting ~30 Python construction sites to pass ~20
#: keywords each; a half-done removal is worse than none.
REGISTERED_DEBT: frozenset[str] = frozenset({
    "__init__(c_puct=1.5)",
    "__init__(c_scale=1.0)",
    "__init__(c_visit=50.0)",
    "__init__(dirichlet_alpha=0.3)",
    "__init__(dirichlet_enabled=True)",
    "__init__(dirichlet_epsilon=0.25)",
    "__init__(fast_prob=0.0)",
    "__init__(fast_sims=50)",
    "__init__(fpu_reduction=0.25)",
    "__init__(full_search_prob=0.0)",
    "__init__(gumbel_explore_moves=10)",
    "__init__(gumbel_m=16)",
    "__init__(leaf_batch_size=8)",
    "__init__(max_moves_per_game=128)",
    "__init__(n_sims_full=0)",
    "__init__(n_sims_quick=0)",
    "__init__(n_simulations=50)",
    "__init__(n_workers=4)",
    "__init__(quiescence_blend_2=0.3)",
    "__init__(quiescence_enabled=True)",
    "__init__(random_opening_plies=0)",
    "__init__(results_queue_cap=10000)",
    "__init__(standard_sims=0)",
    "__init__(temp_min=0.5)",
    "apply_dirichlet_to_root(epsilon=0.25)",
    "get_improved_policy(c_scale=1.0)",
    "get_improved_policy(c_visit=50.0)",
    "sample_graph_batch(augment=False)",
})


def _config_leaf_names() -> set[str]:
    """Every field NAME in `RunConfig`, unqualified — the vocabulary a bridge default could
    shadow.

    `descend_containers` reaches names inside a `list[SubModel]`, and the vocabulary is every
    SEGMENT of every reachable path, block names included, since a block name is a key too.
    """
    return {segment
            for path in leaf_paths(RunConfig, descend_containers=True)
            for segment in path.split(".")}


def test_the_vocabulary_is_not_empty() -> None:
    leaves = _config_leaf_names()
    assert len(leaves) > 100, f"only {len(leaves)} leaf name(s) — the walk is broken"
    for expected in ("n_workers", "max_game_moves", "draw_reward", "c_visit", "c_scale"):
        assert expected in leaves, f"{expected} missing — the census would not cover it"


def test_neither_engine_stub_defaults_a_parameter_that_shadows_a_config_key() -> None:
    """The load-bearing row. A `= <literal>` on a name the schema also owns is a second
    authority over one number, sitting where nobody reads it."""
    leaves = _config_leaf_names()
    #: `max_moves_per_game` is the bridge's spelling of `selfplay.max_game_moves`, renamed at
    #: `build_runner_config`, so the census must know both names.
    aliases = {"max_moves_per_game": "max_game_moves", "epsilon": "dirichlet_epsilon"}
    offenders: list[str] = []
    for stub in _STUBS:
        tree = ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            args = node.args
            positional = args.posonlyargs + args.args
            pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
            pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
            for arg, default in pairs:
                name = aliases.get(arg.arg, arg.arg)
                if name in leaves and isinstance(default, ast.Constant):
                    offenders.append(
                        f"{stub.relative_to(_REPO)}:{node.lineno} {node.name}({arg.arg}="
                        f"{default.value!r})"
                    )
    unregistered = sorted({o.split(" ", 1)[1] for o in offenders} - REGISTERED_DEBT)
    assert not unregistered, (
        "a NEW bridge signature default shadows a config key:\n  " + "\n  ".join(unregistered) +
        "\nThe schema owns these numbers. A default here is the authority the config is "
        "supposed to be, sitting on a path nobody reads until a kwarg is forgotten "
        "(AUDIT-1 F-39)."
    )
    stale = sorted(REGISTERED_DEBT - {o.split(" ", 1)[1] for o in offenders})
    assert not stale, (
        f"a registered debt row no longer exists: {stale}. Delete it — a register that keeps "
        "rows for defaults nobody has any more is how a debt list stops being read."
    )

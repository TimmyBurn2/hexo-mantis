"""Freeze Mctx's own outputs as the Gumbel parity oracle.

Mctx (`github.com/google-deepmind/mctx`) is the reference implementation of Danihelka et al.,
"Policy improvement by planning with Gumbel" (ICLR 2022). This script writes its answers to
``tests/fixtures/mctx_parity/mctx_parity_v1.json``, which the Rust side asserts against. Run it
ONCE: the fixture is the frozen artefact, not this script's availability.

**jax and mctx are deliberately NOT repo dependencies.** Mint in a throwaway environment::

    uv venv --python 3.11 /tmp/mctxenv
    uv pip install --python /tmp/mctxenv/bin/python mctx 'jax[cpu]'
    /tmp/mctxenv/bin/python tools/gen_mctx_parity_fixtures.py

Whole-search parity is not obtainable, so the oracle is taken at the five surfaces where both
implementations compute the same function of the same numbers. Priors are emitted TWICE because
Mctx carries LOGITS and softmaxes internally while this repo's nodes carry PROBABILITIES; every
action is legal, because this repo's tree has no illegal-action slots at all.

Raises:
    SystemExit: mctx or jax is not importable (exit 2), with the mint recipe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# `reportMissingImports` is suppressed AT THE SITE, not by excluding this file: mctx/jax are
# absent BY DESIGN, and a file-wide exclusion would stop checking everything else here too.
try:
    import jax  # pyright: ignore[reportMissingImports]
    import jax.numpy as jnp  # pyright: ignore[reportMissingImports]
    from mctx._src import (  # pyright: ignore[reportMissingImports]
        action_selection,
        qtransforms,
        seq_halving,
    )
    from mctx._src import tree as tree_lib  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover - the mint recipe is the message
    sys.stderr.write(
        f"{exc}\n\nmctx/jax are not repo dependencies. Mint the fixture in a "
        "throwaway environment:\n"
        "  uv venv --python 3.11 /tmp/mctxenv\n"
        "  uv pip install --python /tmp/mctxenv/bin/python mctx 'jax[cpu]'\n"
        "  /tmp/mctxenv/bin/python tools/gen_mctx_parity_fixtures.py\n"
    )
    raise SystemExit(2) from exc

OUT = Path(__file__).resolve().parents[1] / "tests/fixtures/mctx_parity/mctx_parity_v1.json"

#: Sim budgets from the packet's item 8 grid.
BUDGETS = (2, 8, 32, 50, 96)
#: Root candidate counts. 1 and 2 pin the degenerate ends, and 3/5/12 are here because
#: `selfplay.gumbel_m` is an unconstrained `int >= 1`: a power-of-two-only grid cannot tell
#: `ceil(log2(m))` from `trailing_zeros(m)`, and a planted break substituting one for the other
#: passed the whole grid until these three were added.
CONSIDERED = (1, 2, 3, 4, 5, 8, 12, 16)
#: Legal-action counts. 216 sits above the 192 child cap deliberately — the cap is what is
#: removed at the root.
LEGAL_COUNTS = (5, 50, 192, 216)
#: Mctx's own defaults for `qtransform_completed_by_mix_value`.
MAXVISIT_INIT = 50.0
VALUE_SCALE = 0.1


def _f32(x: Any) -> list[float]:
    """Return `x` as a flat list of Python floats at f32 precision."""
    return [float(v) for v in np.asarray(x, dtype=np.float32).reshape(-1)]


def _f32_or_null(x: Any) -> list[float | None]:
    """As `_f32`, but non-finite entries become JSON ``null``: `score_considered` scores an
    ineligible candidate at ``-inf``, and the Rust port models ineligibility as
    `Option::None`."""
    return [
        None if not np.isfinite(v) else float(v)
        for v in np.asarray(x, dtype=np.float32).reshape(-1)
    ]


def _root_tree(
    prior_logits: np.ndarray,
    qvalues: np.ndarray,
    visit_counts: np.ndarray,
    raw_value: float,
    node_value: float,
) -> tree_lib.Tree:
    """Build the minimal single-root `Tree` the q-transform reads; the arrays are rank-2/3 as
    `Tree` declares so `qvalues()`'s `chex.assert_rank` holds."""
    n_actions = prior_logits.shape[0]
    zeros_n = jnp.zeros((2,), dtype=jnp.float32)
    zeros_na = jnp.zeros((2, n_actions), dtype=jnp.float32)
    return tree_lib.Tree(
        node_visits=jnp.array([int(visit_counts.sum()), 0], dtype=jnp.int32),
        raw_values=jnp.array([raw_value, 0.0], dtype=jnp.float32),
        node_values=jnp.array([node_value, 0.0], dtype=jnp.float32),
        parents=jnp.array([-1, 0], dtype=jnp.int32),
        action_from_parent=jnp.array([-1, 0], dtype=jnp.int32),
        children_index=jnp.full((2, n_actions), -1, dtype=jnp.int32),
        children_prior_logits=jnp.stack(
            [jnp.asarray(prior_logits, dtype=jnp.float32), jnp.zeros((n_actions,))]
        ),
        children_visits=jnp.stack(
            [jnp.asarray(visit_counts, dtype=jnp.int32), jnp.zeros((n_actions,), jnp.int32)]
        ),
        children_rewards=zeros_na,
        children_discounts=zeros_na + 1.0,
        children_values=jnp.stack(
            [jnp.asarray(qvalues, dtype=jnp.float32), jnp.zeros((n_actions,))]
        ),
        embeddings=zeros_n,
        root_invalid_actions=jnp.zeros((n_actions,), dtype=jnp.float32),
        extra_data=None,
    )


def _visit_shapes(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """The four visit distributions each q-transform case is taken at: the degenerate root the
    mixed value exists for, the minimum at which `sum_probs` is a single prior, and two that
    bracket the realistic range."""
    concentrated = np.zeros(n, dtype=np.int32)
    concentrated[: min(4, n)] = np.array([25, 12, 8, 5], dtype=np.int32)[: min(4, n)]
    spread = rng.integers(0, 4, size=n).astype(np.int32)
    one = np.zeros(n, dtype=np.int32)
    one[n // 2] = 7
    return {
        "all_unvisited": np.zeros(n, dtype=np.int32),
        "one_visited": one,
        "concentrated": concentrated,
        "spread": spread,
    }


def build() -> dict[str, Any]:
    """Compute every parity case. Returns the JSON-ready fixture document."""
    rng = np.random.default_rng(20260909)
    doc: dict[str, Any] = {
        "schema": 1,
        "source": "google-deepmind/mctx",
        "maxvisit_init": MAXVISIT_INIT,
        "value_scale": VALUE_SCALE,
    }

    # (1) Sequential-Halving schedules. The LENGTH is the exact-budget property.
    doc["seq_halving"] = [
        {
            "max_num_considered_actions": m,
            "num_simulations": n,
            "sequence": list(seq_halving.get_sequence_of_considered_visits(m, n)),
        }
        for m in CONSIDERED
        for n in BUDGETS
    ]

    # (2)-(5) share one synthetic root per (legal count, visit shape).
    qt_cases: list[dict[str, Any]] = []
    for n in LEGAL_COUNTS:
        prior_logits = rng.normal(0.0, 1.5, size=n).astype(np.float32)
        prior_probs = np.asarray(jax.nn.softmax(jnp.asarray(prior_logits)), dtype=np.float32)
        qvalues = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
        raw_value = float(np.float32(rng.uniform(-0.9, 0.9)))
        gumbel = np.asarray(
            jax.random.gumbel(jax.random.PRNGKey(7 + n), (n,), dtype=jnp.float32),
            dtype=np.float32,
        )
        for shape_name, visits in _visit_shapes(rng, n).items():
            # `node_value` is set apart from `raw_value` ON PURPOSE: the completion reads the
            # RAW value, and a case where the two coincide could not tell the readings apart.
            node_value = float(np.float32(raw_value + 0.37))
            tree = _root_tree(prior_logits, qvalues, visits, raw_value, node_value)
            completed = qtransforms.qtransform_completed_by_mix_value(
                tree, tree.ROOT_INDEX, value_scale=VALUE_SCALE, maxvisit_init=MAXVISIT_INIT
            )
            considered_visit = int(visits.max())
            root_score = seq_halving.score_considered(
                jnp.asarray(considered_visit),
                jnp.asarray(gumbel),
                jnp.asarray(prior_logits),
                completed,
                jnp.asarray(visits),
            )
            interior = action_selection._prepare_argmax_input(  # noqa: SLF001
                probs=jax.nn.softmax(jnp.asarray(prior_logits) + completed),
                visit_counts=jnp.asarray(visits),
            )
            action_weights = jax.nn.softmax(jnp.asarray(prior_logits) + completed)
            qt_cases.append(
                {
                    "name": f"n{n}_{shape_name}",
                    "n_actions": n,
                    "prior_logits": _f32(prior_logits),
                    "prior_probs": _f32(prior_probs),
                    "qvalues": _f32(qvalues),
                    "visit_counts": [int(v) for v in visits],
                    "raw_value": raw_value,
                    "node_value": node_value,
                    "gumbel": _f32(gumbel),
                    "considered_visit": considered_visit,
                    "completed_qvalues": _f32(completed),
                    "root_score_considered": _f32_or_null(root_score),
                    "interior_argmax_input": _f32(interior),
                    "action_weights": _f32(action_weights),
                }
            )
    doc["qtransform"] = qt_cases
    return doc


def main() -> int:
    """Write the fixture. Returns the process exit code."""
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    sys.stdout.write(
        f"wrote {OUT} — {len(doc['seq_halving'])} schedules, {len(doc['qtransform'])} q cases\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

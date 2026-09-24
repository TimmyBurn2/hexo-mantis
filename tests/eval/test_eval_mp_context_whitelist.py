"""Only the `'spawn'` multiprocessing context is accepted, and the refusal carries the reason.

Two independent structural reasons: the eval worker needs its OWN CUDA context, which a forked
child inherits poisoned; and `_worker_entry` arms `PR_SET_PDEATHSIG`, which the kernel signals on
the death of the thread that CREATED the child — under `'forkserver'` that thread belongs to the
forkserver process, so the arming would track the wrong process and fire early or never.

THE REFUSAL KEYS OFF THE NAME STRING, never the context object: the eval suites monkeypatch
`multiprocessing.get_context`, so a check that inspected the returned context would red them all.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _pipeline_harness import eval_config, pipeline_kwargs

from mantis.eval.pipeline import build_eval_pipeline
from mantis.config.schema import EvalConfig


def _eval_cfg() -> EvalConfig:
    return eval_config(worker_kill_grace_sec=1.0)


def _pipeline_kwargs(tmp_path: Path, **overrides: Any) -> dict:
    return pipeline_kwargs(
        tmp_path, eval_cfg=_eval_cfg(), run_id="q3_mp_ctx_whitelist",
        drain_caps_sec=5.0, **overrides
    )


def test_a_forkserver_context_is_refused_at_construction(tmp_path) -> None:
    """`'forkserver'` is refused at CONSTRUCTION, not at first round, and with `ValueError` — a bad
    argument VALUE, deliberately distinct from the `RuntimeError` that reports a violated invariant."""
    with pytest.raises(ValueError, match="forkserver"):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx="forkserver"), leaf_batch_size=1)


def test_a_fork_context_is_refused_at_construction(tmp_path) -> None:
    """A mutant that special-cases only `'forkserver'` passes the row above and dies here, which
    forces the refusal to be a WHITELIST equality rather than a blacklist of known-bad names."""
    with pytest.raises(ValueError, match="fork"):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx="fork"), leaf_batch_size=1)


@pytest.mark.parametrize("bad", ["", "SPAWN", "spawn ", "threads"])
def test_any_other_context_name_is_refused_too(tmp_path, bad: str) -> None:
    """Empty string, wrong case, stray whitespace and an invented name are refused by the same
    equality, with no per-value arm to forget."""
    with pytest.raises(ValueError):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx=bad), leaf_batch_size=1)


def test_the_default_context_constructs(tmp_path) -> None:
    """The positive control, asserting past "no exception raised": a constructor that returned
    early would satisfy a bare does-not-raise check while breaking every round."""
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path), leaf_batch_size=1)
    try:
        assert pipeline._mp_ctx_name == "spawn"
        assert pipeline._work_dir.is_dir()
        assert pipeline._poller.is_alive(), "the constructed pipeline never started its poller"
    finally:
        pipeline.stop()


def test_the_refusal_names_the_only_supported_value(tmp_path) -> None:
    """The message must say what IS supported and why, not merely that the value was rejected."""
    with pytest.raises(ValueError) as caught:
        build_eval_pipeline(**_pipeline_kwargs(tmp_path, mp_ctx="fork"), leaf_batch_size=1)
    message = str(caught.value)
    assert "'spawn'" in message
    assert "PR_SET_PDEATHSIG" in message
    assert "CUDA" in message

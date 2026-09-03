"""The eval decode guard refuses BEFORE any model snapshot is loaded (LAW-07 producer).

WP12-R RED-TEAM finding D-2. `mantis.eval.errors.EvalDecodeUnsupportedError`'s docstring,
DESIGN §c.7 and PREREG all state that the guard fires "once per round, at spec-resolution
time, before any model is loaded" — and nothing in the tree tested it. RED-TEAM's mutation
M5 relocated `_assert_decode_implements_declared_pooling(enc_spec)` to below the first
`load_model_snapshot(...)` call in `run_round` and all 85 oracles stayed GREEN with gate 11
at rc 0. A stated property with no producer is exactly what LAW-07/R4 forbid.

The ordering is not stylistic. A refusal that happens AFTER a checkpoint has been
deserialised onto `spec.worker_device` has already paid that memory — on the box that
CARD-RUN5-GPU-OOM is about, "did we load a model before refusing" is a resource question.

Method: `mantis.eval.worker` binds `load_model_snapshot` at module scope, so the loader is
replaced with a recorder that RAISES a sentinel the moment it is reached. The refusal arm
then asserts the sentinel never fired and the recorder was never called; the control arm
asserts that the very same recorder IS reached for an admitted encoding, so the refusal
arm cannot be green because the loader was unreachable for some unrelated reason.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.eval import worker
from mantis.eval.errors import EvalDecodeUnsupportedError
from mantis.eval.rounds import GateSpec, RoundSpec, RungJob

_BOOK = "book_v1_s20260625_p4"
_SEED = 20260625


class _LoaderReached(RuntimeError):
    """Sentinel: `load_model_snapshot` was entered. Named, so no unrelated raise passes."""


def _caps_for(enc_name: str):
    """The fused-forward memory bound this encoding's route needs (F-816-10 D-1).

    Derived from the encoding, not chosen per call site: the graph route resolves the bound
    EAGERLY when its `InferenceServer` is constructed, and the grid route never reads it. The
    value is the template's NON-BINDING-BY-CONSTRUCTION pair, so no round here splits.
    """
    from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
    from mantis.encoding import lookup

    if lookup(enc_name).representation != "graph":
        return None
    return FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)


def _spec(tmp_path: Path, enc_name: str) -> RoundSpec:
    """A real `RoundSpec`. The snapshot paths deliberately do NOT exist — every arm below
    replaces the loader, and a round that reaches the filesystem has already lost."""
    gate = GateSpec(
        stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=2, opening_book=_BOOK,
        bootstrap_resamples=10, min_distinct_per_pair=1, seed_base=_SEED, run_gate=True,
    )
    rung_jobs = [
        RungJob(
            name="random_rung", bot="random", variant="raw", depth=None,
            opponent_sims=None, opening_book=_BOOK, deploy_matched=True, games=0,
        )
    ]
    return RoundSpec(
        leaf_batch_size=1, amp_dtype="bf16", leaf_build_threads=1,
        round_id=f"guard_order_{enc_name}", step=1,
        candidate_snapshot=str(tmp_path / "candidate.pt"),
        best_snapshot=str(tmp_path / "best.pt"), best_step=None,
        encoding=enc_name, worker_device="cpu", gate=gate, rung_jobs=rung_jobs,
        random_floor_games=0, random_model_sims=2, sealbot_model_sims=2,
        kraken_model_sims=2, strix_model_sims=2, seed_base=_SEED,
        round_timeout_sec=600.0, result_path=str(tmp_path / "result.json"),
        progress_path=str(tmp_path / "progress.txt"),
        ladder_bootstrap_resamples=10, ladder_bootstrap_ci_level=0.95,
        ladder_bootstrap_seed=1234,
        ply_cap_adjudication=None, strength_floor=None,
        fused_graph_caps=_caps_for(enc_name),
        inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10),
    )


def _explode_on_load(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the worker's bound loader with a recorder that raises when reached."""
    reached: list[str] = []

    def _recorder(path, device="cpu"):
        reached.append(str(path))
        raise _LoaderReached(f"load_model_snapshot reached for {path!r} on {device!r}")

    monkeypatch.setattr(worker, "load_model_snapshot", _recorder)
    return reached


def test_the_refusal_happens_before_any_snapshot_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The M5 producer: a refused encoding must never reach `load_model_snapshot`.

    RED under M5 (guard relocated below the first load): the loader is entered, so
    `_LoaderReached` propagates instead of `EvalDecodeUnsupportedError` and `reached` is
    non-empty — both assertions fail. GREEN as shipped.
    """
    reached = _explode_on_load(monkeypatch)

    with pytest.raises(EvalDecodeUnsupportedError) as excinfo:
        worker.run_round(_spec(tmp_path, "v6_live2_ls"))

    assert type(excinfo.value) is EvalDecodeUnsupportedError
    assert "legal_set_scatter_max" in str(excinfo.value)
    assert reached == [], (
        f"the guard refused only AFTER loading {reached}: a round whose encoding this "
        f"decode cannot honour must cost zero checkpoint deserialisations, on the "
        f"worker device, before it is refused."
    )


def test_an_admitted_encoding_does_reach_the_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the SAME recorder is reached for `v6`, so the arm above is not vacuous.

    Without this, "the loader was never called" would also be satisfied by a `run_round`
    that could not reach the loader at all, or by a patch that never took effect.
    """
    reached = _explode_on_load(monkeypatch)
    spec = _spec(tmp_path, "v6")

    with pytest.raises(_LoaderReached):
        worker.run_round(spec)

    assert reached == [spec.candidate_snapshot]

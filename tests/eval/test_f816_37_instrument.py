"""R339(c) — `F-816-37` as an INSTRUMENT: 1-in-1 on the eval path, and DUMP-ON-FIRE.

WHAT THE RULING CHANGES AND WHY. `F-816-37` is one run-fatal graph-contract failure on the eval
path — *"edge axis one-hot is not a clean one-hot (edge 803217): [0.0, 0.0, 0.5]"* — that has
never reproduced. Three diagnostics forcing the check on every collate found nothing across
~1 000 collates and a 90-minute burst produced zero occurrences, so hunting it further buys
nothing. What the run cannot afford is the NEXT occurrence taking the run down and leaving
nothing behind, and that is what these two properties fix: the eval path samples the check
1-in-1 rather than 1-in-64 (`canary_period = int(batch_size)` against
`inference.inference_batch_size: 64`), and a failure writes the offending batch before the
exception propagates.

WHY THE CORRUPTION IS REAL AND NOT A PATCHED RAISE. LAW-07 wants a producer test, and a test
that monkeypatches `collate_graph_batch` to raise would prove the dump code runs while proving
nothing about the CHECK — which is the half that has to notice. So the plant goes into the wire
slice, one edge's axis one-hot, exactly where the real failure's signature sits; the real Rust
`verify_edge_geometry` is what fires, and `EdgeAttrGeometryMismatch` is what propagates.

THE NEGATIVE CONTROL IS HALF THE FILE. A dump test that only ever runs a corrupted round would
be green against an instrument that dumps on every round, so the uncorrupted round is asserted
to produce NO dump and to pass.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.eval.rounds import GateSpec, RoundSpec, RungJob
from mantis.eval.snapshot import write_model_snapshot
from mantis.model import GnnArch, build_net

_ENC = "gnn_axis_v1"
_BOOK = "book_v1_s20260625_p4"
_SEED = 20260625


def _net(seed: int):
    spec = lookup(_ENC)
    torch.manual_seed(seed)
    arch = GnnArch(
        in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim, hidden=16, num_layers=2,
    )
    net = build_net(arch)
    net.arch = arch
    net.eval()
    return net


def _round_spec(tmp_path: Path) -> RoundSpec:
    candidate, best = tmp_path / "candidate.pt", tmp_path / "best.pt"
    write_model_snapshot(_net(seed=1), candidate)
    write_model_snapshot(_net(seed=2), best)
    gate = GateSpec(
        stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=2, opening_book=_BOOK,
        bootstrap_resamples=10, min_distinct_per_pair=1, seed_base=_SEED, run_gate=True,
    )
    return RoundSpec(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, amp_dtype="bf16", max_plies=24,
        leaf_build_threads=1, concurrency=1,
        round_id="f816_37_instrument", step=7, candidate_snapshot=str(candidate),
        best_snapshot=str(best), best_step=None, encoding=_ENC, worker_device="cpu",
        gate=gate,
        rung_jobs=[RungJob(name="random_rung", bot="random", variant="raw", depth=None,
                           opponent_sims=None, opening_book=_BOOK, deploy_matched=True,
                           games=0)],
        random_floor_games=0,
        random_model_sims=2, sealbot_model_sims=2, kraken_model_sims=2, strix_model_sims=2,
        seed_base=_SEED, round_timeout_sec=600.0,
        result_path=str(tmp_path / "result.json"),
        progress_path=str(tmp_path / "progress.txt"),
        ladder_bootstrap_resamples=10, ladder_bootstrap_ci_level=0.95,
        ladder_bootstrap_seed=1234,
        ply_cap_adjudication=None, strength_floor=None,
        fused_graph_caps=FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921),
        inference_batching=InferenceBatchingSpec(
            inference_batch_size=64, inference_max_wait_ms=10
        ),
    )


def _dumps(tmp_path: Path) -> list[Path]:
    return sorted(tmp_path.glob("collate_dump_*.json"))


# ── the rate ───────────────────────────────────────────────────────────────────────────
def test_every_collate_path_asks_for_one_in_one() -> None:
    """All three paths' postures, read off the call sites that state them.

    Structural rather than behavioural on purpose: the alternative is counting checks in a
    round, which measures the round's length as much as the posture. What R342(b)(i) rules is
    which VALUE each path passes, and that is what is asserted.

    THIS TEST INVERTED AT R342(b)(i). It used to assert that self-play kept the
    batch-size-derived rate; R339(c)'s ground for that was "a class that has only ever fired on
    eval", and `F-816-37` has since fired on the training path (R340 leg 3). The condition now
    is 1-in-1 on every path for the whole run, so the old assertion would pass only on a
    configuration the ruling forbids.
    """
    import inspect

    eval_src = inspect.getsource(worker)
    assert eval_src.count("collate_check_period=1") == 2, (
        "both eval engines (candidate and best-anchor) must ask for 1-in-1 — arming one "
        "leaves half the round's forwards sampled at the batch-size rate"
    )

    from mantis.selfplay import pool as sp_pool

    pool_src = inspect.getsource(sp_pool)
    assert "collate_check_period=1" in pool_src, (
        "R342(b)(i): the self-play InferenceServer must ask for 1-in-1 for the WHOLE run; "
        "at the batch-size-derived rate a corrupted batch had 63 chances in 64 of passing"
    )
    assert "collate_dump=_collate_dump_target(config)" in pool_src, (
        "R342(b)(i) is 1-in-1 AND dump-on-fire: a check that halts without the artifact is "
        "the exact failure R340 leg 3 recorded on the training path"
    )


def test_the_selfplay_dump_target_is_a_sibling_of_the_other_two() -> None:
    """The self-play dump lands beside the trainer's, under one run record.

    Pins the DERIVATION, not a literal path: all three paths deriving from the run's own
    checkpoint dir is what stops a second path authority appearing.
    """
    from mantis.selfplay.pool import _collate_dump_target

    dump_dir, context_fn = _collate_dump_target({"train": {"checkpoint_dir": "/run/xyz/checkpoints"}})
    assert dump_dir == "/run/xyz/collate_dumps", dump_dir
    ctx = context_fn()
    assert ctx["path"] == "selfplay" and ctx["concurrency"] == 1, ctx

    # A config with no checkpoint_dir must not raise INSIDE a dump target: the dump exists to
    # preserve evidence, so it degrades to a relative default rather than taking down the run.
    fallback, _ = _collate_dump_target({})
    assert fallback == "collate_dumps", fallback


def test_period_one_runs_the_semantic_layer_on_every_batch() -> None:
    """`_canary_should_run(1)` is True forever; `_canary_should_run(64)` is not.

    The mechanism the postures above select, pinned as a function so a change to the cadence
    rule cannot silently turn 1-in-1 into 1-in-something-else.
    """
    from mantis.selfplay import graph_collate as gc

    gc.reset_semantic_canary()
    assert [gc._canary_should_run(1) for _ in range(8)] == [True] * 8
    gc.reset_semantic_canary()
    fired = [gc._canary_should_run(64) for _ in range(65)]
    assert fired[0] is True and sum(fired) == 2, (
        f"1-in-64 should fire on the first batch and once more at 64; fired {sum(fired)}"
    )


# ── the negative control ───────────────────────────────────────────────────────────────
def test_a_clean_round_dumps_nothing(tmp_path: Path) -> None:
    """The control. Without it, an instrument that dumped unconditionally would look correct."""
    worker.run_round(_round_spec(tmp_path))
    assert _dumps(tmp_path) == [], (
        f"a clean round wrote a dump: {[p.name for p in _dumps(tmp_path)]}"
    )


# ── the planted corruption ─────────────────────────────────────────────────────────────
def _plant_one_hot_corruption(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Corrupt ONE edge's axis one-hot in the FIRST wire slice the server collates.

    The plant sits on `slice_graph_wire`'s output — the exact object the check reads — so the
    production `verify_edge_geometry` is what notices. `[0.0, 0.0, 0.5]` is F-816-37's own
    observed signature rather than an invented value: a half-written one-hot.
    """
    from mantis.selfplay import graph_wire_split

    state: dict[str, Any] = {"planted": 0}
    real = graph_wire_split.slice_graph_wire

    def _spy(payload, g0, g1):
        sub = real(payload, g0, g1)
        if state["planted"] == 0:
            attr = np.asarray(sub.edge_attr)
            if attr.size >= 5:
                attr[0:3] = [0.0, 0.0, 0.5]
                state["planted"] = 1
        return sub

    monkeypatch.setattr(graph_wire_split, "slice_graph_wire", _spy)
    return state


def test_a_planted_corruption_DUMPS_and_REDS(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LAW-07's mutation self-test for the instrument: the check notices, and the batch lands.

    Both halves are asserted because either alone is satisfiable by a broken instrument: a
    round that reds without dumping is HEAD's behaviour, and a dump without a red would be an
    instrument that swallowed a run-fatal defect.
    """
    state = _plant_one_hot_corruption(monkeypatch)
    # THE RED HALF, and its shape is the finding's own: the failure does not come back as a
    # `broken` result, it PROPAGATES out of the round. That is exactly what F-816-37 did in
    # production — the STEP 4c witness died at 14 min 50 s — and it is why an unbounded rate
    # on this path is a run-blocker rather than a reporting gap.
    with pytest.raises(ValueError, match="one-hot"):
        worker.run_round(_round_spec(tmp_path))
    assert state["planted"] == 1, "the corruption never reached a wire slice"

    dumps = _dumps(tmp_path)
    assert dumps, (
        "the round produced no dump — an occurrence that leaves nothing behind is exactly "
        "what R339(c) exists to end"
    )
    sidecar = json.loads(dumps[0].read_text(encoding="utf-8"))
    assert sidecar["finding"] == "F-816-37"
    assert sidecar["error_type"] == "EdgeAttrGeometryMismatch", sidecar["error_type"]
    assert "one-hot" in sidecar["error"], sidecar["error"]
    assert sidecar["round_id"] == "f816_37_instrument"
    assert sidecar["step"] == 7
    # THE HALT-CONDITION FIELD. R339(c) turns its arming halt on whether an occurrence
    # happened under concurrency, so a dump that could not answer that would not be the
    # instrument the ruling ordered.
    assert sidecar["concurrency"] == 1
    assert sidecar["phase"].startswith("gate_"), sidecar["phase"]

    saved = sorted(tmp_path.glob("collate_dump_*.npz"))
    assert saved, "the sidecar landed without its batch"
    assert sidecar["batch"] == saved[0].name, "the sidecar names a batch file that is not there"
    with np.load(saved[0]) as arrays:
        assert "edge_attr" in arrays and "edge_index" in arrays and "node_coords" in arrays, (
            f"the batch is missing the check's own inputs: {sorted(arrays.keys())}"
        )
        # The corruption itself is IN the saved batch — which is the whole point: a dump that
        # did not carry the offending bytes would be a timestamp with a message attached.
        attr = arrays["edge_attr"].reshape(-1, 5)
        assert list(attr[0][:3]) == [0.0, 0.0, 0.5], (
            f"the saved batch does not carry the planted edge: {attr[0][:3]}"
        )

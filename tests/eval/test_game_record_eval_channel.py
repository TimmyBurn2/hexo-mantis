"""⊕ R344(b) — the EVAL channels write their games, with the search stats the head already had.

The self-play half is pinned in `tests/monitor/test_game_record.py`. This is the other half,
driven through the REAL `worker.run_round`: a round plays a gate block and a random floor, and
every game of both must land in the store with the fields a viewer needs.

WHY THE STATS ROW IS HERE AND NOT IN THE UNIT TESTS. `DeployHeadPlayer.select_move` computes
`get_root_children_info()` and, before R344(b), threw it away one line before returning the
move. The claim "the eval channel's per-position stats need no engine change" is only worth
anything if a REAL round produces them, so this drives one rather than asserting the head in
isolation.
"""
from __future__ import annotations

from pathlib import Path

import torch

from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.eval.rounds import GameRecordTarget, GateSpec, RoundSpec
from mantis.eval.snapshot import write_model_snapshot
from mantis.model import CnnArch, build_net
from mantis.monitor.game_record import iter_run_games

#: A DENSE encoding at radius 8, not radius-5 `v6`: `book_v1_s20260625_p4` is minted
#: against `gnn_axis_v1` and 292 of its 512 openings need radius >= 6 to replay.
_ENC = "v6w25"
_BOOK = "book_v1_s20260625_p4"
_SEED = 20260625
_RUN_ID = "grec-eval"


def _net(seed: int):
    spec = lookup(_ENC)
    torch.manual_seed(seed)
    arch = CnnArch(board_size=spec.board_size, in_channels=spec.n_planes, filters=8,
                   res_blocks=1)
    net = build_net(arch)
    net.arch = arch
    net.eval()
    return net


def _round_spec(tmp_path: Path, target: GameRecordTarget | None) -> RoundSpec:
    candidate, best = tmp_path / "candidate.pt", tmp_path / "best.pt"
    write_model_snapshot(_net(seed=1), candidate)
    write_model_snapshot(_net(seed=2), best)
    gate = GateSpec(
        stride=1, screen_games=2, confirm_games=2, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=2, opening_book=_BOOK,
        bootstrap_resamples=10, min_distinct_per_pair=1, seed_base=_SEED, run_gate=True,
    )
    return RoundSpec(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, amp_dtype="bf16", max_plies=16,
        leaf_build_threads=1, concurrency=1,
        round_index=0, round_id="r000007_7000", step=7000, candidate_snapshot=str(candidate),
        best_snapshot=str(best), best_step=None, encoding=_ENC, worker_device="cpu",
        gate=gate, rung_jobs=[], random_floor_games=2,
        random_model_sims=2, sealbot_model_sims=2, kraken_model_sims=2, strix_model_sims=2,
        seed_base=_SEED, round_timeout_sec=600.0,
        result_path=str(tmp_path / "result.json"),
        progress_path=str(tmp_path / "progress.txt"),
        ladder_bootstrap_resamples=10, ladder_bootstrap_ci_level=0.95,
        ladder_bootstrap_seed=1234,
        game_record=target,
        ply_cap_adjudication=None, strength_floor=None, fused_graph_caps=None,
        inference_batching=InferenceBatchingSpec(inference_batch_size=64,
                                                 inference_max_wait_ms=10),
    )


def _play(tmp_path: Path) -> list[dict]:
    records_dir = tmp_path / "games"
    worker.run_round(_round_spec(
        tmp_path, GameRecordTarget(record_dir=str(records_dir), run_id=_RUN_ID)))
    return list(iter_run_games(records_dir, _RUN_ID))


def test_a_real_round_writes_every_game_it_played(tmp_path: Path) -> None:
    """Both armed blocks land, each on its own channel and phase.

    MUTATION THAT REDS IT: drop `games.sink(...)` from any one of the five fan-outs — the
    round still plays, still promotes or does not, and silently stops recording one block."""
    records = _play(tmp_path)
    assert records, "a round that played games wrote none of them"

    by_phase: dict[str, int] = {}
    for record in records:
        by_phase[record["phase"]] = by_phase.get(record["phase"], 0) + 1
    assert "gate_screen" in by_phase, f"the gate block did not record; saw {by_phase}"
    assert "random_floor" in by_phase, f"the floor block did not record; saw {by_phase}"

    channels = {r["phase"]: r["channel"] for r in records}
    assert channels["gate_screen"] == "promotion"
    assert channels["random_floor"] == "random_floor"


def test_every_eval_record_carries_what_a_viewer_needs(tmp_path: Path) -> None:
    """The field list R344(b) names, asserted per record rather than on one sample.

    `step_kind` is `round` here and `actor` on the self-play channel: the two channels
    attribute a game to a training step by different measurements, and a record that did not
    say which would invite a reader to plot them on one axis (LAW-03)."""
    for record in _play(tmp_path):
        assert record["run_id"] == _RUN_ID
        assert record["step"] == 7000 and record["step_kind"] == "round"
        assert record["game_id"].startswith("r000007_7000_"), record["game_id"]
        assert record["colors"]["candidate"] in (1, -1)
        assert record["colors"]["opponent"] == -record["colors"]["candidate"]
        assert record["result"] in ("p1", "p2", "draw")
        assert record["plies"] == len(record["moves"]), (
            "the ply count and the move list must agree — they are the same fact"
        )
        assert all(len(move) == 2 for move in record["moves"]), "moves are axial (q, r) pairs"
        assert record["served_sims"] >= 1
        assert isinstance(record["termination"], str) and record["termination"]


def test_the_gate_block_carries_PER_POSITION_SEARCH_STATS(tmp_path: Path) -> None:
    """The claim that made R344(b)'s eval half a wiring job: the deploy head already computes
    the visit distribution and the root value, and only ever discarded them.

    Also pins the two shape decisions a reader depends on.

    **`by` names the side that searched, and on THIS channel that is both of them.** The gate
    plays candidate net against ANCHOR net — two deploy heads — so its stats list covers every
    ply from both sides, while a rung or floor game (a plain bot opponent) carries only the
    candidate's. A list without `by` reads identically in the two cases, and a consumer that
    assumed one-sided would halve every per-move statistic it computed on the gate.

    **`visits` carries the SUPPORT** — visited children only — because a zero-visit child is
    part of the distribution, carries none of its information, and at radius 8 would be most of
    the bytes.

    MUTATION THAT REDS IT: stop stashing `last_root`, or capture it AFTER the argmax where it
    could describe a different search than the move beside it."""
    gate_games = [r for r in _play(tmp_path) if r["channel"] == "promotion"]
    assert gate_games, "no gate games to read stats from"

    with_stats = [r for r in gate_games if r.get("search_stats")]
    assert with_stats, "not one gate game carried search stats"

    for record in with_stats:
        stats = record["search_stats"]
        assert len(stats) <= record["plies"], "more search roots than plies were played"
        plies_seen = [entry["ply"] for entry in stats]
        assert plies_seen == sorted(plies_seen), "stats must be in ply order"
        assert len(set(plies_seen)) == len(plies_seen), "one root per ply, not two"
        assert {e["by"] for e in stats} <= {"candidate", "opponent"}, "unknown searcher"
        assert "opponent" in {e["by"] for e in stats}, (
            "the gate plays two deploy heads, so the anchor's roots must be here and LABELLED "
            "— an unlabelled two-sided list is the defect this field exists to prevent"
        )
        for entry in stats:
            assert -1.0 <= entry["root_value"] <= 1.0, entry["root_value"]
            for q, r, n in entry["visits"]:
                assert isinstance(q, int) and isinstance(r, int)
                assert n > 0, "only the SUPPORT is stored; a zero-visit row is dead weight"


def test_an_EMPTY_support_is_recorded_not_dropped(tmp_path: Path) -> None:
    """An entry whose support is empty is still a search, and it is kept.

    MEASURED, not hypothesised: a local boot of `configs/smoke_preflight_armed.yaml` — which
    mints `eval.gate.deploy_sims: 1` — produced 127 roots and **every one of them had an empty
    support**. At one simulation the root is expanded and nothing is backed up to a child, so
    there is no visited child to record. That is correct behaviour, and a first cut of this
    file asserted it could not happen.

    Dropping such an entry would be worse than keeping it twice over: `root_value` is real
    information, and `len(search_stats)` would stop counting the plies that were searched.

    MUTATION THAT REDS IT: skip the append when the support is empty."""
    records = [r for r in _play(tmp_path) if r.get("search_stats")]
    assert records, "no stats to check"
    for record in records:
        for entry in record["search_stats"]:
            assert "visits" in entry, "the field must be present even when the support is empty"
            assert "root_value" in entry, (
                "root_value is the information an empty-support entry still carries"
            )


def test_a_round_with_no_target_writes_nothing_and_does_not_raise(tmp_path: Path) -> None:
    """`game_record=None` is the no-op arm every test-built spec is in. It must be quiet, and
    it must not create the directory either — an empty `games/` beside a run would read as a
    run that recorded nothing, which is a different claim from a round that was never asked
    to."""
    worker.run_round(_round_spec(tmp_path, None))
    assert not (tmp_path / "games").exists()


def test_the_pipeline_ALWAYS_gives_its_rounds_a_record_target() -> None:
    """`game_record=None` is legitimate for a test-built spec and never for production.

    Nothing else pins that: every row above drives a spec this file constructed, so all of
    them would stay green with the pipeline handing its real rounds `None`. Structural over
    the AST rather than a substring search — `"game_record" in source` passes on a comment,
    and `game_record=None` in the production builder is exactly the defect.

    MUTATION THAT REDS IT: drop the kwarg, or pass `game_record=None`."""
    import ast
    import inspect

    import mantis.eval.pipeline as pipeline_module

    tree = ast.parse(inspect.getsource(pipeline_module))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "RoundSpec"
    ]
    assert len(calls) == 1, f"expected ONE RoundSpec construction, found {len(calls)}"
    bound = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "game_record" in bound, "production rounds are built with no record target"
    assert isinstance(bound["game_record"], ast.Call), (
        "the target must be a constructed GameRecordTarget, not a name that could be None"
    )
    assert getattr(bound["game_record"].func, "id", None) == "GameRecordTarget"


def test_an_unwritable_record_dir_does_NOT_break_the_round(tmp_path: Path, capsys) -> None:
    """The posture inversion between the two writers, pinned because it is easy to get
    backwards and expensive when it is.

    In `mantis.run` an un-openable store RAISES: the run has not started, and a run that
    cannot write its games should say so before it plays 25 000 of them. Inside a ROUND the
    calculus inverts — the round produces the promotion decision the run gates on, so killing
    it over an unwritable directory converts a lost record into a broken round, a skipped
    gate and an `eval_broken` an operator has to read.

    MUTATION THAT REDS IT: let the `GameRecordWriter` construction propagate out of
    `_RoundGameRecords.__init__`."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")   # mkdir will fail on this path

    result = worker.run_round(_round_spec(
        tmp_path, GameRecordTarget(record_dir=str(blocked / "games"), run_id=_RUN_ID)))

    assert result, "the round must still produce a result"
    assert "DISABLED" in capsys.readouterr().err, (
        "the loss must be reported once and loudly, not swallowed"
    )

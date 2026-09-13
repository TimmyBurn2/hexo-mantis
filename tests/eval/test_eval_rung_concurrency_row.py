"""`eval.rung_concurrency` — the RUNG block's own games-in-flight row (R351's 288-game
PUCT-512 point: serial, the rung is ≈ 3 h a round against a 3 600 s timeout).

The same idiom as `eval.concurrency` (R339(b)): `1` is byte-exact the serial arm that ran
before the row existed, an absent row IS a minted `1`, and the rung is the ONE reader.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from pydantic import ValidationError

from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.config.schema import RunConfig
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.eval.rounds import EVAL_RUNG_CONCURRENCY_ROW, GateSpec, RoundSpec, RungJob
from mantis.eval.snapshot import write_model_snapshot
from mantis.model import GnnArch, build_net

_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "configs" / "smoke_preflight_armed.yaml"
_ENC = "gnn_axis_v1"
_BOOK = "book_v1_s20260625_p4"
_SEED = 20260625


def _net(seed: int):
    spec = lookup(_ENC)
    torch.manual_seed(seed)
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)
    net = build_net(arch)
    net.arch = arch
    net.eval()
    return net


def _census(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int | None, bool]]:
    """One `(phase, concurrency-kwarg, factory-present)` row per `play_paired_match` call, the
    phase read off the record sink's closure (the same census `test_eval_concurrency_row` runs)."""
    rows: list[tuple[str, int | None, bool]] = []
    real = worker.play_paired_match

    def _phase(value: Any, depth: int = 0) -> str:
        if depth > 4:
            return "?"
        if isinstance(value, str):
            return value
        if isinstance(value, tuple):
            return next((f for f in (_phase(v, depth + 1) for v in value) if f != "?"), "?")
        for cell in getattr(value, "__closure__", None) or ():
            found = _phase(cell.cell_contents, depth + 1)
            if found != "?":
                return found
        return "?"

    def _spy(candidate, opponent, openings, **kwargs):
        rows.append((_phase(kwargs["record_sink"]), kwargs.get("concurrency"),
                     kwargs.get("player_factory") is not None))
        return real(candidate, opponent, openings, **kwargs)

    monkeypatch.setattr(worker, "play_paired_match", _spy)
    return rows


def _raw() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))


def test_the_row_name_is_the_key_path_the_schema_actually_carries() -> None:
    section, _, field = EVAL_RUNG_CONCURRENCY_ROW.partition(".")
    assert section == "eval"
    assert field in RunConfig.model_fields["eval"].annotation.model_fields


def test_an_absent_row_and_a_minted_one_are_the_SAME_config() -> None:
    raw = _raw()
    assert "rung_concurrency" not in raw["eval"], (
        f"{_CONFIG.name} has grown the key — this row's premise is that some committed config "
        "still omits it, so the default has a live subject")
    minted = copy.deepcopy(raw)
    minted["eval"]["rung_concurrency"] = 1
    assert RunConfig.model_validate(raw).eval.rung_concurrency == 1
    assert RunConfig.model_validate(raw).model_dump() == RunConfig.model_validate(minted).model_dump()


@pytest.mark.parametrize("bad", [0, -1])
def test_the_row_is_refused_below_one(bad: int) -> None:
    raw = _raw()
    raw["eval"]["rung_concurrency"] = bad
    with pytest.raises(ValidationError, match="rung_concurrency"):
        RunConfig.model_validate(raw)


def test_round_spec_requires_the_row() -> None:
    import dataclasses

    field = next(f for f in dataclasses.fields(RoundSpec) if f.name == "rung_concurrency")
    assert field.default is dataclasses.MISSING, "a spec silently carrying 1 is the disabled-knob class"


def _rung_spec(tmp_path: Path, rung_concurrency: int) -> RoundSpec:
    """A round with ONE `random` rung (no vendored bot needed) and the gate off."""
    candidate = tmp_path / "candidate.pt"
    write_model_snapshot(_net(seed=1), candidate)
    gate = GateSpec(
        stride=1, screen_games=0, confirm_games=0, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=2, opening_book=_BOOK,
        bootstrap_resamples=10, min_distinct_per_pair=1, seed_base=_SEED, run_gate=False,
    )
    rung = RungJob(name="r0", bot="random", variant="raw", depth=None, opponent_sims=None,
                   opening_book=_BOOK, deploy_matched=True, games=4)
    return RoundSpec(
        leaf_batch_size=1, c_visit=50.0, c_scale=1.0, q_rescale=True, search_kind="puct",
        gumbel_m=16, max_plies=32, leaf_build_threads=1, concurrency=1,
        rung_concurrency=rung_concurrency,
        round_index=0, round_id="rung_concurrency_wiring", step=1,
        candidate_snapshot=str(candidate), best_snapshot=None, best_step=None, encoding=_ENC,
        worker_device="cpu", gate=gate, rung_jobs=[rung], random_floor_games=0,
        random_model_sims=2, sealbot_model_sims=2, seed_base=_SEED, round_timeout_sec=600.0,
        result_path=str(tmp_path / "result.json"), progress_path=str(tmp_path / "progress.txt"),
        ladder_bootstrap_resamples=10, ladder_bootstrap_ci_level=0.95, ladder_bootstrap_seed=1234,
        game_record=None, ply_cap_adjudication=None, strength_floor=None,
        fused_graph_caps=FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921),
        inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10),
    )


@pytest.mark.parametrize("armed", [1, 2])
def test_the_rung_block_carries_the_row_with_a_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, armed: int
) -> None:
    """At `armed=2` the rung call must carry 2 with a `player_factory`; at 1 the factory is
    still handed over (byte-exact at G=1, never called)."""
    rows = _census(monkeypatch)
    worker.run_round(_rung_spec(tmp_path, armed))
    rung_rows = [r for r in rows if r[0] == "rung"]
    assert rung_rows, f"the rung never played; phases seen: {[r[0] for r in rows]}"
    for phase, conc, has_factory in rung_rows:
        assert conc == armed, f"{phase} played at concurrency={conc}, the spec said {armed}"
        assert has_factory, f"{phase} carries the row without a player_factory — G>1 would raise"

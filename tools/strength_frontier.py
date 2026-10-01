"""STRENGTH-FRONTIER-1 driver: frozen nets x search kind x sims, through the eval child.

>300 justify (R8): one instrument whose halves — cell vocabulary, snapshot builders, the RoundSpec
composition mirroring `mantis.run`'s eval seam, the parallel child runner, the pair-level readout
— are only checkable against each other; every game goes through `python -m mantis.eval.worker`.
A cell: `label`, `candidate` (a checkpoint path, `bc_full` = every head of the BC checkpoint, or
`bc_tp` = the BC net through the config's `identity.warm_start` seam), `search_kind`, `sims`, `games`,
`opponent` (`strix` at its own `strix_sims` — RUNG-2; `six` at `six_nodes` on the pinned network `six_net`;
or a snapshot source played through the GATE block; the sealbot cell went with the sealbot rung), `gumbel_m`, `c_scale`/`q_rescale` (the
deploy head's σ), `concurrency` (games in flight; 1 = the arena's serial loop), `opening_book` (a
manifest id) and `seed_base` — all the config's when absent; BOOK_V2's replays vary the last two. No
random floor, one rung, `round_index` 0; a refused floor probe is a FAILED cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from mantis import _engine as mantis_engine
from mantis.bots.six import CLOSE_LOG_MARKER as SIX_CLOSE_LOG_MARKER
from mantis.bots.six import FINDING_LOG_MARKER as SIX_FINDING_LOG_MARKER
from mantis.bots.six import PROVIDER_LOG_MARKER as SIX_PROVIDER_LOG_MARKER
from mantis.config.loader import load_config
from mantis.config.resolve.allocator_posture import declared_allocator_posture, governs_device
from mantis.config.resolve.eval_posture import resolve_ply_cap_adjudication, resolve_strength_floor
from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps
from mantis.config.resolve.inference_batching import resolve_inference_batching
from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads
from mantis.config.resolve.puct import resolve_puct_constants
from mantis.config.resolve.tactics import ARMS, arm_from_file, resolve_deploy_tactics
from mantis.encoding import lookup
from mantis.eval.aggregate import pair_bootstrap_wr_ci
from mantis.eval.rounds import GameRecordTarget, GateSpec, RoundSpec, RungJob
from mantis.eval.snapshot import write_model_snapshot
from mantis.model import arch_from_spec_and_config, build_net
from mantis.model.identity import net_param_hash
from mantis.monitor.game_record import iter_run_games
from mantis.train.checkpoints import deploy_state, load_checkpoint
from mantis.train.warmstart import apply_bc_warm_start, resolve_bc_warm_start
from mantis.util.determinism import seed_everything
from mantis.util.hashing import sha256_file

BC_FULL = "bc_full"
BC_TP = "bc_tp"
STRIX = "strix"
SIX = "six"
#: The opponents played through the RUNG block; anything else is a snapshot through the gate.
_RUNG_OPPONENTS = (STRIX, SIX)
_RUN_ID = "frontier1"
_REPO = Path(__file__).resolve().parents[1]
_BOOTSTRAP_RESAMPLES = 2000
_CI_LEVEL = 0.95
#: The rung block's own pair-bootstrap terms, the values every receipt on record was aggregated
#: under (`eval.ladder.bootstrap_*` until the block was deleted): the child's `rungs` CI at
#: 1000 resamples, this tool's `pair_readout` at 2000, both seeded here.
_RUNG_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_SEED = 1234


class FrontierCellError(ValueError):
    """A cell this driver cannot compose. Raised, never defaulted past."""


def _snapshot_from_checkpoint(path: Path, out: Path) -> dict[str, Any]:
    ck = load_checkpoint(path)
    if ck.metadata.arch is None:
        raise FrontierCellError(f"{path}: the stamp resolves no arch, so the net cannot be rebuilt")
    net = build_net(ck.metadata.arch)
    # The DEPLOY weights: the EMA shadow when the stamp carries one, else the learner's.
    state, weights = deploy_state(ck)
    net.load_state_dict(state)
    sha = write_model_snapshot(net, out)
    return {"source": str(path), "step": ck.metadata.step, "net_hash": net_param_hash(net),
            "tensors": len(state), "snapshot_sha256": sha, "weights": weights}


def _snapshot_bc_tp(config: Any, out: Path) -> dict[str, Any]:
    """The BC net through the CONFIG's seam (`reinit` decides the fresh heads)."""
    declared = resolve_bc_warm_start(config.model_dump())
    if declared is None:
        raise FrontierCellError("the config declares no identity.warm_start; bc_tp needs one")
    spec = lookup(config.identity.encoding)
    seed_everything(config.seed)
    net = build_net(arch_from_spec_and_config(spec, config.model_dump()))
    report = apply_bc_warm_start(net, declared, spec=spec)
    sha = write_model_snapshot(net, out)
    return {"source": str(declared.checkpoint), "seam": f"reinit={list(declared.reinit)}",
            "seed": config.seed, "loaded_keys": len(report["loaded_keys"]),
            "reinit_keys": report["reinit_keys"],
            "net_hash": net_param_hash(net), "snapshot_sha256": sha}


def _snapshot_bc_full(config: Any, out: Path) -> dict[str, Any]:
    declared = resolve_bc_warm_start(config.model_dump())
    if declared is None:
        raise FrontierCellError("the config declares no identity.warm_start; bc_full needs one")
    info = _snapshot_from_checkpoint(declared.checkpoint, out)
    if info["net_hash"] != declared.net_hash:
        raise FrontierCellError(
            f"{declared.checkpoint}: net hash {info['net_hash']} != declared {declared.net_hash}"
        )
    info["seam"] = "every head from the BC checkpoint"
    return info


def build_snapshot(source: str, config: Any, out: Path) -> dict[str, Any]:
    """Write the snapshot for `source` at `out`; return its provenance."""
    if source == BC_FULL:
        return _snapshot_bc_full(config, out)
    if source == BC_TP:
        return _snapshot_bc_tp(config, out)
    return _snapshot_from_checkpoint(Path(source), out)


def base_round_spec(config: Any, *, work_dir: Path) -> RoundSpec:
    """The run's eval-pipeline seam as `mantis.run` composes it; a cell replaces the varying fields."""
    dump = config.model_dump()
    cfg = config.eval
    graph = config.identity.representation == "graph"
    gate = GateSpec(
        stride=1, deploy_sims=cfg.gate.deploy_sims,
        opening_book=cfg.gate.opening_book, bootstrap_resamples=cfg.gate.bootstrap_resamples,
        min_distinct_per_pair=cfg.gate.min_distinct_per_pair, seed_base=cfg.gate.seed_base,
        run_gate=False, sequential=cfg.gate.sequential.model_dump(),
    )
    return RoundSpec(
        round_id="cell", round_index=0, step=0, candidate_snapshot="", best_snapshot=None,
        best_step=None, encoding=config.identity.encoding, worker_device=cfg.worker_device,
        gate=gate, rung_jobs=[], random_floor_games=0, random_model_sims=cfg.random_model_sims,
        seed_base=cfg.gate.seed_base,
        round_timeout_sec=cfg.round_timeout_sec, result_path="", progress_path="",
        game_record=GameRecordTarget(record_dir=str(work_dir / "games"), run_id=_RUN_ID),
        ply_cap_adjudication=resolve_ply_cap_adjudication(cfg),
        strength_floor=resolve_strength_floor(cfg),
        fused_graph_caps=resolve_fused_graph_caps(dump) if graph else None,
        leaf_batch_size=config.selfplay.leaf_batch_size,
        max_plies=config.eval.max_plies,
        c_visit=config.selfplay.c_visit, c_scale=config.selfplay.c_scale,
        q_rescale=config.selfplay.q_rescale,
        search_kind="", gumbel_m=config.selfplay.gumbel_m, tactics=resolve_deploy_tactics(dump),
        puct=resolve_puct_constants(config),
        inference_batching=resolve_inference_batching(dump) if graph else None,
        leaf_build_threads=resolve_leaf_build_threads(dump) if graph else 1,
        concurrency=1, rung_concurrency=1,
        allocator_posture=(declared_allocator_posture(dump)
                           if governs_device(cfg.worker_device) else None),
    )


def _strix_rung(config: Any, games: int, strix_sims: int, *, solver: bool = True,
                radius: int | None = None) -> RungJob:
    """The strix rung (RUNG-2) at `strix_sims` on the gate's book; `solver` False = `<stem>:net_only`, `radius` N = `<stem>:r<N>`."""
    from mantis.bots import strix as _strix

    pin = _strix._pin()
    if pin is None:
        raise FrontierCellError("vendor/pins.toml declares no [pins.hexo-strix]")
    stem = str(pin["checkpoint"]).rsplit(".", 1)[0]
    if radius is not None and not solver:
        raise FrontierCellError("a strix cell names strix_radius or strix_solver false, not both (no such variant)")
    variant = stem + (f"{_strix.RADIUS_SUFFIX}{int(radius)}" if radius is not None
                      else "" if solver else _strix.NET_ONLY_SUFFIX)
    return _rung_job(config, STRIX, variant, strix_sims, games)


def _rung_job(config: Any, bot: str, variant: str, opponent_sims: int, games: int) -> RungJob:
    """An external rung on the gate's book, at its own sims, under the tool's pinned pair-bootstrap terms."""
    return RungJob(name=bot, bot=bot, variant=variant, opponent_sims=opponent_sims,
                   opening_book=config.eval.gate.opening_book, deploy_matched=True, games=games,
                   bootstrap_resamples=_RUNG_BOOTSTRAP_RESAMPLES, bootstrap_ci_level=_CI_LEVEL,
                   bootstrap_seed=_BOOTSTRAP_SEED)


def _rung_on_cell_book(job: RungJob, cell: Mapping[str, Any]) -> RungJob:
    """The rung on the cell's `opening_book` when it names one, else on the gate's."""
    return replace(job, opening_book=str(cell["opening_book"])) if "opening_book" in cell else job


def cell_opponent(cell: Mapping[str, Any]) -> str:
    """The cell's `opponent`, REQUIRED: the old default (`sealbot_d5`) went with the rung."""
    if "opponent" not in cell:
        raise FrontierCellError(f"{cell.get('label')}: a cell names its opponent ({STRIX!r}, {SIX!r} or a "
                                "snapshot source); the sealbot rung is deleted and there is no default")
    return str(cell["opponent"])


def cell_channel(cell: Mapping[str, Any]) -> str:
    """The game-record channel a cell's games land on: rung opponents write `external`."""
    return "external" if cell_opponent(cell) in _RUNG_OPPONENTS else "promotion"


def armed_sides(cell: Mapping[str, Any]) -> str:
    """The heads a cell's block arms: ours against a ruler, both against a snapshot (the deploy-matched gate pair)."""
    return "candidate" if cell_opponent(cell) in _RUNG_OPPONENTS else "both"


def cell_spec(cell: Mapping[str, Any], base: RoundSpec, *, cell_dir: Path, config: Any) -> RoundSpec:
    """One cell's RoundSpec: the rung at `sims` vs strix or six, or a fixed-N gate block vs a model."""
    games = int(cell["games"])
    kind = str(cell["search_kind"])
    sims = int(cell["sims"])
    opponent = cell_opponent(cell)
    if kind not in ("gumbel", "puct"):
        raise FrontierCellError(f"{cell['label']}: search_kind {kind!r} is not gumbel|puct")
    if games < 2 or games % 2:
        raise FrontierCellError(f"{cell['label']}: games={games} must be an even number >= 2")
    seed_base = int(cell.get("seed_base", base.gate.seed_base))
    common = dict(
        round_id=str(cell["label"]), round_index=0, step=int(cell.get("step", 0)),
        seed_base=seed_base,
        candidate_snapshot=str(cell_dir / "candidate.pt"),
        result_path=str(cell_dir / "result.json"), progress_path=str(cell_dir / "progress.txt"),
        search_kind=kind, gumbel_m=int(cell.get("gumbel_m", base.gumbel_m)),
        c_scale=float(cell.get("c_scale", base.c_scale)),
        q_rescale=bool(cell.get("q_rescale", base.q_rescale)),
        game_record=GameRecordTarget(record_dir=str(cell_dir / "games"), run_id=_RUN_ID),
        concurrency=int(cell.get("concurrency", 1)),
        rung_concurrency=int(cell.get("concurrency", 1)),
        # Against a ruler our head alone plays it; a snapshot opponent is our net's gate pair, armed alike.
        tactics=cell.get("tactics", base.tactics),
    )
    if opponent == STRIX:
        if "strix_sims" not in cell:
            raise FrontierCellError(f"{cell['label']}: a strix cell names strix_sims (its sims per move)")
        job = _strix_rung(config, games, int(cell["strix_sims"]), solver=bool(cell.get("strix_solver", True)),
                          radius=None if cell.get("strix_radius") is None else int(cell["strix_radius"]))
        return replace(base, **common, rung_model_sims=sims, rung_jobs=[_rung_on_cell_book(job, cell)])
    if opponent == SIX:
        for key in ("six_net", "six_nodes"):
            if key not in cell:
                raise FrontierCellError(f"{cell['label']}: a six cell names {key} (six_net: the pinned network, "
                                        "six_nodes: its nodes per turn)")
        job = _rung_job(config, SIX, str(cell["six_net"]), int(cell["six_nodes"]), games)
        return replace(base, **common, rung_model_sims=sims, rung_jobs=[_rung_on_cell_book(job, cell)])
    # A fixed-N match: the GSPRT cannot stop before its last pair and reads its verdict once, which the cell ignores.
    pairs = games // 2
    if pairs < 2:
        raise FrontierCellError(f"{cell['label']}: a model cell plays >= 2 pairs (the GSPRT's LLR reads two); games={games}")
    gate = replace(base.gate, run_gate=True, deploy_sims=sims, seed_base=seed_base,
                   opening_book=str(cell.get("opening_book", base.gate.opening_book)),
                   sequential={**base.gate.sequential, "min_pairs": pairs, "max_pairs": pairs, "check_every_pairs": pairs})
    return replace(base, **common, gate=gate, best_snapshot=str(cell_dir / "opponent.pt"))


def _candidate_outcome(record: Mapping[str, Any]) -> float:
    result = record["result"]
    if result == "draw":
        return 0.5
    candidate_first = int(record["colors"]["candidate"]) == 1
    return 1.0 if (result == "p1") == candidate_first else 0.0


def _dedupe_key(record: Mapping[str, Any]) -> str:
    """The dedupe key as `mantis.eval.aggregate` spells it: the trajectory, qualified by the seat."""
    return f"{int(record['colors']['candidate'])}|{record.get('trajectory_hash') or json.dumps(record['moves'])}"


def pair_readout(records: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    """WR with a PAIR-level bootstrap CI over DISTINCT games; records `2k`/`2k+1` are one opening."""
    ordered = sorted(records, key=lambda r: int(r["game_index"]))
    seen: set[str] = set()
    pairs: list[list[float]] = []
    plies: list[int] = []
    for i in range(0, len(ordered), 2):
        legs = []
        for r in ordered[i:i + 2]:
            key = _dedupe_key(r)
            if key in seen:
                continue
            seen.add(key)
            legs.append(_candidate_outcome(r))
            if r.get("plies") is not None:
                plies.append(int(r["plies"]))
        if legs:
            pairs.append(legs)
    outcomes = [o for legs in pairs for o in legs]
    units = np.asarray([sum(legs) / len(legs) for legs in pairs], dtype=np.float64)
    lo, hi = pair_bootstrap_wr_ci(units, resamples=_BOOTSTRAP_RESAMPLES, ci_level=_CI_LEVEL,
                                  seed=seed)
    n = len(outcomes)
    return {
        "games": len(records), "eff_n": n, "pairs": len(units),
        "wins": sum(1 for o in outcomes if o == 1.0), "losses": sum(1 for o in outcomes if o == 0.0),
        "draws": sum(1 for o in outcomes if o == 0.5),
        "wr": (sum(outcomes) / n) if n else None, "wr_ci_lower": lo, "wr_ci_upper": hi,
        "median_plies": float(np.median(plies)) if plies else None,
    }


def tactics_module_sha256(repo: Path) -> str:
    """The tree's tactics sources (`tactics/**`, `mcts/tactics_*.rs`) hashed in path order; Raises: FrontierCellError, OSError."""
    src = repo / "crates" / "mantis-search" / "src"
    module, wiring = sorted((src / "tactics").rglob("*.rs")), sorted((src / "mcts").glob("tactics_*.rs"))
    if not module or not wiring:
        raise FrontierCellError(f"{src}: no tactics sources where the module hash reads them")
    h = hashlib.sha256()
    for path in [*module, *wiring]:
        h.update(str(path.relative_to(repo)).encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def load_arm(arm: str, block_path: Path | None) -> dict[str, Any] | None:
    """`arm_from_file` for this tool; Raises: FrontierCellError — the arm, the block or its file refused."""
    try:
        return arm_from_file(arm, block_path)
    except (ValueError, OSError) as exc:
        raise FrontierCellError(f"--arm {arm}: {exc}") from exc


def tactics_readout(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The candidate's rows summed over the cell's games, and each game a found root proof did not win, with its end."""
    rows: dict[str, int] = {}
    lost: list[dict[str, Any]] = []
    drawn: list[dict[str, Any]] = []
    for record in records:
        game = record.get("candidate_tactics")
        if game is None:
            continue
        for key, value in game.items():
            rows[key] = rows.get(key, 0) + int(value)
        outcome = _candidate_outcome(record)
        if game.get("root_proofs_found", 0) > 0 and outcome < 1.0:
            entry = {"game_index": int(record["game_index"]), "termination": record.get("termination")}
            (lost if outcome == 0.0 else drawn).append(entry)
    return {"rows": rows, "proof_games_lost": lost, "proof_games_drawn": drawn}


def _records_for(cell_dir: Path, channel: str) -> Iterator[dict[str, Any]]:
    """The cell's OWN games; the floor probe's `random_floor` records are posture, not reading."""
    for record in iter_run_games(cell_dir / "games", _RUN_ID):
        if record.get("channel") == channel:
            yield record


def run_cell(cell: Mapping[str, Any], *, config: Any, base: RoundSpec, work_dir: Path,
             python: str, env: Mapping[str, str]) -> dict[str, Any]:
    """Compose, spawn the eval child, wait, read out; returns the cell's record."""
    label = str(cell["label"])
    cell_dir = work_dir / label
    cell_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"candidate": build_snapshot(str(cell["candidate"]), config, cell_dir / "candidate.pt")}
    opponent = cell_opponent(cell)
    if opponent not in _RUNG_OPPONENTS:
        provenance["opponent"] = build_snapshot(opponent, config, cell_dir / "opponent.pt")
    spec = cell_spec(cell, base, cell_dir=cell_dir, config=config)
    spec_path = cell_dir / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict(), indent=1), encoding="utf-8")
    # Named before the child plays: the tree's tactics sources, and the engine bytes the child loads.
    tactics = None if spec.tactics is None and cell.get("tactics_arm") is None else {
        "arm": cell.get("tactics_arm"), "block": spec.tactics, "module_sha256": tactics_module_sha256(_REPO),
        "sides": armed_sides(cell),
        "engine_sha256": sha256_file(Path(mantis_engine.__file__))}
    started = time.time()
    with (cell_dir / "child.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [python, "-m", "mantis.eval.worker", str(spec_path), spec.result_path],
            stdout=log, stderr=subprocess.STDOUT, env=dict(env), check=False,
        )
    wall = time.time() - started
    record: dict[str, Any] = {
        "label": label, "cell": dict(cell), "provenance": provenance, "rc": proc.returncode,
        "wall_sec": round(wall, 1), "started_utc": time.strftime("%FT%TZ", time.gmtime(started)),
    }
    if tactics is not None:
        record["tactics"] = tactics
    if opponent == STRIX:
        from mantis.bots.strix import FINDING_LOG_MARKER

        lines = [ln.strip() for ln in (cell_dir / "child.log").read_text(encoding="utf-8").splitlines()
                 if FINDING_LOG_MARKER in ln]
        record["strix_findings"] = {"count": len(lines), "first": lines[:5]}
    if opponent == SIX:
        record.update(six_log_record((cell_dir / "child.log").read_text(encoding="utf-8")))
    if proc.returncode == 0:
        result = json.loads(Path(spec.result_path).read_text(encoding="utf-8"))
        record["worker_result"] = {"rungs": result.get("rungs"), "gate": result.get("gate"),
                                   "skipped_rungs": result.get("skipped_rungs")}
        games = list(_records_for(cell_dir, cell_channel(cell)))
        readout = pair_readout(games, seed=_BOOTSTRAP_SEED)
        if tactics is not None:
            tactics.update(tactics_readout(games))
        if readout["eff_n"] == 0:
            # A skipped rung or a refused floor probe exits 0 with no games of the cell's own:
            # a failed cell, never a 0-game reading.
            record["rc"] = -1
            record["error"] = (result.get("skipped_rungs") or result.get("strength_floor")
                               or "no games recorded")
        else:
            readout["sec_per_game"] = round(wall / readout["games"], 2)
            record["readout"] = readout
    (cell_dir / "cell.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def _marked(log: str, marker: str) -> list[dict[str, str]]:
    return [dict(tok.split("=", 1) for tok in ln.split(marker, 1)[1].split() if "=" in tok)
            for ln in log.splitlines() if marker in ln]


def six_log_record(log: str) -> dict[str, Any]:
    """The six engines' starts, provider, hashes and counters, and their forfeits, read off the child's log lines; Raises: ValueError on a non-integer counter."""
    starts, closes = _marked(log, SIX_PROVIDER_LOG_MARKER), _marked(log, SIX_CLOSE_LOG_MARKER)
    findings = [ln.strip() for ln in log.splitlines() if SIX_FINDING_LOG_MARKER in ln]
    providers = sorted({s.get("provider", "?") for s in starts})
    return {"six_engine": {"starts": len(starts), "provider": "+".join(providers) or None,
                           "engine_sha256": sorted({s.get("engine_sha256", "?") for s in starts}),
                           "net_sha256": sorted({s.get("net_sha256", "?") for s in starts}),
                           "runtime_sha256": sorted({s.get("runtime_sha256", "?") for s in starts}),
                           "searches": sum(int(c.get("searches", 0)) for c in closes),
                           "stale_pending": sum(int(c.get("stale_pending", 0)) for c in closes)},
            "six_findings": {"count": len(findings), "first": findings[:5]}}


def format_row(record: Mapping[str, Any]) -> str:
    """One summary line per cell."""
    cell = record["cell"]
    sigma = "".join(f" {k}={cell[k]}" for k in ("c_scale", "q_rescale", "strix_sims", "strix_solver", "strix_radius",
                                                 "six_net", "six_nodes") if k in cell)
    head = f"{record['label']:<28} {cell['search_kind']:<6} {int(cell['sims']):>4}{sigma}"
    if record["rc"] != 0 or "readout" not in record:
        return f"{head}  FAILED rc={record['rc']} ({record['wall_sec']} s) {record.get('error', '')}"
    r = record["readout"]
    med = "-" if r["median_plies"] is None else f"{r['median_plies']:.0f}"
    findings = record.get("strix_findings")
    tail = "" if findings is None else f"  strix findings {findings['count']}"
    forfeits = record.get("six_findings")
    tail += "" if forfeits is None else f"  six forfeits {forfeits['count']}"
    return (f"{head}  WR {r['wr']:.3f} [{r['wr_ci_lower']:.3f}, {r['wr_ci_upper']:.3f}]  "
            f"{r['wins']}-{r['losses']}-{r['draws']} n={r['games']} eff_n={r['eff_n']}  "
            f"{r['sec_per_game']} s/game  med {med} plies{tail}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run every cell of a cells file, `--parallel` at a time, writing `summary.jsonl`."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--cells", required=True, help="JSON list of cells, or a .jsonl")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--only", default=None, help="comma-separated labels to run")
    parser.add_argument("--arm", choices=ARMS, default=None,
                        help="every cell's tactics arm, our head's against a ruler and both heads' against a snapshot; "
                             "absent, the config's own block")
    parser.add_argument("--tactics-block", type=Path, default=None, help="a search.tactics block (JSON) the arm overlays")
    args = parser.parse_args(argv)
    if args.tactics_block is not None and args.arm is None:
        parser.error("--tactics-block names the block an --arm overlays; give the arm")

    config = load_config(args.config)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    text = Path(args.cells).read_text(encoding="utf-8")
    cells = ([json.loads(line) for line in text.splitlines() if line.strip()]
             if args.cells.endswith(".jsonl") else json.loads(text))
    if args.only:
        wanted = set(args.only.split(","))
        cells = [c for c in cells if c["label"] in wanted]
    labels = [c["label"] for c in cells]
    if len(set(labels)) != len(labels):
        raise FrontierCellError(f"duplicate cell labels: {labels}")
    if any("tactics" in c or "tactics_arm" in c for c in cells):
        raise FrontierCellError("a cells file names no tactics block: --arm over --tactics-block is the one way in")
    if args.arm is not None:
        armed = load_arm(args.arm, args.tactics_block)
        # An arm's cells are its own: their label, and so their games' directory, carry the arm.
        cells = [{**c, "label": f"{c['label']}_{args.arm}", "tactics": armed, "tactics_arm": args.arm} for c in cells]
    base = base_round_spec(config, work_dir=work_dir)
    env = dict(os.environ)
    if governs_device(config.eval.worker_device) and base.allocator_posture == "expandable_segments":
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    summary = work_dir / "summary.jsonl"
    summary_lock = threading.Lock()
    print(f"frontier: {len(cells)} cells, parallel={args.parallel}, device={config.eval.worker_device}",
          flush=True)

    def _one(cell: Mapping[str, Any]) -> dict[str, Any]:
        record = run_cell(cell, config=config, base=base, work_dir=work_dir,
                          python=sys.executable, env=env)
        with summary_lock, summary.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(format_row(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        records = list(pool.map(_one, cells))
    failed = [r["label"] for r in records if r["rc"] != 0]
    print(f"done: {len(records) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

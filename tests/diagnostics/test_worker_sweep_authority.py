"""The rows the first cut of the PREREG left out, each for something load-bearing.

>300 justify (R8): ONE SUBJECT — the properties that gate the mint and were checked by nothing.
Each row here pairs a claim with the mutation that survives without it, and the rows are kept
together because the mutations OVERLAP: the posture assertion, the verdict's composition across
both sinks, the pool's fail-fast hook, the sampler's death, the rung-failure taxonomy and the
`--select-only` identity checks are all forms of the same defect — an instrument that produces a
plausible number for a drive that did not happen. Splitting them by subject would let one be
deleted without the argument for the others having to be re-made, and the argument is what this
file is for. The two AST helpers (`_func`, `_called_line`) are shared by the structural rows for
`0bb4381`'s reason: a predicate and the proof it can fire must move together or the proof rots.


Every row here exists because a driver that PASSED every other row in this suite could still be
wrong in the way the row names. That is the test for whether an oracle is worth writing: not "does
it check something", but "what mutation survives without it".

  * THE ALLOCATOR-POSTURE ASSERTION is mint-critical (R308(g)(i)) and had no falsifier at all.
  * THE VERDICT'S COMPOSITION — `governing = max(card, allocator)` — is the one rule the box
    block calls its standing rule, and a driver that silently dropped the card sink passed every
    verdict row in `test_worker_sweep_verdicts.py`.
  * NEVER WRITES A CONFIG is the strongest safety claim made about a tool that runs on the box
    during a mint sitting, and it had no producer. `fusion_calibrate`'s equivalent property is
    stated in prose there too; here it is a census.
  * WARM-UP EXCLUSION: a driver that fed warm-up rounds into the stopping rule passed everything.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mantis.diagnostics import worker_sweep as ws

_MODULE = Path(ws.__file__)
_PLAN = Path(__file__).resolve().parents[2] / "tools" / "worker_sweep_plan.toml"
_MIB = 1024 ** 2


@pytest.fixture()
def plan() -> ws.SweepPlan:
    return ws.load_plan(_PLAN)


def _func(name: str) -> ast.FunctionDef:
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {_MODULE}")


def _called_line(fn: ast.FunctionDef, symbol: str) -> int:
    lines = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
             and ((isinstance(n.func, ast.Name) and n.func.id == symbol)
                  or (isinstance(n.func, ast.Attribute) and n.func.attr == symbol))]
    assert lines, f"{symbol} is never called inside {fn.name}"
    return min(lines)


# ══ the allocator posture ════════════════════════════════════════════════════════════════
def test_the_posture_is_asserted_before_the_ladder_is_walked() -> None:
    """STRUCTURAL, on the tree's own precedent (`tests/config/test_allocator_posture_authority
    .py` reads the AST and pins that the assert precedes `init_trainer`). Here the subject is
    `walk_ladder`, because that is what builds the first pool and therefore takes the first CUDA
    allocation. A posture check after the first allocation is a check of a regime the process is
    already in."""
    fn = _func("run_sweep")
    assert _called_line(fn, "assert_allocator_posture") < _called_line(fn, "walk_ladder"), (
        "the allocator-posture assertion must precede the ladder walk — a cap fitted under one "
        "regime and run under the other is a partition measured for a machine state this "
        "process is not in (3.62 GiB of card high-water, RECAL_EXIT_2026-08-22.md §2)"
    )


def test_the_posture_authority_is_the_one_the_run_root_calls_not_a_copy() -> None:
    from mantis.config.resolve import allocator_posture

    assert ws.assert_allocator_posture is allocator_posture.assert_allocator_posture


class _StopAfterPosture(Exception):
    """Sentinel: the posture check was PASSED and pool construction was reached."""


def _null_posture_twin(tmp_path: Path) -> Path:
    """`configs/run5.yaml` with its posture returned to the R119 `null` placeholder.

    WHY A TWIN AND NOT THE COMMITTED FILE. These rows used to point straight at
    `configs/run5.yaml`, on the stated premise that *"every config on this tree carries the
    R119 `null` placeholder"*. RECAL-SITTING-5's mint (R326) made that premise FALSE — all
    seven now mint a measured posture — so the rows would have gone quietly green against a
    config that no longer exercises the refusal at all. **The property is the refusal, not the
    state of the committed tree**, so the null is CONSTRUCTED rather than borrowed and these
    rows stay live through any future re-mint.
    """
    import yaml

    raw = yaml.safe_load(Path("configs/run5.yaml").read_text(encoding="utf-8"))
    assert raw["train"]["device"] == "cuda", "the refusal is cuda-side; the twin must stay cuda"
    raw["allocator_posture"] = None
    out = tmp_path / "run5_null_posture.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out


def test_a_cuda_config_with_a_null_posture_refuses_before_any_pool_is_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """BEHAVIOURAL half. This is what Phase W does on the box if the posture has not been
    decided — and discovering it HERE costs nothing, while discovering it at the box costs a
    sitting."""
    def explode(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a pool was built despite an unminted allocator posture")

    monkeypatch.setattr(ws, "build_sweep_pool", explode)
    sink = io.StringIO()
    with pytest.raises(ValueError, match="allocator_posture"):
        ws.run_sweep(config_path=_null_posture_twin(tmp_path), plan_path=_PLAN, out=sink,
                     noise_floor_rel_std=0.0)


def test_the_null_posture_refusal_reaches_the_exit_code_as_a_named_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path,
) -> None:
    monkeypatch.setattr(ws, "build_sweep_pool", lambda *a, **k: None)
    rc = ws.main(["--config", str(_null_posture_twin(tmp_path)), "--plan", str(_PLAN)])
    assert rc == ws.RC_REFUSED
    assert "REFUSED" in capsys.readouterr().err


def test_the_MINTED_posture_no_longer_refuses_and_that_is_the_mints_own_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other direction, and the reason the two rows above could be repaired safely.

    A twin-based refusal row proves the refusal fires on a null. It does NOT prove the
    committed tree has stopped tripping it — and a repair that only moved the subject could
    hide a config still carrying the placeholder. This asserts the complement directly:
    `configs/run5.yaml` AS MINTED gets past the posture check and reaches pool construction.

    AND IT CARRIES THE MINT'S SECOND CONSEQUENCE, which is the one that will surprise people:
    a minted posture is a CONTRACT ON THE LAUNCH ENVIRONMENT. With `expandable_segments` minted,
    a cuda process started WITHOUT `PYTORCH_CUDA_ALLOC_CONF` no longer refuses for a missing
    posture — it refuses for a MISMATCHED one, and says so by name. This row therefore launches
    the way the box launches, and the required conf is DERIVED from the resolver rather than
    transcribed, so a third regime cannot leave a stale literal here.

    MUTATION THAT REDS IT: run5 reverting to `null`."""
    from mantis.config.resolve.allocator_posture import resolve_allocator_posture

    spec = resolve_allocator_posture({"allocator_posture": "expandable_segments"})
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF",
                       ",".join(f"{k}:{v}" for k, v in spec.required_conf.items()))
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    reached: list[bool] = []

    def record(*_a: Any, **_k: Any) -> Any:
        reached.append(True)
        raise _StopAfterPosture

    # `walk_ladder` is the step the STRUCTURAL row above names as the one the assertion must
    # precede, so it is the right witness for "the assertion passed": reaching it means the
    # posture check ran and did not refuse. Patched here rather than `build_sweep_pool`, which
    # `run_sweep` reaches only through the ladder's own runner.
    monkeypatch.setattr(ws, "walk_ladder", record)
    sink = io.StringIO()
    with pytest.raises(_StopAfterPosture):
        ws.run_sweep(config_path=Path("configs/run5.yaml"), plan_path=_PLAN, out=sink,
                     noise_floor_rel_std=0.0)
    assert reached, "run5's minted posture must reach the ladder, not refuse before it"


# ══ the verdict's composition — the card sink actually reaches the rule ══════════════════
def test_growth_visible_ONLY_on_the_card_sink_still_verdicts_growing(plan: ws.SweepPlan) -> None:
    """THE MUTATION THIS CATCHES: a driver that verdicts on `allocator_peak_bytes` alone. Every
    row in `test_worker_sweep_verdicts.py` passes under it, because those series move together.
    Here they do not: the allocator's per-round demand is flat while the CARD's committed level
    climbs — which is exactly the shape the 2026-08-22 host produced (§8.3, the sawtooth against
    a rising high-water), and exactly the reason the block's rule is "the larger governs"."""
    flat_alloc = 400 * _MIB
    card = [900, 1000, 1000, 1400, 1900, 2600]
    rounds = tuple(
        ws.RoundReading(index=i, warmup=(i == 0), wall_sec=120.0, games=3, moves=1000,
                        available=True, sampled_peak_bytes=c * _MIB,
                        allocator_peak_bytes=flat_alloc, card_samples=120)
        for i, c in enumerate(card)
    )
    verdict, _refusal = ws._verdict_for(rounds, plan)
    assert verdict == ws.GROWING, (
        "the card sink never reached the stopping rule — the verdict is being taken on the "
        "allocator alone, and the block's standing rule needs both"
    )
    assert all(r.governing_sink == "card" for r in rounds if not r.warmup)


def test_the_sink_disagreement_is_reported_as_a_number_not_only_resolved(
    plan: ws.SweepPlan,
) -> None:
    """"The larger governs AND the disagreement is a finding" — a rule whose second half needs
    the difference on the record to be actionable at all."""
    rounds = tuple(
        ws.RoundReading(index=i, warmup=(i == 0), wall_sec=120.0, games=3, moves=1000,
                        available=True, sampled_peak_bytes=3000 * _MIB,
                        allocator_peak_bytes=400 * _MIB, card_samples=120)
        for i in range(6)
    )
    rung = ws.RungResult(n_workers=4, verdict=ws.PLATEAU, rounds=rounds, refusal=None,
                         produced_by="t")
    row = rung.as_dict(plan.metric, plateau_rounds=plan.plateau_rounds)
    assert row["max_sink_disagreement_bytes"] == 2600 * _MIB
    assert row["governing_sink_counts"] == {"card": 5}


def test_the_masked_rise_flag_fires_on_a_window_the_stopping_rule_cannot_see(
    plan: ws.SweepPlan,
) -> None:
    """`classify` compares the trailing window against the running maximum BEFORE it, so a
    start-up spike in an early measured round raises the baseline and a strictly rising window
    under it still verdicts PLATEAU. That is right for the eval child (fresh process per round)
    and not obviously right for a long-lived pool. The rule is the ruling's and is not changed;
    the SHAPE it cannot see is reported instead."""
    peaks = [900, 8000, 1000, 5000, 5200, 5400]
    rounds = tuple(
        ws.RoundReading(index=i, warmup=(i == 0), wall_sec=120.0, games=3, moves=1000,
                        available=True, sampled_peak_bytes=v * _MIB,
                        allocator_peak_bytes=v * _MIB, card_samples=120)
        for i, v in enumerate(peaks)
    )
    verdict, _refusal = ws._verdict_for(rounds, plan)
    rung = ws.RungResult(n_workers=4, verdict=verdict, rounds=rounds, refusal=None,
                         produced_by="t")
    assert verdict == ws.PLATEAU, "the fixture is only interesting if the rule passes it"
    rise = rung.trailing_rise_pct("governing", plan.plateau_rounds)
    assert rise is not None and rise > plan.band_pct, (
        f"the trailing-window trend ({rise}) did not exceed the band on a rising window; the "
        "record would then carry a PLATEAU with nothing beside it"
    )


def test_the_trend_flag_is_not_a_strict_monotone_check(plan: ws.SweepPlan) -> None:
    """THE MUTATION THIS CATCHES: `all(b > a for a, b in zip(window, window[1:]))`, which was the
    first cut. A single REPEATED or DIPPING value anywhere in the window silences a strict check
    while the window climbs by the same total — and real memory series are not strictly monotone,
    so the shape a strict flag catches is the one a real series is least likely to have."""
    for peaks in ([900, 8000, 1000, 5000, 5000, 5400],      # one equal step
                  [900, 8000, 1000, 5000, 4800, 5400]):     # one dip
        rounds = tuple(
            ws.RoundReading(index=i, warmup=(i == 0), wall_sec=120.0, games=3, moves=1000,
                            available=True, sampled_peak_bytes=v * _MIB,
                            allocator_peak_bytes=v * _MIB, card_samples=120)
            for i, v in enumerate(peaks))
        rung = ws.RungResult(n_workers=4, verdict=ws.PLATEAU, rounds=rounds, refusal=None,
                             produced_by="t")
        rise = rung.trailing_rise_pct("governing", plan.plateau_rounds)
        assert rise is not None and rise > plan.band_pct, (
            f"a window rising {rise}% overall was not flagged; series {peaks}"
        )


# ══ warm-up rounds are excluded, from the verdict AND from the throughput ════════════════
def test_a_warmup_round_reaches_neither_the_verdict_nor_the_throughput(
    plan: ws.SweepPlan,
) -> None:
    """A driver that fed the warm-up round in passes every other row in this suite. The warm-up
    exists because pool start-up is not throughput — and, on the memory side, because a
    start-up transient inside the stopping rule's baseline is the masked-rise shape above."""
    rounds = (
        ws.RoundReading(index=0, warmup=True, wall_sec=120.0, games=0, moves=0, available=True,
                        sampled_peak_bytes=9999 * _MIB, allocator_peak_bytes=9999 * _MIB,
                        card_samples=120),
        *(ws.RoundReading(index=i, warmup=False, wall_sec=120.0, games=3, moves=1000,
                          available=True, sampled_peak_bytes=1000 * _MIB,
                          allocator_peak_bytes=1000 * _MIB, card_samples=120)
          for i in range(1, 6)),
    )
    rung = ws.RungResult(n_workers=4, verdict=ws.PLATEAU, rounds=rounds, refusal=None,
                         produced_by="t")
    assert len(rung.measured) == 5 and len(rung.scored) == 5
    row = rung.as_dict("moves_per_min", plateau_rounds=plan.plateau_rounds)
    assert row["rung_peak_bytes"] == 1000 * _MIB, "the warm-up round's peak reached the report"
    assert row["moves_per_min"] == pytest.approx(500.0), (
        "the warm-up round's zero-throughput window reached the rate"
    )


# ══ the tool never writes a config ═══════════════════════════════════════════════════════
def test_the_tool_writes_exactly_one_thing_and_it_is_the_report(  # noqa: D401
) -> None:
    """THE SAFETY CLAIM, MECHANIZED. This tool runs on the box during a mint sitting. Minting is
    the operator's act (R119) and the pick lands through `tools/mint_config.py` on the sitting's
    branch (R308(b)); a diagnostics tool that could write a config is one editor slip from
    minting a value nobody recorded."""
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))

    def _writes(node: ast.AST) -> list[str]:
        found: list[str] = []
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            name = (sub.func.attr if isinstance(sub.func, ast.Attribute)
                    else sub.func.id if isinstance(sub.func, ast.Name) else "")
            if name in ("write_text", "write_bytes", "writelines", "mkdir", "rename", "unlink"):
                found.append(name)
            if name == "open":
                modes = [a.value for a in sub.args[1:] if isinstance(a, ast.Constant)]
                modes += [k.value.value for k in sub.keywords
                          if k.arg == "mode" and isinstance(k.value, ast.Constant)]
                if any(set("wax+") & set(str(m)) for m in modes):
                    found.append("open(w)")
        return found

    # THE PREDICATE, and it is about the DESTINATION rather than the count: every write in this
    # module must sit in a STATEMENT that also names `args.out`. A count would have to be edited
    # every time a write moved (the probe-then-write shape already made it three), and a count
    # nobody re-derives is the class gate 15's second half governs.
    offenders: list[str] = []
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for stmt in ast.walk(func):
            if not isinstance(stmt, ast.stmt):
                continue
            writes = _writes(stmt)
            if not writes or any(_writes(child) for child in ast.iter_child_nodes(stmt)
                                 if isinstance(child, ast.stmt)):
                continue  # only the innermost statement carrying the call
            names = {f"{n.value.id}.{n.attr}" for n in ast.walk(stmt)
                     if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
            if "args.out" not in names:
                offenders.append(f"{func.name}:{stmt.lineno}:{writes}")
    assert not offenders, (
        f"the sweep writes to a destination other than its own --out report: {offenders}"
    )
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else
              node.func.id if isinstance(node.func, ast.Name) else ""
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
    assert "yaml" not in imported, (
        "a tool whose one safety claim is that it never writes a config imports a yaml writer"
    )
    for banned in ("safe_dump", "dump", "mint", "mint_config"):
        assert banned not in called, f"the sweep calls {banned!r}"


# ══ P9, strengthened: produced_by is per-row, not a constant the renderer supplies ═══════
def test_two_rungs_with_different_producers_print_both(plan: ws.SweepPlan, capsys) -> None:
    """The census this replaces passed on a renderer that printed one constant for every rung."""
    def rung(n: int, label: str) -> ws.RungResult:
        return ws.RungResult(n_workers=n, verdict=ws.PLATEAU, refusal=None, produced_by=label,
                             rounds=tuple(
                                 ws.RoundReading(index=i, warmup=(i == 0), wall_sec=120.0,
                                                 games=3, moves=1000, available=True,
                                                 sampled_peak_bytes=1000 * _MIB,
                                                 allocator_peak_bytes=900 * _MIB,
                                                 card_samples=120)
                                 for i in range(6)))

    prov = {"produced_by": "sweep@aaa", "config_name": "c", "config_sha256": "s", "git_commit": "g",
            "git_dirty": False, "device": "cuda", "torch_version": "2.11", "gpu_name": "X",
            "card_total_bytes": 16 * 1024 ** 3, "declared_allocator_posture": "default",
            "live_allocator_conf": "", "live_allocator_conf_source_var": None,
            "thread_bound": 16, "thread_bound_source": "os.sched_getaffinity(0)",
            "cuda_counters_available": True}
    report = ws.build_report(plan=plan, prov=prov,
                             results=[rung(2, "run5@aaa"), rung(4, "run5@bbb")], stopped="t",
                             noise_floor_rel_std=0.0)
    ws.render(report, __import__("sys").stdout)
    text = capsys.readouterr().out
    assert "run5@aaa" in text and "run5@bbb" in text


def test_the_cpu_device_helpers_refuse_rather_than_return_zero() -> None:
    """`cuda_device_total_bytes` joins the pair `cuda_device_used_bytes` belongs to; a total of
    zero would make every peak in the report look like 100% of the card."""
    from mantis.util.device import cuda_device_total_bytes

    with pytest.raises(ValueError):
        cuda_device_total_bytes("cpu")




# ══ RED-TEAM rows — each closes a probe that produced a wrong answer looking right ════════
def test_growth_visible_ONLY_on_the_allocator_sink_still_fails_the_rung(
    plan: ws.SweepPlan,
) -> None:
    """THE DIRECTION `max()` ACTUALLY EATS, and the one the first oracle did not cover.

    On a real box the card level RATCHETS — torch's caching allocator does not return reserved
    blocks to the driver, so `total - free` climbs to its high-water and sits there. A rung whose
    allocator DEMAND grows 3.3x underneath a flat 15.4 GiB card level therefore verdicted PLATEAU
    on the composite, passed, and entered the knee set, while `classify` on the allocator series
    alone said GROWING. That is `RECAL_EXIT_2026-08-22.md` §11b's own failure — *a term measured
    by watching until it looks flat is not a bound* — reproduced inside the instrument built to
    end it."""
    flat_card = 15400 * _MIB
    demand = [3000, 3200, 3400, 6000, 8000, 10000]
    rounds = tuple(
        ws.RoundReading(index=i, warmup=(i == 0), wall_sec=120.0, games=3, moves=1000,
                        available=True, sampled_peak_bytes=flat_card,
                        allocator_peak_bytes=v * _MIB, card_samples=120)
        for i, v in enumerate(demand)
    )
    rung = ws.RungResult(n_workers=8, verdict="", rounds=rounds, refusal=None, produced_by="t")
    assert rung.sink_verdicts(plateau_rounds=plan.plateau_rounds,
                              band_pct=plan.band_pct)["governing"] == ws.PLATEAU, (
        "the fixture is only interesting if the COMPOSITE passes it"
    )
    verdict, refusal = ws._verdict_for(rounds, plan)
    assert verdict == ws.GROWING, (
        "a rung whose allocator demand grew 3.3x under a flat card level PASSED — `max()` "
        "resolved the pair before the stopping rule ever ran"
    )
    assert "allocator" in (refusal or "")


def test_a_rung_whose_sole_producer_died_does_not_report_a_plateau(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE SHARPEST WRONG-ANSWER-THAT-LOOKS-RIGHT the red team found, and the two halves
    reinforce each other: with the drain dead nothing reaches the replay buffer, so the memory
    series goes FLAT BECAUSE THE RUNG IS BROKEN, while `runner_stats` keeps reporting from the
    RUST counters, which climb regardless. The knee rule compares rates ACROSS rungs, so one dead
    feeder moves the pick.

    `WorkerPool._stats_loop` stores the feeder's death in `_producer_exc` and does NOT raise;
    `check_producer_health` is the pool's own fail-fast hook and the trainer calls it every step.
    This sweep has no trainer, so nothing called it."""
    class _DeadFeeder:
        _producer_exc = RuntimeError("selfplay_producer_died")
        model = type("_NoParams", (), {"state_dict": lambda self: {}})()

        def start(self) -> None: ...
        def stop(self) -> None: ...

        def check_producer_health(self) -> None:
            raise RuntimeError("self-play buffer feeder died") from self._producer_exc

    counter = {"n": 0}

    def stats(_pool: object) -> Any:
        counter["n"] += 1
        return type("S", (), {"games_completed": counter["n"],
                              "positions_generated": counter["n"] * 500})()

    monkeypatch.setattr(ws, "build_sweep_pool", lambda *a, **k: _DeadFeeder())
    monkeypatch.setattr(ws, "cuda_counters_available", lambda _d: False)
    monkeypatch.setattr(ws, "runner_stats", stats)
    result = ws.drive_rung(object(), plan, n_workers=8, device=torch.device("cpu"),
                           label="t", out=io.StringIO(), sleep=lambda _s: None)
    assert result.verdict != ws.PLATEAU
    assert result.verdict == ws.RUNG_ERROR
    assert "feeder" in (result.refusal or "")


def test_the_sweep_calls_the_pools_own_fail_fast_hook(
) -> None:
    """The structural half: `check_producer_health` appeared in `mantis.train` and in four test
    doubles, and in no consumer that actually drives a pool without a trainer."""
    assert "check_producer_health" in _MODULE.read_text(encoding="utf-8")
    assert _called_line(_func("drive_rung"), "check_producer_health") > 0


def test_a_dead_card_sampler_refuses_the_rung_instead_of_switching_instrument(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon thread that dies silently is a MEASUREMENT THAT STOPPED. Unguarded, the rung's
    series switched instrument mid-flight and a series that FELL by 9.3 GiB then rose 47%
    verdicted PLATEAU — the exact hazard `reset_cuda_peak_counters`' own docstring names."""
    class _Pool:
        _producer_exc = None
        # R317(c)(i): drive_rung hashes `pool.model` right after the build; a mock pool needs one.
        model = type("_NoParams", (), {"state_dict": lambda self: {}})()

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def check_producer_health(self) -> None: ...

    class _DeadSampler(ws.CardSampler):
        def start(self) -> None:  # never spawns; the "thread" is already dead
            self._error = ValueError("device 'cuda' names CUDA but this process has no CUDA")

    monkeypatch.setattr(ws, "build_sweep_pool", lambda *a, **k: _Pool())
    monkeypatch.setattr(ws, "cuda_counters_available", lambda _d: True)
    monkeypatch.setattr(ws, "reset_cuda_peak_counters", lambda _d: None)
    monkeypatch.setattr(ws, "CardSampler", _DeadSampler)
    monkeypatch.setattr(ws, "runner_stats",
                        lambda _p: type("S", (), {"games_completed": 1,
                                                  "positions_generated": 500})())
    result = ws.drive_rung(object(), plan, n_workers=4, device=torch.device("cpu"),
                           label="t", out=io.StringIO(), sleep=lambda _s: None)
    assert result.verdict == ws.REFUSED
    assert "sampler thread died" in (result.refusal or "")


@pytest.mark.parametrize("exc", [
    RuntimeError("selfplay runner panicked"),
    RuntimeError("CUDA out of memory"),          # the message, not the type
    ValueError("something unmodelled"),
])
def test_an_unmodelled_rung_failure_is_a_named_verdict_and_not_a_lost_ladder(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch, exc: Exception,
) -> None:
    """`mantis-bridge` builds with `panic = "unwind"` (R2/LAW-13) precisely so a Rust panic
    crosses the FFI as an exception. Only `torch.OutOfMemoryError` was caught, so an escaping
    `RuntimeError` reached the interpreter as SHELL RC 1 — which this tool's contract reserves
    for "no rung PASSED". A crash must never present as a measured memory result, and the ladder
    is the expensive artifact."""
    def explode(*_a: Any, **_k: Any) -> Any:
        raise exc

    monkeypatch.setattr(ws, "build_sweep_pool", explode)
    monkeypatch.setattr(ws, "cuda_counters_available", lambda _d: False)
    result = ws.drive_rung(object(), plan, n_workers=8, device=torch.device("cpu"),
                           label="t", out=io.StringIO())
    assert result.verdict == ws.RUNG_ERROR
    assert repr(exc) in (result.refusal or "")


def test_a_teardown_failure_does_not_erase_the_oom_finding_it_was_teardown_for(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plausible pairing: an OOM is exactly when an inference-server join is likeliest to
    fail. A raise in `finally` REPLACES the return value, so the OOM verdict never left the
    function, the `rung_end` marker was never written, and the ladder died with the teardown's
    traceback."""
    class _Exploding:
        _producer_exc = None
        model = type("_NoParams", (), {"state_dict": lambda self: {}})()

        def start(self) -> None: ...
        def check_producer_health(self) -> None:
            raise torch.OutOfMemoryError("CUDA out of memory (synthetic)")

        def stop(self) -> None:
            raise RuntimeError("inference server join failed")

    monkeypatch.setattr(ws, "build_sweep_pool", lambda *a, **k: _Exploding())
    monkeypatch.setattr(ws, "cuda_counters_available", lambda _d: False)
    monkeypatch.setattr(ws, "runner_stats",
                        lambda _p: type("S", (), {"games_completed": 0,
                                                  "positions_generated": 0})())
    sink = io.StringIO()
    result = ws.drive_rung(object(), plan, n_workers=8, device=torch.device("cpu"),
                           label="t", out=sink, sleep=lambda _s: None)
    assert result.verdict == ws.OOM, "the teardown failure replaced the OOM finding"
    assert "teardown failed" in (result.refusal or ""), "the teardown failure was swallowed"
    assert "rung_end" in [r["phase"] for r in ws.parse_sweep_markers(sink.getvalue())]


def test_select_only_refuses_a_document_that_is_not_this_tools_output(tmp_path: Path) -> None:
    """A three-key hand-written dict used to yield `PICK = 1` — the value R309(f) REJECTS — at
    rc 0. The marker reader invokes the `peaks.py` lesson; this reader is the one that then never
    asked whether the shape was its own."""
    path = tmp_path / "not-ours.json"
    path.write_text(json.dumps({
        "provenance": {"produced_by": "handwritten"},
        "plan": {"knee_pct": 95.0, "metric": "moves_per_min"},
        "rungs": [{"n_workers": 1, "verdict": "PLATEAU", "moves_per_min": 1.0}],
    }), encoding="utf-8")
    assert ws.main(["--select-only", str(path)]) == ws.RC_REFUSED


def test_select_only_refuses_a_report_whose_stated_rule_is_not_the_ruling_s(
    tmp_path: Path,
) -> None:
    base = {"tool": ws.TOOL, "provenance": {"produced_by": "run5@abc"},
            "plan": {"knee_pct": 95.0, "metric": "moves_per_min"},
            "rungs": [{"n_workers": 2, "verdict": "PLATEAU", "moves_per_min": 900.0}]}
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(base), encoding="utf-8")
    # R317(d): --select-only now needs a measured noise floor to select under.
    nf_path = tmp_path / "noise_floor.json"
    nf_path.write_text(json.dumps({"tool": ws.TOOL, "mode": "noise_floor",
                                   "noise_floor": {"rel_std": 0.0}}), encoding="utf-8")
    assert ws.main(["--select-only", str(ok), "--noise-floor-report", str(nf_path)]) == 0
    for edit in ({"knee_pct": 60.0}, {"knee_pct": 100.0}, {"metric": "games_per_min"}):
        bad = tmp_path / f"bad{tuple(edit)[0]}{tuple(edit.values())[0]}.json"
        bad.write_text(json.dumps({**base, "plan": {**base["plan"], **edit}}), encoding="utf-8")
        assert ws.main(["--select-only", str(bad), "--noise-floor-report", str(nf_path)]) \
            == ws.RC_REFUSED, (
            f"a one-field edit ({edit}) re-derived a pick with the ruling's own authority"
        )


def test_a_malformed_report_refuses_at_rc_2_and_never_presents_as_rc_1(tmp_path: Path) -> None:
    """rc 1 is "no rung PASSED", and the Phase W block's failure posture BRANCHES on rc 1 vs
    rc 2. An unhandled `KeyError` used to exit 1 — a malformed artifact presenting as a measured
    memory result."""
    path = tmp_path / "no-produced-by.json"
    path.write_text(json.dumps({"tool": ws.TOOL, "provenance": {},
                                "plan": {"knee_pct": 95.0, "metric": "moves_per_min"},
                                "rungs": []}), encoding="utf-8")
    assert ws.main(["--select-only", str(path)]) == ws.RC_REFUSED


def test_select_only_refuses_an_out_it_would_silently_ignore(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"tool": ws.TOOL, "provenance": {"produced_by": "x"},
                                "plan": {"knee_pct": 95.0, "metric": "moves_per_min"},
                                "rungs": []}), encoding="utf-8")
    assert ws.main(["--select-only", str(path), "--out", str(tmp_path / "o.json")]) \
        == ws.RC_REFUSED


def test_an_unwritable_out_is_refused_BEFORE_the_ladder_not_after_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The destination used to be validated at the moment it is least recoverable — after a
    seventy-minute ladder, outside the guard, and BEFORE the render, so neither the file nor the
    screen survived."""
    def explode(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the ladder was walked before --out was probed")

    monkeypatch.setattr(ws, "run_sweep", explode)
    unwritable = tmp_path / "nodir" / "x" / "report.json"
    monkeypatch.setattr(ws.Path, "mkdir",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("read-only")))
    assert ws.main(["--config", "configs/run5.yaml", "--plan", str(_PLAN),
                    "--out", str(unwritable)]) == ws.RC_REFUSED


# ══ I-3 — the per-round memory-instrument WIRING, on the counters-PRESENT path ═══════════
def test_one_probe_and_one_counter_reset_per_round_on_a_counters_present_host(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE MUTATION THIS CATCHES: hoisting the probe out of the round loop, or dropping the
    per-round `reset_cuda_peak_counters`. Both passed every other row in this packet, because
    every other `drive_rung` row monkeypatches `cuda_counters_available -> False` and never
    enters the counters branch at all.

    The two mechanisms are what make the memory series MEAN what the report says it means: one
    probe per round is what keeps `DeviceMemoryProbe`'s running maxima per-round rather than
    per-rung (it never resets them itself — `eval/child_memory.py` says so, and says why), and
    the reset is the window boundary the "independent per-round peaks" claim rests on. Asserted
    in prose is not asserted."""
    probes: list[str] = []
    resets: list[str] = []
    cards = iter(range(1_000_000_000, 9_000_000_000, 1_000_000_000))

    class _Probe:
        def __init__(self, round_id: str) -> None:
            self._round_id = round_id

        def mark(self, phase: str) -> dict:
            return {"phase": phase, "max_memory_allocated_bytes": 4 * _MIB,
                    "max_memory_reserved_bytes": 5 * _MIB}

    class _Pool:
        _producer_exc = None
        # R317(c)(i): drive_rung hashes `pool.model` right after the build; a mock pool needs one.
        model = type("_NoParams", (), {"state_dict": lambda self: {}})()

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def check_producer_health(self) -> None: ...

    class _Sampler:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            self._peak: int | None = None

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def reset(self) -> None:
            self._peak = None

        def window(self) -> tuple[int | None, int]:
            self._peak = next(cards)
            return self._peak, 120

        def error(self) -> None:
            return None

    def make_probe(_device: str, *, round_id: str, out: Any) -> Any:
        probes.append(round_id)
        return _Probe(round_id)

    monkeypatch.setattr(ws, "build_sweep_pool", lambda *a, **k: _Pool())
    monkeypatch.setattr(ws, "cuda_counters_available", lambda _d: True)
    monkeypatch.setattr(ws, "make_probe", make_probe)
    monkeypatch.setattr(ws, "reset_cuda_peak_counters", lambda d: resets.append(d))
    monkeypatch.setattr(ws, "CardSampler", _Sampler)
    monkeypatch.setattr(ws, "runner_stats",
                        lambda _p: type("S", (), {"games_completed": 1,
                                                  "positions_generated": 500})())
    result = ws.drive_rung(object(), plan, n_workers=4, device=torch.device("cuda:0"),
                           label="t", out=io.StringIO(), sleep=lambda _s: None)
    rounds = plan.rounds_per_rung
    assert probes == [f"w4r{i}" for i in range(rounds)], (
        f"expected ONE probe per round, in round order; got {probes}"
    )
    assert resets == ["cuda:0"] * rounds, (
        f"expected ONE counter reset per round boundary; got {resets}"
    )
    # and the card sink actually reached the readings, per round and independently
    peaks = [r.sampled_peak_bytes for r in result.rounds]
    assert len(set(peaks)) == rounds, f"the card windows are not per-round: {peaks}"


# ══ I-7 — the row that replaces a name claiming more than its assertion ══════════════════
def test_the_report_carries_the_card_total_every_peak_is_measured_against(
    plan: ws.SweepPlan, capsys,
) -> None:
    """D-10's answer and F-WS-6's *"so a reader can do the headroom arithmetic against the report
    itself"*. The row this replaces asserted `torch is not None` and text-scanned a DOCSTRING —
    it never touched the report. That is the overclaiming class this file's own module docstring
    cites."""
    prov = {
        "tool": ws.TOOL, "produced_by": "run5@abc", "config_name": "run5.yaml",
        "config_sha256": "d" * 64, "git_commit": "abc1234", "git_dirty": False,
        "device": "cuda", "torch_version": "2.11", "gpu_name": "X",
        "card_total_bytes": 16 * 1024 ** 3, "declared_allocator_posture": "default",
        "live_allocator_conf": "", "live_allocator_conf_source_var": None,
        "thread_bound": 16, "thread_bound_source": "os.sched_getaffinity(0)",
        "cuda_counters_available": True,
    }
    rounds = tuple(
        ws.RoundReading(index=i, warmup=(i == 0), wall_sec=120.0, games=3, moves=1000,
                        available=True, sampled_peak_bytes=1000 * _MIB,
                        allocator_peak_bytes=900 * _MIB, card_samples=120)
        for i in range(6))
    rung = ws.RungResult(n_workers=4, verdict=ws.PLATEAU, rounds=rounds, refusal=None,
                         produced_by="run5@abc")
    report = ws.build_report(plan=plan, prov=prov, results=[rung], stopped="t",
                             noise_floor_rel_std=0.0)
    assert report["provenance"]["card_total_bytes"] == 16 * 1024 ** 3
    ws.render(report, __import__("sys").stdout)
    assert "card_total=16.0000 GiB" in capsys.readouterr().out, (
        "a peak without the capacity it was taken against is a number a reader cannot size"
    )


def test_the_provenance_docstring_still_states_the_no_host_identifiers_rule() -> None:
    """Kept as its own row, at its own width: this is a claim about the DOCSTRING, and the claim
    about the report is `test_the_report_carries_no_host_identifier_fields`."""
    assert "NO HOST IDENTIFIERS" in (ws.provenance.__doc__ or "")


def test_the_report_carries_the_config_basename_and_never_its_path(
    plan: ws.SweepPlan, tmp_path: Path,
) -> None:
    """R301(d) capture-time redaction, on the report rather than in the docstring. CI gate 17
    cannot see this artifact — it scans the engine tree, and the report lands in the governance
    workspace and in a sitting record — so the redaction has to be in the producer."""
    cfg = tmp_path / "deeply" / "nested" / "run5_local.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("x: 1\n", encoding="utf-8")
    # `seed` joins the stub because `provenance` now carries it: F-RESIT-10's repair makes the
    # ladder's rungs comparable only if they are seeded, and R69 says the report travels with
    # that mechanism rather than leaving a reader to trust it. A stub omitting a field the
    # production path reads raises `AttributeError` — correctly, and it is the same maintenance
    # every new surface has needed here.
    prov = ws.provenance(SimpleNamespace(run_id="r", seed=20260718, identity=SimpleNamespace(
        encoding="gnn_axis_v1", representation="graph"),
        model_dump=lambda: {"allocator_posture": None}),
        cfg, device="cpu", label="t")
    assert prov["config_name"] == "run5_local.yaml"
    assert prov["seed"] == 20260718, (
        "the report must say what the ladder was seeded with; before F-RESIT-10's repair the "
        "honest value of this field was 'unseeded' and nothing in the report said so"
    )
    assert str(tmp_path) not in json.dumps(prov)
    assert prov["config_sha256"] == hashlib.sha256(cfg.read_bytes()).hexdigest(), (
        "the field names SHA-256; a git blob SHA-1 under that name makes a later `sha256sum` "
        "read as a changed config"
    )


# ══ I-11 — the provenance helpers have VALUE rows, not only refusal rows ═════════════════
def test_the_config_digest_is_hashlib_sha256_of_the_bytes(tmp_path: Path) -> None:
    """The field NAMES an algorithm. The first cut used `git hash-object`, which is SHA-1 over
    `blob <len>\\0<content>`; a reader verifying with `sha256sum` gets a mismatch and concludes
    the config changed. A label asserting a fact nobody re-derived is derive-or-delete in a
    different costume."""
    path = tmp_path / "c.yaml"
    path.write_bytes(b"identity:\n  encoding: gnn_axis_v1\n")
    assert ws._sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(ws._sha256(path)) == 64
    assert ws._sha256(tmp_path / "absent.yaml") is None, (
        "an unreadable config must degrade to `null`, never to some other file's digest"
    )


def test_git_degrades_to_none_rather_than_to_a_positive_claim() -> None:
    """`git_dirty` used to read `False` when git could not answer at all — a positive claim of
    cleanliness about a tree nobody looked at, on exactly the host where the tool is launched
    from a scratch directory or a tarball with no `.git`."""
    assert ws._git("rev-parse", "HEAD") is not None, "this repo IS a git tree"
    assert ws._git("this-is-not-a-git-subcommand") is None

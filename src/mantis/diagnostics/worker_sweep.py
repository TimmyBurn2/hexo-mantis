# >300 justify (R8). NO LINE COUNT is stated (G-DFIX-4 / R192(e), derive-or-delete): a
# transcribed tally must be re-edited on every edit, will eventually be wrong, and is then read
# as evidence. This module is ONE CLAIM — "here is the worker count this card wants, here is the
# rule that picked it, and here is every number the rule ran on". The ladder walk, the per-round
# throughput and memory readings, the per-rung verdict and the knee arithmetic are the halves of
# that one claim and are not separable: a pick reported away from the series that produced it is
# a number without its mechanism (R69), and a pick whose selection rule lives in another file is
# a rule a reader must go and trust. `fusion_calibrate.py` carries the identical argument for the
# identical reason, and this tool is its sibling in the same sitting.
"""`python -m mantis.diagnostics.worker_sweep` — what should `selfplay.n_workers` BE on this box?

PHASE W of the re-calibration re-sit (R309(g)), and it exists because of a measured finding —
**the UNBITEABLE CAP**, and not the one it is easy to confuse it with.

`RECAL_EXIT_2026-08-22.md` §11.6: `configs/run5.yaml` mints `selfplay.n_workers: 1`, *"which no
prereg row and no ruling records… at that supply the minted cap can never bite"*. §8 (the STEP 4
burst) had to be run at `n_workers: 4` **for that reason, disclosed with its number** — "run5
mints `n_workers: 1`, and at that supply the cap CANNOT bite… Four workers give supply 32 and the
cap bites". So the caps were fitted at one geometry and validated at another, and neither is the
geometry `configs/run5.yaml` would run.

**WHAT PHASE W IS NOT A CURE FOR, stated so the next reader does not inherit the error.** STEP 4's
falsifier failed on three clauses — the pre-first-step phase peak, the eval-child term at 2.98x,
and the joint peak — and §8.5 attributes the mechanism to inference-phase reservation
fragmentation. **None of them is attributed to `n_workers`, and raising `n_workers` makes the
joint peak WORSE.** Phase W cures the fit-geometry mismatch and the unbiteable cap; it does not
cure the partition, and a reader who believes otherwise will mis-read every number below.

R309(g) states the cure it IS in one line: **caps fit at the config that will run, or they are
stale at birth.** So the worker count is picked FIRST, written into the config on the sitting's
branch, and STEP 1's four terms are measured at that geometry.

**SELF-PLAY ONLY, AND THAT IS STRUCTURAL.** No trainer step may execute before the mint — the
caps are VOID on this host and a training step would cross the voided-caps row. This module
therefore builds its own collaborators and never imports `mantis.run`, which it could not import
anyway: `mantis.run` pulls `mantis.train.orchestrator` at module top level. **The unimportability
and the guarantee are the same fact**, and `tests/diagnostics/test_worker_sweep_reachability.py`
checks it two ways — a whole-closure import walk that counts imports at EVERY scope (a
function-body import is the loophole a top-level-only walk misses) and a fresh-subprocess witness
over `sys.modules`. A subprocess of `python -m mantis.run` was rejected for the opposite reason:
the driver's own closure would read clean while a trainer stepped in the child, which is
structure-not-text (R296(f)) failing while wearing the check's own uniform.

**THE RULE IS NOT IN THIS FILE.** Every threshold — the ladder, the round budget, the window, the
band, the knee percent, and which throughput ranks the rungs — is read from an explicit plan file
(`tools/worker_sweep_plan.toml`). There is no default anywhere here: a missing key raises
`ValueError` naming it and an unknown key raises rather than being ignored. That is R1's shape
applied to a tool, and it is the difference between a pre-registration and a preference — the
plan is committed before the sitting, so `git log` dates the rule against the numbers.

**THE STOPPING RULE IS IMPORTED, NOT RE-WRITTEN.** `eval_child_memory.classify` decides
PLATEAU / GROWING on this tool's series exactly as it decides the eval child's. One stopping rule
in this tree, one place it can be wrong. Its `--plateau-rounds`/`--band-pct` are this tool's
`[stopping_rule]` block, and its refusal semantics carry over whole: **REFUSED is never a
verdict.** Fewer measured rounds than the window needs is a named refusal, never "0 rounds,
plateau".

**TWO SINKS, LARGER GOVERNS — AND THE REPORT SAYS WHICH ONE DID.** Per round the tool records
the CARD's own high-water (sampled every `sampler_interval_sec` through
`mantis.util.device.cuda_device_used_bytes`) AND the caching allocator's
(`torch.cuda.max_memory_allocated`, through the eval child's own probe). These are different
instruments, not two views of one: on the 2026-08-22 host they disagreed by 3.62 GiB of
high-water at matched config across allocator postures. The box block's standing rule is that
the LARGER GOVERNS and the disagreement is a finding; both numbers, their difference and the
WINNER are printed. The winner matters because the two are different KINDS of quantity — `card`
is a LEVEL (context, retained reserve, any co-resident process; unmoved by a peak reset),
`allocator` is a per-round DEMAND peak — so a PLATEAU on a card-governed series means "the
card's committed bytes stopped rising", which is a different sentence from the demand one and
must not be read as it.

**WHERE THE GAME COUNT COMES FROM, AND WHY IT IS NOT AN EVENT.** `runner_stats(pool)` reads the
Rust runner's own counters. The event stream is NOT usable here and the register says why:
`docs/registers/falsified.md` **F-43** — `game_complete` is emitted but DROPPED in production
(the pool is built with `sink=None`) and `iteration_complete.games_total` is gated by
`log_interval`, so neither is a games signal. A sweep that counted games off events would count
zero on a healthy drive. The register was read before this instrument was designed and no other
row transfers.

**EVERY FIGURE CARRIES ITS SAMPLING LIMIT AND ITS PRODUCING RUN** (R287(a)): rounds measured,
wall seconds, card samples taken, and the invocation label, beside the number and not in a
footnote. **Unmeasured rounds are listed, counted, named and excluded BY NAME** — the RECAL-PREP
convention, because silently dropping them biases a series without saying so.

**WHAT A "MOVE" IS, VERIFIED AND NOT ASSUMED (LAW-03).** `moves` is the engine's
`positions_generated`, which increments once per APPLIED COMPOUND TURN — two stones —
immediately after `board.apply_move(move_idx.0, move_idx.1)`
(`crates/mantis-selfplay/src/runner/search_drive.rs:1002`). It is not a ply count. The repo's own
vocabulary already calls this a move (`selfplay/pool_drain.py:105`) and this tool keeps that word
while naming the unit beside every figure.

**TWO MODES, AND THE EXIT CODES BELONG TO BOTH.** `--config` + `--plan` DRIVES; `--select-only`
re-derives a pick from a report this tool already wrote and drives nothing. Neither mode's inputs
are defaulted and the two sets are disjoint — naming an input a mode does not read is itself a
refusal, because it describes a run that did not happen.

    0   a pick was made
    1   no rung PASSED — every measurable rung was GROWING or OOM
    2   REFUSED — the plan, the config, the report or the ladder cannot be answered about

**`rc 1` IS "NO PICK", NOT "THE SITTING FAILED."** R309(f) ends *"never a sitting failure"*, and
that clause governs what a reader may conclude from an OOM. What follows a pickless sweep is the
box block's own pre-registered consequence, and it is the block's to state — not this tool's, and
not a reading anyone takes after seeing the number.

`2` is `eval_child_memory.RC_REFUSED` imported, not re-spelled: a refusal is "you did not give me
something I can answer about", and there is one such code in this family.

**IT NEVER MINTS AND NEVER WRITES A CONFIG.** It measures, it selects under the pre-registered
rule, and it prints the arithmetic with its inputs so the sitting record carries the derivation
rather than the answer alone. Minting is the operator's act (R119) on the sitting's branch
(R308(b)).
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
import tomllib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any

import torch

from mantis.config.loader import load_config
from mantis.config.resolve.allocator_posture import (
    assert_allocator_posture,
    declared_allocator_posture,
    governs_device,
    read_live_allocator_conf,
)
from mantis.config.resolve.coordinator import resolve_coordinator_knobs
from mantis.diagnostics.eval_child_memory import (
    GROWING,
    PLATEAU,
    RC_REFUSED,
    InsufficientRoundsError,
    classify,
)
from mantis.eval.child_memory import make_probe
from mantis.model.identity import net_param_hash
from mantis.selfplay.buffers import BufferKind
from mantis.selfplay.hparams import resolve_pool_encoding
from mantis.selfplay.pool import WorkerPool
from mantis.selfplay.pool_hooks import runner_stats
from mantis.util.determinism import seed_everything
from mantis.util.device import (
    cuda_counters_available,
    cuda_device_total_bytes,
    cuda_device_used_bytes,
    reset_cuda_peak_counters,
)

TOOL = "mantis.diagnostics.worker_sweep"
MARKER = "MANTIS_WORKER_SWEEP"
GIB = 1024 ** 3

#: Rung verdicts beyond the two `classify` returns. `REFUSED` keeps `eval_child_memory`'s
#: meaning exactly — not a verdict, a statement that no verdict is available.
REFUSED = "REFUSED"
OOM = "OOM"
#: A rung that failed for a reason this tool does not model. NOT a tool failure: the ladder is
#: the expensive artifact, and `mantis-bridge` builds with `panic = "unwind"` (R2/LAW-13) exactly
#: so a Rust panic crosses the FFI as an exception rather than aborting the process — which makes
#: it a RUNG failure with a name, not a traceback that loses seventy minutes of box time.
RUNG_ERROR = "RUNG_ERROR"
#: The pool's sole producer died. Its own fail-fast hook (`WorkerPool.check_producer_health`)
#: raises this into view; without the call, the Rust counters keep climbing while nothing reaches
#: the replay buffer, so the rung reports throughput AND a flat memory series — flat BECAUSE it is
#: broken. There is no `SKIPPED_AFTER_OOM` here: R309(f) stops the EXTENSION at an OOM, so the base
#: ladder is walked whole and no rung is ever skipped for one.
PRODUCER_DEAD = "PRODUCER_DEAD"

#: The three series a rung is verdicted on. `governing` is the composite the box block's rule
#: names ("the larger governs"); the other two are the instruments it resolves away, and they are
#: verdicted SEPARATELY because `max()` runs before the stopping rule does.
_SINK_FIELDS = {
    "governing": "governing_peak_bytes",
    "card": "sampled_peak_bytes",
    "allocator": "allocator_peak_bytes",
}

#: The closed token set for `[selection].metric`. Both figures are recorded for every rung
#: whichever is chosen; only the RANKING is single-valued.
METRICS = ("moves_per_min", "games_per_min")

#: R309(f)/(g)'S OWN CONSTANTS, PINNED HERE AND NOT IN THE PLAN FILE. The plan STATES them so
#: the report echoes what it ran under; the loader REFUSES any other value.
#:
#: The asymmetry this closes was a real hole. The plan already refused a rung below 2, citing
#: R309(f)'s REJECTED-1 clause — and then left the OTHER constant from the same sentence ("the
#: smallest rung within 95 percent of the best PASSING rung's throughput") fully editable,
#: validated only for range. A sitting that disliked its pick could have re-run with
#: `knee_pct = 90` and every check would have stayed green, with git history showing only a
#: second plan file. The ruling closes with "No post-hoc movement of any of it"; a constant
#: that can be edited between two runs of the same tool has moved.
#:
#: WHAT IS LEGITIMATELY THE PLAN'S, and the line is not arbitrary: the MEASUREMENT BUDGET
#: (`round_sec`, `warmup_rounds`, `measured_rounds`, `sampler_interval_sec`) and this design's
#: own OPERATIONALIZATIONS of the ruling's words (`band_pct` and `plateau_rounds` make "memory
#: discipline holds" checkable; `min_gain_pct` and `extension_step` make "while gains persist"
#: checkable; `extension_max` is a runaway stop). Those are the prep session's authorship and
#: are amendable by the architect before the sitting. The ruling's own numbers are not.
RULED_RUNGS = (2, 4, 8, 12, 14)
RULED_KNEE_PCT = 95.0

#: THE RANKING METRIC IS ALSO PINNED, and for the same reason one layer down. It is not the
#: ruling's — it is DESIGN AMENDMENT A1's, pre-registered on a measurement taken before any box
#: number existed: on a CPU host at 2 workers a 20 s window produced 22 moves and ZERO completed
#: games, because a game runs to `max_game_moves`. `games_per_min` therefore reads 0 for a HEALTHY
#: rung whenever a round is shorter than a game, and a metric that can be 0 for a healthy rung
#: cannot rank rungs.
#:
#: MEASURED, not argued: with `metric = "games_per_min"` and rounds shorter than a game, the whole
#: ladder ranks identically zero, the knee picks the SMALLEST rung at rc 0, and the ladder-stop
#: line reads "gains no longer persist" — while the moves column says the top rung is 3.7x faster.
#: A pre-registration with a measured basis and no enforcement is a preference; one token in a
#: plan file made the pick arbitrary with every check green.
PREREG_METRIC = "moves_per_min"

#: THE DETERMINISM CONTROL'S FORMER BAND. **SUPERSEDED BY R317(c)** — kept only as a historical
#: constant (old reports cite it, and `render_determinism_control`/the sitting record still print
#: the spread beside it for continuity). R315(c)(i) pinned it at 1%, measured 0.5821% AGREE
#: engine-side; RECAL-SITTING-3 measured the SAME check live on a box at 3.9258%, DIVERGED — the
#: band was a cross-regime carry (R317(b)), not a property of the seeding. The GATE is now
#: net-parameter-hash equality (`_hash_gate`), no band; this constant no longer gates anything.
RULED_DETERMINISM_BAND_PCT = 1.0

#: The band's UPPER bound. The plan file argues 1.0 at length (5.0 would permit 774 MiB of growth
#: per round on this card, against a sitting whose falsifier fired on 343 MiB). A bound is what
#: makes that argument load-bearing instead of decorative: at `band_pct = 500` every rung
#: PLATEAUs and the memory gate is simply off, with nothing in the tree noticing. The architect's
#: amendment right lives BELOW this bound, which is where the reasoning also lives.
MAX_BAND_PCT = 5.0
#: A window of one round makes PLATEAU mean "the final round did not exceed the max of all before
#: it", which is not a convergence test at all — the plan file's own words, now enforced.
MIN_PLATEAU_ROUNDS = 2

#: The plan's exact shape. Missing section, missing key and unknown key are each a named
#: `ValueError`: a silently-ignored key is how a rule someone believed was in force turns out
#: never to have been read.
PLAN_SHAPE: dict[str, tuple[str, ...]] = {
    "provenance": ("prereg_ruling", "prereg_recorded", "authored_by", "note"),
    "ladder": ("rungs", "extension_step", "extension_max", "min_gain_pct"),
    "rounds": ("warmup_rounds", "measured_rounds", "round_sec", "sampler_interval_sec"),
    "stopping_rule": ("plateau_rounds", "band_pct"),
    "selection": ("knee_pct", "metric"),
}


class SweepRefusal(Exception):
    """A named refusal that exits `RC_REFUSED` and emits no pick.

    An exception rather than a `sys.exit` at the raise site so every refusal takes ONE exit
    path with one code and one destination — `fusion_calibrate.CalibrationRefusal`'s reason: a
    refusal that sometimes lands on stdout would be parsed as a report by whatever reads it.
    """


class _Discard:
    """A sink for the eval probe's own marker channel.

    The probe's counter reads, availability decision and running maxima are reused whole; its
    `MANTIS_EVAL_MEM` line is NOT, and is discarded here rather than re-emitted. A sweep log
    carrying eval markers would be read by `mantis.diagnostics.eval_child_memory` as an eval
    drive, which it is not — and that reader is fail-closed precisely so it never reports about
    a file it did not understand.
    """

    def write(self, _text: str) -> int:
        return 0

    def flush(self) -> None:
        return None


# ══ the plan ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SweepPlan:
    """The pre-registered rule, whole. Frozen: a resolved run-scoped constant a consumer could
    rebind is a second authority with extra steps."""

    rungs: tuple[int, ...]
    extension_step: int
    extension_max: int
    min_gain_pct: float
    warmup_rounds: int
    measured_rounds: int
    round_sec: float
    sampler_interval_sec: float
    plateau_rounds: int
    band_pct: float
    knee_pct: float
    metric: str
    provenance: dict[str, Any]

    @property
    def rounds_per_rung(self) -> int:
        return self.warmup_rounds + self.measured_rounds


def _require(section: str, keys: tuple[str, ...], block: Any, path: Path) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise ValueError(f"{path}: [{section}] is missing or is not a table")
    missing = [k for k in keys if k not in block]
    unknown = [k for k in block if k not in keys]
    if missing:
        raise ValueError(
            f"{path}: [{section}] is missing required key(s) {missing}. This plan has no "
            "defaults: a threshold nobody wrote down is a rule nobody chose, applied to a "
            "measurement a mint will rest on."
        )
    if unknown:
        raise ValueError(
            f"{path}: [{section}] carries unknown key(s) {unknown}. Refusing rather than "
            "ignoring them — a silently-ignored key is a rule someone believed was in force."
        )
    return dict(block)


def load_plan(path: str | Path) -> SweepPlan:
    """Load and validate the sweep plan. Every refusal names what is wrong and why it matters."""
    plan_path = Path(path)
    raw = tomllib.loads(plan_path.read_text(encoding="utf-8"))
    unknown_sections = [k for k in raw if k not in PLAN_SHAPE]
    if unknown_sections:
        raise ValueError(f"{plan_path}: unknown section(s) {unknown_sections}")
    blocks = {name: _require(name, keys, raw.get(name), plan_path)
              for name, keys in PLAN_SHAPE.items()}
    ladder, rounds = blocks["ladder"], blocks["rounds"]
    rule, selection = blocks["stopping_rule"], blocks["selection"]
    # TYPES ARE NOT COERCED. `float("95.0")` and `int(2.9)` both succeed, so a plan could STATE
    # one thing and RUN another — `rungs = [2.9, ...]` truncating to the ruled ladder is the
    # measured instance. A pre-registration whose printed form differs from its executed form is
    # not a pre-registration.
    for section, key, value, want in (
        ("ladder", "rungs", tuple(ladder["rungs"]), int),
        ("ladder", "extension_step", ladder["extension_step"], int),
        ("ladder", "extension_max", ladder["extension_max"], int),
        ("ladder", "min_gain_pct", ladder["min_gain_pct"], float),
        ("rounds", "warmup_rounds", rounds["warmup_rounds"], int),
        ("rounds", "measured_rounds", rounds["measured_rounds"], int),
        ("rounds", "round_sec", rounds["round_sec"], float),
        ("rounds", "sampler_interval_sec", rounds["sampler_interval_sec"], float),
        ("stopping_rule", "plateau_rounds", rule["plateau_rounds"], int),
        ("stopping_rule", "band_pct", rule["band_pct"], float),
        ("selection", "knee_pct", selection["knee_pct"], float),
    ):
        values = value if isinstance(value, tuple) else (value,)
        for item in values:
            ok = isinstance(item, int) and not isinstance(item, bool) if want is int else (
                isinstance(item, (int, float)) and not isinstance(item, bool))
            if not ok:
                raise ValueError(
                    f"{plan_path}: [{section}].{key} must be written as "
                    f"{'an integer' if want is int else 'a number'}, got {item!r} "
                    f"({type(item).__name__}). Nothing here is coerced: a plan that states one "
                    "value and runs another is not a pre-registration."
                )
    rungs = tuple(int(r) for r in ladder["rungs"])
    if not rungs:
        raise ValueError(f"{plan_path}: [ladder].rungs is empty — there is no ladder to walk")
    if any(r < 2 for r in rungs):
        raise ValueError(
            f"{plan_path}: [ladder].rungs contains a rung below 2 ({sorted(rungs)}). R309(f) "
            "REJECTS n_workers = 1 as far too low; the rule cannot be un-made by editing the "
            "file it is written in."
        )
    if list(rungs) != sorted(set(rungs)):
        raise ValueError(f"{plan_path}: [ladder].rungs must be strictly increasing: {rungs}")
    if rungs != RULED_RUNGS:
        raise ValueError(
            f"{plan_path}: [ladder].rungs is {list(rungs)}; R309(g) names the base ladder "
            f"{list(RULED_RUNGS)} and R309(f) closes with 'No post-hoc movement of any of it'. "
            "Extension past the top of that ladder is what the ruling permits, and it is the "
            "EXTENSION keys that carry it — not an edit to the base ladder."
        )
    if float(selection["knee_pct"]) != RULED_KNEE_PCT:
        raise ValueError(
            f"{plan_path}: [selection].knee_pct is {selection['knee_pct']}; R309(f) fixes the "
            f"knee rule at {RULED_KNEE_PCT:g} percent of the best PASSING rung's throughput. "
            "This is the ruling's own constant, in the same sentence as 'No post-hoc movement "
            "of any of it', and it is not a plan knob."
        )
    if int(ladder["extension_max"]) < max(rungs):
        raise ValueError(
            f"{plan_path}: [ladder].extension_max {ladder['extension_max']} is below the "
            f"highest base rung {max(rungs)} — the ceiling would refuse the ladder itself"
        )
    if int(rounds["measured_rounds"]) <= int(rule["plateau_rounds"]):
        raise ValueError(
            f"{plan_path}: [rounds].measured_rounds {rounds['measured_rounds']} is not GREATER "
            f"than [stopping_rule].plateau_rounds {rule['plateau_rounds']}. Fewer would make a "
            "verdict unreachable; EQUAL is worse, because it is reachable and wrong: `classify` "
            "then takes its degenerate branch (`running = peaks[0]`, the window covering the "
            "whole series) and the rule silently becomes 'no round exceeds the FIRST by more "
            "than the band', with no history outside the window at all. A plan that cannot "
            "reach a HONEST verdict is a refusal at LOAD, not a surprise hours into a sitting."
        )
    for section, key, value in (
        ("rounds", "round_sec", rounds["round_sec"]),
        ("rounds", "sampler_interval_sec", rounds["sampler_interval_sec"]),
        ("ladder", "extension_step", ladder["extension_step"]),
    ):
        if float(value) <= 0:
            raise ValueError(f"{plan_path}: [{section}].{key} must be positive, got {value}")
    # THE TWO KNOBS THAT DECIDE PASS/FAIL FOR EVERY RUNG, bounded on BOTH sides. The plan file
    # argues both numbers at length and neither argument was enforced: `band_pct = 500` turns the
    # memory gate off entirely and `plateau_rounds = 1` makes PLATEAU mean "the last round did not
    # exceed the max of all before it". Range-checking only the sign is the exact asymmetry the
    # `knee_pct` pin closed one level up.
    if not 0.0 <= float(rule["band_pct"]) <= MAX_BAND_PCT:
        raise ValueError(
            f"{plan_path}: [stopping_rule].band_pct must be in [0, {MAX_BAND_PCT:g}], got "
            f"{rule['band_pct']}. Above that bound the memory conjunct R309(f) makes the PASS "
            "condition is off, and every rung plateaus by construction."
        )
    if int(rule["plateau_rounds"]) < MIN_PLATEAU_ROUNDS:
        raise ValueError(
            f"{plan_path}: [stopping_rule].plateau_rounds must be at least "
            f"{MIN_PLATEAU_ROUNDS}, got {rule['plateau_rounds']}. A window of one round is not a "
            "convergence test — and `classify` refuses 0 or less with a DIFFERENT error, hours "
            "into a sitting, after the first rung has been driven."
        )
    if not 0.0 < float(selection["knee_pct"]) <= 100.0:
        raise ValueError(
            f"{plan_path}: [selection].knee_pct must be in (0, 100], got {selection['knee_pct']}"
        )
    if float(rule["band_pct"]) < 0.0 or float(ladder["min_gain_pct"]) < 0.0:
        raise ValueError(f"{plan_path}: band_pct and min_gain_pct may not be negative")
    if selection["metric"] not in METRICS:
        raise ValueError(
            f"{plan_path}: [selection].metric {selection['metric']!r} is not one of {METRICS}"
        )
    if selection["metric"] != PREREG_METRIC:
        raise ValueError(
            f"{plan_path}: [selection].metric is {selection['metric']!r}; the ranking metric is "
            f"PRE-REGISTERED as {PREREG_METRIC!r} (DESIGN amendment A1, on a measurement taken "
            "before any box number existed). Both figures are recorded and printed for every "
            "rung either way — only the RANKING is single-valued, and it is not a plan knob. "
            "With rounds shorter than a game, `games_per_min` ranks every rung at zero and the "
            "knee then picks the smallest rung at rc 0 while the moves column disagrees."
        )
    if int(rounds["warmup_rounds"]) < 0:
        raise ValueError(f"{plan_path}: [rounds].warmup_rounds may not be negative")
    return SweepPlan(
        rungs=rungs,
        extension_step=int(ladder["extension_step"]),
        extension_max=int(ladder["extension_max"]),
        min_gain_pct=float(ladder["min_gain_pct"]),
        warmup_rounds=int(rounds["warmup_rounds"]),
        measured_rounds=int(rounds["measured_rounds"]),
        round_sec=float(rounds["round_sec"]),
        sampler_interval_sec=float(rounds["sampler_interval_sec"]),
        plateau_rounds=int(rule["plateau_rounds"]),
        band_pct=float(rule["band_pct"]),
        knee_pct=float(selection["knee_pct"]),
        metric=str(selection["metric"]),
        provenance=blocks["provenance"],
    )


# ══ markers ══════════════════════════════════════════════════════════════════════════════
def emit_marker(record: dict[str, Any], *, out: Any) -> None:
    """Write one `MANTIS_WORKER_SWEEP` line. Flushed, because a sweep can be killed at a rung
    and a buffered marker is a measurement that did not survive the thing it was measuring."""
    print(f"{MARKER} {json.dumps(record, sort_keys=True)}", file=out, flush=True)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """`json.loads` takes the LAST of duplicate keys, silently. This reader is the recovery path
    for a sweep that was killed at a rung — i.e. exactly the situation where a log has been
    concatenated or hand-repaired — so a payload carrying `{"n_workers": 2, "n_workers": 99}`
    would yield a wrong record set with no refusal. Everything else in this function fails
    closed; this is the hole in it."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(
                f"{MARKER} payload carries the key {key!r} twice. Refusing rather than taking "
                "the last one: a concatenated or hand-repaired log is exactly what this reader "
                "is for."
            )
        seen[key] = value
    return seen


def _no_non_finite(token: str) -> Any:
    """`NaN`, `Infinity` and `-Infinity` are values to `json.loads` by default. A peak of `NaN`
    is not a measurement and must not travel as one."""
    raise ValueError(f"{MARKER} payload carries the non-finite constant {token!r}")


def parse_sweep_markers(text: str) -> list[dict[str, Any]]:
    """Recover the marker records from a captured sweep log. FAILS CLOSED.

    No markers RAISES; a marker whose payload is not JSON RAISES. Both are the `peaks.py`
    lesson: a reader that guesses at a shape it does not recognise produces a number nobody can
    distinguish from a measurement, and that is how 1 392 GiB was once reported for a 16 GiB card.
    """
    records: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(MARKER):
            continue
        body = line[len(MARKER):].strip()
        try:
            parsed = json.loads(body, object_pairs_hook=_no_duplicate_keys,
                                parse_constant=_no_non_finite)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{MARKER} line carries no readable json payload ({exc}); refusing to guess at "
                f"it: {body[:120]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{MARKER} payload must be a json object, got {type(parsed).__name__}")
        records.append(parsed)
    if not records:
        raise ValueError(
            f"no {MARKER} lines found. This reader does not fall back to guessing at a file's "
            "shape: the substitute reading would be indistinguishable from a measurement."
        )
    return records


# ══ the card-level sink ══════════════════════════════════════════════════════════════════
class CardSampler:
    """Background running-maximum of CARD used bytes. The second instrument, not a second view.

    Runs only where the counters exist. `peak()` and `samples()` are per-WINDOW: `reset()` opens
    a new one at a round boundary the driver owns and declares, which is the condition under
    which a reset is legitimate at all (see `mantis.util.device.reset_cuda_peak_counters`).
    """

    def __init__(self, device: str, interval_sec: float) -> None:
        self._device = device
        self._interval = interval_sec
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak: int | None = None
        self._samples = 0
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                used = cuda_device_used_bytes(self._device)
                with self._lock:
                    self._peak = used if self._peak is None else max(self._peak, used)
                    self._samples += 1
                self._stop.wait(self._interval)
        except Exception as exc:  # noqa: BLE001 — recorded and surfaced, never swallowed
            # A DAEMON THREAD THAT DIES SILENTLY IS A MEASUREMENT THAT STOPPED. Unguarded, the
            # exception printed to stderr — into the same stream the markers go to — and the
            # driver never learned: `window()` kept returning the last peak of the open window
            # and, after the next `reset()`, `(None, 0)` forever. The rung's series then switched
            # INSTRUMENT mid-flight, from card-governed to allocator-only, and a series that FELL
            # by 9.3 GiB and then rose 47% verdicted PLATEAU. That is the exact hazard
            # `reset_cuda_peak_counters` names — *a figure that FELL reads as memory released* —
            # produced by the tool that docstring was written for. The death is recorded and the
            # rung is refused BY NAME, which is this instrument's own convention for a round it
            # could not measure.
            with self._lock:
                self._error = exc

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sweep-card-sampler")
        self._thread.start()

    def reset(self) -> None:
        with self._lock:
            self._peak = None
            self._samples = 0

    def window(self) -> tuple[int | None, int]:
        with self._lock:
            return self._peak, self._samples

    def error(self) -> BaseException | None:
        """The sampler thread's cause of death, or `None` while it is alive."""
        with self._lock:
            return self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


# ══ readings ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RoundReading:
    """One measurement round. `available` False means the host had no CUDA counters — the round
    is still LISTED and COUNTED, and is excluded from the verdict by name."""

    index: int
    warmup: bool
    wall_sec: float
    games: int
    moves: int
    available: bool
    sampled_peak_bytes: int | None
    allocator_peak_bytes: int | None
    card_samples: int

    @property
    def governing_peak_bytes(self) -> int | None:
        """The LARGER of the two sinks — the box block's standing rule, which needs both."""
        peaks = [p for p in (self.sampled_peak_bytes, self.allocator_peak_bytes) if p is not None]
        return max(peaks) if peaks else None

    @property
    def governing_sink(self) -> str | None:
        """WHICH sink governed, reported because the two are different KINDS of quantity.

        `card` is `total - free`: a LEVEL, including the ~0.26 GiB CUDA context, whatever the
        caching allocator retains but has not handed out, and any co-resident process. It is
        unaffected by `reset_peak_memory_stats`. `allocator` is a per-round DEMAND peak that the
        round boundary zeroes. `max()` of the two is the block's standing rule and it is the
        right GATE, but it silently changes what the series MEANS depending on which side wins —
        so the report says which side won, per round and in aggregate, rather than leaving a
        reader to infer that a PLATEAU on a card-level series means "the card's committed bytes
        stopped rising" and not "per-round demand stopped rising".
        """
        if self.governing_peak_bytes is None:
            return None
        return ("card" if self.sampled_peak_bytes is not None
                and self.governing_peak_bytes == self.sampled_peak_bytes else "allocator")

    @property
    def sink_disagreement_bytes(self) -> int | None:
        """How far apart the two instruments were. The block's own rule ends "and the
        disagreement is a finding", which needs the difference reported to be actionable."""
        if self.sampled_peak_bytes is None or self.allocator_peak_bytes is None:
            return None
        return abs(self.sampled_peak_bytes - self.allocator_peak_bytes)

    @property
    def moves_per_min(self) -> float:
        return (self.moves / self.wall_sec) * 60.0 if self.wall_sec > 0 else 0.0

    @property
    def games_per_min(self) -> float:
        return (self.games / self.wall_sec) * 60.0 if self.wall_sec > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "warmup": self.warmup, "wall_sec": round(self.wall_sec, 3),
            "games": self.games, "moves": self.moves, "available": self.available,
            "sampled_peak_bytes": self.sampled_peak_bytes,
            "allocator_peak_bytes": self.allocator_peak_bytes,
            "governing_peak_bytes": self.governing_peak_bytes,
            "governing_sink": self.governing_sink,
            "sink_disagreement_bytes": self.sink_disagreement_bytes,
            "card_samples": self.card_samples,
            "moves_per_min": round(self.moves_per_min, 4),
            "games_per_min": round(self.games_per_min, 4),
        }


@dataclass(frozen=True)
class RungResult:
    """One rung's whole record — the verdict AND everything the verdict was taken on."""

    n_workers: int
    verdict: str
    rounds: tuple[RoundReading, ...]
    refusal: str | None
    produced_by: str
    #: R317(c)(i): the constructed net's parameter hash, post-seed pre-play. `None` when the rung
    #: never reached a built net (OOM/error during construction) — the gate excludes those rather
    #: than treating an absence as an equality or an inequality.
    net_param_hash: str | None = None

    def measured_peaks(self) -> list[int]:
        """The governing peaks of the measured rounds, oldest first.

        The narrowing lives HERE and not at each call site: `measured` already guarantees
        `governing_peak_bytes is not None`, and re-deriving that guarantee three times is three
        places it can be got wrong (and, as the type checker points out, three places where a
        `None` can silently reach `max()` or the stopping rule).
        """
        return [peak for r in self.measured if (peak := r.governing_peak_bytes) is not None]

    @property
    def measured(self) -> tuple[RoundReading, ...]:
        return tuple(r for r in self.rounds
                     if not r.warmup and r.available and r.governing_peak_bytes is not None)

    @property
    def scored(self) -> tuple[RoundReading, ...]:
        """Rounds a THROUGHPUT figure is taken over: non-warm-up, measured or not. Throughput
        does not need CUDA counters and is not thrown away for want of them."""
        return tuple(r for r in self.rounds if not r.warmup)

    def throughput(self, metric: str) -> float:
        wall = sum(r.wall_sec for r in self.scored)
        if wall <= 0:
            return 0.0
        total = sum(r.moves if metric == "moves_per_min" else r.games for r in self.scored)
        return (total / wall) * 60.0

    def spread(self, metric: str) -> dict[str, float | None]:
        """Min / median / max of the per-round rate across the scored rounds.

        A POINT ESTIMATE WITH NO DISPERSION IS THE OMISSION THE SITTING ALREADY RECORDED:
        `RECAL_EXIT_2026-08-22.md` §11.9 — *"the A/B in STEP 1b carries no variance
        information… it establishes a difference between postures and nothing about the
        stability of either."* The knee band is 5 percent; a reader who cannot see the spread
        cannot tell a 5 percent difference from the noise it may be, and LAW-09 carries the same
        discipline for benches. Reported, never gating: the ruling's rule is the ruling's.
        """
        values = sorted(getattr(r, metric) for r in self.scored)
        if not values:
            return {"min": None, "median": None, "max": None, "n_rounds": 0}
        return {"min": round(values[0], 4), "median": round(median(values), 4),
                "max": round(values[-1], 4), "n_rounds": len(values)}

    def series(self, sink: str) -> list[int]:
        """One sink's own per-round peaks, oldest first, warm-up and unmeasured rounds excluded."""
        key = _SINK_FIELDS[sink]
        return [value for r in self.measured if (value := getattr(r, key)) is not None]

    def sink_verdicts(self, *, plateau_rounds: int, band_pct: float) -> dict[str, str]:
        """The stopping rule applied to EACH sink independently, plus the governing composite.

        WHY EACH SINK AND NOT ONLY THE COMPOSITE — the defect this closes, measured. `governing`
        is `max(card, allocator)`, and `max()` resolves the pair BEFORE the stopping rule ever
        runs. On a real box the card level RATCHETS: torch's caching allocator does not return
        reserved blocks to the driver, so `total - free` climbs to its high-water and then sits
        there. A rung whose allocator DEMAND grows 3.3x underneath a flat 15.4 GiB card level
        therefore verdicts PLATEAU on the composite, passes, and enters the knee set — while
        `classify` on the allocator series alone says GROWING.

        That is `RECAL_EXIT_2026-08-22.md` §11b's own failure — *a term measured by watching
        until it looks flat is not a bound* — reproduced inside the instrument built to end it,
        and it is the direction the composite eats rather than the one an oracle first covered.

        THE BLOCK'S RULE IS NOT CHANGED. "The larger governs" still decides the FIGURE. What
        changes is that the rule is asked of each series rather than of the resolved pair, and
        **growth on any sink fails the rung** — strictly more conservative, and it can only ever
        refuse a rung the composite would have passed.
        """
        verdicts: dict[str, str] = {}
        for sink in _SINK_FIELDS:
            series = self.series(sink)
            if not series:
                continue
            try:
                verdicts[sink] = classify(series, plateau_rounds=plateau_rounds,
                                          band_pct=band_pct)
            except InsufficientRoundsError:
                verdicts[sink] = REFUSED
        return verdicts

    def trailing_rise_pct(self, sink: str, plateau_rounds: int) -> float | None:
        """Percentage change from the first to the last round of the trailing window.

        A TREND TEST, replacing a strict round-on-round monotone check that a single repeated or
        dipping value silenced — and real memory series are not strictly monotone, so the shape
        the strict flag caught is the one a real series is least likely to have. DISCLOSED,
        NEVER GATING: `classify` compares the window against the running maximum BEFORE it, so a
        raised baseline can hide a climb underneath; this reports the climb the rule cannot see
        rather than changing a rule the ruling names.
        """
        series = self.series(sink)
        if len(series) < plateau_rounds or plateau_rounds < 2 or series[-plateau_rounds] <= 0:
            return None
        window = series[-plateau_rounds:]
        return (window[-1] - window[0]) / window[0] * 100.0

    def as_dict(self, metric: str, *, plateau_rounds: int | None = None,
                band_pct: float | None = None) -> dict[str, Any]:
        governing = self.measured_peaks()
        disagreements = [r.sink_disagreement_bytes for r in self.measured
                         if r.sink_disagreement_bytes is not None]
        sinks = [r.governing_sink for r in self.measured]
        return {
            "n_workers": self.n_workers, "verdict": self.verdict, "refusal": self.refusal,
            "produced_by": self.produced_by, "net_param_hash": self.net_param_hash,
            "rounds_total": len(self.rounds), "rounds_measured": len(self.measured),
            "rounds_unmeasured": len(self.scored) - len(self.measured),
            "wall_sec": round(sum(r.wall_sec for r in self.rounds), 3),
            "moves_per_min": round(self.throughput("moves_per_min"), 4),
            "games_per_min": round(self.throughput("games_per_min"), 4),
            "moves_per_min_spread": self.spread("moves_per_min"),
            "games_per_min_spread": self.spread("games_per_min"),
            "ranking_metric": metric,
            "rung_peak_bytes": max(governing) if governing else None,
            "governing_sink_counts": {sink: sinks.count(sink) for sink in set(sinks)},
            "max_sink_disagreement_bytes": max(disagreements) if disagreements else None,
            "sink_verdicts": (
                {} if plateau_rounds is None or band_pct is None
                else self.sink_verdicts(plateau_rounds=plateau_rounds, band_pct=band_pct)),
            "trailing_rise_pct": (
                {} if plateau_rounds is None
                else {sink: value for sink in _SINK_FIELDS
                      if (value := self.trailing_rise_pct(sink, plateau_rounds)) is not None}),
            "rounds": [r.as_dict() for r in self.rounds],
        }


# ══ the drive ════════════════════════════════════════════════════════════════════════════
def thread_bound() -> tuple[int, str]:
    """The box's MEASURED thread count and which call answered.

    R309(f) bounds the ladder above by "the box's measured thread count". Measured, never
    typed: `sched_getaffinity` is the honest one inside a container with a CPU mask, and
    `cpu_count` is the fallback where the OS does not offer it. WHICH one answered is stamped,
    because on a box the two can differ and a reader must not have to guess which bound stopped
    the ladder.
    """
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        return len(getter(0)), "os.sched_getaffinity(0)"
    return int(os.cpu_count() or 1), "os.cpu_count()"


def _select_sweep_buffer(config: Any, spec: Any, capacity: int) -> Any:
    """The replay buffer for a trainer-free pool, dispatched through the buffer layer's ONE
    authority (`BufferKind.from_spec`, repo_design §3) — a CLOSED match that raises on an
    unknown representation, which is a stronger dispatch than the run root's string comparison.

    A SECOND SITE, DISCLOSED AND NOT TALKED PAST. The run's selector is `mantis.run::
    _select_buffer`, unreachable from here for the same reason that makes this tool
    trainer-free (`mantis.run` imports `mantis.train.orchestrator` at module top level). No
    literal appears on this path — the graph ring's visit geometry is DERIVED by the engine's
    own `derived_hexg_visit_capacity` from the config's sims regime, which is R255/ADJ-D34's
    actual requirement — and no encoding literal sits in a default position, so CI gate 11
    stays quiet for `_select_buffer`'s own measured reason. The honest repair is to move that
    function into `mantis.selfplay.buffers` and re-point three shipped oracles; that is a
    change to the production boot path and it is FILED (`WORKER_SWEEP_FINDINGS.md` F-WS-1),
    not taken quietly here.
    """
    # WHICH REFUSAL, stated because it is NOT the boot's. `_select_buffer`'s third arm raises
    # `RepresentationRouteError`; `BufferKind.from_spec` delegates to `is_graph_representation`
    # and an unknown or absent representation raises `RepresentationMismatch`. Both are the same
    # LAW-11 refusal — no dense-by-default arm — and they are deliberately not unified here: a
    # diagnostics tool borrowing the BOOT's error family would put a run-fatal route's exception
    # on a path no run takes. The reader of a sweep traceback should see the buffer layer's own
    # error, and `tests/test_run_buffer_route.py` scans the boot's docstring, not this one.
    kind = BufferKind.from_spec(spec)
    if kind is BufferKind.GRAPH:
        from mantis._engine import HexgBuffer, derived_hexg_visit_capacity

        sp = config.selfplay
        pc = sp.playout_cap
        visit_capacity = derived_hexg_visit_capacity(
            n_simulations=sp.mcts.n_simulations, standard_sims=pc.standard_sims,
            fast_prob=pc.fast_prob, fast_sims=pc.fast_sims,
            full_search_prob=pc.full_search_prob, n_sims_quick=pc.n_sims_quick,
            n_sims_full=pc.n_sims_full, leaf_batch_size=sp.leaf_batch_size,
            completed_q_values=sp.completed_q_values,
        )
        return HexgBuffer(capacity, config.identity.encoding, visit_capacity)
    from mantis._engine import ReplayBuffer

    return ReplayBuffer(capacity, config.identity.encoding)


def build_sweep_net(config: Any, arch: Any, device: torch.device) -> Any:
    """SEED, then build this rung's network. The seeding is the point, and it is a REPAIR.

    **F-RESIT-10, measured at the 2026-08-27 re-sit.** This tool built a fresh `build_net(arch)`
    per rung from an UNSEEDED RNG, so every rung of the pre-registered ladder raced a DIFFERENT
    random network. On an unbounded board a network's policy decides how far stones spread, which
    decides the graph's node and edge counts, which decides what every fused forward costs — so
    the ladder's ranking column was a function of `n_workers` **and an uncontrolled draw**. The
    knee rule compares rungs, and a column carrying a term resampled between rungs cannot be
    compared: on the measured ladder the pick moved six rungs depending on which net a rung drew.

    **Size of the effect, measured rather than argued.** At a FIXED worker count, throughput
    varied **1.60x** on the draw alone (277 seeded / 444 on a lucky unseeded draw at
    `n_workers = 4`), against **2.39x** for the entire ladder from 2 workers to 16. The noise was
    roughly 60 % of the signal, constant within a rung and resampled between them — the worst
    possible shape for a rule that compares rungs.

    **R30a's ONE-BOOT-SITE rule is not crossed, and the distinction is not a technicality.**
    `mantis.run.build_run_collaborators` remains the only place a RUN seeds, once, before any
    RNG-consuming object exists. This process is not a run: it builds MANY pools and its entire
    output is a COMPARISON between them, so each one must start from the same RNG state or the
    comparison is not one. `seed_everything` is documented idempotent and is called here per rung,
    from the config's own `seed` — never a literal, so a re-minted seed follows without an edit.

    Placed immediately before the ONE RNG consumer on this path, which is `build_net`.
    """
    from mantis.model import build_net

    seed_everything(int(config.seed))
    return build_net(arch).to(device)


def _hash_gate(pairs: list[tuple[str, str | None]]) -> dict[str, Any]:
    """R317(c)(i), THE GATE: net-parameter hash equality, no band, over whichever drives built
    a net. `pairs` is (label, hash-or-None) per drive; a drive that never reached a built net
    (OOM/error during construction) is excluded rather than counted as an agreement or a
    divergence — the gate answers "did the nets that exist agree", not "did every drive run".
    """
    present = {label: h for label, h in pairs if h is not None}
    if len(present) < 2:
        return {"verdict": REFUSED, "hashes": present,
                "reason": f"only {len(present)} drive(s) built a net to hash; R317(c)(i) needs "
                          "at least two to gate on"}
    distinct = sorted(set(present.values()))
    if len(distinct) > 1:
        return {"verdict": DIVERGED, "hashes": present,
                "reason": f"net-parameter hashes differ across drives (R317(c)(i)): "
                          f"{len(distinct)} distinct value(s) over {len(present)} drive(s) — "
                          "the same seed did not build the same net"}
    return {"verdict": AGREE, "hashes": present,
            "reason": f"net-parameter hashes agree across all {len(present)} drive(s) that "
                      "built one (R317(c)(i))"}


def build_sweep_pool(config: Any, *, n_workers: int, device: torch.device) -> WorkerPool:
    """Build the self-play collaborators for ONE rung — model, buffer, pool. No trainer.

    `n_workers` arrives through `WorkerPool`'s own declared override seam
    (`SelfPlayHParams.from_config(config, n_workers)`), never by editing a config on disk:
    varying it is the measurement, and the config is otherwise the run's own, unchanged.
    """
    from mantis.model import arch_from_spec_and_config

    raw = config.model_dump()
    resolved = resolve_pool_encoding(raw, arch=None)
    arch = arch_from_spec_and_config(resolved.registry_spec, raw)
    model = build_sweep_net(config, arch, device)
    capacity = int(resolve_coordinator_knobs(config.train).capacity)
    buffer = _select_sweep_buffer(config, resolved.registry_spec, capacity)
    return WorkerPool(model=model, config=raw, device=device, replay_buffer=buffer,
                      n_workers=n_workers, arch=arch)


def _verdict_for(rounds: tuple[RoundReading, ...], plan: SweepPlan) -> tuple[str, str | None]:
    """Apply the IMPORTED stopping rule to EVERY sink, and fail the rung on growth in ANY of them.

    REFUSED IS NEVER A VERDICT — `eval_child_memory`'s rule, carried whole. Too few measured
    rounds is a named refusal about the drive, not a statement about the memory, and the message
    is `classify`'s own.

    GROWTH ON ANY SINK FAILS THE RUNG. `RungResult.sink_verdicts` carries the measured argument;
    the short version is that `governing = max(card, allocator)` resolves the pair before the rule
    runs, and on a card whose reserve has saturated the composite is the flat card level in nearly
    every round. This gate is strictly more conservative than the composite alone: it can only
    refuse a rung the composite would have passed, never pass one it would have failed.
    """
    stub = RungResult(n_workers=0, verdict="", rounds=rounds, refusal=None, produced_by="")
    peaks = stub.series("governing")
    try:
        classify(peaks, plateau_rounds=plan.plateau_rounds, band_pct=plan.band_pct)
    except InsufficientRoundsError as exc:
        return REFUSED, str(exc)
    verdicts = stub.sink_verdicts(plateau_rounds=plan.plateau_rounds, band_pct=plan.band_pct)
    growing = sorted(sink for sink, v in verdicts.items() if v == GROWING)
    if growing:
        return GROWING, (
            f"growth seen on: {', '.join(growing)} (all sink verdicts: {verdicts}). The block's "
            "rule decides which FIGURE governs; it does not decide which SERIES the stopping "
            "rule is asked about, and a rung that grows on either instrument has not converged."
        )
    return PLATEAU, None


def drive_rung(
    config: Any, plan: SweepPlan, *, n_workers: int, device: torch.device, label: str,
    out: Any, sleep: Any = time.sleep,
) -> RungResult:
    """Run one rung: build a fresh pool, walk the rounds, verdict the series, tear down.

    FRESH POOL PER RUNG, deliberately: a rung that inherited the previous rung's allocator
    state would be measuring the ladder's history rather than its own worker count.

    **EVERY FAILURE IS A RUNG VERDICT, NOT A TRACEBACK.** The ladder is the expensive artifact —
    the base ladder alone is over an hour of rented box — so an exception this tool does not model
    fails ITS RUNG and the walk continues. Only `torch.OutOfMemoryError` was caught at first, and
    the failure surface is wider than that by construction: `mantis-bridge` builds with
    `panic = "unwind"` (R2/LAW-13) exactly so a Rust panic crosses the FFI as an exception, and an
    escaping `RuntimeError` reached the interpreter as **shell rc 1** — which this tool's own
    contract reserves for "no rung PASSED". A crash must never be able to present as a measured
    memory result.
    """
    device_str = str(device)
    counters = cuda_counters_available(device_str)
    readings: list[RoundReading] = []
    pool: Any = None
    sampler: CardSampler | None = None
    started = False
    teardown_note: str | None = None
    net_hash: str | None = None

    def _finish(verdict: str, refusal: str | None) -> RungResult:
        note = refusal if teardown_note is None else f"{refusal or ''} | {teardown_note}".strip()
        return RungResult(n_workers=n_workers, verdict=verdict, rounds=tuple(readings),
                          refusal=note or None, produced_by=label, net_param_hash=net_hash)

    emit_marker({"phase": "rung_start", "n_workers": n_workers, "counters": counters,
                 "produced_by": label}, out=out)
    try:
        try:
            # The BUILD is inside the guard: a rung can OOM constructing the model or the ring,
            # and that is the same data as a rung that OOMs mid-drive.
            pool = build_sweep_pool(config, n_workers=n_workers, device=device)
            # R317(c)(i): hashed POST-SEED, PRE-PLAY — before `pool.start()` lets any worker
            # touch the net, so a later divergence cannot be blamed on this read.
            net_hash = net_param_hash(pool.model)
            sampler = CardSampler(device_str, plan.sampler_interval_sec) if counters else None
            pool.start()
            started = True
            if sampler is not None:
                sampler.start()
            for index in range(plan.rounds_per_rung):
                warmup = index < plan.warmup_rounds
                probe = make_probe(device_str, round_id=f"w{n_workers}r{index}", out=_Discard())
                if counters:
                    reset_cuda_peak_counters(device_str)
                if sampler is not None:
                    sampler.reset()
                before = runner_stats(pool)
                start = time.monotonic()
                emit_marker({"phase": f"round_start:{index}", "n_workers": n_workers,
                             "warmup": warmup, "produced_by": label,
                             **probe.mark(f"round_start:{index}")}, out=out)
                sleep(plan.round_sec)
                # THE POOL'S OWN FAIL-FAST HOOK, at every round boundary. `WorkerPool._stats_loop`
                # catches the drain loop's every exception, stores it in `_producer_exc` and does
                # NOT raise; `check_producer_health` is the contract, and the trainer calls it
                # each step. This sweep has no trainer, so nothing called it — and the two halves
                # of that omission reinforce each other: with the feeder dead nothing reaches the
                # replay buffer, so the memory series goes FLAT BECAUSE THE RUNG IS BROKEN, while
                # `runner_stats` keeps reporting from the Rust counters, which climb regardless.
                # A rung whose feeder died read as a clean PLATEAU with a plausible rate, and the
                # knee rule compares rates ACROSS rungs, so one dead feeder moves the pick.
                pool.check_producer_health()
                if sampler is not None and (sampler_error := sampler.error()) is not None:
                    raise SweepRefusal(
                        f"the card sampler thread died during round {index}: {sampler_error}. "
                        "The rung's series would change instrument mid-flight, and a figure that "
                        "FELL for that reason reads as memory released."
                    )
                after = runner_stats(pool)
                elapsed = time.monotonic() - start
                end_mark = probe.mark(f"round_end:{index}")
                sampled_peak, samples = sampler.window() if sampler is not None else (None, 0)
                reading = RoundReading(
                    index=index, warmup=warmup, wall_sec=elapsed,
                    games=max(0, after.games_completed - before.games_completed),
                    moves=max(0, after.positions_generated - before.positions_generated),
                    available=bool(end_mark.get("max_memory_allocated_bytes") is not None),
                    sampled_peak_bytes=sampled_peak,
                    allocator_peak_bytes=end_mark.get("max_memory_allocated_bytes"),
                    card_samples=samples,
                )
                readings.append(reading)
                emit_marker({"phase": f"round_end:{index}", "n_workers": n_workers,
                             "produced_by": label, **reading.as_dict(), **end_mark}, out=out)
        finally:
            # TEARDOWN IN ITS OWN GUARD. A raise here REPLACES the return value, so an
            # `InferenceServer.join` failure — likeliest exactly when a rung has just OOM'd —
            # erased the OOM finding, skipped the `rung_end` marker and killed the ladder with
            # its own traceback. `started` is the predicate `mantis.run::
            # _stop_pool_if_start_attempted` measured (an unstarted pool's join raises); the
            # try/except is for every other cause.
            if sampler is not None:
                sampler.stop()
            if started and pool is not None:
                try:
                    pool.stop()
                except Exception as exc:  # noqa: BLE001 — recorded on the rung, never swallowed
                    teardown_note = f"teardown failed after this rung: {exc!r}"
            emit_marker({"phase": "rung_end", "n_workers": n_workers, "produced_by": label,
                         "teardown_note": teardown_note}, out=out)
    except torch.OutOfMemoryError as exc:
        # DATA, not a sitting failure (R309(f)): the rung fails and the EXTENSION stops.
        emit_marker({"phase": "rung_oom", "n_workers": n_workers, "produced_by": label}, out=out)
        return _finish(OOM, f"CUDA out of memory at {n_workers} workers: {exc}")
    except SweepRefusal as exc:
        return _finish(REFUSED, str(exc))
    except KeyboardInterrupt:
        # AN EXPLICIT DECISION, not a default. At ~70 minutes for the base ladder on a rented
        # box, losing every measured rung to a Ctrl-C is the expensive outcome; the rung is named
        # as interrupted and `walk_ladder` re-raises so the sitting stops, with the partial report
        # written by `run_sweep`.
        emit_marker({"phase": "rung_interrupted", "n_workers": n_workers, "produced_by": label},
                    out=out)
        raise
    except Exception as exc:  # noqa: BLE001 — a rung failure with a NAME, never a lost ladder
        emit_marker({"phase": "rung_error", "n_workers": n_workers, "produced_by": label,
                     "error": repr(exc)}, out=out)
        return _finish(RUNG_ERROR, f"{n_workers} workers failed with {exc!r}")

    if (producer_exc := getattr(pool, "_producer_exc", None)) is not None:
        return _finish(PRODUCER_DEAD, f"the self-play buffer feeder died: {producer_exc!r}")
    scored = [r for r in readings if not r.warmup]
    if scored and sum(r.moves for r in scored) == 0:
        return _finish(REFUSED,
                       f"{n_workers} workers generated NO moves across {len(scored)} measured "
                       f"round(s) totalling {sum(r.wall_sec for r in scored):.1f}s — that is a "
                       "sampling limit, not a throughput of zero, and this tool will not report "
                       "it as one")
    verdict, refusal = _verdict_for(tuple(readings), plan)
    return _finish(verdict, refusal)


def walk_ladder(plan: SweepPlan, *, runner: Any, label: str) -> tuple[list[RungResult], str]:
    """Walk the base ladder WHOLE, then extend while gains persist and discipline holds.

    THE BASE BRACKET IS PRE-REGISTERED AND IS WALKED IN FULL. An earlier cut skipped any base
    rung above the box's measured thread count, which on an 8-vCPU instance silently reduced
    R309(g)'s `2, 4, 8, 12, 14` to `2, 4, 8` — half a pre-registered bracket unmeasured, on a
    plausible physical argument nobody had granted. R309(f) attaches the thread bound to the
    EXTENSION (*"extension past 14 permitted while gains persist and memory discipline holds,
    bounded above by the box's measured thread count"*), and that is where it is applied. An
    over-subscribed base rung produces its own verdict, which is data; the bound is reported
    beside the ladder so a reader can see which rungs were over-subscribed.

    THE OOM CLAUSE IS IMPLEMENTED AS THE REGISTER WRITES IT, not as it reads more sensibly.
    R309(f): *"an OOM at a rung is data that fails the rung and stops the ladder's EXTENSION,
    never a sitting failure."* So an OOM fails its own rung, the base rungs above it are STILL
    WALKED, and only the extension is closed off. The widening (stop the whole ladder) is a
    PRE-REGISTRATION CHANGE, which a tool does not get to take; it is filed as an adjudication
    (`WORKER_SWEEP_FINDINGS.md` F-WS-2) and if it comes back granted this is the one function
    that changes.

    EXTENSION STARTS ABOVE THE LAST RUNG RUN, never above the best PASSING one, and the walk
    TERMINATES because that rung's `n_workers` strictly increases. An earlier cut chose the
    highest rung whose verdict was in an enumerated set — and when a later verdict token was
    added and not added to that set, the same extension rung was proposed and re-driven forever:
    a fourteen-minute pool build and teardown per iteration, unbounded, with no report ever
    written. The predicate is now "the last rung run", which needs no enumeration to stay
    correct.

    Returns the results and the STATED reason the walk stopped: a ladder that ends without
    saying why invites the reader to assume it ran out of rungs when it ran out of card — and a
    ladder that states the WRONG why is worse than one that states none.
    """
    bound, bound_source = thread_bound()
    ceiling = min(plan.extension_max, bound)
    results: list[RungResult] = []
    oom_at: int | None = None
    for rung in plan.rungs:
        result = runner(rung)
        results.append(result)
        if result.verdict == OOM and oom_at is None:
            oom_at = rung

    while True:
        highest = results[-1]
        if oom_at is not None:
            return results, (
                f"rung {oom_at} OOM'd; R309(f) stops the ladder's EXTENSION there. The base "
                "rungs above it were still walked, because that is what the clause says")
        if highest.verdict != PLATEAU:
            return results, (f"the highest rung run ({highest.n_workers}) did not PASS "
                             f"({highest.verdict}); no extension")
        passing = [r for r in results if r.verdict == PLATEAU]
        prior = [r for r in passing if r.n_workers != highest.n_workers]
        if prior:
            prior_best = max(r.throughput(plan.metric) for r in prior)
            gain_pct = (((highest.throughput(plan.metric) - prior_best) / prior_best * 100.0)
                        if prior_best > 0 else 0.0)
            if gain_pct <= plan.min_gain_pct:
                return results, (
                    f"rung {highest.n_workers} gained {gain_pct:.2f}% on {plan.metric} over the "
                    f"best prior passing rung, at or below the {plan.min_gain_pct:g}% floor — "
                    "gains no longer persist (R309(f))")
        nxt = highest.n_workers + plan.extension_step
        if nxt > ceiling:
            which = ("[ladder].extension_max" if plan.extension_max <= bound
                     else f"the measured thread bound {bound} ({bound_source})")
            return results, f"the next extension rung {nxt} is above {which}"
        result = runner(nxt)
        results.append(result)
        if result.verdict == OOM:
            oom_at = nxt


# ══ the knee ═════════════════════════════════════════════════════════════════════════════
AGREE = "AGREE"
DIVERGED = "DIVERGED"


def determinism_verdict(first: dict[str, Any], second: dict[str, Any], *,
                        metric: str) -> dict[str, Any]:
    """Two drives of the SAME rung under the SAME seed: did they build the SAME net?

    **RE-SPECIFIED, R317(c). SUPERSEDES R315(c)(i)'s throughput band.** RECAL-SITTING-3 measured
    this control live on a real box, on the SAME rung this docstring used to cite as evidence for
    the band (n_workers=4): 3.9258% spread, DIVERGED against the 1% band, where RESIT-PREP-2's
    engine-side measurement had come back 0.5821%, AGREE. **The defect was the check, not the
    seeding**: `moves_per_min` conflates what a seed controls (the net, the game trajectories)
    with what the machine controls (wall-clock scheduling), and the band was carried from one
    quiet regime to certify a noisier one it was never measured against (R317(b)).

    **THE GATE IS NOW NET-PARAMETER-HASH EQUALITY, NO BAND** (R317(c)(i)): this tests exactly
    what F-RESIT-10's repair claimed — same seed, same net — with nothing about timing in it.
    `_hash_gate` does the comparison; this function's job is to also carry the throughput spread
    as a REPORTED, NON-GATING figure (R317(c)(iii)) so a reader sees both without either one
    controlling the other's answer.

    PURE, over the two rung rows, for `select_knee`'s reason — the sitting reads the arithmetic
    and not an answer, and the same function is driven by its own oracle with rows it constructs.

    **REFUSED IS NEVER A VERDICT** — the rule this tool carries everywhere. A drive that OOM'd,
    errored, lost its producer or ranks at zero has no throughput to compare, and saying
    "they agree" about two numbers that are not measurements is the failure mode the whole
    instrument is built against.

    Raises:
        ValueError: either row is missing `n_workers` or the ranking column, the two rows are
            not the SAME rung (which would make the comparison a ladder step, not a control),
            or a ranking value is non-finite.
    """
    rows = (first, second)
    for row in rows:
        if metric not in row:
            raise ValueError(f"a determinism-control row carries no {metric!r} column")
        value = row[metric]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
            raise ValueError(
                f"a determinism-control row's {metric} is {value!r}. NaN and +/-inf are values "
                "to `json.loads`, and a control that ranks one would report agreement about a "
                "number that is not a measurement"
            )
    if first.get("n_workers") != second.get("n_workers"):
        raise ValueError(
            f"the determinism control compares ONE rung with itself; got n_workers "
            f"{first.get('n_workers')!r} and {second.get('n_workers')!r}. Two different rungs "
            "would make this a ladder step wearing the control's name"
        )
    values = [float(row[metric]) for row in rows]
    undecidable = [row.get("verdict") for row in rows
                   if row.get("verdict") in (REFUSED, OOM, RUNG_ERROR, PRODUCER_DEAD)]
    block = {
        "n_workers": first.get("n_workers"), "metric": metric,
        "first": values[0], "second": values[1],
        "verdicts": [row.get("verdict") for row in rows],
    }
    gate = _hash_gate([("first", first.get("net_param_hash")),
                       ("second", second.get("net_param_hash"))])
    if undecidable or min(values) <= 0:
        return {**block, "spread_pct": None, "net_hash_gate": gate, "verdict": REFUSED,
                "reason": (f"a drive is not a measurement to compare: verdicts {block['verdicts']}"
                           f", {metric} {values}. REFUSED is never a verdict about determinism")}
    spread = abs(values[1] - values[0]) / min(values) * 100.0
    return {**block, "spread_pct": spread, "net_hash_gate": gate, "verdict": gate["verdict"],
            "reason": (f"{gate['reason']}; throughput spread {spread:.4f}% on {metric} "
                       "REPORTED with no band (R317(c)(iii))")}


def select_knee(rows: list[dict[str, Any]], *, knee_pct: float, metric: str,
                noise_floor_rel_std: float) -> dict[str, Any]:
    """R309(f)'s knee rule, as a pure function over the report's own rung rows.

    PURE, so `--select-only` re-derives the pick from a written report through THIS function
    and not a second copy of it. The returned block carries every input the rule ran on, which
    is the difference between a sitting record that carries the arithmetic and one that carries
    the answer.

    **R317(d): the rule is AMENDED, strictly conservative.** `noise_floor_rel_std` is the
    coefficient of variation measured from four fresh same-seed drives at a reference rung
    (`run_noise_floor`) — a MEASURED noise floor, not a guess, and it carries no default: a
    sitting that has not measured one cannot select a pick under this rule. The `within` set
    widens by `3 * noise_floor_rel_std * best`, which can only ADD rungs, never remove one, and
    the pick is still the SMALLEST member of `within` — so the amendment can only move the pick
    toward FEWER workers, never toward more. Pass `0.0` to recover the un-amended rule exactly
    (every existing report re-selected before R317 landed does this, disclosed by the value
    itself rather than by a second code path).

    Raises:
        ValueError: `noise_floor_rel_std` is negative or non-finite, in addition to every
            pre-existing validation this function performs.
    """
    if not isinstance(noise_floor_rel_std, (int, float)) or isinstance(noise_floor_rel_std, bool) \
            or not isfinite(noise_floor_rel_std) or noise_floor_rel_std < 0:
        raise ValueError(
            f"noise_floor_rel_std is {noise_floor_rel_std!r}; R317(d) requires a measured, "
            "non-negative, finite coefficient of variation from run_noise_floor — pass 0.0 "
            "explicitly to select under the un-amended rule, never a default."
        )
    if knee_pct != RULED_KNEE_PCT:
        raise ValueError(
            f"knee_pct is {knee_pct}; R309(f) fixes the knee rule at {RULED_KNEE_PCT:g} percent. "
            "This is checked in BOTH modes, from SOURCE — a report is a file, and a file can be "
            "edited between the drive that wrote it and the reader that quotes it."
        )
    if metric != PREREG_METRIC:
        raise ValueError(
            f"metric is {metric!r}; the ranking metric is pre-registered as {PREREG_METRIC!r} "
            "(DESIGN amendment A1). Both figures are recorded for every rung; only the RANKING "
            "is single-valued."
        )
    passing = []
    for row in rows:
        if row.get("verdict") != PLATEAU:
            continue
        # THE ROWS ARE VALIDATED, because in `--select-only` they are whatever a file says. An
        # earlier cut printed `PICK = 1` — the ONE value R309(f) REJECTS — from a three-key
        # hand-written dict, at rc 0, in the tool's own arithmetic and with its own authority.
        n_workers = row.get("n_workers")
        if not isinstance(n_workers, int) or isinstance(n_workers, bool) or n_workers < 2:
            raise ValueError(
                f"rung row carries n_workers={n_workers!r}. R309(f) REJECTS n_workers = 1 and "
                "this reader will not print a pick it would have refused to measure."
            )
        if metric not in row:
            raise ValueError(f"rung {n_workers} carries no {metric!r} column")
        value = row[metric]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
            raise ValueError(
                f"rung {n_workers}'s {metric} is {value!r}. NaN and +/-inf are values to "
                "`json.loads`: a NaN row used to vanish from the knee set with no refusal, and "
                "an inf row used to capture the pick."
            )
        passing.append({"n_workers": n_workers, "value": float(value)})
    passing.sort(key=lambda p: p["n_workers"])
    notes = sorted({str(r.get("verdict")) for r in rows
                    if r.get("verdict") in (OOM, RUNG_ERROR, PRODUCER_DEAD, REFUSED)})
    if not rows:
        return {"knee_pct": knee_pct, "metric": metric, "passing": [], "best": None,
                "threshold": None, "noise_floor_rel_std": noise_floor_rel_std,
                "noise_floor_adjustment": None, "adjusted_threshold": None,
                "within": [], "picked": None, "notes": notes,
                "reason": "the report carries NO RUNGS at all — this is a statement about the "
                          "document, not about the card's memory"}
    if not passing:
        return {"knee_pct": knee_pct, "metric": metric, "passing": [], "best": None,
                "threshold": None, "noise_floor_rel_std": noise_floor_rel_std,
                "noise_floor_adjustment": None, "adjusted_threshold": None,
                "within": [], "picked": None, "notes": notes,
                "reason": "no rung PASSED (a PLATEAU memory verdict is required, R309(f))"}
    best = max(passing, key=lambda p: p["value"])
    if best["value"] <= 0:
        raise ValueError(
            f"every passing rung ranks at {best['value']} on {metric}: the ranking column cannot "
            "order the ladder, so the knee rule has nothing to apply. Refusing rather than "
            "picking the smallest rung off an identically-zero table."
        )
    threshold = best["value"] * (knee_pct / 100.0)
    # R317(d): the ONLY safe post-hoc direction. Subtracting from the threshold can only ADD
    # rungs to `within`, and the pick is still the SMALLEST member — so this can only pull the
    # pick toward fewer workers, never toward more, whatever `noise_floor_rel_std` turns out to be.
    adjustment = 3.0 * noise_floor_rel_std * best["value"]
    adjusted_threshold = threshold - adjustment
    within = [p for p in passing if p["value"] >= adjusted_threshold]
    picked = min(within, key=lambda p: p["n_workers"])
    return {
        "knee_pct": knee_pct, "metric": metric, "passing": passing, "best": best,
        "threshold": threshold, "noise_floor_rel_std": noise_floor_rel_std,
        "noise_floor_adjustment": adjustment, "adjusted_threshold": adjusted_threshold,
        "within": within, "picked": picked["n_workers"],
        "notes": notes,
        "reason": (f"the smallest passing rung within {knee_pct:g}% of the best passing rung's "
                   f"{metric}, widened by R317(d)'s 3-sigma noise floor "
                   f"({noise_floor_rel_std:.4%} rel. std, -{adjustment:.4f} off the threshold)"),
    }


# ══ provenance, report, render ═══════════════════════════════════════════════════════════
def _sha256(path: Path | str) -> str | None:
    """The config's REAL SHA-256. The first cut used `git hash-object`, which returns git's blob
    hash — SHA-1 over `blob <len>\0<content>` — under a field named `config_sha256`. A later
    reader verifying the config the caps were fitted against runs `sha256sum`, gets a mismatch,
    and concludes the config changed. A label asserting a fact nobody re-derived is the
    derive-or-delete class in a different costume."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(config: Any, config_path: Path, *, device: str, label: str) -> dict[str, Any]:
    """What produced every figure in this report (R287(a)).

    NO HOST IDENTIFIERS — R112 and CI gate 17. GPU model and CPU count are regime facts and are
    carried; a hostname, a home path or a provider name is not and is never read here. The LIVE
    allocator conf is read through the ONE authority and its SOURCE VARIABLE is stamped beside
    it: RECAL-PREP found `fusion_calibrate` stamping `""` for a drive that was in the other
    regime because it read one variable of the two c10 reads.
    """
    live = read_live_allocator_conf()
    bound, bound_source = thread_bound()
    cuda = torch.cuda.is_available()
    commit = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain")
    return {
        "tool": TOOL,
        "produced_by": label,
        # CAPTURE-TIME REDACTION (R301(d)), and it has to be here rather than in a later scan:
        # this report lands in the governance workspace and in a sitting record, and CI gate 17
        # scans neither. The operator's invocation on the box is an absolute path under a home or
        # a provisioning directory; the BASENAME plus the digest carries every fact the report
        # actually uses, and a later scan only ever catches what was already written down.
        "config_name": Path(config_path).name,
        "config_sha256": _sha256(config_path),
        "git_commit": commit,
        # `None`, not `False`, when git could not answer. `bool(None)` is a POSITIVE CLAIM OF
        # CLEANLINESS about a tree nobody looked at — and the tool will be launched from a
        # scratch directory or a tarball with no `.git` on exactly the host that matters.
        "git_dirty": None if commit is None else bool(porcelain),
        "run_id": getattr(config, "run_id", None),
        # THE SEED EVERY RUNG'S NETWORK WAS BUILT FROM (F-RESIT-10). Carried because a ladder
        # whose rungs are comparable is a CLAIM about how they were built, and R69 says a
        # measurement travels with its mechanism: a reader of this report can now see that the
        # ranking column is a function of `n_workers` alone, rather than having to trust it.
        # Before the repair the honest value of this field would have been "unseeded".
        "seed": int(config.seed),
        "encoding": config.identity.encoding,
        "representation": config.identity.representation,
        "device": device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda,
        "cuda_counters_available": cuda_counters_available(device),
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        # The capacity every peak in this report is implicitly measured against. Without it a
        # reader cannot size a figure, and the downstream partition is an inequality against
        # exactly this number.
        "card_total_bytes": (cuda_device_total_bytes(device)
                             if cuda_counters_available(device) else None),
        "thread_bound": bound,
        "thread_bound_source": bound_source,
        "declared_allocator_posture": declared_allocator_posture(config.model_dump()),
        "allocator_posture_governs_device": governs_device(device),
        "live_allocator_conf": live.raw,
        "live_allocator_conf_source_var": live.source_var,
    }


def build_report(*, plan: SweepPlan, prov: dict[str, Any], results: list[RungResult],
                 stopped: str, noise_floor_rel_std: float) -> dict[str, Any]:
    rows = [r.as_dict(plan.metric, plateau_rounds=plan.plateau_rounds,
                      band_pct=plan.band_pct) for r in results]
    # R317(c)(i): the ladder-wide gate. Every rung shares one seed and one config, so every net
    # any rung built must hash equal — a divergence here means the ranking column this ladder
    # exists to produce is not comparable, and no knee arithmetic on it can be trusted.
    gate = _hash_gate([(str(r.n_workers), r.net_param_hash) for r in results])
    selection = select_knee(rows, knee_pct=plan.knee_pct, metric=plan.metric,
                            noise_floor_rel_std=noise_floor_rel_std)
    if gate["verdict"] == DIVERGED:
        selection = {**selection, "picked": None,
                     "reason": f"VOIDED by R317(c)(i): {gate['reason']} — the ranking column is "
                               "not comparable across rungs built from different nets"}
    return {
        "tool": TOOL,
        "prereg": dict(plan.provenance),
        "plan": {
            "rungs": list(plan.rungs), "extension_step": plan.extension_step,
            "extension_max": plan.extension_max, "min_gain_pct": plan.min_gain_pct,
            "warmup_rounds": plan.warmup_rounds, "measured_rounds": plan.measured_rounds,
            "round_sec": plan.round_sec, "sampler_interval_sec": plan.sampler_interval_sec,
            "plateau_rounds": plan.plateau_rounds, "band_pct": plan.band_pct,
            "knee_pct": plan.knee_pct, "metric": plan.metric,
        },
        "provenance": prov,
        "ladder_stopped_because": stopped,
        "rungs": rows,
        "net_hash_gate": gate,
        "selection": selection,
    }


def rc_for(report: dict[str, Any]) -> int:
    """0 a pick · 1 measurable but nothing PASSED · 2 nothing was measurable at all.

    R317(c)(i): a DIVERGED net_hash_gate is neither of the first two — it means the ranking
    column itself is not comparable, which is a statement about the INSTRUMENT, not about the
    card's memory or the ladder's throughput. It REFUSES (rc 2) rather than reporting "no pick"
    the way an all-GROWING ladder does, because the latter is data and the former is not.
    """
    if report.get("net_hash_gate", {}).get("verdict") == DIVERGED:
        return RC_REFUSED
    if report["selection"]["picked"] is not None:
        return 0
    decisive = [r for r in report["rungs"] if r["verdict"] in (PLATEAU, GROWING, OOM)]
    return 1 if decisive else RC_REFUSED


def _gib(value: Any) -> str:
    return "unmeasured" if value is None else f"{value / GIB:.4f} GiB"


def render(report: dict[str, Any], out: Any) -> None:
    """The human screen. Every figure beside its sampling limit and its producing run."""
    prov = report["provenance"]
    plan = report["plan"]
    print(f"{TOOL} — produced_by={prov['produced_by']}", file=out)
    if not prov["cuda_counters_available"]:
        print(
            "MECHANISM EVIDENCE ONLY — this host has no CUDA counters. No figure below is a\n"
            "floor, a bound, or a comparison point for any other host. Cross-host numbers are\n"
            "mechanism evidence and nothing else.", file=out,
        )
    print(f"prereg: {report['prereg'].get('prereg_ruling')} "
          f"(recorded {report['prereg'].get('prereg_recorded')})", file=out)
    print(f"config: {prov['config_name']} sha256={prov['config_sha256']} "
          f"commit={prov['git_commit']} dirty={prov['git_dirty']}", file=out)
    print(f"regime: device={prov['device']} torch={prov['torch_version']} "
          f"gpu={prov['gpu_name']} card_total={_gib(prov.get('card_total_bytes'))} "
          f"posture={prov['declared_allocator_posture']!r} "
          f"live_alloc_conf={prov['live_allocator_conf']!r} "
          f"(from {prov['live_allocator_conf_source_var']})", file=out)
    oversubscribed = [row["n_workers"] for row in report["rungs"]
                      if row["n_workers"] > prov["thread_bound"]]
    print(f"bound:  {prov['thread_bound']} threads via {prov['thread_bound_source']}"
          + (f"  — rungs {oversubscribed} are OVER-SUBSCRIBED and were walked anyway: R309(f) "
             "bounds the EXTENSION by the thread count, and the base bracket is pre-registered"
             if oversubscribed else ""), file=out)
    print(f"rule:   rounds={plan['warmup_rounds']}w+{plan['measured_rounds']}m of "
          f"{plan['round_sec']:g}s · plateau_rounds={plan['plateau_rounds']} "
          f"band_pct={plan['band_pct']:g} · rank on {plan['metric']} · "
          f"knee_pct={plan['knee_pct']:g}", file=out)
    print("", file=out)
    for row in report["rungs"]:
        print(f"  n_workers={row['n_workers']:>3}  {row['verdict']:<18} "
              f"moves/min={row['moves_per_min']:>10.3f}  games/min={row['games_per_min']:>8.3f}  "
              f"peak={_gib(row['rung_peak_bytes'])}", file=out)
        print(f"      sample: rounds_measured={row['rounds_measured']} "
              f"rounds_unmeasured={row['rounds_unmeasured']} wall_sec={row['wall_sec']:.1f} "
              f"— that count and that wall time are the limit, not a bound "
              f"(produced_by={row['produced_by']})", file=out)
        spread = row[f"{row['ranking_metric']}_spread"]
        if spread["n_rounds"]:
            print(f"      spread ({row['ranking_metric']}, n={spread['n_rounds']}): "
                  f"min={spread['min']} median={spread['median']} max={spread['max']} "
                  f"— the knee band is a percentage; read it against this, not against the "
                  f"point estimate alone", file=out)
        if row["governing_sink_counts"]:
            print(f"      sinks: governed by {row['governing_sink_counts']} · "
                  f"max disagreement {_gib(row['max_sink_disagreement_bytes'])} — where the two "
                  f"instruments disagree the larger governs AND the disagreement is a finding",
                  file=out)
        if row["sink_verdicts"]:
            print(f"      verdicts per sink: {row['sink_verdicts']} — the stopping rule is asked "
                  f"of EACH series, not of the pair `max()` already resolved; growth on either "
                  f"instrument fails the rung", file=out)
        rises = {sink: f"{value:+.2f}%" for sink, value in row["trailing_rise_pct"].items()}
        risen = [s for s, v in row["trailing_rise_pct"].items() if v > plan["band_pct"]]
        if rises:
            print(f"      trailing-window rise: {rises}"
                  + (f"  ⚠ {risen} rose more than the band across the window while the verdict "
                     "was taken against the running maximum BEFORE it — disclosed, non-gating: "
                     "this is the shape the stopping rule cannot see" if risen else ""),
                  file=out)
        for reading in row["rounds"]:
            flag = "" if reading["available"] else "  [unmeasured]"
            tag = "warmup" if reading["warmup"] else "      "
            print(f"        r{reading['index']} {tag} moves={reading['moves']:>7} "
                  f"games={reading['games']:>4} "
                  f"card={_gib(reading['sampled_peak_bytes'])} "
                  f"alloc={_gib(reading['allocator_peak_bytes'])} "
                  f"governing={_gib(reading['governing_peak_bytes'])} "
                  f"card_samples={reading['card_samples']}{flag}", file=out)
        if row["refusal"]:
            print(f"      REFUSAL/NOTE: {row['refusal']}", file=out)
    print("", file=out)
    print(f"ladder stopped because: {report['ladder_stopped_because']}", file=out)
    render_selection(report["selection"], out)


def render_determinism_control(control: dict[str, Any], out: Any) -> None:
    """The control's own screen: the net-hash gate (R317(c)(i)), the reported spread with no
    band (R317(c)(iii)), and the verdict."""
    spread = control["spread_pct"]
    shown = "unmeasurable" if spread is None else f"{spread:.4f}%"
    gate = control.get("net_hash_gate", {})
    print("", file=out)
    print(f"DETERMINISM CONTROL — rung {control['n_workers']} driven twice in one process, "
          "same seed", file=out)
    print(f"  drive 1        {control['first']:12.4f} {control['metric']}", file=out)
    print(f"  drive 2        {control['second']:12.4f} {control['metric']}", file=out)
    print(f"  net hashes     {gate.get('hashes', {})}", file=out)
    print(f"  spread         {shown:>12}   REPORTED, no band (R317(c)(iii))", file=out)
    print(f"  rung verdicts  {control['verdicts']}", file=out)
    print(f"  VERDICT        {control['verdict']:>12}   {control['reason']}", file=out)


def render_selection(selection: dict[str, Any], out: Any) -> None:
    """The knee arithmetic WITH its inputs — the derivation, not just the answer."""
    print(f"selection: knee_pct={selection['knee_pct']:g} on {selection['metric']}", file=out)
    if not selection["passing"]:
        print(f"  PICK = none — {selection['reason']}", file=out)
        # THE NOTES PRINT HERE TOO, and this is the run where the reader most needs them: the
        # line was added so "a reader who quotes the arithmetic alone would not otherwise see
        # that the ladder had a failing rung", and it sat BELOW an early return, i.e. unreachable
        # in exactly the case it was written for.
        if selection.get("notes"):
            print(f"  what the ladder DID return: {', '.join(selection['notes'])}", file=out)
        return
    print("  passing rungs (PLATEAU only): "
          + ", ".join(f"{p['n_workers']}@{p['value']:.3f}" for p in selection["passing"]),
          file=out)
    print(f"  best passing:                 {selection['best']['n_workers']} "
          f"@ {selection['best']['value']:.3f}", file=out)
    print(f"  threshold = {selection['best']['value']:.3f} * "
          f"{selection['knee_pct'] / 100.0:g} = {selection['threshold']:.3f}", file=out)
    print(f"  R317(d) noise floor: {selection['noise_floor_rel_std']:.4%} rel. std -> "
          f"-{selection['noise_floor_adjustment']:.3f}, adjusted threshold = "
          f"{selection['adjusted_threshold']:.3f}", file=out)
    print("  at or above adjusted threshold:"
          + ", ".join(f"{p['n_workers']}@{p['value']:.3f}" for p in selection["within"]),
          file=out)
    print(f"  PICK = {selection['picked']}   ({selection['reason']})", file=out)
    if selection.get("notes"):
        print(f"  NOT in the passing set: {', '.join(selection['notes'])} — a reader who quotes "
              "the arithmetic alone would not otherwise see that the ladder had a failing rung",
              file=out)


# ══ entry ════════════════════════════════════════════════════════════════════════════════
def run_determinism_control(*, config_path: Path, plan_path: Path, n_workers: int,
                            out: Any) -> dict[str, Any]:
    """Drive ONE rung TWICE in one process and report whether the two built the SAME net.

    **The control R315(c)(i) orders and R317(c) RE-SPECIFIES, and it is the instrument that
    makes the ladder's ranking column testable rather than trusted.** `build_sweep_net` seeds
    from the config's own `seed` before every pool build, so two drives of the same rung must
    build the SAME network; if they do, their net-parameter hashes are EQUAL — no band. If
    seeding is removed or perturbed, they are not — which is what
    `tests/diagnostics/test_worker_sweep_determinism.py` demonstrates rather than asserts.

    It lives HERE, in the shipped tool, and not in a sitting's script: an instrument a sitting
    authors on the box is an instrument nobody reviewed, and the 2026-08-27 re-sit had to write
    three of them.

    ONE PROCESS, deliberately — the same process the ladder walks in, so the control measures the
    thing the ladder does. The 2026-08-27 discriminator established the residual is position-
    independent (0.58 % apart, first drive against second, network held fixed), so a difference
    here is about the seeding and not about where the drive sat.

    Raises:
        AllocatorPostureMismatchError: the live allocator conf does not match the minted posture.
        SweepRefusal: the plan file is unreadable, incomplete, or moves a ruled constant.
    """
    plan = load_plan(plan_path)
    config = load_config(config_path)
    device = torch.device(config.train.device)
    assert_allocator_posture(config.model_dump(), device_type=device.type)
    label = f"{getattr(config, 'run_id', 'run')}@{_git('rev-parse', '--short', 'HEAD') or 'no-git'}"

    rows: list[dict[str, Any]] = []
    for attempt in (1, 2):
        result = drive_rung(config, plan, n_workers=n_workers, device=device,
                            label=f"{label}#determinism-{attempt}", out=out)
        row = result.as_dict(plan.metric, plateau_rounds=plan.plateau_rounds,
                             band_pct=plan.band_pct)
        row["drive"] = attempt
        rows.append(row)
    control = determinism_verdict(rows[0], rows[1], metric=plan.metric)
    prov = provenance(config, Path(config_path), device=str(device), label=label)
    return {"tool": TOOL, "mode": "determinism_control", "prereg": dict(plan.provenance),
            "provenance": prov, "drives": rows, "control": control}


def run_noise_floor(*, config_path: Path, plan_path: Path, n_workers: int,
                    out: Any) -> dict[str, Any]:
    """R317(d): FOUR fresh same-seed drives at one reference rung — the measured noise floor
    the amended knee rule requires. Walks no ladder and picks nothing, same discipline as
    `run_determinism_control`.

    The coefficient of variation (population std / mean) of the four drive means is the
    RELATIVE noise this sitting carries forward under the stated assumption that it holds
    across every rung of the ladder (R317(d)) — a measured number, not a guess, and one the
    knee rule (`select_knee`) can no longer be asked to select without.

    Raises:
        AllocatorPostureMismatchError: the live allocator conf does not match the minted posture.
        SweepRefusal: the plan file is unreadable, incomplete, or moves a ruled constant.
    """
    from statistics import fmean, pstdev

    plan = load_plan(plan_path)
    config = load_config(config_path)
    device = torch.device(config.train.device)
    assert_allocator_posture(config.model_dump(), device_type=device.type)
    label = f"{getattr(config, 'run_id', 'run')}@{_git('rev-parse', '--short', 'HEAD') or 'no-git'}"

    rows: list[dict[str, Any]] = []
    for attempt in range(1, 5):
        result = drive_rung(config, plan, n_workers=n_workers, device=device,
                            label=f"{label}#noisefloor-{attempt}", out=out)
        row = result.as_dict(plan.metric, plateau_rounds=plan.plateau_rounds,
                             band_pct=plan.band_pct)
        row["drive"] = attempt
        rows.append(row)
    gate = _hash_gate([(f"drive{r['drive']}", r.get("net_param_hash")) for r in rows])
    means = [row[plan.metric] for row in rows]
    undecidable = [row.get("verdict") for row in rows
                   if row.get("verdict") in (REFUSED, OOM, RUNG_ERROR, PRODUCER_DEAD)]
    if undecidable or min(means) <= 0:
        noise_floor = {
            "n_workers": n_workers, "metric": plan.metric, "means": means,
            "mean_of_means": None, "std": None, "rel_std": None, "n_sigma": 3,
            "reason": f"a drive is not a measurement to average: verdicts "
                      f"{[row.get('verdict') for row in rows]}, {plan.metric} {means}",
        }
    else:
        mean_of_means = fmean(means)
        std = pstdev(means)
        noise_floor = {
            "n_workers": n_workers, "metric": plan.metric, "means": means,
            "mean_of_means": mean_of_means, "std": std,
            "rel_std": std / mean_of_means if mean_of_means else None, "n_sigma": 3,
            "reason": f"rung-{n_workers} relative noise (coefficient of variation over "
                      f"{len(means)} fresh same-seed drives), assumed to carry across every "
                      "rung of the ladder (R317(d))",
        }
    prov = provenance(config, Path(config_path), device=str(device), label=label)
    return {"tool": TOOL, "mode": "noise_floor", "prereg": dict(plan.provenance),
            "provenance": prov, "drives": rows, "net_hash_gate": gate,
            "noise_floor": noise_floor}


def render_noise_floor(payload: dict[str, Any], out: Any) -> None:
    """The noise-floor screen: the four drives, the hash gate, and the derived rel_std."""
    nf = payload["noise_floor"]
    gate = payload.get("net_hash_gate", {})
    print("", file=out)
    print(f"NOISE FLOOR — rung {nf['n_workers']} driven {len(nf['means'])} times, same seed "
          "(R317(d))", file=out)
    for i, value in enumerate(nf["means"], start=1):
        print(f"  drive {i}         {value:12.4f} {nf['metric']}", file=out)
    print(f"  net hashes     {gate.get('hashes', {})}  gate={gate.get('verdict')}", file=out)
    if nf["rel_std"] is None:
        print(f"  REFUSED — {nf['reason']}", file=out)
    else:
        print(f"  mean_of_means  {nf['mean_of_means']:12.4f}", file=out)
        print(f"  std            {nf['std']:12.4f}", file=out)
        print(f"  rel_std        {nf['rel_std']:12.4%}   ({nf['reason']})", file=out)


def run_sweep(*, config_path: Path, plan_path: Path, out: Any,
             noise_floor_rel_std: float) -> dict[str, Any]:
    """Load, assert the regime, walk the ladder, build the report. Writes no config, ever."""
    plan = load_plan(plan_path)
    config = load_config(config_path)
    device = torch.device(config.train.device)
    # The SAME authority `mantis.run` calls, imported and not copied: R308(g)(i) says a CUDA
    # process boots on a MINTED regime or not at all, and this sweep is the re-sit's first CUDA
    # process. On a non-CUDA device the resolution is skipped by the resolver's own route
    # scoping, so a cpu drive is untouched.
    assert_allocator_posture(config.model_dump(), device_type=device.type)
    label = f"{getattr(config, 'run_id', 'run')}@{_git('rev-parse', '--short', 'HEAD') or 'no-git'}"

    def runner(n_workers: int) -> RungResult:
        return drive_rung(config, plan, n_workers=n_workers, device=device, label=label, out=out)

    results, stopped = walk_ladder(plan, runner=runner, label=label)
    prov = provenance(config, Path(config_path), device=str(device), label=label)
    return build_report(plan=plan, prov=prov, results=results, stopped=stopped,
                        noise_floor_rel_std=noise_floor_rel_std)


def read_report(path: str | Path) -> dict[str, Any]:
    """Load a report this tool wrote, and REFUSE anything else.

    IDENTITY, NOT ONLY READABILITY. `parse_sweep_markers` invokes the `peaks.py` lesson — *a
    reader that guesses at a shape it does not recognise produces a number nobody can distinguish
    from a measurement* — and the first cut of this reader then required four keys and asked
    nothing about whether the document was its own output. A three-key hand-written dict printed
    `PICK = 1`, the one value R309(f) REJECTS, at rc 0.
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path}: not a report object ({type(report).__name__})")
    if report.get("tool") != TOOL:
        raise ValueError(
            f"{path}: `tool` is {report.get('tool')!r}, not {TOOL!r}. This reader answers about "
            "its own output and refuses to answer about a document it cannot vouch for."
        )
    for key in ("rungs", "plan", "provenance"):
        if not isinstance(report.get(key), (dict, list)):
            raise ValueError(f"{path}: the report carries no `{key}` block")
    stated = report["plan"]
    if stated.get("knee_pct") != RULED_KNEE_PCT or stated.get("metric") != PREREG_METRIC:
        raise ValueError(
            f"{path}: the report states knee_pct={stated.get('knee_pct')!r} "
            f"metric={stated.get('metric')!r}; the pre-registered rule is "
            f"{RULED_KNEE_PCT:g} / {PREREG_METRIC!r}. A report whose stated rule is not the "
            "ruling's was produced under a rule this tool cannot apply, or has been edited."
        )
    return report


def read_noise_floor_report(path: str | Path) -> float:
    """R317(d): load a `--noise-floor` report and REFUSE anything else, returning `rel_std`.

    A separate reader from `read_report`, deliberately: a noise-floor document has no `rungs` or
    `plan` block — it is not a ladder report and must not be read as one just because both come
    from this tool.
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("tool") != TOOL or \
            report.get("mode") != "noise_floor":
        raise ValueError(f"{path}: not a --noise-floor report from {TOOL}")
    rel_std = report.get("noise_floor", {}).get("rel_std")
    if rel_std is None:
        raise ValueError(f"{path}: carries no measured rel_std (its own drive was REFUSED) — "
                         "nothing to select under")
    return float(rel_std)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {TOOL}", description=__doc__)
    parser.add_argument("--config", help="a MINTED RunConfig — the run's own, otherwise unchanged")
    parser.add_argument("--plan", help="the pre-registered sweep plan (tools/worker_sweep_plan.toml)")
    parser.add_argument("--out", help="report destination; stdout when absent")
    parser.add_argument("--select-only",
                        help="re-derive the knee from a report this tool already wrote")
    parser.add_argument("--determinism-control", type=int, metavar="N",
                        help="drive rung N TWICE in one process and report whether the two built "
                             "the SAME net, by hash equality (R317(c)(i)); walks no ladder and "
                             "picks nothing")
    parser.add_argument("--noise-floor", type=int, metavar="N",
                        help="drive rung N FOUR times in one process and report the measured "
                             "relative noise (R317(d)); walks no ladder and picks nothing")
    parser.add_argument("--noise-floor-report",
                        help="a report this tool already wrote with --noise-floor, supplying "
                             "the measured noise floor the amended knee rule requires "
                             "(R317(d)); required by drive mode and --select-only alike")
    args = parser.parse_args(argv)

    if args.determinism_control is not None and args.noise_floor is not None:
        print("REFUSED: --determinism-control and --noise-floor are different drive modes; "
              "naming both describes neither", file=sys.stderr)
        return RC_REFUSED

    if args.determinism_control is not None:
        # Same discipline --select-only carries: an input this mode does not read is refused BY
        # NAME rather than ignored, because naming an input a mode does not read describes a run
        # that did not happen.
        if args.select_only:
            print("REFUSED: --determinism-control drives a rung and --select-only reads a written "
                  "report; they are different modes and naming both describes neither",
                  file=sys.stderr)
            return RC_REFUSED
        if not args.config or not args.plan:
            print("REFUSED: --determinism-control needs both --config and --plan, for the reason "
                  "the ladder does: a config this tool picked would measure a program nobody "
                  "asked about, and a plan it picked would be a pre-registration nobody wrote",
                  file=sys.stderr)
            return RC_REFUSED
        if args.determinism_control < 2:
            print(f"REFUSED: --determinism-control {args.determinism_control}; R309(f) REJECTS "
                  "n_workers = 1 and this tool will not drive a rung it would refuse to pick",
                  file=sys.stderr)
            return RC_REFUSED
        try:
            report = run_determinism_control(config_path=Path(args.config),
                                             plan_path=Path(args.plan),
                                             n_workers=args.determinism_control, out=sys.stderr)
        except KeyboardInterrupt:
            print("REFUSED: interrupted; no control was completed", file=sys.stderr)
            return RC_REFUSED
        except Exception as exc:  # noqa: BLE001 — ONE refusal path, and rc 1 is not it
            print(f"REFUSED: {exc!r}", file=sys.stderr)
            return RC_REFUSED
        if args.out is not None:
            Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True),
                                      encoding="utf-8")
            print(f"{TOOL}: determinism control written to {args.out}", file=sys.stderr)
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        render_determinism_control(report["control"], sys.stderr)
        verdict = report["control"]["verdict"]
        return 0 if verdict == AGREE else (1 if verdict == DIVERGED else RC_REFUSED)

    if args.noise_floor is not None:
        if args.select_only or args.noise_floor_report:
            print("REFUSED: --noise-floor drives a rung and the other two read a written "
                  "report; naming more than one mode describes none of them",
                  file=sys.stderr)
            return RC_REFUSED
        if not args.config or not args.plan:
            print("REFUSED: --noise-floor needs both --config and --plan, for the reason "
                  "every drive mode does: a config this tool picked would measure a program "
                  "nobody asked about, and a plan it picked would be a pre-registration nobody "
                  "wrote", file=sys.stderr)
            return RC_REFUSED
        if args.noise_floor < 2:
            print(f"REFUSED: --noise-floor {args.noise_floor}; R309(f) REJECTS n_workers = 1 "
                  "and this tool will not drive a rung it would refuse to pick", file=sys.stderr)
            return RC_REFUSED
        try:
            report = run_noise_floor(config_path=Path(args.config), plan_path=Path(args.plan),
                                     n_workers=args.noise_floor, out=sys.stderr)
        except KeyboardInterrupt:
            print("REFUSED: interrupted; no noise floor was completed", file=sys.stderr)
            return RC_REFUSED
        except Exception as exc:  # noqa: BLE001 — ONE refusal path, and rc 1 is not it
            print(f"REFUSED: {exc!r}", file=sys.stderr)
            return RC_REFUSED
        if args.out is not None:
            Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True),
                                      encoding="utf-8")
            print(f"{TOOL}: noise floor written to {args.out}", file=sys.stderr)
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        render_noise_floor(report, sys.stderr)
        if report["net_hash_gate"]["verdict"] == DIVERGED:
            return RC_REFUSED
        return 0 if report["noise_floor"]["rel_std"] is not None else RC_REFUSED

    if args.select_only:
        # EVERY input the mode does not read is refused BY NAME. `--out` was silently ignored
        # here while `--config`/`--plan` were refused — one of the two behaviours is wrong, and
        # the docstring already says which: naming an input a mode does not read describes a run
        # that did not happen.
        unread = [name for name, value in (("--config", args.config), ("--plan", args.plan),
                                           ("--out", args.out)) if value]
        if unread:
            print(f"REFUSED: --select-only re-derives a pick from a written report and runs no "
                  f"drive; {', '.join(unread)} name inputs it does not read", file=sys.stderr)
            return RC_REFUSED
        if not args.noise_floor_report:
            print("REFUSED: --select-only needs --noise-floor-report; R317(d) requires a "
                  "measured noise floor before a pick can be selected, and this reader will not "
                  "re-derive one under an implicit 0.0", file=sys.stderr)
            return RC_REFUSED
        try:
            report = read_report(args.select_only)
            noise_floor_rel_std = read_noise_floor_report(args.noise_floor_report)
            selection = select_knee(report["rungs"], knee_pct=RULED_KNEE_PCT,
                                    metric=PREREG_METRIC,
                                    noise_floor_rel_std=noise_floor_rel_std)
            produced_by = report["provenance"].get("produced_by", "<unstated>")
        except (OSError, AttributeError, KeyError, TypeError, ValueError) as exc:
            print(f"REFUSED: cannot re-derive a pick from {args.select_only}: {exc}",
                  file=sys.stderr)
            return RC_REFUSED
        print(f"{TOOL} --select-only {args.select_only}")
        print(f"produced_by={produced_by}")
        render_selection(selection, sys.stdout)
        report["selection"] = selection
        return rc_for(report)

    if not args.config or not args.plan:
        print("REFUSED: both --config and --plan are required. Neither is defaulted: a config "
              "this tool picked would measure a program nobody asked about, and a plan it "
              "picked would be a pre-registration nobody wrote.", file=sys.stderr)
        return RC_REFUSED
    if not args.noise_floor_report:
        print("REFUSED: the ladder needs --noise-floor-report; R317(d) requires a measured "
              "noise floor before a pick can be selected, and this tool will not walk a "
              "seventy-minute ladder toward a selection it cannot make at the end",
              file=sys.stderr)
        return RC_REFUSED
    try:
        noise_floor_rel_std = read_noise_floor_report(args.noise_floor_report)
    except (OSError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return RC_REFUSED
    if args.out is not None:
        # PROBED BEFORE THE FIRST RUNG, because the destination used to be validated at the moment
        # it is least recoverable: the write sat outside the guard and BEFORE the render, so an
        # unwritable path after a seventy-minute ladder produced a traceback, rc 1, and no screen.
        try:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text("", encoding="utf-8")
        except OSError as exc:
            print(f"REFUSED: --out {args.out} is not writable ({exc}); refusing BEFORE the "
                  "ladder rather than after it", file=sys.stderr)
            return RC_REFUSED
    try:
        report = run_sweep(config_path=Path(args.config), plan_path=Path(args.plan),
                           out=sys.stderr, noise_floor_rel_std=noise_floor_rel_std)
    except KeyboardInterrupt:
        print("REFUSED: interrupted; no complete ladder was measured", file=sys.stderr)
        return RC_REFUSED
    except Exception as exc:  # noqa: BLE001 — ONE refusal path, and rc 1 is not it
        # An escaping exception exits the interpreter with rc 1, which this tool's contract
        # reserves for "no rung PASSED — every measurable rung was GROWING or OOM". A malformed
        # input or an unmodelled failure must never be able to present as a measured memory
        # result, because the block's Phase W posture branches on exactly that distinction.
        print(f"REFUSED: {exc!r}", file=sys.stderr)
        return RC_REFUSED

    payload = json.dumps(report, indent=2, sort_keys=True)
    render(report, sys.stderr)          # the screen FIRST: the numbers survive a failed write
    if args.out:
        try:
            Path(args.out).write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"REFUSED: could not write {args.out} ({exc}); the report follows on stdout so "
                  "the drive is not lost", file=sys.stderr)
            print(payload)
            return RC_REFUSED
    else:
        print(payload)
    return rc_for(report)


__all__ = [
    "GROWING",
    "MARKER",
    "METRICS",
    "OOM",
    "PLATEAU",
    "REFUSED",
    "RC_REFUSED",
    "PRODUCER_DEAD",
    "RUNG_ERROR",
    "CardSampler",
    "RoundReading",
    "RungResult",
    "SweepPlan",
    "SweepRefusal",
    "build_report",
    "build_sweep_pool",
    "drive_rung",
    "emit_marker",
    "load_plan",
    "main",
    "parse_sweep_markers",
    "provenance",
    "AGREE",
    "DIVERGED",
    "build_sweep_net",
    "determinism_verdict",
    "render_determinism_control",
    "run_determinism_control",
    "rc_for",
    "read_report",
    "read_noise_floor_report",
    "render",
    "render_noise_floor",
    "render_selection",
    "run_noise_floor",
    "run_sweep",
    "select_knee",
    "thread_bound",
    "walk_ladder",
]

if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())

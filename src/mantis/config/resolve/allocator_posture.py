# >300 justify (R8). NO LINE COUNT is stated (G-DFIX-4 / R192(e), derive-or-delete). This
# module is ONE claim with three faces that cannot be separated without creating a second
# authority over the same fact: what the config MINTED (the token), what the process is
# ACTUALLY RUNNING UNDER (the environment pair, read in c10's own precedence order), and the
# comparison between them. Splitting the reader out would put "which variable does torch
# read" in one module and "which posture was minted" in another, and the whole defect class
# this exists to close is exactly those two answers drifting apart. The prose below is the
# per-decision rationale R8's clause protects; executable content is a minority of the file.
"""`allocator_posture` — THE one authority over the CUDA caching allocator's REGIME
(R308(g)(i); R1, LAW-08, LAW-11, R119).

WHY THIS KNOB EXISTS, measured rather than argued. The 2026-08-22 re-calibration sitting
measured, on one host at one sha with one config, a card high-water of **14.98 GiB under the
DEFAULT allocator posture against 11.36 GiB under `expandable_segments:True`** — a 3.62 GiB
difference in the quantity the whole memory partition is denominated in. It nevertheless kept
DEFAULT, and the reason it gave was not a measurement: *a cap fitted under
`expandable_segments:True` is valid only while that variable is set, and no config mints it, no
gate checks it, and `armed_aborts.py` has no row for it* — a minted value depending on an
unminted, unenforced launch fact, which is the silent-authority class R1 exists to kill, with
the worst failure mode available: launch without the variable and you get caps fitted for a
posture you are not in AND the fragile regime together.

**This module is the removal of that objection, and nothing else.** It does not pick a value.
Every committed config and both mint templates carry the R119 `null` placeholder, because
R308(g)(i) reserves the posture VALUE for the re-calibration sitting, measurement-derived under
R282(b). What lands here is the machinery that makes a value mintable, threadable to every
process that allocates, and *unrunnable under a posture it was not fitted for*.

A CLOSED TOKEN SET, NOT THE RAW ENVIRONMENT STRING. The config mints `default` or
`expandable_segments`; this module owns the one mapping token -> required allocator conf. A
free-text env string in a minted config would be a second spelling authority for the regime the
caps are fitted under: `expandable_segments:True`, `expandable_segments:true` and
`expandable_segments:True,max_split_size_mb:128` would be three configs claiming one regime, and
the third is a regime nobody fitted. The closed set makes "a posture with no fit"
unrepresentable, which is the property R79's no-off-sentinel rule buys elsewhere in this schema.
Adding a member is a mint-level act with its own measurement.

`null` IS NOT AN OFF STATE. It is R119's placeholder — schema-VALID, so gate 7 stays green and
the repo ships complete configs, and runtime-REFUSED, so a CUDA process on an unminted posture
cannot boot. `UncalibratedAllocatorPostureError` is a SUBCLASS of the general absence for
`resolve_fused_graph_caps`'s recorded reason: "you never minted this" carries a remedy that
"your config is malformed" does not, and a caller handling the general absence must not miss the
placeholder.

DEVICE-SCOPED, AND THE SCOPING IS PRINCIPLED RATHER THAN CONDITIONAL AUTHORITY. The posture
governs the CUDA caching allocator. A process with no CUDA device has no CUDA caching allocator,
so it has no posture to enforce and no term of a partition fitted under one — the same shape as
`resolve_fused_graph_caps` being reached from the GRAPH branch and never the grid one. Each
process asserts for ITS OWN device: the run process for `config.train.device`, the eval child
for `RoundSpec.worker_device`. The eval child is a SECOND allocator on the same card that no
in-process bound can see, which is why it gets its own assertion rather than inheriting the
parent's verdict.

THE ASSERTION RAISES; IT DOES NOT SET. `PYTORCH_CUDA_ALLOC_CONF` is consumed when the caching
allocator is constructed, so a process that sets it after import may or may not be heard
depending on what has already touched CUDA — a check whose correctness depends on import order
is not a check. The launch environment stays the operator's act, and this makes a wrong one
loud at boot instead of a mystery at hour three. `AllocatorPostureMismatchError` is therefore
its own type on `RuntimeError` rather than a `ValueError`: the config is well-formed and the
INVOCATION is wrong, and sending an operator to the config file would be the wrong remedy.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

_KEY = "allocator_posture"

#: The two environment variables c10 reads, IN C10'S OWN PRECEDENCE ORDER, taken from the
#: pinned wheel's own source: `c10/cuda/CUDAAllocatorConfig.h` reads
#: `PYTORCH_CUDA_ALLOC_CONF` FIRST and falls back to `PYTORCH_ALLOC_CONF` only when the first
#: is unset. That order is the REVERSE of what torch's own prose suggests —
#: `c10/core/AllocatorConfig.h` calls `PYTORCH_ALLOC_CONF` "the primary environment variable
#: for configuration" and `PYTORCH_CUDA_ALLOC_CONF` a backward-compatibility name "with lower
#: priority". A check written from the documentation would read the pair backwards, and a
#: one-direction test would not see it: with only one variable set, both orders agree, and one
#: variable set is every ordinary launch. Both directions are pinned in
#: `tests/config/test_allocator_posture_authority.py`.
ALLOC_CONF_VARS: tuple[str, str] = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")

#: The entry point that PRODUCES the value, named in every refusal so an operator is never
#: left to guess where a measured regime comes from (R69: a number without its producing
#: mechanism is struck).
_MINT = "uv run python tools/mint_config.py"


class AllocatorPosture(StrEnum):
    """The closed regime set. A member here is a regime a cap can be fitted under."""

    DEFAULT = "default"
    EXPANDABLE_SEGMENTS = "expandable_segments"


#: token -> the allocator conf that posture REQUIRES, as a parsed mapping. DEFAULT requires
#: the EMPTY conf: "no allocator configuration at all", which is what the sitting's own
#: provenance stamped (`pytorch_cuda_alloc_conf: ""`) and what every banked oracle, the leg-2
#: budget and the banked bench side were taken under.
_REQUIRED_CONF: dict[AllocatorPosture, dict[str, str]] = {
    AllocatorPosture.DEFAULT: {},
    AllocatorPosture.EXPANDABLE_SEGMENTS: {"expandable_segments": "True"},
}


class MissingAllocatorPostureError(ValueError):
    """The allocator posture is not declared, or is not a member of the closed set.

    A `ValueError` for `MissingFusedGraphCapsError`'s reason: an absent regime declaration is
    a configuration ERROR, not a condition to recover from. Never caught anywhere.
    """


class UncalibratedAllocatorPostureError(MissingAllocatorPostureError):
    """The posture is the `null` placeholder: the key exists and has no minted value.

    A SUBCLASS, deliberately — an uncalibrated posture IS a special case of an unusable one,
    and it carries a remedy (measure it at the box, mint what was measured) the general
    absence does not.
    """


class AllocatorPostureMismatchError(RuntimeError):
    """The process is not running under the posture its config mints.

    NOT a `ValueError`, and the distinction is the remedy: the config is well-formed, the
    LAUNCH is wrong. Also raised for an AMBIGUOUS environment (both variables set to
    different confs) and for a CUDA process that was handed no posture at all.
    """


@dataclass(frozen=True)
class AllocatorPostureSpec:
    """The resolved regime. FROZEN for `FusedGraphCapsSpec`'s reason: a resolved run-scoped
    constant a consumer could rebind is a second authority with extra steps, and this one
    crosses a process seam (`RoundSpec`), where a rebind in the child would be invisible to
    the parent that measured the budget."""

    posture: AllocatorPosture

    @property
    def required_conf(self) -> dict[str, str]:
        return dict(_REQUIRED_CONF[self.posture])

    def required_env(self) -> dict[str, str]:
        """The environment this posture REQUIRES, as a mapping a launcher can splat.

        Landed with the R309(e) grant because the granted diff calls it: a test that must launch
        a child IN the config's minted posture needs the env as DATA, not as the human-readable
        `launch_hint` string beside it. Both render from `_REQUIRED_CONF`, so they cannot drift —
        which is the whole reason this is a method here rather than a dict built at a call site.

        The CUDA-named variable is the one written, because c10 reads it FIRST
        (`c10/cuda/CUDAAllocatorConfig.h`) — writing the generic one instead would produce an
        environment this module's own reader calls AMBIGUOUS the moment anything sets the other.
        DEFAULT renders as the EMPTY STRING rather than as an absent key: a launcher splatting
        this mapping over an inherited environment must be able to OVERRIDE an inherited posture,
        and a missing key cannot. `read_live_allocator_conf` parses `""` to the empty conf, which
        is exactly what DEFAULT requires.
        """
        conf = _REQUIRED_CONF[self.posture]
        rendered = ",".join(f"{k}:{v}" for k, v in sorted(conf.items()))
        return {ALLOC_CONF_VARS[0]: rendered}

    @property
    def launch_hint(self) -> str:
        """How to launch under this posture, in the shape an operator can paste."""
        conf = _REQUIRED_CONF[self.posture]
        if not conf:
            return f"unset {ALLOC_CONF_VARS[0]} and {ALLOC_CONF_VARS[1]}"
        rendered = ",".join(f"{k}:{v}" for k, v in sorted(conf.items()))
        return f'{ALLOC_CONF_VARS[0]}="{rendered}"'


@dataclass(frozen=True)
class LiveAllocatorConf:
    """What this process is ACTUALLY running under, and where that was read from.

    `source_var` is `None` only when neither variable is set. It is carried rather than
    discarded because a refusal that does not name the variable it read sends an operator to
    the wrong one — and the pair's precedence is the exact thing this module exists to get
    right.
    """

    source_var: str | None
    raw: str
    parsed: dict[str, str]
    #: BOTH variables are set to confs that disagree, in the one case where c10's own
    #: precedence could not be verified from the shipped headers (the CUDA-named variable set
    #: to the empty string). A separate flag rather than a sentinel inside `parsed`, so no
    #: caller can mistake it for a conf: the answer here is "this process cannot be told what
    #: it is running under", which is a different thing from "it is running under nothing".
    ambiguous: bool = False


def parse_alloc_conf(raw: str) -> dict[str, str]:
    """Parse an allocator conf string into a `key -> value` mapping, on c10's own grammar.

    STRUCTURAL, NEVER STRING EQUALITY — but structural means *c10's* structure, and the
    difference is not academic. Read from the pinned wheel's own source
    (`c10/core/AllocatorConfig.h`):

    * **Whitespace is stripped ENTIRELY**, not merely trimmed — `ConfigTokenizer` skips every
      `isspace` character wherever it appears, so ` expandable_segments : True ` and
      `expandable_segments:True` are the same conf to torch, and are the same conf here.
    * **NOTHING IS LOWER-CASED, and case is load-bearing.** `toBool` accepts EXACTLY `"True"`
      or `"False"` and `TORCH_CHECK_VALUE`s on anything else, and the key set is matched by
      exact token equality against lower-case literals. So `expandable_segments:true` is not
      a spelling variant of the regime — **torch REFUSES it**, and a check that normalised
      case would bless an environment the allocator will not accept. An earlier cut of this
      module did exactly that, and its refusal message told operators to launch with the
      spelling torch rejects; the red-team pass caught it against the header.

    Conversely `expandable_segments:True,max_split_size_mb:128` is a DIFFERENT conf and must
    not compare equal to the regime a cap was fitted under, which string containment would get
    wrong.

    Tolerant of shapes it does not understand rather than crashing on them: torch's grammar
    admits bracketed list values, and a part with no `:` is not a conf this repo mints. Both
    parse to something that simply will not equal a required conf, so they REFUSE at the
    comparison — the correct verdict for a foreign conf — instead of raising here, where the
    message would be about parsing rather than about the regime.
    """
    stripped = "".join(ch for ch in raw if not ch.isspace())
    parsed: dict[str, str] = {}
    for part in stripped.split(","):
        if not part:
            continue
        key, sep, value = part.partition(":")
        parsed[key] = value if sep else ""
    return parsed


def read_live_allocator_conf(environ: Mapping[str, str] | None = None) -> LiveAllocatorConf:
    """Read the live allocator conf from the environment, in c10's own precedence order.

    THE LIMIT, STATED, because it is the honest boundary of this check. The PARSED allocator
    configuration exists in C++ (`SnapshotInfo.config_metadata` carries both
    `expandable_segments` and `last_allocator_settings`), and whether it is reachable from
    Python could not be settled in this tree: the pinned wheel here is `2.11.0+cpu`, its CUDA
    bindings are compiled out, and an absent symbol in a CPU wheel is evidence about the wheel
    rather than about the API. No speculative branch is written for a getter this dispatch
    cannot reach and therefore cannot give a producer test — an untested branch on a
    memory-safety assertion is the phantom-input class R4/LAW-07 exist to kill. The
    environment pair, read the way c10 reads it, is what a Python process can observe; if the
    re-sit's cu128 wheel does expose the parsed config, promoting this reader to use it is an
    amendment worth taking, because it moves the assertion from "the launch environment says
    X" to "the allocator says X".

    ONE EDGE IS FAIL-CLOSED RATHER THAN GUESSED. The precedence itself is verified from the
    header. What is NOT verifiable from the shipped headers is whether c10's `get_env` treats
    an env var set to the EMPTY STRING as set. That only matters in one case — the CUDA-named
    variable set to `""` while the generic one carries a real conf — and there the two
    readings disagree about what the allocator is doing. This function reports that case as
    AMBIGUOUS (`source_var=None`, `raw` naming both) and `assert_posture_token` refuses it,
    rather than picking whichever answer happens to be convenient.
    """
    env = os.environ if environ is None else environ
    present = {name: env[name] for name in ALLOC_CONF_VARS if name in env}
    if not present:
        return LiveAllocatorConf(source_var=None, raw="", parsed={})
    primary, fallback = ALLOC_CONF_VARS
    if primary in present and present[primary].strip():
        return LiveAllocatorConf(primary, present[primary], parse_alloc_conf(present[primary]))
    if primary in present and fallback in present:
        if parse_alloc_conf(present[fallback]) != parse_alloc_conf(present[primary]):
            return LiveAllocatorConf(
                source_var=None,
                raw=f"{primary}={present[primary]!r} {fallback}={present[fallback]!r}",
                parsed=parse_alloc_conf(present[fallback]),
                ambiguous=True,
            )
    name = primary if primary in present else fallback
    return LiveAllocatorConf(name, present[name], parse_alloc_conf(present[name]))


def device_type_of(device: str) -> str:
    """`"cuda:0"` -> `"cuda"`. The ordinal is not part of the regime: one process, one
    caching allocator configuration, whichever card index it lands on."""
    return str(device).split(":", 1)[0].strip().lower()


#: The one device type the CUDA caching allocator's posture governs. Held HERE, once, and
#: never spelled at a call site: `tests/config/test_train_device_authority.py::
#: test_the_composition_root_hardcodes_no_device_string` bans a device string literal in
#: `mantis.run` outright (R126, DESIGN §1.2 item 3), and it is right to — the composition root
#: reads the device from the config and from nowhere else. `governs_device` below is the
#: predicate the root asks instead.
_GOVERNED_DEVICE_TYPE = "cuda"


def governs_device(device: str) -> bool:
    """True iff a process on `device` has a CUDA caching allocator for the posture to govern.

    The predicate lives beside the posture rather than at each call site so the device token
    is spelled ONCE in the repo, and so the two consumers — the composition root and the eval
    child — cannot disagree about which devices the regime applies to.
    """
    return device_type_of(device) == _GOVERNED_DEVICE_TYPE


def resolve_allocator_posture(full_config: Any) -> AllocatorPostureSpec:
    """Return the declared allocator posture. Absence raises, naming the level."""
    if not isinstance(full_config, Mapping):
        raise MissingAllocatorPostureError(
            f"{_KEY}: the config is not a mapping ({type(full_config).__name__}), so no "
            "posture can be read — a CUDA process would then run under whatever the launch "
            "happened to be, which is the unminted-precondition class this key exists to end"
        )
    if _KEY not in full_config:
        raise MissingAllocatorPostureError(
            f"{_KEY} is absent. Absent is an ERROR, never a default (R1/LAW-11): the caps on "
            "this card are fitted under ONE allocator regime, and a config that does not say "
            "which one cannot be checked against the process running it. The key is REQUIRED "
            "by the schema, so a config that reaches here without it was not built through "
            "the one loader."
        )
    declared = full_config[_KEY]
    if declared is None:
        raise UncalibratedAllocatorPostureError(
            f"{_KEY} is null — the R119 PLACEHOLDER, not an off state. `null` is schema-valid "
            "so the repo ships complete configs, and refused here so a CUDA run on an "
            "unminted regime cannot boot. The VALUE is a MEASUREMENT taken at the box "
            "(R308(g)(i) reserves it for the re-calibration sitting under R282(b)); mint what "
            "was measured, never a hand-picked token:\n"
            f"    {_MINT} --template <t> --out <this config> --force "
            f"--set {_KEY}=<{'|'.join(p.value for p in AllocatorPosture)}>"
        )
    try:
        return AllocatorPostureSpec(AllocatorPosture(declared))
    except ValueError as exc:
        raise MissingAllocatorPostureError(
            f"{_KEY}={declared!r} is not a member of the closed regime set "
            f"{[p.value for p in AllocatorPosture]}. The set is closed BECAUSE each member is "
            "a regime some cap was fitted under; a token with no fit behind it would read as "
            "a minted value and be one."
        ) from exc


def declared_allocator_posture(full_config: Any) -> str | None:
    """Return the DECLARED posture token, or `None` for the R119 placeholder. No verdict.

    The difference from `resolve_allocator_posture` is deliberate and is about WHO knows the
    device. This reads the key, REFUSES a token outside the closed regime set (garbage is
    always an error, whatever runs later), and passes the placeholder through untouched — so a
    caller that is only THREADING the value across a seam does not have to pronounce on
    whether the process at the other end can run.

    It exists for `compose_run`, threading the posture onto every `RoundSpec`.
    (AUDIT-1 F-52: this read "exactly ONE caller"; `diagnostics/worker_sweep.py` imports it
    too, to publish the DECLARED posture beside the LIVE one in its provenance block. The
    design claim is unchanged — this function threads and does not judge — but "exactly one"
    is a census, and it was wrong.) The eval child is the process that knows its own device and its own
    environment, and `assert_posture_token` there RAISES on a `None` token whenever
    `worker_device` is cuda — so the placeholder is refused, at the seam that can name what is
    wrong, rather than being silently accepted anywhere.

    WHY NOT ASSERT AT BOOT ON THE EVAL SIDE TOO — the honest version, because an earlier cut of
    this module did. It read better and it broke a FROZEN oracle
    (`tests/train/test_terminal_eval_rc.py`, which monkeypatches the collaborator builder and
    drives `compose_run` with a cuda-eval config). A frozen edit without a grant is an absolute
    exclusion, and "my check reads better" is not a grant. The enforcement lost is smaller than
    it looks: every committed config that declares a cuda EVAL device also declares a cuda
    TRAIN device, and the train-side boot assertion refuses those at
    `build_run_collaborators` before any allocation. The uncovered shape — cpu trainer, cuda
    eval child — is one no committed config has, and on it the child still refuses at its first
    round. Recorded as a DISCLOSED RESIDUAL rather than closed.
    """
    if not isinstance(full_config, Mapping):
        raise MissingAllocatorPostureError(
            f"{_KEY}: the config is not a mapping ({type(full_config).__name__})"
        )
    if _KEY not in full_config:
        raise MissingAllocatorPostureError(
            f"{_KEY} is absent. Absent is an ERROR, never a default (R1/LAW-11)."
        )
    declared = full_config[_KEY]
    if declared is None:
        return None
    try:
        return AllocatorPosture(declared).value
    except ValueError as exc:
        raise MissingAllocatorPostureError(
            f"{_KEY}={declared!r} is not a member of the closed regime set "
            f"{[p.value for p in AllocatorPosture]}"
        ) from exc


def assert_posture_token(
    token: str | None, *, device_type: str, environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assert this process's live allocator conf matches `token`. Returns the READING.

    The reading is returned rather than merely logged so a caller can record WHAT WAS
    CHECKED — including the not-enforced arm, which states its reason. "Not enforced" with no
    reason attached is indistinguishable from "not checked", and this repo has already paid
    for one gate whose silence read as a pass.

    `token=None` on a CUDA device RAISES. That combination means the caller resolved no
    posture for a process that has a caching allocator — the eval child's case, where the
    parent's resolution and the child's device would have to disagree for it to happen — and
    a silent skip there is the whole hole this assertion exists to close.
    """
    dtype = device_type_of(device_type)
    if not governs_device(dtype):
        return {
            "enforced": False,
            "reason": (
                f"device_type={dtype!r} has no CUDA caching allocator, so no allocator "
                "posture governs this process"
            ),
            "posture": token,
            "expected_conf": None,
            "observed_conf": None,
            "source_var": None,
        }
    if token is None:
        raise AllocatorPostureMismatchError(
            "a cuda process was handed no allocator posture. The posture is resolved in the "
            "parent and carried across the process seam; `None` here means the two disagree "
            "about the device, and running on would put a second allocator on the card under "
            "a regime nobody declared."
        )
    spec = AllocatorPostureSpec(AllocatorPosture(token))
    live = read_live_allocator_conf(environ)
    expected = spec.required_conf
    if live.ambiguous:
        raise AllocatorPostureMismatchError(
            f"{_KEY}={spec.posture.value!r} is minted, and this process cannot be told which "
            f"allocator configuration it is running under: {live.raw}. c10 reads "
            f"{ALLOC_CONF_VARS[0]} first and {ALLOC_CONF_VARS[1]} as a fallback, but whether "
            "an EMPTY value counts as set is not verifiable from the shipped headers, so the "
            "two readings disagree here. Set exactly one of the two variables and re-launch — "
            "guessing which one the allocator honoured is how a cap gets certified against a "
            "regime nobody was in."
        )
    if live.parsed != expected:
        raise AllocatorPostureMismatchError(
            f"{_KEY}={spec.posture.value!r} is minted, but this cuda process is running under "
            f"a different allocator configuration: observed {live.parsed or '{} (unset)'} "
            f"from {live.source_var or 'neither variable (or an ambiguous pair)'}, required "
            f"{expected or '{} (no allocator configuration at all)'}. A cap fitted under one "
            "posture is invalid under the other — this is not a warning, because the failure "
            "it prevents is a memory partition measured for a machine state you are not in. "
            f"Launch with: {spec.launch_hint}"
        )
    return {
        "enforced": True,
        "reason": "live allocator conf matches the minted posture",
        "posture": spec.posture.value,
        "expected_conf": expected,
        "observed_conf": dict(live.parsed),
        "source_var": live.source_var,
    }


def assert_allocator_posture(
    full_config: Any, *, device_type: str, environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the posture from a config and assert it, for a process on `device_type`.

    The resolution is SKIPPED on a non-CUDA device, deliberately: resolving would raise the
    R119 placeholder refusal on a cpu run, which has no CUDA caching allocator and therefore
    no regime to be wrong about. That is the same route-scoping `resolve_fused_graph_caps`
    takes on the grid arm, and it is why a cpu developer run is untouched by this key.
    """
    if not governs_device(device_type):
        return assert_posture_token(None, device_type=device_type, environ=environ)
    spec = resolve_allocator_posture(full_config)
    return assert_posture_token(spec.posture.value, device_type=device_type, environ=environ)


__all__ = [
    "ALLOC_CONF_VARS",
    "AllocatorPosture",
    "AllocatorPostureMismatchError",
    "AllocatorPostureSpec",
    "LiveAllocatorConf",
    "MissingAllocatorPostureError",
    "UncalibratedAllocatorPostureError",
    "assert_allocator_posture",
    "assert_posture_token",
    "declared_allocator_posture",
    "device_type_of",
    "governs_device",
    "parse_alloc_conf",
    "read_live_allocator_conf",
    "resolve_allocator_posture",
]

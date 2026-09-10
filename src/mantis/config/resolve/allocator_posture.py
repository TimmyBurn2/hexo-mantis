# >300 justify (R8). NO LINE COUNT is stated (derive-or-delete). This module is ONE claim with
# three faces that cannot be separated without creating a second authority over the same fact:
# what the config MINTED, what the process is ACTUALLY RUNNING UNDER (the environment pair, in
# c10's own precedence order), and the comparison between them. Splitting the reader out would
# put "which variable does torch read" in one module and "which posture was minted" in another,
# which is exactly the drift this exists to close.
"""`allocator_posture` — THE one authority over the CUDA caching allocator's REGIME.

WHY THIS KNOB EXISTS, measured: the 2026-08-22 sitting measured a card high-water of 14.98 GiB
under the DEFAULT posture against 11.36 GiB under `expandable_segments:True` — 3.62 GiB in the
quantity the whole memory partition is denominated in. It kept DEFAULT anyway, because a cap
fitted under that variable is valid only while it is set, and nothing minted, checked or armed
it. This module removes that objection and nothing else: it does not pick a value, and every
committed config carries the `null` placeholder.

A CLOSED TOKEN SET, NOT THE RAW ENVIRONMENT STRING: free text would make
`expandable_segments:True`, `...:true` and `expandable_segments:True,max_split_size_mb:128`
three configs claiming one regime, the third being a regime nobody fitted.

`null` IS NOT AN OFF STATE: schema-VALID so gate 7 stays green, runtime-REFUSED so a CUDA
process on an unminted posture cannot boot.

DEVICE-SCOPED, and each process asserts for ITS OWN device — the run process for
`config.train.device`, the eval child for `RoundSpec.worker_device`, since the child is a
second allocator on the same card that no in-process bound can see.

THE ASSERTION RAISES; IT DOES NOT SET. `PYTORCH_CUDA_ALLOC_CONF` is consumed when the caching
allocator is constructed, so a check whose correctness depends on import order is not a check.
`AllocatorPostureMismatchError` is a `RuntimeError`: the config is well-formed and the
INVOCATION is wrong.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

_KEY = "allocator_posture"

#: The two environment variables c10 reads, IN C10'S OWN PRECEDENCE ORDER, taken from the
#: pinned wheel's source and NOT from torch's prose, which states the reverse. A one-direction
#: test cannot see it: with only one variable set, both orders agree.
ALLOC_CONF_VARS: tuple[str, str] = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")

#: The entry point that PRODUCES the value, named in every refusal so an operator is never
#: left to guess where a measured regime comes from.
_MINT = "uv run python tools/mint_config.py"


class AllocatorPosture(StrEnum):
    """The closed regime set. A member here is a regime a cap can be fitted under."""

    DEFAULT = "default"
    EXPANDABLE_SEGMENTS = "expandable_segments"


#: token -> the allocator conf that posture REQUIRES, as a parsed mapping. DEFAULT requires
#: the EMPTY conf — no allocator configuration at all — which is what the sitting's provenance
#: stamped and what every banked oracle and bench side was taken under.
_REQUIRED_CONF: dict[AllocatorPosture, dict[str, str]] = {
    AllocatorPosture.DEFAULT: {},
    AllocatorPosture.EXPANDABLE_SEGMENTS: {"expandable_segments": "True"},
}


class MissingAllocatorPostureError(ValueError):
    """The allocator posture is not declared, or is not a member of the closed set. A
    `ValueError`: a configuration ERROR, not a condition to recover from."""


class UncalibratedAllocatorPostureError(MissingAllocatorPostureError):
    """The posture is the `null` placeholder: the key exists and has no minted value. A
    SUBCLASS, because "you never minted this" carries a remedy the general absence does not."""


class AllocatorPostureMismatchError(RuntimeError):
    """The process is not running under the posture its config mints — a `RuntimeError`, not a
    `ValueError`, because the config is well-formed and the LAUNCH is wrong. Also raised for an
    AMBIGUOUS environment and for a CUDA process handed no posture at all."""


@dataclass(frozen=True)
class AllocatorPostureSpec:
    """The resolved regime, FROZEN: a run-scoped constant a consumer could rebind is a second
    authority with extra steps, and this one crosses a process seam where a rebind in the child
    would be invisible to the parent that measured the budget."""

    posture: AllocatorPosture

    @property
    def required_conf(self) -> dict[str, str]:
        return dict(_REQUIRED_CONF[self.posture])

    def required_env(self) -> dict[str, str]:
        """The environment this posture REQUIRES, as a mapping a launcher can splat.

        Renders from `_REQUIRED_CONF` alongside `launch_hint`, so the two cannot drift. The
        CUDA-named variable is the one written, because c10 reads it FIRST. DEFAULT renders as
        the EMPTY STRING and not an absent key, since a launcher splatting this over an
        inherited environment must be able to OVERRIDE an inherited posture.
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
    """What this process is ACTUALLY running under, and where that was read from. `source_var`
    is `None` only when neither variable is set, and is carried because a refusal that does not
    name the variable it read sends an operator to the wrong one."""

    source_var: str | None
    raw: str
    parsed: dict[str, str]
    #: BOTH variables set to confs that disagree, in the one case where c10's precedence could
    #: not be verified from the shipped headers. A separate flag rather than a sentinel inside
    #: `parsed`, so no caller can mistake "cannot be told what it is running under" for
    #: "running under nothing".
    ambiguous: bool = False


def parse_alloc_conf(raw: str) -> dict[str, str]:
    """Parse an allocator conf string into a `key -> value` mapping, on c10's own grammar.

    STRUCTURAL, NEVER STRING EQUALITY, and structural means *c10's* structure: whitespace is
    stripped ENTIRELY rather than trimmed, and NOTHING is lower-cased, because `toBool` accepts
    exactly `"True"`/`"False"` — so `expandable_segments:true` is not a spelling variant, torch
    REFUSES it, and an earlier cut that normalised case told operators to launch with the
    spelling torch rejects. Conversely `expandable_segments:True,max_split_size_mb:128` is a
    DIFFERENT conf that string containment would wrongly accept. Unrecognised shapes parse to
    something that cannot equal a required conf, so they refuse at the comparison.
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

    THE LIMIT, STATED: the parsed configuration exists in C++, and whether it is reachable from
    Python could not be settled here — the pinned wheel is `2.11.0+cpu` with its CUDA bindings
    compiled out, so an absent symbol is evidence about the wheel, not the API. No speculative
    branch is written for a getter that can be given no producer test.

    ONE EDGE IS FAIL-CLOSED: whether c10 treats an env var set to the EMPTY STRING as set is
    not verifiable from the shipped headers, and it matters only when the CUDA-named variable
    is `""` while the generic one carries a real conf. That case is reported AMBIGUOUS and
    refused rather than guessed.
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


#: The one device type the posture governs, held HERE once and never spelled at a call site: a
#: test bans a device string literal in `mantis.run` outright, and `governs_device` below is
#: the predicate the composition root asks instead.
_GOVERNED_DEVICE_TYPE = "cuda"


def governs_device(device: str) -> bool:
    """True iff a process on `device` has a CUDA caching allocator for the posture to govern.
    Beside the posture so the device token is spelled ONCE, and so the composition root and the
    eval child cannot disagree about which devices apply."""
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
    """Return the DECLARED posture token, or `None` for the placeholder. No verdict.

    Unlike `resolve_allocator_posture` this refuses a token outside the closed set but passes
    the placeholder through, so a caller only THREADING the value across a seam need not
    pronounce on whether the far end can run. `compose_run` threads it onto every `RoundSpec`,
    and `worker_sweep` reads it to publish the DECLARED posture beside the LIVE one.

    WHY THE EVAL SIDE IS NOT ASSERTED AT BOOT: an earlier cut did and broke a FROZEN oracle.
    The enforcement lost is small — every committed config with a cuda eval device also has a
    cuda train device, which the train-side assertion already refuses, and the uncovered shape
    still refuses at the child's first round. A DISCLOSED RESIDUAL.
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

    Returned rather than logged so a caller can record WHAT WAS CHECKED, including the
    not-enforced arm and its reason: "not enforced" with no reason is indistinguishable from
    "not checked". `token=None` on a CUDA device RAISES — a silent skip there is the hole this
    assertion exists to close.
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
    Resolution is SKIPPED on a non-CUDA device: resolving would raise the placeholder refusal
    on a cpu run, which has no caching allocator and so no regime to be wrong about."""
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

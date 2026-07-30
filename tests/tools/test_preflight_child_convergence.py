"""⊕ WPMAIN ORACLE — the preflight child boots THE path, proved at the process boundary
(DESIGN §1.4/§9, oracles O-C1 + O-C2).

RED at drive time until IMPL re-points the child: the spawn below passes NO `--device`, and
at `b482243` the tool's `_require_preflight_args` refuses that with rc 2 (R126 kills the flag
— DESIGN ADDENDUM C.1.2). So the first assertion is the RED anchor, and it is red for the
right reason: the surface it drives is the one this WP deletes.

Why a SUBPROCESS oracle exists at all when `tests/test_run_one_authority.py` already censuses
the child's source: the dispatch's minimum set says "at SUBPROCESS level (real spawn, not
import-level only)", and it is right to. Every static census in this WP reads the file the
parent RE-EXECS — but the parent re-execs `os.path.abspath(__file__)` in a fresh interpreter,
and what that process actually did is only observable from what it left behind. The child's
own JSONL segment is that evidence, and it is evidence no AST can forge:

- `run_boot_identity` and `resolved_config` are emitted by `compose_run` and by nothing else
  in the tree. Their presence in the CHILD's log directory is proof the child went through
  the composition root, at the process boundary (O-C1).
- `run_boot_identity.config_sha256` is `config_identity_sha256` of the config the child
  ACTUALLY composed. The parent hashes the config IT loaded with the same one authority — so
  a child that read a different file, or composed a differently-overridden config, is a
  named mismatch instead of an invisible one. That is the F-B1 closure, re-run through the
  new path (O-C2).
- `run_boot_identity.run_id` is the CONFIG's run_id. At `b482243` the child passes
  `run_id=booted.run_id` explicitly; after R123 the parameter is gone and the composer reads
  `config.run_id` itself. Either way the published id must be the minted one and never the
  retired `"run"` default — this is the behavioural half of O-A5.

`tests/tools/test_preflight_armed_smoke.py` remains the R103 live consumer of
`configs/smoke_preflight_armed.yaml` and success criterion 3's own oracle; its ASSERTIONS go
green on the new path unedited (its one mechanical argv hunk — dropping the dead `--device
cpu` — is IMPL's, per the R88 census). This file does not restate it: it asserts the
convergence property that file never could, because a green report proves the tool worked,
not that the child took THE path.

INTEGRATION tier: a real ~30 s CPU boot + burst, the same drive class as the armed smoke.
Fakes: none. Real tool, real subprocess, real config, real burst.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

from mantis.config.loader import config_identity_sha256, load_config
from mantis.config.schema import RunConfig

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_TOOL = _REPO / "tools" / "ci_gates" / "preflight_mint.py"
_CONFIG = _REPO / "configs" / "smoke_preflight_armed.yaml"

#: The armed smoke's minimum legal burst plus headroom, and the number
#: `test_preflight_armed_smoke.py` drives — one burst length, one authority.
_BURST_STEPS = 16

#: The four events `compose_run` alone publishes into a run's own segment.
_COMPOSER_EVENTS = ("run_boot_identity", "resolved_config")


@pytest.fixture(scope="module")
def preflight_child(tmp_path_factory):
    """ONE real preflight spawn, shared by every assertion below (a second ~30 s boot to
    re-assert the same process would be waste, not independence)."""
    out_dir = tmp_path_factory.mktemp("preflight_convergence")
    proc = subprocess.run(
        [sys.executable, str(_TOOL), "--config", str(_CONFIG),
         "--burst-steps", str(_BURST_STEPS), "--out-dir", str(out_dir),
         "--timeout-sec", "400"],
        cwd=str(_REPO), capture_output=True, text=True, timeout=500,
    )
    return proc, out_dir


def _code_text(path: Path) -> str:
    """Source with COMMENT / STRING / f-string-literal tokens removed — the same helper, by
    the same guard idiom, as `tests/tools/test_preflight_mint.py`'s. FSTRING_MIDDLE is 3.12+
    (PEP 701); on the 3.11 floor f-strings lex as STRING."""
    skip = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    with path.open("rb") as handle:
        return "\n".join(tok.string for tok in tokenize.tokenize(handle.readline)
                         if tok.type not in skip)


def _child_events(out_dir: Path) -> list[dict]:
    """The CHILD's own event stream. `build_run_safety(log_dir=out_dir/"logs")` is the only
    thing that writes here, and only `compose_run` calls it."""
    segments = sorted((out_dir / "logs").glob("*.jsonl"))
    assert segments, (
        f"the child left no JSONL segment under {out_dir / 'logs'} — it never reached "
        "`build_run_safety`, which means it never reached the composition root"
    )
    return [json.loads(line) for segment in segments
            for line in segment.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_child_boots_green_with_no_device_flag_on_the_argv(preflight_child) -> None:
    """O-C1, arm 1 — and the RED anchor.

    The invocation carries `--config --burst-steps --out-dir --timeout-sec` and nothing
    else. At `b482243` that is rc 2 (`_require_preflight_args` lists `--device` among the
    required four). After R126 the flag does not exist and the child boots the CONFIG's own
    device — which is the whole property: preflighting run5 boots run5's minted device, so
    a `--device cpu` invocation can no longer false-clear a cuda run's memory wall
    (the WPBOX 16 GiB OOM; LAW-03's instrument-that-cannot-false-clear corollary).

    MUTATION THAT REDS IT: keep the flag. Nothing else in the WP drives the argv the parent
    actually builds — the source censuses read `_child_argv`'s TEXT; this runs it."""
    proc, out_dir = preflight_child
    tail = (proc.stdout + proc.stderr)[-3000:]
    assert proc.returncode == 0, (
        f"the preflight must be green on the new path (rc {proc.returncode}):\n{tail}"
    )
    reports = sorted(out_dir.glob("preflight_*.json"))
    assert reports, f"no evidence report written:\n{tail}"
    report = json.loads(reports[-1].read_text())
    assert report["verdict"] == "pass" and report["child"]["rc"] == 0


def test_the_child_process_left_the_composition_roots_own_boot_events(preflight_child) -> None:
    """O-C1, arm 2 — the convergence proof, at the process boundary.

    `run_boot_identity` and `resolved_config` are emitted by `compose_run` and by nothing
    else. A child that boots an approximation of the composition root — which is exactly
    what `_boot_main` IS at `b482243`, and what it would remain if the re-point were partial
    — cannot produce them.

    MUTATION THAT REDS IT: re-point the child at a shim that rebuilds the composition
    inline. rc stays 0, the evidence report stays `pass`, `test_preflight_armed_smoke.py`
    stays green, and the two boot paths are two again. No other oracle in this WP observes
    the child PROCESS."""
    _proc, out_dir = preflight_child
    names = [event.get("event") for event in _child_events(out_dir)]
    for event in _COMPOSER_EVENTS:
        assert names.count(event) == 1, (
            f"the child's segment must carry exactly one {event} — the composition root's "
            f"own boot record; got {names.count(event)} in {sorted(set(names))}"
        )


def test_the_childs_published_identity_is_the_config_it_actually_composed(
    preflight_child,
) -> None:
    """O-C2 — the `run_boot_identity` sha handshake, green through the new path.

    F-B1's defect was precisely that parent and child read the config INDEPENDENTLY and only
    the parent's identity was published, so a child that read a different file was invisible
    in the evidence artefact. The hash is recomputed here from the config the child was
    ASKED to boot, with the burst override applied the same way the child applies it
    (`dump -> mutate one key -> model_validate`, `preflight_mint.py:544-553`).

    MUTATION THAT REDS IT: publish the pre-override config's hash (the run then advertises a
    posture it is not running); or compose a config other than the burst-overridden one —
    the two-config boot O-A4 forbids structurally, caught here behaviourally, end to end.

    The `run_id` assertion is O-A5's behavioural half: the published id is the MINTED one,
    never `compose_run`'s retired `run_id: str = "run"` default in any guise. A segment
    named for a default id is a run whose logs cannot be attributed."""
    _proc, out_dir = preflight_child
    raw = load_config(_CONFIG).model_dump()
    raw["train"]["max_train_steps"] = _BURST_STEPS
    booted = RunConfig.model_validate(raw)

    identity = next(event for event in _child_events(out_dir)
                    if event.get("event") == "run_boot_identity")
    assert identity["config_sha256"] == config_identity_sha256(booted), (
        "the child published the identity of a DIFFERENT config than the one it was asked "
        "to boot — the F-B1 defect, at the seam F-B1 closed"
    )
    assert identity["run_id"] == booted.run_id == "smoke_preflight_armed", (
        f"the published run id must be the config's own; got {identity['run_id']!r}"
    )


def test_the_tool_no_longer_builds_a_single_collaborator_for_itself() -> None:
    """O-C1, arm 3 — the INVERTED O-9 census (§4): the four builder tokens
    (`init_trainer`, `WorkerPool`, `HexgBuffer`, `ReplayBuffer`) were REQUIRED to be present
    in the tool; they are now BANNED from it.

    RED FOR A REASON OTHER THAN "not built yet": this asserts a DELETION. At `b482243` all
    four are present and O-9 (`test_preflight_mint.py:940-970`) requires them to be. The
    predicate does not weaken — it inverts, which is the presence->ban pattern R121(a)
    sanctions and which O-10 already used. Its equal-or-stronger successors for the
    "the collaborators are real" claim are O-A3 (the builder-reality census, at the code's
    new home) and O-F1/O-B1 (behavioural drives through the real objects), named in DESIGN
    §4 — the census alone never earned that claim, and the tree measured why.

    MUTATION THAT REDS IT: leave one construction behind "just for the preflight". One is
    all it takes: a tool that builds even one collaborator differently is a tool that
    preflights a run5 nobody will launch.

    `build_run_safety(` and `StepCoordinatorConfig(` stay banned verbatim, as O-9 had them —
    those are `compose_run`'s to construct, and the throwaway coordinator-config call
    (`preflight_mint.py:906-911`) dies with D-3.

    The scan is over CODE with comment/string tokens removed — O-9's own `_code_text`
    instrument, kept for O-9's own stated reason: a raw-text census flags the tool's prose,
    and that is the false positive which teaches people to word comments around a gate. The
    tool's docstrings will go on NAMING these builders (they describe what the child boots);
    what must disappear is the tool constructing them."""
    source = _code_text(_TOOL)
    for token in ("init_trainer", "WorkerPool", "HexgBuffer", "ReplayBuffer",
                  "build_run_safety", "StepCoordinatorConfig"):
        assert token not in source, (
            f"the tool still names {token!r}: the boot lives at `mantis.run` now, and a CI "
            "gate that builds its own collaborators is the one-authority violation this WP "
            "exists to end (R121(a))"
        )

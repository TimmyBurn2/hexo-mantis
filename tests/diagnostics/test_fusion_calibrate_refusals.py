"""⊕ F-816-10 F8 — the calibration tool's REFUSALS (the only half that runs off the box).

Written by ORACLE-WRITE **before** the feature exists.

SCOPE, STATED FIRST SO THIS SUITE IS NOT OVER-READ. `python -m mantis.diagnostics.
fusion_calibrate` measures peak CUDA allocation over a sweep of fused batches and fits
`peak ~ a + b*E + c*N`. That measurement NEEDS A GPU and it is the box's job (design §9,
`plan/F816_10_BOX_PROCEDURE.md`). Nothing here asserts a fit, a byte count or a cap value —
oracles for things this machine cannot run are decoration, and this file deliberately does not
contain any. What it pins is the three behaviours that are fully determined OFF the box:

1. **A non-CUDA host REFUSES and emits NO cap** (design §9.3). The failure mode this prevents
   is the worst one in the packet: a calibration that "succeeds" on CPU produces a number with
   no producing mechanism, and R69 strikes a number without one. A CPU-derived cap minted into
   `configs/run5.yaml` would be exactly the guessed value R119 exists to forbid, wearing the
   tool's authority.
2. **`--shapes-only` reports the device-free half with NULLS, never extrapolations** — the
   unproduced-field convention (`docs/contracts/event_manifest.md`) applied to a report, plus
   an explicit `"calibrated": false` and NO mint line. A shapes-only report that printed a
   mint line would be a copy-pasteable command to mint an uncalibrated cap.
3. **`--budget-bytes` has NO default** (design §9.1 step 6): R1's shape applied to a tool. A
   default budget is a value nobody minted, and every cap the tool emits is a function of it.

Every row drives the REAL module as a SUBPROCESS with `CUDA_VISIBLE_DEVICES=""`, so the
non-CUDA arm is exercised deterministically on any host, GPU box included.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "configs" / "smoke_gnn.yaml"
_MODULE = "mantis.diagnostics.fusion_calibrate"
#: Any budget at all — the rows below are about the REFUSALS, and none of them reaches a fit.
_BUDGET = "9431000000"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the module with CUDA masked off. `-m` and not a loose script: entry points are
    `python -m mantis.*` or console scripts repo-wide (CLAUDE.md)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONWARNINGS"] = "ignore"
    return subprocess.run(
        [sys.executable, "-m", _MODULE, *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=600,
    )


def _report(proc: subprocess.CompletedProcess[str], cwd: Path) -> dict:
    """The tool's JSON report, from stdout or from the file it wrote in `cwd`.

    Deliberately permissive about WHERE the report lands — the design fixes its CONTENT
    (§9.3/§9.5) and leaves the destination open, so this helper accepts either rather than
    inventing a flag the design never specified."""
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        pass
    for path in sorted(cwd.rglob("*.json")):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    raise AssertionError(
        "no JSON report found on stdout or written under the working directory.\n"
        f"stdout:\n{proc.stdout[:2000]}\nstderr:\n{proc.stderr[:2000]}")


def _output(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout + proc.stderr


# ═══ FG8-01 — the non-CUDA refusal ═══════════════════════════════════════════════════════
def test_fg8_01_a_non_cuda_host_refuses_by_name_and_emits_no_cap(tmp_path) -> None:
    """FG8-01 — `torch.cuda.is_available()` False ⇒ exit 2 with a NAMED refusal, and NO cap.

    Not a warning, not a degraded CPU estimate, not an extrapolation from tensor sizes. The
    quantity being fitted is a CUDA allocator peak; on a host without one there is nothing to
    measure, and a number produced anyway would carry the tool's authority without its
    mechanism (R69: a number without its producing mechanism is struck)."""
    proc = _run("--config", str(_CONFIG), "--budget-bytes", _BUDGET,
                "--source", "synthetic", "--repeats", "1", cwd=tmp_path)
    out = _output(proc)
    assert proc.returncode == 2, (
        f"expected the named CUDA refusal (exit 2); got {proc.returncode}.\n{out[:2000]}")
    assert "cuda" in out.lower(), (
        f"the refusal does not name CUDA as the reason: {out[:1000]!r}")
    assert "max_fused_edges=" not in out and "max_fused_nodes=" not in out, (
        "the tool emitted a cap on a host that cannot measure one")
    assert "mint_config" not in out, (
        "the tool printed a copy-pasteable mint line for a value it never measured — that "
        "line is the operator's whole interface to R119 and it must only appear behind a "
        "real measurement")


# ═══ FG8-02 — --shapes-only reports nulls, not extrapolations ════════════════════════════
def test_fg8_02_shapes_only_reports_nulls_and_says_it_is_uncalibrated(tmp_path) -> None:
    """FG8-02 — the device-free half runs, and every measured field is `null`.

    `peak_bytes: null` and `fit: null` are the unproduced-field convention applied to a
    report: `null` says "no producer on this path", where a `0` or an extrapolated estimate
    would read as a measurement. `calibrated: false` is the same statement in one flag, so a
    consumer of the report cannot miss it by reading only the top level."""
    proc = _run("--config", str(_CONFIG), "--budget-bytes", _BUDGET, "--shapes-only",
                "--source", "synthetic", "--repeats", "1", cwd=tmp_path)
    out = _output(proc)
    assert proc.returncode == 0, (
        f"`--shapes-only` must SUCCEED on a non-CUDA host (that is its whole reason to "
        f"exist); got {proc.returncode}.\n{out[:2000]}")

    report = _report(proc, tmp_path)
    assert report.get("calibrated") is False, (
        f"the report does not declare itself uncalibrated: calibrated="
        f"{report.get('calibrated')!r}")
    assert "peak_bytes" in report, "the report omits `peak_bytes` instead of nulling it"
    assert report["peak_bytes"] is None, (
        f"`peak_bytes` is {report['peak_bytes']!r} on a host that measured nothing — an "
        "extrapolated or zeroed peak reads as a real measurement (the F-10 class)")
    assert "fit" in report, "the report omits `fit` instead of nulling it"
    assert report["fit"] is None, (
        f"`fit` is {report['fit']!r} with no measurement behind it")


def test_fg8_02_shapes_only_prints_no_mint_line(tmp_path) -> None:
    """FG8-02 second limb — stated separately because it is the one an implementer is most
    likely to leave in while making the rest of the report honest.

    The mint line is the operator's whole interface to R119. A `--shapes-only` run that
    printed one would hand over a copy-pasteable command to mint a cap that was never
    measured — which is worse than printing nothing, because it looks like the output of a
    calibration."""
    proc = _run("--config", str(_CONFIG), "--budget-bytes", _BUDGET, "--shapes-only",
                "--source", "synthetic", "--repeats", "1", cwd=tmp_path)
    out = _output(proc)
    assert proc.returncode == 0, (
        f"`--shapes-only` did not run at all, so the absence of a mint line below would be "
        f"vacuous; got {proc.returncode}.\n{out[:2000]}")
    assert "mint_config" not in out, (
        f"a `--shapes-only` run printed a mint line:\n{out[:2000]}")
    assert "--set inference.fused_graph_caps" not in out


def test_fg8_02_shapes_only_still_reports_the_shapes_it_did_measure(tmp_path) -> None:
    """FG8-02 third limb — the LAW-07 clean twin: `--shapes-only` is not simply refusing
    everything. The device-free half genuinely runs and the report carries the per-batch
    `(N, E)` and the operating ratio, which is the input the box sitting needs to choose its
    sweep before it ever allocates."""
    proc = _run("--config", str(_CONFIG), "--budget-bytes", _BUDGET, "--shapes-only",
                "--source", "synthetic", "--repeats", "1", cwd=tmp_path)
    report = _report(proc, tmp_path)
    text = json.dumps(report)
    assert "sweep" in report or "points" in report, (
        f"the shapes-only report carries no sweep at all: {text[:1000]}")
    for token in ("nodes", "edges"):
        assert token in text, (
            f"the shapes-only report never mentions {token} — it measured no shapes, which "
            "is the only thing it CAN measure")


# ═══ FG8-03 — --budget-bytes has no default ══════════════════════════════════════════════
def test_fg8_03_the_budget_has_no_default_and_omitting_it_is_an_error(tmp_path) -> None:
    """FG8-03 — R1's shape applied to a tool: a default budget is a value nobody minted, and
    every cap the tool emits is a function of it.

    Asserted as a REFUSAL that names the flag, not merely as a non-zero exit: an
    argparse-shaped error that named something else would be indistinguishable from the
    non-CUDA refusal at the exit code, which is 2 for both."""
    proc = _run("--config", str(_CONFIG), "--source", "synthetic", "--repeats", "1",
                cwd=tmp_path)
    out = _output(proc)
    assert proc.returncode != 0, (
        f"the tool ran with NO budget; `--budget-bytes` carries a default.\n{out[:2000]}")
    assert "--budget-bytes" in out, (
        f"the refusal does not name the missing flag: {out[:1000]!r}")


def test_fg8_03_the_help_text_does_not_advertise_a_budget_default(tmp_path) -> None:
    """FG8-03 second limb — the census over the tool's own interface. `--help` exits 0 and
    lists `--budget-bytes`; it must not print a `(default: ...)` for it, because a documented
    default is a value an operator will reach for without minting it."""
    proc = _run("--help", cwd=tmp_path)
    assert proc.returncode == 0, f"`--help` must succeed:\n{_output(proc)[:1000]}"
    out = proc.stdout
    assert "--budget-bytes" in out, "`--budget-bytes` is not an option at all"
    assert "--shapes-only" in out, "`--shapes-only` is not an option at all"
    # argparse wraps an option's help onto its own continuation lines, so the whole BLOCK is
    # read — from the line naming the flag up to the next option — not just that one line.
    lines = out.splitlines()
    # The USAGE line also mentions the flag; the OPTION entry is the one whose own text
    # begins with it. Matching the usage line instead would read the wrong block and the row
    # would pass against a tool that does advertise a default.
    starts = [i for i, line in enumerate(lines) if line.lstrip().startswith("--budget-bytes")]
    assert starts, (
        f"`--budget-bytes` never appears as an option entry in the help:\n{out[:1500]}")
    start = starts[-1]
    block: list[str] = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip().startswith("-") or not line.strip():
            break
        block.append(line)
    assert "default" not in " ".join(block).lower(), (
        f"`--budget-bytes` advertises a default: {block!r}. A documented default is a value "
        "an operator will reach for without minting it, and every emitted cap is a function "
        "of the budget (R1 applied to a tool)")


def test_the_margin_pin_is_0_85_READ_OFF_THE_PARSER_not_the_help(tmp_path) -> None:
    """R327(c) as a producer, not a memory — and read off the MECHANISM.

    The whole of conjunct 2's pass at the R326 mint is 0.79 % of margin-headroom: the partition
    closes at `k = 0.849998` and refuses at 0.86. `k` turned out to BE this knob, so the pin is
    the criterion, and a criterion that drifts to whatever the card afforded AFTER the card was
    measured has stopped being one.

    The value is taken from the argparse action argparse itself uses, never from the help text
    or from the `(0.85 when unset.)` note in it — a string an edit can move without moving the
    default is the proxy-not-mechanism trap. `margin_achieved` in the report would be the other
    mechanism reading, but it is `None` unless the tool RECOMMENDS, which needs the GPU this
    suite deliberately does not have.
    """
    from mantis.diagnostics.fusion_calibrate import build_parser

    action = next(a for a in build_parser()._actions if "--margin" in a.option_strings)
    assert action.default == 0.85, (
        f"the --margin pin moved to {action.default}; R327(c) pins it at 0.85 as PROCEDURE, "
        "and moving it to the measured affordability edge is criterion movement, not "
        "calibration"
    )


def test_the_margin_pins_rationale_names_the_value_that_would_refuse(tmp_path) -> None:
    """The second half, and the one a reader actually meets. A bare `0.85` reads as a round
    number nobody derived; the operator needs to see that 0.86 REFUSES the same partition on the
    same card, or the next sitting re-opens a settled question by looking reasonable."""
    proc = _run("--help", cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0.8568" in proc.stdout and "0.86" in proc.stdout, (
        "the --margin rationale no longer names the affordability edge or the refusing value:\n"
        + proc.stdout[:2000]
    )


@pytest.mark.parametrize("flag", ["--config", "--budget-bytes", "--shapes-only",
                                  "--source", "--repeats", "--margin"])
def test_fg8_03_the_designed_interface_exists(tmp_path, flag: str) -> None:
    """FG8-03 third limb — the six flags design §9 specifies are the interface the box
    procedure was written against. A tool whose flags drifted from the procedure is a box
    sitting that fails at the first command, hours from the machine that could fix it."""
    proc = _run("--help", cwd=tmp_path)
    assert flag in proc.stdout, (
        f"{flag} is missing from the tool's interface; `plan/F816_10_BOX_PROCEDURE.md` is "
        "written against it")

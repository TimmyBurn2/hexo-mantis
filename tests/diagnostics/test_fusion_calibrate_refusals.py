"""The calibration tool's REFUSALS — the only half that runs off the GPU box.

The fit needs a GPU, so nothing here asserts a fit, a byte count or a cap value. Every row
drives the REAL module as a subprocess with `CUDA_VISIBLE_DEVICES=""`, so the non-CUDA arm is
exercised deterministically on any host, GPU box included.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "configs" / "smoke_preflight_armed.yaml"
_MODULE = "mantis.diagnostics.fusion_calibrate"
#: Any budget at all — none of these rows reaches a fit.
_BUDGET = "9431000000"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the module as `python -m` with CUDA masked off."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONWARNINGS"] = "ignore"
    return subprocess.run(
        [sys.executable, "-m", _MODULE, *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=600,
    )


def _report(proc: subprocess.CompletedProcess[str], cwd: Path) -> dict:
    """Return the tool's JSON report, from stdout or from the file it wrote in `cwd` — the
    design fixes the report's content and leaves its destination open."""
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


def test_fg8_01_a_non_cuda_host_refuses_by_name_and_emits_no_cap(tmp_path) -> None:
    """A non-CUDA host exits 2 with a NAMED refusal, no cap and no mint line: the quantity
    fitted is a CUDA allocator peak, so there is nothing to estimate from."""
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


def test_fg8_02_shapes_only_reports_nulls_and_says_it_is_uncalibrated(tmp_path) -> None:
    """`--shapes-only` runs the device-free half and nulls every measured field: `null` says
    "no producer", where a `0` or an extrapolation would read as a measurement."""
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
    """`--shapes-only` prints no mint line — a copy-pasteable command to mint a cap nothing
    measured is worse than nothing, because it looks like the output of a calibration."""
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
    """The clean twin: `--shapes-only` is not refusing everything — the report carries the
    per-batch `(N, E)` the box sitting needs to choose its sweep before it allocates."""
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


def test_fg8_03_the_budget_has_no_default_and_omitting_it_is_an_error(tmp_path) -> None:
    """Omitting `--budget-bytes` is an error that NAMES the flag — a bare non-zero exit would
    be indistinguishable from the non-CUDA refusal, which is 2 as well."""
    proc = _run("--config", str(_CONFIG), "--source", "synthetic", "--repeats", "1",
                cwd=tmp_path)
    out = _output(proc)
    assert proc.returncode != 0, (
        f"the tool ran with NO budget; `--budget-bytes` carries a default.\n{out[:2000]}")
    assert "--budget-bytes" in out, (
        f"the refusal does not name the missing flag: {out[:1000]!r}")


def test_fg8_03_the_help_text_does_not_advertise_a_budget_default(tmp_path) -> None:
    """`--help` lists `--budget-bytes` with no `(default: ...)`: a documented default is a
    value an operator reaches for without minting it."""
    proc = _run("--help", cwd=tmp_path)
    assert proc.returncode == 0, f"`--help` must succeed:\n{_output(proc)[:1000]}"
    out = proc.stdout
    assert "--budget-bytes" in out, "`--budget-bytes` is not an option at all"
    assert "--shapes-only" in out, "`--shapes-only` is not an option at all"
    # argparse wraps help onto continuation lines, so the whole block is read.
    lines = out.splitlines()
    # The usage line also mentions the flag; the OPTION entry is the one whose text begins
    # with it, and matching the usage line would read the wrong block.
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
    """The `--margin` default is pinned at 0.85, read off the argparse action and never the
    help text. The pass has 0.79 % of headroom — the partition closes at `k = 0.849998` and
    refuses at 0.86 — so the pin IS the criterion."""
    from mantis.diagnostics.fusion_calibrate import build_parser

    action = next(a for a in build_parser()._actions if "--margin" in a.option_strings)
    assert action.default == 0.85, (
        f"the --margin pin moved to {action.default}; R327(c) pins it at 0.85 as PROCEDURE, "
        "and moving it to the measured affordability edge is criterion movement, not "
        "calibration"
    )


def test_the_margin_pins_rationale_names_the_value_that_would_refuse(tmp_path) -> None:
    """The help names the affordability edge and the refusing value: a bare `0.85` reads as a
    round number nobody derived, and the operator needs to see that 0.86 refuses."""
    proc = _run("--help", cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0.8568" in proc.stdout and "0.86" in proc.stdout, (
        "the --margin rationale no longer names the affordability edge or the refusing value:\n"
        + proc.stdout[:2000]
    )


@pytest.mark.parametrize("flag", ["--config", "--budget-bytes", "--shapes-only",
                                  "--source", "--repeats", "--margin"])
def test_fg8_03_the_designed_interface_exists(tmp_path, flag: str) -> None:
    """The six designed flags exist: a tool whose interface drifted from the box procedure is
    a sitting that fails at the first command, hours from the machine that could fix it."""
    proc = _run("--help", cwd=tmp_path)
    assert flag in proc.stdout, (
        f"{flag} is missing from the tool's interface; `plan/F816_10_BOX_PROCEDURE.md` is "
        "written against it")

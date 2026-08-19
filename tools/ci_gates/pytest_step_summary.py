"""Emit a pytest junit-xml digest to the GitHub step summary.

WHY THIS EXISTS. The 2026-08-19 gate-3b red on run 32195010976 was undiagnosable from
outside: step logs are admin-gated on this repo (403 for unauthenticated readers), and
check-run annotations carried only "Process completed with exit code 1". The tier is green
locally, so the ONLY place the failure exists is the runner — and the runner's evidence was
unreadable. Job summaries ($GITHUB_STEP_SUMMARY) render on the public run page, so a digest
written there is readable by anyone with the run URL, dispatcher sessions included.

Reads one junit xml (pytest --junitxml), writes: failed/errored test ids with the first
lines of their message, pass/fail/skip counts, and the slowest cases. Exit code is ALWAYS 0
unless the xml itself is unreadable — this tool reports on a gate, it is not the gate
(the pytest step's own exit code remains the verdict; R4/LAW-07 unaffected).
"""
from __future__ import annotations

# stdlib ElementTree is acceptable here: the xml is written by pytest in the SAME job
# (trusted, self-produced), xml.etree rejects external entities outright, and the 3.11
# floor means expat >= 2.4 with billion-laughs amplification protection. Revisit only if
# this tool ever reads xml it did not produce.
import argparse
import os
import sys
import xml.etree.ElementTree as ET


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("junit_xml")
    ap.add_argument("--title", default="pytest")
    ap.add_argument("--slowest", type=int, default=15)
    args = ap.parse_args()

    out_path = os.environ.get("GITHUB_STEP_SUMMARY")
    out = open(out_path, "a", encoding="utf-8") if out_path else sys.stdout

    if not os.path.exists(args.junit_xml):
        print(f"## {args.title}: no junit xml at `{args.junit_xml}` "
              "(step likely died before pytest wrote it)", file=out)
        return 0

    try:
        root = ET.parse(args.junit_xml).getroot()
    except ET.ParseError as exc:
        print(f"## {args.title}: junit xml unparseable: {exc}", file=out)
        return 1

    suites = root.iter("testsuite")
    cases = list(root.iter("testcase"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for s in suites:
        for k in totals:
            totals[k] += int(s.get(k, 0) or 0)

    print(f"## {args.title}: {totals['tests']} tests — "
          f"{totals['failures']} failed, {totals['errors']} errored, "
          f"{totals['skipped']} skipped", file=out)

    bad = [(c, kind)
           for c in cases
           for kind in ("failure", "error")
           if c.find(kind) is not None]
    for case, kind in bad:
        node = case.find(kind)
        if node is None:  # unreachable by construction of `bad`; narrows the type
            continue
        msg = (node.get("message") or (node.text or "")).strip()
        first = "\n".join(msg.splitlines()[:6])
        test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
        print(f"\n### {kind.upper()}: `{test_id}` "
              f"({float(case.get('time', 0) or 0):.1f}s)\n```\n{first}\n```", file=out)

    timed = sorted(cases, key=lambda c: float(c.get("time", 0) or 0), reverse=True)
    print(f"\n### slowest {args.slowest}", file=out)
    for c in timed[: args.slowest]:
        print(f"- {float(c.get('time', 0) or 0):.1f}s "
              f"`{c.get('classname', '')}::{c.get('name', '')}`", file=out)

    if out is not sys.stdout:
        out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

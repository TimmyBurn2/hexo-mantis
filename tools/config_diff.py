"""Config delta assertions: two-file --expect diff, and a self-contained --from-header check.

Two-file mode (unchanged):
    uv run python tools/config_diff.py A.yaml B.yaml --expect dotted.key [--expect k2 ...]
  Exit 0 iff {keys whose values differ} == {expected}; exit 1 on any mismatch; exit 2 on load error.

Lying-header mode (B3, red-team #3):
    uv run python tools/config_diff.py --from-header <config.yaml>
  Parses the config's stamped header (`# template:` + `# delta:` lines) → the CLAIMED delta-key
  set, re-diffs the config against tools/config_templates/<t>.yaml, and asserts claimed == actual.
  Exit 0 MATCH; exit 1 naming the lie (an omitted real diff OR a claimed-but-unchanged key); exit 2
  on load/parse error (missing template, unparseable header, invalid config).

Both configs must schema-validate (an invalid config can't be diff-asserted).
"""
import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from mantis.config.loader import load_config

TEMPLATES_DIR = Path(__file__).resolve().parent / "config_templates"


def _flatten(node: object, prefix: str = "") -> dict[str, object]:
    if isinstance(node, dict):
        out: dict[str, object] = {}
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
        return out
    return {prefix: node}


def _diff_keys(flat_a: dict[str, object], flat_b: dict[str, object]) -> set[str]:
    return {k for k in set(flat_a) | set(flat_b) if flat_a.get(k) != flat_b.get(k)}


def _covers(claimed: str, actual: str) -> bool:
    """True iff a delta claimed on `claimed` accounts for the real diff at `actual`.

    Exact match, or `actual` sits INSIDE the block `claimed` names. A mint delta may set a
    whole BLOCK — `--set 'train.draw_rate_abort={threshold: …, min_step: …, N_pool_min: …}'`
    — and in some cases it MUST: `mint_config._resolve_parent` requires every path segment to
    exist in the template, so a leaf inside a template block that ships `null` cannot be
    addressed at all. The header then truthfully claims one key while the flattened diff
    reports its leaves, and comparing the two sets literally reads that as a lying header.

    The widening is bounded and it is one-way: a claimed block with NO real diff under it is
    still reported (`HEADER CLAIMS … but it is unchanged`), so a header cannot claim a delta
    it did not make; and a real diff OUTSIDE every claimed block is still reported, so a
    header cannot hide one. What it stops asserting is that a delta names a LEAF, which was
    never the rule — it was an artefact of every previous delta happening to be one.
    """
    return actual == claimed or actual.startswith(f"{claimed}.")


def _parse_header(text: str) -> tuple[str | None, set[str]]:
    """Return (template_name, claimed_delta_keys) from a stamped mint header.

    Reads leading `#`-comment lines only (header ends at the first non-comment line). A
    `# delta: <dotted>: <old> -> <new>` line contributes its <dotted> key.
    """
    template: str | None = None
    claimed: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        if line.startswith("# template:"):
            template = line.split(":", 1)[1].strip()
        elif line.startswith("# delta:"):
            key = line[len("# delta:") :].strip().split(":", 1)[0].strip()
            if key:
                claimed.add(key)
    return template, claimed


def _run_from_header(config_path: str) -> int:
    path = Path(config_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read config: {exc}", file=sys.stderr)
        return 2
    template, claimed = _parse_header(text)
    if not template:
        print("unparseable header: no '# template:' line found", file=sys.stderr)
        return 2
    template_path = TEMPLATES_DIR / f"{template}.yaml"
    if not template_path.is_file():
        print(f"missing template: {template_path}", file=sys.stderr)
        return 2
    try:
        flat_cfg = _flatten(load_config(path).model_dump())
        flat_tmpl = _flatten(load_config(template_path).model_dump())
    except (ValidationError, yaml.YAMLError, OSError, TypeError) as exc:
        print(f"load/validation error: {exc}", file=sys.stderr)
        return 2
    actual = _diff_keys(flat_cfg, flat_tmpl)
    unclaimed = {k for k in actual if not any(_covers(c, k) for c in claimed)}
    empty_claims = {c for c in claimed if not any(_covers(c, k) for k in actual)}
    if not unclaimed and not empty_claims:
        print("MATCH:", ", ".join(sorted(actual)))
        return 0
    for k in sorted(unclaimed):
        print(f"HEADER OMITS a real diff on {k}: {flat_tmpl.get(k)!r} -> {flat_cfg.get(k)!r}")
    for k in sorted(empty_claims):
        print(f"HEADER CLAIMS {k} but it is unchanged vs template")
    return 1


def _run_expect(config_a: str, config_b: str, expect: list[str]) -> int:
    try:
        flat_a = _flatten(load_config(config_a).model_dump())
        flat_b = _flatten(load_config(config_b).model_dump())
    except (ValidationError, yaml.YAMLError, OSError, TypeError) as exc:
        print(f"load/validation error: {exc}", file=sys.stderr)
        return 2
    diff_keys = _diff_keys(flat_a, flat_b)
    expected = set(expect)
    if diff_keys == expected:
        print("MATCH:", ", ".join(sorted(diff_keys)))
        return 0
    for k in sorted(diff_keys - expected):
        print(f"UNCLAIMED diff on {k}: {flat_a.get(k)!r} -> {flat_b.get(k)!r}")
    for k in sorted(expected - diff_keys):
        print(f"expected diff on {k} but values are identical")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-header", dest="from_header", metavar="CONFIG",
        help="re-diff a config against its header-named template; assert claimed == actual",
    )
    parser.add_argument("config_a", nargs="?")
    parser.add_argument("config_b", nargs="?")
    parser.add_argument(
        "--expect", action="append", metavar="dotted.key", default=[],
        help="key expected to differ (repeatable); the diff must be exactly this set",
    )
    args = parser.parse_args(argv)

    if args.from_header:
        return _run_from_header(args.from_header)
    if not (args.config_a and args.config_b and args.expect):
        print(
            "usage: config_diff.py A B --expect KEY [...]  |  config_diff.py --from-header CONFIG",
            file=sys.stderr,
        )
        return 2
    return _run_expect(args.config_a, args.config_b, args.expect)


if __name__ == "__main__":
    sys.exit(main())

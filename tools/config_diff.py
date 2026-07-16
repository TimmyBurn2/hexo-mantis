"""One-key-diff assert: verify two configs differ exactly where claimed.

CLI: uv run python tools/config_diff.py A.yaml B.yaml --expect dotted.key [--expect k2 ...]

Both inputs must schema-validate (an invalid config can't be "diff-asserted").
Exit 0 iff {keys whose values differ} == {expected} (prints MATCH + keys); exit 1 with
the symmetric difference printed otherwise; exit 2 on load/validation error.
"""
import argparse
import sys

import yaml
from pydantic import ValidationError

from mantis.config.loader import load_config


def _flatten(node: object, prefix: str = "") -> dict[str, object]:
    if isinstance(node, dict):
        out: dict[str, object] = {}
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
        return out
    return {prefix: node}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_a")
    parser.add_argument("config_b")
    parser.add_argument(
        "--expect", action="append", required=True, metavar="dotted.key",
        help="key expected to differ (repeatable); the diff must be exactly this set",
    )
    args = parser.parse_args(argv)

    try:
        flat_a = _flatten(load_config(args.config_a).model_dump())
        flat_b = _flatten(load_config(args.config_b).model_dump())
    except (ValidationError, yaml.YAMLError, OSError, TypeError) as exc:
        print(f"load/validation error: {exc}", file=sys.stderr)
        return 2

    diff_keys = {k for k in flat_a if flat_a[k] != flat_b[k]}
    expected = set(args.expect)
    if diff_keys == expected:
        print("MATCH:", ", ".join(sorted(diff_keys)))
        return 0
    for k in sorted(diff_keys - expected):
        print(f"UNCLAIMED diff on {k}: {flat_a[k]!r} -> {flat_b[k]!r}")
    for k in sorted(expected - diff_keys):
        print(f"expected diff on {k} but values are identical")
    return 1


if __name__ == "__main__":
    sys.exit(main())

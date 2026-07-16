"""Mint a complete, schema-valid config from a named template + explicit deltas.

CLI: uv run python tools/mint_config.py --template dev --out configs/<name>.yaml
     [--set dotted.key=value ...] [--force]

Every --set key MUST already exist in the template (creating a new key via mint is an
error — the schema would reject it anyway). The delta is stamped into the file header
(repo_design §5 copy-drift antidote). Exit 0 ok; 2 on unknown template, unknown delta
key, schema-invalid result, existing output without --force, or usage error.
"""
import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from mantis.config.schema import RunConfig

TEMPLATES_DIR = Path(__file__).resolve().parent / "config_templates"


def _resolve_parent(data: dict, dotted: str) -> tuple[dict, str]:
    """Return (parent mapping, leaf key) for an EXISTING dotted key; KeyError otherwise."""
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    leaf = parts[-1]
    if not isinstance(node, dict) or leaf not in node:
        raise KeyError(dotted)
    return node, leaf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template", required=True, help="template name under tools/config_templates/"
    )
    parser.add_argument("--out", required=True, help="output config path (e.g. configs/name.yaml)")
    parser.add_argument(
        "--set", dest="deltas", action="append", default=[], metavar="dotted.key=value",
        help="delta applied to the template; the key must already exist",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = parser.parse_args(argv)

    template_path = TEMPLATES_DIR / f"{args.template}.yaml"
    if not template_path.is_file():
        print(f"unknown template: {args.template} (no {template_path})", file=sys.stderr)
        return 2
    data = yaml.safe_load(template_path.read_text())

    delta_lines: list[str] = []
    for raw in args.deltas:
        if "=" not in raw:
            print(f"malformed --set (need dotted.key=value): {raw}", file=sys.stderr)
            return 2
        dotted, _, raw_value = raw.partition("=")
        try:
            parent, leaf = _resolve_parent(data, dotted)
        except KeyError:
            print(f"unknown delta key (not in template): {dotted}", file=sys.stderr)
            return 2
        old = parent[leaf]
        new = yaml.safe_load(raw_value)
        parent[leaf] = new
        delta_lines.append(f"# delta: {dotted}: {old} -> {new}")

    try:
        RunConfig.model_validate(data)
    except ValidationError as exc:
        print(f"minted config does not schema-validate:\n{exc}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite {out_path} (use --force)", file=sys.stderr)
        return 2
    header = "\n".join(
        ["# minted-by: tools/mint_config.py", f"# template: {args.template}", *delta_lines]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "\n" + yaml.safe_dump(data, sort_keys=False))
    print(f"minted {out_path} from template {args.template} ({len(delta_lines)} delta(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())

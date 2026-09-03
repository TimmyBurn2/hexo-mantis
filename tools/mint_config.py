"""Mint a complete, schema-valid config from a named template + explicit deltas.

CLI: uv run python tools/mint_config.py --template dev --out configs/<name>.yaml
     [--set dotted.key=value ...] [--force]

Every --set key MUST already exist in the template (creating a new key via mint is an
error — the schema would reject it anyway). The delta is stamped into the file header
(repo_design §5 copy-drift antidote) as REPLAYABLE YAML — see `_render_value` (R187). Exit 0
ok; 2 on unknown template, unknown delta key, a delta value the header cannot record
replayably, schema-invalid result, existing output without --force, or usage error.

`--out` takes a FREE path and puts no constraint on its shape (R75 declined the suffix guard
that briefly stood here). The mint path is covered shape-agnostically instead: a config minted
under `configs/` is enumerated by `discover_configs` whatever it is called, and one minted
anywhere else is audited by `preflight_mint.py --config <path>`, which unions any named path
into the audit set regardless of shape.
"""
import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from mantis.config.schema import RunConfig
from mantis.util.yaml_io import parse_config_yaml

TEMPLATES_DIR = Path(__file__).resolve().parent / "config_templates"

#: The `# delta:` line's old/new separator. A rendered value may not contain it, or the line
#: stops being splittable back into two values (R187).
HEADER_SEP = " -> "

#: Emitter width. The header is ONE line per delta by format, so line-wrapping must be off;
#: PyYAML has no "never wrap" switch, so the width is set past any plausible value.
_UNWRAPPED = 1 << 30


class HeaderRenderError(ValueError):
    """A delta value that cannot be recorded in the header as replayable YAML.

    Raised, never swallowed: a header that cannot be read back is the defect R187 names, and a
    serializer that silently emits an approximation is the same defect with a different input.
    """


def _identical(left: object, right: object) -> bool:
    """Type-strict structural equality — the round-trip predicate.

    `==` alone is too weak here: `True == 1`, `1.0 == 1`, and `[1] == [True]` all hold, so an
    `==`-only check would bless a render that silently retypes a value. NaN needs `repr`
    because it is not equal to itself.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        assert isinstance(right, dict)
        return len(left) == len(right) and all(
            any(_identical(key, other) and _identical(value, right[other]) for other in right)
            for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple))
        return len(left) == len(right) and all(
            _identical(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float):
        assert isinstance(right, float)
        return repr(left) == repr(right)
    return bool(left == right)


def _render_value(value: object, *, where: str) -> str:
    """Render a delta value as the one-line YAML text `--set` parses back into that value.

    The header's value domain is EXACTLY the image of `yaml.safe_load`: both slots come from
    it — the old value from the template load, the new one from `yaml.safe_load(raw_value)`
    below. `yaml.safe_dump` is that loader's inverse over its own image, which is why it is
    the serializer here and Python `str()` is not: `str()` is neither total (`None` -> `None`,
    which reads back as the STRING `"None"`; `inf`, `nan`, `set`, `bytes`, `('a','b')` all
    read back wrong or not at all) nor injective (`None` and `"None"` render identically), so
    a header written with it is not replayable by the tool that wrote it — and R1's "configs
    are minted, never hand-varied" rests on exactly that replayability (R187).

    Totality is not assumed, it is CHECKED per value: the rendering is parsed back and must be
    structurally identical, and must be one line and separator-free. Anything else raises. The
    two values in the domain that `safe_dump` cannot round-trip — `!!omap` and `!!pairs`, which
    load as lists of tuples and dump as lists of lists — therefore refuse the mint loudly
    instead of being written down wrong.
    """
    try:
        dumped = yaml.safe_dump(
            [value], default_flow_style=True, sort_keys=False, width=_UNWRAPPED,
            allow_unicode=True,
        ).strip()
    except yaml.YAMLError as exc:
        raise HeaderRenderError(f"{where}: value is not YAML-serialisable: {exc}") from exc
    # Dumped as a one-element flow sequence so a top-level scalar cannot pick up a `...`
    # document-end marker; the brackets come straight back off.
    if not (dumped.startswith("[") and dumped.endswith("]")):
        raise HeaderRenderError(f"{where}: unexpected serialisation {dumped!r}")
    rendered = dumped[1:-1].strip()
    if "\n" in rendered:
        raise HeaderRenderError(f"{where}: value does not fit one header line: {rendered!r}")
    if HEADER_SEP in rendered:
        raise HeaderRenderError(
            f"{where}: rendered value contains the header separator {HEADER_SEP!r} and would "
            f"make the delta line ambiguous: {rendered!r}"
        )
    try:
        back = yaml.safe_load(rendered)
    except yaml.YAMLError as exc:
        raise HeaderRenderError(f"{where}: rendered value does not parse back: {exc}") from exc
    if not _identical(back, value):
        raise HeaderRenderError(
            f"{where}: rendered value does not round-trip: {value!r} -> {rendered!r} -> {back!r}"
        )
    return rendered


def _delta_line(dotted: str, old: object, new: object) -> str:
    """The header's provenance record for one delta, both slots replayable (R187)."""
    old_text = _render_value(old, where=f"delta {dotted} (old)")
    new_text = _render_value(new, where=f"delta {dotted} (new)")
    return f"# delta: {dotted}: {old_text}{HEADER_SEP}{new_text}"


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
    # AUDIT-1 F-45. THE config parser, not a third `yaml.safe_load`: a template with a
    # duplicate key would mint a config whose loaded value silently differs from the one
    # the template appears to state, and the minted file is the artifact of record.
    data = parse_config_yaml(template_path)

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
        try:
            line = _delta_line(dotted, old, new)
        except HeaderRenderError as exc:
            print(f"cannot stamp a replayable header: {exc}", file=sys.stderr)
            return 2
        parent[leaf] = new
        delta_lines.append(line)

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
    out_path.write_text(header + "\n" + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"minted {out_path} from template {args.template} ({len(delta_lines)} delta(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# >300 justify (R8): the behavioural half defines what the renderer does and the census half
# asserts the committed configs are in that form; split, either half asserts a format no test
# defines, and the parser under test is duplicated.
"""A minted header is REPLAYABLE: every delta value round-trips through the tool.

Both header slots come from `yaml.safe_load`, so the value domain is exactly the image of
PyYAML's `SafeLoader`, and Python `str()` is neither total nor injective over that image — it
maps `None` and the string `"None"` onto the same text. The census half checks every committed
header both textually (is the slot the canonical rendering?) and semantically (does it equal
the body's value at that dotted path?).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mantis.config.loader import discover_configs, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MINT = REPO_ROOT / "tools" / "mint_config.py"
BASELINE = REPO_ROOT / "tests" / "fixtures" / "wpmain" / "config_baseline_b482243"

#: The mint tool as a module, so the census calls the SAME renderer the tool stamps with.
_SPEC = importlib.util.spec_from_file_location("_mint_config_under_test", MINT)
assert _SPEC is not None and _SPEC.loader is not None
mint_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mint_config)

#: One source document per `SafeConstructor` tag plus the strings YAML would retype;
#: `test_the_domain_table_covers_every_safe_loader_tag` holds the table to that domain.
_DOMAIN_DOCS = (
    "null", "~", "true", "false", "42", "0x1f", "1.5", ".inf", "-.inf", ".nan",
    "!!binary 'AAEC'", "2026-08-03", "2026-08-03 11:22:33", "!!set {a, b}",
    "'None'", "'null'", "'yes'", "'0123'", "'a: b'", "''", "'#hash'", "'- dash'", "'%pct'",
    "[1, 2, 3]", "{a: 1, b: null}", "[{name: x, opponent_sims: null, deploy_matched: true}]",
)

#: The two tags `safe_dump` cannot invert (list of TUPLES out, list of lists back).
_UNRENDERABLE_DOCS = ("!!omap [{a: 1}]", "!!pairs [{a: 1}]")

#: A byte-frozen snapshot under a FROZEN manifest — a record, not a mintable config.
_BASELINE_KNOWN_BAD = {
    ("run6.yaml", "train.draw_rate_abort"),
    ("smoke_preflight_armed.yaml", "train.draw_rate_abort"),
    ("smoke_preflight_armed.yaml", "eval.ladder.rungs"),
}


def _run_mint(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(MINT), "--template", "dev", *argv],
                          cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)


def _header_deltas(path: Path) -> list[tuple[str, str, str]]:
    """(dotted key, old slot text, new slot text) for every `# delta:` line in the header."""
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        if not line.startswith("# delta:"):
            continue
        key, _, rest = line[len("# delta:"):].strip().partition(":")
        old, sep, new = rest.strip().partition(mint_config.HEADER_SEP)
        assert sep, f"{path.name}: undelimited delta line {line!r}"
        rows.append((key.strip(), old, new))
    return rows


def _at(data: object, dotted: str) -> object:
    for part in dotted.split("."):
        assert isinstance(data, dict), f"{dotted} does not resolve"
        data = data[part]
    return data


def _stringified_none(value: object) -> bool:
    if isinstance(value, str):
        return value == "None"
    if isinstance(value, dict):
        return any(_stringified_none(k) or _stringified_none(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_stringified_none(item) for item in value)
    return False


def test_a_None_bearing_delta_mints_a_header_that_replays_green(tmp_path: Path) -> None:
    """A `None`-bearing delta mints a header whose slot replays to the SAME BYTES.
    Killer: `str()` in the delta line, which replays `opponent_sims: 'None'` and exits 2."""
    rungs = ("[{name: sealbot_d1, bot: sealbot, variant: d1, depth: 1, opponent_sims: null, "
             "opening_book: book_v1_s20260625_p4, deploy_matched: true, games_max: 1}]")
    first = tmp_path / "first.yaml"
    minted = _run_mint("--out", str(first), "--set", "run_id=none_bearing",
                       "--set", f"eval.ladder.rungs={rungs}")
    assert minted.returncode == 0, (minted.stdout + minted.stderr)[-2000:]
    assert load_config(first).eval.ladder.rungs[0].opponent_sims is None

    deltas = dict((key, new) for key, _old, new in _header_deltas(first))
    assert yaml.safe_load(deltas["eval.ladder.rungs"])[0]["opponent_sims"] is None, (
        f"the header recorded {deltas['eval.ladder.rungs']!r}, which reads back with a "
        "stringified None -- the R187 defect"
    )
    second = tmp_path / "second.yaml"
    replayed = _run_mint("--out", str(second), *[
        arg for key, new in deltas.items() for arg in ("--set", f"{key}={new}")
    ])
    assert replayed.returncode == 0, (
        "a minted config's own header must replay through its own minter (R1/R187); rc "
        f"{replayed.returncode}\n{(replayed.stdout + replayed.stderr)[-2000:]}"
    )
    assert second.read_text(encoding="utf-8") == first.read_text(encoding="utf-8"), (
        "the replay must reproduce the file byte-for-byte, header included"
    )


def test_None_and_the_string_None_are_distinguishable_in_the_header(tmp_path: Path) -> None:
    """`None` and the string `"None"` mint distinguishable headers. Killer: any renderer that
    emits a bare `None` for either, which cannot say which value was minted."""
    as_null = tmp_path / "null.yaml"
    as_text = tmp_path / "text.yaml"
    rung = ("[{{name: r, bot: sealbot, variant: d1, depth: 1, opponent_sims: null, "
            "opening_book: {book}, deploy_matched: true, games_max: 1}}]")
    assert _run_mint("--out", str(as_null), "--set", "train.draw_rate_abort=null"
                     ).returncode == 0
    minted = _run_mint("--out", str(as_text), "--set",
                       "eval.ladder.rungs=" + rung.format(book="'None'"))
    assert minted.returncode == 0, (minted.stdout + minted.stderr)[-2000:]

    null_slot = dict((k, n) for k, _o, n in _header_deltas(as_null))["train.draw_rate_abort"]
    text_slot = dict((k, n) for k, _o, n in _header_deltas(as_text))["eval.ladder.rungs"]
    assert yaml.safe_load(null_slot) is None, f"null slot rendered {null_slot!r}"
    assert yaml.safe_load(text_slot)[0]["opening_book"] == "None", (
        f"the string 'None' came back as something else: {text_slot!r}"
    )
    assert null_slot == "null", (
        f"a None delta must render as YAML null, not {null_slot!r} -- `None` is a plain "
        "STRING in YAML, which is the whole defect"
    )


def test_the_header_renders_every_value_the_set_parser_can_produce() -> None:
    """Every value the `--set` parser can produce renders one-line and round-trips.
    Killer: `str()`, measured to fail 17 of these 26 rows."""
    for doc in _DOMAIN_DOCS:
        value = yaml.safe_load(doc)
        rendered = mint_config._render_value(value, where=doc)
        assert "\n" not in rendered and mint_config.HEADER_SEP not in rendered, doc
        assert mint_config._identical(yaml.safe_load(rendered), value), (
            f"{doc}: rendered {rendered!r} does not round-trip"
        )


def test_the_domain_table_covers_every_safe_loader_tag() -> None:
    """The domain table spans every `SafeConstructor` tag, so the totality row is not vacuous
    — a new PyYAML tag, or a table trimmed to the passing rows, reds here."""
    from yaml.constructor import SafeConstructor

    tags = {tag.rsplit(":", 1)[1] for tag in SafeConstructor.yaml_constructors if tag}
    covered = {type(yaml.safe_load(doc)).__name__
               for doc in _DOMAIN_DOCS + _UNRENDERABLE_DOCS}
    assert tags == {"null", "bool", "int", "float", "binary", "timestamp", "str", "seq", "map",
                    "set", "omap", "pairs"}, f"the SafeLoader domain moved: {sorted(tags)}"
    assert {"NoneType", "bool", "int", "float", "bytes", "date", "datetime", "str", "list",
            "dict", "set"} <= covered, f"the domain table lost a shape: {sorted(covered)}"


@pytest.mark.parametrize("doc", _UNRENDERABLE_DOCS)
def test_a_value_the_header_cannot_record_refuses_the_mint_loudly(doc: str,
                                                                  tmp_path: Path) -> None:
    """A value the header cannot record refuses the mint at rc 2, with no file left behind.
    Killer: drop the `_identical` check, or fall back to `str()` on `HeaderRenderError`."""
    value = yaml.safe_load(doc)
    with pytest.raises(mint_config.HeaderRenderError, match="does not round-trip"):
        mint_config._render_value(value, where="probe")

    out = tmp_path / "refused.yaml"
    refused = _run_mint("--out", str(out), "--set", f"run_id={doc}")
    assert refused.returncode == 2, (refused.stdout + refused.stderr)[-2000:]
    assert "cannot stamp a replayable header" in refused.stderr, refused.stderr[-2000:]
    assert not out.exists(), "a refused mint must leave no file behind"


def test_no_committed_minted_config_carries_a_stringified_None_header() -> None:
    """No committed header carries a stringified `None`, swept through the ONE discovery
    authority — a flat glob is blind to `configs/prod/`, and both slots must be honest."""
    configs = discover_configs(REPO_ROOT / "configs")
    assert configs, "no committed configs found -- a vacuous census is not a census"
    offenders = [
        (path.name, key, slot, text)
        for path in configs
        for key, old, new in _header_deltas(path)
        for slot, text in (("old", old), ("new", new))
        if _stringified_none(yaml.safe_load(text))
    ]
    assert not offenders, (
        f"minted headers carrying a stringified None (re-mint them, never hand-edit -- R1): "
        f"{offenders}"
    )


def test_every_committed_minted_header_is_in_canonical_replayable_form() -> None:
    """Each committed slot is exactly what the tool would stamp today, which is what makes a
    re-mint a no-op on the header and therefore reviewable."""
    drifted = [
        (path.name, key, slot, text, mint_config._render_value(yaml.safe_load(text), where=key))
        for path in discover_configs(REPO_ROOT / "configs")
        for key, old, new in _header_deltas(path)
        for slot, text in (("old", old), ("new", new))
        if mint_config._render_value(yaml.safe_load(text), where=key) != text
    ]
    assert not drifted, f"non-canonical minted header slots (re-mint): {drifted}"


def test_every_committed_header_delta_agrees_with_the_config_body() -> None:
    """The value a header RECORDS is the value the config CARRIES, or the provenance is
    fiction. Only the new slot is checkable this way — the old slot belongs to the template."""
    mismatched = []
    for path in discover_configs(REPO_ROOT / "configs"):
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key, _old, new in _header_deltas(path):
            recorded, actual = yaml.safe_load(new), _at(body, key)
            if not mint_config._identical(recorded, actual):
                mismatched.append((path.name, key, recorded, actual))
    assert not mismatched, (
        f"minted headers whose recorded delta is not the body's value: {mismatched}"
    )


def test_the_frozen_wpmain_baseline_set_is_gone_and_stays_gone() -> None:
    """The `b482243` baseline set is deleted and stays deleted. Killer: re-add the directory
    without re-arming the retired remint instrument, leaving a snapshot nothing reads."""
    assert not BASELINE.exists(), (
        f"{BASELINE} is back. It is a byte-frozen snapshot whose only reader was the retired "
        "remint instrument; a baseline nobody diffs is a golden that cannot go stale loudly. "
        "Re-arm the instrument in the same commit or delete the directory again."
    )

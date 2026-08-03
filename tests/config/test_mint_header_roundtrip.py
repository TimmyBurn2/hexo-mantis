# >300 justify (R8), stated at this file's MEASURED size of 330 lines. The two halves are one
# claim — "a minted config's header is replayable by its own minter" — and they cannot be split:
# the behavioural half establishes what the renderer does and the census half asserts the
# committed configs are in exactly that form, so a split leaves either half asserting a format
# no test defines. Both share `_header_deltas` / `_stringified_none` / the domain table, and R5
# bars cross-test imports, so splitting means duplicating the parser that IS the subject.
"""R187 — the minted header is REPLAYABLE: every delta value round-trips through the tool.

`tools/mint_config.py` stamped its `# delta:` lines with Python `str()`. That is neither total
nor injective over the header's value domain, and the domain is not a guess: both slots come
from `yaml.safe_load` (`mint_config.py:60` loads the template, `:74` parses the `--set` value),
so the domain is exactly the image of PyYAML's `SafeLoader`. Over that image `str()` fails on
`None` -> `None` (reads back as the STRING `"None"`), `inf`/`nan`, `set`, `bytes`, tuples from
`!!omap`/`!!pairs`, and any string YAML would retype (`yes`, `0123`, `null`, `''`, `a: b`) --
and it maps `None` and `"None"` onto the SAME text, so even a correct-looking header cannot be
read back unambiguously.

The measured consequence: `configs/smoke_preflight_armed.yaml`'s `eval.ladder.rungs` delta
carries `opponent_sims: None` inside a list of dicts, and replaying that header through the
tool that wrote it raises `Input should be a valid integer [input_value='None']`. **A minted
config's own provenance was not replayable by its own minter** -- and R1's "configs are minted,
never hand-varied" rests on precisely that replayability, so the defect sits under the rule.

This file is the producer test the surface never had (LAW-07), in two halves:

- **behavioural** — mint through the real CLI and read the header back, including the `None`
  case, the injectivity case, and the loud-refusal case (a value the header cannot record
  refuses the mint at rc 2 rather than writing an approximation; a serializer that swallows an
  un-encodable value is the same class of defect).
- **census** — every committed minted config's header, checked BOTH textually (is the slot the
  canonical rendering?) and semantically (does the slot's value equal the config body's value
  at that dotted path?). The second is the property that actually matters and the first is what
  keeps it cheap to see.

The `tests/fixtures/wpmain/config_baseline_b482243/` baselines are swept too, and they are the
one place a stringified-`None` still lives: they are a byte-frozen snapshot of `b482243` whose
sha256s sit in the FROZEN `tests/fixtures/manifest.toml`, so they cannot be re-minted -- a
historical record is not a defect to fix, it is the record of one. They are pinned as a closed,
named set instead of skipped, so a NEW bad header appearing there still reds.
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

#: The mint tool loaded as a module, so the census can call the SAME renderer the tool stamps
#: with. Importing it any other way would mean transcribing the format into the test, and a
#: transcribed format is a second authority (R1).
_SPEC = importlib.util.spec_from_file_location("_mint_config_under_test", MINT)
assert _SPEC is not None and _SPEC.loader is not None
mint_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mint_config)

#: The header's value domain, by its generator: one source document per `SafeConstructor` tag
#: plus the strings YAML would retype. Established from `yaml.constructor.SafeConstructor.
#: yaml_constructors` (null, bool, int, float, binary, timestamp, str, seq, map, set, omap,
#: pairs), not from imagination -- `test_the_domain_table_covers_every_safe_loader_tag` holds
#: it to that.
_DOMAIN_DOCS = (
    "null", "~", "true", "false", "42", "0x1f", "1.5", ".inf", "-.inf", ".nan",
    "!!binary 'AAEC'", "2026-08-03", "2026-08-03 11:22:33", "!!set {a, b}",
    "'None'", "'null'", "'yes'", "'0123'", "'a: b'", "''", "'#hash'", "'- dash'", "'%pct'",
    "[1, 2, 3]", "{a: 1, b: null}", "[{name: x, opponent_sims: null, deploy_matched: true}]",
)

#: The two `SafeLoader` tags `safe_dump` genuinely cannot invert: both construct a list of
#: TUPLES and dump as a list of lists. The tool must refuse these, loudly.
_UNRENDERABLE_DOCS = ("!!omap [{a: 1}]", "!!pairs [{a: 1}]")

#: The one place a stringified-`None` header still lives, closed and named: a byte-frozen
#: snapshot of `b482243` under a FROZEN manifest, which is a record, not a mintable config.
_BASELINE_KNOWN_BAD = {
    ("run5.yaml", "train.draw_rate_abort"),
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


# ── behavioural half ────────────────────────────────────────────────────────────────────────
def test_a_None_bearing_delta_mints_a_header_that_replays_green(tmp_path: Path) -> None:
    """The headline row: mint -> load -> validate green, with `None` inside the delta value.

    `eval.ladder.rungs` is the real shape (`configs/smoke_preflight_armed.yaml:27`): a list of
    dicts with `opponent_sims: null`. The replay is the whole point -- the header slot is fed
    straight back to `--set` and the second mint must produce the SAME BYTES, which is what
    "the provenance is replayable" means operationally.

    MUTATION THAT REDS IT: `str()` in the delta line. The replayed `--set` then carries
    `opponent_sims: 'None'` and the second mint exits 2 on `Input should be a valid integer`.
    """
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
    """Injectivity, the sharper half of the defect: `str()` maps `None` and `"None"` onto the
    same six characters, so the header cannot say which was minted even when nothing else is
    wrong. Two mints, two different values, two different headers.

    MUTATION THAT REDS IT: any renderer that emits a bare `None` for either value.
    """
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
    """Totality over the declared domain, checked at the renderer rather than asserted in
    prose: load each source document with the SAME parser `--set` uses, render it, and require
    the rendering to be one line, separator-free and structurally identical on the way back.

    MUTATION THAT REDS IT: `str()` — measured, it fails 17 of these 26 rows — or a renderer that
    drops the round-trip verification and starts emitting approximations.
    """
    for doc in _DOMAIN_DOCS:
        value = yaml.safe_load(doc)
        rendered = mint_config._render_value(value, where=doc)
        assert "\n" not in rendered and mint_config.HEADER_SEP not in rendered, doc
        assert mint_config._identical(yaml.safe_load(rendered), value), (
            f"{doc}: rendered {rendered!r} does not round-trip"
        )


def test_the_domain_table_covers_every_safe_loader_tag() -> None:
    """Anti-vacuity: the totality row above is only meaningful if its table spans the domain,
    and the domain's generators are enumerable -- `SafeConstructor.yaml_constructors`. A new
    PyYAML tag (or a table quietly trimmed to the passing rows) reds this."""
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
    """No silent fallback. `!!omap`/`!!pairs` load as lists of tuples and `safe_dump` writes
    them back as lists of lists, so the round-trip check fails -- and the tool must then exit
    2 naming the value, not stamp the lossy text and carry on.

    MUTATION THAT REDS IT: dropping the `_identical` check, or catching `HeaderRenderError` and
    falling back to `str()`. Either turns an unrecordable value into a lying header, which is
    the defect this file exists to close.
    """
    value = yaml.safe_load(doc)
    with pytest.raises(mint_config.HeaderRenderError, match="does not round-trip"):
        mint_config._render_value(value, where="probe")

    out = tmp_path / "refused.yaml"
    refused = _run_mint("--out", str(out), "--set", f"run_id={doc}")
    assert refused.returncode == 2, (refused.stdout + refused.stderr)[-2000:]
    assert "cannot stamp a replayable header" in refused.stderr, refused.stderr[-2000:]
    assert not out.exists(), "a refused mint must leave no file behind"


# ── census half ─────────────────────────────────────────────────────────────────────────────
def test_no_committed_minted_config_carries_a_stringified_None_header() -> None:
    """R187's census over the LIVE set, through the ONE discovery authority (the precedent is
    `test_config_diff_from_header.py:121` -- a flat glob is blind to `configs/prod/`).

    Both slots are swept, not just the replayed one: a reader cannot tell which side of a
    delta line is load-bearing, so both must be honest.
    """
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
    """The textual half: each slot must be exactly what the tool would stamp today. This is
    what makes a re-mint a no-op on the header and therefore reviewable.

    MUTATION THAT REDS IT: a hand-edited header line (R1's "never hand-varied" -- the header is
    part of the file), or the renderer changing without the configs being re-minted.
    """
    drifted = [
        (path.name, key, slot, text, mint_config._render_value(yaml.safe_load(text), where=key))
        for path in discover_configs(REPO_ROOT / "configs")
        for key, old, new in _header_deltas(path)
        for slot, text in (("old", old), ("new", new))
        if mint_config._render_value(yaml.safe_load(text), where=key) != text
    ]
    assert not drifted, f"non-canonical minted header slots (re-mint): {drifted}"


def test_every_committed_header_delta_agrees_with_the_config_body() -> None:
    """The semantic half, and the one that would have caught R187's defect at run5 and at
    smoke_preflight_armed: the value the header RECORDS must be the value the config CARRIES.

    A header slot is provenance for a body value; if the two disagree the provenance is
    fiction, whether the cause is a lossy renderer, a hand-edit, or a drifted re-mint. Only the
    new slot can be checked this way -- the old slot belongs to the template -- and that is
    exactly the slot a replay feeds back to `--set`.
    """
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


def test_the_frozen_wpmain_baselines_carry_exactly_the_known_historical_bad_headers() -> None:
    """The baselines are swept too (R187 says every config AND the fixture baselines), but they
    are a byte-frozen snapshot of `b482243` under the FROZEN `tests/fixtures/manifest.toml`:
    re-minting them would falsify a record and red the manifest oracle. So they are PINNED, not
    skipped -- the known-bad set is closed at three named deltas.

    MUTATION THAT REDS IT: a fourth bad header appearing in the baselines (someone re-cutting
    them from a broken minter), or one of the three quietly disappearing (someone hand-fixing a
    frozen record). Both are things a skip would not see.
    """
    baselines = sorted(BASELINE.glob("*.yaml"))
    assert baselines, f"no baseline configs under {BASELINE}"
    found = {
        (path.name, key)
        for path in baselines
        for key, old, new in _header_deltas(path)
        if _stringified_none(yaml.safe_load(old)) or _stringified_none(yaml.safe_load(new))
    }
    assert found == _BASELINE_KNOWN_BAD, (
        "the frozen baselines' stringified-None set is closed and historical; got "
        f"{sorted(found)} against {sorted(_BASELINE_KNOWN_BAD)}"
    )

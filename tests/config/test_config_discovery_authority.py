"""The ONE authority on "what is a config file on disk" — `mantis.config.loader`.

WPAX ADJ-13 F-1, corrective pass (recheck R-2). This file is the loader-level half; the
gate-level half (gate 7 as a process, gate 12's declaration partition, the mint and launch
routes) lives in `tests/tools/test_preflight_mint_process.py`, beside the rig that can drive
both gates.

**The class, stated at the level the three escapes actually shared.** Not "gate 12's glob is
narrower than gate 7's" — that framing produced two fixes and two escapes. The class is that
**discovery answered "is this a config?" by EXTENSION while loading answered it by NOTHING AT
ALL**: `load_config` was `yaml.load(Path(path).read_text())`, which accepts any suffix, no
suffix, or a suffix nobody has thought of yet. While that asymmetry stood, the set of files a
run could be LAUNCHED from was strictly larger than the set either gate could SEE, and every
fix that widened discovery by one more extension just moved the boundary:

    configs/run6.yaml   (MF-7)      -> fixed, and run6.yml walked through
    configs/run6.yml    (RED-TEAM)  -> fixed, and run6.txt walked through
    configs/prod/*.yaml (RED-TEAM)  -> fixed
    configs/run6.txt / .YAML (RECHECK R-2)

Enumerating extensions can never close it, because the loader accepted the complement of every
enumeration. So the loader narrows: `is_config_path` is ONE predicate, read by discovery (what
the gates enumerate) and by `load_config` (what a run can be launched from, what
`tools/mint_config.py --out` may write, and what `python -m mantis.run <path>` may consume).

Every row below is written against the BICONDITIONAL rather than against a list of suffixes,
because a list of suffixes is exactly what has been fixed twice and escaped twice.
"""
from pathlib import Path

import pytest

from mantis.config.loader import (
    CONFIG_SUFFIXES,
    ConfigSuffixError,
    discover_configs,
    is_config_path,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN5 = REPO_ROOT / "configs" / "run5.yaml"

#: The corpus is the COMPLEMENT of an enumeration, plus the enumeration, so a row cannot pass
#: by knowing the answer for `.yaml` alone. Every name is planted as a byte-for-byte copy of a
#: real, schema-valid config, so "it is not a config" is a statement about the NAME and never
#: an accident of the contents.
_NAMES = (
    "run6.yaml", "run6.yml",                     # in CONFIG_SUFFIXES
    "prod/run6.yaml", "prod/nested/run6.yml",    # in, at depth
    "run6.txt", "run6.conf", "run6.json",        # a plain unknown suffix
    "run6.YAML", "run6.YML", "run6.Yaml",        # a CASE variant of a known one
    "run6", "run6.",                             # no suffix at all
    "run6.yaml.bak", "run6.yml.orig",            # a known suffix that is not final
    ".yaml", ".yml",                             # a dotfile NAMED like a suffix
    "run6.yamlx", "run6.xyaml",                  # a known suffix as a substring
)


@pytest.fixture
def planted(tmp_path: Path) -> Path:
    """Every name in `_NAMES` on disk under one `configs/` directory, each a real config."""
    configs = tmp_path / "configs"
    body = RUN5.read_text()
    for name in _NAMES:
        target = configs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return configs


def test_the_loaders_accept_set_IS_discoverys_filter_and_not_a_second_answer(planted) -> None:
    """**The class boundary, as a biconditional over the complement of the enumeration.**

    For every name: discovery enumerates it IFF `is_config_path` accepts it IFF `load_config`
    will read it. Any one of the three drifting from the other two re-opens F-1, and the
    failure message says which pair disagreed rather than merely that a count was wrong.

    This is the row that would have gone red on the ADJ-13 fix as shipped: `configs/run6.txt`
    was rejected by discovery and ACCEPTED by the loader, which is the whole finding.
    """
    discovered = {path.relative_to(planted).as_posix() for path in discover_configs(planted)}
    for name in _NAMES:
        path = planted / name
        accepted = is_config_path(path)
        try:
            load_config(path)
        except ConfigSuffixError:
            loadable = False
        else:
            loadable = True
        assert accepted == loadable, (
            f"{name!r}: is_config_path says {accepted} and the loader says {loadable}. That "
            "asymmetry IS ADJ-13 F-1 — while it stands, the launchable set is larger than the "
            "set either gate can see, and the next unenumerated suffix walks through"
        )
        assert accepted == (name in discovered), (
            f"{name!r}: is_config_path says {accepted} but discovery says {name in discovered}. "
            "Discovery must SELECT with the predicate, never re-derive it"
        )


def test_every_suffix_in_the_authority_is_LIVE_in_both_directions(planted) -> None:
    """The inverse. A fix shaped as "refuse everything but `.yaml`" would pass every rejection
    row above while silently killing `.yml`, so each declared suffix is driven end to end."""
    discovered = {path.relative_to(planted).as_posix() for path in discover_configs(planted)}
    for suffix in CONFIG_SUFFIXES:
        name = f"run6{suffix}"
        assert is_config_path(planted / name)
        assert name in discovered, f"{suffix} is declared but not discovered; got {discovered}"
        assert load_config(planted / name).run_id == "run5", (
            f"{suffix} is declared and must LOAD — a declared-but-unreadable suffix is the "
            "same two-authorities defect pointing the other way"
        )


def test_discovery_is_RECURSIVE_and_SORTED(planted) -> None:
    """`tools/mint_config.py --out` takes a free path, so `configs/prod/run6.yaml` is a
    supported output of the repo's own minting tool. Sorted so two consumers of one tree cannot
    disagree about order — the cross-gate equality row in the process file compares lists."""
    found = [path.relative_to(planted).as_posix() for path in discover_configs(planted)]
    assert "prod/run6.yaml" in found and "prod/nested/run6.yml" in found, (
        f"a config in a subdirectory must be discovered at any depth; got {found}"
    )
    assert found == sorted(found), f"discovery must be ordered; got {found}"


def test_the_predicate_is_a_NAME_test_and_never_a_stat(tmp_path) -> None:
    """Recheck R-4, at the predicate. A config-SHAPED path that cannot be read — a directory,
    a dangling symlink — must stay INSIDE the answer set so gate 7 fails loudly on it. An
    `is_file()` conjunct here is what made gate 7 stop rejecting both shapes (HEAD rc 1 on each,
    delta rc 0, both gates silent), which is a regression the delta introduced in a green gate.
    """
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "adir.yaml").mkdir()
    (configs / "broken.yaml").symlink_to(tmp_path / "nowhere.yaml")
    (configs / "real.yaml").write_text(RUN5.read_text())

    found = {path.name for path in discover_configs(configs)}
    assert found == {"adir.yaml", "broken.yaml", "real.yaml"}, (
        "config-shaped-and-broken is a LOUD failure, not something to filter into silence; "
        f"got {found}"
    )
    for name in ("adir.yaml", "broken.yaml"):
        with pytest.raises(OSError):
            load_config(configs / name)


def test_a_refused_suffix_says_WHY_and_names_the_authority(tmp_path) -> None:
    """The refusal is an operator-facing message on the launch path (`python -m mantis.run`),
    so it must name the predicate and the one place to change it — not merely fail."""
    bad = tmp_path / "run6.txt"
    bad.write_text(RUN5.read_text())
    with pytest.raises(ConfigSuffixError) as caught:
        load_config(bad)
    message = str(caught.value)
    assert "CONFIG_SUFFIXES" in message and str(CONFIG_SUFFIXES) in message, message
    assert "run6.txt" in message, message


def test_the_authority_is_EXPORTED_from_the_package_that_claims_it(tmp_path) -> None:
    """Recheck R-12. `mantis.config.loader`'s docstring calls the module the ONE authority on
    what counts as a config; a package whose public surface does not carry it invites the next
    consumer to write a sixth glob rather than import the answer."""
    import mantis.config as package

    for name in ("CONFIG_SUFFIXES", "ConfigSuffixError", "discover_configs", "is_config_path",
                 "load_config"):
        assert name in package.__all__, f"{name} is not on mantis.config.__all__"
        assert getattr(package, name) is not None

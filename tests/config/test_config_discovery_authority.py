"""The **shared-authority invariant** — `load_config` accepts ⇒ `discover_configs` sees.

WPAX ADJ-13 F-1, re-ruled by **R75**. This file is the loader-level half; the gate-level half
(gate 7 as a process, gate 12's declaration partition, the mint and launch routes) lives in
`tests/tools/test_preflight_mint_process.py`, beside the rig that can drive both gates.

**The class, in one sentence.** A file under the audit root that the loader will READ but
discovery will not ENUMERATE is a production config nobody audits — and every name-based
discovery filter creates exactly that gap, because the loader's accept-set is defined by
CONTENT, not by name.

    configs/run6.yaml   (MF-7)      -> fixed, and run6.yml walked through
    configs/run6.yml    (RED-TEAM)  -> fixed, and configs/prod/ walked through
    configs/prod/*.yaml (RED-TEAM)  -> fixed, and run6.txt walked through
    configs/run6.txt / .YAML (RECHECK R-2)

Enumerating extensions can never close it, because the loader accepts the complement of every
enumeration. The corrective pass closed it by narrowing the LOADER; **R75 declined that** — a
run may be launched from a path of any shape. What closes it instead is the invariant, held on
the DISCOVERY side:

    load_config(p) succeeds  =>  p in discover_configs(root)      for every p under root

or, in the form the rows below drive: *a file discovery skips must be a file the loader
refuses.* R75 names the invariant as THE protection, so under LAW-07 / R4 it needs a live
producer rather than only an implementation. That producer is
`test_the_shared_authority_INVARIANT_holds_over_the_whole_corpus`, and every other row in this
file exists to keep it from passing vacuously.
"""
from pathlib import Path

import pytest

from mantis.config.loader import discover_configs, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN5 = REPO_ROOT / "configs" / "run5.yaml"

#: The corpus is the COMPLEMENT of an enumeration, plus the enumeration, so a row cannot pass by
#: knowing the answer for `.yaml` alone. Every name is planted as a byte-for-byte copy of a real,
#: schema-valid config, so each one really IS loadable — "discovery must see it" is then a claim
#: about a genuine hazard and never about a stub nothing would read. The four historical escapes
#: are in here by name, as are the three the recheck and RED-TEAM added.
_NAMES = (
    "run6.yaml", "run6.yml",                     # the original two
    "prod/run6.yaml", "prod/nested/run6.yml",    # at depth
    "run6.txt", "run6.conf", "run6.json",        # a plain unknown suffix
    "run6.YAML", "run6.YML", "run6.Yaml",        # a CASE variant of a known one
    "run6", "run6.",                             # no suffix at all
    "run6.yaml.bak", "run6.yml.orig",            # a known suffix that is not final
    ".yaml", ".yml",                             # a dotfile NAMED like a suffix
    ".hidden/run6.yaml",                         # a config under a HIDDEN directory
    "run6.yamlx", "run6.xyaml",                  # a known suffix as a substring
)


def _loadable(path: Path) -> bool:
    """Ground truth for the invariant's left-hand side: does the loader READ this path?

    Any exception is a refusal — `OSError` for a directory or a dangling link, a YAML error, a
    `TypeError` for a non-mapping root, a `ValidationError` for an incomplete config. The
    invariant is about the loader SUCCEEDING, so the catch is deliberately total; narrowing it
    to one exception type would let a new refusal mode read as an acceptance.
    """
    try:
        load_config(path)
    except Exception:  # noqa: BLE001 — total by design; see the docstring above
        return False
    return True


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


def test_the_shared_authority_INVARIANT_holds_over_the_whole_corpus(planted) -> None:
    """**THE producer for R75's protection.** For every path under the root: if `load_config`
    accepts it, discovery enumerates it.

    Quantified over the tree as it is on disk — `rglob` on the fixture, not over `_NAMES` — so a
    path the fixture creates incidentally (the `prod/`, `prod/nested/` and `.hidden/`
    directories) is inside the claim too. Driven in BOTH directions:

    * every loadable path is discovered — the invariant itself; this is the assertion that goes
      red the moment discovery grows any name filter, which is the class;
    * every path discovery SKIPS is one the loader refuses — the contrapositive, stated
      separately because it is the sentence R75 uses and because a discovery that returned the
      empty list would satisfy neither half.

    A shipped `configs/run6.txt` was measured schema-valid, `audit_arming`-DISARMED on the one
    REQUIRED row, mintable, launchable, and rc 0 from gate 7 AND gate 12. Under this invariant it
    is enumerated, so gate 12 reports it UNDECLARED instead.
    """
    discovered = {path.resolve() for path in discover_configs(planted)}
    on_disk = sorted(planted.rglob("*"))
    assert len(on_disk) > len(_NAMES), "the corpus must include the directories rglob walks"

    loadable = [path for path in on_disk if _loadable(path)]
    assert len(loadable) == len(_NAMES), (
        "every planted name is a byte-for-byte copy of a real config, so all of them must be "
        f"loadable or this row is measuring the fixture rather than the invariant; got "
        f"{len(loadable)} of {len(_NAMES)}"
    )
    for path in loadable:
        assert path.resolve() in discovered, (
            f"{path.relative_to(planted).as_posix()!r}: the loader READS it and discovery does "
            "not enumerate it. That gap IS ADJ-13 F-1 — a file a run can be launched from and "
            "no gate can see. Discovery must not filter by name; the loader decides by content"
        )

    skipped = [path for path in on_disk if path.resolve() not in discovered]
    for path in skipped:
        assert not _loadable(path), (
            f"{path.relative_to(planted).as_posix()!r} was skipped by discovery and the loader "
            "reads it — a file discovery skips must be a file the loader refuses (R75)"
        )
    assert skipped, "the corpus must exercise the skip arm, or the contrapositive is vacuous"


def test_the_ONLY_thing_discovery_skips_is_a_REAL_directory(planted) -> None:
    """The exclusion, named exactly, so a later widening of it is a visible edit.

    A real directory is the one path type the loader refuses BY TYPE — `read_text()` raises
    `IsADirectoryError` for every directory, unconditionally — and `rglob` recurses through it,
    so everything loadable beneath it is enumerated anyway. That is a proof rather than a
    heuristic, and it is the only such proof available; anything else discovery dropped would be
    dropped on a guess about names.
    """
    discovered = {path.resolve() for path in discover_configs(planted)}
    skipped = [path for path in planted.rglob("*") if path.resolve() not in discovered]
    assert {path.relative_to(planted).as_posix() for path in skipped} == {
        "prod", "prod/nested", ".hidden",
    }, f"only real directories may be skipped; got {[str(p) for p in skipped]}"
    for path in skipped:
        assert path.is_dir() and not path.is_symlink()
        with pytest.raises(IsADirectoryError):
            load_config(path)


def test_a_symlinked_DIRECTORY_is_enumerated_because_rglob_will_not_walk_it(tmp_path) -> None:
    """**The input just outside the boundary** (R71), found by walking the fix's own rule rather
    than the demonstration inputs.

    `pathlib.rglob` does not recurse THROUGH a symlink to a directory (measured: with
    `configs/link -> outside/`, `link/hidden_cfg.yaml` is absent from `rglob("*")`). So the
    obvious spelling of the exclusion — "skip directories" — would drop `configs/link` while
    never walking it, and an entire subtree of loadable, disarmed configs would be invisible to
    both gates. Exactly the class, one refactor later.

    Keeping the symlink IN the enumeration is what closes it: gate 7 hits `IsADirectoryError` on
    it and goes loud, which is the only outcome that mentions the subtree at all.
    """
    configs = tmp_path / "configs"
    configs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hidden_cfg.yaml").write_text(RUN5.read_text())
    (configs / "link").symlink_to(outside)

    assert "link/hidden_cfg.yaml" not in {
        path.relative_to(configs).as_posix() for path in configs.rglob("*")
    }, "premise of this row: rglob does not walk a symlinked directory"
    assert (configs / "link").resolve() in {path.resolve() for path in discover_configs(configs)}, (
        "a symlinked directory hides a subtree from discovery, so it must be enumerated itself "
        "and fail loudly — dropping it is a silent hole the size of that subtree"
    )
    with pytest.raises(IsADirectoryError):
        load_config(configs / "link")


def test_the_loader_accepts_a_config_at_ANY_shape(tmp_path) -> None:
    """R75, driven directly: the accept-set narrowing is OUT and the loader reads by content.

    Each name below was refused by `load_config` between `4d11147` and this pass. A run may be
    launched from a path of any shape again; what makes that safe is the invariant above, not a
    suffix test. The `ConfigSuffixError` symbol is asserted GONE rather than merely unused —
    a constant with no live consumer is R1 / LAW-08's shape, and a dead exception class invites
    the next reader to re-arm the refusal it named.
    """
    import mantis.config as package
    import mantis.config.loader as loader

    for name in ("run6.txt", "run6.YAML", "run6", "run6.yaml.bak", "run6.yamlx", ".yaml"):
        path = tmp_path / name
        path.write_text(RUN5.read_text())
        assert load_config(path).run_id == "run5", f"{name} must load — R75 declined the refusal"

    for dead in ("CONFIG_SUFFIXES", "ConfigSuffixError", "is_config_path"):
        assert not hasattr(loader, dead), (
            f"{dead} lost its last live consumer when R75 removed the refusal and the mint "
            "guard; a constant nothing reads is exactly what R1 / LAW-08 forbid"
        )
        assert dead not in package.__all__, f"{dead} is still on mantis.config.__all__"


def test_discovery_is_RECURSIVE_and_SORTED_and_does_not_skip_dotfiles(planted) -> None:
    """`tools/mint_config.py --out` takes a free path, so `configs/prod/run6.yaml` is a supported
    output of the repo's own minting tool. Sorted so two consumers of one tree cannot disagree
    about order — the cross-gate equality row in the process file compares lists. Dotfiles are
    called out because `glob.glob` DOES skip them while `pathlib.rglob` does not, and
    `configs/.yaml` was one of the seven escapes: a discovery built on the other module would
    have re-opened the class silently.
    """
    found = [path.relative_to(planted).as_posix() for path in discover_configs(planted)]
    assert "prod/run6.yaml" in found and "prod/nested/run6.yml" in found, (
        f"a config in a subdirectory must be discovered at any depth; got {found}"
    )
    assert ".yaml" in found and ".hidden/run6.yaml" in found, (
        f"a dotfile config and a config under a hidden directory are both loadable; got {found}"
    )
    assert found == sorted(found), f"discovery must be ordered; got {found}"


def test_a_config_SHAPED_but_BROKEN_path_stays_INSIDE_the_answer_set(tmp_path) -> None:
    """Recheck R-4, at the loader. A dangling symlink must stay enumerated so gate 7 fails
    loudly on it: it is a broken FILE reference, the loader's refusal of it is a TOCTOU accident
    of the target's absence rather than a property of its type, and filtering it into silence is
    how `configs/` acquires residents no gate ever mentions.
    """
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "broken.yaml").symlink_to(tmp_path / "nowhere.yaml")
    (configs / "broken.txt").symlink_to(tmp_path / "nowhere.txt")
    (configs / "real.yaml").write_text(RUN5.read_text())

    found = {path.name for path in discover_configs(configs)}
    assert found == {"broken.yaml", "broken.txt", "real.yaml"}, (
        f"a dangling symlink is a LOUD failure, not something to filter into silence; got {found}"
    )
    for name in ("broken.yaml", "broken.txt"):
        with pytest.raises(FileNotFoundError):
            load_config(configs / name)


def test_the_invariant_holds_on_the_REAL_configs_tree() -> None:
    """The rows above all run on planted trees; this one runs on the tree that ships, so the
    invariant is a statement about `configs/` and not only about `tmp_path`."""
    configs = REPO_ROOT / "configs"
    discovered = {path.resolve() for path in discover_configs(configs)}
    assert discovered, "gate 7 must never be vacuous"
    for path in configs.rglob("*"):
        if _loadable(path):
            assert path.resolve() in discovered, (
                f"{path} is loadable and undiscovered on the REAL tree — a launchable config no "
                "gate audits"
            )


def test_the_authority_is_EXPORTED_from_the_package_that_claims_it() -> None:
    """Recheck R-12. `mantis.config.loader`'s docstring calls the module the ONE enumeration; a
    package whose public surface does not carry it invites the next consumer to write a sixth
    glob rather than import the answer."""
    import mantis.config as package

    for name in ("discover_configs", "load_config"):
        assert name in package.__all__, f"{name} is not on mantis.config.__all__"
        assert getattr(package, name) is not None

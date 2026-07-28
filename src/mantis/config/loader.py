"""Config loader: yaml.safe_load (duplicate-key-rejecting) -> schema validation. No merge,
no defaults, no env expansion. Missing key and unknown key both raise pydantic.ValidationError
(loud, listing every error). A duplicate YAML key (frozen loader silently last-won) is a
HARD error here (judgment #8).

This module is also the ONE authority on **what counts as a config file on disk**
(`CONFIG_SUFFIXES` + `is_config_path` + `discover_configs`) — R71 / ADJ-13 F-1, corrected by
the post-recheck corrective pass (R-2).

**The class, stated at the level the three escapes actually shared.** It is not "gate 12's glob
is narrower than gate 7's". It is: **discovery answered "is this a config?" by EXTENSION while
loading answered it by NOTHING AT ALL** — `load_config` was `yaml.load(Path(path).read_text())`,
which accepts any suffix, no suffix, or a suffix nobody has thought of yet. While that asymmetry
stood, the set of files a run could be LAUNCHED from was strictly larger than the set either
gate could SEE, and every fix that widened discovery by one more extension just moved the
boundary: `configs/run6.yaml` (MF-7) → `configs/run6.yml` + `configs/prod/` (RED-TEAM) →
`configs/run6.txt` + `configs/run6.YAML` (RECHECK R-2). Enumerating extensions can never close
it, because the loader accepts the complement of every enumeration.

So the loader NARROWS (recheck shape (b)). `is_config_path` is the ONE predicate, and it governs
BOTH directions:

* `discover_configs` selects with it — what the gates look at;
* `load_config` REFUSES anything it rejects (`ConfigSuffixError`) — what a run can be launched
  from, `tools/mint_config.py --out` can write, and `python -m mantis.run <path>` can consume.

A file under `configs/` whose suffix is not in `CONFIG_SUFFIXES` is therefore not a config
ANYWHERE, which is what makes a gate's silence about it correct rather than a hole. Measured
before this pass: `configs/run6.txt` was schema-valid, `audit_arming`-DISARMED on the required
`actor_lag` row, loadable, launchable, and BOTH gates returned rc 0.

SF-4 is preserved: `discover_configs` takes the directory as an ARGUMENT and resolves no repo
root. `REPO_ROOT` stays in the tool (`preflight_mint.py:104`), where `parents[2]` is structurally
sound rather than dependent on an editable install.

**What this does NOT close, stated so the next reader does not read more into it than was
built:** the loader answers "what is a config", not "where may a config live". A `.yaml` file
outside `configs/` — `<repo>/scratch/run6.yaml`, `/tmp/run6.yaml` — is still loadable and still
launchable, and `discover_configs(configs/)` will never see it. That is deliberate (an operator
preflighting a candidate from a scratch directory is the normal case, and every `--config`
route depends on it) and it is carded as `CARD-CONFIG-DISCOVERY-ROOT` rather than taken here.
"""
from pathlib import Path

import yaml

from mantis.config.schema import RunConfig

#: The extensions a config file may carry. DATA, and the reason it is a tuple rather than a
#: glob or three: it is read by ONE predicate (`is_config_path`) that both discovery and the
#: loader consume, so widening it widens what the gates SEE and what a run can be LAUNCHED from
#: **together**. Those two moving apart is the whole of ADJ-13 F-1.
CONFIG_SUFFIXES: tuple[str, ...] = (".yaml", ".yml")


class DuplicateKeyError(ValueError):
    """A YAML mapping declared the same key twice (silent last-wins is banned)."""


class ConfigSuffixError(ValueError):
    """A path whose suffix is not in `CONFIG_SUFFIXES` was handed to the loader.

    Loud rather than best-effort, and that direction is the fix (recheck R-2): a loader that
    reads `configs/run6.txt` makes that file launchable while leaving it invisible to every
    gate, which is a disarmed production config nobody audits. A file the gates do not count
    is a file the loader does not read.
    """


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that HARD-ERRORS on a duplicate key at any nesting depth.

    The frozen utils/config.py merely LOGGED a warning and kept the last value; a duplicate
    key is a copy-paste hazard, so it becomes a load-time error. Safe construction is preserved
    (SafeLoader subclass — no arbitrary object construction).
    """

    def construct_mapping(self, node, deep=False):
        seen: set = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise DuplicateKeyError(
                    f"duplicate key {key!r} at {key_node.start_mark}"
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def is_config_path(path: str | Path) -> bool:
    """THE predicate — "is this path a config file?" — with exactly one implementation.

    Read by `discover_configs` (what the gates enumerate) and by `load_config` (what a run can
    be launched from). One predicate is the entire point: two, however carefully kept in step,
    is the R1 / LAW-08 two-authorities shape, and the gap between them is where every one of
    F-1's three escapes lived.

    Deliberately a NAME test, not a `stat`: it must answer the same way for a path that is a
    directory or a broken symlink. A `configs/adir.yaml` directory and a dangling
    `configs/broken.yaml` are CONFIG-SHAPED and BROKEN, which is a loud gate-7 failure
    (`OSError` out of `read_text`) — not something to filter into silence. Adding an
    `is_file()` conjunct here is what made gate 7 stop rejecting both shapes (recheck R-4).
    """
    return Path(path).suffix in CONFIG_SUFFIXES


def discover_configs(configs_dir: str | Path) -> list[Path]:
    """Every config file under `configs_dir`, RECURSIVELY, in sorted order.

    The one authority both gate 7 and gate 12 consume (R71). Recursive because
    `tools/mint_config.py --out` takes a free path, so `configs/prod/run6.yaml` is a supported
    output of the repo's own minting tool, not a contrived input; sorted so two consumers of
    the same tree cannot disagree about order.
    """
    return sorted(path for path in Path(configs_dir).rglob("*") if is_config_path(path))


def load_config(path: str | Path) -> RunConfig:
    """Load and schema-validate one complete config file (duplicate keys rejected).

    Refuses a path `is_config_path` rejects (recheck R-2). This is a PRODUCTION behaviour
    change and it is the fix: while the loader was extension-agnostic, `configs/run6.txt` was
    schema-valid, DISARMED on the manifest's one required row, mintable via
    `tools/mint_config.py --out`, launchable via `python -m mantis.run <path>`, and invisible
    to gate 7 and gate 12 alike. Now discovery's filter and the loader's accept-set are the
    same call, so "not discovered" and "not loadable" cannot come apart.
    """
    if not is_config_path(path):
        raise ConfigSuffixError(
            f"{path}: not a config file — its suffix is not in CONFIG_SUFFIXES "
            f"{CONFIG_SUFFIXES}. A file the gates do not enumerate is a file this loader does "
            "not read: reading it would make it launchable while leaving it unaudited by "
            "gate 7 and gate 12 (ADJ-13 F-1). Rename it, or widen CONFIG_SUFFIXES — which "
            "widens discovery and the loader together, which is the point."
        )
    raw = yaml.load(Path(path).read_text(), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: config root must be a mapping")
    return RunConfig.model_validate(raw)

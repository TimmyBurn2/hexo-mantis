# >300 justify (R8): the selector's closed registry, the incumbent pin that keeps production
# on the arch it has always built, and the end-to-end round trip that proves a NON-incumbent
# arch is reachable are one unit — the round trip is only evidence while the incumbent pin says
# nothing quietly changed underneath it.
"""T10 — CANDIDATE D: the arch selector, and what it does and does not close (R322(d)).

B1 landed `GnnNetV2` behind the model contract and disclosed one limitation in its own exit:
**V2 is not selectable from a minted config.** R322(d) orders the selector, "suite-proven via a
throwaway diagnostic config", with "every shipped production config still selecting its current
arch, byte-unchanged".

WHAT LANDED, AND IT IS THE MECHANISM IN FULL:

  * `ARCH_KINDS` — the ONE arch-kind vocabulary, keyed by the declared dataclass's own name.
    It used to be a private table inside `mantis.train.checkpoints`, authored at B1 because the
    LOADER is where the need first bit; a kind vocabulary is a model fact, so it moved to
    `mantis.model.arch` and the loader now imports it. Two copies of a discriminator is the
    duplicate-authority class that file's own docstring warns about.
  * `ARCH_KINDS_BY_REPRESENTATION` — the pairing rule (`SEAM_V1_DESIGN` §2.1's missing half,
    stated for the arch side). `graph` admits TWO kinds now, which is exactly why a selector
    has to exist at all.
  * `select_arch(spec, config, *, arch_kind)` — keyword-only, NO default. A caller that does
    not know which kind it wants is not entitled to one.
  * `INCUMBENT_ARCH_KIND` — a statement about HISTORY, not a default, and pinned below against
    the real minted configs: what this tree has always built, per representation.

THE CONFIG ROW LANDED AT R330(e), AND IT IS EMPTY IN EVERY SHIPPED CONFIG. B2 reported that a
selector key would be REQUIRED under R1 and would put a minted row into the two production
configs, which R322(d) makes a HALT; R323(b) then ruled the key "enters production configs only
as a minted row at run6's mint". R330(e) lands the plumbing under that ruling: `identity.arch_kind`
is an OPTIONAL schema leaf (the one optional identity leaf, enumerated as such in
`tests/config/test_schema.py`), `arch_from_spec_and_config` honours a present row and resolves an
absent one to the incumbent, and the rows below pin BOTH halves — no shipped config carries the
row yet, and a config that does carry it builds what it names. Config-less call sites (a
checkpoint's legacy read, `strip_and_restamp`, the pretrain validator) never reach this table:
their authority is the artifact's stamp, `mantis.train.checkpoints.stamped_arch_kind`.

WHAT "THROWAWAY DIAGNOSTIC CONFIG" MEANS BELOW, stated precisely so the round trip is not read
as more than it is: the test MINTS a real config file, loads it through the ONE loader, builds
V2 through the selector against that config's own resolved encoding spec, serves a batch, and
deletes the file. Everything except the KIND comes from the minted config; the kind is the
selector's explicit argument, because no key can carry it yet. That is the honest boundary of
what candidate D closes at B2.
"""
from __future__ import annotations

import hashlib
import re

import pytest
import torch
import yaml

from mantis.config.loader import load_config
from mantis.encoding import lookup
from mantis.model import (
    ARCH_KIND_ROW,
    ARCH_KINDS,
    ARCH_KINDS_BY_REPRESENTATION,
    INCUMBENT_ARCH_KIND,
    GnnArch,
    GnnArchV2,
    UnknownArchKind,
    arch_from_spec_and_config,
    build_net,
    net_param_hash,
    select_arch,
)

from _corpus import ConformanceRefusal

from test_config_partition_shared_vs_arch_scoped import CONFIGS

#: The seed the diagnostic build runs under. An instrument parameter: it fixes WHICH random net
#: is hashed, and no claim below depends on its value.
_SEED = 20260830

#: Widths for the diagnostic build — small enough for the default tier. The config does not
#: carry them (no arch width key is a live `RunConfig` leaf, which is its own finding), so the
#: dataclass defaults would otherwise apply and make this a slow test for no gain.
_WIDTHS = {"hidden": 8, "num_layers": 2, "policy_hidden": 8, "value_hidden": 8}


class SelectorWentVacuous(ConformanceRefusal):
    """The selector's subject is missing, so a green result would mean nothing."""


def _config_source(representation: str) -> tuple[dict, str]:
    """A shipped config's raw YAML for `representation`, as a diagnostic config's base.

    Copied from a real minted file rather than authored here: a hand-built config would drift
    from the schema the moment a key is added, and the point of the round trip is that the
    diagnostic config is a config the REAL loader accepts.
    """
    for path in sorted(CONFIGS.glob("*.yaml")):
        if load_config(path).identity.representation == representation:
            return yaml.safe_load(path.read_text(encoding="utf-8")), path.name
    raise SelectorWentVacuous(
        f"no shipped config selects {representation}, so the diagnostic config has no base and "
        "the round trip below would prove nothing"
    )


def _graph_config_source() -> tuple[dict, str]:
    return _config_source("graph")


@pytest.fixture
def diagnostic_config(tmp_path):
    """A THROWAWAY minted graph config: written, loaded through the one loader, then deleted.

    It lives under `tmp_path`, never under `configs/` — a file there would be discovered by
    `discover_configs`, which gates 7 and 12 both consume, so a diagnostic config parked in the
    audit root is a config nobody declared. Deleted at the end of the test either way, and the
    deletion is asserted, because "throwaway" is a property of the file's lifetime and not of
    its name.
    """
    raw, base = _graph_config_source()
    raw["run_id"] = "seam-b2-arch-selector-diagnostic"
    path = tmp_path / "arch_selector_diagnostic.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    yield path, base
    path.unlink()
    assert not path.exists()


# ── the vocabulary and the pairing rule ──────────────────────────────────────────────────
def test_the_kind_vocabulary_is_set_equal_to_build_nets_dispatch(derived):
    """The registry and the dispatch are one claim, checked as SET EQUALITY in both directions.

    A kind in the registry that `build_net` cannot build is a name that resolves to nothing; an
    arch `build_net` dispatches on that the registry does not name is an arch no selector, no
    stamp and no manifest can reach. The suite's T7 section already derives the dispatch census
    from `build_net` itself, and this row consumes the same source rather than a second list.
    """
    import inspect

    from mantis.model import build as build_module

    source = inspect.getsource(build_module.build_net)
    dispatched = {name for name in ARCH_KINDS if f"isinstance(arch, {name})" in source}
    derived("t10.arch_kinds", sorted(ARCH_KINDS))
    derived("t10.dispatched", sorted(dispatched))
    assert dispatched == set(ARCH_KINDS), (
        f"`build_net` dispatches on {sorted(dispatched)} and ARCH_KINDS names "
        f"{sorted(ARCH_KINDS)}; a kind with no branch resolves to nothing and a branch with no "
        "kind is unreachable from the selector, the stamp and every manifest"
    )


def test_the_pairing_rule_partitions_the_kinds_across_the_representations(derived):
    """Every kind is admitted by exactly one representation, and every representation admits at
    least one. A kind admitted by none is unreachable; a kind admitted by two would make
    `INCUMBENT_ARCH_KIND` ambiguous in a way no test below would see."""
    admitted = [k for kinds in ARCH_KINDS_BY_REPRESENTATION.values() for k in kinds]
    derived("t10.pairing", {r: list(k) for r, k in ARCH_KINDS_BY_REPRESENTATION.items()})
    assert sorted(admitted) == sorted(ARCH_KINDS), (
        "the pairing rule and the kind vocabulary disagree about which kinds exist"
    )
    assert len(admitted) == len(set(admitted)), "a kind is admitted by two representations"
    for representation, kinds in ARCH_KINDS_BY_REPRESENTATION.items():
        assert kinds, f"{representation} admits no arch kind at all"


def test_the_selector_has_a_representation_with_a_REAL_choice(derived):
    """The vacuity guard this whole section needs. If every representation admitted exactly one
    kind, a selector would be indistinguishable from a lookup and every row below would pass
    while proving nothing. `graph` admits two BECAUSE `GnnNetV2` landed."""
    choices = {r: len(k) for r, k in ARCH_KINDS_BY_REPRESENTATION.items()}
    derived("t10.choices_per_representation", choices)
    assert max(choices.values()) >= 2, (
        "no representation admits more than one arch kind, so nothing in this file is "
        "exercising a CHOICE — the selector is a lookup and candidate D has no subject"
    )


# ── the incumbent: history, pinned against the real minted files ─────────────────────────
def test_every_shipped_config_still_selects_the_arch_it_has_always_selected(derived):
    """R322(d)'s "every shipped production config still selects its current arch".

    EXECUTED against every minted file, not argued: each config is loaded through the one
    loader, its encoding resolved through the registry, and the arch the production entry point
    (`arch_from_spec_and_config`) returns is asserted to be the incumbent for its
    representation. A selector that quietly re-pointed production is what this catches, and it
    is the row that makes the round trip below safe to add.
    """
    seen = 0
    for path in sorted(CONFIGS.glob("*.yaml")):
        config = load_config(path)
        spec = lookup(config.identity.encoding)
        arch = arch_from_spec_and_config(spec, config.model_dump())
        expected = ARCH_KINDS[INCUMBENT_ARCH_KIND[config.identity.representation]]
        derived(f"t10.incumbent.{path.name}", type(arch).__name__)
        assert type(arch) is expected, (
            f"{path.name} (representation={config.identity.representation}) now builds "
            f"{type(arch).__name__}; the incumbent is {expected.__name__} and production must "
            "not move without a ruling"
        )
        seen += 1
    assert seen, "no shipped config was checked, so this pin asserts nothing"


def test_the_incumbent_is_a_KIND_the_pairing_rule_admits():
    """The incumbent is history, but it still has to be reachable: an incumbent the pairing rule
    does not admit would make `arch_from_spec_and_config` raise on every production config."""
    assert set(INCUMBENT_ARCH_KIND) == set(ARCH_KINDS_BY_REPRESENTATION)
    for representation, kind in INCUMBENT_ARCH_KIND.items():
        assert kind in ARCH_KINDS_BY_REPRESENTATION[representation]


# ── the selector's refusals ──────────────────────────────────────────────────────────────
def test_an_UNKNOWN_kind_is_REFUSED_and_not_resolved_to_the_nearest_fit():
    """PB-T10a. The class LAW-11 exists for, transposed from encodings to arches: a name this
    build does not have is an error, never the closest member of the union."""
    spec = lookup("gnn_axis_v1")
    with pytest.raises(UnknownArchKind, match="not a known model kind"):
        select_arch(spec, {}, arch_kind="GnnArchV3")


def test_a_kind_the_REPRESENTATION_does_not_admit_is_REFUSED():
    """PB-T10b. The pairing rule with teeth: a grid arch on a graph encoding builds nothing."""
    spec = lookup("gnn_axis_v1")
    with pytest.raises(UnknownArchKind, match="not admitted by representation"):
        select_arch(spec, {}, arch_kind="CnnArch")


def test_select_arch_takes_NO_default_kind():
    """The absence of a default is the property that keeps `select_arch` from becoming a second
    answer to "what does production build". Read off the signature, not asserted in prose."""
    import inspect

    parameter = inspect.signature(select_arch).parameters["arch_kind"]
    assert parameter.default is inspect.Parameter.empty, (
        "`select_arch` grew a default arch kind; the incumbent is `arch_from_spec_and_config`'s "
        "job and a second defaulting surface is a second authority"
    )
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


# ── the round trip: a minted config, V2, a served batch, a stable hash ───────────────────
def _batch(in_dim: int, edge_dim: int, n_real: int = 12, n_stones: int = 3) -> dict:
    """A synthetic star graph on the `gnn_axis_v1` wire's dummy topology.

    SYNTHETIC AND LABELLED AS SUCH (T6's rule): it reproduces the wire's shape — one dummy node
    bidirectionally connected to every real node, all-zero edge attrs — so `forward_batch` has
    something legal to serve. It is not a position and no reachability is implied by it; the
    claim under test is that the SELECTED net serves the wire at all, not what it says.
    """
    dummy = n_real
    n = n_real + 1
    src = torch.cat([torch.arange(n_real), torch.full((n_real,), dummy)])
    dst = torch.cat([torch.full((n_real,), dummy), torch.arange(n_real)])
    edge_index = torch.stack([src, dst])
    stone_mask = torch.zeros(n, dtype=torch.bool)
    stone_mask[:n_stones] = True
    legal_mask = torch.zeros(n, dtype=torch.bool)
    legal_mask[n_stones:n_real] = True
    torch.manual_seed(_SEED)
    return {
        "x": torch.randn(n, in_dim),
        "edge_index": edge_index,
        "edge_attr": torch.zeros(edge_index.shape[1], edge_dim),
        "stone_mask": stone_mask,
        "legal_index": legal_mask.nonzero(as_tuple=True)[0],
    }


def _serve(net, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        policy, value, _bins = net.forward_batch(
            batch["x"], batch["edge_index"], batch["edge_attr"],
            batch["legal_index"], batch["stone_mask"],
        )
    return policy, value


def _digest(*tensors: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    for tensor in tensors:
        hasher.update(tensor.detach().to(torch.float64).contiguous().numpy().tobytes())
    return hasher.hexdigest()


@pytest.mark.parametrize("arch_kind", ARCH_KINDS_BY_REPRESENTATION["graph"])
def test_a_minted_config_round_trips_through_the_selector_to_a_SERVED_batch(
    arch_kind, diagnostic_config, derived
):
    """CANDIDATE D's proof: select → build → serve → hash, over a real minted config.

    Parametrized over BOTH graph kinds rather than V2 alone, and that is the row's point: the
    two runs differ in exactly one argument, so "V2 is reachable" is demonstrated against V1 as
    its own control rather than asserted on its own.
    """
    path, base = diagnostic_config
    config = load_config(path)
    assert config.identity.representation == "graph"
    spec = lookup(config.identity.encoding)
    torch.manual_seed(_SEED)
    arch = select_arch(spec, {**config.model_dump(), **_WIDTHS}, arch_kind=arch_kind)
    assert type(arch) is ARCH_KINDS[arch_kind]
    net = build_net(arch).eval()
    policy, value = _serve(net, _batch(arch.in_dim, arch.edge_dim))
    derived(f"t10.roundtrip.{arch_kind}.base_config", base)
    derived(f"t10.roundtrip.{arch_kind}.param_hash", net_param_hash(net))
    derived(f"t10.roundtrip.{arch_kind}.output_digest", _digest(policy, value))
    assert policy.shape[0] == 9, policy.shape          # one logit per legal node
    assert value.shape == (1, 1), value.shape          # one graph, one value


def test_the_round_trip_is_STABLE_across_two_builds_of_the_same_kind(diagnostic_config):
    """The hash is only evidence if it reproduces. Two independent builds at the same seed must
    agree bit-for-bit on both the parameter hash and the served output — otherwise the digest
    recorded above is a measurement of the RNG, not of the arch."""
    path, _base = diagnostic_config
    config = load_config(path)
    spec = lookup(config.identity.encoding)
    digests, hashes = [], []
    for _ in range(2):
        torch.manual_seed(_SEED)
        arch = select_arch(spec, {**config.model_dump(), **_WIDTHS}, arch_kind="GnnArchV2")
        net = build_net(arch).eval()
        hashes.append(net_param_hash(net))
        digests.append(_digest(*_serve(net, _batch(arch.in_dim, arch.edge_dim))))
    assert hashes[0] == hashes[1]
    assert digests[0] == digests[1]


def test_the_two_kinds_are_DIFFERENT_functions_on_the_same_minted_config(diagnostic_config):
    """The control that keeps the round trip from being satisfiable by a selector that ignores
    its argument. Same config, same seed, same batch, one argument different — and the served
    outputs must NOT agree. (`GnnNetV2`'s value readout consumes a max statistic V1 cannot see;
    the witnesses for WHY that is the right difference live in `test_gnn_v2_witnesses.py`, and
    nothing here is a strength claim in either direction — F-01 is the standing fence.)"""
    path, _base = diagnostic_config
    config = load_config(path)
    spec = lookup(config.identity.encoding)
    served = {}
    for kind in ("GnnArch", "GnnArchV2"):
        torch.manual_seed(_SEED)
        arch = select_arch(spec, {**config.model_dump(), **_WIDTHS}, arch_kind=kind)
        net = build_net(arch).eval()
        served[kind] = _digest(*_serve(net, _batch(arch.in_dim, arch.edge_dim)))
    assert served["GnnArch"] != served["GnnArchV2"], (
        "both kinds served an identical batch identically, so the selector's argument changed "
        "nothing — the round trip above would pass on a selector that ignored it"
    )


def test_the_selected_V2_arch_is_the_SIBLING_dataclass_and_not_V1(diagnostic_config):
    """The dispatch hazard B1 filed, re-checked at the SELECTOR rather than at `build_net`: a
    selector that returned `GnnArch` for `arch_kind="GnnArchV2"` would satisfy every shape
    assertion above, because the two dataclasses have identical field sets."""
    path, _base = diagnostic_config
    config = load_config(path)
    spec = lookup(config.identity.encoding)
    arch = select_arch(spec, config.model_dump(), arch_kind="GnnArchV2")
    assert isinstance(arch, GnnArchV2)
    assert not isinstance(arch, GnnArch)


# ── the row (R330(e)): it exists, it is empty everywhere shipped, and it is honoured ──────
def test_the_selector_row_is_the_ONE_config_key_naming_an_arch_and_no_shipped_config_carries_it(
    derived,
):
    """R323(b): the row enters production configs ONLY as a minted row at run6's mint. Pinned in
    both directions — the key is live in the schema (R330(e)), and every committed config omits
    it — so a row minted early or a second arch-naming key both red this test."""
    from mantis.config.schema import RunConfig

    from test_config_partition_shared_vs_arch_scoped import live_leaf_paths

    leaves = live_leaf_paths(RunConfig)
    # Word-boundaried on the LEAF SEGMENT: a bare `in` match reads the "arch" inside
    # `full_search_prob` and turns this pin into a permanent red on an unrelated key.
    naming = re.compile(r"(^|_)arch(_|$)")
    named = sorted(leaf for leaf in leaves if naming.search(leaf.split(".")[-1]))
    derived("t10.arch_naming_leaves", named)
    assert named == [ARCH_KIND_ROW], named
    carrying = sorted(
        path.name for path in CONFIGS.glob("*.yaml")
        if yaml.safe_load(path.read_text(encoding="utf-8")).get("identity", {}).get("arch_kind")
        is not None
    )
    derived("t10.configs_carrying_the_row", carrying)
    assert not carrying, (
        f"{carrying} carry identity.arch_kind before the run6 mint; R323(b) reserves the row to "
        "that mint act"
    )


@pytest.mark.parametrize("kind", sorted(ARCH_KINDS_BY_REPRESENTATION["graph"]))
def test_a_minted_arch_kind_row_is_honoured_by_the_production_entry_point(tmp_path, kind):
    """The row's reader is `arch_from_spec_and_config` — the function every config-holding
    production site calls — so a config that carries `identity.arch_kind` builds what it names,
    incumbent or not, and the mint later writes only the row."""
    raw, _base = _graph_config_source()
    raw["run_id"] = "r330e-row-honoured"
    raw["identity"]["arch_kind"] = kind
    path = tmp_path / "with_row.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(path)
    assert config.identity.arch_kind == kind
    arch = arch_from_spec_and_config(lookup(config.identity.encoding), config.model_dump())
    assert type(arch) is ARCH_KINDS[kind], type(arch).__name__


def test_a_row_naming_a_kind_the_representation_does_not_admit_is_refused_at_construction(
    tmp_path,
):
    """The schema cannot import the vocabulary (a config↔model cycle, gate 9), so the refusal
    lives in `select_arch` and fires at the first net built — before anything trains or serves.
    A GRID config naming a graph kind is refused by name, never resolved to the nearest fit."""
    raw, _base = _config_source("grid")
    raw["run_id"] = "r330e-row-refused"
    raw["identity"]["arch_kind"] = "GnnArchV2"
    path = tmp_path / "bad_row.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(path)  # schema-valid: the refusal is downstream, and this pins where
    with pytest.raises(UnknownArchKind, match="not admitted by representation='grid'"):
        arch_from_spec_and_config(lookup(config.identity.encoding), config.model_dump())
    raw["identity"]["arch_kind"] = "NoSuchArch"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(path)
    with pytest.raises(UnknownArchKind, match="not a known model kind"):
        arch_from_spec_and_config(lookup(config.identity.encoding), config.model_dump())

# >300 justify (R8). NO LINE COUNT is stated, per G-DFIX-4 and R192(e)'s derive-or-delete:
# R8 asks for a one-line justification, not a tally. (Grown from WPMAIN by R178(a)'s ruled
# deletion, R187's ruled header re-render, and WP12-R F2's block-level expression of the
# insert-only rule with its own closure guard.) The oracles are ONE claim — "a
# re-mint of the six configs changes exactly what was ruled and nothing else" — driven from ONE
# committed baseline through two instruments that must stay in the same file because each
# allowance (`_REMOVED_LINES`, `_REHEADERED_DELTAS`) is read by both the tolerance it widens and
# the closure guard that keeps it closed. Splitting them separates an allowance from its guard.
"""⊕ WPMAIN ORACLE — the re-mint is ADDITIVE, and run5's armed values survive it (O-E2).

This WP adds three items to the schema — `eval_enabled` (R120), the `monitor.disk_guard`
family (R122) and `train.device` (R126) — and re-mints all six configs by replaying each
one's header-recorded template + deltas. A replay is a REGENERATION: it can silently move any
value the template owns. `configs/run5.yaml` carries the three pre-registered armed values
`0.25 / 25000 / 50` (`train.draw_rate_abort`), and R119's HARD STOP is verbatim: **any change
to run5's armed values** is mint-prereg-only. This file is that stop's instrument.

The baseline is committed, not computed: `tests/fixtures/wpmain/config_baseline_b482243/`
holds the six configs' bytes at `b482243`, the commit this WP branched from, with manifest
rows and sha256s (`tests/fixtures/manifest.toml`). A golden that lives in git history alone
is a golden no test can read.

Two independent instruments, because they see different mutations:

- **structural** — the parsed leaf key-path -> value map. Every pre-existing leaf keeps its
  value byte-for-byte, exactly the new leaves appear, and exactly the RULED-DELETED leaf
  disappears. This sees a moved VALUE regardless of formatting.
- **textual** — the line-level diff must be INSERT-ONLY apart from that one ruled deletion.
  This sees a dropped minted-header line, a reordered block, a rewritten comment: things a
  value map is blind to, and things a regenerating minter does easily.

WHY THERE IS A DELETION AT ALL, AND WHY THE BASELINE WAS NOT RE-CUT (WP12-R, R178(a) as
assigned by R183(a)). R178(a) DELETES `train.buffer_save_interval` — a key minted into
`run5.yaml` whose only consumer chain ended in `_try_save_buffer`, which WP12-R Phase CS
(F-CS-2) measured production-dead on every leg. The re-mint that rides that deletion is the
first NON-insertion this file has ever seen, and closing it had exactly two shapes:

1. **Re-cut the baseline.** REJECTED, on three grounds and one of them is dispositive.
   (a) `tests/fixtures/manifest.toml` — which carries each baseline file's sha256 and is what
   `test_fixtures_manifest.py` checks — is a FROZEN oracle in this WP; re-cutting the baseline
   means editing it, and no grant exists. (b) The directory name `config_baseline_b482243` IS
   a commit pin; a baseline re-cut against a later tree makes its own name false. (c) A re-cut
   baseline equals the live configs, so `_ADDED_LEAVES` becomes empty and every assertion in
   this file goes vacuous — that is destroying the instrument, not maintaining it.
2. **Teach the instrument that ONE ruled deletion is legal, by name.** ADOPTED. `_REMOVED_
   LEAVES` is a closed, one-element set carrying its R-number, and both halves assert set
   EQUALITY against it rather than relaxing to "deletions are fine": a second dropped leaf, a
   dropped header line, a reformat, a reorder and a rewritten comment all still red. The
   textual half additionally pins the exact deleted TEXT and forbids `replace` outright, so a
   deletion cannot smuggle a substitution in beside it.

The instrument's reach is therefore unchanged except on one named line. Anything the
pre-R178 version caught, this version still catches — asserted directly by
`test_the_permitted_deletion_is_exactly_one_named_line`, which fails if the allowance ever
widens beyond that one leaf.

Device VALUES are deliberately NOT pinned per config. R126 routes them to mint prereg
(DEP-DEVICE-MINT-VALUES): `smoke_preflight_armed` is the one unambiguous case — its own
header already records `eval.worker_device: cuda -> cpu` and it is the CI-runnable smoke — so
cpu is transcription there and IS pinned. The other five are the operator's decision at mint,
and an oracle that pinned them would be this phase making a mint decision.

The six configs are ENUMERATED here rather than read from `tests/conftest.py:52-53`'s
`MINTED_CONFIGS`, which lists FIVE — it omits `smoke_preflight_armed.yaml`. That gap is
pre-existing and recorded, not fixed here (F-12); inheriting it would leave the config the
preflight actually boots out of the HARD-STOP sweep.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import pytest
import yaml

from mantis.config.loader import discover_configs, load_config
from mantis.config.schema import ARCH_SCOPED_KEYS

_REPO = Path(__file__).resolve().parents[2]
_LIVE = _REPO / "configs"
_BASELINE = _REPO / "tests" / "fixtures" / "wpmain" / "config_baseline_b482243"

_CONFIGS = ("dev_example.yaml", "run5.yaml", "smoke_gnn.yaml", "smoke_preflight_armed.yaml",
            "smoke_radius_curriculum.yaml", "sustained_kcluster.yaml")

#: Configs minted AFTER the b482243 baseline was cut — `(name, template)` rows, named here,
#: closed, one per row (F-P2B, R259 shakedown). This is the "its own baseline row" the
#: membership premise demands, in the only shape the frozen oracle permits:
#: `config_baseline_b482243/` and its manifest sha256s are FROZEN (§1 of this docstring's
#: re-cut rejection), the directory name IS a commit pin, and a config that did not exist at
#: b482243 has no bytes to freeze there — a "baseline" equal to the live file would make the
#: O-E2 added-leaves equality false and the diff vacuous, destroying the instrument for a
#: subject it never had. So a post-baseline mint is DECLARED here instead: the membership
#: assert enumerates it by name (an eighth FLAT `configs/*.yaml` still reds; subdirectory/
#: `.yml` shapes are gate 12's subject per ADJ-13 F-1/R75), the baseline-free value sweeps
#: below run over it, and its provenance instruments are its own minted header (pinned live
#: by `test_mint_header_roundtrip.py`) plus gate 12's by-name production audit.
#:
#: WHY THE TEMPLATE IS PART OF THE ROW (review F-P2B finding 5c): a post-baseline mint
#: escapes the six configs' structural/textual diffs, so its drift detection rides the
#: BASELINED configs that share its template — dev-template drift still reds through them.
#: A row minted from a template NO baselined config uses would have zero drift witness, so
#: the membership test asserts each declared row's own `# template:` header line matches the
#: declared template AND that some baselined config carries the same line (derived from the
#: baseline files at point of use, never transcribed).
_POST_BASELINE_MINTS = (("shakedown_20260807.yaml", "dev"),)

#: Exactly what this WP's re-mint may add — one key, one family of three leaves, one key.
_ADDED_LEAVES = {
    "eval_enabled",
    # RECAL-PREP (R308(g)(i)): the CUDA caching allocator's REGIME, minted `null` (R119's
    # placeholder) in every config because the VALUE is a measurement the re-calibration
    # sitting takes under R282(b) — never a dispatcher's act. An INSERTION in every one of the
    # seven configs and a value nowhere, so both halves of this file's instrument stay live:
    # the structural half sees exactly one added leaf, the textual half exactly one added line.
    "allocator_posture",
    "monitor.disk_guard.interval_sec",
    "monitor.disk_guard.warn_gb",
    "monitor.disk_guard.fail_gb",
    "train.device",
    # WP12-R dispatch 6 phase F2 (CARD-RUN5-GPU-OOM, R179): `train.microbatch_caps` is a
    # REQUIRED schema block, so every config necessarily gains both leaves — the re-mint is
    # still purely ADDITIVE and the textual half stays insert-only. run5 additionally gains
    # one `# delta:` HEADER line, which is an insertion too. The values are the operator's at
    # mint prereg (R119/R85) and this instrument does not pin them; `_RUN5_ARMED` below is the
    # closed set of values that ARE pinned, and it is deliberately not widened here.
    "train.microbatch_caps.max_edges",
    "train.microbatch_caps.max_nodes",
    # R242 (ADJ-D12): `monitor.gate_interval` is a REQUIRED schema leaf, so every config
    # necessarily gains it and the re-mint stays purely ADDITIVE. Its VALUE is not pinned
    # here for the same reason the caps' are not — but it is not an operator choice either:
    # the bundle mints it EQUAL to each config's own `train.log_interval`, which is what
    # keeps the arming cadence byte-identical in effect across the split. The re-derived
    # stride is a mint-prereg row. `smoke_preflight_armed.yaml` additionally gains ONE
    # `# delta:` HEADER line (`monitor.gate_interval: 1000 -> 10`), which is an insertion too.
    "monitor.gate_interval",
    # The eval-posture bundle (F-R-P2B-5): two REQUIRED schema blocks, so every config
    # necessarily gains both, and both are minted `null` — the DISARMED posture, which is
    # also the identity value. TWO leaves and not five, because `_leaves` stops at a `None`
    # (a null block has no inner keys in the FILE); the schema-side walker descends the
    # blocks and sees five, and the two counts differ for that reason alone. Each config
    # gains exactly two body lines and NO `# delta:` header line — the template value and the
    # config value are the same `null` everywhere, so there is no delta to stamp and the
    # textual half stays purely insert-only. No VALUE is pinned here: arming either block is
    # a mint-prereg row, and this instrument is the thing that would catch an arming that
    # arrived without one.
    "eval.ply_cap_adjudication",
    "eval.strength_floor",
    # F-816-10 (R276(f)): `inference.fused_graph_caps` is a REQUIRED schema block, so every
    # config necessarily gains both leaves and the re-mint stays purely ADDITIVE — the textual
    # half is still insert-only. TWO leaves and not one, because the block is minted as a
    # MAPPING everywhere (unlike the eval-posture blocks, whose `null` value stops `_leaves`
    # at the block): the five non-production configs carry derived non-binding ints and the
    # two production configs carry `{max_fused_edges: null, max_fused_nodes: null}`, which is
    # a mapping with two null members, not a null block. That difference is deliberate — a
    # null BLOCK would be an off state, and the off state for this bound is unrepresentable.
    # `run5.yaml` and `shakedown_20260807.yaml` additionally gain ONE `# delta:` HEADER line
    # each, which is an insertion too. No VALUE is pinned here: the production pair is the
    # operator's measurement at the box (R119), and `tests/config/
    # test_fused_graph_caps_authority.py` is where the `null` placeholder and the derived
    # non-binding values are each asserted on their own terms.
    "inference.fused_graph_caps.max_fused_edges",
    "inference.fused_graph_caps.max_fused_nodes",
}

#: The subset of `_ADDED_LEAVES` that is ARCH-SCOPED (R322(d)) — added only to the configs
#: whose representation HAS the block, and added to no other.
#:
#: **This is a NARROWING of an allowance, not a widening**, and that is why it needs no ruled
#: deletion beside it. Both blocks arrived AFTER the `b482243` baseline, so on a grid config
#: they are not removals at all — they are additions that no longer happen, and the re-mint
#: stays PURELY INSERT-ONLY on every config. `_REMOVED_LEAVES` and `_REMOVED_LINES` are
#: untouched and still closed at one element each.
#:
#: NO MINTED ROW IS TOUCHED, and that is a precondition rather than a remark: the two grid
#: configs are not in `PRODUCTION_CONFIGS` (`run5.yaml` + `shakedown_20260807.yaml`, both
#: GRAPH), no armed value moves, and the four values that stop being written are the
#: templates' own NON-BINDING-BY-CONSTRUCTION numbers — never a sized cap. R322(d) makes a
#: repair that would touch a minted row a HALT; this one does not reach one.
#:
#: DERIVED from the schema's own partition, not typed: `ARCH_SCOPED_KEYS` is the ONE authority
#: on which block belongs to which arch, so a third scoped block needs no edit here and a
#: block that stopped being scoped cannot leave a stale entry behind.
_ARCH_SCOPED_ADDED_LEAVES: dict[str, frozenset[str]] = {
    f"{key.section}.{key.field}": frozenset(
        leaf for leaf in _ADDED_LEAVES
        if leaf.startswith(f"{key.section}.{key.field}.")
    )
    for key in ARCH_SCOPED_KEYS
}


def _added_leaves_for(name: str) -> frozenset[str]:
    """`_ADDED_LEAVES`, minus every arch-scoped block this config's representation lacks.

    The representation is read off the LIVE config through the one loader — structure, not a
    name list — so re-minting a config to the other representation moves its expectation with
    it instead of leaving a stale entry behind.
    """
    arch = load_config(_LIVE / name).identity.representation
    excluded: set[str] = set()
    for key in ARCH_SCOPED_KEYS:
        if arch != key.arch:
            excluded |= _ARCH_SCOPED_ADDED_LEAVES[f"{key.section}.{key.field}"]
    return frozenset(_ADDED_LEAVES) - excluded

#: Exactly what a re-mint may REMOVE, ever — the one dead knob R178(a) ORDERED deleted
#: (WP12-R, assigned to its dispatcher by R183(a); grounds R116/LAW-08 + the F-CS-2
#: measurement that the replay-buffer save is production-dead on every leg). A closed set of
#: one. Widening it is a ruling, not an edit.
_REMOVED_LEAVES = {"train.buffer_save_interval"}

#: The one deleted LINE, byte-exact, that the textual half will tolerate. Every minted config
#: writes this leaf identically (`yaml.safe_dump`, two-space indent, value `0`), so pinning
#: the text costs nothing and buys the guarantee that the tolerated deletion is the ruled one
#: and not merely a deletion of the same SIZE somewhere else in the file.
_REMOVED_LINES = ["  buffer_save_interval: 0"]


#: Exactly which BODY leaves a re-mint may MOVE, by (config, dotted key). A closed set of one,
#: and the THIRD non-insertion this file has ever admitted.
#:
#: **THE OPERATOR'S GRANT, 2026-08-27.** Phase W of the re-calibration sitting mints a
#: MEASUREMENT-DERIVED `selfplay.n_workers` under R309(f)'s knee rule, and `n_workers: 1` — the
#: value the `b482243` baseline carries — is the ONE value that rule REJECTS. So every pick the
#: pre-registered ladder can legally produce moves this leaf, and both halves of this file
#: refused it: the structural half as a moved value, the textual half as a `replace`. The
#: operator granted the widening rather than let a measurement-derived mint be blocked by an
#: instrument written before the measurement existed. Widening it FURTHER is a ruling, not an
#: edit — the same sentence `_REMOVED_LEAVES` carries.
#:
#: **THE WIDENING, RULED: R326(c), 2026-08-31, enacted by the operator's forwarding of the
#: RECAL-SITTING-5 launcher.** The 2026-08-27 grant named `run5.yaml` alone; R316(f) then ruled
#: that the same measurement-derived pick mints into ALL SEVEN configs — *"the corrected block
#: mints shakedown's `n_workers` to the Phase W pick with the other six configs"* — which
#: post-dates the grant and collides with it. R326(c) resolves the collision the way this file
#: demands: by a RULING that names the change, never by relaxing the instrument to "moves are
#: fine". **The widening is along the CONFIG axis ONLY. The KEY axis stays closed at one**, and
#: `test_a_second_moved_leaf_still_reds` asserts exactly that, so a second ruled KEY still needs
#: a second ruling.
#:
#: DERIVED from this file's own two config authorities rather than typed as seven pairs: a
#: config that joins the live set joins it through `_POST_BASELINE_MINTS` with grounds, and
#: `test_the_committed_baseline_covers_every_minted_config` refuses any live config that is in
#: neither. A transcribed list of seven would go stale the first time that happened and would
#: then be read as evidence (R192(e)).
#:
#: **SUBSET, not equality, and the difference is deliberate.** `_REMOVED_LEAVES` asserts
#: equality because its deletion HAS happened and must stay happened. This is a PERMISSION
#: whose exercise depends on a mint: a clone from before the mint moves nothing and must stay
#: green; the tree after it moves exactly this leaf. Nothing is given up by the weaker
#: relation — a move on any key OUTSIDE this set still reds, which is the whole of the "and
#: only those rows" bite, and `test_a_second_moved_leaf_still_reds` is its producer.
_MOVED_LEAF_KEY = "selfplay.n_workers"
_MOVED_LEAVES = {(name, _MOVED_LEAF_KEY)
                 for name in (*_CONFIGS, *(n for n, _t in _POST_BASELINE_MINTS))}

#: Exactly which minted-header delta lines R187's re-mint may RE-RENDER, by (config, key). A
#: closed set of five, and the second non-insertion this file has ever seen.
#:
#: `tools/mint_config.py` stamped its header values with Python `str()`, so a `None` came out
#: as the six characters `None` and read back as the STRING `"None"` — which is why replaying
#: `smoke_preflight_armed.yaml`'s `eval.ladder.rungs` delta through the tool that wrote it
#: failed schema validation. R187 orders the serializer fixed and the affected configs
#: re-minted; a re-mint rewrites the header line, which is a `replace`, which this file
#: forbids. Same two shapes as the R178(a) deletion above, same resolution: teach the
#: instrument the ruled change BY NAME rather than relax it.
#:
#: The allowance is not a free pass on these five lines. `_is_ruled_reheader` re-derives each
#: one: the key must be unchanged, and applying the OLD renderer (`str()`) to the NEW slot's
#: parsed value must reproduce the BASELINE slot byte-for-byte. That is a proof that the ONLY
#: thing that changed is the rendering — a re-mint that moved a value cannot satisfy it, and
#: neither can a hand-edit. Reach beyond that is held by `test_mint_header_roundtrip.py`,
#: which pins every live header slot to canonical form AND to the config body's own value.
_REHEADERED_DELTAS = {
    ("run5.yaml", "monitor.actor_lag_abort_enabled"),
    ("run5.yaml", "train.draw_rate_abort"),
    ("smoke_preflight_armed.yaml", "train.draw_rate_abort"),
    ("smoke_preflight_armed.yaml", "monitor.actor_lag_abort_enabled"),
    ("smoke_preflight_armed.yaml", "eval.ladder.rungs"),
}

#: R122's minted family (revisable at mint prereg per the R85 pattern — the literals were
#: dead, so nothing has ever measured them; a revision is a prereg row, not an IMPL edit).
_DISK_GUARD = {"monitor.disk_guard.interval_sec": 60.0,
               "monitor.disk_guard.warn_gb": 10.0,
               "monitor.disk_guard.fail_gb": 5.0}

#: R119's HARD STOP, in the config's own units.
_RUN5_ARMED = {"train.draw_rate_abort.threshold": 0.25,
               "train.draw_rate_abort.min_step": 25000,
               "train.draw_rate_abort.N_pool_min": 50}


def _leaves(node, prefix: str = "") -> dict[str, object]:
    """Every leaf key-path -> value. A list is a LEAF: reordering a ladder's rungs is a
    value change, not a structural one, and must be seen as such."""
    if isinstance(node, dict):
        out: dict[str, object] = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): node}


def _is_ruled_reheader(name: str, old_line: str, new_line: str) -> bool:
    """True iff `new_line` is `old_line` with its delta values RE-RENDERED and nothing else.

    Derived, not transcribed: the two slots are split back out, and `str()` — the renderer
    R187 replaced — applied to the NEW slot's parsed value must reproduce the BASELINE slot
    exactly. A replay that moved a value, renamed a key, touched a body line, or re-rendered a
    delta outside `_REHEADERED_DELTAS` fails at least one conjunct."""
    if not (old_line.startswith("# delta:") and new_line.startswith("# delta:")):
        return False
    old_key, _, old_rest = old_line[len("# delta:"):].strip().partition(":")
    new_key, _, new_rest = new_line[len("# delta:"):].strip().partition(":")
    if old_key.strip() != new_key.strip():
        return False
    if (name, new_key.strip()) not in _REHEADERED_DELTAS:
        return False
    old_before, old_sep, old_after = old_rest.strip().partition(" -> ")
    new_before, new_sep, new_after = new_rest.strip().partition(" -> ")
    if not (old_sep and new_sep):
        return False
    return all(str(yaml.safe_load(new)) == old
               for old, new in ((old_before, new_before), (old_after, new_after)))


def _ruled_move_paths(name: str) -> dict[str, str]:
    """`leaf name -> dotted path` for this config's ruled moves, REFUSING an ambiguous leaf.

    The textual half sees `  n_workers: 8`, a body line with no section context, so the leaf
    name has to resolve to a dotted path. That resolution is DERIVED from the config's own
    leaf map, and it REFUSES rather than guesses when a leaf name is not unique — a predicate
    that silently picked one of two same-named keys would tolerate a move on the wrong one,
    which is exactly the class this file exists to catch."""
    live = _live_leaves(name)
    resolved: dict[str, str] = {}
    for config_name, dotted in _MOVED_LEAVES:
        if config_name != name or dotted not in live:
            continue
        leaf = dotted.rsplit(".", 1)[-1]
        matches = [path for path in live if path.rsplit(".", 1)[-1] == leaf]
        assert matches == [dotted], (
            f"{name}: the ruled-move leaf {leaf!r} resolves to {matches}, not uniquely to "
            f"{dotted!r}. This predicate refuses an ambiguous leaf rather than picking one"
        )
        resolved[leaf] = dotted
    return resolved


def _is_ruled_move(name: str, old_line: str, new_line: str) -> bool:
    """True iff the two BODY lines are the SAME ruled leaf carrying its baseline and live values.

    Every conjunct is DERIVED from the two configs, none transcribed: same indentation, same
    key, the key resolves (uniquely) to a `_MOVED_LEAVES` path for this config, the OLD value
    parses to the value the BASELINE config holds there, and the NEW value parses to the value
    the LIVE config holds there. A line that disagrees with the body it claims to describe
    fails, and so does a move on any key nobody named."""
    if old_line.lstrip().startswith("#") or new_line.lstrip().startswith("#"):
        return False
    if len(old_line) - len(old_line.lstrip(" ")) != len(new_line) - len(new_line.lstrip(" ")):
        return False
    old_key, old_sep, old_value = old_line.strip().partition(": ")
    new_key, new_sep, new_value = new_line.strip().partition(": ")
    if not (old_sep and new_sep) or old_key != new_key:
        return False
    dotted = _ruled_move_paths(name).get(old_key)
    if dotted is None:
        return False
    try:
        parsed_old, parsed_new = yaml.safe_load(old_value), yaml.safe_load(new_value)
    except yaml.YAMLError:
        return False
    return (parsed_old == _baseline_leaves(name).get(dotted)
            and parsed_new == _live_leaves(name).get(dotted))


def _replace_is_reheaders_plus_insertions(name: str, old: list[str], new: list[str]) -> bool:
    """True iff every OLD line is matched, IN ORDER, by a NEW line that is IDENTICAL or a
    RULED re-header, and every remaining NEW line is an addition.

    THIS IS NOT A NEW ALLOWANCE — it is the SAME rule, expressed at block level instead of
    requiring the two blocks to be the same length. `difflib` merges an insertion that touches
    a `replace` op INTO that op, so a config that both gains a header line and carries a ruled
    re-render on an adjacent line arrives as one `2 -> 3 replace` that the length-equal form
    cannot decompose. WP12-R F2 is the first such case: `run5.yaml` gains ONE
    `# delta: train.microbatch_caps: ...` line beside the two R187 re-renders.

    Nothing that the length-equal form rejects passes here. EVERY old line must still find a
    counterpart that is identical or satisfies `_is_ruled_reheader` — which still requires a
    named `(config, delta key)` in `_REHEADERED_DELTAS` and still PROVES the change is
    rendering-only. A vanished old line, a rewritten old line, a moved value, a re-render on
    an unnamed key: all still fail, because the loop runs out of new lines or never matches.
    `test_the_block_level_replace_rule_still_rejects_a_rewrite` is that closure's producer.
    """
    i = 0
    for old_line in old:
        while i < len(new) and not (new[i] == old_line
                                    or _is_ruled_reheader(name, old_line, new[i])
                                    or _is_ruled_move(name, old_line, new[i])):
            i += 1                       # an ADDED line: allowed, it is an insertion
        if i == len(new):
            return False                 # an old line vanished or was rewritten
        i += 1
    return True


def _live_leaves(name: str) -> dict[str, object]:
    return _leaves(yaml.safe_load((_LIVE / name).read_text(encoding="utf-8")))


def _baseline_leaves(name: str) -> dict[str, object]:
    return _leaves(yaml.safe_load((_BASELINE / name).read_text(encoding="utf-8")))


def test_the_committed_baseline_covers_every_minted_config() -> None:
    """Premise — an oracle whose golden is missing is an oracle that passes vacuously. The
    fixtures manifest holds the sha256 of each of these six (added_by = "WPMAIN"), so a
    baseline edited to make a diff go away is caught by `test_fixtures_manifest.py`, not by
    nobody.

    `_LIVE` (`configs/`) is swept through `discover_configs` (R71/R75), the ONE discovery
    authority gates 7 and 12 both consume — a second flat `*.yaml` glob there would be
    exactly the divergence ADJ-13 F-1 was: a subdirectory/`.yml` shape both gates make
    legal could join `configs/` and slip past this sweep unseen (N4, F-P2B/N4). `_BASELINE`
    stays a flat glob deliberately: it is a frozen fixture snapshot pinned by
    `tests/fixtures/manifest.toml`, not the live directory this row exists to police —
    a file added there needs its own manifest row regardless of how it is discovered."""
    assert sorted(path.name for path in _BASELINE.glob("*.yaml")) == sorted(_CONFIGS)
    assert sorted(path.name for path in discover_configs(_LIVE)) == sorted(
        (*_CONFIGS, *[name for name, _template in _POST_BASELINE_MINTS])), (
        "the live set must be the six this WP re-mints plus exactly the DECLARED "
        "post-baseline mints (`_POST_BASELINE_MINTS`) — a config appearing mid-WP needs its "
        "own row there, with grounds, before this sweep means anything"
    )
    # The template half of each post-baseline row (finding 5c): the row's declared template
    # must be the file's own `# template:` header line, and some BASELINED config must carry
    # the same line — that is what gives the un-baselined mint a drift witness at all.
    baselined_template_lines = {
        next(line for line in (_BASELINE / name).read_text(encoding="utf-8").splitlines()
             if line.startswith("# template:"))
        for name in _CONFIGS
    }
    for name, template in _POST_BASELINE_MINTS:
        line = next(line for line in (_LIVE / name).read_text(encoding="utf-8").splitlines()
                    if line.startswith("# template:"))
        assert line == f"# template: {template}", (
            f"{name}: the declared template {template!r} must be the file's own header line; "
            f"got {line!r} — a declaration the file contradicts is a false provenance row"
        )
        assert line in baselined_template_lines, (
            f"{name}: template {template!r} is used by NO baselined config, so template "
            "drift for this row has no witness anywhere — a post-baseline mint from a new "
            "template needs its own instrument (or a baseline re-cut ruling), not a row here"
        )


@pytest.mark.parametrize("name", _CONFIGS)
def test_the_remint_adds_the_new_keys_and_moves_nothing_else(name: str) -> None:
    """O-E2, structural half — set equality on what appeared, and value equality on
    everything that was already there.

    MUTATION THAT REDS IT: a replay that regenerates from a drifted template and moves,
    say, `train.batch_size` or an eval ladder rung. That is invisible in review (the diff is
    "just a re-mint") and it silently re-postures every run. Value equality over ~170 leaves
    is the only instrument that sees it."""
    baseline, live = _baseline_leaves(name), _live_leaves(name)
    removed = set(baseline) - set(live)
    assert removed == _REMOVED_LEAVES, (
        f"{name}: a re-mint may drop exactly {sorted(_REMOVED_LEAVES)} (R178(a), the one "
        f"ruled dead-knob deletion); got {sorted(removed)}"
    )
    added = set(live) - set(baseline)
    allowed_added = _added_leaves_for(name)
    assert added == allowed_added, (
        f"{name}: the re-mint may add exactly {sorted(allowed_added)} (R120 + R122 + R126, "
        f"minus the arch-scoped blocks this config's representation does not have — "
        f"R322(d)); got {sorted(added)}"
    )
    ruled = {path for config_name, path in _MOVED_LEAVES if config_name == name}
    moved = {path: (baseline[path], live[path])
             for path in baseline
             if path not in _REMOVED_LEAVES and live[path] != baseline[path]}
    unruled = {path: pair for path, pair in moved.items() if path not in ruled}
    assert not unruled, (
        f"{name}: the re-mint MOVED existing values {unruled} — the three schema items are "
        "additive by construction (§6), so any delta outside the named ruled moves "
        f"{sorted(ruled)} is template drift"
    )


@pytest.mark.parametrize("name", _CONFIGS)
def test_the_remint_diff_is_insert_only_apart_from_the_one_ruled_deletion(name: str) -> None:
    """O-E2, textual half — nothing is rewritten, including the minted header, and the ONLY
    thing deleted is R178(a)'s named line.

    The header records `minted-by`, `template` and every delta (`configs/run5.yaml:1-7`); it
    is the provenance a mint record is reconstructed from. A value map cannot see a lost
    header line, a reordered section, or a rewritten comment.

    A `delete` is tolerated only when its removed lines are EXACTLY `_REMOVED_LINES` — same
    text, same count — so a second dropped line, a dropped header line, or a different line
    of the same length all still red.

    `replace` was forbidden outright until R187, and it is not open now: it is tolerated only
    where every replaced line satisfies `_is_ruled_reheader`, which requires a named
    (config, delta key) from `_REHEADERED_DELTAS` and PROVES the change is rendering-only by
    reproducing the baseline slot from the live slot through the old `str()` renderer. A
    re-render nobody ruled, on a key nobody named, or one that moved a value, all still red.

    MUTATION THAT REDS IT: a minter that reformats (re-wraps a long ladder line, re-orders
    keys, normalises quoting) anywhere outside those five lines — and, inside them, one that
    changes what the delta SAYS while reformatting it. The diff stops being insert-only, and
    the re-mint stops being reviewable as "the new keys, minus the one ruled deletion, plus
    the five ruled re-renders, and nothing else"."""
    baseline = (_BASELINE / name).read_text(encoding="utf-8").splitlines()
    live = (_LIVE / name).read_text(encoding="utf-8").splitlines()
    ops = difflib.SequenceMatcher(a=baseline, b=live, autojunk=False).get_opcodes()
    offending = [
        (tag, baseline[i1:i2], live[j1:j2])
        for tag, i1, i2, j1, j2 in ops
        if not (tag in ("equal", "insert")
                or (tag == "delete" and baseline[i1:i2] == _REMOVED_LINES)
                or (tag == "replace"
                    and _replace_is_reheaders_plus_insertions(
                        name, baseline[i1:i2], live[j1:j2])))
    ]
    assert not offending, (
        f"{name}: the re-mint diff must be INSERT-ONLY apart from the single ruled deletion "
        f"{_REMOVED_LINES} (R178(a)) and the five ruled header re-renders (R187); found "
        f"{offending[:3]}"
    )
    deleted = [line for tag, i1, i2, _, _ in ops if tag == "delete" for line in baseline[i1:i2]]
    assert deleted == _REMOVED_LINES, (
        f"{name}: exactly one line may be deleted, once — R178(a)'s; got {deleted}"
    )


def test_the_permitted_deletion_is_exactly_one_named_line() -> None:
    """The allowance's own guard: this file tolerates deletion at all only because R178(a)
    ORDERED one, and the tolerance must never become "deletions are fine".

    MUTATION THAT REDS IT: a later phase that needs its own key gone and widens
    `_REMOVED_LEAVES` / `_REMOVED_LINES` instead of getting a ruling. The sets are closed at
    one element each and the leaf and the line must name the SAME key, so a widened allowance
    cannot pass as a maintenance edit."""
    assert _REMOVED_LEAVES == {"train.buffer_save_interval"}, (
        "the ruled deletion set is R178(a)'s and is closed at one leaf; widening it is a "
        "ruling (R183(a) assigned this one deletion, not a deletion policy)"
    )
    assert _REMOVED_LINES == ["  buffer_save_interval: 0"], (
        "the tolerated line must be the ruled leaf's own minted line"
    )
    assert not (_REMOVED_LEAVES & _ADDED_LEAVES), (
        "a leaf cannot be both added and removed by one re-mint"
    )


def test_the_arch_scoped_narrowing_is_exactly_the_schema_partition() -> None:
    """The new allowance's own guard (R322(d)), in the shape the two above already have.

    MUTATION THAT REDS IT: a later phase that wants a leaf to stop appearing in some config and
    writes it into `_ARCH_SCOPED_ADDED_LEAVES` by hand instead of scoping the key in the
    schema. The mapping is DERIVED from `ARCH_SCOPED_KEYS`, so the only way to narrow this
    instrument is to make the schema itself refuse the key on that arch — which is the change
    the narrowing is supposed to be recording.

    Both directions: every scoped block must actually contribute leaves (a scoped block that
    contributes none is an entry that can never go stale, i.e. a dead allowance), and the
    narrowing must never reach a leaf that is not inside a scoped block."""
    assert _ARCH_SCOPED_ADDED_LEAVES, (
        "no key is arch-scoped, so this narrowing is unused and should go"
    )
    for block, leaves in _ARCH_SCOPED_ADDED_LEAVES.items():
        assert leaves, (
            f"{block} is arch-scoped but contributes no ADDED leaf, so excluding it narrows "
            "nothing and the entry cannot be seen to go stale"
        )
        assert all(leaf.startswith(f"{block}.") for leaf in leaves), block
    narrowed = set().union(*_ARCH_SCOPED_ADDED_LEAVES.values())
    assert narrowed <= _ADDED_LEAVES, (
        "the narrowing names leaves the re-mint never added; it can only subtract from "
        f"`_ADDED_LEAVES`, and {sorted(narrowed - _ADDED_LEAVES)} is outside it"
    )
    assert _REMOVED_LEAVES == {"train.buffer_save_interval"}, (
        "R322(d) is a NARROWING of an addition, not a deletion — it must not have widened the "
        "ruled-deletion set on its way through"
    )
    graph = {n for n in _CONFIGS if load_config(_LIVE / n).identity.representation == "graph"}
    assert graph and graph != set(_CONFIGS), (
        "the baselined configs no longer span both representations, so `_added_leaves_for` "
        "returns one answer for every config and the narrowing is untested"
    )


def test_a_second_moved_leaf_still_reds() -> None:
    """The allowance's own guard: this file tolerates a MOVED value at all only because the
    operator granted it, and the tolerance must never become "moves are fine".

    MUTATION THAT REDS IT: a later sitting that needs its own leaf moved and widens
    `_MOVED_LEAVES` instead of getting a ruling. **The KEY axis is where the closure lives** —
    R326(c) widened the CONFIG axis to the seven R316(f) mints and widened nothing else — so
    this row asserts the key set is exactly one NAMED key, and `_is_ruled_move` refuses any
    line whose key is not IN it. The two ways a second move could slip through (a second key,
    or a predicate that matched on shape rather than on name) are both closed here.

    The `_is_ruled_move` arms are driven against the LIVE tree rather than against a fixture,
    so they stay true of whatever `configs/run5.yaml` currently holds: before the mint the
    baseline and live values agree and every arm is False for want of a difference; after it
    the ruled pair is the one that passes. Neither state can make an UNNAMED key pass."""
    assert {path for _c, path in _MOVED_LEAVES} == {"selfplay.n_workers"}, (
        "the ruled-move KEY set is closed at one named key (R326(c) widened the CONFIG axis "
        "only); a second key is a ruling, not a maintenance edit"
    )
    assert {name for name, _p in _MOVED_LEAVES} == set(_CONFIGS) | {
        name for name, _t in _POST_BASELINE_MINTS
    }, (
        "R326(c) grants the move on the seven configs R316(f) mints the pick into — which is "
        "this file's own live set, derived; a hand-typed subset or superset is drift"
    )
    assert not {path for _c, path in _MOVED_LEAVES} & _ADDED_LEAVES, (
        "a leaf cannot be both added by the re-mint and moved by it"
    )
    assert not {path for _c, path in _MOVED_LEAVES} & _REMOVED_LEAVES, (
        "a leaf cannot be both removed by the re-mint and moved by it"
    )
    assert not (set(_MOVED_LEAVES) & _REHEADERED_DELTAS), (
        "the body-move allowance and the header re-render allowance must not name the same "
        "(config, key): one tolerates a moved VALUE, the other proves nothing moved"
    )
    # An UNNAMED key of exactly the tolerated SHAPE must still fail — the predicate matches on
    # the ruled name, never on "looks like a scalar that changed".
    assert not _is_ruled_move("run5.yaml", "  batch_size: 256", "  batch_size: 128"), (
        "a body line of the tolerated shape on a key nobody named must not pass"
    )
    # An UNNAMED key in a config the grant DOES cover must still fail — the replacement for
    # the pre-R326(c) arm that used `dev_example.yaml`'s `n_workers`, which the widening makes
    # a ruled pair. The negative control has to move to an axis the ruling did not widen, or
    # it stops being a control: this one is the KEY axis, checked on a non-run5 config so it
    # is not merely the `batch_size` arm above wearing a different file name.
    assert not _is_ruled_move("dev_example.yaml", "  seed: 20260716", "  seed: 20260717"), (
        "R326(c) names one key on seven configs; a DIFFERENT key on a covered config is unruled"
    )
    # THE ARM THAT ISOLATES THE NAME CONJUNCT, and it is here because R326(c)'s flip-one-byte
    # pass found the docstring's second closure ASSERTED and not driven. Every arm above is
    # refused by the VALUE conjunct before the name conjunct is ever reached: a predicate that
    # dropped the `_MOVED_LEAVES` lookup entirely and resolved any unique live leaf still
    # passed all of them. This one cannot be: it feeds an UNNAMED key carrying its OWN
    # baseline and live values, so both value conjuncts hold by construction and only the
    # name lookup can return False. Values are DERIVED from the two configs — a transcribed
    # pair would stop satisfying "its own values" the moment either config moved.
    unnamed = "train.batch_size"
    assert unnamed not in {path for _c, path in _MOVED_LEAVES}, (
        f"{unnamed} must be OUTSIDE the ruled set for this to be a control at all"
    )
    unnamed_base = _baseline_leaves("run5.yaml")[unnamed]
    unnamed_live = _live_leaves("run5.yaml")[unnamed]
    leaf = unnamed.rsplit(".", 1)[-1]
    assert not _is_ruled_move("run5.yaml", f"  {leaf}: {unnamed_base}",
                              f"  {leaf}: {unnamed_live}"), (
        "an UNNAMED key carrying its own baseline and live values must be refused BY NAME — "
        "every other arm here is refused by the value conjunct first, so this is the only one "
        "that can see a predicate matching on shape instead of on the ruled name"
    )
    # A line that disagrees with the body it claims to describe must still fail, whichever
    # side disagrees — this is the conjunct that makes the predicate DERIVED rather than a
    # pattern match, and it is the one a hand-edited config would trip.
    live_workers = _live_leaves("run5.yaml")["selfplay.n_workers"]
    base_workers = _baseline_leaves("run5.yaml")["selfplay.n_workers"]
    assert not _is_ruled_move("run5.yaml", f"  n_workers: {base_workers}",
                              f"  n_workers: {int(live_workers) + 1}"), (
        "a NEW slot that does not equal the live config's own value must not pass"
    )
    assert not _is_ruled_move("run5.yaml", f"  n_workers: {int(base_workers) + 1}",
                              f"  n_workers: {live_workers}"), (
        "an OLD slot that does not equal the baseline config's own value must not pass"
    )


def test_the_block_level_replace_rule_still_rejects_a_rewrite() -> None:
    """The closure guard for `_replace_is_reheaders_plus_insertions`, mirroring the deletion
    and re-header guards. The block-level form exists ONLY so a `replace` op that difflib
    merged an insertion into can be decomposed; it must reject everything the length-equal
    form rejected.

    MUTATION THAT REDS IT: relaxing the predicate to "the new block contains the old block's
    keys somewhere", or to a length/tag check — either would let a re-mint that MOVED a value
    or DROPPED a header line pass as an insertion."""
    ruled_old = "# delta: monitor.actor_lag_abort_enabled: False -> True"
    ruled_new = "# delta: monitor.actor_lag_abort_enabled: false -> true"
    added = "# delta: train.microbatch_caps: {max_edges: 1} -> {max_edges: 2}"
    # the real case: one ruled re-header plus one inserted line
    assert _replace_is_reheaders_plus_insertions("run5.yaml", [ruled_old], [added, ruled_new])
    assert _replace_is_reheaders_plus_insertions("run5.yaml", [ruled_old], [ruled_new, added])
    # a DROPPED old line is still a failure, however many new lines surround it
    assert not _replace_is_reheaders_plus_insertions("run5.yaml", [ruled_old], [added])
    # a MOVED value inside a ruled key is still a failure — `_is_ruled_reheader` re-derives
    # the baseline slot through the OLD renderer and a moved value cannot reproduce it
    moved = "# delta: monitor.actor_lag_abort_enabled: false -> false"
    assert not _replace_is_reheaders_plus_insertions("run5.yaml", [ruled_old], [moved])
    # a re-render on a key nobody named is still a failure
    unnamed_old = "# delta: train.batch_size: 8 -> 256"
    unnamed_new = "# delta: train.batch_size: 8 -> 257"
    assert not _replace_is_reheaders_plus_insertions("run5.yaml", [unnamed_old], [unnamed_new])
    # order is still enforced: two old lines cannot match one new line, and a swap fails
    other_old = "# delta: train.draw_rate_abort: None -> {'threshold': 0.25, 'min_step': 25000, 'N_pool_min': 50, 'consec': 3}"
    other_new = "# delta: train.draw_rate_abort: null -> {threshold: 0.25, min_step: 25000, N_pool_min: 50, consec: 3}"
    assert _replace_is_reheaders_plus_insertions(
        "run5.yaml", [ruled_old, other_old], [added, ruled_new, other_new])
    assert not _replace_is_reheaders_plus_insertions(
        "run5.yaml", [ruled_old, other_old], [other_new, ruled_new])


def test_the_permitted_reheader_is_exactly_five_named_delta_lines() -> None:
    """The re-render allowance's own guard, mirroring the deletion guard above.

    R187 ordered two configs re-minted because their headers carried a stringified `None`;
    exactly five delta lines moved, and the allowance is closed at those five. It must never
    become "header churn is fine" — that would retire the reach this file exists for, since a
    minted header IS provenance and a silently rewritten one is a silently rewritten record.

    MUTATION THAT REDS IT: a later phase adding its own key to `_REHEADERED_DELTAS` instead of
    getting a ruling; or the predicate loosening to "any `# delta:` line may be replaced" —
    the fabricated pair below must stay rejected even though its key IS named."""
    assert _REHEADERED_DELTAS == {
        ("run5.yaml", "monitor.actor_lag_abort_enabled"),
        ("run5.yaml", "train.draw_rate_abort"),
        ("smoke_preflight_armed.yaml", "train.draw_rate_abort"),
        ("smoke_preflight_armed.yaml", "monitor.actor_lag_abort_enabled"),
        ("smoke_preflight_armed.yaml", "eval.ladder.rungs"),
    }, "the ruled re-render set is R187's and is closed at five delta lines"
    assert _is_ruled_reheader(
        "run5.yaml",
        "# delta: monitor.actor_lag_abort_enabled: False -> True",
        "# delta: monitor.actor_lag_abort_enabled: false -> true",
    ), "premise: the predicate accepts the ruled re-render it exists for"
    assert not _is_ruled_reheader(
        "run5.yaml",
        "# delta: monitor.actor_lag_abort_enabled: False -> True",
        "# delta: monitor.actor_lag_abort_enabled: false -> false",
    ), "a re-render that changed the VALUE is not a re-render"
    assert not _is_ruled_reheader(
        "run5.yaml",
        "# delta: seed: 20260716 -> 20260718",
        "# delta: seed: 20260716 -> 20260719",
    ), "a delta key nobody ruled may not be rewritten at all"
    assert not _is_ruled_reheader(
        "run5.yaml", "  buffer_save_interval: 0", "  buffer_save_interval: 1"
    ), "the allowance is for header delta lines, never body lines"


@pytest.mark.parametrize("name", (*_CONFIGS, *[n for n, _t in _POST_BASELINE_MINTS]))
def test_every_config_mints_the_disk_guard_family_at_the_ruled_values(name: str) -> None:
    """O-E2, the R122 family arm. The three values are the previously-DEAD code-side
    literals (`subsystems.py:163-168`), minted so the guard R121(b) mandates has operator-
    visible thresholds instead of numbers nobody could see. Post-baseline mints are swept
    too (F-P2B) — this arm reads only the LIVE file, so it needs no baseline row, and the
    armed-abort manifest's terminal-eval residual leans on the `eval_enabled` half holding
    over every committed config.

    MUTATION THAT REDS IT: mint a different value on one config (a per-config guard posture
    nobody decided), or mint the block on all but one."""
    leaves = _live_leaves(name)
    for path, value in _DISK_GUARD.items():
        assert leaves[path] == value, (
            f"{name}: {path} must mint at R122's ruled {value}; got {leaves[path]!r}"
        )
    assert leaves["eval_enabled"] is True, (
        f"{name}: eval_enabled mints True everywhere — today's effective posture is the "
        "code default True, so True is a zero-behaviour mint (R120; run5's True is LAW-15)"
    )


@pytest.mark.parametrize("name", (*_CONFIGS, *[n for n, _t in _POST_BASELINE_MINTS]))
def test_every_config_declares_a_train_device_from_the_closed_vocabulary(name: str) -> None:
    """O-E2, the R126 arm — the KEY is pinned on every committed config (post-baseline mints
    included, F-P2B: the arm reads only the live file); the VALUE is pinned only where the
    design states it.

    `smoke_preflight_armed` = cpu is transcription, not a decision: its own minted header
    records `eval.worker_device: cuda -> cpu`, every in-repo drive ran it on cpu, and it is
    the CI-runnable smoke (§C.1.6). The other five are operator mint decisions
    (DEP-DEVICE-MINT-VALUES) — pinning them here would be this phase deciding run5's device,
    which is not its call.

    MUTATION THAT REDS IT: transcribe `eval.worker_device` into `train.device` across the
    board. R126 rules them DIFFERENT facts (split topology legitimate), so that inference is
    exactly the proxy the ruling refuses — and it would silently mint the armed smoke to
    cuda, un-CI-running the one config the preflight boots."""
    device = _live_leaves(name)["train.device"]
    assert device in ("cpu", "cuda"), (
        f"{name}: train.device must be a member of the closed vocabulary; got {device!r}"
    )
    if name == "smoke_preflight_armed.yaml":
        assert device == "cpu", (
            "the armed smoke is a genuinely-cpu smoke because its CONFIG is cpu, not because "
            "a flag said so (R64/R61, §C.1.2)"
        )


def test_run5s_armed_draw_rate_values_survive_the_remint_byte_identical() -> None:
    """O-E2's HARD-STOP arm (R119, verbatim: "any change to run5's armed values — 0.25 /
    25000 / 50 — mint-prereg-only").

    Asserted THREE ways, because each sees a different way to lose them: the parsed values
    (a changed number), the literal source lines (a reformat that rounds `0.25` to `0.3`
    through a float round-trip), and the minted header's delta line (a replay that dropped
    the delta and fell back to the template's null, which DISARMS the abort entirely while
    every schema check stays green).

    The header line moved ONCE, at R187, from `None -> {'threshold': 0.25, …}` to
    `null -> {threshold: 0.25, …}` — the same delta, re-rendered so the header replays through
    its own minter. The three numbers are byte-identical across that move and the structural
    arm above proves no leaf shifted, so R119's stop is untouched: nothing was re-armed, a
    rendering was repaired. The pinned text is the CURRENT canonical form, so a regression to
    the old renderer reds this row too.

    MUTATION THAT REDS IT: any of the three. This is the one assertion in the WP that is not
    about correctness but about authority: these numbers were pre-registered at mint prereg
    (R82/R85/R92) and no work package may move them."""
    leaves = _live_leaves("run5.yaml")
    for path, value in _RUN5_ARMED.items():
        assert leaves[path] == value, (
            f"run5 {path} is {leaves[path]!r}, not the pre-registered {value!r} — HARD STOP"
        )
    text = (_LIVE / "run5.yaml").read_text(encoding="utf-8")
    for line in ("    threshold: 0.25", "    min_step: 25000", "    N_pool_min: 50"):
        assert line in text, f"run5's armed line {line!r} is not byte-identical after re-mint"
    assert ("# delta: train.draw_rate_abort: null -> {threshold: 0.25, min_step: 25000, "
            "N_pool_min: 50, consec: 3}") in text, (
        "the minted header's own record of the arming delta must survive the replay — "
        "without it the provenance of the armed values is gone even if the values are not"
    )

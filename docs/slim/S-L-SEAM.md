# S-L-SEAM — LENS (whole tree): the R321 model seam

scope: src/, tools/, crates/, configs/, tests/ (docs only where a new kind would have to edit them).
method: the kind vocabulary taken from `src/mantis/model/arch.py::ARCH_KINDS` at HEAD `69e1532`; a throwaway
AST scan (in the scratchpad) of every tracked `*.py` for `isinstance`/`issubclass` on an arch or net class,
`Compare` nodes carrying a kind string or reading `ARCH_KINDS`/`SOFT_POLICY_ARCH_KINDS`/`INCUMBENT_ARCH_KIND`, and
dict/set/list literals keyed by kind or net names; `git grep` for every kind, net, archived-net and
kind-specific-row name; `git log -S` + `git show --stat/--numstat` on the last landing; PZ.md for membership.
No test was run: every seam test imports torch, which this environment lacks.

"Own scope" of an arch = its net module, its rows in `mantis.model.{arch,build,__init__}`, its conformance
section under tests/model/conformance/, and its own witness tests in tests/model/. `plan/SEAM_V1_DESIGN.md` is
not tracked (PZ-2), so the boundary comes from R321's accept bar ("a second arch must need no edits to the
trainer, server, arena or config schema outside its own scope") and the PZ-2 member table.

## Summary

Findings: 6. DUP 2 (C 2) · SIMPLIFY 3 (C 2, B 1) · TEST 1 (B 1). By lane: B 2, C 4, A 0.
Top 3 by Δ: S-L-SEAM-01 −24 · S-L-SEAM-04 −17 · S-L-SEAM-05 −10. Total −72.

Headline numbers:
- **Leak sites: 36** (list (a) below). 11 branch on a kind or restate a kind table, 7 are the soft-policy kind's
  training feature living in src outside the model package, and 18 are files forced to change by that kind's
  REQUIRED config rows. Rust has **0**: `git grep -c -E "GnnArch|GnnNet|\barch_kind\b|HexTacToe|HeXONet|SoftPolicy|soft_policy" -- crates`
  returns nothing. Rust matches only on `Representation`, which is the encoding identity key and has one
  variant; `grid` is refused by name in `crates/mantis-encoding/src/spec/mod.rs`.
- **The last landing** (`f000803`, GnnArchV2SoftPolicy) touched 39 files. 9 are the arch's own scope
  (+184/−20). **30 are leaks (+455/−27)**, and they include the trainer and the config schema.
- **What a new kind costs today, outside its own scope:**
  - A PLAIN kind (a trunk swap with one policy head and the shared field set) needs **1 forced edit**
    (`tests/eval/test_snapshot_payload_keys.py::_TINY`, which reds until edited) and 3 should-edits: the
    two hand-typed ban censuses and repo_design §3. Those three still pass if missed, but they stop covering
    the new kind. If the class name does not start with `GnnArch`, it also needs **3 more** name-coupled
    edits: the two conformance `specs_for` and `test_anchor`. The trainer, server, arena and config schema
    need 0 edits, so R321's bar HOLDS for a plain kind.
  - A kind that brings its own head, objective and config rows needs **30 files** (measured on `f000803`).
    The bar did NOT hold.
  - With S-L-SEAM-04 and S-L-SEAM-05 applied, the plain-kind forced edits drop to 0 and the name coupling
    is gone.
- **Worst three leaks:**
  1. The trainer carries the soft-policy objective and a kind-table cross-check
     (`src/mantis/train/trainer/core.py`, +119/−8 in the landing). The accept bar names the trainer. The
     objective is ratified as landed (R367, PZ-6), so only the cross-check is proposed (S-L-SEAM-03).
  2. The kind's rows are a REQUIRED key in the SHARED `ModelConfig`. That forces `aux_soft_policy: null`
     into every config, template, payload builder and the strip's synthetic config: 18 files. The
     partition's own vocabulary probe cannot see a kind-scoped key. The regex in
     `test_config_partition_shared_vs_arch_scoped.py::_ARCH_VOCABULARY` returns False on
     `model.aux_soft_policy.*`. The per-representation partition has no per-kind half. Not proposed:
     the fix changes what configs are accepted and adds schema.
  3. The kind discriminator is written three ways: `arch_kind` in `checkpoints._arch_to_dict`,
     `__arch_type__` in `snapshot._arch_to_plain_dict`, and `checkpoints._LEGACY_BY_REPRESENTATION`
     beside `arch.INCUMBENT_ARCH_KIND`. This is the AUDIT-1 F-16 class (snapshot's private table once
     lagged a kind). Proposed in S-L-SEAM-01 and S-L-SEAM-02.

### (a) THE LEAKS — sites outside an arch's own scope that name or branch on a specific arch

A. Branches on a kind, restates a kind table, or couples to kind names (11)

| # | site | what it branches on / names | why it is outside the seam |
|---|---|---|---|
| A1 | src/mantis/config/schema/core.py::SOFT_POLICY_ARCH_KINDS | a per-kind table `{"GnnArchV2SoftPolicy"}` | a kind fact kept in the config schema; a new head-carrying kind edits the schema |
| A2 | src/mantis/config/schema/core.py::RunConfig._soft_policy_rows_pair_with_their_head | `identity.arch_kind in SOFT_POLICY_ARCH_KINDS` | a schema validator that branches on kind membership |
| A3 | src/mantis/train/trainer/core.py::Trainer.__init__ | `type(self.arch).__name__ in SOFT_POLICY_ARCH_KINDS` | the trainer names a kind table (S-L-SEAM-03) |
| A4 | src/mantis/train/checkpoints.py::_LEGACY_BY_REPRESENTATION | `{"graph": GnnArch}` | names V1 and restates `INCUMBENT_ARCH_KIND` (S-L-SEAM-02) |
| A5 | src/mantis/eval/snapshot.py::_arch_to_plain_dict, ::_plain_dict_to_arch, ::_ARCH_TYPES | its own kind discriminator `__arch_type__` | a second serializer beside the loader's (S-L-SEAM-01) |
| A6 | tests/eval/test_snapshot_payload_keys.py::_TINY | a dict keyed by the three arch classes, with an `assert cls in _TINY` | forced edit per kind; the landing edited it (S-L-SEAM-05) |
| A7 | tests/model/test_one_amp_dtype_authority.py::test_no_net_is_constructed_outside_the_one_builder | the set `{"GnnNet","GnnNetV2","HexTacToeNet"}` | hand-typed; already misses GnnNetV2SoftPolicy (DEFECTS) |
| A8 | tests/model/test_arch_ban.py::_RE_ISINSTANCE, ::_RE_TYPENAME | the alternation `GnnNet\|HexTacToeNet` | hand-typed; `\bGnnNet\b` does not match `GnnNetV2*` (DEFECTS) |
| A9 | tests/model/test_flatten_ban_p3.py::KNOWN_DENSE_HEADS_EXEMPT_FROM_FLATTEN_BAN (+ its test) | the buried `HexTacToeNet` | a dead arch name; T11 GRAVES already buries it (S-L-SEAM-06) |
| A10 | tests/train/test_anchor.py::test_a_NON_INCUMBENT_lineage_survives_a_save_load_round_trip | `k.startswith("Gnn")` | reads the representation off a kind NAME instead of `ARCH_KINDS_BY_REPRESENTATION` (0 Δ fix, no finding) |
| A11 | tests/train/test_lr_schedule.py::test_every_cosine_config_builds_the_floored_schedule_from_its_own_rows | `ARCH_KINDS[arch_kind or INCUMBENT_ARCH_KIND[rep]]` | re-types the selector rule that `arch_from_spec_and_config` owns (0 Δ fix, no finding) |

B. The soft-policy kind's feature, in src outside the model package (7). B2 and B5
(`resolve/aux_soft_policy.py`, `Trainer._aux_soft_policy_terms`, `Trainer._policy_head_grad_norms`) are
ratified as landed (R367, PZ-6), so they are listed here and not proposed.

| # | site | what it names |
|---|---|---|
| B1 | src/mantis/config/schema/model.py::AuxSoftPolicyConfig, ::ModelConfig.aux_soft_policy | the per-kind rows, REQUIRED (`default=...`) on every config |
| B2 | src/mantis/config/resolve/aux_soft_policy.py (module) | the rows' resolver |
| B3 | src/mantis/config/resolve/__init__.py | re-exports AuxSoftPolicySpec, MissingAuxSoftPolicyError, resolve_aux_soft_policy |
| B4 | src/mantis/config/schema/__init__.py | re-exports SOFT_POLICY_ARCH_KINDS, AuxSoftPolicyConfig |
| B5 | src/mantis/train/trainer/core.py::TrainHParams.aux_soft_policy, ::Trainer._aux_soft_policy_terms, ::_aux_soft_policy_block, ::_policy_head_grad_norms, the aux arm of ::train_step_from_graph_batch | the kind's objective and its LAW-18 rows |
| B6 | src/mantis/train/losses.py::soft_policy_target | the kind's target construction |
| B7 | src/mantis/train/checkpoints.py::strip_and_restamp | synthetic `"model": {"aux_soft_policy": None}` |

C. Files forced to change by the REQUIRED per-kind key (18)

| # | files | the edit |
|---|---|---|
| C1–C5 | configs/{dev_example,run6,run7,run8,smoke_preflight_armed}.yaml | `aux_soft_policy: null` (run9 also carried it until R367 deleted it; run10 arms it, which is the expected mint) |
| C6 | tools/config_templates/dev.yaml | the row plus a 3-line comment naming GnnArchV2SoftPolicy |
| C7–C10 | tests/config/{test_actor_sync_schema::_payload, test_eval_schema_bounds::_payload, test_schema_strict::_valid_payload, test_train_policy_value_target_consistency::_payload} | `"aux_soft_policy": None` in the payload |
| C11–C14 | tests/train/{conftest::make_run_config (+make_full_train_hparams), test_checkpoint_conformance::test_reads_full_v1_envelope_via_field_map, test_launch_path_smoke::_config, test_resume_wiring_integration::_full_config} | the same |
| C15 | tests/train/_microbatch_harness.py | `aux_soft_policy=None` in graph_hparams; ::tiny_soft_policy_arch, ::soft_policy_graph_trainer |
| C16–C17 | tests/config/test_every_key_has_consumer{,_p2}.py::CONSUMER_REGISTRY | two rows each (LAW-08 bijection) |
| C18 | tests/train/test_aux_soft_policy.py | the trainer feature's tests |

Counted as NAMES ONLY, not as leaks, because a new kind does not have to edit them:
- 4 comments or docstrings name `GnnNet`/`GnnNetV2` as the owner of a method every net has, or as history:
  src/mantis/selfplay/inference_server.py (the `forward_batch` pyright comment),
  src/mantis/train/trainer/core.py (the same comment at `forward_batch_heads`),
  src/mantis/selfplay/graph_collate.py::GraphBatch, and src/mantis/eval/snapshot.py (the `_ARCH_TYPES`
  comment).
- 44 direct arch-dataclass constructions in 39 test files outside the arch's own tests
  (`git grep -n -E "\b(GnnArch|GnnArchV2|GnnArchV2SoftPolicy)\(" -- tests tools ':!tests/model/conformance' …`
  → 44 lines / 39 files). Most use V1 `GnnArch` as the tiny-net fixture, and each re-types
  `in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim`. They cost nothing to ADD a kind, but they are
  the cost of RETIRING V1 (HANDOFF L-DUP).
- `GnnArchV2` is used as "the non-incumbent kind" in test_mint_row, test_arch_stamp_authority,
  test_gnn_widths_resolver, test_fused_graph_caps_authority, test_resume_identity_and_strictness,
  tests/tools/{conftest,test_ladder_backends}, test_compile_trunk and test_graph_microbatch_bound.
- configs name kinds only through `identity.arch_kind`, the selector row, which is expected.

### (b) LOOKS DUPLICATE, IS THE SEAM — do not merge (for L-DUP and the area scouts)

| subject | why it looks duplicate | why it is the contract |
|---|---|---|
| src/mantis/model/arch.py::GnnArch, ::GnnArchV2, ::GnnArchV2SoftPolicy | three frozen dataclasses with identical field sets | deliberate SIBLINGS, never subclasses: `build_net` and the loader dispatch by type, and "the swap is the ARCH, not a knob" (GnnArchV2 docstring). One class with a kind field would re-open the isinstance-twin hazard that T11 checks |
| src/mantis/model/build.py::build_net isinstance chain | could be `ARCH_KINDS[...]` lookup | the conformance suite PARSES this chain (`test_arch_states_its_perf_floor.py::arch_kinds_dispatched` over BUILD_SOURCE; `test_arch_selector_makes_v2_selectable.py` greps `isinstance(arch, {name})`). A table lookup blinds three census tests |
| src/mantis/model/arch.py::ARCH_KINDS, ::ARCH_KINDS_BY_REPRESENTATION, ::INCUMBENT_ARCH_KIND, ::ModelArch | the same three names four times | registry row, pairing rule, history pin (R323) and the union the symmetry-claim census parses. Each has its own conformance test |
| src/mantis/model/gnn.py::GnnNet vs gnn_v2.py::GnnNetV2 / ::GnnNetV2SoftPolicy; ::segment_mean_with_fallback vs ::segment_max_with_fallback; gine.py::RepresentationNetwork vs gnn_v2.py::RepresentationNetworkV2 | parallel forward_batch / forward_single / readout bodies | R321's "hot paths compiled per-arch with zero runtime indirection". Merging them behind a flag is the indirection the contract bans |
| gnn.py::GnnNet.forward_batch_heads / ::policy_heads and their GnnNetV2SoftPolicy overrides | a second forward entry | the trainer's ONE per-net entry (3d93447: "the trainer calls one method and never branches on a kind") |
| tests/model/conformance/test_arch_states_its_perf_floor.py::registered_probes; test_arch_states_its_memory_envelope.py::registered_envelopes | three near-identical rows each, derivable from ARCH_KINDS | each arch must STATE its floor and envelope, set-equal to the dispatch; the suite says a derived list "can never notice one" |
| tests/model/conformance/test_arch_reachability_and_graves.py::GRAVES | names dead archs (HeXONet, ValueHead, HexTacToeNet) | T11 graves prove the dead stay dead (fixture goldens in tests/fixtures/model_graves) |
| src/mantis/config/schema/core.py::ARCH_SCOPED_KEYS + src/mantis/config/resolve/arch_scope.py::refuse_outside_its_arch | the same partition checked twice | schema half and read-path half. The docstring says why: a resolver that refuses only on absence is green by accident |
| src/mantis/model/arch.py::declared_gnn_widths / ::gnn_widths_block; train/checkpoints.py::_SYNTH_ARCH_SCOPED | width read and its inverse; a synth table | an inverse pair, and a loop over ARCH_SCOPED_KEYS that fails loudly on a new block |
| tests/model/{test_arch_v2_dispatch,test_gnn_v2_witnesses,test_gnn_v2_soft_policy}.py | overlap the conformance suite | the arch's own witnesses (W-C1, dispatch, served-output identity); PZ-2 lists the first two |
| tests/fixtures/train/legacy_payload_shapes.json, tests/fixtures/model_graves/hexonet_grave_v1.json | name HexTacToeNet / HeXONet | frozen goldens (PZ-3) |

NOT the seam, although it sits beside it: `eval/snapshot.py`'s serializer pair is a real duplicate of
`checkpoints._arch_to_dict/_arch_from_dict` (S-L-SEAM-01), and the two conformance `specs_for` are copies
with a kind-name sniff (S-L-SEAM-04).

### The last landing, classified

Command: `git log -S GnnArchV2SoftPolicy --oneline`. The landing is `f000803` feat(model). The follow-ups
are `6da4e4c` (run10 mint, expected), `3d93447` REVIEW-1 F4 (moved the kinds table into config and gave
every net `forward_batch_heads`/`policy_heads`) and `deb16b8` REVIEW-2 (added the trainer's kind-table
cross-check). The GnnNetV2 landing (`c5c25cb`) is beyond the shallow clone
(`git rev-parse --is-shallow-repository` → true, 50 commits).

`git show --numstat f000803` → 39 files, split by `^src/mantis/model/|^tests/model/`:
`seam files 9 +184/-20 ; leak files 30 +455/-27`.

- Seam-internal (9): src/mantis/model/{arch,build,__init__,gnn_v2}.py; src/mantis/model/gnn.py (+2/−1:
  `load_from_bc` admits a `reinit` head absent from the source; this is warm-start plumbing, in V1's module);
  tests/model/test_gnn_v2_soft_policy.py; conformance {test_arch_selector_makes_v2_selectable,
  test_arch_states_its_memory_envelope, test_arch_states_its_perf_floor}.py.
- Leak (30): trainer/core.py +119/−8; losses.py +39; config/resolve/aux_soft_policy.py +46;
  resolve/__init__.py +8; schema/core.py +23; schema/model.py +9/−1; schema/__init__.py +4/−1;
  checkpoints.py +1/−1; configs ×6 (+1 each; run9 since deleted); tools/config_templates/dev.yaml +4;
  docs/contracts/{run_config_schema,event_manifest}.md (forced by gate 13 and the manifest);
  tests/config ×6; tests/eval/test_snapshot_payload_keys.py; tests/train ×6 (conftest, harness, aux test,
  3 payload builders).

## Findings

### S-L-SEAM-01 | DUP | C
subject: src/mantis/eval/snapshot.py::_arch_to_plain_dict, ::_plain_dict_to_arch, ::_ARCH_TYPES
claim: The snapshot serializes the declared arch with its own kind discriminator (`__arch_type__`). That
duplicates `train/checkpoints.py::_arch_to_dict/_arch_from_dict` (`arch_kind`, same `dataclasses.asdict` +
`ARCH_KINDS[kind](**d)` shape). Snapshot should call the loader's pair, and its copy, the alias and two
imports go.
evidence: `sed -n 26,47p src/mantis/eval/snapshot.py` → the alias plus two 5–7-line functions keyed on
`__arch_type__`; `sed -n 169,206p src/mantis/train/checkpoints.py` → the same round trip keyed on `arch_kind`.
`git grep -n "__arch_type__" -- tests` → 0, so no fixture or test pins the spool key. anchor.py already
imports `checkpoints._arch_to_dict` (precedent). An eval→train top-level edge is acyclic today: an AST scan of
top-level `from mantis.*` imports shows no train→eval edge, so gate 9's condensed graph stays acyclic.
deliberate?: The alias was made an IDENTITY alias at R330(e) to kill a private kind table (F-16). The
serializer functions were never examined, and they are exactly the second authority F-16 was about: one
more spelling of the discriminator. Two behaviour changes, both lane-C material. (1) The spool file's arch
key changes from `__arch_type__` to `arch_kind`. Snapshots are spool files written and read within one run,
never persisted fixtures. (2) An unknown arch type is refused at load (RepresentationMismatch) instead of at
write (TypeError). It is unreachable, because `.arch` is only planted by `build_net`. The alternative
home, moving the pair into mantis.model.arch, pushes arch.py (271) over the R8 cap and costs a header.
Δlines: −24 (`sed -n 26,47p src/mantis/eval/snapshot.py | wc -l` → 22; plus `import dataclasses` and
`from typing import Any`, whose only uses are inside the deleted lines per `grep -n "dataclasses\|\bAny\b"`).
The import `from mantis.model import ARCH_KINDS, build_net` becomes a checkpoints import in place (0).
Conformance `test_no_second_arch_kind_table.py::test_the_snapshot_module_reads_the_shared_vocabulary_and_keeps_no_copy`
(8 lines) is rewritten to assert the shared serializer by identity (0 Δ); deleting it instead would lower
the test floor.
witness: tests/eval/test_snapshot_payload_keys.py (every-kind round trip; payload keys `{"state_dict","arch"}`);
tests/model/conformance/test_no_second_arch_kind_table.py; tests/eval/test_round_end_to_end.py
depends: —

### S-L-SEAM-02 | SIMPLIFY | C
subject: src/mantis/train/checkpoints.py::_ARCH_KINDS, ::_LEGACY_BY_REPRESENTATION
claim: `_ARCH_KINDS = ARCH_KINDS` is a pure alias. `_LEGACY_BY_REPRESENTATION = {"graph": GnnArch}` restates
`arch.INCUMBENT_ARCH_KIND = {"graph": "GnnArch"}`: both are documented as "a fact about history, not a
default". The four readers should use `ARCH_KINDS` and `INCUMBENT_ARCH_KIND` directly.
evidence: `sed -n 133,141p src/mantis/train/checkpoints.py` → both comments say "history, not a default" and
"imported rather than restated". Readers are `_arch_from_dict` (2) and `stamped_arch_kind` (2). The
`GnnArch` import's only code use is the legacy dict (`grep -n "GnnArch\b" src/mantis/train/checkpoints.py` →
import, a comment, the dict).
deliberate?: The ruling body must confirm that "a pre-discriminator STAMP" and "a config without the row"
name ONE history. Today both answer GnnArch, and `INCUMBENT_ARCH_KIND` is ruling-pinned (R323, PZ-6) as
history that does not move. The conformance PK3 test imports `train.checkpoints._ARCH_KINDS` and would
re-point to `mantis.model.ARCH_KINDS` (0 Δ). PZ-2 lists `_ARCH_KINDS` as a seam member.
Δlines: −6 (`sed -n 136,141p src/mantis/train/checkpoints.py | wc -l` → 6: two comment+assignment blocks
and the blank between them). The import list swaps `GnnArch` for `INCUMBENT_ARCH_KIND` (0).
witness: tests/model/test_arch_v2_dispatch.py (legacy dict → `GnnArch`, unknown representation refused);
tests/train/test_arch_stamp_authority.py; tests/model/conformance/test_arch_states_its_perf_floor.py::test_the_CHECKPOINT_LOADERS_arch_registry_matches_the_same_dispatch
depends: S-L-SEAM-01 (same module family; independent edits)

### S-L-SEAM-03 | SIMPLIFY | C
subject: src/mantis/train/trainer/core.py::Trainer.__init__ (the `SOFT_POLICY_ARCH_KINDS` cross-check) and its import
claim: The trainer's `type(self.arch).__name__ in SOFT_POLICY_ARCH_KINDS` refusal is implied by two checks
that already exist. The trainer's own next check is net heads ⇔ `model.aux_soft_policy` rows. The schema's
`_soft_policy_rows_pair_with_their_head` is rows ⇔ kind in the table. Deleting the cross-check removes the
trainer's only reference to a kind table, which is leak A3.
evidence: `sed -n 187,199p src/mantis/train/trainer/core.py` → the check at `self.soft_policy != (type(self.arch).__name__ in …)`
followed by `self.soft_policy != (self.hp.aux_soft_policy is not None)`. The pinning test
`tests/train/test_aux_soft_policy.py::test_the_net_is_the_authority_for_its_heads_and_the_kinds_table_must_agree`
monkeypatches the net to one head with the rows ARMED, so the rows check fires on the same plant. Only its
`match="kinds table"` changes.
deliberate?: REVIEW-2 G4.4 asked for this check ("the kinds table a cross-check refused by name", deb16b8).
At that time `_aux_soft_policy_block` was gated on the table-derived flag. At HEAD the block is gated on
`hp.aux_soft_policy` and the flag is net-derived. G4.4's hazard, rows reading as a measured head with no
head, is closed by the rows check. The one case left is a hand-built `TrainHParams` that bypasses the
schema (tests only). In that case the trainer trains no aux term and emits no aux rows: an absence, not a
zero. It reverses a reviewed fix, so lane C.
Δlines: −7 (`sed -n 188,193p src/mantis/train/trainer/core.py | wc -l` → 6, plus the import line
`from mantis.config.schema.core import SOFT_POLICY_ARCH_KINDS`).
witness: tests/train/test_aux_soft_policy.py::test_the_net_is_the_authority_for_its_heads_and_the_kinds_table_must_agree
(re-matched), ::test_the_head_and_its_rows_are_one_fact_at_the_trainer_and_at_the_schema
depends: —

### S-L-SEAM-04 | DUP | C
subject: tests/model/conformance/test_arch_states_its_memory_envelope.py::specs_for (copy of test_arch_states_its_perf_floor.py::specs_for)
claim: Two conformance modules define the same `specs_for`. They differ only in the refusal class and
message, and both read an arch's representation off its NAME (`arch_kind.startswith("GnnArch")`). Keep one
in perf_floor, parametrised by the refusal class and reading `ARCH_KINDS_BY_REPRESENTATION` (the seam's own
pairing rule). The envelope module already imports from perf_floor, so it imports this one too.
evidence: throwaway AST spans → perf_floor `specs_for` 274–291 (18 lines), envelope `specs_for` 307–322
(16 lines), identical bodies except the exception. `git grep -n 'startswith("GnnArch")' -- tests` → exactly
these two. Envelope line 29 already does `from test_arch_states_its_perf_floor import BUILD_SOURCE, arch_kinds_dispatched`.
Envelope's `roster` import has no other use (`grep -n roster` → the import plus inside specs_for).
deliberate?: Not a per-arch row. It is a helper, and the name sniff is a leak INSIDE the conformance
section: a kind not named `GnnArch*` gets no graph encodings and is refused for the wrong reason. `_corpus.py`
is NOT the home, because importing `mantis.model` there would make the torch-free conformance modules
(roster guard, config partition, window-frame) torch-dependent. perf_floor already imports mantis.model.
Δlines: −17 (−16 span −2 separating blank lines, +1 for the `refusal` parameter's wrapped signature). No R8
crossing (envelope 494 → 477, still over 300 with its header).
witness: tests/model/conformance/test_arch_states_its_memory_envelope.py (parametrised pairs, EVERY-arch
test); tests/model/conformance/test_arch_states_its_perf_floor.py; test_conformance_roster_guard.py
depends: —

### S-L-SEAM-05 | SIMPLIFY | B
subject: tests/eval/test_snapshot_payload_keys.py::_TINY (+ ::_net, ::test_every_arch_kind_in_the_vocabulary_round_trips_through_the_snapshot)
claim: `_TINY` is a per-class dict holding three IDENTICAL kwargs rows. It is the only site outside the seam
that a new plain kind is FORCED to edit (`assert cls in _TINY`). One kwargs dict applied to every
`ARCH_KINDS` class does the same job with no per-kind edit, and `_net` can reuse it. The two
`net.arch = arch` lines re-do what `build_net` already does.
evidence: `sed -n 86,92p` → GnnArch, GnnArchV2 and GnnArchV2SoftPolicy each map to
`in_dim=11, edge_dim=5, hidden=8, num_layers=1, policy_hidden=8, value_hidden=8`. `_net()` (line 25) types
the same widths. `src/mantis/model/build.py::build_net` ends `net.arch = arch`. The landing `f000803`
edited this dict (+3/−1).
deliberate?: The assert was meant to make a growing vocabulary visible. A new kind whose field set differs
still reds, with a TypeError at `cls(**_TINY)`, and GnnArchV2's docstring states the field set is shared by
design. So detection is kept and only the forced edit goes. This is a design choice (what should red), so
lane B.
Δlines: −10. `_TINY` 7 lines → 1 (−6, AST span 86–92). `assert cls in _TINY` −1. `net.arch = arch` at
lines 28 and 105, −2. `_net`'s two-line ctor → `GnnArch(**_TINY)`, −1.
witness: the test itself (every-kind round trip) and ::test_roundtrip_still_rebuilds_the_identical_net
depends: —

### S-L-SEAM-06 | TEST | B
subject: tests/model/test_flatten_ban_p3.py::KNOWN_DENSE_HEADS_EXEMPT_FROM_FLATTEN_BAN, ::test_hextactoe_net_is_the_named_exempt_dense_head
claim: The test asserts that a test-local constant equals its own literal `{"HexTacToeNet"}`. It names a net
that R346(f) deleted and that the conformance T11 GRAVES table already proves stays dead.
evidence: `sed -n 29,31p;87,88p tests/model/test_flatten_ban_p3.py` → the constant and
`assert KNOWN_DENSE_HEADS_EXEMPT_FROM_FLATTEN_BAN == {"HexTacToeNet"}`. `git grep -n HexTacToeNet -- src` → 0.
tests/model/conformance/test_arch_reachability_and_graves.py::GRAVES carries a `"HexTacToeNet"` row.
deliberate?: The comment calls it a "documentation anchor" for a dense head exempt from the flatten ban.
Nothing dense is left to exempt, and the grave is the standing record.
Δlines: −8 (the constant plus its 2-line comment plus a blank, 4; the test plus 2 blanks, 4). The module
docstring's exemption sentence goes stale and is reworded (not counted). Gate 3: collected count −1, so this
is a test-floor move (tools/ci_gates/test_count_floor.txt). Gate 14: comment runs fall, and the floors may
be lowered, never raised. Overlaps slice T8.
witness: NONE (tautology); tests/model/conformance/test_arch_reachability_and_graves.py::test_a_GRAVE_stays_dead keeps the fact
depends: —

## DEFECTS
- tests/model/test_one_amp_dtype_authority.py::test_no_net_is_constructed_outside_the_one_builder bans direct
  ctors of `{"GnnNet","GnnNetV2","HexTacToeNet"}`. It misses GnnNetV2SoftPolicy, which the last landing did
  not add, and it names a buried net.
- tests/model/test_arch_ban.py::_RE_ISINSTANCE / ::_RE_TYPENAME use `\b(GnnNet|HexTacToeNet)\b`, which does
  not match `GnnNetV2` or `GnnNetV2SoftPolicy` (checked: `isinstance(m, GnnNetV2)` → no match). The sniff
  ban is blind to every kind since V1.
- docs/design/repo_design.md §3 says build_net's "dispatch ORDER is load-bearing, `GnnArchV2` first, because
  `GnnArchV2` is a subclass of `GnnArch`". That is false at HEAD: the arch.py docstrings say "a SIBLING of
  `GnnArch`, never a subclass". It also lists two kinds of three.
- src/mantis/model/__init__.py module docstring says "nets (GNN + CNN)" and lists only GnnArch/GnnArchV2.
  CNN was deleted (R346(f)), and a third kind exists.
- tests/model/conformance/test_config_partition_shared_vs_arch_scoped.py::_ARCH_VOCABULARY has no kind or
  head vocabulary, so `model.aux_soft_policy.*`, a key meaningful on one kind, landed in the SHARED half
  unflagged (regex checked → False on both leaves).

## PARKED
none

## HANDOFF
- L-DUP: "checkpoint → net" is re-typed 7 times: `if ck.metadata.arch is None: raise …; net = build_net(ck.metadata.arch); net.load_state_dict(…)`
  in src/mantis/train/warmstart.py, tools/analyzer/engines.py, tools/bench_server.py,
  tools/ladder/backends.py, tools/probe1/nets.py, tools/probe1/swa.py and tools/strength_frontier.py. They
  use the seam correctly, so this is not a leak.
- L-DUP / T-slices: 44 direct `GnnArch*(in_dim=spec.node_feat_dim, edge_dim=spec.edge_feat_dim, …)` tiny-net
  constructions in 39 test files re-type what `select_arch(spec, cfg, arch_kind=…)` derives.
- C3: src/mantis/eval/worker.py::build_candidate_player still has a `spec.representation == "grid"` arm. That
  representation is refused at registry parse (R346(f)), so the arm is dead.
- C2: `IdentityConfig.representation: Literal["grid","graph"]` and `ArchScopedKey.__post_init__` still admit
  `grid`. These are dense vestiges at the representation level, not the arch-kind level.
- D3: the repo_design §3 subclass claim above.

## Not covered
- The GnnNetV2 landing (`c5c25cb`) and the arch-selector landing are beyond the 50-commit shallow clone, so
  only the GnnArchV2SoftPolicy landing was classified.
- No test or probe was run. Every seam, trainer and snapshot test imports torch, which is absent here, so
  every witness above is named but not executed.
- `plan/SEAM_V1_DESIGN.md` is not tracked. "Own scope" is read from R321's text and PZ-2, not from the
  design's seven parts.
- docs/ were read only for what a new kind must edit (contracts, repo_design). Doc findings belong to D1/D3.
- src/mantis/config/schema/core.py::ARCH_SCOPED_KEYS gaining a per-kind half, which would dissolve C1–C18,
  is not proposed. It changes what configs are accepted (null → absent) and adds schema lines.

## Review
reviewer: fresh read-only agent (not the author); no lane-A finding, so no DELETE-PROBE and no worktree was created.
Reviewed at HEAD 7188aea. `git diff --stat 69e1532 HEAD` shows 6 files, all under docs/slim, so every subject is unchanged from the census base.

| ID | verdict | lane | Δlines | note |
|---|---|---|---|---|
| S-L-SEAM-01 | AMENDED | C | −22 (was −24) | `build_net` still needs its own import line, and a second blank is needed before `def write_model_snapshot`. Adds a top-level eval→train edge on a PRIVATE symbol. The spool reader inherits the legacy no-kind→GnnArch arm. The refusal moves from the parent to the eval child |
| S-L-SEAM-02 | ARCHITECT | C | −6 (confirmed) | Is "a stamp that predates `arch_kind`" the same history as "a config that predates `identity.arch_kind`" (R323 INCUMBENT_ARCH_KIND), or two facts that coincide today? |
| S-L-SEAM-03 | ARCHITECT | C | −7 (confirmed) | This reverses REVIEW-2 G4.4's landed fix (deb16b8, "the table a cross-check and the net the authority"). Should the by-name cross-check stay as defence in depth? |
| S-L-SEAM-04 | CONFIRMED | C | −17 (unprobed; torch-bound, protected suite) | Swapping the name sniff for ARCH_KINDS_BY_REPRESENTATION changes semantics inside PZ-1/PZ-2. The result is identical for all 3 kinds today |
| S-L-SEAM-05 | CONFIRMED | B | −10 | Changes what reds. `build_net` plants `net.arch` (src/mantis/model/build.py) |
| S-L-SEAM-06 | CONFIRMED | B | −8 | Test-floor move (tools/ci_gates/test_count_floor.txt). Module docstring line 4 must be reworded |

### Leak list, re-derived
Method (different from the scout's AST pass): a `tokenize` scan of every tracked .py/.rs/.yaml/.toml under src, tools, tests, crates and configs. Each hit is classed as a code, string or comment token against the kind vocabulary regex: the 3 arch classes, 3 net classes, RepresentationNetworkV2, the 2 buried nets, the 4 tables and their aliases, the `aux_soft_policy`/`soft_policy*` family, `__arch_type__` and `arch_kind`. Own scope was excluded (src/mantis/model/, tests/model/conformance/, the 3 own witness tests). Script and output are in the scratchpad (rev_seam/scan.py, scan.out).

That found 87 files. A second pass took those files and grepped for branch shapes outside own scope: `isinstance`/`issubclass` on an arch or net class, `__name__ ==|in`, `startswith("Gnn`, `type(..) is Gnn*`, `in`/`[...]` on a kind table.
- Branch sites in src: exactly `schema/core.py::_soft_policy_rows_pair_with_their_head` (A2) and `trainer/core.py::Trainer.__init__` (A3).
- Branch sites in tests: A10 and A11. The remaining isinstance/type-is hits are premise asserts (a V1 fixture or the legacy-stamp→GnnArch fact). They are names-only and correctly left uncounted.
- **Count HOLDS at 36**: A 11 · B 7 · C 18. Every row was matched to a scan hit, and the names-only 4 were confirmed by `git grep -n -E "GnnNet|GnnArch"` on the 4 files.
- Rust: 0 confirmed. The crates .rs/.toml files are in the scan, with no hit.
- Caveats, count unchanged:
  - C18 (tests/train/test_aux_soft_policy.py) is the feature's own test, not a file forced by the REQUIRED key. It is misfiled but still a leak.
  - docs/contracts/{run_config_schema,event_manifest}.md were also forced by the landing (gate 13 / the manifest). With docs in scope the count is 38.
- **Worst three: agreed.** Trainer (A3+B5) is first, and the REQUIRED shared key (B1+C1–C18) second. Third is the discriminator restated (A4+A5), which PZ-2 already names as a seam member.

### Last landing, re-derived
`git log -S GnnArchV2SoftPolicy` puts the landing at `f000803`, followed by 6da4e4c, 3d93447 and 9548904.
`git show --numstat --format= f000803`, split on `^src/mantis/model/|^tests/model/`, gives: own 9 +184/−20 · leak 30 +455/−27. All 30 leak paths were listed and match the scout's list exactly: 6 configs, 2 contract docs, 8 src files, 13 tests and 1 template.

### Per-finding notes
S-L-SEAM-01 — AMENDED.
- `grep -n "dataclasses\|\bAny\b" src/mantis/eval/snapshot.py` → lines 17/36 and 20/32/41. Both imports are used only in the deleted span, so their −2 holds.
- `from mantis.model import ARCH_KINDS, build_net` cannot become the checkpoints import "in place (0)": `build_net` stays. That costs +1 for a new `from mantis.train.checkpoints import _arch_from_dict, _arch_to_dict`, and +1 for the second blank that the deleted span carried. Net Δ is −22.
- DUP question: no arch code moves into the trainer, server, arena or schema, and no runtime branch on arch is added.
- Acyclicity was re-checked with my own condensed first-level AST graph: train reaches {_engine, config, encoding, model, monitor, selfplay, util}, and eval is not among them.
- Three behaviour costs the scout did not state:
  - the eval CHILD would import `mantis.train` at top level (today only pipeline.py does, lazily);
  - a private `_`-name would be imported across packages (the anchor.py precedent is intra-package);
  - `_arch_from_dict`'s legacy arm (no kind → GnnArch) would widen to spool files.
- It rewrites a PZ-2-pinned identity test, so it stays lane C.

S-L-SEAM-02 — ARCHITECT.
- `grep -n "_ARCH_KINDS\b\|_LEGACY_BY_REPRESENTATION\|GnnArch\b" src/mantis/train/checkpoints.py` → the alias at 137 and the dict at 141. With comments 136 and 139–140 and blank 138, that is −6. There are 4 readers (185/189/221/224 for the alias, 194/228 for the dict), plus the docstring at 211 to reword.
- Not stated by the scout: once readers use ARCH_KINDS directly, PK3 ("third registry", perf_floor's `from mantis.train.checkpoints import _ARCH_KINDS`) compares the shared table with itself. It should then be deleted, not re-pointed, and that is a test-floor move.

S-L-SEAM-03 — ARCHITECT.
- REVIEW2 G4.4's Fix text (docs/audits/REVIEW2_2026-09-21.md) keeps the table as the cross-check, and deb16b8 landed it. Deleting it reverses a landed review fix.
- The scout's implication argument holds on every production path. `git grep -n "Trainer(" -- src tools` finds only orchestrator.py and pretrain/graph_route.py. Both build `arch` with `arch_from_spec_and_config` over the same config the schema validator pairs.
- `grep -n SOFT_POLICY_ARCH_KINDS src/mantis/train/trainer/core.py` → the import at 28 and the check at 188–193, so −7 holds.
- The pinning test's NAME ("…kinds_table_must_agree") also goes stale (0 Δ).

S-L-SEAM-04 — CONFIRMED.
- An AST dump of both `specs_for` (perf_floor 274–291, envelope 307–322) shows identical bodies apart from the refusal class, its message and the docstring.
- Callers: envelope 330/335; perf_floor 472/506/535.
- `wc -l` → envelope 494, so it stays over 300 and keeps its R8 header.
- DUP question: the code stays in tests, and no pair is a parity-pinned oracle or twin.

S-L-SEAM-05 — CONFIRMED.
- `sed -n 80,110p` shows 3 identical kwargs rows. src/mantis/model/build.py::build_net ends `net.arch = arch`, so the test's two re-plants are redundant.
- `git grep` of docs/governance (archive excluded) for `_TINY`/`test_snapshot_payload_keys` → 0, so it is not ruling-named. Lane B is right: it changes what reds.

S-L-SEAM-06 — CONFIRMED.
- `git grep -n "KNOWN_DENSE_HEADS_EXEMPT\|test_flatten_ban_p3"` → one other hit, docs/governance/archive/RULINGS_ACTIVE.md, a historical spot-check of the line. It is archive, not standing, so the lane does not change.
- GRAVES carries a "HexTacToeNet" row (R346(f)).
- The file is 99 lines, so there is no R8 effect. Gate 3 still moves the floor, so lane B.

DEFECTS, all 5 re-checked:
- (1) The ctor ban set `{"GnnNet","GnnNetV2","HexTacToeNet"}` lacks GnnNetV2SoftPolicy. The gap is latent: `git grep -n -E "\bGnnNet(V2)?(SoftPolicy)?\(" -- src tools tests ':!src/mantis/model'` finds docstrings only.
- (2) A regex probe of `_RE_ISINSTANCE` on [GnnNetV2, GnnNetV2SoftPolicy, GnnNet] → [False, False, True]. It is confirmed, but narrower than stated: the NETS are subclasses (gnn_v2.py `class GnnNetV2(GnnNet)`), so an `isinstance(m, GnnNet)` sniff still catches V2 nets at runtime. The blind spot is a sniff that NAMES a V2 class.
- (3) repo_design.md §3 says GnnArchV2 "is a subclass of GnnArch", but arch.py has `class GnnArchV2:` with no base. The doc confuses the ARCH siblings with the NET subclass chain.
- (4) src/mantis/model/__init__.py opens "nets (GNN + CNN)". Confirmed.
- (5) Evaluating `_ARCH_VOCABULARY` on the two aux leaves and `model.gnn.hidden` → [False, False, True]. Confirmed.

### Missed by the scout
- NEW-1 | names-only | — : tests/selfplay/test_graph_collate_masking_authority.py imports `mantis.model.gnn.GnnNet` and pins the collate contract against `GnnNet.forward_batch`'s signature alone. V1 stands in for every kind here, and none of the scout's names-only buckets lists it.
- NEW-2 | DOC | B : tests/config/test_every_key_has_consumer{,_p2}.py CONSUMER_REGISTRY strings for `model.gnn.*` name "GnnArch/GnnArchV2" as the width consumers and omit GnnArchV2SoftPolicy. The strings are stale, and a new kind leaves them staler.
- NEW-3 | DOC | B : the tests/encoding/test_encoding_round_trip.py module docstring still carries a DEFERRED "HexTacToeNet-forward leg" for a net R346(f) deleted.

### Tally: raised 6 | confirmed 3 | amended 1 | refuted 0 | pending 0 | architect 2

# SLIM-FIX EXIT — R368, packet closed 2026-09-25

Branch `claude/slim-fix-r368`, cut from the SLIM-SCOUT census `1e8d6d6` (base `dev` `69e1532`).
This record is everything that dies with `docs/slim/` (dissolved at this packet's close, R368):
the durable resume point from here is THIS file plus `docs/governance/STATE.md`. The full census,
per-row ledger and per-wave dispatch logs remain in git history at the dissolution commit's parent
(`git show <parent>:docs/slim/PROGRESS.md`, `:docs/slim/LEDGER.md`, `:docs/slim/01_DEFECTS.md`,
`:docs/slim/00_MAP.md`, `:docs/slim/REPROBE.md`, `:docs/slim/S-*.md`, `:docs/slim/handoff/*`).

**Status: READY-TO-MERGE.** The wave STOPS here by ruling: no fast-forward, no push, `dev`
untouched. The operator enacts the merge after reading this record. run10's launch base rule
(R368(i)) applies to the merge: run10 launches from the merged tip only if the full gate set is
green at it, slow tier included (it is, below); otherwise from `69e1532`.

## 1. BEFORE and AFTER figures

Both commands verbatim from `docs/slim/00_MAP.md` §1; unit = text lines from
`git diff --numstat <empty-tree> HEAD` (git's own count; binary files count 0 and are tallied
separately). BEFORE at `1e8d6d6`; AFTER at `2e69d0e2` (the post-W8-D tip, before the exit-record
commits). `E=$(git hash-object -t tree /dev/null)`.

Per top dir (`$E` and the tip substituted for `HEAD` at each measurement):

```
git diff --numstat $E HEAD | awk -F'\t' '{p=$3; top=(index(p,"/")?substr(p,1,index(p,"/")-1):"<root>"); F[top]++; if($1=="-")B[top]++; else L[top]+=$1} END{for(k in F) printf "%-10s %5d %4d %7d\n",k,F[k],B[k]+0,L[k]}' | sort -k4 -n -r
```

| top dir | BEFORE files | BEFORE lines | AFTER files | AFTER lines | Δ files | Δ lines |
|---|---:|---:|---:|---:|---:|---:|
| tests | 546 | 279 440 | 539 | 142 562 | −7 | −136 878 |
| docs | 67 | 40 710 | 114 | 44 407 | +47 | +3 697 |
| crates | 145 | 40 012 | 145 | 36 576 | 0 | −3 436 |
| src | 193 | 38 133 | 176 | 32 850 | −17 | −5 283 |
| tools | 98 | 16 362 | 97 | 15 921 | −1 | −441 |
| (root files) | 13 | 3 646 | 13 | 3 173 | 0 | −473 |
| configs | 6 | 1 055 | 4 | 673 | −2 | −382 |
| vendor | 2 | 149 | 1 | 29 | −1 | −120 |
| .github | 1 | 141 | 1 | 141 | 0 | 0 |
| **all** | **1 071** (33 binary) | **419 648** | **1 090** (30 binary) | **276 332** | **+19** | **−143 316** |

Per package (same pipe with the package key; full AFTER table):

```
git diff --numstat $E HEAD | awk -F'\t' '{p=$3; c=split(p,a,"/"); if(c==1)k="<root>"; else if(a[1]=="src"&&c>=4)k=a[1]"/"a[2]"/"a[3]; else if(a[1]=="src")k="src/mantis/<files>"; else if((a[1]=="crates"||a[1]=="tests"||a[1]=="tools"||a[1]=="docs")&&c>=3)k=a[1]"/"a[2]; else k=a[1]"/<files>"; F[k]++; if($1=="-")B[k]++; else L[k]+=$1} END{for(k in F) printf "%-32s %5d %4d %7d\n",k,F[k],B[k]+0,L[k]}' | sort
```

| package | files | lines | | package | files | lines |
|---|---:|---:|---|---|---:|---:|
| configs/\<files\> | 4 | 673 | | src/mantis/monitor | 12 | 2 555 |
| crates/mantis-bridge | 13 | 4 598 | | src/mantis/selfplay | 13 | 3 896 |
| crates/mantis-core | 18 | 3 623 | | src/mantis/train | 38 | 8 448 |
| crates/mantis-encoding | 9 | 1 510 | | src/mantis/util | 10 | 447 |
| crates/mantis-graph | 8 | 2 696 | | tests/arena | 13 | 1 512 |
| crates/mantis-search | 35 | 10 472 | | tests/bots | 2 | 288 |
| crates/mantis-selfplay | 62 | 13 677 | | tests/bridge | 15 | 1 583 |
| docs/audits | 12 | 4 041 | | tests/config | 53 | 9 840 |
| docs/contracts | 9 | 1 487 | | tests/data | 3 | 611 |
| docs/design | 42 | 12 242 | | tests/diagnostics | 20 | 4 292 |
| docs/governance | 9 | 15 440 | | tests/encoding | 14 | 1 277 |
| docs/slim (dissolved at close) | 42 | 11 197 | | tests/eval | 48 | 8 021 |
| .github/\<files\> | 1 | 141 | | tests/\<files\> | 32 | 5 766 |
| \<root\> | 13 | 3 173 | | tests/fixtures | 56 | 53 595 |
| src/mantis/arena | 9 | 856 | | tests/model | 32 | 7 757 |
| src/mantis/bots | 5 | 413 | | tests/monitor | 18 | 3 729 |
| src/mantis/config | 45 | 5 085 | | tests/selfplay | 61 | 9 910 |
| src/mantis/data | 2 | 420 | | tests/tools | 71 | 13 918 |
| src/mantis/deploy | 1 | 1 | | tests/train | 95 | 19 903 |
| src/mantis/diagnostics | 11 | 4 113 | | tests/util | 6 | 560 |
| src/mantis/encoding | 6 | 1 401 | | tools/analyzer | 13 | 1 306 |
| src/mantis/eval | 11 | 2 938 | | tools/ci_gates | 21 | 5 334 |
| src/mantis/\<files\> | 3 | 921 | | tools/config_templates | 1 | 247 |
| src/mantis/model | 10 | 1 356 | | tools/dashboard | 17 | 2 144 |
| | | | | tools/\<files\> | 22 | 4 788 |
| | | | | tools/ladder | 10 | 984 |
| | | | | tools/probe1 | 8 | 750 |
| | | | | tools/viewer | 5 | 368 |
| | | | | vendor/\<files\> | 1 | 29 |

`docs/slim`'s 42 files / 11 197 lines leave the tree at the dissolution commit (the figure above
is the pre-dissolution tip, as the addendum orders). Reading notes: tests' −136 878 is dominated by
fixture deletions (the 190 638-line "other" fixture column at BEFORE); docs' +47 is the audits and
the packet's own working docs; files grew +19 because reviews, addenda and this record are new
files while deletions were mostly lines within kept files.

## 2. Per-wave ledger summary (W0–W8)

Full per-row detail: `git show <dissolution-parent>:docs/slim/PROGRESS.md` (per-wave exit facts and
dispatch logs) and `:docs/slim/LEDGER.md` §6 (the census row table, verdicts and legs).

- **W0 entry (2026-09-23).** BEFORE sweep `make gates.exit` ALL GREEN, 20 gates incl. slow, at
  `1e8d6d6` in an own worktree. R368 landed (`9ee503d7`). test_count_floor 4862→5112 (`1fbddf17`).
  The 30 PENDING-PROBE census rows re-probed on the torch host: 29 GREEN, 1 AMENDED (TESTS-5-06,
  Δ −48), 0 RED (`docs/slim/REPROBE.md`, in history).
- **W1 correctness (01_DEFECTS 1–39, 42). EXITED** at `cadcc367`. The defect repairs and their
  planted breaks; REVIEW-W1 0 MUST / 3 SHOULD / 8 NOTE, all closed.
- **W2 Rust. EXITED** at `3db6ab5a`. REVIEW-W2 0 MUST / 3 SHOULD / 9 NOTE, all closed except the
  recorded deviation below (finding 3, par.rs fan-out).
- **W3 Python src. EXITED.** 10 implementation commits; REVIEW-W3 0 must-fix beyond a ruff red in
  the handoff helper, 2 should-fix, 4 notes, closed; the exit sweep caught one re-point miss
  (`test_graph_round_encoding.py`'s `_is_graph`) fixed at `09fa8e6b`.
- **W4 tools. EXITED.** 10 commits (sealbot vendor side, the by-path loaders → `mantis.util.loadpkg`,
  run_shard_paths, board.js, gate 14's two new gated measures, gate 16 tree-wide zero — 203 sites).
  REVIEW-W4 0/0/5, all recorded with grounds.
- **W5 tests + fixtures. EXITED** at `8297c74d`. 108 commits, +3 302/−138 142 (fixture bytes
  dominate). Collected 5035→4927. REVIEW-W5 3 MUST / 5 SHOULD / 9 NOTE, fixed in ONE loop. **The
  KNOWN-RED cured** by `cdbc8000` (§6 below).
- **W6 docs + configs. EXITED** at `0c8b9176`. 61 commits. run7.yaml and run8.yaml deleted
  (`8b00b4dd`); **run6.yaml STAYED on a HALT** (§3 below); RULINGS_ACTIVE.md deleted
  (`33e32a16`), CARDS 1028→730 lines; STATE.md rewritten to current facts. REVIEW-W6 1/7/10, fixed
  in ONE loop (three notes left with grounds, each later closed: #12 by W8-D item 4; #3's class by
  W8-D item 2; #13/#15 on contact in W6). Exit sweep ALL GREEN, 19 gates.
- **W7 style pass (R368(g)'s sanctioned pass). EXITED.** 63 commits over
  `d18dc5bc..816c003e`, every one comment-only by the AST/token verifier. Measures: ruling_cite
  1163→91, comment_excess 3124→2144, textfile 466→228. REVIEW-W7 5/7/6, fixed in ONE loop (15
  commits); its carries landed as W8-D item 10. Exit sweep ALL GREEN, 19 gates, at `816c003e`.
- **W8 the close (this wave).** W8-D's operator grants (§5), the AFTER figure (§1), this record,
  the STATE paragraph, the dissolution of docs/slim, REVIEW-W8 (`docs/audits/REVIEW_W8_<date>.md`)
  and the full `make gates.exit` (slow tier included) in `.wt/gates`.

Refuted-at-contact rows and still-C/KEEP rows with grounds are listed per wave in PROGRESS's W5
"REFUTED at contact" / "Still C" sections (in history): TESTS-5-17, TESTS-1-14, TESTS-6-06
refuted; the `_model_samples` pair and the receipt pair, the 13 `train_step_from_tensors` stubs
(the REPROBE verdict), the drawrate trio's private fakes, `_real_graph_ring`, L-STYLE-04/10,
and the closed-unverifiable NEW rows stayed C/KEEP with the grounds recorded there.

## 3. HALTs, and how each closed

- **W6 HALT (run6.yaml stays):** `tests/tools/test_preflight_mint_process.py` is R310-frozen and
  its `RUN5` bound run6 by name; the measured un-narrowing fix needed a grant (R43/R310). The
  filed subset-narrowing shape is preserved at `docs/audits/SLIM_FIX_halt_twin_heldout.diff`
  (moved from `docs/slim/handoff/` at this close; REVIEW-W6 #7 showed it NARROWS the oracle and
  the grant rejected it). **CLOSED by W8-D item 1** (equality-keeping census derivation).
- **W6 HALT 2 (same grant):** `test_the_ruled_deploy_sims_are_pinned` binds run6.yaml by path.
  **CLOSED by W8-D item 1(b)** (deleted; the 160 stays recorded in R346(b) and run6.yaml).
- **W7 stash trap:** L7's `stash pop` took L6's stash (refs/stash is shared across worktrees);
  both sides restored by sha, nothing lost. Closed in-wave; legs were told never to stash again.
- **W8 recovery note (not a HALT, recorded for the record):** the second W8 dispatcher's
  `git checkout -- <file>` on w8-preflight's uncommitted grant-1(a) diff wiped it; it was
  recovered byte-identical from the session's saved diff log (`git apply --check` first). Plants
  in worktrees now use file copies, never checkout.

## 4. The KNOWN-RED (cured)

`tests/monitor/test_supervisor_signal_posture.py::test_the_stop_handlers_are_installed_by_main_and_never_at_import`
reded in FULL default-tier runs on this host (~half of them) across W3–W4 and passed standalone:
the pytest process's SIGINT disposition was transiently SIG_IGN (C-level) when the test spawned
its probe child, and CPython preserves an inherited SIG_IGN across exec. **Cured in W5 by
`cdbc8000`**: the probe child starts from SIG_DFL. REVIEW-W5 §3 H reproduced the old red under
`trap '' TERM INT`, saw the new pass, and confirmed the import-time plants still red (the row is
not narrowed). Every sweep since (W5 re-run, W6, W7) has 3a at 0 failed.

## 5. W8-D — the operator grants of 2026-09-25 (nine items, plus item 10's carries)

Decided by the operator, delegated to the session with "go with the architecturally cleanest
decisions". Each landed as ONE commit naming the grant; every item was executed exactly as scoped.

1. **run6.yaml STAYS in configs/, nothing binds it by name.**
   - (a) `ddad53e0` — the R310-frozen preflight-process oracle re-pointed with the
     EQUALITY-KEEPING shape: `forced_twin_leaves(base_config)` derives the forced set off the base
     the oracle runs on, the `train.heldout_gap` leaves only when the base carries the block;
     base-dependent rows run `@_OVER_THE_CENSUS` over `census.production_configs`. `differing ==
     forced` survives. NOT the filed subset-narrowing diff. Plant: a twin that fails to replace
     the `run_id` forced leaf reds both census members ("Extra items in the right set: 'run_id'").
     169 default + 12 integration rows green.
   - (b) `0f4bab11` — `test_the_ruled_deploy_sims_are_pinned` deleted (floor 4928→4927 in the
     same commit). A minted value is provenance, not expectation.
   - (c) `d03e3d00` — `tests/tools/test_preflight_mint.py`'s modelled constants read off the
     census with an all-members-agree premise (`_modelled_monitor`); the corpus default config and
     the three tool invocations use the census's first member; the audit-green prose names no
     file. 30 rows green.
   - Exit check: `git grep -n 'run6\.yaml' -- src tools tests` is EMPTY (not even history prose
     remains — grant 2's prose rewrite removed the last owner/note mentions).
2. `0affd3ce` — armed_aborts MANIFEST PROSE only: the trough and ply-cap rows lose the stale
   run-named mint wording (now "no production config has armed it yet; a mint that pre-registers
   {…} flips this row REQUIRED" / "flips when every production config arms it" — run10 arms the
   ply-cap values, so the note names what the minted value is); `terminal_eval_broken`'s RESIDUAL
   cites `tests/test_run_eval_enabled_authority.py::test_every_production_config_declares_eval_enabled_true`
   (the census witness c79b4e79 landed). The nine rows' `required`/`config_path`/`source_pin`
   AST-identical before/after; gate 12 rc 0 before (W7 exit sweep, W8 entry) and after;
   `tests/config/test_armed_abort_manifest.py` 22 passed.
3. `16c1b4ed` — `src/mantis/eval/pipeline.py`'s round-completion catch-all: the `logger.exception`
   message no longer repeats `repr(exc)`. The O-30 witness
   (`test_the_round_completion_route_logs_a_traceback_and_the_detail`) reads the detail off the
   log record's `exc_info`. Plants: `_LOG.error` reds ("none carrying exc_info"); `str(exc)` as
   detail reds the re-pointed assertion. 42 passed before/after.
4. `8d77b5da` — R368's Amends line annotated at the RULINGS.md register foot (ANNOTATION under
   R368's foot (A1)): R336(e)'s on-contact clause is the rustfmt one, so (g) replaces nothing in
   it. CLAUDE.md's Rust bullet dropped the "comment-application half" gloss.
5. `c09823e0` — `docs/governance/archive/README.md` keeps ONE tombstone line for RULINGS_ACTIVE.md
   (deleted by R368(e) in `33e32a16`); the narrative paragraph dropped.
6. `5a905f7b` — `Trainer.load_checkpoint` deleted (zero callers; LAW-12's one loader is
   `mantis.train.checkpoints.resume_trainer`/`load_checkpoint`). Resume + checkpoint rows green
   before AND after (131 passed incl. integration-marked; `tests/model/conformance/` 200 passed,
   read-only); run10 MATCH.
7. **CARD-POISON-STANCE stays CARDED** (a runner error-behaviour change beyond slimming; lands
   after run10 STARTs). Recorded recommendation: `into_inner` for the latch/stop/finalize/spawn
   faces, a named error for the drain faces. Ground: the card's own row in
   `docs/governance/CARDS.md` (the poison-error shapes are a behaviour change, not dead code, and
   touching them pre-START would move runner numerics under R368(i)).
8. **REVIEW-W2 #3 stays a recorded deviation:** the shared `par::map_in_order` fan-out
   (`6c30e948`) was kept, not reverted — it dedups fan-out plumbing across the crate; a revert or
   a LAW-09 bench on it is post-run10 work.
9. **run7.yaml and run8.yaml stay deleted** per R368(e) (deleted at `8b00b4dd`). History keeps
   them; every run dir carries its `resolved_config.yaml`.
10. **W7 carries:** `2e69d0e2` — validate.rs's three refusal strings state what the registry
    refuses at load, not the deleted `sym_tables_for` (registry_census 6 + axis_pin 7 green,
    clippy/rustfmt clean, no other reader of the old text). **The style verifier's method limits
    (REVIEW-W7 #18) are NOT landed** — recorded: they matter only if a later style pass reuses
    the verifier; the needed upgrades are a literal-exact Rust pass (token-text equality without
    the doc-comment carve-out that let `///` reflow through), a pragma grep and a heredoc-range
    guard. The verifier itself died with docs/slim; its spec is this paragraph.

## 6. Reviews (R367(b): a fresh read-only agent per wave; filed verbatim with §5 dispositions)

| review | path (docs/audits/) | MUST | SHOULD | NOTE | closure |
|---|---|---:|---:|---:|---|
| W1 | REVIEW_W1_2026-09-23.md | 0 | 3 | 8 | all closed |
| W2 | REVIEW_W2_2026-09-24.md | 0 | 3 | 9 | closed except the recorded par.rs deviation (grant 8) |
| W3 | REVIEW_W3_2026-09-24.md | 0 | 2 | 4 | all closed |
| W4 | REVIEW_W4_2026-09-24.md | 0 | 0 | 5 | all recorded with grounds |
| W5 | REVIEW_W5_2026-09-24.md | 3 | 5 | 9 | one loop, closed; the reviewer verified with its own plants |
| W6 | REVIEW_W6_2026-09-25.md | 1 | 7 | 10 | one loop, closed (3 notes on grounds; #12 closed by grant 4) |
| W7 | REVIEW_W7_2026-09-25.md | 5 | 7 | 6 | one loop (15 commits), closed; carries = grant 10 |
| W8 | REVIEW_W8_<date>.md | — | — | — | see its §5 |

## 7. Models per leg (as recorded in the wave dispatch logs)

- **W1–W4:** PROGRESS's dispatch logs for these waves carry no per-leg model tokens (the recording
  convention began at W5); their legs' reports are in history.
- **W5:** first dispatcher (9839becd..9d520676) slice legs unrecorded; second dispatcher —
  AQ-CARD-FAKES finisher **opus**, drain re-base + follow-up **opus**, train helper hoists
  **opus**, residue2 (Rust + config) **sonnet**, misc (eval/model/tools rows) **sonnet**,
  L20–L34 + W6 inventories **sonnet** (read-only), REVIEW-W5 + closure **opus**, fix loop 1
  **opus**.
- **W6:** configs + configs-b **opus**, cards **sonnet**, mech **sonnet**, reg **opus**;
  REVIEW-W6 **opus**; the third dispatcher's fix legs as recorded in PROGRESS W6.
- **W7:** L1 selfplay, L2 search+encoding, L3 ci_gates, L7 train/config/eval/root **opus**;
  L4 core/bridge/graph, L5, L6, L8 **sonnet**; REVIEW-W7 **opus**; both fix legs **opus**.
- **W8:** grant 3 (o30) **opus**; grants 1(a)+(c) (preflight) **opus** (completed by the second
  dispatcher's session); grants 2/4/5 (prose) **sonnet**; grants 1(b)/6/10 (code) — 1(b) **sonnet**
  (first session), 6 and 10 completed by the second dispatcher's session; grants 7/8/9 are
  records in this file; REVIEW-W8 **opus**.

## 8. The exit sweep (the full gate set, slow tier included)

Procedure (identical to W7's, the pinned way to run a sweep here):
`git -C .wt/gates checkout --detach <tip> && cd .wt/gates && uv sync`, then run
`UV_NO_SYNC=1 make gates.exit` (which is `tools/ci_gates/run_all.sh --with-slow`) as a TRANSIENT
systemd user unit so it survives the session:
`systemd-run --user --unit=mantis-gates-w8 --working-directory=<wt> --setenv=UV_NO_SYNC=1 …`
Logs: `journalctl --user -u mantis-gates-w8`. Its result line is recorded below when it lands
(see §9). Gate 1 (fresh-clone uv sync) stays opt-in per the runner; the slow tier's inclusion is
what makes this the packet-exit gate (R333(b)).

## 9. Exit facts (filled at the close)

- Collected at the post-grant tip `2e69d0e2`: **4 948** = floor (up from 4 928: grant 1(a)'s
  census parametrize added rows; grant 1(b) deleted one). Comment measures at floor:
  comment_excess 2 143, banner 0, docstring_excess 11 897, private_docstring 1 283, rust_doc
  2 235, ruling_cite 91, textfile 228.
- run10 MATCH at the post-grant tip and at the final tip (the pinned procedure: §10's appendix).
- `make gates.exit` at the final tip in `.wt/gates`, user unit `mantis-gates-w8`:
  **ALL GREEN — 20 gates** (result appended below at the sweep's landing).
- The branch does NOT fast-forward `dev` and is NOT pushed. ENACTS 4 is the operator's.

## 10. Appendix — the run10 MATCH procedure (verbatim, before docs/slim dies)

```
.venv/bin/python docs/slim/handoff/run10_resolved.py configs/run10.yaml \
  | cmp - docs/slim/handoff/run10_resolved_base.json
```

`run10_resolved.py` (verbatim, 8 lines):

```python
"""Print run10's resolved config (the post-validation model_dump written to a run dir) as sorted JSON."""
import json
import sys

from mantis.config.loader import load_config

cfg = load_config(sys.argv[1])
print(json.dumps(cfg.model_dump(mode="json"), sort_keys=True, indent=1))
```

NOTE: the base JSON lives in git history after the dissolution
(`git show <dissolution-parent>:docs/slim/handoff/run10_resolved_base.json`); re-create the script
from this appendix (or history) and compare with `cmp`. The handoff tool was dev-only, never tree
code, so its `sys.argv` shape and output formatting are the pinned contract: `model_dump(mode="json")`,
sorted keys, `indent=1`. A mismatch is a HALT under R368(i).

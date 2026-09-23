# S-A-DOCS-1 — D1 (docs/governance non-archive, docs/audits/**, docs/contracts/**)
scope: docs/governance/{STATE,CARDS,RULINGS,LAWS,falsified,COMMS_STYLE}.md, docs/audits/** (incl. archive/), docs/contracts/*.md (9 files); docs/slim/ skipped (this census's own output).
method: `sed`/`awk`/`wc` section spans; `git grep` path and marker citations; AST resolution of backticked names in the contract docs (throwaway script in scratchpad `d1/symcheck.py`: Python defs by `ast`, Rust items by regex); `tools/ci_gates/contract_doc_gate.py` (gate 13) with its flagged symbols re-resolved by AST; `python3 tools/ci_gates/rule7_gate.py --full-tree`; `git grep` comparison of `src/mantis/monitor/producer_manifest.yaml` ids against the event-manifest table. Every finding is a governance doc or a PZ-2 contract doc, so every finding is lane C. Nothing was edited.

## Summary
- 14 findings, all DOC | C. No DEAD, DUP or TEST findings: nothing in the slice is code.
- Top 3 by Δlines: S-A-DOCS-1-01 (STATE history) −674 · S-A-DOCS-1-03 (closed CARDS rows) −313 · S-A-DOCS-1-09 (AUDIT-3, no path citation) −272.
- Kept, with nothing to report: LAWS.md (61 lines, current: 17 laws, LAW-10 retired, the 11-item protected set); REVIEW_2026-09-21.md and REVIEW2_2026-09-21.md (cited by path in STATE, CARDS and RULINGS R367; CLAUDE.md R10 puts review reports under docs/audits/); the 9 contract docs as documents (PZ-2 seam contracts, repo_design §4). Gate 13's 28 distinct flagged symbols all resolve by AST, so its red here comes only from the missing torch.
- Four handoffs from other scouts were verified and folded in: (a) and (c) → 07, (b) → 10, (d) → 13.

## Findings

### S-A-DOCS-1-01 | DOC | C
subject: docs/governance/STATE.md
claim: About 91 % of STATE is dated history, which breaks its own opening rule "Rewritten in place, never appended to". Only the header, the R367 half of the "Current phase" line and dispatcher item (19) describe where the run is now.
evidence: `grep -n '^## \|^([0-9]*)' STATE.md` gives these spans. Header 1–9. Line 10 is 7 092 bytes, of which the text before its first "Before it:" is 1 645 bytes (the rest restates R366…R358). History runs 11–595 (585 lines, 57 264 bytes): the 2026-09-15 run7 resume, the SEALBOT-TT A/B, PERF-A4, Holds 1–2, R349(b), run7's minted values, and dispatcher items (1)–(18). Item (19) is 596–637 (42 lines, 5 000 bytes). "Exit facts — the R353 packet" plus Provenance is 638–726 (89 lines, 9 110 bytes), covering the R353…R363 legs. Whole file: 726 lines, 79 023 bytes.
callers: n/a (DOC). The file is cited by CLAUDE.md, the CALLERS §5 procedures (by tool name, and those tools are all in the tree) and RULINGS.
Δlines: −674 (`sed -n 11,595p | wc -l` = 585, plus `sed -n 638,726p | wc -l` = 89), before the one fresh provenance line a rewrite adds. The history is already carried by RULINGS entries R353–R367 and git.
witness: gate 10 (every `docs/…`/`tools/…` token left in STATE must stay tracked). No other gate reads STATE.
depends: —

### S-A-DOCS-1-02 | DOC | C
subject: docs/governance/STATE.md (§"Dispatcher state for a fresh session", §"Minted values", §"Exit facts", header)
claim: Several present-tense STATE facts disagree with the tree or with STATE's own later items.
evidence: "Dispatcher state…" opens "run7 is LIVE", but item (8) of the same file records run7 STOPPED 2026-09-18. §"Minted values — configs/run7.yaml (the running config…)": the armed config is run10 (`grep -c '^# delta:' configs/run10.yaml` → 44), and run7 now has 35 deltas, not the heading's 41. §Exit facts gives "4 988 collected" and comment floors "3 451 / 0 / 13 387, private 1 530, Rust-doc 2 677"; the tree has `test_count_floor.txt` 4862, and `comment_length_floor.txt` has 3385 / 0 / 13079 (CALLERS §11: 1485 / 2676), with 5 112 collected per item (19). The header says "everything else is the 2026-09-15 rewrite and reads as of that date", but the items run to 2026-09-21. The "Protected set, laws, cards" line is fine: LAWS.md lists 17 laws.
callers: n/a (DOC)
Δlines: 0 (repair in place, R311(c)). Discharged by 01 if STATE is rewritten.
witness: NONE
depends: 01

### S-A-DOCS-1-03 | DOC | C
subject: docs/governance/CARDS.md (closed rows)
claim: 33 card rows that are CLOSED, LANDED, SPENT, DECIDED, DISCHARGED, FOLDED or SETTLED stay in "the open work". The file allows this ("until a ruling sweeps its section"), so a sweep ruling would remove 313 lines. The section "Opened by R346 itself" is closed in full.
evidence: throwaway block splitter (scratchpad `d1_cards.py`: one top-level `- **` bullet per block, and each block's status was read in full) → 33 blocks, 313 lines. They are: the 4 R346 rows, CARD-REVIEW-1, RUN10-SIZE-PARENT, PROBE-1 (SPENT by R366), RUN10-RULE (SPENT), EVAL-GATE-FIELDS-IN-STREAM, STRIX-NET-ONLY, PERF-3 (SPENT), SERVER-OWNED-COPY (CLOSED), PUCT-ATTRACTOR, SEALBOT-GIL-SERIAL, SEALBOT-TT-SEAT, SELFPLAY-SEARCH-STATS (LANDED), STAMP-FLOOR, A4-MINT, PREFLIGHT-CADENCE, PHASE-W-AT-GUMBEL, ALPHA-TARGET-FORM, EVAL-CADENCE, SEALBOT-HORIZON, BOX-VOLUME, WARMSTART-STAMP-SCHEMA, TIER-HOST, GATES-ON-CUDA-VENV, CHECKER-THREAD-LEVER, F-816-24, kill-grace SETTLED, run5 prereg SPENT, R359(e), R245(c). The same file's tables also list F-816-24 CLOSED a second time, and CARD-MAXPLIES as "CLOSED IN CODE".
callers: n/a (DOC)
Δlines: −313 (block script above)
witness: gate 10 (paths cited in removed rows are irrelevant to it; removing text cannot red it)
depends: —

### S-A-DOCS-1-04 | DOC | C
subject: docs/governance/CARDS.md §"CARD-* that exist only as in-source markers"
claim: The section's "status comes from the code" premise is false at HEAD. 6 of its 7 markers are gone from the tree, and 10 markers that are in the tree are missing from CARDS.md; 4 of those 10 appear in no governance doc.
evidence: per marker, `git grep -c -F "<CARD>" -- <named file>` plus a tree-wide `git grep -l`. Only CARD-MINT-RESOLVE-PARENT-CONJUNCT has a hit; BUDGET-AUTHORITY-CONSOLIDATION, GAME-RECORD-SELFPLAY-STATS, CONFIG-DISCOVERY-ROOT, EXEMPT-CONFIGS-OPERATOR-CONFIRM, PREFLIGHT-ORACLE-OUTDIR-CLEANUP and DESIGN-P-3.4-ORDERING have 0 hits. In reverse, `git grep -h -o "CARD-[A-Z0-9-]*" -- src tests tools crates configs pyproject.toml .github` finds these absent from CARDS.md: CARD-ABORT-EXIT, CARD-COORD-KNOBS (7 hits), CARD-CS2, CARD-ORPHAN-WORKERS, CARD-POOL-ENCODING-BRIDGE, CARD-RUN-MAIN and CARD-LINT-GATE, which RULINGS names. With 0 governance mentions: CARD-PREFLIGHT-OUTDIR-REUSE, CARD-PREFLIGHT-CHILD-STDERR-BUDGET (tests/tools/test_preflight_pfc_cards.py), CARD-RING-SAMPLER-SEED (crates/mantis-selfplay/tests/replay_sampler_seed.rs) and CARD-TRAINSTEP-ADAPTER (tests/tools/test_preflight_mint_process.py).
callers: n/a (DOC)
Δlines: −9 (the six dead entries: 2+2+2+1+1+1 lines, read off `awk '/^## CARD-\* that exist only/,/^## RQ/'`). The 10 missing rows would add lines back; that is the ruling's choice.
witness: NONE (no gate derives this section)
depends: —

### S-A-DOCS-1-05 | DOC | C
subject: docs/governance/CARDS.md (headlines, and the section "What holds run6")
claim: Several open-looking headlines disagree with the tree or with the record, which is the trap the file itself warns about ("read the whole row").
evidence: (i) CARD-OC7-OVERRUN is still "BLOCKING … no `make gates.exit` run … can complete". Its own body says "DISCRIMINATED — HOST", CARD-TIER-HOST's carve-out landed (tests/train/test_law06_cpu_carveout.py), and STATE records `gates.exit` green with the integration tier at 49 passed (items 17–19). (ii) "What holds run6" (36 lines) is run6's START hold record; run6 was stopped by R350(a), and its "REPAIR-A2 leg 5 … BLOCKING … `MAX_CHILDREN_PER_NODE = 192`" contradicts `crates/mantis-search/src/mcts/mod.rs::MAX_CHILDREN_PER_NODE` = 1024 (R346(c)). (iii) CARD-RUN9-STOP-LAW is "carried until run9's own stop discharges it", but run9 was never started and `configs/run9.yaml` was deleted (R367), so the row cannot discharge. CARD-EVAL-ROUND-OVERRUN closes on run7's rounds, and run7 is stopped. (iv) CARD-PYRIGHT-STRICT says "live marker at `pyproject.toml:92`", but the marker sits elsewhere in pyproject.toml (`git grep -n PYRIGHT-STRICT pyproject.toml`). "F-01..F-52 at this writing" is wrong because falsified.md has F-53. "1661 lines" is a transcribed count (R192(e)).
callers: n/a (DOC)
Δlines: −36 for "What holds run6" (`sed -n` of the section, header to the next header); the rest is 0 (repair).
witness: NONE
depends: 03

### S-A-DOCS-1-06 | DOC | C
subject: docs/governance/falsified.md rows F-04 and F-43 (append-only; corrects only by annotation)
claim: F-04's pointers resolve to nothing at HEAD, and F-43's annotation cites line coordinates that have drifted.
evidence: `git grep -n "aggregate_cluster_values_min\|MINPIN" -- crates src tests` → 0. `crates/mantis-encoding/src/registry.toml` has `value_pool = "none"` in both rows, where F-04 says `"min"`. The CARDS row CARD-MINPIN ("the parity pin landed") has no in-tree pin either. F-43 cites `src/mantis/run.py:528, 538`; the two `sink=_DeferredSink()` sites are now elsewhere in `src/mantis/run.py` (git grep).
callers: n/a (DOC)
Δlines: 0 (a scope annotation adds lines)
witness: NONE
depends: —

### S-A-DOCS-1-07 | DOC | C
subject: docs/governance/RULINGS.md (the "Coverage" header paragraph, R346's Status line, R271)
claim: This finding carries three register facts that disagree with the tree. Two of them were handed off by other scouts and are verified here.
evidence: (i) The header says "323 entries over 322 numbers" for R23–R345. `grep -o '^### R[0-9]*' | sort -un` gives 343 numbers present for R23–R367, and only R227 and R228 are missing, so R23–R345 has 321 numbers and 322 headings (the extra one is R279(g)-ANNEX). (ii) Handoff (a): R346's Status line says "AUDIT-2 filing is DISCHARGED (`docs/audits/AUDIT_2026-09-09.md`", but the file lives at docs/audits/archive/ (R355(f)). RULINGS is exempt from gate 10, so no gate catches this. (iii) Handoff (c): R271(b)/(d) is `Status: standing` and names `RULINGS_ACTIVE.md` as "the DERIVED working index … curated at session close", yet R346(e) froze it under docs/governance/archive/, and R271 carries no superseded/amended note.
callers: n/a (DOC)
Δlines: 0 (annotation)
witness: NONE (gate 10 exempts RULINGS by declaration)
depends: —

### S-A-DOCS-1-08 | DOC | C
subject: docs/governance/COMMS_STYLE.md
claim: No standing record cites this file. CLAUDE.md's map lists STATE, RULINGS, CARDS, LAWS and falsified but not COMMS_STYLE, and its only citers are the frozen archive and this census.
evidence: `git grep -l COMMS_STYLE` → the file itself, docs/governance/archive/{RULINGS_ACTIVE,rulings_register}.md, docs/slim/00_MAP.md. Four of its 17 lines are a repair narrative (R311(c)).
callers: n/a (DOC)
Δlines: −17 (`wc -l`) if it is folded into CLAUDE.md or deleted. 0 if CLAUDE.md names it instead.
witness: gate 10 (fine either way: no gate-10-scanned file cites it)
depends: —

### S-A-DOCS-1-09 | DOC | C
subject: docs/audits/AUDIT_2026-09-11.md (AUDIT-3)
claim: Nothing in the tree cites this file by path. It is named only as "AUDIT-3" in RULINGS R348(e)/(h) and CARDS. It was taken at wave-3 `5ab4aa7b`, and its lead finding (F-816-24) is recorded as CLOSED by R349(a). Its peer AUDIT-2 went to archive/ under R355(f); AUDIT-3 did not.
evidence: `git grep -l -F AUDIT_2026-09-11 -- . ':!docs/slim'` → 0 files. `git grep -n AUDIT-3 -- docs/governance/*.md` → RULINGS R348(e)/(h), and CARDS CARD-OC7-OVERRUN and F-816-24.
callers: n/a (DOC)
Δlines: −272 (`wc -l`) if deleted. 0 plus one README row if moved to docs/audits/archive/ (that edits a FROZEN README, R355(f), so either route needs a ruling).
witness: NONE
depends: —

### S-A-DOCS-1-10 | DOC | C
subject: docs/audits/archive/AUDIT_2026-09-09.md (FROZEN, R355(f))
claim: Handoff (b) verified. The frozen audit cites paths from before the move: docs/governance/RULINGS_ACTIVE.md and docs/governance/rulings_register.md (both now under archive/), and docs/registers/{falsified,laws}.md (dissolved). This is expected in a point-in-time record, and the archive README already carries the caveat "not a statement about HEAD". No edit is proposed.
evidence: path tokens `docs/(registers|governance|…)/…` each checked with `git ls-files --error-unmatch` → the 4 above are missing.
callers: n/a (DOC)
Δlines: 0 (keep, frozen)
witness: NONE (gate 10 does not scan docs/audits)
depends: —

### S-A-DOCS-1-11 | DOC | C
subject: docs/contracts/{eval_instrument,event_manifest,run_config_schema,checkpoint_envelope}.md (small drifts)
claim: Four contract claims disagree with the code they describe.
evidence: (i) eval_instrument.md says `EvalBrokenReason` has "seven members" ("Which of the seven broke"), and event_manifest.md's rc-48 paragraph says "ONE code for seven reason classes". An AST read of `src/mantis/eval/errors.py::EvalBrokenReason` finds 8 members, `ABANDONED` added with CARD-STOP-DRAIN-VS-GRACE, and event_manifest.md's own `eval_broken` row lists all 8. (ii) eval_instrument.md says the candidate and best are built "through one `_build_candidate_player` call site". HEAD has the public `src/mantis/eval/worker.py::build_candidate_player` with 6 calls in worker.py, plus diagnostics/acceptance_witness.py and tools/analyzer/engines.py. (iii) run_config_schema.md's "Who asserts" table names `mantis.config.loader (_UniqueKeyLoader -> DuplicateKeyError)`. `_UniqueKeyLoader` has 0 hits in src; the loader imports `DuplicateKeyError`/`parse_config_yaml` from `mantis.util.yaml_io`. Gate 13 resolves only `mantis.*` tokens, so bare names like this one escape it. (iv) checkpoint_envelope.md: `metadata.arch` is "the declared `GnnArch` dataclass", but three arch kinds are live (`src/mantis/model/arch.py::ARCH_KINDS`).
callers: n/a (DOC)
Δlines: 0 (repair in place; each contract's version and tests move together, per repo_design §4)
witness: gate 13 for (iii) only in principle (it does not read bare names); NONE for the rest
depends: —

### S-A-DOCS-1-12 | DOC | C
subject: docs/contracts/event_manifest.md §"Shipped rows (WP13-A)" vs src/mantis/monitor/producer_manifest.yaml
claim: The table presents 6 ids as manifest rows that the manifest does not contain, and it omits a manifest row that does exist. No test binds the doc to the manifest.
evidence: `grep -n "id:" producer_manifest.yaml` gives 16 ids. The doc's table rows (`| \`<id>\` | symbol\|event_literal`) include eval_round_wall, eval_round_progress, eval_broken, eval_strength_floor, eval_ply_cap_adjudication and eval_round_device_memory, each with 0 `id:` rows in the manifest. `resolved_config` has a manifest row and 0 mentions in the doc. `git grep -l event_manifest.md -- tests` → only tests/train/test_training_step_absence.py, which checks something else.
callers: n/a (DOC)
Δlines: 0 (repair). A derive-don't-transcribe check in the style of gate 13 would be a lane-B design choice.
witness: NONE
depends: —

### S-A-DOCS-1-13 | DOC | C
subject: docs/contracts/event_manifest.md training_step roster (handoff (d); S-A-CORE-1-02)
claim: The roster documents payload fields (`loss_ownership`, `loss_threat`, `avg_sigma`, `n_rows_total`, `value_accuracy`, among S-A-CORE-1-02's ten) that no trainer produces. The contract describes fields that read None on every row.
evidence: `git grep -n -E "[\"'](ownership_loss|threat_loss|avg_sigma|value_accuracy|n_rows_total)[\"']" -- src` → only the `measured(loss_info, …)` READS in `src/mantis/train/events.py::emit_training_step_event`, and no writer exists (S-A-CORE-1-02 has the per-key evidence).
callers: n/a (DOC)
Δlines: follows S-A-CORE-1-02 (the doc rows and the code rows go in one commit)
witness: tests/train/test_training_step_absence.py (ABSENCE_CAPABLE)
depends: S-A-CORE-1-02

### S-A-DOCS-1-14 | DOC | C
subject: docs/contracts/run_config_schema.md §"Version history"
claim: The contract carries 37 one-line version rows totalling 59 946 bytes, averaging 1.6 KB each. Each row narrates its amendment, and those reasons already live in RULINGS and git. A pointer per version would keep the contract readable. The "Deliberately absent" section is separate and gated (gate 13 checks it in reverse), so this finding leaves it alone.
evidence: `grep -c '^| v[0-9]'` → 37; the byte sum of those rows → 59 946.
callers: gate 13 (tools/ci_gates/contract_doc_gate.py) reads the whole doc. Check which sections it scans before cutting: a history row that names a deleted key must not start failing its forward check.
Δlines: −36 net if the 37 rows collapse to one pointer line. If the rows are only shortened, Δlines is 0 and roughly 55 KB is removed.
witness: gate 13, tests/tools/test_contract_doc_gate.py
depends: —

## DEFECTS
- docs/governance/STATE.md has 15 lines of box-local specifics: box-local absolute run paths, a port-forward tunnel invocation, the rental provider's CLI command with an instance id, and pids. CLAUDE.md gate 17 / Rule 7 says these live off-tree. `python3 tools/ci_gates/rule7_gate.py --full-tree` → "no host content", because the tracked pattern floor does not cover these forms (the local supplement is absent here).

## PARKED
none

## HANDOFF
- C2 (config): `src/mantis/config/schema/core.py::IdentityConfig.representation` is still `Literal["grid","graph"]`, while the registry refuses "grid" by name (registry.md, R346(f)). run_config_schema.md faithfully mirrors the code.
- Dispatcher (PZ.md, PZ-6): lists `CARD-EXEMPT-CONFIGS-OPERATOR-CONFIRM` as a marker in src/mantis/config/armed_aborts.py, but there are 0 hits in the tree at HEAD (see 04).

## Not covered
- RULINGS.md bodies (4 630 lines) were not read beyond the header, R271, R346's Status line, the R367 heading and the head of the Annotations inventory. R346–R367 span 1 651 lines of verbatim packets; they were not checked against the tree.
- Commit-hash citations across STATE, CARDS and RULINGS were not verified. The clone is shallow, and 5 of 25 sampled hashes are absent locally (likely box branches carried by bundle).
- The REVIEW/REVIEW2/AUDIT-3 bodies were read only at their headers; their internal path cites were not checked (point-in-time records).
- game_record.md, preflight_report.md, graph_wire.md, registry.md and replay_persist.md were only spot-checked by symbol resolution (all names resolved or are named as deleted). graph_wire.md's "13 per-array getters" was not counted.
- Gate 13 was not run clean (torch is absent); its 28 flagged symbols were resolved by AST instead.
- docs/slim/ was skipped (census output).

## Review
reviewer: fresh read-only agent (not the author); no lane-A finding, so no delete-probe and no worktree was created (the worktrees listed at the end of review belong to other reviewers and were left alone)

| ID | verdict | lane | Δlines | note |
|---|---|---|---|---|
| 01 | CONFIRMED | C | −674 | spans re-measured; "91 %" is a byte share, by lines it is 92.8 % |
| 02 | AMENDED | C | 0 | the "4 988 vs floor" item is not a disagreement: the floor STATE states is the tree's |
| 03 | CONFIRMED | C | −313 | own splitter, same 33 blocks; CARD-E1-RULER-R6 may also be sweepable |
| 04 | AMENDED | C | −9 | the reverse set has 11 markers, not 10 (7 named in RULINGS + 4 in no governance doc) |
| 05 | CONFIRMED | C | −36 | (iv): "F-01..F-52" is dated, not wrong; "1661 lines" is still correct at HEAD |
| 06 | CONFIRMED | C | 0 | |
| 07 | CONFIRMED | C | 0 | plus NEW-1 in the same paragraph |
| 08 | CONFIRMED | C | −17 | gate 10's docs/governance glob floor (4) survives a delete (5 files left) |
| 09 | CONFIRMED | C | −272 | |
| 10 | AMENDED | C | 0 | 6 cited paths are missing, not 4 |
| 11 | CONFIRMED | C | 0 | (ii) has a 4th caller site; (iii) the loader was renamed and moved, not deleted |
| 12 | CONFIRMED | C | 0 | |
| 13 | CONFIRMED | C | follows S-A-CORE-1-02 | |
| 14 | AMENDED | C | −36 only if the latest `\| vN \|` row is kept | gate 13 reads the version table and reds without it |

### Per-finding notes
S-A-DOCS-1-01 — CONFIRMED.
- Command: per-span `sed -n A,Bp | wc -lc` over STATE, with dated lines counted by a `2026-..-..|R3NN` grep.
- Result: 11–595 is 585 lines / 57 264 B (90 of those lines dated). 638–726 is 89 lines / 9 110 B. Line 10 is 7 093 B, of which 1 645 B come before "Before it:". The whole file is 726 lines / 79 023 B.
- My definition of "dated history": a section or item whose subject is a completed, dated leg (R353…R366, the 09-15 resume, the Holds, the A/B), as opposed to where the run is now.
- On that definition, 674 of 726 lines (92.8 %) and 71 822 of 79 023 bytes (90.9 %) are history. The scout's "91 %" is the byte figure.
- The span includes the 8-line §"Protected set, laws, cards". Its law-count sentence is current, but its cards half is the R353 leg's history. A rewrite keeps about one line of it, so the net stays about −673.

S-A-DOCS-1-02 — AMENDED.
- Confirmed:
  - "run7 is LIVE" opens §Dispatcher state, while item (8) records "run7 STOPPED 2026-09-18".
  - `grep -c '^# delta:'` gives run7 35 against the heading's 41, and run10 44.
  - The comment ratchet in §Exit facts is 3 451 / 13 387 / 1 530 / 2 677. `comment_length_floor.txt` has 3385 / 13079 / 1485 / 2676. The stated numbers exceed the down-only floors, so they are stale.
  - The header's "reads as of 2026-09-15" does not hold, because the items run to 2026-09-21.
- Changed: the §Exit facts line reads "4 988 collected against the committed floor 4 862". Its floor equals `test_count_floor.txt` (4862), so it agrees with the tree. Only the collected count is dated (5 112 per item 19), and it cannot be re-derived here because torch is absent.
- Also: "the Protected-set line is fine" holds only for the law count.

S-A-DOCS-1-03 — CONFIRMED.
- Command: an awk splitter (one block = one top-level `- **` bullet up to the next blank line or `## `), with each block's first status word taken from anywhere in the block, not only the headline.
- Result: over the scout's 33 names it gives 313 lines, after dropping two false pattern matches (CARD-CHECKER-THREAD-GIL and the RQ-7 row).
- A headline-only scan finds 30 blocks. That is a different set, which shows the classification is a reading, not a regex.
- Possible omission: CARD-E1-RULER-R6 (21 lines) carries a "READ 2026-09-21", and STATE records that its cell ran. Whether to sweep it is the ruling's call.

S-A-DOCS-1-04 — AMENDED (the count only).
- Forward check: for each of the 7 markers, `git grep -l -F` over src, tests, tools, crates, configs, pyproject.toml and Makefile. Only MINT-RESOLVE-PARENT-CONJUNCT hits. The Δ of −9 (2+2+2+1+1+1) re-reads correctly off the section.
- Reverse check: `git grep -h -o 'CARD-…'` over the code tree, each token then tested with `grep -F` against CARDS.md. It finds exactly the scout's 11 names. The claim's "10 markers … 4 of those 10" should read 11: 7 are named in RULINGS and 4 in no governance doc.
- A truncated token, CARD-PREFLIGHT-CHILD, also sits in tests/tools/test_preflight_pfc_cards.py.

S-A-DOCS-1-05 — CONFIRMED.
- Commands and results:
  - `grep -E 'const|static'` for MAX_CHILDREN_PER_NODE gives `crates/mantis-search/src/mcts/mod.rs` = 1024. The CARDS run6 hold says "= 192".
  - `ls configs/run9.yaml` gives no such file.
  - The OC7 headline still reads "BLOCKING", while its body reads "DISCRIMINATED … HOST, not code".
  - `git grep -n PYRIGHT-STRICT pyproject.toml` finds the marker on a different line from the one cited.
  - The run6 section, from its header to the next header, is 36 lines.
- Two notes on (iv):
  - "(`F-01`..`F-52` at this writing; derive the last row from the file)" dates itself and tells the reader to derive the last row, so it is dated rather than wrong.
  - The "1661 lines" matches `wc -l` of the archived AUDIT-2 at HEAD. It is a transcribed count, but a correct one.

S-A-DOCS-1-06 — CONFIRMED.
- `git grep -i 'aggregate_cluster_values_min\|MINPIN\|min_pin'` over crates, src, tests and tools gives 0 hits.
- registry.toml has `value_pool = "none"` in both rows. `ValuePool::Min` still exists as a legal spec variant, but nothing uses it.
- F-43's annotation cites run.py:528 and 538. `git grep '_DeferredSink()' src/mantis/run.py` puts the two sites elsewhere in the file.

S-A-DOCS-1-07 — CONFIRMED.
- Command: `grep -o '^### R[0-9]+' | sort -n | uniq`, compared with `comm` against `seq 23 367`.
- Result: 344 headings over 343 distinct numbers; only R227 and R228 are missing. For R23–R345 there are 322 headings over 321 numbers; the one duplicate number is R279 and its R279(g)-ANNEX.
- The coverage note's own premise ("every number from R23 to R345 … except R227 and R228") also yields 321, not the "322 numbers" it states. So the note contradicts itself as well as the tree.
- (ii) `ls docs/audits/archive/` shows AUDIT_2026-09-09.md only there, while R346's Status line cites the pre-move path.
- (iii) R271's block ends "Status: standing [INLINE]". The only other R271 mention in RULINGS is R272's ratification heading. No amendment note exists.

S-A-DOCS-1-08 — CONFIRMED.
- `git grep -l COMMS_STYLE` gives: the file itself, the two frozen archive files, and docs/slim. CLAUDE.md does not name it.
- Gate 10's `GLOB_SCOPE` sets a floor of 4 `.md` files for docs/governance. Deleting this file leaves 5, so gate 10 stays green.

S-A-DOCS-1-09 — CONFIRMED.
- `git grep -l -F AUDIT_2026-09-11 -- . ':!docs/slim'` gives 0 files.
- By name, "AUDIT-3" appears only in CARDS, RULINGS, the frozen AUDIT-2 and the file itself.
- `wc -l` gives 272.

S-A-DOCS-1-10 — AMENDED (the count only).
- Command: every `docs/…\.md` token extracted from the frozen audit, each checked with `git ls-files --error-unmatch`.
- Result: 6 are missing. They are the scout's 4 plus two top-level docs/ spec files that the scout's subdirectory-only pattern could not match.
- The keep-frozen verdict and Δ0 are unchanged.

S-A-DOCS-1-11 — CONFIRMED.
- (i) An AST read of EvalBrokenReason gives 8 members, ABANDONED included. "seven" appears at eval_instrument.md (twice) and in event_manifest.md's rc-48 paragraph.
- (ii) `git grep -w build_candidate_player` also finds tools/ladder/backends.py.
- (iii) The doc's `_UniqueKeyLoader` is not gone. It is now the public `UniqueKeyLoader` in `src/mantis/util/yaml_io.py` (renamed and moved), so the repair is a rename in the table.
- (iv) An AST read of `ARCH_KINDS` gives 3 kinds.

S-A-DOCS-1-12 — CONFIRMED.
- Command: a yaml load of producer_manifest (16 ids) against a regex over the doc's §"Shipped rows (WP13-A)" table.
- Result: the same 6 ids appear in the doc but not in the manifest.
- `resolved_config` is absent from the whole doc. The other manifest ids missing from that table (heartbeat.*, actor_lag, target_integrity_counters, terminal_eval_broken, warn.*) are named elsewhere in the doc.
- tests/monitor/test_manifest_contract.py and tests/monitor/test_monitor_census.py read the yaml, never the doc.

S-A-DOCS-1-13 — CONFIRMED.
- Command: `git grep -n -w <key> -- src` for all seven roster names, plus a dict-literal and subscript writer regex.
- Result: the only hits are in src/mantis/train/events.py, all of them reads or payload assembly. No writer exists. The finding moves with S-A-CORE-1-02.

S-A-DOCS-1-14 — AMENDED.
- The counts are right: 37 rows and 59 946 B.
- The risk the scout named is not the real one. Gate 13 READS this table:
  - `_TABLE_VERSION_RE` parses the `| vN |` rows.
  - The gate reds on "no version-table rows parsed".
  - It also reds when the `- version: vN` header differs from max(row), with the message "the table is the authority — a row lands when a version does".
- A collapse to one pointer line reds gate 13 unless the latest `| vN |` row is kept. Shortening the rows keeps it green. Either way it amends the stated design of a gate: a gate-13 design choice under lane C, not a doc trim.

DEFECT (gate 17 / STATE) — CONFIRMED in kind, with a different count.
- `rule7_gate.py --full-tree` gives rc 0, "no host content". It also prints loudly that the operator-term supplement is absent and that the scan ran at the TRACKED FLOOR only.
- My own kind-only scan of STATE finds 21 lines (the classes overlap):
  - box-local absolute run-tree paths: 11
  - a port-forward or remote-shell invocation: 2
  - the rental provider's name or host alias: 5
  - process ids: 6
  - an instance id: 1
- The scout's 15 draws a narrower class boundary. Most of these lines sit inside the history spans that 01 would remove.

Handoff PZ-6: CONFIRMED. `git grep -l -F CARD-EXEMPT-CONFIGS-OPERATOR-CONFIRM` over the code tree gives 0.

### Missed by the scout
NEW-1 | DOC | C — the RULINGS §Coverage note says "Four entries record an absence rather than a decision", then names five: R24, R29, R32, R33 and R267 (`sed -n 30,36p RULINGS.md`). Δ0; the fix is an annotation.

### Tally: raised 14 | confirmed 10 | amended 4 | refuted 0 | pending 0 | architect 0 (+1 NEW)

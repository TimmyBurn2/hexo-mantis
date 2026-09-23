# S-A-DOCS-2 — D2 docs/governance/archive (frozen)
scope: docs/governance/archive/** (4 files); one citation row each for docs/design/archive and docs/audits/archive (owned by D3 / D1).
method: `wc -l`, `git cat-file -s`, `git grep -c/-l/-n -F` for each basename and path over the whole tree (excluding docs/slim), header reads (`head -25`), `sed -n` spot reads at the cited coordinates only, `python3 tools/ci_gates/check_tracked_refs.py` (gate 10), `python3 tools/ci_gates/rule7_gate.py --full-tree` (gate 17). The files were NOT read end to end.

## Summary
- Findings: 2, both lane C. PACK/C x2 (one retirement ARCHITECT question, one KEEP-frozen verdict for the rest of the pack). DOC: 0 in slice, and 2 go to HANDOFF.
- Top by Δlines: S-A-DOCS-2-01 -7722 (ARCHITECT question, with a blocker named); S-A-DOCS-2-02 0 (KEEP).
- Line coordinates into the archive from live docs: 33 coordinates on 30 lines, all in docs/governance/CARDS.md (`A:` / `R:`). **33/33 resolve at HEAD** (each checked by `sed -n` at the line ±3 for the subject token or the ruling). No DEFECT. CLAUDE.md, STATE.md, LAWS.md and README.md cite the archive by path only, never by line.

### Per-file table (docs/governance/archive)

| path | lines (`wc -l`) | bytes | citing files (by path, outside the archive and docs/slim) | readers (gate/test/tool/code) | verdict |
|---|---|---|---|---|---|
| docs/governance/archive/README.md | 39 | 2 318 | 0 by file path. The directory `docs/governance/archive/` is cited by 6 files: CLAUDE.md, README.md, docs/design/repo_design.md, docs/governance/{CARDS,LAWS,RULINGS}.md | none | KEEP-frozen. R346(e) says "one directory, one README, no tooling" |
| docs/governance/archive/RULINGS_ACTIVE.md | 7 722 | 701 549 | 1: docs/governance/CARDS.md (the path at "Reading the identifiers", plus 30 `A:` coordinate lines). Frozen citers: docs/audits/archive/AUDIT_2026-09-09.md x4, at the pre-move path | none | ARCHITECT question (S-A-DOCS-2-01) |
| docs/governance/archive/rulings_register.md | 9 349 | 760 912 | 2: docs/governance/RULINGS.md (header + annotations-inventory `Source:`) and docs/governance/CARDS.md (path + 4 `R:` coordinates). Frozen citer: AUDIT_2026-09-09.md x1, at the pre-move path | none | KEEP-frozen. The only verbatim text of R23–R345 (S-A-DOCS-2-02) |
| docs/governance/archive/laws.md | 106 | 8 437 | 3: CLAUDE.md ("Laws digest (full text: …)"), docs/governance/LAWS.md (header), docs/governance/CARDS.md (CARD-CLAUDEMD-REPOINT). RULINGS.md also names the bare basename twice (the LAW-10 annotation) | none | KEEP-frozen. The earned mechanism per law, and the R345(e) LAW-10 annotation (S-A-DOCS-2-02) |
| docs/design/archive/** (8 files, D3) | 1 057 | — | 3: docs/design/repo_design.md, docs/governance/CARDS.md, docs/governance/STATE.md | none | owned by D3. Recorded as a count only |
| docs/audits/archive/** (2 files, D1) | 1 666 | — | 2: docs/governance/CARDS.md, docs/governance/STATE.md | none | owned by D1. Recorded as a count only |

Evidence for the table:
- Sizes: `git ls-files docs/*/archive docs/governance/archive | xargs wc -l` returned 39 / 7722 / 106 / 9349, with a total of 17 216. Bytes came from `git cat-file -s`.
- Path citers: `git grep -c -F "governance/archive/<name>" -- . ':!docs/slim'` returned: rulings_register RULINGS.md:2, CARDS.md:1; RULINGS_ACTIVE CARDS.md:1; laws.md LAWS.md:1, CARDS.md:1, CLAUDE.md:1; README 0; the directory RULINGS 3, CARDS 3, LAWS 1, repo_design 1, README.md 1, CLAUDE.md 1.
- Readers: `git grep -n -i archive -- tools tests src crates Makefile .github pyproject.toml` returned only unrelated hits (`archive/grid-path` tag, npz/zip "archive"). `git grep -n -E "rulings_register|RULINGS_ACTIVE|archive/laws" -- tools tests src` returned 0. `git grep -n "docs/governance" -- tests tools src crates` returned only test_gate_vacuity.py, which asserts that LAWS/CARDS/STATE are scanned, plus gate 10's own scope.
- Gate 10 does NOT scan the archive. `tools/ci_gates/check_tracked_refs.py::GLOB_SCOPE` = `{"docs/contracts": 5, "docs/governance": 4}`, and `Path(directory).glob("*.md")` is non-recursive. RULINGS.md is in `SCAN_EXEMPT`. The run returned `gate 10: scanning 17 file(s)`, rc=0. So the stale paths inside the archive (for example `docs/registers/falsified.md` in laws.md) never red, and deleting any archive file whose path CLAUDE.md, LAWS.md or CARDS.md names WOULD red gate 10.
- Gate 17 scans archive content but does not read or consume it. `rule7_gate.py --full-tree` returned "no host content in the tracked tree (1041 text file(s) …)".
- Gate 6: the two registers are 76% / 70% of the 1 MB add ceiling. Both are frozen, so they cannot grow.
- Freeze integrity: `git log -- docs/governance/archive` shows no edit inside this checkout's history. The clone is shallow (`--is-shallow-repository` true, 54 commits back to 2026-09-19).

## Findings

### S-A-DOCS-2-01 | PACK | C
subject: docs/governance/archive/RULINGS_ACTIVE.md
claim: ARCHITECT question: the archive README says this derived index is "superseded in every function" (laws by LAWS.md, values by STATE.md, cards by CARDS.md, texts by RULINGS.md), and its one live citer is CARDS.md. Should a ruling amending R346(e) retire it from the tree, after CARDS.md's `A:` cites are re-pointed? Two blockers apply first: (1) three OPEN CARDS rows, F-816-34/35/36, are recorded ONLY here. (2) The file is frozen by R346(e).
evidence: `git grep -c -E "\b[AR]:[0-9]{2,5}\b"` → only docs/governance/CARDS.md:30 lines. 27 of those 30 lines also carry a ruling id beside the coordinate (for example "R343(a); A:1927-1939"), which resolves in canonical RULINGS.md. The 3 without one are the F-816-34/35/36 rows (ruling column "none"). `grep -c -F F-816-3{4,5,6}` gives register 0/0/0, RULINGS.md 0/0/0 and ACTIVE 3/2/2. Their only fuller source, `plan/ADJUDICATION_QUEUE.md`, is not tracked (archive README §"Sitting records and plan/"). File shape: `grep -n "^## "` shows lines 1–1294 are a `#`-comment version log (v1.1…v3.67), and §8 curation log runs from line 4097 to the end.
callers: n/a (not a DEAD claim). Every path citer is listed in the table.
Δlines: -7722 (`wc -l docs/governance/archive/RULINGS_ACTIVE.md`). CARDS.md stays net ~0: coordinates are swapped for ruling ids, and the three F-816 rows would need their text carried into CARDS first, so it could grow by a few lines.
witness: gate 10. CARDS.md is in its glob scope and names `docs/governance/archive/RULINGS_ACTIVE.md` under "Reading the identifiers", so it reds if the file goes before the cite does.
depends: —

### S-A-DOCS-2-02 | PACK | C
subject: docs/governance/archive/{rulings_register.md, laws.md, README.md}
claim: KEEP-frozen, with no slimming lane. rulings_register.md is the only in-tree verbatim text of R23–R345 (its last header is R345; RULINGS.md carries condensed entries of 10 lines or fewer by R346(e), and 321 of them fall in R23–R345). laws.md holds each law's earned mechanism and the R345(e) LAW-10 annotation, which LAWS.md points to by path. README.md is the freeze statement that R346(e) mandates. None of them has a tool, test or code reader.
evidence: `grep -o -E "^#{2,3} R[0-9]+" RULINGS.md | … awk '$1<=345' | sort -un | wc -l` → 321. `grep -n -E "^#{1,2} R[0-9]+" rulings_register.md | tail -1` → R345. R271(a) in RULINGS.md reads "append-only VERBATIM ARCHIVE — never compressed, rewritten or pruned". R346(e) reads "The old register, ACTIVE, sitting records and plan/ are FROZEN into docs/governance/archive/ (one directory, one README, no tooling)". The annotation positions that RULINGS.md "Annotations inventory" states (12 under R282, 7 under R308, 8 under R310, 13–14 under R336) all match, checked by awk for the enclosing `# R` header of each ANNOTATION line.
callers: n/a (not a DEAD claim).
Δlines: 0
witness: gate 10 (CLAUDE.md, LAWS.md and CARDS.md name laws.md; CARDS.md names rulings_register.md)
depends: —

## DEFECTS
none in the live-doc → archive coordinates. All 33 of CARDS.md's `A:`/`R:` coordinates resolve at HEAD. Three matched by content rather than token and are correct: A:7687 is the leg-5 paragraph ("SEVEN OF EIGHT LEGS …"), A:7718 is the "OWED … `supervisor_kill_grace_sec` reading" line, and A:1927-1939 is the R341(b) block with its G=8 discharge.

## PARKED
none

## HANDOFF
- D4 (CLAUDE.md): the heading "Laws digest (full text: docs/governance/archive/laws.md)" labels the frozen pre-R346 register as the law's "full text". That file still carries LAW-10 (DELETED by R347(d), per LAWS.md) and the dissolved path `docs/registers/falsified.md`. CLAUDE.md's own opening line and R9 name docs/governance/LAWS.md as the law text. This is lane C, because CARD-CLAUDEMD-REPOINT closed with this naming. ARCHITECT question: relabel it "standing text LAWS.md; earned mechanisms archive/laws.md"?
- D1 (docs/audits/archive/AUDIT_2026-09-09.md, frozen): it cites the pre-move paths `docs/governance/RULINGS_ACTIVE.md:7197-7210` (x3), `:7120-7124`, `docs/governance/rulings_register.md:3519` and `docs/registers/laws.md:55-60`. The ranges give 0 hits for the cited facts. "1 581 steps/h" sits at ACTIVE:7260/7323, and "RTX 5080" at ACTIVE:7710. This is frozen text, so any fix is annotation-only.
- D1 (docs/governance/RULINGS.md, R346 `Status:`): cites `docs/audits/AUDIT_2026-09-09.md`, which is now under docs/audits/archive/. RULINGS.md is in gate 10's SCAN_EXEMPT, so no gate catches this, and any fix is annotation-only.

## Not covered
- The archive bodies were not read end to end, by instruction, so they were not checked for internal consistency. That is frozen text, and it corrects only by annotation.
- Git history before the shallow boundary (2026-09-19) was not examined, so freeze integrity is shown for this checkout's window only.
- Whether full remote history keeps RULINGS_ACTIVE.md recoverable after a retirement was not verified, because the clone is shallow.

## Review
reviewer: fresh read-only agent (not the author); no lane-A deletions raised, so no delete-probe and no worktree was created
| ID | verdict | lane | Δlines | note |
|---|---|---|---|---|
| S-A-DOCS-2-01 | AMENDED | C | -7722 (awk NR count; unchanged) | the ARCHITECT question stands. "Its one live citer is CARDS.md" is wrong: standing RULINGS.md::R271(b)–(d) (and R272(b)) name RULINGS_ACTIVE.md as the working index that is curated at session close, so a retirement ruling must annotate R271 as well as amend R346(e). The coordinate split is 26 + 1 + 3, not 27 + 3 |
| S-A-DOCS-2-02 | CONFIRMED | C | 0 | KEEP-frozen. R271(a) and R346(e) are verbatim at HEAD. 321 of R23–R345, and the register's last header is R345 |

### Per-finding notes
S-A-DOCS-2-01 — AMENDED:
- Sizes: `awk 'END{print NR}'` + `stat -c %s` over `git ls-files docs/governance/archive` -> 39/7722/106/9349 lines. Bytes match the table.
- Citers: `git ls-files -z | xargs -0 grep -l -F RULINGS_ACTIVE` (excluding slim/archive) -> CARDS.md:1, AUDIT_2026-09-09.md:4 AND RULINGS.md:1. The RULINGS.md hit is R271 ("Status: standing [INLINE]"), and (b) reads "RULINGS_ACTIVE.md is the DERIVED working index; sessions seed from ACTIVE + laws.md + CLAUDE.md". (d) says ACTIVE is curated at session close. R272(b) extends that curation practice. R346(e) names "ACTIVE" in its freeze list. So the subject is ruling-named twice, and `grep -n R271` shows no annotation recording the supersession. Lane C holds.
- Coordinates: a python sweep of CARDS.md with `\b[AR]:\d{2,5}(-\d+)?` -> 33 coordinates on 30 lines. Every one is in range and resolves by content. I re-read the 7 the heuristic flagged, and each shows the row's subject token at the coordinate. CARDS.md is the only file with coordinates.
- The split: 3 are the F-816-34/35/36 rows (ruling column "none"; `git grep -c` -> CARDS 1 + ACTIVE 3/2/2, nowhere else). CARDS:976 cites A:3901 as "the stale line in the archive" itself, so it is not a swap to a ruling id; it would be deleted instead. That leaves 26 that can be swapped for a ruling id. CARDS:805 and CARDS:876 get theirs from the preceding lines of the same paragraph (R348(e)/R349; R227/R228/R267).
- R267: the register's line "R267 REMAINS A GAP" (rulings_register.md) also carries the only-a-digest-line fact, so ACTIVE is not the sole record there.
- Witness: tools/ci_gates/check_tracked_refs.py::TOKEN_RE matches `docs/governance/archive/RULINGS_ACTIVE.md`, and CARDS.md is inside GLOB_SCOPE's non-recursive `docs/governance/*.md`. The gate 10 run prints "scanning 17 file(s)", with no failures. The witness holds.
S-A-DOCS-2-02 — CONFIRMED:
- `.venv/bin/python -S` regex over RULINGS.md `^#{2,3} R(\d+)` -> 321 distinct ids in 23..345 (max 367). Over the register, `^#{1,3} R(\d+)` -> last = max = R345.
- R271(a) "append-only VERBATIM ARCHIVE — never compressed, rewritten or pruned" and R346(e) "FROZEN into docs/governance/archive/ (one directory, one README, no tooling)" were both read at HEAD.
- `git grep -E 'governance/archive|RULINGS_ACTIVE|rulings_register|archive/laws' -- ':!docs'` -> only CLAUDE.md:94 and README.md:38, both prose. No tool, test or src reader.
- Gate 6: tools/ci_gates/artifact_gate.py::MAX_ADDED_BYTES = 1_000_000, so the files are at 76.1% and 70.2%, as the scout said.
HANDOFF check:
- CLAUDE.md:94 "Laws digest (full text: docs/governance/archive/laws.md)" is confirmed.
- archive/laws.md still has LAW-10 x1 and `docs/registers/falsified.md` x2, while LAWS.md reads "LAW-10 DELETED by R347(d)". CARD-CLAUDEMD-REPOINT is CLOSED (CARDS.md).
- The R346 Status cites `docs/audits/AUDIT_2026-09-09.md`, which `git ls-files docs/audits` shows only under archive/. All three handoffs stand.

### Missed by the scout
NEW-1 | DOC | C | subject: docs/governance/RULINGS.md::R271(b)–(d) (+R272(b)) | claim: this standing text directs sessions to seed from ACTIVE and curate it at session close, but R346(e) froze ACTIVE, and no annotation records that it was superseded | Δlines: 0 (annotation-only, register corrects only by annotation) | ARCHITECT: annotate R271(b)–(d) as superseded by R346(e), and name it in any ruling from S-A-DOCS-2-01?

### Tally: raised 2 | confirmed 1 | amended 1 | refuted 0 | pending 0 | architect 1 (S-A-DOCS-2-01, plus NEW-1)

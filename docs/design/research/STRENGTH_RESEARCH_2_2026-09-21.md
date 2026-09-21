# STRENGTH RESEARCH 2 — PACKET RESEARCH-STRENGTH-2 (2026-09-21)

Multi-agent packet on ONE question: WHAT MOVES STRENGTH PER BOX-HOUR. Orchestrated from worktree
`research-strength-2` at HEAD `649852ddd7a9e367de3b0a00b87fc86297815824` (`dev` at packet start);
14 subagent runs (P1 research, P2 review, five briefs, five rebuttals, audit, judge) + this
orchestrator. No repo file changed; this file is the deliverable. Nothing in it is a decision.

**Reading order for the architect:** the ONE SCREEN below → P6 (judge) → P5 (audit) → the DISSENT
inside P6 → then the phases in order. Phase artifacts are inlined verbatim from the working files
(they carried the packet's own tagging discipline; untagged claims were struck by P2/P5 and the
strikes stand in the text).

**Record drift during the packet** (P5 caught it): the untracked freshest record
`PARAM_DISTANCE_2026-09-21.md` (main checkout, NOT at worktree HEAD) re-hashed mid-packet
`4c6759b9cbe2eec4…` → `60ca27c41b3ab2a2…` — the run9 mint session edited it while this packet ran;
briefs citing the first sha read the earlier text. `RUN9_PREREG §3`'s SUCCESS ≥ 0.192 /
FALSIFIED ≤ 0.142 lines exist only in that same uncommitted rewrite (P5's worst mis-tag). The
pack's premises section flags three more: `handoff §5/§9` = NOT FOUND as a file (STATE.md is the
handoff, per RULINGS); R364 verbatim is not at HEAD; `configs/run9.yaml` unminted at HEAD.

**Re-pin, 2026-09-21 (R365 §0(6), by the landing session; one line, the phases below unedited):**
`docs/design/measurements/PARAM_DISTANCE_2026-09-21.md` is COMMITTED at `203e7060` and hashes sha256
`60ca27c41b3ab2a20a34474d3b058e051813e7178e1afbce6c778577bf800905` at HEAD; every `4c6759b9cbe2eec4…`
cite below is the pre-commit draft, content verified by P5's spot-checks (L7). `configs/run9.yaml` is
minted at `61bd2fd1` and R364 is in the register at `b5a5f51e`; both were in flight when this packet ran.

**HALT recommendation to the architect: NO** (P6's verdict, MEASURED STATE item 15 — full-span
draws 0.5 % / ply-cap 0.53 % are measured-inert under the only "rows are wrong" candidate; better
rows exist, wrong rows do not).

---

## ONE SCREEN — P6's ranking (verbatim; full entries in the P6 section)


is the run itself — held at bottom of builds).

| # | name | grp | Δ/bh (order · box-h · claimed Δ) | conf | carrying tag | falsifier | pre-START |
|---|------|-----|-------------------------------|------|--------------|-----------|-----------|
| 1 | E4 ruler-resolution | PREMISE | high · 0 bh · what the ±3.9-pp cell can/can't certify | high | MEASURED PD:48 CI+slopes (P5 PASS) | is the check (0 bh) | YES — mint in flight |
| 2 | E3 mr-calibration | PREMISE | high · 0.5 bh · ≥2× mr=1 error ⇒ named target for B/C2/A | medium | MEASURED F-45 (25/25 @ mr==1) | 0.5 bh census | YES |
| 3 | C3 probe→size | BUILD 2-stg | high · stg1 0 bh · decisive either way; stg2 55 bh, ≥0.161 | medium | MEASURED-absent §Q3(b) + PD walk | stage 1 IS it (0 bh) | YES (stage 1) |
| 4 | P-B2 proof-as-target | BUILD | med-high · 0-bh falsifier; build 22 bh, 45k ≥0.18 | medium | MEASURED CARD A1 + F-38/39/40 | 0 bh offline | YES (beside run9) |
| 5 | E1 ruler-r6 | PREMISE | med-high · 1.2 bh · unit-stability of EVERY bar | medium | MEASURED §Q7.1.2 + driver:67 | the cell (1.2 bh) | rec: preflight; not gating |
| 6 | P-B1 soft-policy | BUILD | medium · KL 0 bh gates; build 20 bh, 45k ≥0.18 | medium | MEASURED STATE 13–15 h_full + SOURCED KataGo T2 | 0 bh KL | falsifier now; build run11+ |
| 7 | A-P2 EMA probe→row | KNOB | medium · probe 1.5 bh; row 17 bh; "free strength" ≥0.192-class | medium | MEASURED PD §2 cos 0.15→0.03 (sha stale, P5 L7) | 1.5 bh (none <1) | beside run9 |
| 8 | D-1 fill-at-knee | KNOB | medium · 0.75 bh · fill 72→knee ⇒ steps/h (gated premise) | medium | MEASURED PERF3 fill/queue_wait | 0.2 bh bench | YES — step-3 window reserved |
| 9 | P-B3 sims 128 | KNOB | med · bench 0.2 bh; cell 19 bh · ×1.6–1.8 games/h, wall ≤0.85× | low-med | SOURCED §1b strix-128 (P5 #4) + MEASURED F-47/PERF3 | 0.2 bh bench; 3-bh twin | YES (bench, preflight) |
| 10 | D-2 edge-cutoff | BUILD | low-med · 0.2 bh · D-1's conversion, device-side term | low | MEASURED PERF3 §6 buckets; transfer ARGUED | 0 bh regression (weak, ±6%) | no — waits step 3 |
| 11 | A-P1 LR floor | KNOB | medium · 17 bh (floor est.) · slope restore, 15k ≥0.192 | medium; bar ARGUED (P5 #1) | MEASURED PD §2 noise stats | none <1 bh; 0-bh control | no — run10's (R364(d)) |
| 12 | C2 value-scalar | BUILD | low-med · 18–20 bh/15k, ≈50 equal · ≥0.161 | low-med | MEASURED §Q7.2 diff (regime-confounded) | 0 bh probe = E3 bucketed | no — run11+ |
| 13 | P-B4 λ target | BUILD | low-med · 22 bh/15k · 45k ≥0.18 + loss <0.50 | medium; no cheap falsifier | MEASURED PD §2 value ‖Δ‖+norm | none <1 bh offline | no |
| 14 | D-3 off-box eval | BUILD | low-med · money + 0.5 bh · +15–25% steps/h (ARGUED) | low-med | MEASURED EVAL_COST 51%/rates | 0.2 bh — but needs the card | no |
| 15 | C1 arch-D6 | BUILD | low · 50–55 bh · ≥0.161; removes equivariance noise floor | low | MEASURED spread 0.146 n=1; tie UNMEASURED (P5 2d) | 0 bh conformance + series | no |
| 16 | E2 cap-census | PREMISE | low · 0.2 bh · refinement only (headline answered) | LOW (P5 #2 struck) | MEASURED STATE item 15 (0.5%/0.53%) | 0.2 bh census | optional, pre-series-read |
| 17 | A-P3 batch-512 | KNOB | low · 17 bh · same samples, half steps; slope SPECULATION | low | SOURCED parity §Q7.2 + F-44 (run6 regime) | ≈10-min memory smoke (arm only) | no |

**HALT: NO** — MEASURED(STATE item 15, 3b597edc): 53 084 full-span games, "draws 0.5 %, ply-cap
0.53 %", under E2's own ≤3 % bar: the only "rows are wrong" candidate (the armed −0.5 rows) is
measured-inert; all else is better-rows (LR = R364(d), run10's) or unread cost, not a wrong row.
**NEXT BOX-HOUR:** (1) E4 — free, deadline = the in-flight mint; (2) E3's one mirror pull (0.5 bh)
carrying B's decomposer + C2's calibration + A-P2's mr partition; (3) the 0.2-bh bench pair in
run9's reserved preflight window — bench_server at 128 sims (P-B3's pre-check) + the B-curve knee
(D-1's falsifier / CARD-PERF-3 step 3). ≈0.7 box-h total; E1 (1.2 bh) if the window allows.


---

## §0 RULES (the packet's own text, binding every phase)

- Tag every claim: MEASURED (path, sha) · SOURCED (URL/repo path, the source's own sentence, date
  read) · ARGUED · SPECULATION. Untagged → struck by the reviewer. "The literature says" without
  a sentence = struck.
- falsified.md F-01…F-52 is read first; nothing on it is re-proposed without naming the NEW
  measurement that would reopen it. Regime of every number stated (game, net, compute, unit).
- A proposal is ONE swap: rows or a build behind the seam; cost in box-hours to first reading;
  success line at EQUAL GAMES vs a named parent; a falsifier that costs < 1 box-hour if one
  exists, and a statement if none does.
- Progress counts: expected Δ strength per box-hour, not Δ strength. A 1.3× on throughput ranks
  with a knob.
- Question premises, including ours: name any statement in the record you could not find a
  measurement for.

§2 PROPOSAL TEMPLATE: name · thesis (one line) · evidence (tagged lines) · the swap (rows or
build, verify-at-HEAD keys) · cost to first reading (box-hours, dev-hours) · success line at
equal games vs named parent · falsifier < 1 box-hour (or "none; a run is the only test") · what
kills it · interactions with run9's rows.

---

## P0 — CONTEXT PACK (as every subagent received it)


# P0 CONTEXT PACK — PACKET RESEARCH-STRENGTH-2 (2026-09-21)

Read this file fully, then read the files it names. No summaries of summaries: every item below
IS the file, in the worktree `/home/tom/Work/HeXO/hexo-mantis-rs2` (branch `research-strength-2`,
HEAD `649852ddd7a9e367de3b0a00b87fc86297815824`, "ours at HEAD" = that tree). The 16-hex prefixes
are sha256 of each file's content at that HEAD. Phase outputs go to `/tmp/rs2/` — you write ONLY
your own assigned file there; this repo is READ-ONLY to you.

## §0 RULES of this packet (bind every agent, every phase)

- Tag every claim: MEASURED (path, sha) · SOURCED (URL/repo path, the source's own sentence,
  date read) · ARGUED · SPECULATION. Untagged → struck by the reviewer. "The literature says"
  without a sentence = struck.
- falsified.md F-01…F-52 is read first; nothing on it is re-proposed without naming the NEW
  measurement that would reopen it. State the regime of every number (game, net, compute, unit).
- A proposal is ONE swap: rows or a build behind the seam; cost in box-hours to first reading;
  success line at EQUAL GAMES vs a named parent; a falsifier that costs < 1 box-hour if one
  exists, and a statement if none does.
- Progress counts: expected Δ strength per box-hour, not Δ strength. A 1.3× on throughput ranks
  with a knob.
- Question premises, including ours: name any statement in the record you could not find a
  measurement for.

## §2 PROPOSAL TEMPLATE (every proposal, every brief)

name · thesis (one line) · evidence (tagged lines) · the swap (rows or build, verify-at-HEAD keys)
· cost to first reading (box-hours, dev-hours) · success line at equal games vs named parent ·
falsifier < 1 box-hour (or "none; a run is the only test") · what kills it · interactions with
run9's rows.

## THE GAME (verify at HEAD, do not trust prose)

`README.md` (2c05982c2d0a60ad) — hex grid, 6-in-a-row to win, unbounded board, compound 2-stone
turns. `docs/design/repo_design.md` (7ed05270a7b25ef6) — the structural contract; its §on
measurement units. Rules as code: `crates/mantis-core/src/` (rules, Ply/Turn vocabulary);
LAW-03: a compound turn places TWO stones; turn vs ply is a founding unit — plies here are
single placements unless a line says "compound turns". Legal set = union of hex-radius-8 balls
around stones (fence); ≈ 355 legal moves at radius 8 (F-51). 256-ply cap → `ply_cap_value`.

## THE CONFIG SURFACE

- `configs/run8.yaml` (f600bfb412325b14) — the RUNNING config (run8, live since 2026-09-18):
  Gumbel self-play 320/64 p 0.25, m 16, c_visit 50, c_scale 1.0, `q_rescale: false`, raw q;
  PUCT-256 deploy/eval; AdamW 1e-3 → 5e-4 cosine over 1 M steps; batch 256; ring 100 000;
  `training_steps_per_game 1.0`; `augment: true` (armed R358); warm start run7@42k; 65-bin
  value over pure outcome z; `ema.enabled: false`.
- **`configs/run9.yaml` is NOT MINTED at HEAD** (the mint is in flight beside this packet; the
  main checkout `/home/tom/Work/HeXO/hexo-mantis` carries uncommitted mint work — READ-ONLY to
  you). What run9 IS, at HEAD: `docs/design/measurements/RUN9_PREREG_2026-09-19.md`
  (3e40594afb6a7320) — eval rows FIXED (eval_interval 15000 on the strix cell's checkpoint, sealbot
  rung DELETED, gate 256/GSPRT unchanged, gate fields on the stream), parent BLANK ruled at 45k,
  §1c the pre-stated PARENT RULE, §5 the sequence.
- run9's ruled shape (operator R364, NOT yet in RULINGS.md at HEAD — cited by the untracked
  PARAM_DISTANCE below, see PREMISES): parent = run8@45k (net `3aef7883…`); the ONE training
  swap = DATA REGIME — `replay_capacity` 500 000 held at replay ratio ≈ 8 (CARD-RUN9-QUEUE (i));
  LR is RUN10's (R364(d)), not run9's.

## THE RECORD (paths relative to the worktree; every file at the shas above)

The run9-relevant state, in reading order:

1. `docs/design/research/STRENGTH_RESEARCH_2026-09-18.md` (699ae9c96483330b) — PACKET 1. §0 the
   Q1–Q11 ranking; §1b the regime table (READ before quoting any outside number); §Q7 strix
   itself (the pinned checkpoint's full record: solver, targets, buffer, LR, curriculum); §Q2
   replay; §Q5 Gumbel; Appendix A the scratch D6 lossless test.
2. `docs/design/measurements/STRIX_RUN7_60K_2026-09-17.md` (2c1507a730843437) — the run7 strix
   series + the 0.78 book ceiling + §D throughput units.
3. run8's series, gate table, ring audits: `docs/governance/STATE.md` (3b597edc1b306bf4)
   "Current phase" + "Dispatcher state" items (8)–(15) — the 15k/30k/45k cells (0.104 / 0.135 /
   0.142 vs parent 0.111), r1–r17 gate rounds (anchor frozen at 15k, r11–r17 all rejected),
   ring audits (`one_hot_share_full` ≈ 0.27 flat vs < 0.25 band; h_full 0.209 → 0.286;
   replay_ratio 3.0–3.6), shakedown8, EVAL-CENSUS prices. And
   `/home/tom/Work/HeXO/hexo-mantis/docs/design/measurements/PARAM_DISTANCE_2026-09-21.md`
   (sha256 4c6759b9cbe2eec4…; UNTRACKED in the MAIN checkout, not at worktree HEAD — the freshest
   record): the 18-checkpoint parameter-distance table (flat ‖Δ‖ ≈ 22/3k steps, cos(Δ,Δprev)
   0.15 → 0.03, weight norm +49 % with value head +96 %, strix + gate columns beside).
4. `docs/design/measurements/EVAL_COST_2026-09-19.md` (b0e7c3de28e069f3) — who consumes
   best_model; rounds cost (51 % of run8's wall in flight; trainer 356–415 steps/h in a cell vs
   ~1000 alone); GSPRT long end; the pre-stated directions.
5. `docs/design/measurements/PERF3_2026-09-18.md` (fbf3377f4d15e77d) — serving decomposition:
   3 208 leaves/s alone at B 46, CPU stage 11.46 of 14.44 ms the bound, 1 222 beside a cell;
   `tools/bench_server.py`; step 3 (box B-curve) reserved for run9's preflight window.
6. `docs/design/measurements/LADDER_SHAKEDOWN_2026-09-19.md` (8fea860176d84da8) — the HeXO
   server shakedown: 0 of 20 = 0 of 2 DISTINCT games, opening-diversity finding, s/turn.
   `docs/design/measurements/CPU_HEAD_PROFILE_2026-09-20.md` (23cca1828889941a) — the CPU
   deploy head: 5.0 s/stone at 256 sims, per-leaf forward ≈ 20 ms flat in batch size, VPS sizing.
7. The ANALYZER symmetry reading: `docs/design/analyzer_design.md` (2f94bc34160f986b) — the
   instrument; the measurement itself is R363(d) in `docs/governance/RULINGS.md`
   (7e21c43ade95b6ea, entry near the file top): symmetry spread 0.146 over the 12 elements at
   run8@18k, translation exact, argmax 12/12 — filed as a GNN-2 witness.
8. `docs/governance/falsified.md` (0ac5043b1fc2355e) — F-01…F-52 + scope annotations. READ FIRST.
   Load-bearing for this packet: F-44 (trainer idle 92 %), F-47 (server-thread bound), F-48/F-50/
   F-51 SUSPENDED (Gumbel head readings; the re-derivation cells), F-52 (PUCT trainer attractor),
   F-38/F-39/F-40 (search-in-the-loop probes), F-15 (expansion-time short-circuit hazard).
9. `docs/governance/CARDS.md` (9a2c84ba3cb61de9) — the queue: CARD-RUN9-QUEUE (the (i)–(vii)
   order), CARD-PERF-3 (B-curve, contended arm), CARD-EVAL-REDESIGN, CARD-STRIX-NET-ONLY (READ,
   A1: the play-time solver is not the gap; the training-side proof-target is UNTESTED); the
   PARKED list under R358(e) (aux targets, net size, curriculum, opponent diversity, teacher).
10. `docs/governance/LAWS.md` (9f31b9741dd8584b) — the seventeen + the protected set.
11. `docs/governance/RULINGS.md` (7e21c43ade95b6ea) — R356–R363 verbatim at the top; R364 is NOT
    at HEAD (see PREMISES).
12. Adjacent, read what your brief needs: `RUN8_PREREG_2026-09-17.md` (15ab2a88431e42bc),
    `RUN7_EVAL_COST_2026-09-15.md` (03ddcd1ae5ae8b8a — the 0.78 book ceiling's derivation),
    `STRIX_RUNG_2026-09-14.md` (c28708a7fda4f19e — equal-work 256/256 cell),
    `STRENGTH_FRONTIER_1_2026-09-13.md` (6ff026abfdb8e8aa — the σ cells),
    `INVESTIGATION1_2026-09-15.md` (cd062e7bbe3e9303 — the Gumbel head residue red team),
    `crates/mantis-encoding/src/registry.toml` (f03c78eb47a6d9a3 — `gnn_axis_r8`, 11 node / 5
    edge features), `src/mantis/model/value_targets.py` (the λ-return codec, landed UNARMED).

## PREMISES THIS PACK FLAGS (per §0 — question premises, including ours)

- "handoff §5/§9" (packet §0 P0 list): NO file named handoff exists at HEAD. RULINGS line "STATE.md
  is the handoff at every step" governs → STATE.md's Dispatcher section is included above; the
  nearest literal "§5" is RUN9_PREREG §5 (the run9 sequence). An operator SESSION_HANDOFF_v6 is
  cited by R545-era rulings and is NOT in the repo. NOT FOUND — flagged, not guessed.
- R364 (the parent + swap ruling) is NOT in RULINGS.md at HEAD; its content is known only through
  STATE item 15 and PARAM_DISTANCE's citations (R364(b) data regime, R364(d) LR run10's). Any
  brief that needs R364's verbatim text says so and cites those paths.
- run9.yaml does not exist at HEAD (mint in flight). Claims about "run9's rows" cite RUN9_PREREG,
  CARD-RUN9-QUEUE and PARAM_DISTANCE, never an imagined config.
- The box (RTX 5080) and the mirror are NOT reachable from this session; run8 numbers are read
  off the record (STATE, PARAM_DISTANCE), never re-derived.

## THE BRIEFS (P3) — one thesis each, template §2 above, ≤ 2 screens each

A DATA & OPTIMISATION — "the knobs are wrong": window, reuse, LR shape/floor, batch, EMA,
  augmentation, warm-start dynamics. MUST address: what run9's reuse-8 will and won't show.
B TARGETS & SEARCH — "what a leaf teaches": σ/c_scale, sims, m, proof-as-target, surprise
  weighting, soft-policy head, λ value target, exploration (Gumbel noise vs visit sampling),
  the 27 % one-hot share.
C NET & ENCODING — "the net is the ceiling": D6 equivariance, JK-cat, width/depth, node/edge
  features, 65-bin vs scalar value, threat features, what the symmetry spread 0.146 means for
  strength.
D SYSTEMS — "learning per box-hour": leaves/s (B-curve, edge-batching, CPU stage), batch fill
  75 %, worker count, eval cost, hardware choice (a second card for eval; a different box),
  what a 2× buys.
E PREMISES — "the target may be wrong": is beat-strix the right ruler; BC warm start vs scratch;
  curriculum by radius; teacher distillation; two-stone turn semantics in the encoding; whether
  value from outcome z can work in a game this tactical; the ply cap / draw utility / fence radius.


*Drift note (added at assembly, after P5): the PARAM_DISTANCE sha above (`4c6759b9…`) was the
file's content when the pack was written; the mint session edited it mid-packet to
`60ca27c4…`. Every MEASURED citation of it in the briefs names the earlier text.*

---

## P1 — GENERAL RESEARCH (web, sourced, dated 2026-09-21)


# P1 GENERAL RESEARCH BRIEF — PACKET RESEARCH-STRENGTH-2 · read 2026-09-21

Web-research brief for P3. Six areas as tasked; every claim tagged SOURCED with URL, the source's
own sentence (verbatim), and read date 2026-09-21. Each source's regime stated in one clause.
Nothing here is a proposal; nothing is about the mantis repo. Packet-1 sources (AGZ/AZ, KataGo,
Gumbel paper, MiniZero, LZ/lc0, ELF, Tablut, SAZ, Polygames, Jones 2021, strix) are NOT re-cited
except where a NEW sentence was fetched for a NEW question.

**Fetch inventory (honesty first).** Worked: arXiv export API + abs pages (all cited IDs returned
HTTP 200 this session), the AAAI open-access PDF, GitHub REST + raw at sha, DuckDuckGo HTML (early
queries; later ones returned an empty result page). Failed: `api.semanticscholar.org` — 429
"Too Many Requests" on EVERY attempt incl. 40–50 s backoff (≈6 tries, 06:27–06:36 UTC);
`api2.openreview.net` — `{"notes":[],"count":0,"searchUnavailable":true}`; `www.connect6.org` —
TLS failure, curl exit 60 (http and https, with and without `-k`). Items those would have carried
are marked below.

---

## 1 · NEW since 2026-09-18: strength-per-compute in small AZ-style systems

**The exact packet window is empty on arXiv.** Queries run (read 2026-09-21):
`cat:cs.LG AND ("AlphaZero" OR "MuZero" OR "self-play" OR "Monte Carlo tree search") AND
submittedDate:[202609160000 TO 202609230000]` → 4 hits, none on strength-per-compute
(Multiplicative Optimism for Constant Regret in Games 2609.21976; Graph-Based Stochastic Power-UCT
2609.19956; Steering Equilibrium Selection in Regularized Self-Play 2609.19820; Online Robust RL
Through Monte-Carlo Planning 2609.18599). A wider `(all:"AlphaZero" OR all:"self-play
reinforcement") AND submittedDate:[202609100000 TO 202609220000]` → 0 hits. NOT FOUND for the
3-day window itself; the items below are the newest on-topic work (2026-06 → 09-08) that packet 1's
source index (its Appendix B) does NOT list.

**1a. Approximate Value Iteration vs AlphaZero at small scale — the strongest new item.**
SOURCED — https://arxiv.org/abs/2609.09094 (v1, published 2026-09-08; Boige, Boumaza, Scherrer),
read 2026-09-21: "Contrary to expectations, our results demonstrate the surprising effectiveness of
AVI: it learns more accurate value functions than those learned by AlphaZero, while its
one-step-lookahead greedy policies remain competitive with MCTS-based policies at substantially
lower training and inference costs." Regime: Connect Four, Hex(7x7), synthetic games (preliminary
Othello, Go(9x9)); "a minimal self-play implementation of Approximate Value Iteration (AVI)";
compute not quantified in the abstract. Relevance flag for P3-B (targets): a non-MCTS target
generator competitive with AZ on a small line/connection game is now on the record — packet 1 had
nothing of this shape.

**1b. WallZero (CG 2026): an AZ agent for a wall-placement game, small board.**
SOURCED — https://arxiv.org/abs/2606.17847 (2026-06-16; Chen, Arjonilla, I.-C. Wu, T.-R. Wu), read
2026-09-21: "We introduce tailored action and feature designs to improve playing performance
significantly." and "WallZero defeats two professional Go players who participated in this study,
securing on average 1.98x more territory per game." Regime: two-player WallGo, 7×7 board, "high
game-tree complexity" from stone movement + wall placement (a compound-ish turn); AlphaZero-based;
compute not stated in abstract. Feature/action-design (not search or net-size) credited for the
gain — the same lever class as CARD-STRIX-NET-ONLY's question.

**1c. A 100+-run "gold-standard" ablation study of a lightweight self-play agent.**
SOURCED — https://arxiv.org/abs/2607.06854 (v1 2026-07-07, v2 2026-08-29; Kelidari, Haghi,
Salmani), read 2026-09-21: "Trust region updates, a well-aimed reward, a curriculum of tougher
opponents, warm starting, and keeping the best checkpoint all help, and stacking them lifts a
self-play champion from about 30 to 36 percent against the expert." and "The result is a
lightweight, game-agnostic recipe that trains competitive agents without training on the expert,
for any game a small model can handle, reported with robust statistics and released as a reusable
package." Regime: Gin Rummy (imperfect-information card game; model-free self-play, NO search),
Leduc Hold'em cross-check, "more than a hundred runs", small nets. Search-in-the-loop AZ is a
different family — the value is the method discipline (fixed expert yardstick, stacked factors,
negative results), and its negative list: "Several ideas did not pay off. Short-term and
longer-term reward shaping, learned state embeddings, imitation and DAgger, and a live large
language model opponent were each unhelpful, too slow, or too heavy to train at scale."

**1d. Searchless chess: exploration regulariser swap, matched-compute sweep.**
SOURCED — https://arxiv.org/abs/2608.27757 (2026-08-27; Miłosz, Duch, Grabowski), read 2026-09-21:
"We replace it with a forward, mass-covering KL toward the network's own MCTS prior (prior-directed
exploration), so exploration covers the moves the prior judges promising" and "a control fine-tuned
on puzzles alone posts the study's largest tactical gains while shedding roughly 260 Elo; a better
puzzle-solver is not thereby a stronger player." Regime: fine-tuning Lc0's released Chessformer
(single GPU-class), ~2000 steps, ratings from test suites + engine play. Relevant to P3-B's
exploration/target-shaping questions (the 27 % one-hot share): the regulariser that anchors
self-play to the search prior beat entropy-style exploration in their matched sweep.

**1e. Flexer: search depth adapted to prior quality (minor).**
SOURCED — https://arxiv.org/abs/2608.15700 (2026-08-16; Rens), read 2026-09-21: "reduce the search
depth proportionally to the quality of the policy network priors" (abstract phrasing of the
objective) and "Flexer outperforms a version of AlphaZero (and DQN and ADP) for some experiments on
three toy symbolic problems." Regime: symbolic toy problems, not board games — included only
because adaptive sim-allocation per position is a compute-per-strength lever; weak evidence.

**1f. Go-Exploit (AAMAS 2023) — NOT in packet 1's index; sample-reuse via start states.**
SOURCED — https://arxiv.org/abs/2302.12359 (v2; Trudeau, Bowling), read 2026-09-21: "In the games
of Connect Four and 9x9 Go, we show that Go-Exploit learns with a greater sample efficiency than
standard AlphaZero, resulting in stronger performance against reference opponents and in
head-to-head play." and "Go-Exploit's sample efficiency improves when KataGo's other innovations
are incorporated." Also on targets: "Producing shorter self-play trajectories allows Go-Exploit to
train upon more independent value targets, improving value training." Regime: Connect Four + 9×9 Go
(one run each, AZ baseline with KataGo cross), MCTS self-play, sample-efficiency metric. Note the
mechanism is start-state reuse from an archive — a cousin of reanalyse (§5) and of strix's staged
curriculum, NOT of a bigger window.

**1g. Project news: KataGo v1.18.0–v1.18.2 (backend work, no method claim).**
SOURCED — https://github.com/lightvector/KataGo/releases (API, read 2026-09-21): v1.18.0
(2026-08-22): "This release adds major optimizations to the CUDA backend, and two new backends -
ROCm for AMD GPUs, and ONNX for various other GPUs/accelerators." v1.18.1 (2026-08-24) "Better
Benchmark/Genconfig, Minor bugfixes"; v1.18.2 (2026-08-30) "CUDA Speedup for Turing (RTX 20xx,
etc)". Regime: inference-engine engineering, not a training-method result. P3-D relevance only.

## 2 · D6 / hexagonal / dihedral-equivariant nets for games; equivariance vs augmentation

**2a. Equivariant MuZero: the equivariance-through-search theorem.**
SOURCED — https://arxiv.org/abs/2302.04798 (2023-02-09; Deac, Weber, Papamakarios), read
2026-09-21: "We prove that, so long as the neural networks used by MuZero are equivariant to a
particular symmetry group acting on the environment, the entirety of MuZero's action-selection
algorithm will also be equivariant to that group." and "We propose improving the data efficiency
and generalisation capabilities of MuZero by explicitly incorporating the symmetries of the
environment in its world-model architecture." Regime: MiniPacman + ProcGen Chaser, rotation groups
on gridworlds, learned model; NOT board games, no equivariant-vs-augmentation Elo number in the
abstract. This is the closest formal statement to "what does a D6-equivariant GNN buy under search"
(packet 1 measured our symmetry spread 0.146 at run8@18k; the theorem says if the net were exactly
equivariant, the SEARCH output is too — that is the GNN-2 hypothesis stated as someone's theorem).

**2b. SLAP: equivariant-style sharing vs augmentation, measured in Gomoku RL.**
SOURCED — https://arxiv.org/abs/2301.04746 (v5; Suen, Alonso; AISB 2023), read 2026-09-21: "SLAP
improved the convergence speed of convolutional neural network learning by 83% in the experiments
with Gomoku game states, with only one eighth of the sample size compared with data augmentation."
and — the honest half — "In reinforcement learning for Gomoku, using AlphaGo Zero/AlphaZero
algorithm with data augmentation as baseline, SLAP reduced the number of training samples by a
factor of 8 and achieved similar winning rate against the same evaluator, but it was not yet
evident that it could speed up reinforcement learning." Regime: Gomoku (a line game, bounded
board), CNN policy nets, AGZ/AZ-family training. This is the ONLY fetched head-to-head
equivariance-family-vs-augmentation number in a line-game AZ setting; note it reaches parity on
8× fewer samples, not superiority, and the authors themselves flag the RL-speedup gap.

**2c. HexaConv: hexagonal-lattice group convolutions (architecture family only).**
SOURCED — https://arxiv.org/abs/1803.02108 (2018-03-06; Hoogeboom, Peters, Cohen, Welling), read
2026-09-21: "Whereas the square tiling provides a 4-fold rotational symmetry, a hexagonal tiling
of the plane has a 6-fold rotational symmetry." and "we find that the increased degree of symmetry
of the hexagonal grid increases the effectiveness of group convolutions, by allowing for more
parameter sharing." Regime: image classification (AID aerial scenes), planar + p6 group conv, no
games, no RL. The parameter-sharing argument (6-fold > 4-fold for weight tying) is the citation P3-C
would need if it argues hex-native layers; it is NOT a GNN and NOT a games result.

**2d. The equivariance-vs-augmentation debate, stated generally (2026).**
SOURCED — https://arxiv.org/abs/2606.26273 (2026-06-24; Dong, Flinth, Gerken), read 2026-09-21:
"Although equivariant networks are well-studied theoretically, much less is known about data
augmentation, since analyzing augmentation requires control over the training dynamics." and "We
conduct extensive numerical experiments which show that one of our symmetrization methods (orbit
expansion) outperforms the baseline in both equivariance and overall performance." Regime:
Bayesian NNs under variational inference, UCI-style tasks — no games. Use: the framing sentence
(theory gap is on the AUGMENTATION side, i.e. our current default), not the numbers.

**NOT FOUND:** a dihedral-D6- or hexagonal-equivariant GRAPH network applied to a board game or to
any AZ-style self-play. Queries that came back empty (read 2026-09-21): arXiv
`"equivariant" AND "graph neural network" AND ("board game" OR "AlphaZero" OR "game-playing")`,
`"dihedral" AND "symmetry" AND "reinforcement learning"`, `ti:"hexagonal" AND abs:"symmetry" AND
cat:cs.LG`, `"graph neural network" AND "Monte Carlo tree search" AND "game"` (only hit: QSAT
neural MCTS, 2101.06619, a SAT domain). Semantic Scholar (would have widened this) was 429 all
session — UNVERIFIED-FETCH, not searched.

## 3 · Compound-move search: two placements per turn

**3a. Two-Stage MCTS for Connect6 — the canonical compound-turn MCTS paper.**
SOURCED — Yen & Yang, "Two-Stage Monte Carlo Tree Search for Connect6", IEEE Trans. Comp. Intell.
AI in Games 2011; full text at https://docslib.org/doc/7457944/two-stage-monte-carlo-tree-search-
for-connect6-shi-jim-yen-member-ieee-and-jung-kuei-yang (read 2026-09-21; two-column PDF, quotes
de-interleaved, words verbatim): "Except for the first move, which is limited to placing just one
stone, in subsequent moves players are allowed to place two stones simultaneously." Mechanism:
"this paper thus proposes a new MCTS variant related to Connect6, called two-stage MCTS. The first
stage focuses on threat space search (TSS), which is designed to solve the sudden-death problem.
[...] The second stage uses MCTS to estimate the game-theoretic value of the initial position."
Result: "two-stage MCTS is considerably more efficient than traditional MCTS on those positions
with TSS solution in Connect6" (paraphrase-safe: sentence is split across columns; marked
paraphrase for the result clause). Regime: classic (pre-deep) MCTS with TSS, bounded 19×19 Connect6,
no learning. The load-bearing fact for P3: the two stones are handled as TWO STAGES of selection
(threat-first, then value MCTS), not as one flat 355² joint action — the same decomposition choice
our compound-turn tree already makes; their move-ordering insight is threat-first ordering WITHIN
the turn.

**3b. MA Gumbel MuZero: compound (joint) actions as the action space, Gumbel-Top-k over them.**
SOURCED — Hao, Hao, Xiao, Li, Li, Zheng, "Multiagent Gumbel MuZero: Efficient Planning in
Combinatorial Action Spaces", AAAI-24; open PDF https://ojs.aaai.org/index.php/AAAI/article/
download/29121/30120 (read 2026-09-21, pdftotext): joint actions ⃗a = ⟨a1,…,an⟩ with
"A = A1 × ··· × An is the joint action space", and the trick that makes a compound action
searchable without enumeration: "This relation is the key to convert the bottom-up sampling
procedure to a top-down one, with the benefit that we do not have to enumerate all actions in A."
Regime: cooperative multi-agent control (matrix games, MA-Gym Switch, SMAC appendix), joint spaces
exponential in agents, 5 seeds, small nets — NOT a board game, but it is the only fetched work
that SEARCHES a per-turn compound action by sampling it autoregressively rather than enumerating
(the exact alternative to two-stage decomposition for a two-stone turn).

**3c. Live Connect6 self-play practice (2026, coursework tier).**
SOURCED — https://github.com/jesusobando32/Connect6-AI-Agent at commit
`36ea4183c48c38bb0d7b169db869e5d49a1a924a` (README, read 2026-09-21): "combina la exploración
profunda de algoritmos de búsqueda con la intuición de redes neuronales" and "Utiliza una
combinación de **Monte Carlo Tree Search (MCTS)** con selección de política **UCB1** para la
búsqueda de jugadas óptimas, guiado por una **Red Neuronal (Deep Learning)** pre-entrenada
(*playout policy*)". Regime: university coursework (UCAB, Mar–Jul 2026), 0 stars, MCTS+UCB1 with a
playout net — no AZ training, no compound-turn novelty. Cited only to show the 2026 state of
public Connect6 practice: nothing beyond classic MCTS.

**NOT FOUND:** (i) any paper on move-ORDERING within a compound turn beyond Yen & Yang's
threat-first stage (arXiv `"Connect6"` → 0 hits; `"compound action" AND "Monte Carlo"` → 0;
`"multiple moves per turn"` → 0; `"Arimaa"` → 0 relevant — read 2026-09-21); (ii) any deep-RL/AZ
result on Connect6 or any two-placement game (packet 1's Soemers transfer paper remains the only
scholarly touch). `www.connect6.org` (Wu's publication list) TLS-failed all session —
UNVERIFIED-FETCH; Semantic Scholar 429 all session — UNVERIFIED-FETCH.

## 4 · Small-net saturation: strength vs net size at fixed game/compute budget

**4a. Encoder-capacity sweep in the Gin Rummy study (§1c).**
SOURCED — https://arxiv.org/abs/2607.06854, read 2026-09-21: "Comparing MLP, convolutional,
set-based, attention, and recurrent encoders shows that extra capacity does little to break the
ceiling, suggesting the limit is information rather than network size." Regime: imperfect-info
card game, model-free (no search), small nets, 100+ runs — the only fetched multi-architecture
capacity comparison at small scale with strength as the outcome. It SATURATES: capacity is not the
binding constraint in their regime.

**4b. AVI (§1a) — small-net competitiveness without search.**
SOURCED — https://arxiv.org/abs/2609.09094, read 2026-09-21: "we train a minimal self-play
implementation of Approximate Value Iteration (AVI) and use ground-truth oracles for exact
evaluation." Regime: minimal nets, Connect Four/Hex(7x7). One net size, no sweep — cited as
context: a minimal net + exact targets matched AZ value quality in their small games.

**NOT FOUND (still):** a strength-vs-net-size (width or depth) curve at FIXED games/compute for an
AZ-style system — the same NOT FOUND packet 1 recorded at Q3. New queries this session (read
2026-09-21): arXiv `"network size" AND "self-play"` (only §1c), `"scaling" AND "board games"`
(nothing on-point), `"compute-optimal" AND "self-play"` → 0. Jones 2021 (already in packet 1:
fully-connected nets, Elo-per-compute axis) remains the nearest primary curve; no GNN or CNN AZ
size-sweep at equal games has appeared.

## 5 · Replay-ratio ablations at small compute

**5a. Fedus et al., the canonical capacity/ratio ablation (model-free, Atari).**
SOURCED — https://arxiv.org/abs/2007.06700 (2020-07-13; Fedus, Ramachandran, et al., ICML 2020),
read 2026-09-21: "Our additive and ablative studies upend conventional wisdom around experience
replay -- greater capacity is found to substantially increase the performance of certain
algorithms, while leaving others unaffected." and "by directly controlling the replay ratio we
contextualize previous observations in the literature and empirically measure its importance
across a variety of deep RL algorithms." Regime: Atari, Q-learning family (DQN/Rainbow-class), no
search, no self-play — capacity helps SOME algorithms and not others; the ratio is measured
important but the optimum is regime-specific. No number transfers to AZ-style; the transferable
point is that capacity and ratio were only decoupled by DIRECT control, which is what run9's
reuse-8-with-500k-ring holds fixed.

**5b. Reanalyse: reuse by regenerating targets on existing data (MuZero Unplugged).**
SOURCED — https://arxiv.org/pdf/2104.06294v1 (packet-1 source, NEW sentence fetched read
2026-09-21): "we describe the Reanalyse algorithm which uses model-based policy and value
improvement operators to compute new improved training targets on existing data points, allowing
efficient learning for data budgets varying by several orders of magnitude." Regime: Atari 100k
and RL Unplugged offline benchmarks, MuZero-sized nets, TPU-class training. The mechanism P3-A
should see: replay ratio can be raised WITHOUT new self-play if targets are recomputed by search —
an option packet 1's window/ratio table did not list (it is strix's proof-target idea from the
consumption side, not the production side).

**5c. Go-Exploit (§1f): reuse by re-entering archived states.**
SOURCED — https://arxiv.org/abs/2302.12359, read 2026-09-21: "Go-Exploit samples the start state
of its self-play trajectories from an archive of states of interest." Regime: Connect Four + 9×9
Go, AZ baseline. Different reuse axis (states, not steps), measured stronger per sample than both
plain AZ and KataGo's search control — the second fetched instance (with reanalyse) of "reuse the
EXPENSIVE part (search/targets), not the cheap part (forward passes)".

**5d. The 2025 survey framing: replay-ratio work is model-free territory.**
SOURCED — https://arxiv.org/abs/2508.03194 / ar5iv full text (read 2026-09-21): "Lastly, in
training budget scaling, we evaluate the impact of distributed training, high replay ratios, large
batch sizes, and auxiliary training on training efficiency and convergence." Its Table 1 tags
replay-ratio scaling to Simba, SimbaV2, BBF, BRO, BRC, Scaling CRL, SYNTHER, PGR, OpenAI Five —
ALL model-free or actor-critic; NO AlphaZero-style system appears under replay-ratio scaling.
Regime: survey of model-free DRL at large batch/agent scale.

**NOT FOUND (still):** an AZ/MuZero-style replay-ratio-vs-strength ABLATION at single-GPU scale
with a stated optimum — packet 1 had setpoints (strix 8.0, KataGo 3.96, AGZ ≈2–3) and still called
the curve absent; this session adds the mechanism-level alternatives (reanalyse, state archives)
but no ratio sweep. Queries (read 2026-09-21): arXiv `"replay ratio"` (nothing AZ-style),
`"replay buffer" AND "self-play"` (only Tablut 2604.05476 + off-topic), `"reanalysis" AND
"MuZero"` → 0, `"data efficiency" AND "AlphaZero"` → 0 on-point.

## 6 · Gumbel σ / c_scale / m at large branching factors (our ≈355 legal moves)

**6a. The Gumbel paper's own sensitivity statement (re-fetched for this question).**
SOURCED — Danihelka et al. 2022, PDF at https://web.archive.org/web/20230218064046id_/
https://openreview.net/pdf?id=bERaNdoegnO (read 2026-09-21, pdftotext): "In all Go and chess
experiments, Gumbel MuZero scales the Q-values by cvisit = 50 and cscale = 1.0. On the
perfect-information game of Go, Gumbel MuZero is not very sensitive to the scale of the Q-values.
Any cvisit ≥ 50 produced similar results (Figure 8a)." and "We use the normalized Q-values also in
Full Gumbel MuZero, but the algorithm does not require Q-values from a specific interval."
CAUTION on regime: their Go experiments are 9×9 (≈82 actions) and chess ≈35 — the insensitivity
statement was NOT measured at 300+ actions. Also from the same text, exploration: "During
training, MuZero acts with explorative actions in the first 30 moves of each self-play game.
MuZero samples the explorative actions proportionally to the visit counts, like AlphaGo Zero
(Silver et al., 2017)."

**6b. MA Gumbel MuZero: Gumbel root sampling at exponential action counts.**
SOURCED — AAAI-24 PDF (§3b URL), read 2026-09-21: "Applying AlphaZero to such problems is
challenging as larger action spaces require much more simulations to keep a good performance."
Why root sampling is the fix: "As MuZero selects actions according to Eq. 1 deterministically, some
actions may never be selected especially when Nsim < |A|. To help explore different actions, Gumbel
MuZero randomly samples k actions without replacement according to πθ as the candidate actions to
search by using the Gumbel-Top-k trick (Vieira 2014)." Their top-k setting — the only fetched
number tying m to sims: "We set the sample number k = clamp(nsim /2, min=2, max=16)." Robustness:
"MA Gumbel AlphaZero can even achieve good performance with only 2 simulations, which shows that
the method is more robust to the simulation budget." Their σ restatement matches ours: "σ(q) =
(cvisit + maxb N(b))^cscale q is a monotonically increasing transformation function" — no c_scale
value sweep appears in the PDF. Regime: multi-agent coordination tasks (joint spaces exponential,
matrix games / Switch / SMAC), 5 seeds — not a board game, no σ ablation.

**6c. Sampled MuZero (as characterised by 6b's authors — their sentence, not the primary).**
SOURCED — same AAAI-24 PDF, read 2026-09-21: "Hubert et al. (2021) propose Sampled MuZero, which
extends MuZero to more complex action spaces such as high-dimensional continuous ones. Rather than
enumerating all possible actions, its idea is to sample a small subset of actions with replacement
and do policy evaluation and improvement with respect to the sampled actions." (The primary
arXiv ID for Sampled MuZero resolved to a DIFFERENT paper in this session's arXiv —
2104.06300 = "Chaos and turbulence in clouds" — so the primary text is UNVERIFIED-FETCH here; only
this secondary characterisation is quotable.)

**NOT FOUND:** a c_scale or σ-form sweep at |A| ≈ 300+ in ANY board-game setting, and any direct
treatment of m (top-k) at fixed large branching for a single-agent game. The nearest regimes
fetched: 9×9 Go/chess insensitivity (82/35 actions, 6a) and exponential joint spaces with k≤16
(6b). Queries (read 2026-09-21): arXiv `"Gumbel" AND "action space"` (nothing beyond 6b/6c
context), `"branching factor" AND "Monte Carlo tree search"` (survey 2103.04931 + continuous/
geometry variants, none on Gumbel root constants). mctx (google-deepmind) was cited by packet 1 at
sha 88f92056 — no new m/c_scale content was sought there this session.

---

## SOURCE INDEX (all read 2026-09-21)

| # | URL | what was taken | loads? |
|---|---|---|---|
| S1 | https://arxiv.org/abs/2609.09094 (AVI self-play, 2026-09-08) | §1a quotes; §4b quote | 200 + API |
| S2 | https://arxiv.org/abs/2606.17847 (WallZero, 2026-06-16) | §1b quotes | 200 + API |
| S3 | https://arxiv.org/abs/2607.06854 (Gin Rummy gold-standard, v2 2026-08-29) | §1c, §4a quotes | 200 + API |
| S4 | https://arxiv.org/abs/2608.27757 (searchless chess, 2026-08-27) | §1d quotes | 200 + API |
| S5 | https://arxiv.org/abs/2608.15700 (Flexer, 2026-08-16) | §1e quotes | 200 + API |
| S6 | https://arxiv.org/abs/2302.12359 (Go-Exploit, AAMAS 2023) | §1f, §5c quotes | 200 + API |
| S7 | https://github.com/lightvector/KataGo/releases (API) | §1g release names/bodies | 200 |
| S8 | https://arxiv.org/abs/2302.04798 (Equivariant MuZero, 2023) | §2a quotes | 200 + API |
| S9 | https://arxiv.org/abs/2301.04746 (SLAP, v5) | §2b quotes | 200 + API |
| S10 | https://arxiv.org/abs/1803.02108 (HexaConv, 2018) | §2c quotes | 200 + API |
| S11 | https://arxiv.org/abs/2606.26273 (Equivariance/augmentation BNNs, 2026) | §2d quotes | 200 + API |
| S12 | https://docslib.org/doc/7457944/… (Yen & Yang, Two-Stage MCTS for Connect6, IEEE TCIAIG 2011) | §3a quotes (de-interleaved) | 200 |
| S13 | https://ieeexplore.ieee.org/document/5740585 (same paper, publisher page) | located via DDG only | not fetched (paywall) |
| S14 | https://ojs.aaai.org/index.php/AAAI/article/download/29121/30120 (MA Gumbel MuZero, AAAI-24) | §3b, §6b quotes; §6c characterisation | 200, PDF 3.7 MB |
| S15 | https://github.com/jesusobando32/Connect6-AI-Agent @ 36ea4183c48c38bb0d7b169db869e5d49a1a924a | §3c README quotes (Spanish) | 200 |
| S16 | https://arxiv.org/abs/2007.06700 (Fedus et al., ICML 2020) | §5a quotes | 200 + API |
| S17 | https://arxiv.org/pdf/2104.06294v1 (MuZero Unplugged; packet-1 source) | §5b Reanalyse sentence | 200 |
| S18 | https://arxiv.org/abs/2508.03194 + https://ar5iv.labs.arxiv.org/html/2508.03194 (DRL scaling survey) | §5d quote + Table 1 read | 200 both |
| S19 | https://web.archive.org/web/20230218064046id_/https://openreview.net/pdf?id=bERaNdoegnO (Gumbel MuZero; packet-1 source) | §6a quotes | 200, PDF |
| — | https://api.semanticscholar.org/graph/v1/… | NOTHING — 429 on every attempt | failed |
| — | https://api2.openreview.net/notes/search | NOTHING — searchUnavailable:true | failed |
| — | https://www.connect6.org/ | NOTHING — TLS error, curl exit 60 | failed |

Empty-result arXiv queries (run 2026-09-21, all returned zero on-topic hits): `"Connect6"`;
`"compound action" AND "Monte Carlo"`; `"multiple moves per turn"`; `"Arimaa"`; `"compute-optimal"
AND "self-play"`; `"reanalysis" AND "MuZero"`; `"equivariant" AND "graph neural network" AND
("board game" OR "AlphaZero" OR "game-playing")`; `"dihedral" AND "symmetry" AND "reinforcement
learning"`; window queries of §1. Packet-1 falsified register read before framing: F-44, F-47,
F-48/F-50/F-51 (SUSPENDED), F-52, F-15, F-51 — nothing here re-proposes them.

---

## P2 — REVIEW OF P1 (fresh agent; P1 was not passed on unreviewed)

# P2 REVIEW — PACKET RESEARCH-STRENGTH-2 · 2026-09-21

Fresh-eyes audit of /tmp/rs2/P1_RESEARCH.md. Method: every cited URL re-fetched today (arXiv
export API + abs pages; AAAI/docslib/web.archive PDFs via pdftotext; GitHub API + raw at sha;
ar5iv HTML); every quoted sentence grepped verbatim in the fetched text (whitespace-normalised);
every NOT-FOUND arXiv query re-run where cheap (7 of ~12); all three "failed host" reports
reproduced. Artifacts in /tmp/rs2/p2/.

## VERDICT TABLE

| ref | verdict | one-line evidence |
|---|---|---|
| §1 window-empty | VERIFIED | re-ran both queries: 4 entries, exactly the 4 cited IDs+titles; wider query totalResults=0 |
| 1a AVI | VERIFIED | both quotes verbatim in abs 2609.09094; C4/Hex(7x7)/Othello/Go(9x9) as stated; abstract carries no compute number |
| 1b WallZero | VERIFIED | both quotes verbatim (abs 2606.17847); 7×7 two-player WallGo, AlphaZero-based, "high game-tree complexity" from stone movement + wall placement |
| 1c Gin Rummy | VERIFIED | all 3 quotes verbatim (incl. negative list); "more than a hundred runs", Gin Rummy, Leduc cross-check; v1 07-07/v2 08-29, authors match |
| 1d searchless chess | VERIFIED | both quotes verbatim (abs 2608.27757); "~2000 steps" = "In about two thousand steps"; caveat: "(single GPU-class)" is P1's gloss, source states no hardware |
| 1e Flexer | VERIFIED | both quotes verbatim; regime "three toy symbolic problems" is the abstract's own phrase |
| 1f Go-Exploit | **WRONG (regime)** | 3 quotes verbatim ✓, but "one run each" is false: paper ran 10 independent runs per hyperparameter setting and 30 validation runs with 95% CIs (full text) |
| 1g KataGo | VERIFIED | API bodies: v1.18.0 2026-08-22 quote verbatim; v1.18.1/v1.18.2 names+dates exact |
| 2a Equivariant MuZero | VERIFIED | both quotes verbatim; MiniPacman + ProcGen Chaser, no board games |
| 2b SLAP | VERIFIED | both quotes verbatim; "AISB 2023" confirmed on abs page (Comments: AISB Convention 2023); parity-not-superiority reading is the authors' own |
| 2c HexaConv | VERIFIED | both quotes verbatim; AID aerial scenes, planar+p6, no games/RL |
| 2d equiv/aug BNNs | **WRONG (regime gloss)** | both quotes verbatim ✓, but "UCI-style tasks" is false: experiments are FashionMNIST (C4); no UCI dataset appears in the paper |
| §2 NOT FOUND | VERIFIED | 2 of 4 queries re-run → 0 hits (eq-GNN-boardgame; dihedral+RL); 2101.06619 resolves to "Solving QSAT problems with neural MCTS" |
| 3a Yen & Yang | VERIFIED | all quotes word-for-word in the docslib text (two-column interleaved exactly as disclosed); "19 19 board" confirmed; no learning (1 incidental 'neural' hit) |
| 3b MA Gumbel MuZero | VERIFIED | joint-space def + Nsim<\|A\| + robustness + k=clamp sentences verbatim; matrix games / MA-Gym Switch / SMAC-in-Appendix-C / 5 seeds all confirmed. Nit: "top-down one" quote omits one word ("top-down sampling one") — meaning unchanged |
| 3c Connect6 coursework | VERIFIED | both Spanish quotes verbatim in README at the cited sha; UCAB, Mar–Jul 2026, stars=0 (API) |
| §3 NOT FOUND | VERIFIED | 3 of 4 queries re-run → 0 hits ("Connect6"; "compound action"+"Monte Carlo"; "multiple moves per turn"); Arimaa query not re-run |
| 4a capacity sweep | VERIFIED | quote verbatim (abs 2607.06854) |
| 4b AVI small-net | VERIFIED | quote verbatim; full text: one architecture (3 residual blocks), no sweep/ablation of size in PDF |
| §4 NOT FOUND | VERIFIED | "compute-optimal"+"self-play" re-run → 0; 2 other queries not re-run |
| 5a Fedus | VERIFIED | both quotes verbatim; "Q-learning methods" in abstract; Atari ×9 in PDF; ratio-measured-important claim is the abstract's |
| 5b Reanalyse | VERIFIED | quote verbatim in 2104.06294v1 PDF (= MuZero Unplugged, v1 title "Online and Offline RL by Planning with a Learned Model"); TPU mention present |
| 5c Go-Exploit reuse | VERIFIED | quote verbatim (abs); regime C4+9×9 Go, AZ baseline |
| 5d DRL survey | VERIFIED | abstract quote verbatim; Table 1 tags Replay Ratio to exactly OpenAI Five, SYNTHER, PGR, BRO, BRC, Simba, SimbaV2, BBF, Scaling CRL; "AlphaZero"/"MuZero" appear 0 times in the whole survey |
| §5 NOT FOUND | VERIFIED | "reanalysis"+"MuZero" re-run → 0; other queries not re-run |
| 6a Gumbel σ/cvisit | **WRONG (regime caution)** | all 3 quotes verbatim ✓, incl. "In all Go and chess experiments… cvisit = 50 and cscale = 1.0" — but the caution "their Go experiments are 9×9 (≈82 actions)" is false: §7.2 runs LARGE-SCALE 19×19 Go ("19x19 Go has 362 possible actions") with those same constants. Only the cvisit-sensitivity figure (Fig 8a) is 9×9-only, so the narrower sub-claim "insensitivity not measured at 300+ actions" survives |
| 6b MA-GM constants | VERIFIED | all quotes verbatim; "no c_scale value sweep appears in the PDF" confirmed (cscale occurs only inside formulas); regime as stated |
| 6c Sampled MuZero | VERIFIED | characterisation verbatim in the AAAI PDF; 2104.06300 re-resolves today to "Chaos and turbulence in clouds" — P1's mis-resolution report is accurate |
| §6 NOT FOUND | VERIFIED-in-part | queries not re-run; 2103.04931 resolves to "MCTS: A Review of Recent Modifications and Applications" as cited |
| Fetch inventory | VERIFIED | reproduced today: S2 API → 429 "Too Many Requests"; openreview → `{"notes":[],"count":0,"searchUnavailable":true}`; connect6.org → TLS cert mismatch (curl 60) + http connection reset |

## STRUCK / FLAGGED (claims without a SOURCED tag)

No untagged EXTERNAL literature claims found — nothing to strike on packet rules. Five untagged
INTERNAL cross-references (not web claims; P1's header rules cover external claims only) —
disposition: keep only if P3 confirms against packet 1 / the repo, else delete:

1. §1 intro — "packet 1's source index (its Appendix B) does NOT list" the below items. Check packet 1 App. B.
2. §2a — "packet 1 measured our symmetry spread 0.146 at run8@18k". Internal repo number; re-derive before use.
3. §5a — "which is what run9's reuse-8-with-500k-ring holds fixed". Internal repo state.
4. §5 NOT FOUND — "packet 1 had setpoints (strix 8.0, KataGo 3.96, AGZ ≈2–3)". Packet-1 numbers, unfetched here.
5. §4 NOT FOUND — "Jones 2021 … remains the nearest primary curve". Inherited packet-1 pointer, no sentence fetched in either session — do not lean on it without packet 1's own quote.

## BOTTOM SUMMARY

Counts: **VERIFIED 27 · UNVERIFIED 0 · WRONG 3 · struck 0** (5 internal cross-refs flagged).
Every quoted sentence checked today is present verbatim in its fetched source (two one-word
quote nits: 3b omits "sampling"; both immaterial). All arXiv IDs resolve to the titled papers.
All three "failed host" reports reproduce exactly, so the UNVERIFIED-FETCH annotations are honest.

**P3 may safely lean on:** 1a, 1b, 1c, 1d, 1e, 1g, 2a, 2b, 2c, 3a, 3b, 3c, 4a, 4b, 5a, 5b, 5c,
5d, 6b, 6c — quotes verbatim and regimes as the sources state them. The NOT-FOUND blocks are
trustworthy: every re-run query (7/7) reproduced P1's zero/4-hit result exactly.

**Needs a caveat:**
- **1f (WRONG, understates the source):** Go-Exploit is STRONGER evidence than P1 says — 10
  runs/sweep setting + 30 validation runs with CIs, not "one run each". Safe to lean on harder.
- **6a (WRONG, understates the source):** the Gumbel paper DID run 19×19 Go (362 actions) with
  cvisit=50/cscale=1.0 at n=400 and matched/exceeded MuZero. Only the cvisit SWEEP is 9×9-only.
  P3-C/P3-F must not repeat "the constants were never exercised above ~82 actions".
- **2d (WRONG, minor):** dataset is FashionMNIST, not "UCI-style"; the framing sentence P3 would
  use (theory gap sits on the augmentation side) is unaffected.
- 1d: "(single GPU-class)" is P1's gloss; source names no hardware.
- 3b: quote nit "top-down [sampling] one".

---

## P3 + P4 — THE FIVE BRIEFS, REVISED AFTER REBUTTAL (the revised versions govern; [REV:] marks the changes; each brief is followed by its objections to the other four)


### BRIEF A — DATA & OPTIMISATION (revised)

# P3 BRIEF A — DATA & OPTIMISATION, REVISED (PACKET RESEARCH-STRENGTH-2, 2026-09-21, rev 1 post-P4)

**Thesis.** Run8's late flatness (+3.1 pp then +0.7 pp per 15k games) is a data/optimisation regime
artifact — window turnover, LR shape/floor, step noise — not a net or search ceiling; the knobs are
one-run row swaps that beat any build on expected Δ per box-hour. [REV: scope narrowed — B holds
the packet's only MEASURED knob change (σ); "the knobs" here means OUR rows, and only while the
step still teaches, which run9's 36k-steps cell prices.]

## Proposals (template §2; shas from P0 §record; parent run8@45k 0.142 [0.104,0.181] unless named)

**P1 — LR-SHAPE-FLOOR** (run10's lever; R364(d) reserves it).
- thesis: a cosine that completes inside the block at strix's AdamW pair restores the slope.
- evidence: MEASURED(PARAM_DISTANCE_2026-09-21.md, 4c6759b9cbe2eec4): ‖Δ‖/3k-steps flat
  18.7–24.3 while cos(Δ,Δprev) fell 0.15→0.03 and the strix gain fell +3.1→+0.7 pp; LR 9.97e-4
  at 51k of a 1e-3→5e-4-over-1M cosine = flat. SOURCED(STRENGTH_RESEARCH §Q6, 699ae9c96483330b,
  read 2026-09-18): strix 2e-4→2e-5; "a fact about two runs, not a measured effect".
  [REV: E's bundle objection accepted — strix differs on MSE value, draw→0, batch 512, max_moves
  300 too (§Q6/§Q7.2); P1's warrant is the noise statistic, not strix parity.]
- swap (rows verified at HEAD, run8.yaml:88/94/95): `train.lr 0.001→2e-4`; `train.eta_min
  5e-4→2e-5`; `train.scheduler_t_max null→<stop-in-steps>` (= the pre-registered stop;
  CosineAnnealingLR cycles back up past t_max). LR warmup needs a NEW schema row = a build;
  excluded here.
- cost: ≈ 17 box-h to first reading + ≈ 1 dev-h [REV: D's correction — 17 assumed the ALONE
  999–1,069 steps/h (EVAL_COST:150–151); in flight run8 averaged 871, a cell costs 0.38× — 17 is
  a floor, not an estimate].
- success line: ≥ 0.192 at 15k GAMES vs run8@45k (256/256) [REV: unit qualifier per E1 — stated in
  strix units; if E1's r6 cell moves ≥ CI half-width, this and every series line inherit the
  radius qualifier at once].
- falsifier: none < 1 box-h; a run is the only test. 0-box-h control: PARAM_DISTANCE re-run on
  run9's checkpoints (coherence stuck ≈0.03 on shared-window data indicts the step).
- what kills it: run9 alone reaching ≥ 0.192; P2's probe null; the value-head norm still climbing
  under 2e-5 with no gain; [REV: C3-stage-1 reading UNDERFIT (train≈held-out, both high) — an LR
  floor deepens underfit; the 0-box-h probe gates P1's direction before mint].
- interactions: ruled run10's (R364(d)); cannot ride run9 (one swap); if run9 is FALSIFIED, run10
  = run9's parent at run8's rows (RUN9_PREREG §3) and P1 waits. [REV: E2 — a hot cap/draw census
  (≥5 %) puts draw-utility in run10's queue ahead of LR; R364(d) rides on E2 reading ≤3 %.]

**P2 — EMA-PROBE-THEN-ROW.**
- thesis: ≥80 % of each 3k-step displacement is orthogonal walk; averaging it off is free strength.
- evidence: MEASURED(PARAM_DISTANCE, 4c6759b9cbe2eec4): cos(Δ,Δprev) 0.13–0.20 → 0.03 late;
  weight norm +49 %, value head +96 %. SOURCED(KataGo arxiv.org/abs/1902.10565 §2, read
  2026-09-18): EMA-of-snapshots decay 0.75 — practice only, no ablation number (§Q6 NOT FOUND).
- swap: STEP 1, a measurement — offline EMA over run8's 18 checkpoints (mirror) through ONE
  288-game 256/256 cell. STEP 2, a row: `train.ema.enabled: true` (decay 0.999, update_every 10;
  the EMA state_dict is what self-play/eval/promotion consume, trainer/core.py:491-493).
- cost: step 1 ≈ 1.5 box-h + ≈ 3 dev-h; step 2 rides its run ≈ 17 box-h [REV: same duty floor].
- success line: step 1 — EMA cell > 0.181 → arm; inside [0.104,0.181] → do not arm; ≤ 0.142 → the
  walk is not averageable noise. Step 2: ≥ 0.192 at 15k games vs run8@45k.
- falsifier: none strictly < 1 box-h; step 1 is cheapest at ≈ 1.5 box-h (stated per §2).
- what kills it: cadence mismatch (3k-checkpoint EMA ≈ SWA, not decay-0.999-per-10-steps —
  transfer ARGUED); a null cell; the coherent drift being off-target; [REV: E3 — the walk may be
  part STRUCTURED: F-45's 25 α=1.0 rows all sit at moves_remaining==1 in lost positions
  (MEASURED falsified.md F-45); the EMA cell's read must bucket by root mr — a gain confined to
  mr==1 roots is turn-structure repair, not noise averaging].
- interactions: run10's ONE swap, never beside P1; consumes run9's checkpoint series, forces no
  run9 change. [REV: C2's 0-box-h dist65 calibration probe co-reads the same checkpoints — one
  mirror pass serves both; a miscalibrated E[v] confounds the EMA cell's direction.]

**P3 — BATCH-512-AT-HELD-REUSE.** [REV: retained, demoted third — parity + arithmetic evidence,
  and B-P3 competes for the same fewer-better-steps axis at rows-only cost.]
- thesis: at fixed reuse ≈ 8, doubling the batch halves the step count for the same samples.
- evidence: SOURCED(STRENGTH_RESEARCH §Q7.2): strix batch 512 — parity only. MEASURED(F-44,
  falsified.md): trainer idle 92 % at 1 step/game. Slope gain: SPECULATION.
- swap: `train.batch_size 256→512` (schema/train.py:248); `train.training_steps_per_game →1.2`
  (512×1.2÷80.3 rows/game = 7.66 reuse, inside run9's [7,9] band; mixing carry verified at HEAD).
- cost: ≈ 17 box-h [REV: same in-flight-duty floor]; ≈ 1 dev-h. Falsifier: the ≈10-min memory
  smoke (8.94 GiB request of 15.48 doubling; microbatch_caps must absorb) — falsifies the ARM.
- success line: ≥ 0.192 at 15k games vs run8@45k.
- what kills it: weakest evidence of the three; the memory smoke refusing; P1 winning first;
  [REV: D-1's duty arithmetic — if fill rows lift games/h, steps/game must rise to HOLD reuse 8].
- interactions: [REV: B-P3 moves target noise the OPPOSITE way (noisier completed-Q, more
  games/h) — the two cannot share a window without confounding; if both queue, P3 first or never.
  C3 UNDERFIT also demotes P3: batch shrinks gradient noise, not capacity.]

## What run9's reuse-8 WILL and WON'T show

WILL (RUN9_PREREG 3e40594afb6a7320 §3): the PAIR at equal GAMES — SUCCESS ≥ 0.192 @30k-games,
FALSIFIED ≤ 0.142 at 15k AND 30k, else 45k THEN STOP; the shakedown's ratio ∈ [7,9] and games/h
replacing the ≈16 h ESTIMATE; the trainer's card share under 2.4× steps/game.
WILL, 0-box-h controls: the turnover confound removed (100k ring turned over every ≈1 250 steps;
500k every ≈14 900 at 2.4/game — PARAM_DISTANCE §3); a PARAM_DISTANCE re-run on run9's
checkpoints: coherence rising on shared data indicts the window, stuck ≈0.03 indicts the step (and
arms P1). [REV: three more free riders on ONE mirror pass — C3's gap probe, C §3's spread series
(≥12 positions/checkpoint, not 1), the warm-start dip's first datum off run9's opening 3k
checkpoints (E §3.2: no cheap falsifier exists anywhere; the dip's game cost is unattributed).]
WON'T: capacity vs ratio — the swap moves BOTH (100k→500k AND 3.2→≈7.7); Fedus et al. (SOURCED
/tmp/rs2/P1_RESEARCH.md §5a, arxiv.org/abs/2007.06700, read 2026-09-21, P2-VERIFIED: "by
directly controlling the replay ratio we contextualize previous observations … and empirically
measure its importance") decoupled them only by direct control, model-free Atari — one run
cannot.
WON'T: equal-games ≠ equal-steps ≠ equal-box-hours. At 15k games run9's net has 36k steps vs
run8's 15k — [REV: promoted from caveat to discriminator — the 36k cell is what prices D's
steps/h→strength conversion going forward: flat at equal games despite 2.4× steps says the step
teaches nothing even on shared data (arms P1, defuses D); ≥ 0.192 says steps still teach and the
binder was data (D's conversion holds, C's ceiling was premature). Either reading is a result.]
WON'T: LR. The LR rows are byte-equal; run9's cells cannot speak to R364(d) — only its checkpoint
series can (above).
WON'T: the warm-start transient's downstream cost — fresh optimizer moments, no path in
warmstart.py; run8's own re-warm dipped to 0.056 at 3k steps (MEASURED PARAM_DISTANCE §2).

## Unmeasured premises (no measurement found for any)

- "reuse ≈ 8 healthy": strix's author comment; no ratio-vs-strength ablation anywhere (§Q2).
- Augmentation's EFFECT in our regime: run8 changed σ AND augment together (R358) — never
  separated; the ×8 is AlphaZero Fig. S1, 19×19 Go, 5000 TPUs (§Q1).
- EMA/SWA benefit: practice only (§Q6). "LR 1e-3 too high": a two-run bundle fact (§Q6; E's
  bundle-citing objection accepted — see P1 evidence REV). Batch 256: parity only (§Q7.2).
- Coherence-fall ⇒ strength cost: one run's correlation (PARAM_DISTANCE §2–§3 says so itself);
  value-head norm +96 %: "filed here, not read" (§3). Warm-start later cost: NOT FOUND (§Q10).
  Gate sims 64: UNMEASURED (STATE item 13). Run9's ≈16 h/15k games: an ESTIMATE (RUN9_PREREG §3).
- [REV: three more, E's finds — cap/draw fire-rates past the 600-game boot window (E2); mr=1 vs
  mr=2 value calibration (E3; F-45 the one artifact); the ruler's radius provenance (E1). All
  sub-box-hour tests; none touches a run row.]

## Dissent space

Weakest point, sharpened [REV]: no knob in MY lane has a MEASURED strength cost — one same-game
setpoint (strix's 8.0, bundle-caveated), one noise statistic of unproven strength-cost, parity;
B's σ swap is the packet's only measured knob change. If run9's regime alone restores the slope,
this thesis collapses to "the window was wrong" and C's or B's lane owns the gap — R364(d)
prices it: LR waits on run9. [REV: the rebuttal's product — a pre-run10 cheap list, all on
mirrors, ≤0.5 box-h each: E2 census, E3 mr-calibration, C3 stage-1 gap probe, spread series
(≥12 positions), PARAM_DISTANCE re-run, the memory smoke. None of the four briefs contradicts
any of the six; all four depend on at least one.]

#### Objections by A to the other four

# P4 A OBJECTIONS — BRIEF A vs B, C, D, E (PACKET RESEARCH-STRENGTH-2, 2026-09-21)

**B — objection.** P-B3 prices a throughput row but smuggles a target change: the σ-softening coincided with one-hot 62–65 % → 22 % AND strength rising, read at 320 sims on run8 (MEASURED STATE.md:226/359, sha 3b597edc; regime run8 Gumbel 320/64 warm-start); halving sims re-noises completed-Q under that same σ — a data-stream change P-B3 prices nowhere — and the one equal-work cell that touched sims read "the sims are not the lever; the net is" (MEASURED STRIX_RUNG_2026-09-14.md:35, 256/256, 288 games).
**B — concede.** The σ swap is the only MEASURED knob change on record that coincided with strength rising (0.104→0.135→0.142; MEASURED STATE items 10/13–15) — target-side knobs hold one live measurement; my lane holds none.

**C — objection.** C1 spends 50–55 box-h and the warm start to buy the residue over a mechanism already armed and measured uniform (sym_bin0/mean 1.000 at every run8 audit, MEASURED STATE item 15), priced off a spread that is one position × one checkpoint (C's own §3 "repair first"), with the best external reading PARITY not gain (SOURCED SLAP, arXiv 2301.04746: equivariance-family ≈ augmentation at 1/8 samples; regime CNN/AGZ Gomoku) — the flagship is ~3× P1's cost on n=1 evidence.
**C — concede.** C3 stage 1 (train-vs-held-out gap, 0 box-h, mirror CPU) is the one instrument that arbitrates our theses — OVERFIT arms my data lane, UNDERFIT caps it — and I accept it reads run9's checkpoints before run10's queue fixes.

**D — objection.** D's steps/h→strength conversion prices at run8's early slope, but the last interval is +0.7 pp inside both CIs with cos(Δ,Δprev) 0.03 (MEASURED PARAM_DISTANCE_2026-09-21.md §2, 3k-step segments, 18 checkpoints); D's own §3 concedes a 2× is "indistinguishable from noise" there — D-1/D-2's success lines are throughput claims wearing strength clothing, and D-1's knee is ALONE-arm arithmetic (B 46/64, PERF3 step 1) beside a run living contended at fill 75 % with D's own ARGUED ≈43 % trainer duty at 2.4 reuse capping steps/h below the fill gain.
**D — concede.** My ledger was wrong: P1/P2/P3's "≈17 box-h" assumed the ALONE 999–1,069 steps/h, but EVAL_COST measured 682–724 inside a round and 356–415 inside a cell (MEASURED EVAL_COST_2026-09-19.md:150–151) — every first reading in my brief rides the in-flight eval share, and D-3-style isolation is a real correction to it.

**E — objection.** E's §3.6 verdict "value from z — ARGUED yes" leans on strix training scalar ±1 MSE from z and beating us, but E's own E2 evidence has strix differing on ALL rows at once (max_moves 300, draw→0, scalar MSE, §Q7.2, plus LR/batch per §Q6/Q7.2) — the bundle confound E polices in E1 is committed in §3.6; as a single-row fact it is what STRENGTH_RESEARCH §Q6 calls "a fact about two runs, not a measured effect" (SOURCED, said there of LR).
**E — concede.** E3 targets a measured artifact in MY stream: all 25 α=1.0 rows sit at moves_remaining==1 in lost positions (MEASURED falsified.md F-45) — if mr=1 calibration is off, part of the +96 % value-head norm and the 27 % one-hot share is turn structure, not step noise, and E3 at ≈0.5 box-h partitions my P2 claim for free.

### BRIEF B — TARGETS & SEARCH (revised)

# P3 BRIEF B (REVISED) — TARGETS & SEARCH: "what a leaf teaches" (RS-2, 2026-09-21; rev 1 after P4)

## 1. Thesis

The net only ever sees what a 320-sim Gumbel root writes down — a σ-scaled completed-Q target that is one-hot on ~27 % of rows,
proof-free, paid for at 0.515 ms/leaf; softening, proving, or buying that lesson cheaper is where strength-per-box-hour lives.
[REV: A — third flatness reading: at FIXED sims the teacher is near-stationary, so the coherence fall 0.15→0.03 is as consistent
with the net reaching the CURRENT target's fixed point (drift 6.3→1.3/segment, "not a noise ball", PARAM_DISTANCE §2 — MEASURED
4c6759b9…) as with A's noise; only a target/search swap moves that branch; the tie is ARGUED.]

## 2. Proposals (parent every success line: run8@45k 0.142 [0.104, 0.181], equal-work 256/256, 288 paired games, CONTENDED,
MEASURED STATE 15 — [REV: E — that parent carries E1's "strix @ r8" radius qualifier until E1 reads]; equal GAMES = equal rows)

**P-B1 SOFT-POLICY HEAD (queue (vii)) — the 27 % population is the target.**
- thesis: auxiliary CE on target^(1/4) (renormalised), weight 8, re-grades the decided-quarter rows the hard CE over-teaches.
  [SOURCED KataGo Table 2 aux-policy 1.30×, arXiv 1902.10565, packet-1 §Q4; T=4/w=8 are KataGoMethods fragments, unverified]
- evidence: one_hot_share_full 0.2745/0.2761/0.2705 MISS vs band < 0.25 at 15k/30k/45k, shakedown 22.1 % [MEASURED STATE 13–15].
  [REV: A — the "σ swap took 62–65 %→22–27 % while strength rose" line is VOID as σ-credit: R358 armed σ AND augment in one
  re-mint; only the no-harm co-movement survives. The decided-population case now rests on h_full_median 0.209→0.286, mean
  0.465→0.594 at flat share plus KataGo — MEASURED STATE 13–15.]
- swap: BUILD — second head + rows `train.aux_policy_temperature: 4.0`, `train.aux_policy_weight: 8.0` (kin to
  `train.fast_policy_weight: 0.0`); producer = in-run KL(hard‖soft) (LAW-18). Recorded target unchanged.
- cost: ≈ 20 box-h to a 15k cell, ~8 dev-h; ≈ 55 box-h to the 45k equal-games reading. success: 45k cell ≥ 0.18 with CI lb > 0.142.
- falsifier < 1 box-h: offline target^(1/4) KL on the mirrored ring (masses + `tail` f32 per row, ring_reader.py:17,127); median
  KL < 0.02 nats ⇒ dead, 0 box-h. [REV: A+E — the §3 decomposer (now carrying E3's moves_remaining split) is promoted to a GATE:
  with σ-credit void, build nothing until it reads the 27 % as undecided-defect; decided-and-correct ⇒ P-B1 loses its mechanism.]
- kills it: the KL; a 15k cell ≤ 0.104; F-32's class.
- interactions: (iv) prior temperature changes the top-m draw — after, never beside; run11+ (run9 = data regime, run10 = LR,
  R364(b)/(d) via PARAM_DISTANCE + P0 premise).

**P-B2 PROOF-AS-TARGET (CARD-STRIX-NET-ONLY A1's untested half).**
- thesis: strix's net TRAINED under one/two-hot proof targets; the play-time cell (Δ +0.3 pp) never touched this. [MEASURED
  CARDS CARD-STRIX-NET-ONLY A1 — "the training-side hypothesis … UNTESTED and UNRANKED, not refuted".] [REV: E — E's §3.4 names
  proof-as-target the ONE alive distillation shape (F-35/36/37 killed feature-side), and E's §3.6 hands "which loop signal" to B.]
- evidence: F-38 bounded minimax proves ALL 33 value-blind losses at depth 6–8 TURNS; F-39 widening REGRESSES 0/38 at ~100× —
  pattern-guided solver; F-40 the net guides 0/14. [MEASURED falsified.md]
- swap: BUILD — bridged `TacticalSolver` at mr≥1 roots behind `selfplay.mcts.root_solver_*` rows; on a proven win the POLICY
  target becomes the proof one/two-hot; leaf eval and value stream untouched (F-15: injection only).
- cost: ≈ 22 box-h to a 15k cell, 12–20 dev-h. success: 45k cell ≥ 0.18; secondary: in-run proof rate ≥ 3 % of roots at ≤ 5 % games/h.
- falsifier < 1 box-h: offline on mirrored ring boards — proof rate, outcome agreement, ms/root, NOVELTY. Proof rate < 2 %,
  ms/root > 50 (F-47; PERF3 CPU 11.46 of 14.44 ms), or novelty < 5 % ⇒ dead.
- kills it: those three numbers; proof one-hots pushing one_hot_share into a permanently red band (re-derive WITH the arm);
  F-02/F-22 attractor class.
- interactions: run11+; the falsifier runs NOW beside run9. [REV: C — FIRST build-priced lane to spend: its 0-box-h falsifier
  discriminates the training-regime half of C's own exhaustion premise before a 50–55 box-h D6 build forfeits its warm start.]

**P-B3 SIMS 320/64 → 128/64 (queue (v)) — buy the same lesson cheaper.**
- thesis: nothing at our regime shows 320 > 128; strix TRAINS at 128/m16 and reads 0.83–0.92 on every run7 net [MEASURED
  STRIX_RUN7_60K §A, 288 games/cell]; the box is serving-bound (F-47 0.515 ms/leaf; PERF3 3 208 leaves/s alone).
- evidence: Gumbel Table C.1 n=32 → 5.9× step speedup at like Elo [SOURCED 9×9 Go Gumbel MuZero, 2 seeds]; MiniZero Othello
  n=2 ≈ n=16 [SOURCED §Q5]; 128 ≥ m·log2(m)=64 [SOURCED strix config]. [REV: A — regime flag: those are low-reuse readings; at
  run9's reuse ≈ 8 each row is re-taught ~8×, so RANDOM target error averages but SYSTEMATIC coarsening repeats — the reuse-8
  read is not the paper's read. ARGUED.]
- swap: ROWS — `selfplay.playout_cap.n_sims_full: 320 → 128` (m/c_visit/c_scale/p unchanged). [REV: D — ×1.6–1.8 games/h was
  ARGUED arithmetic; D's bench_server measures leaves/s at 128 directly (≈ 0.2 box-h) — run it as the pre-check, don't assert it.]
- cost: ≈ 19 box-h to a 15k cell, 1–2 dev-h (+ 0.2 box-h bench pre-check [REV: D]).
- success [REV: D — wall clause restated; old ≤ 0.7× was unreachable]: 45k-equal-games cell ≥ 0.142 AND games/h ≥ 1.5× measured
  in the 3 h shakedown AND wall-to-cell ≤ 0.85× with eval in flight as-is (72 % of run8's wall had a round/cell in flight —
  32.6 % + 39 %; rounds play the same 38–40 s/game at 256/256 however fast self-play runs — MEASURED EVAL_COST §(ii)/§(iv),
  b0e7c3de…); ≤ 0.7× returns only behind D-3's off-box eval.
- falsifier: none < 1 box-h for strength; the 3 h shakedown twin (h_full/one_hot/counter-threat at 128/64) is the cheapest
  target-quality gate; a run is the only strength test.
- kills it: 15k ≤ 0.104 with games/h < 1.3×; STRIX_RUNG §B.2 read across; F-52's depth-drift watch.
- interactions: trainer absorbs games/h at reuse 8 (F-44, run6 regime — duty at 2.4 unmeasured until run9's twin); PERF-3
  step 3's B-curve is run9's preflight — read this arm inside it.

**P-B4 λ VALUE TARGET (codec landed UNARMED) — the leaf's second lesson.**
- thesis: a 40.6-turn game delivers z ~30 turns after the leaf; λ-mixing MCTS root values shortens credit assignment.
  [SOURCED Go-Exploit P1 §1f VERIFIED; AVI P1 §1a VERIFIED]
- evidence: value loss 0.64→0.50 over 51k [MEASURED STATE 15]; codec imported by nothing [MEASURED value_targets.py].
  [REV: C — add the strongest line, from C's own instrument: value-head per-segment ‖Δ‖ swings 4.6–10.8 vs policy's steady
  7.7–9.1, head norm +96 %, while loss FALLS — a head moving most per step with flattening loss is target-misspecification
  evidence. MEASURED PARAM_DISTANCE §2.]
- swap: BUILD — `train.value_target: pure_outcome_z → lambda_return`, `train.value_lambda: 0.8`; the ring must RECORD root v̂
  (ring_reader.py:17) — format change with consumers, dev M–L.
- cost: ≈ 22 box-h to a 15k cell, 10–15 dev-h. success: 45k cell ≥ 0.18 AND value loss at equal games below the parent's 0.50 track.
- falsifier: none < 1 box-h offline (ring stores no root values); the 3 h shakedown divergence producer; for strength "none; a
  run is the only test".
- kills it: bootstrap feedback (F-22…F-33 attractor family); λ ≈ z at our lengths (median |λ−z| < 0.05).
- interactions [REV: C+E — two cheap GATES before any build: (1) C2's 0-box-h dist65 calibration probe — a miscalibrated head
  confounds any target swap; (2) E2's 0.2-box-h cap/draw census — a hot ply_cap/draw rate (−0.5 labels strix writes as 0)
  poisons z itself, and λ propagates a poisoned leaf value into every backup above it; a hot census queues draw-utility AHEAD
  of λ]; recorder lands in a mint window, never on a live ring.

## 3. The 27 % one-hot share — evidence FOR and AGAINST

- Instrument first: `explicit_entropy` counts a tail-only row (α = 1.0, NO explicit mass) as H = 0 "as a one-hot does"
  [MEASURED ring_reader.py:135-137] — yet that row's trained target is the tail rebuilt OVER THE PRIOR, the softest row in the
  ring. The 27 % is an UPPER bound; the α column is stored and never printed for run8 (F-45: ~1/1 000 rising with saturation).
- FOR benign: h_full_median RISES 0.209→0.286, mean 0.465→0.594, share flat — a stable DECIDED population beside a softening
  undecided one; wrong-signed class fixed (counter-threat 0.009–0.031 %, residue PASS) [MEASURED items 13–15, R355(a)].
  [REV: A — σ-causality WITHDRAWN as confounded (R358 armed σ AND augment together); only the no-harm co-movement survives.]
- FOR defect: 27 % of rows teach near-certainty from 320 sims at branching ≈ 355, m 16; KataGo built soft targets for exactly
  this (Go, 800 sims — different regime); the < 0.25 band was calibrated on a 3 h shakedown (22.1 %, value head in warm-up)
  [MEASURED item 10] — the band may be miscalibrated, not the target. [REV: E — defect gains a mechanism: F-45's saturation
  rows ALL sit at moves_remaining == 1, the exact mid-turn state E3 flags; mid-turn decidedness is the unmeasured slice.]
- Verdict [ARGUED]: unproven either way. The 0-box-h decomposer settles it: recompute the share excluding tail-only rows; split
  by moves_remaining and outcome-agreement (ring boards reconstruct). [REV: E — the mr split IS E3's instrument, priced 0.5 box-h
  with a calibration read beside it; adopt rather than duplicate.] Decided-and-correct ⇒ re-derive the band; P-B1 loses its mechanism.

## 4. Unmeasured premises in OUR record (worktree HEAD; 1–6 unchanged from rev 0)

1. The band `one_hot_share_full < 0.25` — one 3 h shakedown reading; no outcome-linked threshold (RUN8_PREREG §3a).
2. R355(d)'s "sharpen CORRECTLY" half — never measured (RULINGS R355). 3. λ parked by analogy; zero λ measurements (CARDS).
4. Gumbel-draw exploration sufficiency past ply 10 unmeasured; strix visit-samples 20 % (search_drive.rs:747; §Q7.6).
5. The five R355(c) re-derivation cells — NO reading at HEAD; F-48/F-50/F-51 suspended (STATE:170; R355(c)).
6. run8's σ ground is MiniZero's DEFAULT, not our-regime (F-50).
7. [REV: A — added] coherence-fall ⇒ wasted step: one of THREE live readings (noise / window turnover / stationary-teacher
   fixed point); no measurement separates them (PARAM_DISTANCE §3 leaves it open).
8. [REV: E — added] the parent 0.142 inherits E1's radius qualifier: "strix @ r8" is a rules row neither side trained at
   (§Q7.1.2, strix_driver.py:67), and E1's ±4 pp stability band exceeds run8's whole gain (+3.1 pp) — every success line above
   can wobble by more than the gain it measures.

## 5. Where the thesis is weakest

Still: no measurement ties TARGET SHAPE to strength — and rebuttal cut the props: A voided my only in-regime σ-credit line, D cut
P-B3's wall claim to ≤ 0.85× until eval moves off-box, C's calibration probe and E's census now gate P-B4, E1 can stamp a ±4 pp
qualifier on the parent [REV: all four]. The convergent node is whether the step still teaches (A's noise vs B's fixed point vs
the window — run9's reuse-8 cell decides): if run9 restores the slope, the target lane owns the residue; if not, the 0-box-h set
— decomposer, E2, E3, C2's probe — is the only honest spend before any run11 queue, and the ceiling talk stays C's.

#### Objections by B to the other four

# P4 BRIEF B — OBJECTIONS (PACKET RESEARCH-STRENGTH-2, 2026-09-21)
## vs A (data & optimisation)
- OBJECTION: A reads cos(Δ,Δprev) 0.15→0.03 as "the step is spent on noise", but PARAM_DISTANCE §2's own
  next bullet — "The centre drifts; it is not a noise ball", drift/segment slowing 6.3→1.3 — is a
  convergence signature: a self-play pool at FIXED 320 sims is a near-stationary teacher, and coherence
  can fall because the net approaches that target's fixed point, which no LR/EMA knob buys back.
  [MEASURED PARAM_DISTANCE §2–3, 4c6759b9…; third reading ARGUED]
- CONCEDE: A is right that R358 armed σ AND augment in one re-mint — my §3's "share fell 62–65 %→22–27 %
  while strength ROSE" cannot credit σ; B's benign chain drops to instrumentation only. [MEASURED STATE
  item 10, R358 re-mint]

## vs C (net & encoding)
- OBJECTION: C's thesis is CARD-STRIX-NET-ONLY's headline ("What is left … is the net and the head") while
  the SAME card's A1 scope note holds the training-side proof target "UNTESTED and UNRANKED, not refuted"
  — exhaustion by Q7.6's "NO" column cannot rank a 50–55 box-h warm-start-forfeit build above its own
  untested training-regime half, whose discriminator (proof-rate/novelty on the mirrored ring) costs
  0 box-h. [MEASURED CARDS.md CARD-STRIX-NET-ONLY A1, 9a2c84ba]
- CONCEDE: C2's 0-box-h calibration probe (dist65 E[v] vs realized z) must run before my P-B4 — head
  calibration and target shape confound — and PARAM_DISTANCE's value-head ‖Δ‖ swings 4.6–10.8 vs policy's
  steady 7.7–9.1 with norm +96 % is the best measured evidence the VALUE half of the leaf's lesson is
  misspecified. [MEASURED PARAM_DISTANCE §2]

## vs D (systems)
- OBJECTION: every D falsifier measures serving (leaves/s via bench_server), none measures D's own thesis
  condition — that steps still teach — which PARAM_DISTANCE's last interval (+0.7 pp per 15k games, inside
  both CIs) says may already be false; until run9's reuse-8 cell reads, D's rows carry schedule rank, not
  §0's Δstrength/box-hour rank. [MEASURED PARAM_DISTANCE §2 "+3.1 pp … then +0.7 pp"; rank ARGUED]
- CONCEDE: D's eval-wall arithmetic is correct and bites P-B3 — a gate round plays the same 256/256 games
  however fast self-play runs (38–40 s/game) and 72 % of run8's wall had a round or cell in flight
  (32.6 % + 39 %), so a self-play sims cut alone cannot reach a ≤ 0.7× wall line. [MEASURED EVAL_COST
  §(ii) + §(iv), b0e7c3de28e069f3]

## vs E (premises)
- OBJECTION: E1's "within ±4 pp ⇒ ruler stable" band equals a 288-game cell's CI half-width (0.142
  [0.104, 0.181] = ±3.85 pp) and exceeds run8's ENTIRE parent-relative gain (+3.1 pp, 0.111→0.142) — the
  cell can certify "stable" a radius bias larger than everything run8 learned, and E names but does not
  price the second anchor that would replace a hot ruler. [MEASURED STATE items 13–15; arithmetic ARGUED]
- CONCEDE: E3 is a B-lane instrument B lacked — F-45's 25/25 α=1.0 rows all sit at moves_remaining == 1,
  exactly the tail-only/mid-turn population my §3 instrument conflates; E3's mr-bucketing is adopted into
  the 27 % decomposer and gates P-B4. [MEASURED falsified.md F-45, 0ac5043b1fc2355e]

### BRIEF C — NET & ENCODING (revised)

# P3-C REVISED — NET & ENCODING — "the net is the ceiling" (PACKET RESEARCH-STRENGTH-2, 2026-09-21, P4 pass)
Base unchanged: worktree `hexo-mantis-rs2` @ `649852dd`; strix numbers are 288-paired-game equal-work
256/256 CONTENDED book_v1 cells. No proposal dropped; C1 re-ranked, C2 amended, C3 strengthened.

## Thesis (revised)
The net and its head are the CEILING — the binding constraint once the ruled knobs (run9 data regime,
run10 LR, R364(b)/(d)) are spent — not the next swap; what C owes the packet NOW is its three
0-box-hour probes (spread series, value calibration, train/held-out gap), which price the run11+ builds.
[REV: A holds the queue through run10 (R364(d) via P0 §2) and §0 counts Δ/box-hour, so the P3 phrasing
"the next strength step is a build" overstated the ordering; the honest claim is binding-constraint-after-
knobs, with probes-first delivery.]

## §0 · PROBES FIRST (new — all 0 box-h on the mirror, decide the run11+ build order)
1. SPREAD SERIES: analyzer over ≥ 12 positions (24×12 d6_lossless corpus) × run8's 18 checkpoints, then
   run9's. 2. CALIBRATION: dist65 E[v] vs realized z (C2's falsifier), bucketed by moves_remaining.
   3. GAP: train-vs-held-out loss on run8's + run9's nets (C3 stage 1). Beside them, co-instrumentation
   adopted from the rebuttal: A's shared-window PARAM_DISTANCE control on run9's checkpoints, and B's
   one-hot decomposer (tail-only rows excluded). [REV: A's P2-step-1 and B's §3 decomposer showed the
   probe-led pattern beats build-led on Δ/box-hour; E3's mr bucketing folds into the calibration probe.]

## C1 · ARCH-D6 — a D6-equivariant trunk behind `identity.arch_kind`
- thesis: unchanged — the net cannot be D6-equivariant by construction (its own features break it);
  building equivariance removes a value-noise floor augmentation only dilutes.
- evidence: unchanged (spread 0.146, R363(d) `7e21c43a`; D6 lossless §Q1.2/App A `699ae9c9`;
  augmentation uniform, STATE item 15 `3b597edc`; SLAP 2301.04746 parity; Equivariant MuZero 2302.04798;
  F-01 static-probe caveat `0ac5043b`).
- swap: unchanged — BUILD, new ARCH_KINDS sibling, NEW registry row, irrep edge features, JK-cat kept.
- run8@45k: warm start FORFEITED; BC restart; comparator run8@45k 0.142 [0.104, 0.181].
- cost: ≈ 50–55 box-hours to equal games + 2–3 dev-days. [REV: D's §3 arithmetic — a 2× serving gain
  degrades to ≈1.5–1.7× effective steps/h with eval in flight — prices C1's ~2×-per-leaf irrep MORE
  harshly than P3 said: the equal-games read pays it on the whole trainer+eval chain, not serving alone.
  Offset if B's P-B3 lands: ×1.6–1.8 games/h buys back roughly half the irrep cost — read C1's rank
  AFTER P-B3's cell. MEASURED(EVAL_COST §iv; PERF3 `fbf3377f`) + ARGUED.]
- success at equal games vs run8@45k: ≥ 0.161 at 45k games. [REV: carries E1's qualifier — if the r6/r8
  ruler cell moves ≥ CI half-width, this and every cell line read "strix@r8" as a stated unit. ARGUED(E1).]
- falsifier < 1 box-hour: conformance test + the §0 spread series (falling spread ⇒ down-rank).
- kills it: spread → 0 with strength flat; the ~2× per-leaf cost on a serving-bound chain; dev failure
  to hold the irrep through 4×128.
- run9 interactions: CARD-ARCH-D6 rules it after the queue; a run9 reuse-8 gain lowers its rank
  (unchanged); [REV: its analyzer witness now also rides run9's checkpoints per §0 — adopted from A's
  shared-window control.]

## C2 · VALUE-SCALAR — strix's scalar tanh head on our trunk
- thesis: unchanged — the one measured net-side asymmetry vs strix is the value head; swap the head,
  keep the trunk. [REV: the swap as written in P3 was incomplete — strix's head also trains on a
  DIFFERENT draw label (draw → 0, §Q7.2 row, verified at HEAD) vs ours (draw_reward −0.5,
  ply_cap_value −0.5, run8.yaml `f600bfb4`), so "MSE on mover-frame z" must FIX the draw label or it is
  not strix's recipe — E2's finding, adopted.]
- evidence: as in P3 (strix scalar tanh + MSE + stone pooling, value_hidden 32, §Q7.2; ours 34 945
  params, norm +96 % while loss fell 0.64 → 0.50, PARAM_DISTANCE `4c6759b9` "filed here, not read";
  F-35/F-37 dense-era precedent only). [REV: A's P1 offers a competing mechanism for the same +96 %
  (flat LR, no EMA — their lane, run10); C2's calibration probe at 0 box-h should run BEFORE A's 17-box-h
  LR arm because a calibrated head kills C2's mechanism without a run — ordering claim, ARGUED.]
- swap: BUILD behind the seam; value loss → MSE on mover-frame z WITH the draw label decided as a named
  row in the mint (0 vs −0.5 is a measured strix delta, not ours to inherit silently).
- run8@45k: PARTIAL warm start (LAW-12 weights-only strip, 267 521 of 302 466 params kept).
- cost: dev 0.5–1 day; ≈ 18–20 box-h to a 15k cell, ≈ 50 to equal games.
- success at equal games vs run8@45k: ≥ 0.161 at 45k games, draw label stated.
- falsifier < 1 box-hour: the calibration probe (0 box-h); [REV: second 0-box-h gate adopted from E3 —
  mr-bucketed calibration: if error at mr=1 ≥ 2× mr=2, the mechanism is mid-turn structure (F-45's
  α=1.0 rows all sit at mr=1, verified falsified.md), and B's λ/proof targets outrank a head-type swap
  in the same run11+ slot. MEASURED(F-45; lib.rs:572).]
- kills it: probe clean AND cell flat; [REV: also killed by E3's reading landing ≥2× at mr=1 with the
  head calibrated — then the target, not the head, is the lever and C2 yields its slot to B's P-B4.]
- run9 interactions: unchanged — read run9's value loss + head norm before minting; one swap per run;
  [REV: C2 and B's P-B4 are COMPETING value-target swaps for the same run11+ slot — the §0 probes order
  them, not queue position.]

## C3 · CAPACITY-PROBE→SIZE — reopen the parked net-size row with the measurement it lacks
- thesis, stage-1 probe, stage-2 rows (gnn_num_layers 4→6 / gnn_hidden 128→192, live keys), costs,
  success line (≥ 0.161 at 45k games), falsifier (stage 1 IS it): unchanged.
- evidence: unchanged (no size-vs-strength curve anywhere, §Q3(b) NOT FOUND; walk ‖Δ‖ ≈ 22/3k flat,
  cos 0.15 → 0.03, norm +49 %; KataGo data-rich rule; Gumbel large-net Elo penalty).
- [REV: strengthened by A's own WON'T — run9 moves capacity AND ratio together (100k→500k AND 3.2→≈7.7)
  so no cell it runs isolates them; the §0 gap probe over run8's vs run9's checkpoints is the ONLY
  instrument that separates capacity from data regime, which raises its priority, not just its scope.
  MEASURED/ARGUED(A §"WON'T: capacity vs ratio").]
- kills it (stage 2): Gumbel frontier, ~2× net doubling per-leaf cost; CPU deploy head 5.0 → ~10 s/stone
  (`CPU_HEAD_PROFILE` `23cca182`); [REV: + trainer duty — F-44's 0.32 s/step is run6's regime and the
  500k ring + burst 8 step cost is unread until run9's 4-h twin (D §4 premise), so stage 2's wall bill
  is unpriced at 2.4 reuse too. MEASURED-ABSENT, ARGUED(D).]

## §3 — what spread 0.146 means (amended)
Unchanged in substance: raw-read VALUE spread, one orbit, ONE position, run8@18k (R363(d),
`2f94bc34` instrument); not a strength number; readings (i) matters / (ii) mostly-doesn't (augmentation
uniform + SLAP parity + argmax 12/12). What decides it: the §0 spread SERIES, ≥ 12 positions before any
correlation. [REV: B's instrument caution transfers — ring_reader's H=0-for-tail-rows shows a scalar
summary can silently conflate two populations, so the series reports the spread DISTRIBUTION per orbit,
not one scalar per checkpoint. ARGUED(from B §3, MEASURED ring_reader.py:135–137).] [REV: run9's
checkpoints join the series — if spread falls on shared-window data while strength moves, the noise
floor tracked the training regime, not the architecture, and C1's rank drops a second way. ARGUED.]

## §4 — unmeasured premises in our record (paths)
1–8 of P3 unchanged (net∧head∧regime confound, JK parity at HEAD, 65-bin never tested, no gap probe,
augment effect unseparated, spread n=1, strix wire semantics undiffed, dense-lineage transfer).
9. [REV: draw-label parity — strix trains draw → 0, ours −0.5 on BOTH draw and ply-cap (§Q7.2 row
   verified; run8.yaml:133–134); never diffed as a VALUE-target variable, only recorded. E2's find.]
10. [REV: the ruler's radius provenance — every cell in this packet, including all of C's, rides
    "strix@r8" at a row neither side trained at (§Q7.1.2; strix_driver.py:67); E1's cell decides
    whether all our numbers carry a unit qualifier. MEASURED(§Q7.1.2) + ARGUED.]
11. [REV: "the step teaches at run9's regime" — the shared premise under A's P1, D's whole conversion,
    and C's equal-games cost accounting: +0.7 pp last interval, inside both CIs, cos ≈ 0.03
    (PARAM_DISTANCE §2); no producer until run9's 36k cell. MEASURED(absent).]

## §5 — weakest point (revised)
P3's weakest point stands and sharpens: the ceiling claim rests on EXHAUSTION (Q7.6's every-other-row
"NO"), not on a net-side measurement, and run9's data regime alone can move 0.142 without touching a
weight shape. [REV: two of the §0 probes are aimed at MY OWN lane — a calibrated dist65 demotes C2, an
underfit-null demotes C3 — so the revised brief is self-falsifying at 0 box-h, which is the honest
state; and E1 can qualify every number cited in it, mine included. ARGUED.]

#### Objections by C to the other four

# P4-C OBJECTIONS (PACKET RESEARCH-STRENGTH-2, 2026-09-21) — C vs A/B/D/E
## vs A — DATA & OPTIMISATION
- OBJECTION: P1's "flat step + falling coherence = noise" cannot separate noise from directed
  value-head growth: the same table concentrates the walk in ONE module (value per-segment ‖Δ‖ swings
  4.6–10.8 vs policy's steady 7.7–9.1; value norm +96 % while value loss FELL 0.64→0.50) — head
  parameterisation (C2, 0-box-h falsifier), not step size; A's own premises admit no experiment links
  coherence to strength. MEASURED(PARAM_DISTANCE_4c6759b9 §2–§3).
- CONCEDE: the queue and cost ladder are A's — R364(d) rules LR run10's, so 17 box-h of rows outrank
  any 50-box-h build on Δ/box-hour until a net-side probe says otherwise; A's shared-window
  PARAM_DISTANCE control on run9's checkpoints is the right 0-box-h instrument, adopted by C.
  MEASURED(P0 §2; RUN9_PREREG §3) + ARGUED.
## vs B — TARGETS & SEARCH
- OBJECTION: P-B1 aims at a population B's own §3 reads "mostly benign + instrument conflation" —
  `explicit_entropy` counts tail-only rows (the ring's SOFTEST targets) as H=0, so 0.27 is an upper
  bound, and the only measured co-movement of target shape with strength (σ: 62–65 %→22–27 % while
  cells ROSE 0.104→0.142) runs AGAINST the defect story; by B's own logic the 0-box-h decomposer, not
  a 20-box-h head, is the next act. MEASURED(STATE 3b597edc items 10/13–15; ring_reader.py:135–137).
- CONCEDE: P-B3's two-part success line (equal-games ≥ parent AND ≤ 0.7× wall) is §0's per-box-hour
  rule stated cleanly, and on a serving-bound box (F-47 0.515 ms/leaf; PERF3 fbf3377f 3 208 leaves/s
  alone) a ×1.6–1.8 games/h cut is real progress; STRIX_RUNG c28708a7 §B.2 "the sims are not the
  lever; the net is" binds P-B3 only as a PLAY-time reading — verified at HEAD. MEASURED/ARGUED.
## vs D — SYSTEMS
- OBJECTION: D's title metric — learning per box-hour — has NO measured numerator at HEAD: the last
  interval is +0.7 pp per 15k games inside both CIs (0.135 [0.097,0.177] → 0.142 [0.104,0.181]) with
  cos(Δ,Δprev) ≈ 0.03 (≥ 80 % orthogonal walk), and every D falsifier gates leaves/s while none gates
  strength-per-step; D's own §5 concedes the rank drops if run9 lands flat, yet D-1/D-2's success
  lines still cite strength cells. MEASURED(PARAM_DISTANCE 4c6759b9 §2).
- CONCEDE: the eval wall is measured and large (51 % of run8's wall in flight; 356–415 vs 999–1 069
  steps/h in a cell), and experiment RATE is a packet good — D-3's isolation would shorten C1's
  50-box-h read by the same ≈1.5–1.7× D prices for itself, and D-1's 0.2-box-h bench falsifier is
  the cheapest in the packet. MEASURED(EVAL_COST b0e7c3de §iv; STATE item 15; PERF3 §5).
## vs E — PREMISES
- OBJECTION: E1 buys a qualifier, not a decision — even a ≥ CI-half-width r6/r8 move annotates every
  series point without moving run10's queue, because no deploy-matched alternative ruler exists
  (sealbot saturates 0.54–0.71 under the ≈0.78 book ceiling; self-referential gates F-30/F-48-flagged)
  and E's own §5 leaves the settling anchor named but unpriced. MEASURED(RUN7_EVAL_COST 03ddcd1a §B).
- CONCEDE: E3 is a fact my P3 brief missed — F-45's α=1.0 rows sit exactly at moves_remaining == 1
  (verified at HEAD, falsified.md 0ac5043b F-45), the one scalar carrying two-stone semantics; a ≥2×
  calibration error there hands C2 a measured mechanism, and E2's draw-label diff (strix draw → 0 vs
  our −0.5, §Q7.2 row, verified) exposes a real confound in C2 as P3 wrote it.

### BRIEF D — SYSTEMS (revised)

# P3 BRIEF D (REVISED) — SYSTEMS: learning per box-hour (PACKET RS2, 2026-09-21; rev 1 after A/B/C/E)

## 1. Thesis
Throughput is a strength lever exactly insofar as steps teach: run8 moved its strix series +3.8 pp
per 30k steps at 871 steps/h averaged (STATE item 15; PARAM_DISTANCE §2), so every × on leaves/s
converts through steps/h — while the server thread stays the bound (MEASURED, PERF-3 step 1) and
while the step is signal, not noise. [REV: that gate is not D's to test — A's P2 step-1 probe
(≈1.5 box-h) / run9's reuse-8 read, C's spread series (0 box-h), E's E1/E2 own it; D EXECUTES the
conversion, the others own its premise. Every lever below is conditional on those reading clean.]

## 2. Proposals

### D-1 FILL-AT-THE-KNEE — lift batch fill (workers × leaf_batch) at the B step 3 names
- thesis: the fill term is the cheap half of the cycle — alone B 46 of 64 (fill 72%; live 75%),
  `queue_wait` 2.96 ms = 20% of the 14.44-ms cycle, and in-flight supply caps B at
  ≈ workers × leaf_batch / 2 = 128. MEASURED PERF3 (B 46.3, saturation 85%, supply cap); MEASURED
  PERF_A4 §4.2/§9 (+14% w32ct); MEASURED STATE item 15 (3,193 leaves/s, fill 75%). F-47's "+5% at
  32 workers" is the PRE-pipeline tree, not a ceiling for the pipeline at HEAD.
- [REV: SUBORDINATED to B's P-B3 — at 128 sims the per-worker pop rate ≈ doubles, in-flight
  supply rises, fill improves with NO worker change; P-B3 delivers part of D-1's claim at 1–2
  dev-h. Sequence: P-B3's 3-h shakedown first (it re-measures games/h at the same operating
  point), then D-1's knee ON TOP. D-1's residual claim: the fill between P-B3's supply gift and
  the knee.]
- swap: rows — `selfplay.n_workers` 32→48, `selfplay.leaf_batch_size` 8→12, `inference.
  inference_batch_size` 64→ the step-3 knee. Verify at HEAD: `saturation_threshold = batch_size /
  2` (crates/mantis-selfplay/src/queues/graph.rs); the run8.yaml rows named.
- cost: ≈ 0.75 box-hours (3 bench_server cells × 220 s) · ≈ 2 dev-hours.
- conversion: leaves/s→steps/h ≈ 1:1 only while the CPU stage stays the bound (PERF-3 §5); at
  run9's 2.4 steps/game a 2× serving puts trainer duty ≈ 43% of wall (ARGUED from F-44's 0.32
  s/step, run6 regime) — decays near ~1.7×. steps/h→strength: +3.1 pp per 15k games
  decelerating to +0.7 (PARAM_DISTANCE §2).
- success: at EQUAL GAMES (15k) vs parent run8@45k (0.142 [0.104, 0.181], 288 paired games): the
  cell inside that band AND ≤ 0.7× run8's 17.2 box-hours per 15k games.
- falsifier < 1 box-hour: `tools/bench_server.py --workers 48 --batch-sizes 64,128` (≈ 0.2 h) —
  b_mean ≤ 50 or leaves/s ≤ 3,208 × 1.1 ⇒ fill is not the lever.
- kills it: step 3's curve flat above B 64; trainer duty at 2.4 reuse capping steps/h; [REV:
  P-B3 landing first and leaving fill ≥ 85% at 128 sims.]
- interactions: CARD-PERF-3's rule — NO design before step 3's numbers; step 3's window RESERVED
  in run9's preflight; nothing rides run9; rows land run10+.

### D-2 EDGE-CUTOFF — wake the pop on a fused-edge budget, not a graph count
- thesis: our pops vary 2×+ in edge load (80.2% in [512k, 1M) fused edges, 8.1% ≥ 1M against the
  1,373,143 cap) while the pop rule counts graphs — the one serving pattern strix holds, we lack.
  MEASURED PERF3 §6 (the buckets, `fusion_splits` 0.06%); SOURCED strix `max_batch_edges =
  45000`, "+35 % A/B" — a config comment, no record; ARGUED: our per-pop load is 15–30× past it.
- [REV: GAINS A CONSUMER — C. C1/C3 price a ×2 per-leaf model at "games/h halves" 1:1, but
  PERF3 §5 has device 9.36 ms OVERLAPPING CPU 11.46 ms in the 14.44-ms cycle: a device-side ×2
  lands ≈0.65–0.75× leaves/s. Step 3's B-curve and D-2's 0-box-h regression now price C's build
  tax too. The regression's ±6% spread stays WEAK; a null is inconclusive — say so.]
- swap: build — an edge-budget wake beside the count wake in `pop_graph_batch_blocking`; verify at
  HEAD: `fused_batch_edges.total`, the `_pow2_bucket` histogram, `inference_batch_size`.
- cost: ≈ 0.2 box-hours · ≈ 6–10 dev-h. Conversion/success: as D-1.
- falsifier < 1 box-hour: 0 box-hours — regress the 24 step-1 windows' gpu_wait/pop against
  edges/graph (14.9k–16.8k spread, mirror JSONL pulled); flat ⇒ nothing to collect.
- kills it: device stage flat in edge count at our graph size; strix's knee never measured at
  16k-edge graphs.
- interactions: waits on step 3's ms/batch; nothing rides run9.

### D-3 OFF-BOX EVAL — a second card (or box) for gate rounds and strix cells
- thesis: eval was 51% of run8's wall in flight and costs the trainer 24–32% (rounds) / 60–64%
  (cells); cadence 36k and the rung deletion are landed — the remaining lever is isolation.
  MEASURED EVAL_COST §(iv) (999–1,069 steps/h alone, 682–724 in a round, 356–415 in a cell; 72%
  of wall with eval in flight); MEASURED STATE item 15 (51%, 871 avg).
- [REV: STRENGTHENED twice. (1) B's P-B3 and every D serving gain RAISE eval's share: per 15k
  games the trainer-alone wall falls while the round+cell wall (≈3–5 h) is per-GAME and
  unchanged — the faster the box, the more eval binds; D-3 is the only lever on that term.
  (2) A's trainer-duty warning at 2.4 reuse cuts the serving-side term the same direction.]
- swap: build — the eval child and follower cell on a second device/endpoint, a MINTED row, never
  a default (LAW-11). Cost: hardware money + ≈ 0.5 box-hours + ≈ 4–8 dev-hours.
- conversion: schedule, not serving: effective steps/h → the alone rate, +15–25% averaged
  (ARGUED from 871 vs 1,035, EVAL_COST §iv).
- success: at equal games vs run8@45k: the 36k cell inside 0.142's band, wall ≤ run9's expected
  wall − 30%.
- falsifier < 1 box-hour: the PERF-3 contended-arm protocol at an OFF-box consumer (one 220-s
  bench_server cell beside it): leaves/s ≥ 0.95× alone ⇒ isolated.
- kills it: contention is CPU-side (PERF-3 §4: every CPU term ×3–7, GPU +2 ms) — a second CARD
  on the same 12-core box may not isolate; a different BOX may be the only true isolation, and
  that split is unmeasured; our GNN on non-CUDA is unmeasured. The follower rides run9 at
  `--cadence 36000` from START; arming D-3 re-labels every future cell CONTENDED/IDLE.
- interactions: none with run9's rows.

## 3. What a 2× buys
[REV: rows first — the cheapest multiplier on the table is B's P-B3 (rows, 1–2 dev-h, ×1.6–1.8
games/h claimed) ahead of any D lever at equal confidence; D's honest stack is P-B3 → step-3
knee (D-1) → edge cutoff (D-2), each multiplicative on serving, ALL capped by the same trainer
duty at 2.4 reuse.]
A 2× leaves/s is NOT a 2× steps/h: eval's wall is per-GAME (a gate round plays the same 256-sim
games however fast self-play is), so the round-in-flight share RISES as serving speeds. ARGUED on
EVAL_COST's rates: per 15k games the trainer-alone wall halves (≈ 15 h → 7.5 h) while the
round+cell wall (≈ 3–5 h) is unchanged → effective ≈ 1.5–1.7× steps/h; with D-3 the full 2×
holds. At run8's slope that is +3.8 pp in ≈ 60% of the box-hours; at the LAST interval's slope
(+0.7 pp per 15k, inside CI) it is indistinguishable from noise — the deceleration is the
conversion's own warning. If the LR/data regime is the true binder (PARAM_DISTANCE: cos(Δ,Δprev)
0.15→0.03 — the 3k step is ≥ 80% noise walk), a 2× buys NO strength per step: it buys experiment
rate and the option to spend the same box-hours on 2× sims per leaf (B's lane). [REV: E2 can no
longer re-price the slope much — run8's FULL-SPAN rates are on record (STATE item 15: draws 0.5%,
ply-cap 0.53%, 53,084 games), far under E2's own 3% bar; the windowed series is E's refinement.]

## 4. Unmeasured premises
- "A second card isolates the trainer": no measurement anywhere; contention reads CPU-side
  (PERF3 §4) — D-3's falsifier is the first datum. "A different box moves the CPU stage": the
  server is ONE thread (F-47: 90.6–99% busy); CPU_HEAD_PROFILE §3 rules out core COUNT, never
  clock — no bench on another host.
- "Steps/h converts at run8's slope past 45k": the last interval is +0.7 pp inside both CIs
  (PARAM_DISTANCE §2). Path: the run9 36k cell.
- "Trainer duty stays sub-bound at 2.4 reuse": F-44's 0.32 s/step is run6's regime (1.0 steps/
  game, 100k ring); the 500k-ring per-step cost is unread until run9's 4-h twin. [REV: A's every
  box-h price inherits this same unread number — the shared denominator, flagged to A in P4.]
- "strix's +35% edge-cutoff transfers": config comment, no record; our per-pop load sits 15–30×
  past that knee.
- [REV: "Augmented steps teach like raw rows": run8 swapped σ AND augment together (R358; C §4.5)
  — never separated; steps/h counts ×8-augmented rows at full value, so if augmentation dilutes,
  the step→strength conversion overcounts. Path: none named; C's spread series is adjacent, not
  a test.]
- [REV: "×2 per-leaf ⇒ ×2 wall": C1/C3's kills price the conversion 1:1; the PERF3 §5 overlap
  (9.36 vs 11.46 ms) bounds it at ≈0.65–0.75× on the device side (ARGUED); step 3's curve is the
  missing instrument.]

## 5. Weakest line
The conversion itself: every D lever delivers steps the PARAM_DISTANCE record says may be noise.
[REV: it is now THREE gates, all owned by others and all cheaper than D's cheapest falsifier:
A's P2 step-1 probe and run9's reuse-8 read decide whether steps teach; C's spread series decides
whether a net noise floor caps them; E1 decides whether the slope's units are stable. D's honest
sequencing for run10: those read FIRST. If any kills the conversion, this lane's surviving output
is instrumentation — step 3's B-curve and the bench protocol, which B (games/h at 128 sims) and
C (per-leaf build tax) consume regardless — and the strength rank drops below A's and B's.]

#### Objections by D to the other four

# P4 REBUTTAL — D (SYSTEMS) vs A, B, C, E (PACKET RESEARCH-STRENGTH-2, 2026-09-21)
## vs A (DATA & OPTIMISATION)
- OBJECTION: the knobs-beat-builds ranking is computed on an unmeasured denominator — the only
  per-step cost on record is F-44's 0.32 s/step at 1.0 steps/game, 100k ring, run6 regime
  (MEASURED, falsified.md 0ac5043b), P1/P2 run at run9's 2.4 + 500k ring, and A's own premise
  list names the ≈16 h/15k-games base an ESTIMATE — if duty bites there, all three A prices
  inflate together and the ordering flips (ARGUED).
- CONCEDE: A's "equal-games ≠ equal-steps ≠ equal-box-hours, wall rows are the guard" is the
  stricter form of D's own rule (MEASURED basis, EVAL_COST §iv: 356–415 vs 999–1,069 steps/h),
  and A's P2 step 1 (≈1.5 box-h, offline) tests the shared step-is-noise premise (MEASURED,
  PARAM_DISTANCE §2 cos 0.15→0.03) cheaper than any D falsifier — D waits on it (ARGUED).
## vs B (TARGETS & SEARCH)
- OBJECTION: P-B3's ×1.6–1.8 games/h is not ×1.6–1.8 progress — eval's cost is per-GAME: every
  15k games carries a round+cell wall ≈3–5 h at 0.68×/0.38× trainer cost (MEASURED, EVAL_COST
  §ii/§iv: 72 % of run8's wall had eval in flight) that B never prices; T/1.7+E vs T+E at
  T≈15 h, E≈3.7 h ⇒ ≈×1.5 steps/h, trimmed further by 2.4-reuse trainer duty (ARGUED).
- CONCEDE: P-B3's two-part success line (equal-games cell AND ≤0.7× wall) is the packet's
  cleanest §0 progress line, and at 1–2 dev-h for the largest multiplier on the table it
  outranks D-1 on expected Δ/box-hour — D concedes the sequencing (ARGUED; games/h is B's own
  unmeasured term).

## vs C (NET & ENCODING)
- OBJECTION: C1/C3 price a ×2 per-leaf net as "games/h halves" — but at HEAD the device stage
  9.36 ms OVERLAPS the CPU stage 11.46 ms in the 14.44-ms cycle, B≈46 alone (MEASURED, PERF3
  fbf3377f §5): a device-side ×2 lands ≈19–22 ms ⇒ ≈0.65–0.75× leaves/s, not 0.5× — C overprices
  its own kill by the overlap; the true curve is step 3's, unrun; launch-side CPU added by the
  build would convert 1:1 (ARGUED).
- CONCEDE: C's spread-series repair (≥12 positions over the 18 mirrored checkpoints, 0 box-h —
  the 0.146 is n=1, MEASURED R363(d)) is the cheapest instrument that could prove a standing
  value-noise floor; flat-vs-plateau caps what ANY steps/h buys and ranks C1 above every D lever —
  D's conversion depends on it reading the other way (ARGUED).

## vs E (PREMISES)
- OBJECTION: E2's "only a boot window was ever read" is false at HEAD — STATE item 15 reads
  run8's full span: "self-play 53 084 games … draws 0.5 %, ply-cap 0.53 %" (MEASURED, STATE
  3b597edc item 15; item 14's audit adds cap 0.16 %), both far under E2's own ≤3 % closure bar —
  the headline is answered; the windowed series and −0.5-row count are refinement (ARGUED).
- CONCEDE: E1's ruler-stability strike lands on D too — D's +3.8 pp/30k constant and every D
  success line are quoted in strix-r8 units whose radius provenance D never priced (MEASURED,
  §Q7.1.2: pin saved at placement_radius 6, rung plays 8; strix_driver.py:67) — the ≈1.2 box-h
  cell should run before run10's queue is fixed (ARGUED).

### BRIEF E — PREMISES (revised)

# P3-E (REVISED) — PREMISES: "the target may be wrong" (BRIEF E, PACKET RESEARCH-STRENGTH-2, 2026-09-21)

## 1. Thesis (one line)

All seven premises name real seams, but our record has measured only one (the play-time solver); the other six ride unmeasured statements, and four — ruler radius-provenance, ruler RESOLUTION, cap/draw fire-rates, mid-turn value calibration — have sub-box-hour (or zero-box-hour) falsifiers that should run BEFORE run10's queue is fixed; a premise proposal changes WHAT WE MEASURE. [REV: D — resolution added as a fourth cheap check: D's own §3 concedes the current slope (+0.7 pp/15k games) is invisible to the 288-game cell's ±3.9-pp CI.]

## 2. Proposals / premise-checks (§2 template)

**E1 · RULER-R6 — the ruler's radius premise, one cell.** [REV: unchanged in substance; cost caveat added from D]
- thesis: the strix series' "same game" premise rides a rules row neither side trained at — the pin was saved in a radius-6 stage, every rung plays it at radius 8 — and its sensitivity is unmeasured.
- evidence: MEASURED (STRENGTH_RESEARCH §Q7.1.2: embedded `game_config placement_radius 6`; `tools/strix_driver.py:67` `req.get("placement_radius", 8)`, verified at HEAD; registry `graph_radius 8`); ARGUED (§Q7.6: radius axis "isolated by a measurement? NO").
- swap: NO row — a PREMISE swap changing what we measure: one equal-work cell, ours PUCT-256 `3aef7883…` (run9's ruled parent; R364 NOT at HEAD — via P0 §PREMISES + PARAM_DISTANCE) vs strix 256 with driver `placement_radius` 6, 288 paired games, `book_v1_s20260625_p4`, CONTENDED, beside the r8 cell. Verify at HEAD: strix_driver.py:67; 288-pair unit (LAW-04).
- cost: ≈ 1.2 box-h contended (68–90 min/cell, STRIX_RUN7_60K §B.5) [REV: D — eval/cell wall is per-GAME at 38–40 s/game, EVAL_COST §ii/:138, so D-1/D-3 serving levers do NOT shorten this; the estimate is contention-regime-dependent, re-priced if D's rows land]; ≈ 2 dev-h.
- success line: re-defined instrument, `strix-r6 equal-work cell`: within ± 4 pp of the r8 cell's 0.142 [0.104, 0.181] (PARAM_DISTANCE §2/:48) ⇒ ruler stable, caveat dropped; a move ≥ CI half-width ⇒ "strix @ r8" becomes a stated unit qualifier on every follower point — including A's, B's, C's and D's success lines, all of which are priced in this same cell unit. [REV: A/B/C/D — the qualifier propagates to all four briefs' bars at once.]
- falsifier: the cell itself, < 1 box-h (idle ≈ 7.5 s/game, §B.5). kills it: driver error at r6 ("a legal-set choice, not a weight mismatch", §Q7.1.2); or Δ inside noise forever. run9: NO row; annotates the follower's series unit (RUN9_PREREG §3).

**E2 · CAP-CENSUS8 — the cap/draw rows' fire-rate over run8's whole span.**
- thesis: `draw_reward −0.5`, `ply_cap_value −0.5`, `max_game_moves 256` (run8.yaml:133,134,141) are armed and inherited verbatim by run9 (RUN9_PREREG §1a), but only a boot window was ever read (draw 1.7 %, ply-cap 2.7 %, 600 games, STATE:222); F-52's 0.11 → 0.86 climb was the PUCT twin.
- evidence: MEASURED (STATE:122,222; F-52); MEASURED (strix differs on all three rows — `max_moves 300`, draw → `draw_value 0`, scalar MSE, §Q7.2; driver default verified at HEAD); MEASURED (draws 0 in 1 728 strix-cell games — STRIX_RUN7_60K §A).
- swap: no rows — a NEW measurement: windowed `monitor_gates.ply_cap_rate` (event_manifest.md:126) over run8's full span + count of value rows written at `terminal_reason == ply_cap|other_draw` under the −0.5 label. Verify at HEAD: `game_complete.terminal_reason` (game_record.md:56); `train.ply_cap_abort {0.5, 600, 3000}` armed (STATE:122).
- cost: ≈ 0.2 box-h off the mirrored events; ≈ 3 dev-h.
- success line: re-defined instrument, `run8 cap/draw fire-rate census`: ≤ 3 % flat ⇒ the −0.5 rows are inert in training, premise closes "rung-only"; ≥ 5 % or rising ⇒ draw-utility enters run10's queue as a measured lever.
- falsifier: the census itself, < 1 box-h. kills it: a dead mirror, not a dead premise (F-52 read the field in windows). run9: none armed; a hot result queues a RUN10 row, never a mid-run9 edit.
- [REV: A — sequencing: a hot census ALSO contaminates A's own evidence base — PARAM_DISTANCE's value-loss 0.64→0.50 and the coherence series are computed over rows that include any −0.5-labelled games; read E2 BEFORE interpreting run9's checkpoint series, or A's 0-box-h control inherits unread labels.]

**E3 · MR-CALIBRATION — is the compound turn's mid-state a value blind spot?** [REV: B/C — now the SHARED instrument: one mirrored pull, three readings]
- thesis: two-stone semantics live in ONE scalar (`moves_remaining / 2`, crates/mantis-graph/src/lib.rs:572, verified at HEAD, node feat idx 3); whether value calibration at mr=1 (mid-turn) matches mr=2 is unmeasured — the one artifact found sits exactly there (F-45: all α=1.0 rows at `moves_remaining == 1`).
- evidence: MEASURED (lib.rs:572; F-45; per-ply `root_value` on eval + sampled self-play since R355(d), game_record.md:81–85); SOURCED (Yen & Yang 2011, P1 §3a: Connect6's two stones as two STAGES — our threats are inputs, not turn structure).
- swap: no rows — a NEW measurement over run8 eval games + sampled self-play shards: |root_value − z| and policy entropy bucketed by root moves_remaining (mr from ply index, game_record.md:57). [REV: B — MERGED with B's §3 decomposer: the same mirrored pull also recomputes one_hot_share excluding tail-only rows split BY mr (B's reading); one pull, two readings, no second shard pass.]
- cost: ≈ 0.5 box-h; ≈ 4 dev-h. [REV: B merge — dev-h shared, marginal cost to B ≈ 0.]
- success line: re-defined instrument, `mr-bucketed value calibration`: equal within CI ⇒ one scalar suffices, premise closes; ≥ 2× error ratio at mr=1 ⇒ B's λ/proof lane gets a named measured target. [REV: C — and C2's kill condition: C's calibration probe must bucket by mr or a pooled E[v]|z reads clean while the mr=1 bucket stays broken (F-45); E3's census IS that bucketed probe — C2 cites it or its "clean" branch kills nothing.]
- falsifier: the census itself, < 1 box-h. kills it: sampled shards too few for a CI (eval channel alone answers at lower power — stated). run9: none; this IS the "new measurement" two R358(e)-parked items require.

**E4 · RULER-RESOLUTION — a 0-box-h power calculation on the cell unit.** [REV: NEW, from D — D's bar cannot see its own effect]
- thesis: the 288-paired-game cell's CI (±3.9 pp on 0.142 [0.104, 0.181], PARAM_DISTANCE:48) cannot resolve the effects the systems lane prices: at the last interval's slope (+0.7 pp per 15k games, PARAM_DISTANCE §2) resolving +0.7 pp needs ≈ (3.9/0.7)² ≈ 30× the games ≈ 8 700 pairs ≈ 34–45 box-h of cells (68–90 min/288).
- evidence: MEASURED (PARAM_DISTANCE:38,48 CI + slopes); MEASURED (cell walls, STRIX_RUN7_60K §B.5); ARGUED (D §3 itself: "indistinguishable from noise").
- swap: no rows, no box — arithmetic on the record, stated once: which decisions the cell CAN support (≥ ~4 pp moves — A's P1 bar, C's builds) and which it CANNOT (D's "same strength at ≤ 0.7× wall" — a non-inferiority read inside ±3.9 pp cannot exclude a real 2–3-pp regression; D's lane needs wall-tracked progress or a cheaper equivalence instrument, named not assumed).
- cost: 0 box-h, ≈ 1 dev-h. success line: the statement itself in RUN9_PREREG's unit section. falsifier: none needed — it IS the check. kills it: a derivation error. run9: annotates every cell's stated decision-resolution.

## 3. The seven premise questions — verdicts (tag; what moves it)

1. **beat-strix ruler — ARGUED** (right by elimination), residue unmeasured — TWO unmeasured halves now: radius provenance (MEASURED §Q7.1.2 pin@r6 vs driver@r8; solver not the gap, 0.111 vs 0.115, CARD-STRIX-NET-ONLY; sealbot saturates 0.54–0.71 under the ≈ 0.78 book ceiling, RUN7_EVAL_COST §B) and RESOLUTION (E4). E1 moves the first, E4 the second. [REV: D — resolution added.]
2. **BC warm start vs scratch — ARGUED keep**, cost unmeasured everywhere: MEASURED for (F-49 BC value head 0.413 vs 0.038; step-0 0.083 vs strix, STRIX_RUN7_60K §A); SOURCED against at scale (AGZ "within the first 24 hours", §Q10). [REV: C — C1's D6 build FORFEITS warm start by construction (in_dim moves) and restarts on BC at 50–55 box-h: if C1 ranks up, this premise stops being "parked, unpriced" and becomes run-critical — C1's price IS the first datum on the scratch side.]
3. **curriculum by radius — SPECULATION** as strength (SOURCED: no staged ablation in any source; strix S1 r2 "degenerate", §Q8; KataGo MIXES sizes); MEASURED-adjacent as throughput (F-47). PARKED R358(e); reopens only on a radius-4 shakedown twin's games/h (≈ 4 box-h). (Uncontested by all four briefs — silence is not evidence.)
4. **teacher distillation — ARGUED park**, one shape alive: F-35/F-36/F-37 MEASURED-against (dense era); strix q_head weight 0, unmeasured (§Q4). Alive: proof-as-target — [REV: B — now PRICED with a real falsifier: P-B2's offline proof-rate/novelty/ms-per-root on mirrored ring boards, 0 box-h; the "new measurement" this parking lacked exists on paper — run B's falsifier beside run9 before any build ruling.]
5. **two-stone semantics — MEASURED present** (lib.rs:572) **, MEASURED suspect** (F-45 at mr==1; quiescence override backup.rs:259) **, unmeasured sufficient**. [REV: B — B's §4.4 adds an adjacent seam: visit-sampling sufficiency past ply 10 unmeasured (`gumbel_explore_moves: 0`; strix visit-samples 20 %, isolation "NO", §Q7.6) — compound-turn EXPLORATION joins value calibration as unmeasured; E3's census reads the value half only.]
6. **value from z — ARGUED yes** at system level with a MEASURED boundary: SOURCED strix trains scalar ±1 MSE from z, draw → 0, and beats us (§Q7.2); F-38 MEASURED the boundary (all 33 blind losses tactical, depth-6–8 turns); PARAM_DISTANCE value head +96 % vs loss 0.64 → 0.50, "no line … connects the two". [REV: C — C2 prices the head swap with a PARTIAL warm start (LAW-12 strip keeps 267 521 of 302 466 params) and C itself concedes strix's scalar head stays confounded with its whole regime (§Q7.6 "NO") — head swap is a hypothesis with a cheap probe, not a measured asymmetry.]
7. **ply cap / draw utility / fence radius — MEASURED as armed rows, unmeasured as effects**: rows run8.yaml:133–141; strix differs on all three; F-52 MEASURED the cap as a PUCT-trainer attractor; fence r8 ⇒ ≈ 355 moves (F-51), chosen for corpus coverage, never strength. E2 moves cap/draw; fence only via E1 + the radius twin (verdict 3).

## 4. Unmeasured premises in our record (paths)

1–10 unchanged from P3-E (ruler-as-trained-game; cap/draw past boot; mr calibration; compute parity "positions written"; strix-WR ⇒ server outcomes, 0 of 2 DISTINCT; value-norm meaning; proof-target untested; rung pair ceiling; radius-throughput; warm-start's later cost). [REV: compressed, no content change.]
11. the cell's DECISION-RESOLUTION — no games-count for a throughput-sized effect is stated anywhere; (3.9/0.7)² ≈ 30× needed. [REV: NEW, from D.] Path: none — E4's arithmetic.
12. visit-sampling sufficiency past ply 10 at `gumbel_explore_moves: 0` — strix visit-samples 20 %. [REV: NEW, from B §4.4.] Path: search_drive.rs:747; STRENGTH_RESEARCH §Q7.6.
13. [REV: B — premise 7 gains a citation: B's P-B2 offline falsifier (proof rate, novelty, ms/root on mirrored boards) is now the named cheap instrument; still UNRUN at HEAD.]

## 5. Where this thesis is weakest

I call the ruler "right by elimination" while proposing to re-derive its units: if E1 moves the reading ≥ CI half-width, §3.1 collapses to "right at a rules row we never chose" and every series point inherits the qualifier at once. [REV: D — and the bluntness makes it worse: E4 says the same CI cannot resolve the systems lane's effects, so the ruler may be BOTH unit-unstable and too blunt — two defects, one instrument.] The second anchor that would settle the ruler (strix at another checkpoint, or a strengthened sealbot) is the measurement I name and do not price — and after reading all four briefs, none of them prices it either; that anchor, not any row or build in A–D, is the packet's largest unpriced item.

#### Objections by E to the other four

# P4-E OBJECTIONS — brief E (premises) vs A/B/C/D (PACKET RESEARCH-STRENGTH-2, 2026-09-21)

## vs A — DATA & OPTIMISATION
- OBJECTION [ARGUED, regime=net/training-rows]: A adopts strix's setpoints (LR 2e-4→2e-5, batch 512, reuse 8) as the healthy target, but strix also writes labels our run never writes — `draw_value 0` vs our −0.5, `max_moves 300` vs 256, scalar MSE (MEASURED STRENGTH_RESEARCH §Q7.2; run8.yaml:133,134,141) — so P1/P3 tune optimizer knobs on a target whose LABELS are still unread; if E2's census reads ≥5 % of rows carrying the −0.5 label strix never writes, no LR floor recovers them — the labels are upstream of every knob A mints.
- CONCEDE [MEASURED, regime=net/steps]: run9's ring-turnover arithmetic is real — 500 000 ÷ 80.3 rows/game ≈ 6 227 games ⇒ turnover every ≈14 900 steps vs run8's ≈1 250 (PARAM_DISTANCE §3, 4c6759b9…) — so the turnover confound gets its reading from run9 alone, whether or not A's rows ever arm; that is one premise of mine the record will close for free.

## vs B — TARGETS & SEARCH
- OBJECTION [ARGUED + MEASURED, regime=net/value-rows]: B's lane changes WHAT the leaf teaches (P-B2 proof targets, P-B4 λ-mixing) without first reading the leaf's CURRENT calibration — the one artifact on record sits at mid-turn (F-45: every α=1.0 row at `moves_remaining == 1`, MEASURED falsified.md 0ac5043b…), and P-B4 would mix a possibly miscalibrated mid-turn root value into every target while its own kill line (median |λ−z|) is only readable AFTER the M–L recorder build; E3's census (≈0.5 box-h) is upstream of both builds and B never sequences it.
- CONCEDE [MEASURED, regime=instrument]: B's instrument finding guts the band premise — `explicit_entropy` counts a tail-only row (α = 1, no explicit mass) as H = 0 "as a one-hot does" (ring_reader.py:135–137, verified at HEAD), so the 27 % is an UPPER bound; and B's 0-box-h decomposer splits BY moves_remaining — the same mirrored pull E3 needs: one pull, two readings.

## vs C — NET & ENCODING
- OBJECTION [ARGUED + MEASURED, regime=net/value-head]: C2's 0-box-h kill ("dist65 E[v] vs realized z ⇒ a calibrated 65-bin kills the mechanism") reads a POOLED scalar — a pooled E[v]|z can read clean while a mid-turn bucket stays broken (F-45 at mr == 1, MEASURED); without moves_remaining bucketing the probe's "clean" branch kills nothing, and E3's census IS the probe C2's kill condition actually needs.
- CONCEDE [MEASURED, regime=instrument]: C's three sub-box-hour falsifiers (d6_lossless conformance, the spread SERIES over ≥ 12 positions repairing the n=1 reading, the calibration probe) are E's own method landed in the net lane — E will not re-price them — and C's §4.2 repair (JK-cat already concatenated at HEAD, gnn_v2.py 93a4811e) is a premise-kill in E's genre: adopted.

## vs D — SYSTEMS
- OBJECTION [MEASURED, regime=game/cell]: D's bar cannot see its own effect — every success line reads "inside 0.142 [0.104, 0.181]" (PARAM_DISTANCE:48 — CI half-width ≈ ±3.9 pp) while D's own conversion says the current slope is +0.7 pp per 15k games, "indistinguishable from noise" (D §3); resolving a +0.7-pp effect at that CI needs ≈ (3.9/0.7)² ≈ 30× the games ≈ 8 700 paired games ≈ 34–45 box-h of cells — a "same strength at less wall" read inside a ±3.9-pp CI cannot exclude a real 2–3-pp regression; the ruler's RESOLUTION is an unmeasured premise before any worker row mints.
- CONCEDE [MEASURED, regime=game/wall]: D's per-game eval-wall arithmetic is correct and binds E too — eval plays fixed 256-sim games at 38–40 s/game (EVAL_COST §ii/:138), so serving levers do NOT shorten cells: E1's 68–90 min/cell estimate (STRIX_RUN7_60K §B.5) is contention-regime-dependent and must be re-priced beside D-1/D-3, never assumed.

---

## P5 — AUDIT (fresh agent; a deliverable on its own)

# P5 AUDIT — PACKET RESEARCH-STRENGTH-2 (2026-09-21)
Repo at HEAD `649852dd` (rs2 worktree, clean); the freshest record is the main checkout's UNCOMMITTED
mint tree. Method: every load-bearing MEASURED tag in the REVISED briefs traced to its cited path+sha
(all P0 §record shas re-hash and match, except PARAM_DISTANCE — L1-7); code cites opened at HEAD.

## LIST 1 — MIS-TAGGED CLAIMS (revised briefs govern; objections checked where load-bearing)

1. **A REV, "What run9's reuse-8 WILL show":** "WILL (pre-registered, RUN9_PREREG 3e40594afb6a7320
   §3): … SUCCESS ≥ 0.192 @30k-games, FALSIFIED ≤ 0.142 at 15k AND 30k, else read 45k THEN STOP; the
   shakedown's replay_ratio ∈ [7,9] (§1b rule)". At that sha RUN9_PREREG contains NO 0.192, no ≤ 0.142
   run9 line, no [7,9] band; §1b is "BLANK BY DESIGN", §3 is the stream section — the lines exist only
   in the UNCOMMITTED main-checkout rewrite (:114–115, :50). SHOULD BE: ARGUED/proposed, or cited to
   the uncommitted file with its own sha — a proposed success line laundered as pre-registered, and
   A's P1 bar ("≥ 0.192 at 15k GAMES") builds on it. Worst mis-tag in the packet.2. **E REV, E2 thesis:** "only a boot window was ever read (draw 1.7 %, ply-cap 2.7 %, 600 games,
   STATE:222)". False at HEAD: STATE item 15 reads the full span (53 084 games, draws 0.5 %, ply-cap
   0.53 %) and item 14's audit cap 0.16 %. P4-D sustained exactly this objection; the revision did not
   amend it. SHOULD BE: struck; the surviving true claim is "no WINDOWED series, no −0.5-row count".
3. **C REV, C1 run9 interactions:** "CARD-ARCH-D6 (CARDS 9a2c84ba) rules it AFTER the queue's knobs".
   No ARCH/D6/equivariance card exists in CARDS.md at that sha (grep: zero hits); the R358(e) parked
   list (CARDS:228) has no arch row. SHOULD BE: ARGUED — the ordering is the packet's, not a card's;
   a governance cite with no referent.
4. **B REV, P-B3 evidence:** "strix TRAINS at 128/m16 and reads 0.83–0.92 on every run7 net [MEASURED
   STRIX_RUN7_60K §A, 288 games/cell]". §A holds 0.083–0.170 (run7's WRs); "0.83–0.92" is the exact
   complement (draws 0) but is not in the path; "trains at 128/m16" is STRENGTH_RESEARCH §1b/§Q7.2.
   SHOULD BE: MEASURED(§A, as 0.083–0.170) + SOURCED(§1b) — two claims under one tag whose path
   contains neither number as written.
5. **C REV, C2:** "PARTIAL warm start (LAW-12 weights-only strip keeps representation + policy
   (267 521 of 302 466 params))" — no cited path contains 267 521, 302 466 or the 34 945 value-head
   count; PARAM_DISTANCE (C's own evidence file) says 302 470, STRENGTH_RESEARCH §1b says 302 466.
   SHOULD BE: the counts carry their own derivation path (a checkpoint state_dict count), not PARAM_DISTANCE.
6. **D REV, D-2 falsifier:** "the 24 step-1 windows' gpu_wait/pop against edges/graph (14.9k–16.8k
   spread)". PERF3's 24 windows span 14 953–17 016 edges/graph; the max is misquoted (17.0k → 16.8k)
   on a power claim (±6 % vs ±6.6 %). Minor; the MEASURED path contains 17,016.
7. **All briefs:** every MEASURED(PARAM_DISTANCE, 4c6759b9…) rides a sha that matches NO file on disk
   at audit time — the untracked record now hashes 60ca27c4… (edited by the in-flight mint). Content
   spot-checks PASS (below), but the §0 (path, sha) handshake fails; re-pin before any ruling cites it.
### PASSED spot-checks (coverage; one line each)
- PARAM_DISTANCE §2/§3: ‖Δ‖ 18.7–24.3 flat, cos 0.13–0.20→0.03, norms +49 %/+96 %, value ‖Δ‖ 4.6–10.8
  vs policy 7.7–9.1, strix 0.104/0.135/0.142, +3.1→+0.7 pp, LR 9.97e-4, re-warm dip 0.056@3k; 45k CI
  ±3.85 pp; E4's (3.9/0.7)² ≈ 30× ≈ 8 700 pairs, 34–45 h. PASS (A, B, C, D, E).
- STATE 13–15: one_hot 0.2745/0.2761/0.2705 (+51k 0.2754), h_full 0.209/0.200/0.286, mean 0.465→0.594,
  counter-threat 0.009–0.031 %, residue 0/4–0/8, loss 2.62/2.12/0.50 vs 3.19, 871 steps/h, 3 193
  leaves/s, fill 75 %, 31.0 h = 51 %, 53 084 games, draws 0.5 %/cap 0.53 % (shakedown 22.1 % = item 10,
  nit). PASS (B, C3, D, P4-D).
- EVAL_COST §(ii)/(iv): 38–40 s/game; 32.6 %+39 %=72 %; 999–1 069 / 682–724 / 356–415 steps/h; cell
  walls 4 256–5 776 s (1.2–1.6 h); sealbot 23 readings 0.540–0.733 mean 0.657 sd 0.045, promoted
  0.733/0.670/0.660/0.681 inside rejected 0.540–0.720 (R362(a)'s grounds are real). PASS (B, D, A).
- PERF3: 3 208/1 222 leaves/s, B 46.3 (fill 72 %), CPU 11.46/14.44 ms, GPU 9.36, queue_wait 2.96 (20 %),
  sat 85 %, buckets 80.2/8.1 % vs cap 1 373 143, 15–30× knee; PERF_A4 w32ct 2 144 vs w16ct 1 875
  (+14.3 %). PASS (D-1/D-2, C REV C1).
- STRIX_RUN7_60K: 0.083–0.170, seat-decided 16–28 % vs 37–57 %, sealbot 0.54–0.71, 68–90 min/cell,
  idle 7.0–7.8 s/game, draws 0/1 728; RUN7_EVAL_COST :35–37 rung 44/47/42 %, 1 − s/2 ≈ 0.78. PASS (E1, E §3.1).
- falsified.md: F-44 (92 % idle, 0.32 s/step, run6), F-45 (25 rows all mr==1, ~1/1 000), F-47 (0.515
  ms/leaf, +5 % @32w), F-49 (0.413 vs 0.038), F-51 (≈ 355), F-52 (0.11→0.86); R363(d) spread 0.146 /
  12 elements / run8@18k / argmax 12/12. PASS (A, B, C, E).
- HEAD code/config: run8.yaml:133–141, lib.rs:572, search_drive.rs:747, ring_reader.py:17/135–137,
  strix_driver.py:67, registry 34.76 %, value_targets.py unarmed, trainer/core.py:491–493; CPU_HEAD
  ~20 ms/leaf flat, 5.0 s/stone. PASS (B, C, E).
- P2's corrections held: no revised brief repeats P1's false "9×9-only" Gumbel caution; Go-Exploit's
  "one run each" understatement appears nowhere revised. PASS.

## LIST 2 — THE RECORD'S UNMEASURED STATEMENTS

**(a) "the ≈ 0.78 book ceiling"** — RUN7_EVAL_COST_2026-09-15.md:37; used at STRIX_RUN7_60K §B.4
("under a ≈ 0.78 book ceiling"), E §3.1, P4-C. Backing: :35 rung seat-decided 44/47/42 % (measured —
run7's sealbot-RUNG population, book_v1); 0.78 = 1 − 0.44/2 is arithmetic. Not measured: the ceiling
as a property of the BOOK — §B.4's own sealbot cells span seat-decided 37–57 % (ceiling 0.72–0.82),
and strix cells measure 16–28 % (ceiling ≈ 0.86–0.92). VERDICT: PARTLY — the fraction is measured, of
the rung population; the single constant is not. Close: quote per-population ceilings off the two
existing measurements (0 box-h); nothing new needs to run.

**(b) "trainer idle 92 %"** — falsified.md:56 (F-44). Backing: MEASUREMENT_STARTPATH_2026-09-11.md §C
(py-spy, 32-min burst, run6 regime: 1.0 steps/game, burst 1, 100k ring). Not measured: run8's mint —
`train.augment: true` puts D6 draws on the trainer thread (PERF3 lists it UNSEPARATED) — and run9's
2.4 steps/game + 500k ring. Carried present-tense as fact at STRENGTH_RESEARCH:178 (§Q1(i)), :243
(§Q2(iii) — the load-bearing grounds that run9's reuse-8 swap is FREE) and CARDS.md:468. VERDICT:
PARTLY — MEASURED for run6; UNMEASURED at run8's mint and run9's envelope (D REV's premise 4 concedes
it). Close: run9's 4-h twin's trainer-duty reading (owed, RUN9_PREREG §5) + regime annotations at the
three forward uses.

**(c) "the sealbot rung never separated promoted from rejected rounds"** — RULINGS.md R362(a) (~:97).
Backing: EVAL_COST §(ii) verbatim — promoted rounds 0.733/0.670/0.660/0.681 inside rejected 0.540–0.720
over 23 readings; the deletion WAS read off it (R362(c): "the census having shown it reads nothing
strix does not"; grounds cite EVAL_COST (i)–(iv)). VERDICT: MEASURED — one caveat the ruling omits:
separation is bounded by instrument power (sd 0.045 ≈ one reading's CI half-width; n=5 promoted), so
it is "not separated at this power", not proven undetectable. Close: nothing owed; an E4-style power
note (0 box-h) if ever re-litigated.
**(d) "GNN-2 non-invariance matters"** — RULINGS.md R363(d) (:59–60, "a GNN-2 witness, filed").
Backing: the 0.146 IS measured (one orbit, ONE check position, run8@18k, value units). Not measured:
any tie to strength — no spread series exists (n=1), no noise-injection test, and the same reading's
argmax 12/12 is evidence AGAINST policy-side mattering. "Witness" hedges the ruling, but C1 prices
50–55 box-h off it and no record doc states "no strength tie measured". VERDICT: PARTLY — the number
MEASURED, the "matters" half UNMEASURED. Close: C's own §0 spread series (≥12 positions × 18
checkpoints, 0 box-h) — proposed, unrun at HEAD.
### Own finds (same shape)
1. **The `one_hot_share_full < 0.25` band** (RUN8_PREREG:132, printed PASS/FAIL at every run8/run9
   audit): calibrated on ONE 3-h shakedown ring at step 3 000 (22.1 %, value head in warm-up); no
   outcome-linked measurement ever backed the threshold; the live run reads 0.27 flat at every audit
   and the miss "stops nothing" (R359(b)). UNMEASURED (the threshold; the shares are measured).
   Close: B's decomposer + re-derive, or strike the band.
2. **strix's reuse ≈ 8 as "healthy"** (run9's ONE swap's grounds; STRENGTH_RESEARCH:243; filled
   RUN9_PREREG:50 "strix's setpoint"): measured only as strix's own setpoint — different head (scalar
   MSE), LR floor 2e-5, curriculum with buffer clears, ROCm APU — carried into our Gumbel-320/65-bin/
   flat-9.97e-4 regime with no transfer test. UNMEASURED as a transfer (§Q2 says so itself); run9's
   cells are the first datum and are in flight.
3. **Cross-instrument series joins**: R362(b) (RULINGS:106) reads run8's 3k/6k dip (0.111→0.056,
   equal-work 256/256) as "matching" run7's 9k→15k dip (0.097→0.045 — cell A: ours PUCT-512 vs strix
   128 sims, STRIX_RUN7 instrument line). Two instruments, one ruling-level narrative; the only bridge
   point is the parent's 0.111, measured in both. PARTLY — the bridge is measured, the "matches" is
   ARGUED inside a ruling. Close: state the instrument beside each number, or one 256/256 cell on a
   run7 mid-checkpoint (≈ 1.2 box-h).4. **"`train.augment: true` helps"** — armed by R358 (bundled with the σ swap; never separated),
   inherited byte-equal by run9's mint; the record's grounds were strix-does-it plus AZ's ×8 on
   19×19 Go at 5 000 TPUs (STRENGTH_RESEARCH:178(i)/(iii)) — a regime-crossing carry into a
   GNN/RTX-5080/288-cell regime with no in-regime measurement. UNMEASURED, and armed in production.
5. **"≈ 16 h/15k-games" / "≈ 17 box-h" prices** (RUN9_PREREG's ESTIMATE, inherited by A's P1/P2/P3 and
   D's conversions as if measured — only cell walls 1.2–1.6 h and steps/h regimes are): tag the derived
   prices ARGUED. And **302 466 vs 302 470 params**: STRENGTH_RESEARCH §1b vs PARAM_DISTANCE §1 disagree
   by 4, both cited as exact (C's strip arithmetic; A's "302 470-param net") — the record disagreeing
   with itself where an exact-sounding count is quoted.

## SUMMARY
- Worst mis-tag: A REV's "pre-registered RUN9_PREREG §3 … SUCCESS ≥ 0.192 / FALSIFIED ≤ 0.142 /
  [7,9] band" — none of it exists in the sha'd file (skeleton, "BLANK BY DESIGN"); it is the
  in-flight uncommitted rewrite, i.e. a proposed line carried as pre-registered.
- Most consequential unmeasured statement: F-44's "trainer idle 92 %" used present-tense at
  STRENGTH_RESEARCH §Q2(iii) as the grounds that run9's reuse-8 swap costs nothing — run6 regime,
  unmeasured at run8's mint (augment on the trainer thread) and run9's 2.4 steps/game.
- Stop citing as fact: (a) "the ≈ 0.78 book ceiling" as a book constant — it is the rung population's
  number (strix cells' measured 16–28 % seat-decided implies ≈ 0.86–0.92 there); quote per-population.

---

## P6 — JUDGE

# P6 JUDGE — PACKET RESEARCH-STRENGTH-2 (2026-09-21)
P5 downgrades applied (L1–L7 + List 2). Rule-2 sweep: no proposal mechanically bottomed — nearest
brushes D-1 (F-47 reopen named = PERF3-at-HEAD + step-3 knee) and C1 (F-01 named; its only reopen
is the run itself — held at bottom of builds).

| # | name | grp | Δ/bh (order · box-h · claimed Δ) | conf | carrying tag | falsifier | pre-START |
|---|------|-----|-------------------------------|------|--------------|-----------|-----------|
| 1 | E4 ruler-resolution | PREMISE | high · 0 bh · what the ±3.9-pp cell can/can't certify | high | MEASURED PD:48 CI+slopes (P5 PASS) | is the check (0 bh) | YES — mint in flight |
| 2 | E3 mr-calibration | PREMISE | high · 0.5 bh · ≥2× mr=1 error ⇒ named target for B/C2/A | medium | MEASURED F-45 (25/25 @ mr==1) | 0.5 bh census | YES |
| 3 | C3 probe→size | BUILD 2-stg | high · stg1 0 bh · decisive either way; stg2 55 bh, ≥0.161 | medium | MEASURED-absent §Q3(b) + PD walk | stage 1 IS it (0 bh) | YES (stage 1) |
| 4 | P-B2 proof-as-target | BUILD | med-high · 0-bh falsifier; build 22 bh, 45k ≥0.18 | medium | MEASURED CARD A1 + F-38/39/40 | 0 bh offline | YES (beside run9) |
| 5 | E1 ruler-r6 | PREMISE | med-high · 1.2 bh · unit-stability of EVERY bar | medium | MEASURED §Q7.1.2 + driver:67 | the cell (1.2 bh) | rec: preflight; not gating |
| 6 | P-B1 soft-policy | BUILD | medium · KL 0 bh gates; build 20 bh, 45k ≥0.18 | medium | MEASURED STATE 13–15 h_full + SOURCED KataGo T2 | 0 bh KL | falsifier now; build run11+ |
| 7 | A-P2 EMA probe→row | KNOB | medium · probe 1.5 bh; row 17 bh; "free strength" ≥0.192-class | medium | MEASURED PD §2 cos 0.15→0.03 (sha stale, P5 L7) | 1.5 bh (none <1) | beside run9 |
| 8 | D-1 fill-at-knee | KNOB | medium · 0.75 bh · fill 72→knee ⇒ steps/h (gated premise) | medium | MEASURED PERF3 fill/queue_wait | 0.2 bh bench | YES — step-3 window reserved |
| 9 | P-B3 sims 128 | KNOB | med · bench 0.2 bh; cell 19 bh · ×1.6–1.8 games/h, wall ≤0.85× | low-med | SOURCED §1b strix-128 (P5 #4) + MEASURED F-47/PERF3 | 0.2 bh bench; 3-bh twin | YES (bench, preflight) |
| 10 | D-2 edge-cutoff | BUILD | low-med · 0.2 bh · D-1's conversion, device-side term | low | MEASURED PERF3 §6 buckets; transfer ARGUED | 0 bh regression (weak, ±6%) | no — waits step 3 |
| 11 | A-P1 LR floor | KNOB | medium · 17 bh (floor est.) · slope restore, 15k ≥0.192 | medium; bar ARGUED (P5 #1) | MEASURED PD §2 noise stats | none <1 bh; 0-bh control | no — run10's (R364(d)) |
| 12 | C2 value-scalar | BUILD | low-med · 18–20 bh/15k, ≈50 equal · ≥0.161 | low-med | MEASURED §Q7.2 diff (regime-confounded) | 0 bh probe = E3 bucketed | no — run11+ |
| 13 | P-B4 λ target | BUILD | low-med · 22 bh/15k · 45k ≥0.18 + loss <0.50 | medium; no cheap falsifier | MEASURED PD §2 value ‖Δ‖+norm | none <1 bh offline | no |
| 14 | D-3 off-box eval | BUILD | low-med · money + 0.5 bh · +15–25% steps/h (ARGUED) | low-med | MEASURED EVAL_COST 51%/rates | 0.2 bh — but needs the card | no |
| 15 | C1 arch-D6 | BUILD | low · 50–55 bh · ≥0.161; removes equivariance noise floor | low | MEASURED spread 0.146 n=1; tie UNMEASURED (P5 2d) | 0 bh conformance + series | no |
| 16 | E2 cap-census | PREMISE | low · 0.2 bh · refinement only (headline answered) | LOW (P5 #2 struck) | MEASURED STATE item 15 (0.5%/0.53%) | 0.2 bh census | optional, pre-series-read |
| 17 | A-P3 batch-512 | KNOB | low · 17 bh · same samples, half steps; slope SPECULATION | low | SOURCED parity §Q7.2 + F-44 (run6 regime) | ≈10-min memory smoke (arm only) | no |

**HALT: NO** — MEASURED(STATE item 15, 3b597edc): 53 084 full-span games, "draws 0.5 %, ply-cap
0.53 %", under E2's own ≤3 % bar: the only "rows are wrong" candidate (the armed −0.5 rows) is
measured-inert; all else is better-rows (LR = R364(d), run10's) or unread cost, not a wrong row.
**NEXT BOX-HOUR:** (1) E4 — free, deadline = the in-flight mint; (2) E3's one mirror pull (0.5 bh)
carrying B's decomposer + C2's calibration + A-P2's mr partition; (3) the 0.2-bh bench pair in
run9's reserved preflight window — bench_server at 128 sims (P-B3's pre-check) + the B-curve knee
(D-1's falsifier / CARD-PERF-3 step 3). ≈0.7 box-h total; E1 (1.2 bh) if the window allows.

## Ranking entries — Δ/bh columns in the table above; here: evidence, kills, pre-START
**1 · E4 (PREMISE).** 0 bh, ~1 dev-h; Δ = decision-resolution itself: the ±3.9-pp CI cannot
resolve +0.7 pp (≈30× games ≈ 34–45 bh of cells), so every "inside the band" line gets a stated
resolution. High — MEASURED(PD:38/:48; STRIX_RUN7 §B.5). It IS the falsifier; killed only by a
derivation error. Pre-START YES: lands in RUN9_PREREG's unit section while the mint is in flight.
**2 · E3 (PREMISE).** 0.5 bh + 4 dev-h (shared with B); Δ = a measured target or a closed premise
for three lanes (B's λ/proof, C2's head, A-P2's noise partition). Medium — MEASURED(F-45: 25/25
α=1.0 rows at mr==1; lib.rs:572). Falsifier = itself; killed only by too-few shards (stated).
Pre-START YES: upstream of interpreting run9's checkpoint series (A's REV).
**3 · C3 (BUILD 2-stage).** Stage 1: 0 bh (~10 min/checkpoint), the ONLY capacity-vs-data-regime
separator (A's WON'T: run9 moves both); stage 2: 55 bh rows on UNDERFIT only, ≥0.161 at 45k.
Medium — MEASURED-absent(§Q3(b)) + PD walk. Kills: OVERFIT parks the row measured; Gumbel frontier
+ 2× per-leaf + CPU head 5→10 s/stone kill stage 2. Pre-START: stage 1, free.
**4 · P-B2 (BUILD).** CARD-STRIX-NET-ONLY's untested half; the 0-bh falsifier (proof rate /
novelty / ms-root on mirrored boards) discriminates C's exhaustion premise before any 50-bh
build. Build 22 bh to 15k, 12–20 dev-h; claims 45k ≥0.18, proof ≥3 % of roots at ≤5 % games/h.
Medium — MEASURED(CARD A1 "UNTESTED and UNRANKED"; F-38/39/40). Dead at proof <2 %, ms/root
>50, novelty <5 %. Falsifier NOW beside run9; build run11+.
**5 · E1 (PREMISE).** 1.2 bh contended; Δ = unit-stability of every number in the packet — a ≥
CI-half-width r6/r8 move stamps "strix@r8" on all four other briefs' bars at once (all adopted).
Medium — MEASURED(§Q7.1.2 pin@r6 vs driver@r8; strix_driver.py:67). Falsifier = the cell; killed
by driver error at r6 or Δ-inside-noise-forever (C's dissent). Preflight-recommended, not gating.
**6 · P-B1 (BUILD).** 0-bh KL falsifier (median <0.02 nats ⇒ dead) + decomposer GATE precede the
build; build 20 bh to 15k, ~8 dev-h; claims 45k ≥0.18 with CI lb >0.142. Medium — MEASURED(STATE
13–15: h_full 0.209→0.286 at flat share; σ-credit void per A) + SOURCED(KataGo T2, 1.30×). Kills:
the KL; a 15k cell ≤0.104; decided-and-correct decomposer; F-32's class. Build run11+ (queue vii).
**7 · A-P2 (KNOB 2-step).** Step 1 (1.5 bh, one 288-game EMA cell over run8's 18 checkpoints) is
the cheapest STRENGTH reading in the packet and the cheapest test of the step-is-noise premise D
needs; step 2 rides ≈17 bh only on >0.181 (≤0.142 kills). Medium — MEASURED(PD §2 cos 0.15→0.03;
sha stale per P5 L7, content verified — re-pin). Kills: SWA-cadence mismatch (ARGUED); a null
cell; mr==1-confined gain (E3 buckets the read). Beside run9, mirror.
**8 · D-1 (KNOB).** 0.75 bh to the knee; 0.2-bh falsifier (b_mean ≤50 or ≤1.1× leaves/s ⇒
dead); Δ = fill 72–75 %→knee on a measured supply cap — real physics, strength only through the
steps-teach premise D concedes is not D's to test. Medium — MEASURED(PERF3 fill 72 %, queue_wait
20 %; PERF_A4 +14 %). Kills: curve flat above B 64; duty at 2.4 reuse; P-B3 first at ≥85 % fill.
Pre-START YES: step 3's window is RESERVED in run9's preflight; rows run10+.
**9 · P-B3 (KNOB).** Rows-only, 1–2 dev-h, 19 bh to 15k; claims ×1.6–1.8 games/h (MEASURED by
the 0.2-bh bench, not asserted — D's correction) at equal-games ≥0.142, wall ≤0.85×. Low-medium
after P5 #4 split the tag ("strix trains at 128" = SOURCED §1b; §A holds 0.083–0.170); A's
reuse-8 re-coarsening caveat stands. Kills: 15k ≤0.104 with games/h <1.3×; STRIX_RUNG §B.2; F-52
watch. Pre-START: bench, YES, same window.
**10 · D-2 (BUILD).** 0.2 bh + 6–10 dev-h; the 0-bh regression falsifier is honest but WEAK (±6 %;
D: "a null is inconclusive"), and strix's +35 % is a config comment, no record. Low — MEASURED
(PERF3 §6 buckets), transfer ARGUED. Kills: device stage flat in edge count; the knee unmeasured
at 16k-edge graphs. Pre-START no — waits on step 3. Gains C as a consumer (per-leaf build tax).
**11 · A-P1 (KNOB).** 17 bh — a FLOOR, not an estimate (in-flight 871 steps/h vs alone
999–1,069; D's sustained correction); claims the slope restored, 15k ≥0.192 vs 0.142. Medium on
MEASURED(PD §2 flat ‖Δ‖ with cos→0.03), but P5 #1 downgraded its "pre-registered" bar to ARGUED
(0.192/[7,9] exist only in the uncommitted rewrite); C3-UNDERFIT reverses direction; E2-hot
re-queues it. No falsifier <1 bh. Pre-START no — R364(d) rules it run10's.
**12 · C2 (BUILD).** 18–20 bh to 15k (≈50 to equal games), 0.5–1 dev-day; claims ≥0.161 at 45k
with the draw label a named row (E2's find). Low-medium — the scalar-head asymmetry is MEASURED as
a diff (§Q7.2) but regime-confounded; param counts unpathed, the record self-disagrees (302 466
vs 302 470 — P5 #5). Kills: probe clean AND cell flat; E3 ≥2× at mr=1 yields the slot to P-B4.
**13 · P-B4 (BUILD).** 22 bh to 15k, 10–15 dev-h (M–L: the ring must record root v̂); claims
45k ≥0.18 AND value loss below the parent's 0.50 track. Medium on MEASURED(PD §2: value ‖Δ‖
4.6–10.8 vs policy's steady 7.7–9.1, norm +96 % while loss falls) — but NO offline falsifier
(ring stores no root values), so rule 1 demotes it below equal-claim peers; double-gated (E3,
E2). Kills: bootstrap feedback (F-22…F-33); λ≈z. Pre-START no.
**14 · D-3 (BUILD).** Money + 0.5 bh + 4–8 dev-h; claims +15–25 % effective steps/h — ARGUED (871
vs 1,035) — on an unmeasured isolation premise (contention reads CPU-side; a second card on the
same 12-core box may not isolate). MEASURED(EVAL_COST: 51 % of wall, 356–415 vs 999–1,069 steps/h)
carries the term's SIZE only; the 0.2-bh falsifier presupposes the purchase. Pre-START no.
**15 · C1 (BUILD).** 50–55 bh + 2–3 dev-days, warm start forfeited; claims ≥0.161 at 45k
games. Low: spread 0.146 MEASURED at n=1, the "matters" half UNMEASURED (P5 2d); best external
= parity (SLAP); P5 #3 struck its governance cite (no ARCH-D6 card exists). Rule-2 note: F-01
falsified a hex-native trunk once; C1's reopen IS the run. Kills: spread→0 with strength flat;
the ~2× per-leaf tax (D: 0.65–0.75×); irrep failure. Pre-START no.
**16 · E2 (PREMISE).** 0.2 bh + 3 dev-h. P5 #2 STRUCK the headline ("only a boot window was ever
read" — false; STATE item 15 read the full span), capping it low; the survivors (windowed series,
−0.5-row count) are refinements whose ≤3 % closure bar the record already meets at 0.5 %/0.53 %.
Remaining value: sequencing — read labels before run9's checkpoint series is interpreted.
**17 · A-P3 (KNOB).** 17 bh; claims the same samples at half the steps (reuse ≈7.66 held). The
weakest evidence in its own brief (SOURCED parity + arithmetic; slope gain SPECULATION) on an
unread denominator (F-44 is run6 regime; P5 List 2(b)). The ≈10-min memory smoke falsifies the
ARM's memory, not the thesis. Kills: P1 first, the smoke refusing, plus the P-B3 window conflict
("P3 first or never"). Pre-START no.

## (a) DISSENT — verbatim, not smoothed
1. A vs B (noise vs fixed point; only run9 separates): B — "A reads cos(Δ,Δprev) 0.15→0.03 as 'the
   step is spent on noise', but PARAM_DISTANCE §2's own next bullet — 'The centre drifts; it is not
   a noise ball', drift/segment slowing 6.3→1.3 — is a convergence signature".
2. C vs B (P-B1 kept behind the gate): C — "by B's own logic the 0-box-h decomposer, not a 20-box-h
   head, is the next act."
3. E vs B (P-B4; gates adopted, the gap stands): E — "P-B4 would mix a possibly miscalibrated
   mid-turn root value into every target while its own kill line (median |λ−z|) is only readable
   AFTER the M–L recorder build; E3's census (≈0.5 box-h) is upstream of both builds and B never
   sequences it."
4. D vs B (P-B3's wall; clause cut to ≤0.85×, the Δ-rank residual): D — "P-B3's ×1.6–1.8 games/h is
   not ×1.6–1.8 progress — eval's cost is per-GAME … ⇒ ≈×1.5 steps/h, trimmed further by 2.4-reuse
   trainer duty."
5. C vs E (E1 kept high by E): C — "E1 buys a qualifier, not a decision — even a ≥ CI-half-width
   r6/r8 move annotates every series point without moving run10's queue, because no deploy-matched
   alternative ruler exists."
6. D vs E (E2; P5 sided with D, revision did not amend): D — "E2's 'only a boot window was ever
   read' is false at HEAD — STATE item 15 reads run8's full span … both far under E2's own ≤3 %
   closure bar — the headline is answered."
7. E vs A (labels; moot at 0.5 %/0.53 %, kept for the record): E — "if E2's census reads ≥5 % of
   rows carrying the −0.5 label strix never writes, no LR floor recovers them — the labels are
   upstream of every knob A mints."
8. A vs B (window conflict; both retained): A — "B-P3 moves target noise the OPPOSITE way (noisier
   completed-Q, more games/h) — the two cannot share a window without confounding; if both queue,
   P3 first or never."
9. B vs C (C1 ordering; C now reads it after P-B3's cell — partial): B — "exhaustion by Q7.6's
   'NO' column cannot rank a 50–55 box-h warm-start-forfeit build above its own untested
   training-regime half, whose discriminator … costs 0 box-h."

## (b) CONVERGENCE — all five, independently
- **Run9's reuse-8 / 36k-steps cell gates every lane's next spend.** A: "if run9 is FALSIFIED … P1
  waits"; B: "run9's reuse-8 cell decides"; C §4.11: "the shared premise under A's P1, D's whole
  conversion, and C's equal-games cost accounting"; D: "D waits on it"; E: "run9's cells are the
  first datum". No brief proposes touching run9's rows; every swap is run10+ or beside-run.
- **The 0/sub-box-hour mirror probes before run10's queue is fixed.** A's six-item list ("None of
  the four briefs contradicts any of the six; all four depend on at least one"), C's §0
  probes-first, B's "the 0-box-h set … is the only honest spend before any run11 queue", D's
  "those read FIRST", E's four cheap checks. Ranks 1–4 are this set.
- **E1's qualifier propagates to all bars** ([REV] markers in A, B, C, D); **E3's mr-bucketing is
  the shared instrument** (adopted by A, B, C) — one mirror pull serves five readings.

## (c) The HALT question — NO
A halt needs evidence the run's OWN rows are wrong. The one candidate (E2's armed −0.5 rows) is
answered by MEASURED(STATE item 15: draws 0.5 %, ply-cap 0.53 % over 53 084 full-span games) —
inert in training, far under E2's own ≤3 % bar. LR is ruled run10's (R364(d)); trainer duty at
2.4 reuse is an unread COST (P5 List 2(b): F-44 is run6 regime), not a wrong row; R364 rules the
swap and its reading gates the queue (all five agree). Two pre-START hygiene items ride with this
verdict, neither halt grounds: land E4's unit note in the in-flight mint; re-pin PARAM_DISTANCE's
sha (P5 L7 — content verified, handshake failed) before any ruling cites it.

---

## ORCHESTRATOR'S CLOSE

Premises this packet questioned and could not close (all tagged NOT FOUND or UNMEASURED in the
phases above): `handoff §5/§9` (no such file at HEAD; STATE.md's Dispatcher section and
RUN9_PREREG §5 are the nearest readings); R364's verbatim text (known only through STATE item 15
and PARAM_DISTANCE's citations); the four record statements the packet named as audit candidates
(P5 verdicts: 0.78 book ceiling PARTLY — a rung-population number quoted as a book constant;
trainer idle 92 % PARTLY — run6 regime, unread at run8's mint; sealbot rung never separated
MEASURED with a power caveat; GNN-2 non-invariance PARTLY — the spread is measured, its strength
tie is not). What this packet does NOT decide: anything — rulings R365+ are the architect's from
P6, P5 and the dissent; the packet cannot arm and did not touch the box, the mint, or the tree.


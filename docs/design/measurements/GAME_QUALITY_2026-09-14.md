# MEASUREMENT — game quality: self-play, the eval games, and the "PUCT exploits sealbot" question (2026-09-14)

The operator's question, verbatim, from looking at games in the viewer: *"the selfplay games etc
really have no good gameplay and it rather looks like our puct is exploiting sth in sealbot or sth
that it does not defend."* This record measures both halves on the GAME-RECORD-1 shards and the
human corpus, with instruments a reader can re-run, and states a verdict on each.

Instrument (all CPU, this machine, 16 cores shared with a foreign `pytest` for most of the run;
scripts under the session scratchpad, listed in `## Sources`):

1. **Exact one-turn tactics** (`tactics.py`, validated against the engine's `winning_moves` ∪
   `threat_moves` on 18 190 human positions, 0 disagreements). Vocabulary, all in 6-cell windows
   along the three hex axes: a **four** (hot window) of P is a window with ≥ 4 P stones and no
   opponent stone — P completes six inside it within ONE turn unless the opponent puts a stone in
   it. **WIN1**: the side to move has a four whose empties fit its remaining stones. **CHECK**:
   the opponent has ≥ 1 four and some ≤ 2-cell set hits every one. **LOST1**: the opponent has
   fours no 2 cells can all hit — a ≥ 3-fold threat, the game is decided whatever is played.
   **missed block**: in CHECK, the turn left an opponent four standing. **missed win**: WIN1 not
   taken. **check-run**: the number of plies from the loser's earliest turn start after which it
   was in CHECK or LOST1 at every one of its remaining turn starts, to the end — the length of the
   forcing sequence the loser could not step out of. Run over EVERY game of every source
   (`run_autopsy.py`, `agg_autopsy.py`).
2. **Stone-purpose fingerprint** (`run_style.py`): each placed stone's first matching class —
   `win`, `block4` (lands in an opponent four), `four` (makes one), `block3` (lands in an opponent
   window with exactly 3 opp stones, a pre-emptive block), `three`, `connect` (adjacent to an own
   stone), `near` (within 2 of any stone), `far` (≥ 3 from every stone).
3. **Judges** (`run_judges.py`, `agg_judges.py`): sealbot d5 (`resolve_bot("sealbot", depth=5)`,
   its pair and its `last_score`) and strix at 128 sims (`resolve_bot("strix", opponent_sims=128)`,
   first stone; second stone only on CHECK/WIN1/LOST1 positions) on turn-start positions sampled
   per source, stratified over five ply bands, never the final winning turn.
4. **Sealbot re-evaluation trajectory** (`run_sealtraj.py`, `agg_sealtraj.py`): sealbot d5
   re-searched at EVERY turn start of all 288 run7 external games (its own turns and ours), giving
   its score, its pair, and whether it reproduces its recorded move — the horizon reading.
5. **Pure continuous-fours solver** (`vcf.py`, `run_vcf.py`): a proof that the winner had a forced
   win by continuous fours from a given turn, within a node budget; UNKNOWN is never a proof of
   safety. Budget-limited on every game (stated where used).
6. **Control** (`run_strix_vs_seal.py`): strix-128 vs sealbot d5 on the same opening book, paired
   colours — an independent opponent's way of beating (or losing to) the same bar.

Sources: the human corpus (8 698 rated decisive games, Elo 753–1 626, median 1 055); run6 (35 287
self-play, 1 105 `external` = the Gumbel-128 deploy head vs sealbot_d5, 1 776 `promotion`);
shakedown7 (2 421 PUCT self-play); the run7 stamp (288 `external` = PUCT-512 vs sealbot_d5 with
`search_stats`, 208 `promotion`, 195 self-play); all 33 frontier cells (run6 checkpoints and the BC
nets at PUCT/Gumbel × 128/256/512 vs sealbot_d5, 288 games each). Every set was read whole for
instruments 1–2; samples for 3–6 are stated per table.

## Verdict (the detail is in §B–§E; every number below is repeated there with its n)

1. **run6's self-play is tactically honest, not degenerate.** Over all 35 112 decisive games:
   0.0 % missed wins-in-turn, 3.6 % of the loser's must-block turns left a four standing and
   0.0 % of the winner's (human losers: 14.1 % corpus-wide, 7.0 % in the ≥ 1300 band), 81 % of games won by a constructed ≥ 3-fold threat
   after a 16-ply forcing run (humans 59 %), 0.5 % draws, p1 0.505, flat over the block. What a
   viewer sees as "no gameplay" is STYLE, and it is measurable: half as many stones connected to
   an own stone as humans (6.9 % vs 13.5 %), more loose stones near the action (24.7 % vs 15.5 %),
   fewer pre-emptive blocks of threes (9.7 % vs 15.7 %), and p2 opening ≥ 6 cells from p1's first
   stone in 28 % of games (69 % for run7's net; humans 12 %). Strix, the independent reference,
   agrees with self-play moves more often than with human moves (28 % vs 15 % of quiet turn
   starts). shakedown7's PUCT self-play IS degenerate (25 % of fours unanswered, 3 % of wins
   missed, rising with the drift); run6's Gumbel self-play is not.
2. **"Our PUCT exploits something sealbot does not defend" — not in the sense meant.** In 288
   PUCT-512 games sealbot missed 0 blocks and 0 wins; 286 games (both directions) end in a
   ≥ 3-fold threat after a forcing run of continuous fours; our runs are 20 plies long, sealbot's
   12. Sealbot enters the run reading itself even-to-ahead (median +1 084) and reports the mate
   8 plies later; a depth-6 sealbot changes its move at that turn in 15 of 20 cases — the
   depth-5 horizon, not a hole. The same motif is how the BC net beats it (0.41), how every run6
   checkpoint beats it, how sealbot beats us back, how 25 % of human games end — and how strix at
   128 sims beats it **30–2** (first reading, n = 32). Sealbot d5 is beatable by continuous fours
   by any competent MCTS player; our 0.507 is a weak instance of that, not a special one.
3. **The finding the operator's viewer impression most likely came from is elsewhere: the Gumbel
   deploy head leaves an opponent four standing in ≈ 3 of 4 must-block turns** (run6's external
   and promotion channels, all 13 Gumbel frontier cells, at 128–512 sims). Those games look like
   nobody is defending because one side is not. PUCT on the same nets: 0–4 %. This is
   CARD-GUMBEL-HEAD-RESIDUE read on the games.

## A. Descriptive

| source | games | decisive | p1 wins (of decisive) | plies median | p10 | p90 | stones placed ≥ 3 from every stone |
|---|---|---|---|---|---|---|---|
| human | 8 698 | 8 698 | 0.502 | 49 | 27 | 113 | 4.0 % |
| run6 self-play (Gumbel 320/64) | 35 287 | 35 112 | 0.505 | 61 | 31 | 131 | 4.4 % |
| run7 self-play (Gumbel 320/64, 195 games at step ≤ 101) | 195 | 195 | 0.421 | 37 | 27 | 77 | 4.8 % |
| shakedown7 self-play (PUCT 320/64, τ 0.5 throughout) | 2 421 | 1 666 | 0.491 | 143 | 31 | 256 | 0.7 % |
| run7 external (PUCT-512 vs sealbot_d5) | 288 | 288 | 0.382 | 35 | 27 | 69 | 6.3 % |
| run7 promotion (PUCT-512 vs the step-0 anchor) | 208 | 207 | 0.488 | 41 | 29 | 87 | 5.7 % |
| run6 external (Gumbel-128 head vs sealbot_d5) | 1 105 | 1 105 | 0.472 | 27 | 19 | 41 | 9.1 % |
| run6 promotion (Gumbel-160 vs anchor) | 1 776 | 1 776 | 0.529 | 21 | 15 | 39 | 12.5 % |
| frontier ck18k PUCT-512 vs sealbot (WR 0.774) | 288 | 288 | 0.469 | 49 | 29 | 91 | 4.5 % |
| frontier ck18k Gumbel-512 vs sealbot (WR 0.066) | 288 | 288 | 0.483 | 25 | 17 | 37 | 9.9 % |

Human games by mean Elo of the two players (the corpus is weak: 5 442 of 8 698 games have a mean
below 1 100):

| mean-Elo band | games | p1 wins | plies median | p90 | ends LOST1 | ends by missed block | ends by missed own win | loser missed-block per CHECK turn | check-run median |
|---|---|---|---|---|---|---|---|---|---|
| < 1000 | 1 981 | 0.496 | 41 | 89 | 45.7 % | 49.7 % | 4.6 % | 22.0 % | 4 |
| 1000–1099 | 2 877 | 0.505 | 47 | 103 | 55.4 % | 41.6 % | 3.1 % | 16.4 % | 8 |
| 1100–1199 | 2 027 | 0.504 | 51 | 113 | 69.9 % | 28.7 % | 1.5 % | 10.2 % | 12 |
| 1200–1299 | 1 229 | 0.487 | 57 | 143 | 69.4 % | 29.0 % | 1.6 % | 10.2 % | 12 |
| ≥ 1300 | 584 | 0.531 | 69 | 165 | 68.3 % | 30.7 % | 1.0 % | 7.0 % | 12 |

The Elo gradient is the instrument's own validation: stronger humans blunder less on every column
and their games end in constructed ≥ 3-fold threats more often, after longer forcing runs.

## B. Tactical honesty, per source (every game, every turn)

"Ends LOST1" = the loser's final turn started in LOST1 (a constructed win: no 2 stones could stop
it). "Ends by missed block" = the loser's final turn started in CHECK with a 2-stone defence
available and it was not played. Rates are per turn / per CHECK turn over all decisive games.

| source | ends LOST1 | ends missed block | ends missed own win | winner missed-win per turn | loser missed-win per turn | loser missed-block per CHECK | winner missed-block per CHECK | check-run median | p90 |
|---|---|---|---|---|---|---|---|---|---|
| human | 59.4 % | 37.9 % | 2.7 % | 1.2 % | 0.9 % | 14.1 % | 4.5 % | 8 | 20 |
| run6 self-play | 80.9 % | 19.0 % | 0.0 % | 0.0 % | 0.0 % | 3.6 % | 0.0 % | 16 | 24 |
| run7 self-play | 79.0 % | 21.0 % | 0.0 % | 0.2 % | 0.0 % | 5.6 % | 0.0 % | 16 | 20 |
| shakedown7 self-play | 30.6 % | 65.9 % | 3.5 % | 3.0 % | 2.2 % | 25.4 % | 10.8 % | 4 | 15 |
| run7 external (PUCT-512 vs sealbot) | 99.3 % | 0.7 % | 0.0 % | 0.0 % | 0.0 % | 0.2 % | 0.0 % | 16 | 20 |
| run7 promotion | 94.2 % | 5.8 % | 0.0 % | 0.0 % | 0.0 % | 1.5 % | 0.0 % | 16 | 20 |
| run6 external (Gumbel-128 head vs sealbot) | 23.3 % | 76.7 % | 0.0 % | 0.0 % | 0.0 % | 47.4 % | 0.0 % | 4 | 20 |
| run6 promotion (Gumbel-160 vs Gumbel anchor) | 1.9 % | 98.1 % | 0.0 % | 0.1 % | 0.0 % | 82.9 % | 0.0 % | 4 | 8 |
| frontier ck18k PUCT-512 | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 20 | 24 |
| frontier ck18k Gumbel-512 | 26.4 % | 73.6 % | 0.0 % | 0.0 % | 0.0 % | 60.2 % | 0.0 % | 4 | 8 |

By ROLE in the eval channels (candidate vs opponent):

| source | n | cand wins | cand missed-block per CHECK | opp missed-block per CHECK | cand loses by LOST1 / missed block / missed win | opp loses by LOST1 / missed block / missed win |
|---|---|---|---|---|---|---|
| run7 external (PUCT-512 vs sealbot_d5) | 288 | 50.7 % | 0.3 % | 0.0 % | 140 / 2 / 0 | 146 / 0 / 0 |
| run7 promotion (PUCT-512 vs anchor) | 207 | 51.7 % | 0.0 % | 2.5 % | 100 / 0 / 0 | 95 / 12 / 0 |
| run6 external (Gumbel-128 head vs sealbot_d5) | 1 105 | 13.8 % | **72.6 %** | 0.0 % | 106 / **847** / 0 | 152 / 0 / 0 |
| run6 promotion (Gumbel-160 vs Gumbel-160 anchor) | 1 776 | 53.2 % | **71.3 %** | **78.9 %** | 22 / 810 / 0 | 12 / 932 / 0 |
| frontier ck18k PUCT-512 | 288 | 77.4 % | 0.0 % | 0.0 % | 65 / 0 / 0 | 223 / 0 / 0 |
| frontier ck18k Gumbel-512 | 288 | 6.6 % | **75.7 %** | 0.0 % | 57 / 212 / 0 | 19 / 0 / 0 |
| every frontier Gumbel cell (13 cells) | 288 each | 0.7–43.8 % | 67.5–94.6 % | 0.0–1.6 % | — | — |
| every frontier PUCT cell (14 cells) | 288 each | 3.8–77.4 % | 0.0–3.7 % | 0.0–1.0 % | — | — |

run6 self-play by actor-step band is flat: ends-LOST1 79–84 %, loser missed-block per CHECK
2.8–3.9 %, missed wins 0.0 %, median length 47 → 69 plies, check-run 16 at every band (the 167
games at step −1, the untrained net, miss 78.5 % of blocks). shakedown7 by 600-game window: loser
missed-block per CHECK 9.3 → 15.7 → 33.0 → 24.6 %, missed-win per turn 1.1 → 2.1 → 4.4 → 3.4 %,
draws 6 → 11 → 40 → 66 % — the blunder rate rises with the drift.

What B says:

- **The PUCT-512 games against sealbot contain no one-turn blunder on either side.** 288 games,
  ≈ 2 800 turn starts, 617 CHECK turns for our net, 2 missed blocks (0.3 %), 0 missed wins;
  sealbot 0 and 0. 286 of 288 games end in a constructed ≥ 3-fold threat. The same holds for all
  14 PUCT frontier cells (9 000+ games): sealbot never leaves a four standing and never misses a
  win in-turn. "Something sealbot does not defend" is NOT a one-turn hole.
- **The Gumbel deploy head leaves an opponent four standing in ≈ 3 of 4 CHECK turns** — the
  run6 `external` rung, the run6 promotion gate (BOTH seats), and all 13 Gumbel frontier cells
  (67–95 %), at 128, 256 and 512 sims alike, on the BC nets and on every run6 checkpoint. Its
  losses are 72–88 % missed blocks. PUCT on the same nets: 0–4 %. This is the F-51 residue
  (CARD-GUMBEL-HEAD-RESIDUE) read on the games: the Gumbel head is tactically blind at the
  one-turn level, and no opponent needs to exploit anything to beat it — it walks into fours.
  Mechanism NOT shown here; one observation for the card: in `r000025_25000_rung_00014` ply 22
  (run6 external) the blocking cell (0,−4) is among the root's visited children (6 visits vs the
  chosen (0,−1)'s 9; the root's top children carry 3–9 visits each at 128 sims) and the root
  value already reads −0.88.
- **run6 self-play is cleaner than the strongest human band on every one-turn column**: 3.6 %
  missed blocks per CHECK turn vs 7.0 % for humans ≥ 1300 (14.1 % corpus-wide), 0.0 % missed wins
  vs 1.0 %, 81 % constructed endings vs 68 %. The 19 % of self-play games that do end by a missed
  block are stable over training; the record does not say which arm produces them — the 64-sim
  quick searches (`full_search_prob 0.25`), the first-10-ply τ-0.5 sampling, or the 320-sim
  search itself — because a self-play record carries no per-move sims or root.
- **shakedown7 (PUCT self-play, τ 0.5 for the whole game) is a different regime**: a quarter of
  CHECK turns go unblocked, 3 % of wins-in-turn are missed, and both rates rise as the games drift
  toward the cap. Read against CARD-PUCT-ATTRACTOR: hypothesis (2) "defends but cannot attack" is
  contradicted as stated — the cap games show unanswered fours (25 % of CHECK turns); hypothesis
  (3), τ-0.5 sampling picking non-best moves, is consistent with these blunder rates but not
  separated from the 64-sim arm by this record.

## C. Judge agreement

Turn-start positions (mover has 2 stones), 24 per ply band (1–10, 11–20, 21–40, 41–80, 81+) per
source where the band has that many games, never the final winning turn; sealbot d5 built fresh
per position (see `## Sources` on its transposition table), strix at 128 sims. "first in played"
= the judge's first stone is one of the two stones actually played; "pair" = both. run7 external
is split by who moved. Sealbot re-judging its own recorded moves reproduces the pair 90.6 % of
the time — the instrument's own check.

| source (mover) | n | sealbot first in played | sealbot pair == played | strix first in played | strix first == played first | sealbot first == strix first | CHECK turns: played left a four | strix's pair left a four |
|---|---|---|---|---|---|---|---|---|
| human | 120 | 40.0 % | 18.3 % | 16.7 % | 9.2 % | 13.3 % | 5.9 % (2/34) | 2.9 % |
| run6 self-play (step ≥ 10 000) | 120 | 29.2 % | 12.5 % | 31.7 % | 15.8 % | 18.3 % | 5.0 % (1/20) | 10.0 % |
| run7 self-play | 110 | 57.3 % | 20.9 % | 30.0 % | 14.5 % | 19.1 % | 6.8 % (3/44) | 6.8 % |
| run7 external / our PUCT-512 moves | 53 | 45.3 % | 18.9 % | 34.0 % | 18.9 % | 22.6 % | 0.0 % (0/13) | 7.7 % |
| run7 external / sealbot's moves | 64 | 93.8 % | 90.6 % | 34.4 % | 12.5 % | 14.1 % | 0.0 % (0/16) | 18.8 % |

QUIET positions only (no four on the board for either side — the positions where a judge's
disagreement is a matter of taste, not of tactics):

| source (mover) | n | sealbot first in played | strix first in played | sealbot first == strix first |
|---|---|---|---|---|
| human | 81 | 25.9 % | 14.8 % | 11.1 % |
| run6 self-play | 94 | 21.3 % | 27.7 % | 14.9 % |
| run7 self-play | 63 | 39.7 % | 22.2 % | 17.5 % |
| run7 external / ours | 32 | 34.4 % | 31.2 % | 28.1 % |
| run7 external / sealbot | 39 | 89.7 % | 30.8 % | 15.4 % |

Agreement by ply band (judge's first stone in the played pair, sealbot / strix, n = 24 per cell
unless shown): human 12/17 · 50/17 · 42/12 · 46/21 · 50/17; run6 self-play 4/8 · 25/25 · 42/42 ·
42/50 · 33/33; our PUCT-512 moves 23/15 (13) · 44/67 (9) · 62/23 (13) · 33/50 (6) · 58/33 (12).
Sealbot reads a mate (its own or against it) at 25 % of human positions, 20 % of run6 self-play,
27 % of run7 self-play positions.

What C says:

- **Strix, the independent AlphaZero-style reference, agrees with our moves MORE than with the
  humans' moves** — 28–31 % of quiet self-play / PUCT-512 turn starts vs 15 % of human ones
  (and 31 % of sealbot's). Sealbot agrees with humans a little more than with run6 self-play on
  quiet positions (26 % vs 21 %) and most with run7's BC-warm-started net (40 %).
- **The two judges agree with each other rarely** (11 % on human positions, 15 % run6, 28 % on
  our PUCT-512 positions): a quiet turn start in this game has many defensible first stones, and
  "the judges agree more on human than on self-play positions" is false on this sample.
- On CHECK positions every source blocks nearly always (0–7 % left a four); strix-128 itself
  leaves a four standing in 6 of 93 CHECK positions sampled (a reminder that 128 sims is not
  tactically clean either).


## D. How our net beats sealbot, and how sealbot beats it — the exploit hypothesis

The 288 run7 external games (PUCT-512, `search_stats` on our moves; 146 wins, 142 losses; 288
distinct trajectories), with the 33 frontier cells as the wider sample.

**D.1 The seat.** Our net wins 56/144 as p1 and 90/144 as p2; sealbot the mirror; the p2 seat wins
178/288 = 0.62 on this random 4-ply book (`book_v1_s20260625_p4`: four uniform-random plies).
Pairs (same opening, colours swapped): candidate sweeps 32, sealbot sweeps 30 — 62 pairs (43 %)
decided by the player; the p1 seat sweeps 24, the p2 seat 58 — 82 pairs (57 %) decided by the
opening. The 0.507 is a paired mean over a book where more than half the pairs are seat-decided. The same
holds across the frontier: seat-decided pairs are 37 % (ck18k PUCT-512), 41 % (ck25k), 55 % (ck3k)
and 51 % (BC net PUCT-150), the p2 seat sweeping more pairs than the p1 seat in every cell; in
the strix control (D.8) only 2 of 16 pairs are seat-decided.

**D.2 How the wins look (146).** Length median 31 plies (p10 27, p90 67). Sealbot is in
continuous CHECK for the last 20 plies (median; p10 16, p90 20; 137/146 ≥ 16) — five sealbot
turns in a row where it must block, then a ≥ 3-fold threat. The forcing run begins at ply 9 or 11
in 71 of 146 wins and by ply 15 in 99; the game ends at ply 27–31 in 83. Our first four comes at
ply 10 in the p2 wins (p25 10, p75 12; made by us in 84 %) and at ply 13.5 in the p1 wins. Our
root value reads ≥ +0.5 a median 18 plies before the end and ≥ +0.9 at 9 — the value head reads
the forcing run from its start. First-searched-ply root value: mean +0.07.

**D.3 How the losses look (142).** Length median 37. We are in continuous CHECK for the last 12
plies (median; p10 4, p90 20). Sealbot's first four comes at ply 16–18. Our root value reads
≤ −0.5 only a median 8 plies before the end (p10 4, p90 12) and ≤ −0.9 at 4: **sealbot's forcing
runs arrive as a surprise to our search** — root value at our CHECK turn starts averages −0.06
(n 617), at LOST1 −0.97 (n 140), at WIN1 +0.95 (n 146), at QUIET +0.33 (n 1 857).

**D.4 Sealbot's own reading of the games (instrument 4, all 288 games re-searched at d5).**

Fresh d5 engine at every turn start of all 288 games (2 904 sealbot turns; sealbot reproduces its
recorded pair at 86.5 % of them, the rest tie-breaks under a different transposition-table state).
Mate distances ≤ 2 in its score (1 = a win in-turn, 2 = a win on its following turn) are
proofs and were converted 100 % of the time by both sides; its mate claims at distance ≥ 3 are NOT
proofs (8 of 150 games in which sealbot reported a mate for itself it went on to lose; 6 of 152 in
which it reported a mate for us we lost), so only the ≤ 2 band is read as a verdict below.

| | sealbot's 146 losses | sealbot's 142 wins |
|---|---|---|
| the loser's continuous-CHECK run before the end (plies) | 20 (p10 16, p90 20) | 12 (p10 4, p90 20) |
| sealbot first reports the mate (plies before the end) | 12 (p10 12, p90 16); never: 0 | 10 (p10 6, p90 10) |
| … which is, after the run began | 8 plies (p10 4, p90 8) | — |
| sealbot's score at ITS last turn before the run (mover's view) | median +1 084 (p10 −2 490, p90 +3 339); negative in 33 %; mate-against in 0 % | — |
| our root value at OUR last turn before the run | — | median +0.13 (p10 −0.20, p90 +0.37) |
| our root value first ≤ −0.5 (plies before the end) | — | 8 (p10 4, p90 12) |
| our root value first ≥ +0.5 (plies before the end) | 18 (p10 13, p90 22) | — |

Our root value against sealbot's score at the same candidate turn starts (n 2 760) (rows: sealbot's
bucket; columns: ours > +0.5 / within / < −0.5): sealbot mate-for-us → 446 / 14 / 0; sealbot
+2 000.. → 259 / 417 / 0; sealbot within ±2 000 → 25 / 880 / 12; sealbot −2 000.. → 4 / 347 / 5;
sealbot mate-against-us → 3 / 73 / 275. The two evaluations never point in opposite directions
with confidence (0 + 0 + 3 + 4 of 2 760); ours is more decided than sealbot's on the winning
side (259 of the 676 sealbot-"+" turns already read > +0.5 to us) and LESS decided on the losing
side (73 of 351 sealbot-mate-against-us turns still read within ±0.5 to us).

What D.4 says: **sealbot walks into the run reading the position as even-to-favourable (median
+1 084) and reports the mate only 8 plies after it started, 12 plies before the end — the depth-5
horizon.** Our net does the mirror image at a shorter horizon: it reads +0.13 before sealbot's
run and sees the loss 8 plies before the end, while it sees its own wins 18 plies out.


**D.5 Is the run forced, and from when?** The pure continuous-fours solver (instrument 5, budget
40 000 nodes, ≤ 5 attacker turns) on 24 sampled wins proves the win forced from 2 attacker turns
before the end in 24/24, from 3 in 6/24 and from 5 in 2/24 (`…rung_00213`: forced from ply 11 of
29; `…rung_00465`: from ply 11 of 29) — and hits its budget in every game, so these are FLOORS:
the observed check-run (16–20 plies) is longer than what the solver can prove. The same solver on
24 sampled sealbot wins proves 2 turns in 23/24 and 4 in 1/24; on 24 run6 self-play games 2+ turns
in 22/24 (one from 6 turns, 22 plies out); on 24 human games only 1 turn in 12/24 — i.e. the
human loser had a defence at its last turn (the corpus's 38 % missed-block endings) — and 2+ in
12/24.

**D.6 The same motif everywhere.** The 20-ply check-run before a candidate win is the same in
every PUCT frontier cell — ck3k, ck18k and ck25k at 512 sims and the BC net (behaviour-cloned from
humans, no RL) at PUCT-150 (WR 0.413) — and the human corpus has a ≥ 16-ply check-run in 2 139 of
8 698 games (25 %; median 12 in the ≥ 1100 bands). Sealbot's wins over us have the same shape
with a shorter run (12). Stone geometry does not differ: 6.9 % of our stones vs sealbot land ≥ 3
from every stone, 5.7 % of sealbot's (4.0 % human); no distant-stone or off-window pattern.

**D.7 Where sealbot's move choice differs from a deeper sealbot.** Reading the vendored search:
depth is in TURNS (both sides), `_quiescence` extends only instant wins and blocks (no four-making
extension), candidates are cells within hex distance 2 of a stone, capped at 15 (20 at the root).
So a d5 sealbot sees at most three of the attacker's turns; a run of five forcing turns is past
its horizon by construction. On 20 sampled wins, at sealbot's last free turn (the turn before its check-run began), d5 vs d6
(`run_horizon.py`; d6 costs 6.7 s median, 20.7 s max per search here, which is why it is not a
rung — R326): d5 reproduces the recorded pair 20/20; **d6 picks a different pair in 15/20**; d6's
score is lower than d5's in 20/20 (median +1 286 → −1 930; negative in 3/20 at d5, 17/20 at d6).
One more turn of depth flips sealbot's own read of these positions from "slightly ahead" to
"behind" and changes its move three times in four — the run begins exactly where its horizon
ends. In `…rung_00213` d6 and d7 both switch from building their own group to contesting ours.


**D.8 The control: strix-128 vs sealbot d5 on the same book.**

16 openings from `book_v1_s20260625_p4` (seed 20260625), each played with both colours, strix at
128 sims (0.9 s/move here) vs a fresh sealbot d5 per game (`run_strix_vs_seal.py`; 32 games,
16.6 s per game median). **Strix wins 30 of 32 (0.938)**, sweeping 14 of 16 pairs (the p2 seat
sweeps the other 2). All 30 strix wins end in a constructed ≥ 3-fold threat after a 16-ply
continuous-CHECK run (p10 12, p90 20; games 38 plies median; strix's first four at ply 14,
made by strix in 22/30); sealbot's 2 wins are strix missed blocks (2 of its 27 CHECK turns);
sealbot missed 0 of 93 blocks. No prior strix-vs-sealbot reading exists in the record; this is
the first, and n = 32.

What D.8 says: an independent bot with no knowledge of sealbot beats it by exactly the motif our
net uses — early four, continuous fours, a threat it cannot hit with two stones — and far more
decisively (0.94 vs our 0.51). The motif is sealbot's weakness under any competent MCTS player;
our net's 0.507 is a WEAK exploitation of it, not a special one.


**Verdict on "our PUCT is exploiting something sealbot does not defend":** what the games show
is a fixed-depth minimax being beaten by forcing sequences longer than its depth, from positions
its own evaluation reads as roughly even; there is no one-turn hole, no geometric trick, no
recurring shape beyond "make the first four early and never stop making fours". The same
weakness is available to any opponent that reads five forcing turns ahead — the BC net at 150 sims
does it, and sealbot does it back to us at a shorter horizon. That is the ordinary way one
tactical engine beats another in this game and it is not a defect a stronger opponent would fail
to have. What IS specific to this bar, and worth carrying: (i) at depth 5 sealbot has no forcing
extension, so its horizon against continuous fours is three attacker turns; (ii) sealbot's
evaluation reads the pre-run position as near-even while the deeper d6/d7 engine picks a
different, contesting move; (iii) the random book makes 37–57 % of pairs seat-decided across cells (57 % on this stamp).

## E. Self-play quality per se

Stone-purpose fingerprint (instrument 2; % of stones, first matching class; human 3 000 games,
run6 self-play 3 000 games at step ≥ 10 000, the rest whole):

| source / side | stones | win | block4 | four | block3 | three | connect | near | far |
|---|---|---|---|---|---|---|---|---|---|
| human / all | 188 838 | 1.6 | 16.8 | 14.5 | 15.7 | 17.8 | 13.5 | 15.5 | 4.6 |
| human / winner | 95 893 | 3.1 | 12.2 | 18.8 | 14.9 | 18.7 | 12.9 | 14.7 | 4.6 |
| human / loser | 92 945 | 0.0 | 21.5 | 10.0 | 16.5 | 16.9 | 14.2 | 16.3 | 4.6 |
| run6 self-play / all | 227 711 | 1.3 | 19.6 | 18.3 | 9.7 | 14.6 | **6.9** | **24.7** | 5.0 |
| run6 self-play / winner | 113 481 | 2.6 | 14.7 | 22.1 | 9.1 | 15.9 | 6.7 | 24.0 | 4.9 |
| run6 self-play / loser | 110 646 | 0.0 | 24.3 | 14.2 | 10.1 | 13.2 | 7.1 | 25.9 | 5.3 |
| run7 self-play / all | 9 088 | 2.1 | 26.0 | 21.9 | 8.2 | 16.2 | 10.1 | 9.1 | 6.3 |
| shakedown7 self-play / all | 149 866 | 0.5 | 10.0 | 9.5 | 19.4 | 18.7 | 26.4 | 14.3 | 1.2 |
| run7 external / ours (PUCT-512) | 6 240 | 2.3 | 19.8 | 20.9 | 15.6 | 14.4 | 7.4 | 11.3 | 8.3 |
| run7 external / sealbot | 6 235 | 2.3 | 26.2 | 17.9 | 6.2 | 19.4 | 13.7 | 6.3 | 8.0 |
| frontier ck18k PUCT-512 / ours | 7 994 | 2.8 | 12.8 | **28.3** | 16.8 | 13.0 | 5.8 | 14.3 | 6.2 |
| frontier ck18k PUCT-512 / sealbot | 7 835 | 0.8 | **34.3** | 12.3 | 4.4 | 20.8 | 14.6 | 6.5 | 6.3 |
| frontier ck18k Gumbel-512 / ours | 3 640 | 0.5 | **8.5** | 14.6 | 14.4 | 12.2 | 9.1 | 26.3 | 14.4 |

Opening habit — hex distance of p2's FIRST stone from p1's first stone: human d = 1 in 55 %, d ≤ 2
in 79 %, d ≥ 6 in 12 % (d = 8, the legal maximum, in 8 %); run6 self-play (step ≥ 10 000) d = 1
in 27 %, d ≥ 6 in 28 %; run7 self-play (the BC-warm-started net at step ≤ 101) d = 1 in 12 %,
**d ≥ 6 in 69 %, d = 8 in 43 %**; shakedown7 d ≥ 6 in 4 %. The first 10 plies of a Gumbel
self-play game are τ-0.5 samples of the visit distribution (`gumbel_explore_moves: 10`,
`temp_min 0.5`), so this is the search's spread, not the head's argmax.

What E says, and the verdict on self-play quality:

- **By the tactical yardstick, run6 self-play is honest play, not degenerate play.** Threats are
  made (18 % of stones make a four, vs 14.5 % human) and answered (96.4 % of CHECK turns block,
  vs 86 % human, 93 % for humans ≥ 1300); no win-in-turn is ever missed; 81 % of games are won by
  a constructed ≥ 3-fold threat at the end of a 16-ply forcing run, 19 % by a missed block. The
  human corpus: 59 % / 38 % / 3 %. Games are decisive (0.5 % draws), balanced (p1 0.505), and
  longer than human games (61 vs 49 plies). None of this moved over the block (step bands flat).
- **What is different from human play is style, and it is visible in the viewer.** Self-play
  stones are less often connected to an own stone (6.9 % vs 13.5 %) and more often loose "near"
  stones within 2 of the action (24.7 % vs 15.5 %); pre-emptive blocks of an opponent three are
  rarer (9.7 % vs 15.7 %) while forced blocks of fours are commoner (19.6 % vs 16.8 %) — the net
  answers fours, not threes, and builds toward fours rather than shapes. p2 opens far from p1's
  stone in 28 % of games (69 % for run7's net). These are the features a human reader sees as "no
  gameplay"; the numbers say the play is tactically sound and stylistically un-human.
- **The one self-play set that IS degenerate is shakedown7**: PUCT at τ 0.5 for the whole game
  leaves a quarter of fours unanswered, misses 3 % of wins-in-turn, and the rates rise as the
  games drift to the cap. Its fingerprint is the inverse of run6's — 26 % `connect`, 19 %
  `block3`, only 9.5 % `four`: shape-building without threats, then blunders.
- **The Gumbel deploy head's games are not a reading of the net.** run6's `external` and
  `promotion` channels and every Gumbel frontier cell are games in which one or both sides leave
  fours standing three times out of four; the promotion gate at 160 sims compares two such heads.


## F. Positions to look at (viewer links)

Viewer: `http://127.0.0.1:8765/viewer/index.html?g=<run>/<game_id>&ply=N` (runs run7, shakedown7,
run6). Human games are not in the viewer; their move lists are in the corpus file by `game_hash`.

1. **`run7/r000001_101_terminal_rung_00213`, ply 9** — our net (X, p1) has a four after its 5th
   stone; sealbot's four stones are the two random book stones and a pair it built next to them
   at ply 5–6, all ≥ 5 from the action. Sealbot is in CHECK from here to the end (20 plies);
   the pure-fours solver proves the win forced from ply 11; sealbot's own d5 score is −2 916 at
   ply 5 and −1 972 at 9, and it first reports the mate at ply 17. At ply 5 sealbot d6 and d7 pick
   (2,3),(0,4) — contesting — where d5 picked (7,−1),(6,−1). Our root value: +0.61 at ply 7.
   `…&ply=5`, `…&ply=9`, `…&ply=11`.
2. **`run7/r000001_101_terminal_rung_00214`, ply 19** — the paired game (colours swapped).
   Sealbot (X) begins its forcing run at 19 and wins at 36; our root value reads +0.25 at ply
   17–18, −0.12 at 25, −0.92 at 29, LOST1 at 33. Sealbot's own score: +3 812 at 19, +9 466 at 23,
   the mate at 27. The same mechanism pointed the other way. `…&ply=17`, `…&ply=29`.
3. **`run7/r000001_101_terminal_rung_00215`, ply 29** — a long one (53 plies): sealbot's own d5
   score reads +5 180 at its turn 29 and +3 557 at 33; our four at 39 starts a 12-ply run it
   cannot leave; it reports the mate at 41. Sealbot's evaluation is optimistic right up to the
   run. `…&ply=29`, `…&ply=39`.
4. **`run6/r000025_25000_rung_00014`, ply 21** — the Gumbel-128 head (O) in CHECK against three
   sealbot fours whose empties are {(6,−10),(7,−11)}, {(1,−5),(0,−4)}, {(1,−5),(6,−10)}; the pairs
   {(1,−5),(6,−10)} or {(0,−4),(6,−10)} hold. It plays (6,−10) and then (0,−1) with the root
   value at −0.88 and the blocking cell (0,−4) visited 6 times against the chosen cell's 9; it
   loses at 24. One of the 847 run6-external losses of this kind. `…&ply=21`, `…&ply=22`.
5. **`run6/eadc34bd08c04f0f97e1df4325d6091d`, ply 13** (self-play, actor step 28 660) — O is in
   CHECK from ply 13 to LOST1 at 29: a 20-ply constructed self-play win, the 81 % case.
   `…&ply=13`, `…&ply=29`.
6. **`run6/0496b33d7e8545baa37cfb52ac733bd1`, ply 33** (self-play, step 30 266) — three X fours,
   hitting pairs ((−1,−1),(−1,5)), ((−1,0),(−1,5)), ((−1,0),(−1,6)) all available; O plays
   (11,−7),(10,−6), eleven cells away, and loses at 36. The 19 % case (a 64-sim quick search or
   a τ sample; the record does not say which). `…&ply=33`.
7. **`shakedown7/769c27516b95437b826c8a0ed77c6c3a`, ply 123** — X has a four with empties
   {(−15,4),(−15,5)}, a win in-turn, and plays (−16,7),(−14,5); O wins at 126. shakedown7's
   missed-own-win case (3.5 % of its decisive games; 0 in run6). `…&ply=123`.
8. **`run7/a564023aa77f4c24a11e2db5acdcb1d3`, ply 2** — p2's first stones at (6,−4),(5,−3), six
   away from p1's (0,0): the far-opening habit of 69 % of run7 self-play games. `…&ply=2`.
9. **Human `2961e3ba64746030`** (Elo 1114 vs 983, 33 plies): O in CHECK from ply 9 to LOST1 at
   29 — the identical 24-ply constructed motif in a human game (25 % of the corpus has a
   ≥ 16-ply run). **Human `251237b3f396cb20`** (1350 vs 1160), ply 21: two X fours, four hitting
   pairs, O plays (0,−5),(−1,2) and loses at 24 — the corpus's 38 % case.


## G. What this record does NOT say

- Nothing here is a strength number; the instruments are one-turn tactics, a budget-limited
  forcing solver, two judges of unknown absolute strength, and an evaluation trajectory from the
  bar itself. "Cleaner than the ≥ 1300 human band" is a statement about blunder rates, not Elo.
- The check-run measures how long the loser was forced, not that the attacker's win was
  inevitable from the run's first ply; the solver proves only the last 2–5 attacker turns within
  budget.
- Judge disagreement is not judge disapproval: this game has many reasonable moves at a quiet
  turn start, and sealbot and strix agree with each other on only 11–28 % of quiet positions.
- The Gumbel-head finding names the symptom (unanswered fours) and one observation; it does not
  name the mechanism inside `gumbel_root_best_move` / sequential halving.
- No human-vs-sealbot record exists; the control is strix, not a human.
- The human corpus is weak (median 1 055 on its site's scale); "human-like" here means like these
  humans.
- Absent is not zero: `search_stats` exist only on eval channels; self-play carries no root
  values, so self-play calibration is not measured.

## Sources

Game sets (read with `mantis.monitor.game_record.iter_run_games`, all shards, no dedupe needed:
288/288 and 25 110/25 110 distinct trajectories where checked):

- the operator's `hexo-bootstrap-corpus/hexo_human_corpus.jsonl` checkout (+ its `SCHEMA.md`, `README.md`)
- the operator's `mantis-mirror/run6/logs/games` (run_id `run6`: 35 287 selfplay, 1 105
  external, 1 776 promotion, 820 random_floor)
- `mantis-mirror/shakedown7/logs/games` (run_id `shakedown7`: 2 421 selfplay)
- `mantis-mirror/run7-stamp-d13bc7c3/games` (run_id `run7`: 288 external,
  208 promotion, 195 selfplay, 24 random_floor)
- `mantis-mirror/frontier/phase{1,2,3}/<cell>/games` (run_id `frontier1`,
  33 cells; `cells_phase*.json` and each cell's `result.json` for the cell definitions and WRs)
- `src/mantis/arena/books/book_v1_s20260625_p4.json` (the eval book; `tools/mint_opening_book.py`
  for how it was minted: four uniform-random legal plies)

Engine surfaces: `mantis._engine.Board` (`winning_moves`, `threat_moves`, `forced_win_move`,
`legal_moves`; `crates/mantis-core/src/board/moves.rs`, `crates/mantis-bridge/src/board.rs`);
`mantis._engine.TacticalSolver` was evaluated and NOT used (its threat-space candidates are
4→5 conversions only, so it returns UNKNOWN at every four-making turn; with quiet-move widening
it exhausts 20 000 nodes at depth 4 in 2.5 s per position). `mantis.bots.resolve.resolve_bot`
for sealbot d5/d6/d7 and strix-128; `vendor/external/sealbot/current/engine/{search,movegen,
constants,bot}.h` for the depth-in-turns, the block-only quiescence, `NEIGHBOR_DIST 2`,
`CANDIDATE_CAP 15 / ROOT_CANDIDATE_CAP 20`, and the transposition table's key
(`bot.h:214`, position ⊕ cur_player ⊕ moves_left — NOT the root player whose perspective the
stored scores carry; it persists across `get_move` calls, so every instrument here builds a FRESH
`SealBotAdapter` per search. The eval rung reuses one adapter per thread across paired games
with swapped seats; a cross-game hit needs an identical position with the same side to move,
which the random book makes rare, and this record did not measure whether it ever happens).

Scripts (the agent session's scratchpad directory, subfolder `quality/`, outputs under its `out/`;
untracked, dev-only, none of it is repo code): `qlib.py` (loaders, replay, turn clock), `tactics.py` (windows, fours,
hitting sets, the per-game autopsy), `run_autopsy.py` / `agg_autopsy.py` (§A, §B, per-band
tables), `run_style.py` (§E fingerprint), `run_judges.py` / `agg_judges.py` (§C),
`run_sealtraj.py` / `agg_sealtraj.py` (§D.4), `vcf.py` / `run_vcf.py` (§D.5), `run_horizon.py`
(§D.7), `run_strix_vs_seal.py` (§D.8), `render.py` (ASCII boards used to eyeball §F).
Outputs: `out/autopsy_{human,run6,shakedown7,run7,frontier}.jsonl`, `out/style_*.jsonl`,
`out/judges_*.jsonl`, `out/sealtraj_run7_external.jsonl`, `out/vcf_*.jsonl`,
`out/horizon_run7_candwins.jsonl`, `out/strix_vs_seal.jsonl`. Two TT-contaminated first
passes are kept as `*_TTCONTAMINATED.jsonl` and were not used.

Governance read: `docs/governance/CARDS.md` (CARD-PUCT-ATTRACTOR, CARD-GUMBEL-HEAD-RESIDUE),
`docs/governance/falsified.md` (F-51, F-52), `docs/design/measurements/SHAKEDOWN7_2026-09-14.md`
and `RUN7_STAMP2_2026-09-14.md` for the run context.


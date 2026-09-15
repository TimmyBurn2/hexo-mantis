# MEASUREMENT — INVESTIGATION-1 re-aimed at run6's trough (R350(g), 2026-09-13)

What the targets were between 8k and 14k, the KL from the prior, and the α distribution — read
off the run record itself. The game shards carry NO per-position search stats for self-play
games (the self-play recorder writes the move list and the result only; R344's 1-in-N sample of
search stats is not producing), so the reconstruction uses the RINGS: every bundle's
`.ring.bin` holds the last 100 000 training rows (≈ the last hour of games) with their stored
targets — the sparse Gumbel row's explicit cells and its tail mass α — and the checkpoint of the
same step holds the net those rows were about to train. For each step the script rebuilds the
target exactly as the trainer does (`ragged_policy_ce_and_target_entropy`, the tail from the
net's own detached prior), over 16 batches × 256 sampled rows (4 096 rows, `seed 20260913 +
step`, `recent_frac 0`, no augment) on the dev box's CPU, and reads:

- `CE` — the policy loss the trainer would report on that ring with that net;
- `H(target)` — the entropy of the reconstructed target, same reduction;
- `KL` = CE − H — KL(target ‖ prior), R350(b)(iv)'s first line, reconstructed post hoc;
- the α distribution of the sampled rows and the α = 1.0 count per 1 000 (the row the loss now
  excludes, R350(e)).

Instrument: `scratchpad/trough_kl.py` (this session's scratchpad; the checkpoints are read
through the one loader after `strip_and_restamp` — the loader's tolerance for a stamp that
predates a schema leaf landed later the same day). Every row is `full_search` (α ≥ 0 on all).
[Corrected in place 2026-09-15 (R311(c)): the rows are NOT all full-search — `is_full_search` is
set on ≈ 25 % of every ring's rows (`full_search_prob 0.25`), the quick 64-sim arm on the rest;
see `FORCED_MOVE_CENSUS_2026-09-15.md`. The KL/CE readings above pooled both arms.]

| step | ring rows | CE (nats) | H(target) | KL(target ‖ prior) | α mean | α = 1.0 per 1 000 |
|---|---|---|---|---|---|---|
| 1 000 | 48 884 | 2.221 | 0.143 | 2.079 | 0.0025 | 1.2 |
| 3 000 | 100 000 | 2.378 | 0.130 | 2.248 | 0.0069 | 4.4 |
| 5 000 | 100 000 | 2.654 | 0.115 | 2.540 | 0.0085 | 6.6 |
| 8 000 | 100 000 | 2.574 | 0.120 | 2.454 | 0.0070 | 5.9 |
| 10 000 | 100 000 | 2.715 | 0.096 | 2.619 | 0.0073 | 5.1 |
| 12 000 | 100 000 | 2.835 | 0.111 | 2.725 | 0.0086 | 6.6 |
| 14 000 | 100 000 | 2.882 | 0.128 | 2.754 | 0.0049 | 2.9 |
| 18 000 | 100 000 | 2.580 | 0.113 | 2.467 | 0.0078 | 5.1 |
| 25 000 | 100 000 | 2.637 | 0.113 | 2.524 | 0.0060 | 3.7 |

## What it says

1. **The targets are near one-hot from the first ring on.** H(target) sits at 0.10–0.14 nats
   for the whole block (a two-move 50/50 target would read 0.69; 0.11 nats is ≈ 97 % of the
   mass on one move). Under `c_scale 1.0` × Mctx's min-max rescale the completed-Q improved
   policy IS an argmax on Q, as R350(a) states — and it is that at step 1 000 with a value head
   1 000 steps old as much as at 25 000.
2. **The KL from the prior is 2–2.75 nats throughout and RISES through the trough**: 2.08 at 1k,
   2.62 at 10k, 2.75 at 14k, back to ≈ 2.5 after. The prior's probability on the target's move
   averages exp(−CE) ≈ 8–11 % — the search's answer disagreed with the policy on essentially
   every row, all block. The trough's signature in this line is the rise 2.08 → 2.75 with the
   targets not getting any softer (H flat), i.e. the argmax moved AWAY from the prior, not the
   prior toward a broader target.
3. **α is small and its tail is the story.** Mean α 0.25–0.86 %, p90 ≈ 1e-5: on almost every
   row the 16 explicit cells hold the whole target; the α = 1.0 rows (3–7 per 1 000 in these
   4 096-row samples, the block record's 4.98 flat over the whole stream) are the exception the
   loss now excludes, not the rule.
4. **What this does NOT separate**: whether the argmax was RIGHT. A KL of 2.5 nats from a
   correct search is learning; from an argmax of a random value head it is the overwrite R350(a)
   names. The frontier's PUCT-vs-Gumbel cells on the same nets are the discriminator for that
   (`STRENGTH_FRONTIER_1_2026-09-13.md`), and R350(b)(iv)'s live line makes the next run read
   this table from step 0 rather than after the fact.

Owed still: the three-row α = 1.0 reconstruction R349(c) ordered (CARDS.md, owed section); the
self-play search-stats sample R344 ordered, without which per-position KL cannot be read from
the game record (a card, below the line).

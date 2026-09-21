# PARAMETER DISTANCE — run8's checkpoints against the running mean, beside the strix and gate series (R364 §0(7), 2026-09-21)

Ordered by R364 §0(7): the numbers that decide LR's place for run10. Nothing here is a design or a
lever; the last section names what the numbers rule out and what they leave open, and stops.

## 1. What was measured, and how

Eighteen checkpoints off the operator's mirror (`mantis-mirror/{run7,run8}/checkpoints/`): the parent
`run7_00042000_46fdb931.ckpt` as run8's step 0 (its weights are run8's step-0 net, `bc_warmstart_loaded`
verified over 50 tensors), then every periodic save `run8_00003000_861fa9f0` … `run8_00051000_a0a5b077`
(3 000-step cadence, 17 saves). Each is read through `mantis.train.checkpoints.load_checkpoint` (the
LAW-12 loader, the stamp validated), its 50 floating tensors concatenated in sorted-name order into one
float64 vector θ of 302 470 parameters (three groups by the state-dict prefix: `representation` 42
tensors, `policy_head` 4, `value_head` 4). Per checkpoint k, with θ̄ₖ the running mean of θ₀ … θₖ:
‖θₖ‖, ‖θₖ − θ₀‖, ‖Δₖ‖ = ‖θₖ − θₖ₋₁‖, ‖θₖ − θ̄ₖ‖, cos(Δₖ, Δₖ₋₁) (the coherence of successive 3k-step
displacements), cos(Δₖ, θₖ − θ₀) (the displacement against the cumulative drift), and ‖Δₖ‖ per group.
Beside each row: the strix follower's equal-work 256/256 point where one exists (CONTENDED, 288 paired
games) and the gate round on that checkpoint (candidate pooled WR against the anchor, pairs, promoted;
derived from the mirror's `logs/games/` promotion-channel records, `phase == gate_sequential`, the
candidate's colour read per game — the box's pre-R362 tree emits no gate fields). The harness is a
one-file scratch script (≈ 60 lines: load, concatenate, the norms above, JSON out) kept out of the tree;
everything below is derived from its JSON. The 45k net's `state_dict_param_hash` read here is
`3aef78834b042d333400f12f6afb15e41cb716712f84386297bfbb1dd4126f42` — the hash `configs/run9.yaml`
declares.

## 2. The numbers

LR over the whole span: `cosine` from 1.0e-3 to `eta_min` 5.0e-4 over `total_steps` 1 000 000 — at
51 000 the schedule reads 9.97e-4, so the step size below is that of a FLAT LR.

| step | ‖θ‖ | ‖θ−θ₀‖ | ‖Δ‖ | ‖θ−θ̄‖ | cos(Δ,Δprev) | cos(Δ,θ−θ₀) | ‖Δ‖ repr / policy / value | strix 256/256 | gate: WR · pairs · promoted |
|---|---|---|---|---|---|---|---|---|---|
| 0 (parent) | 156.6 | 0 | — | 0 | — | — | — | 0.111 [0.073, 0.149] | — |
| 3 000 | 160.0 | 20.7 | 20.7 | 10.4 | — | — | 18.4 / 8.0 / 5.2 | 0.056 [0.031, 0.083] | 0.616 · 88 · yes |
| 6 000 | 163.4 | 29.9 | 18.7 | 15.1 | 0.146 | 0.728 | 16.0 / 8.2 / 5.4 | 0.0625 [0.038, 0.090] | 0.646 · 48 · yes |
| 9 000 | 167.5 | 38.1 | 19.1 | 19.7 | 0.149 | 0.634 | 16.4 / 7.7 / 6.0 | — | 0.536 · 104 (cap) · no |
| 12 000 | 171.9 | 46.3 | 20.3 | 24.7 | 0.158 | 0.588 | 17.1 / 8.4 / 7.0 | — | 0.505 · 48 · no |
| 15 000 | 176.1 | 53.7 | 20.6 | 28.9 | 0.126 | 0.524 | 17.8 / 8.2 / 6.2 | 0.104 [0.069, 0.139] | 0.641 · 80 · yes (the anchor since) |
| 18 000 | 182.1 | 62.5 | 21.6 | 34.5 | 0.167 | 0.551 | 17.6 / 7.9 / 9.8 | — | 0.422 · 32 · no |
| 21 000 | 187.0 | 70.5 | 21.5 | 39.2 | 0.188 | 0.505 | 18.1 / 8.4 / 8.0 | — | 0.469 · 40 · no |
| 24 000 | 191.3 | 78.1 | 21.0 | 43.3 | 0.159 | 0.477 | 17.7 / 8.4 / 7.8 | — | 0.458 · 24 · no |
| 27 000 | 196.4 | 86.1 | 21.8 | 48.0 | 0.160 | 0.479 | 18.3 / 7.9 / 8.8 | — | 0.537 · 80 · no |
| 30 000 | 201.5 | 94.4 | 22.2 | 52.9 | 0.169 | 0.473 | 18.3 / 8.7 / 9.1 | 0.135 [0.097, 0.177] | 0.391 · 32 · no |
| 33 000 | 205.7 | 101.0 | 22.0 | 56.2 | 0.125 | 0.399 | 19.3 / 8.2 / 6.7 | — | 0.312 · 24 · no |
| 36 000 | 209.7 | 107.1 | 22.0 | 59.1 | 0.090 | 0.373 | 19.3 / 8.5 / 6.3 | — | 0.450 · 40 · no |
| 39 000 | 215.1 | 115.1 | 22.1 | 63.7 | 0.114 | 0.446 | 18.0 / 8.7 / 9.3 | — | 0.508 · 64 · no |
| 42 000 | 221.7 | 124.6 | 24.3 | 70.0 | 0.198 | 0.473 | 19.9 / 8.8 / 10.8 | — | 0.375 · 16 · no |
| 45 000 | 227.1 | 132.2 | 22.9 | 74.1 | 0.143 | 0.411 | 19.5 / 8.8 / 8.0 | **0.142 [0.104, 0.181]** | 0.512 · 40 · no |
| 48 000 | 231.0 | 137.7 | 22.6 | 76.2 | 0.079 | 0.320 | 19.9 / 9.1 / 5.7 | — | 0.524 · 104 (cap) · no |
| 51 000 | 233.8 | 142.2 | 23.2 | 77.5 | 0.032 | 0.272 | 21.0 / 8.9 / 4.6 | — | 0.469 · 32 · no |

Derived from the table:

- **The step size is flat.** ‖Δ‖ per 3 000 steps sits in 18.7–24.3 over the whole run (median 21.6),
  as a flat LR predicts; it does not shrink as strix flattens.
- **Successive displacements are nearly orthogonal.** cos(Δₖ, Δₖ₋₁) reads 0.13–0.20 from 6k to 30k
  (mean 0.158) and 0.03–0.20 from 33k to 51k (mean 0.111), with the last three rows 0.14 → 0.08 → 0.03.
  By direction the 3k-step walk is ≥ 80 % noise throughout and the coherent share is falling.
- **The cumulative path is between diffusive and ballistic.** ‖θₖ − θ₀‖ against the segment count k
  fits k^0.71 (log–log slope over k = 1 … 17); at k = 17 a pure random walk of these step sizes
  would read ≈ 86, a straight line ≈ 353, the run reads 142. cos(Δ, θ − θ₀) falls 0.73 → 0.27:
  each new segment aligns less with where the run has gone.
- **The centre drifts; it is not a noise ball.** ‖θ − θ̄‖ grows monotonically 10 → 78 and its ratio
  to ‖θ − θ₀‖ holds at 0.50–0.56 (a walk around a fixed centre would saturate ‖θ − θ̄‖; a straight
  line holds the ratio at 0.5). The growth per segment slows late: 6.3 (39k → 42k), 4.1, 2.1, 1.3.
- **The weight norm grows +49 %,** 156.6 → 233.8, +4.5 per 3 000 steps with no sign of settling
  (`weight_decay` 1e-4 is not holding it): `representation` 125.4 → 149.6 (+19 %), `policy_head`
  34.7 → 56.2 (+62 %), **`value_head` 87.2 → 170.7 (+96 %)** — the value head's per-segment step
  swings 4.6–10.8 while the policy head's holds 7.7–9.1 and the representation's rises 16.0 → 21.0.
- **Beside the series:** strix moved +4.8 pp from 3k to 15k (0.056 → 0.104) while cos(Δ, Δprev)
  read ≈ 0.15, then +3.1 pp from 15k to 30k and +0.7 pp from 30k to 45k while the coherence fell
  toward 0.03 and the step size did not; the gate, frozen on the 15k anchor, read the trainer net at
  0.31–0.54 over r6–r17 (two cap rounds at 0.536 and 0.524, rejected by the LLR's sign under
  H0 0.52 / H1 0.62) — a band, not a trend, over 36 000 steps of ‖Δ‖ ≈ 22 each.

## 3. What the numbers rule out, and what they leave open

Ruled out: that run8's late flatness (30k → 45k +0.7 pp, 45k → 51k a gate band) is a SMALL step size
— the step is as large at 51k as at 3k. Ruled out: a settled centre — the run is still drifting, and its
weight norm (the value head's above all) is still growing. Left open, for the operator's LR decision:
whether the falling coherence (0.15 → 0.03) is the LR spending its step on noise (an LR drop or the
schema's dormant `train.ema` lever would test it — R364(d): LR is run10's, after the data regime), or the
data regime's own signature (1 step per game on a 100 000-row window turns the buffer over every ≈ 1 250
steps, so successive segments train on disjoint data — which run9's 500 000-row window at ≈ 8 reuse
changes first, R364(b)). The value head's norm doubling is a finding filed here, not read: the value loss
fell 0.64 → 0.50 over the same span (STATE item 15), and no line on the record connects the two.
Reproduce: the eighteen checkpoints named above, the loader, sorted-name concatenation, float64 norms.

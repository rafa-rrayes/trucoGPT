# Plan: Customizable bot + sampled tournament → high-quality corpus

## Phase 0 — Engine: widen the señal channel (prerequisite)
- [x] state.py: `HandState.signals` → `list[list[int]]` (multiple per seat/hand)
- [x] engine.py: `legal_actions` offers each not-yet-sent code; `_apply_play` appends
- [x] observation.py: `my_signal/partner_signal` → `my_signals/partner_signals` tuples; `encode()` multi-hot
- [x] cli/render.py: render the list of partner/own signals
- [x] tests green (tokenizer redaction computes counts dynamically — should pass)

## Phase 1 — Fully-customizable bot (`truco/agents/configurable.py`)
- [x] `BotConfig` dataclass: style (bluff, signal_prob, aggression) + p_break_R1..R11
- [x] Signalling R1–R4 (lie / late / partial / contradict), up-front multi-signal
- [x] Card play R6,R7,R9,R10 (overtrump / burn / overkill / bad-facedown)
- [x] Betting R5,R8 (ignore-partner conditioning / fold-won)
- [x] Eleven R11 (bad eleven math)
- [x] keep `HeuristicAgent` byte-identical (still the eval baseline)

## Phase 2 — Config sampling (`truco/tournament/configs.py`)
- [x] discrete knob levels + Latin-hypercube sampler → N configs
- [x] granular knobs (0.05/0.1/0.25 steps); rule-breaks capped at [0, 0.5]
- [x] verified 10k LHS configs come out all-distinct

## Phase 3 — Tournament (`truco/tournament/run.py`, `match.py`)
- [x] `play_config_match(cfgA, cfgB, seed, dealer)` pure engine, team A={0,2} B={1,3}
- [x] random pairings, many matches, win-rate ranking (seat/dealer balanced)

## Phase 4 — Corpus regen (`truco/tournament/corpus.py`)
- [x] top-K configs play each other; tokenize all 4 streams → new corpus.npz

## Phase 5 — Verify
- [x] tests (16 passed); smoke tournament + smoke corpus; sanity-check rankings

## Run log — 10k tournament (2026-06-17)
- 10,000 configs × 10 games = 100,000 matches in ~100s on 10 cores (~980 match/s).
- Textbook (all rules obeyed) ranked **522/10000, wr=0.700** — NOT optimal. Top configs
  blend modest bluff + positive aggression + light señal deception. Confirms the textbook
  zero-point is too passive/predictable; some "rule-breaking" is good play.
- New corpus → `checkpoints/corpus_top.npz` (3000 matches from top-15, 12k streams, valid).
  Live `checkpoints/corpus.npz` left untouched.
- CAVEAT: ~14–23 games/config makes the *precise* top-15 noise-dominated (leader 19/20 over
  20 games). The broad trend is reliable; the exact ordering at the top is not. For a
  trustworthy top-K, re-run with 50–100 games/config.

## Run log — 50k tournament, best-of-3, top-30% winners-only (2026-06-17)
- Engine: `MatchState.sets_to_win` + `new_match(sets_to_win=)` parametrize match length
  (default 1 → all existing callers byte-identical). `--sets 2` = best-of-3.
  `tokenize_config_match` → `(tok, winner)`; `generate_topk_corpus(winners_only=)`.
- 50,000 configs × 10 games = 500,000 best-of-3 matches (~19 min, ~438 match/s on 10 cores).
- Textbook ranked **4759/50000, wr=0.682** — top ~9.5%, still not optimal (same trend).
- Corpus: top-30% (15,000 configs) play each other, winners-only → `checkpoints/corpus_top.npz`
  = 6,000 streams, 3.45M tokens, avg 576. Top-fraction (not top-K) defuses the win-rate noise.
- OPEN ISSUE: best-of-3 streams are long (median 525, max 2070); ~52% exceed block_size=512 and
  get left-truncated in pretraining. Decide: bump block_size→1024 + retrain (corpus has full
  streams), regen single-set (`--sets 1`), or accept truncation. Recommended: block_size→1024.

================================================================================
# GOAL (2026-06-17): make the agent unbeatable

Baseline: `checkpoints/rl_vs_pool.pt` = 0.545 vs strong pool @ iter 75, 0.365 vs default; plateaus
& walks off; ~50/50 vs the rank-1 pool bot. Model: 3.25M params (d_model 225, 5L/5H, block 512).

## Diagnosis (code-grounded)
1. Structural ceiling: only ever trains vs FIXED heuristics → learns to exploit them, not to play
   low-exploitability. Unbeatable ⇒ self-play / population training.
2. Card-luck variance dominates the gradient: terminal ±1 + per-hand value/12 shaping (env.py:86-88),
   gamma 0.99 across the WHOLE match, pooled team return (rollout.py:273-284), ~4-15 decisions/match,
   batch-mean baseline only.
3. Single-use REINFORCE: 1 grad step/batch, no clipping → wasteful, unstable, walks off the peak.
   value_head EXISTS (model.py) but unused (prior shared-trunk MSE was mis-scaled).
4. Reward misaligned with skill (betting/bluffing) — only pays hand value.
5. Policy degenerate/exploitable: raw head collapsed (97% illegal top token); argmax play is readable.
6. No exploitability metric. Capacity is NOT the bottleneck.

## Plan
### Phase 1 — Stabilize & de-variance RL  [IN PROGRESS]
- [x] Config flags (default "reinforce" → existing path byte-identical): rl_algo, ppo_clip,
      ppo_epochs, value_coef, use_value_baseline. (config.py:53-58; not arch fields → ckpts unaffected)
- [x] Decision records old_logp + value (no_grad) in rollout (`_batched_pick` now returns
      (idxs, logps, vals); `commit` stores them). multinomial RNG draw unmoved → sampled rollouts
      bit-identical; argmax fidelity cross-check PASS.
- [x] update() dispatches on rl_algo; `_update_ppo`: clipped surrogate, ppo_epochs passes/batch,
      A = standardize(whitened-return − V_old) [or batch-mean when no baseline], value MSE on
      whitened returns (value_coef), entropy bonus. REINFORCE body left verbatim.
- [x] Unit tests: PPO step runs + keys/shapes finite + clipped surrogate rises on a fixed batch;
      value-baseline variant trains the value head. 6/6 test_rl green, 18/18 full suite.
- [~] Train vs pool w/ ppo (scripts/train_vs_pool_ppo.py, 150 iters, no value baseline first;
      same lr 3e-5 / entropy 0.01 / batch 8 as the 0.545 baseline → isolates clip+K-epoch effect).
      RUNNING → checkpoints/rl_vs_pool_ppo.pt. Smoke (4 iters): clipfrac 0.09–0.19, KL ~0.01 (stable).
      Compare peak & stability to 0.545; if promising, run --value variant.

### Phase 2 — Self-play + league  [PENDING]
- [ ] TokenPolicyAgent(net) (reconstruct seat stream from live tokenizer → net → action).
- [ ] rollout: opponent team = frozen net snapshot; league mixture {current, snapshots, pool}.
- [ ] Train self-play; verify it beats the pool-only model head-to-head.

### Phase 3 — Exploitability metric  [PENDING]
- [ ] Best-response evaluator (freeze agent, train fresh RL agent to beat it; report BR win-rate).
- [ ] Adopt as north-star (~0.50 = unexploitable); eval sampled policy, not just argmax.

### Phase 4 (stretch) — skill-aligned reward (betting EV) + test-time search (PIMC/MCTS).

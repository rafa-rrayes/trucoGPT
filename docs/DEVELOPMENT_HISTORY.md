# TrucoGPT — Development History

This document is the canonical record of how **TrucoGPT** was built. TrucoGPT is a
reinforcement-learning agent that plays **Truco** (the Brazilian 2v2 card game). It started life
as a clean, I/O-free Python game engine, grew an MLP-based PPO self-play trainer, and ended up as a
**~10M-parameter decoder-only transformer** (nanoGPT-style) trained in a three-stage pipeline —
supervised next-token pretraining on a tokenized corpus of expert-bot games, PPO fine-tuning
against an opponent pool, and a self-play league — accelerated by a custom **Rust game engine** and
an **RTX 4080** GPU.

It is reconstructed from the working-session logs of the project, covering
**2026-06-15 through 2026-06-20**. Numbers, file names, and decisions below are quoted from those
logs; where the record is ambiguous it is omitted rather than guessed.

> **Note on naming.** The repository is called `trucoDQN`, but the agent is **not** a DQN. It is a
> policy-gradient method — REINFORCE first, then PPO — over a transformer policy. The "DQN" in the
> name is a historical artifact. Early work ran on a Mac (Apple
> Silicon / MPS); from 2026-06-17 onward it ran on a Linux box with an NVIDIA RTX 4080.

---

## Architecture & metric evolution at a glance

| Date | Phase | Policy network | Algorithm | Best result |
|---|---|---|---|---|
| 06-15 | Engine + env | — | — | 91 tests pass; pure I/O-free engine |
| 06-16 AM | First RL | MLP 519→128→128, ~95K params | PPO self-play | vs random 0.98, **vs heuristic ~0.45 (plateau)** |
| 06-16 PM | Tuning + parallelism | same MLP | PPO + league + multiprocessing | vs heuristic 0.52→0.56; **5.1× rollout speedup** |
| 06-16 PM | Plateau debug | same MLP | PPO, 100% heuristic opponents | **vs heuristic 0.82** (lr 2.5e-4 + constant shaping) |
| 06-17 | Transformer + Rust + scale | GPT d_model 250→343, 7 layers, **10.2M params** | pretrain → PPO → self-play | **0.734 vs pool**; **~3,000,000 matches/s** in Rust |
| 06-17 PM | Legality penalty | same GPT (10.2M) | PPO + aux legality loss | unmasked illegal **47.9% → 7.6%** |

The single most important measured arc, all vs the strong-config opponent pool:
**0.550 (baseline PPO) → 0.588 (tuned PPO) → 0.646 (self-play league, 250-dim) → 0.734 (10.2M
transformer + 211M-token corpus + self-play).**

---

## 2026-06-15 (evening) — Founding session: the game engine

**Session:** `read-truco-md` (22:23 → next morning 10:24). Goal: read the rules doc, then
implement Truco "in code, CLI for now." Mid-session the user reframed the whole effort:

> "Our endgoal is to train an AI to play this game, keep that in mind before developing."

That single sentence set the architecture: the deliverable became a **pure, I/O-free, RL-ready
game engine**, with the CLI as just one frontend. The user also added two real mechanics —
**partner signals** (a private team communication channel) and **playing face-down** (hidden
information) — so the game is genuinely imperfect-information.

### What was built

A greenfield Python 3.14 package, zero runtime dependencies (only `pytest` for dev). Design rule:
`cards/state/actions/engine/observation/env` import no I/O, take an injected `random.Random` (no
global RNG, no print/input) so self-play is reproducible; `apply_action` deep-copies state so
transitions are pure.

The rules (per `TRUCO.md`): 2v2, 40-card deck (8/9/10/Jokers removed), rank order worst→best
`4 5 6 7 Q J K A 2 3`. The **manilha** is the rank one above the flipped *vira* (wraps 3→4) and
beats everything; the manilha suit tiebreak is worst→best `♦ ♠ ♥ ♣`, and the clubs manilha is the
**Zap**. A hand is best-of-3 rounds; the raise ladder is Truco(3) → Six(6) → Nine(9) → Twelve(12);
there is a special "Hand of Eleven" rule.

Key modules and constants created (built and test-verified one step at a time):

- **`cards.py`** — `RANK_ORDER=("4","5","6","7","Q","J","K","A","2","3")`; `Suit` IntEnum
  (DIAMONDS=0 < SPADES=1 < HEARTS=2 < CLUBS=3 = tiebreak order); `MANILHA_BASE=100`,
  `FACE_DOWN_STRENGTH=-1`; `Card.id` is a 0–39 rank-major bijection (`card_from_id`);
  `card_strength` = `100+suit` for a manilha else the rank index.
- **`state.py`** — `VALUES=(1,3,6,9,12)`, `SET_TARGET=12`, `ELEVEN=11`; helpers `team_of(s)=s%2`,
  `partner_of((s+2)%4)`, `next_seat`; `Phase` IntEnum (ELEVEN_CHOICE / PLAY / RESPOND / HAND_DONE /
  MATCH_DONE); `HandState`, `MatchState`, `PendingRaise`, `PlayedCard`.
- **`actions.py`** — 8 `SignalCode`s (WEAK, STRONG, HAS_ZAP, HAS_HEARTS, HAS_SPADES, HAS_DIAMONDS,
  HAS_THREE, HAS_TWO); action dataclasses `PlayCard(card, face_down)`, `Signal`, `CallTruco`,
  `AcceptTruco`, `RunTruco`, `ReraiseTruco`, `ElevenChoice(play)`; `describe(action)`.
- **`engine.py`** — `new_match`, `_deal_hand` (vira = `deck[12]`), `legal_actions`, `apply_action`,
  `_resolve_round`, `_decide_hand` (best-of-3 §9 tie rules, `REDEAL` sentinel = -2), 12 event
  dataclasses (HandDealt, CardPlayed, TrucoCalled, RoundResolved, HandResolved, MatchWon, …).
- **`observation.py`** — `observe(state, seat)` produces a **per-seat imperfect-information view**
  (hides other hands, hides face-down identity from non-players, exposes only the partner's signal);
  `encode(obs)` flattens it **seat-relative** ("index 0 is always me") so one network can play any
  seat. `ENCODED_SIZE` is computed from the schema. Card slots are 42-way one-hots
  (`_CARD_HIDDEN=40`, `_CARD_EMPTY=41`, `_CARD_SLOT=42`).
- **`env.py`** — `TrucoEnv`, a PettingZoo-AEC-shaped wrapper. **Reward is zero-sum and
  team-relative**: terminal **±1** on `MatchWon`, plus optional dense **`reward_shaping`** paying
  `HandResolved.value / SET_TARGET` per resolved hand.
- **`agents/`** — `Agent` protocol (`act(obs)`), `RandomAgent`, `HeuristicAgent` (rule-based, with
  `bluff=0.12`, `signal_prob=0.30`, hand-strength logic for calling/responding/eleven), `HumanAgent`.
- **`cli/`** — privacy-safe `render_board`/`narrate` (never prints opponents' signals or others'
  face-down identities) and an interactive `play.py`.

### Dead ends / fixes

- Tests initially failed with `ModuleNotFoundError: No module named 'truco'`; fixed by adding
  `[tool.pytest.ini_options] pythonpath=["."]` plus the `pytest>=8` dev dependency.
- The engine briefly carried a hacky module-level `_DUMMY_RNG_SENTINEL` because the redeal path
  needed the RNG; resolved by threading `rng` cleanly through `_apply_play → _after_round → _deal_hand`.

### Milestone

Tests grew incrementally — 21 → 26 → … → **73 passed** (engine complete) → **91 passed in 0.25s**
(full project, wheel builds). Manual smoke runs (`uv run truco --seed 3`, `python -m truco`) played
full matches end to end.

The session produced **no RL code** — that was explicitly out of scope. It ended with the user
asking Claude to *write a prompt* for the next session to build the RL trainer.

---

## 2026-06-16 (morning) — First RL: MLP + PPO self-play

**Session:** `implement-rl-training-for-truco-agent` (10:25 → 10:58). Built the first learner.

### Key decisions

- **Algorithm: PPO** (clipped) with a **single shared policy** playing all four seats — justified by
  the seat-relative encoder (one net can be any seat). A `Plan` subagent verified the only gating
  risk first: does a torch wheel exist for Python 3.14? `uv add torch` resolved **torch==2.12.0**
  (CPU, no CUDA on the Mac); a benign "Failed to initialize NumPy" warning persisted throughout
  (NumPy was never installed — the design is torch-only).
- **Action space**: a fixed **`ACTION_SIZE=94`**, absolute card-id-based layout (in
  `truco/rl/action_space.py`): `0–39` play face-up, `40–79` play face-down, `80–87` signals, `88`
  CallTruco, `89` AcceptTruco, `90` RunTruco, `91` ReraiseTruco, `92/93` ElevenChoice. Illegal
  logits are **masked to `-1e9`** (finite, to avoid NaNs) before softmax. `ENCODED_SIZE` was
  recomputed to **519**.
- **Network** (`truco/rl/policy.py`, `PolicyNet`): MLP `Linear(519,128) → Tanh → Linear(128,128) →
  Tanh`, then a `policy_head` (94) and a `value_head` (1) on the shared trunk — **~95K params**.
- **Credit assignment**: turn-based — after each `env.step`, each seat's step reward is added to its
  *most recent* transition; per-seat GAE (`gamma=0.99`, `gae_lambda=0.95`, terminal bootstrap 0).
- **`TrainConfig` defaults**: `total_updates=200`, `matches_per_update=16`, `lr=3e-4`, `clip=0.2`,
  `epochs=4`, `minibatch_size=256`, `ent_coef=0.01`, `vf_coef=0.5`, `hidden=128`, grad-clip 0.5.
- New `truco/rl/ppo.py` (`train`, `_rollout_match`, `_ppo_update`, `evaluate`),
  `truco/rl/train.py` (argparse auto-generates a CLI flag per config field), `PolicyAgent`.

### Results

- Sanity run (15 updates): vs random climbed to **0.915**, vs heuristic to **0.43**.
- Full default run (200 updates, the user ran it): **vs random 0.980**, but **vs heuristic
  plateaued ~0.40–0.475** — best checkpoint was update 160 at **0.475**, never crossing 0.5.
- The full test suite went 91 → **96 passed**. The `--policy` CLI wiring ran 20 full matches with
  **zero illegal-action errors**.

### The first diagnosis

The assistant called the heuristic plateau **a methodology ceiling, not a bug**: pure self-play
optimizes against your own clones (wrong opponent distribution); reward shaping biases toward
point-greedy rather than match-optimal play; and the eval methodology was flawed —
`eval_seed = seed + 7_000_000 + update` uses **different decks every eval**, so the "best checkpoint"
chases deck luck, and 200 eval matches give ~3.5% std (so 0.40 vs 0.475 is mostly noise).

The user requested four improvements (heuristic variants + opponent mixing; anneal reward shaping;
temperature sampling; a league) and again asked Claude to *write a handoff prompt* rather than
implement directly.

---

## 2026-06-16 (late morning) — "Beat HeuristicAgent": variants, league, annealed shaping

**Session:** `beat-heuristicagent-with-improved-rl-training` (10:58 → 11:25). Implemented all four
improvements under a hard constraint: the engine/data layer (`engine.py`, `state.py`,
`observation.py`, `env.py`, `actions.py`, `cards.py`) must stay **byte-for-byte unchanged**, and any
change to `heuristic.py`/`policy_agent.py` must default to today's behavior.

### Key decisions

- **Parameterized heuristic** — `HeuristicAgent` gained an `aggression` scalar (`_risk_shift =
  0.15 * aggression`) that shifts every decision gate; `aggression=0.0` is byte-identical to before.
- **`truco/rl/opponents.py`** (new) — `make_heuristic_variants(rng)` builds 5 named styles:
  `default`, `tight-passive`, `loose-aggressive`, `big-bluffer`, `honest-rock`. A `League` class
  holds the heuristics plus **FIFO-bounded frozen learner snapshots**; `sample_opponent(net)` draws
  self / heuristic / snapshot by weight.
- **Reward moved into the trainer** (env built with `reward_shaping=False`): `_step_rewards(events,
  shaping_coef)` pays `MatchWon` ±1 always and `HandResolved` a `shaping_coef`-scaled,
  target-normalized point value. `shaping_coef(update)` **linearly anneals** start→end.
- **Temperature sampling** everywhere — `distribution(obs, mask, temperature)` divides masked logits
  by T; `rollout_temperature=1.0`, `eval_temperature=0.75`; eval reports both greedy and sampled.
- **Eval-seed fix**: `eval_seed = seed + 7_000_000` (fixed, no `+update`); `eval_matches` raised to
  400. New config knobs: `opponent_mix_prob`, `league_self_prob/heuristic_prob/snapshot_prob`,
  `league_snapshot_every/max_snapshots`, the shaping anneal fields.

### Results & how it ended

- 9 new tests in `tests/test_rl_improvements.py` (variants distinct; mixed rollouts record only
  learner seats; shaping schedule; temperature→entropy monotonic; league FIFO bound) — all passed;
  total suite **105**.
- Calibration showed **eval dominated runtime** (a single eval point = four 400-match tournaments),
  so `eval_every` was bumped 10 → 20. A 60-update probe still only reached ~0.27 vs heuristic, so the
  decisive 200-update verdict was deferred. The user killed the run; the >0.5 acceptance criterion
  was **not yet verified** at session end.

---

## 2026-06-16 (midday) — Performance & evaluation investigation

**Session:** `investigate-model-training-performance-and-parameters` (11:27, spanning to evening).
The user asked, repeatedly and pointedly, "Why is training taking so long? How many params? What is
taking so long?"

### Findings (concrete)

- **Model size: 95,327 params** (`519→128→128`, heads). Tiny — microseconds per forward pass.
- The bottleneck is **single-threaded pure-Python match simulation with batch-1 inference**, not the
  net. One self-play match ≈ 26 ms; **eval was the dominant cost** — 16,000 eval matches vs 3,200
  rollout matches in a default run, because every eval cycle ran four 400-match tournaments.
- vs random was saturated (0.90→0.965, uninformative); vs heuristic peaked **0.412 at update 40**
  then decayed — and the decay **coincided with shaping annealing to ~0** around update 120.

A `Plan` subagent surfaced the core insight: a match is best-of-3 sets to 12 points (~10–25+ hands,
100–400 steps) but pays only **one** terminal ±1; with `gamma=0.99`, `0.99^200 ≈ 0.13` discounts
that terminal reward to near-irrelevance for early-hand decisions, so once shaping hits 0 there is
almost no gradient left.

### The config-only A/B (plan `witty-frolicking-quokka.md`)

Changed `TrainConfig` defaults — **`shaping_coef_end` 0.0 → 0.15** (a permanent dense-signal floor,
the headline fix), `matches_per_update` 16 → **64** (4× terminal outcomes), `minibatch_size` 256 →
**1024**, `ent_coef` 0.01 → **0.02**, heavier heuristic exposure (`opponent_mix_prob` 0.5 → 0.6,
`league_heuristic_prob` 0.5 → 0.65), `total_updates` → 150. Saved to a separate
`truco_policy_v2.pt` to preserve the 0.412 baseline.

### Result

The v2 checkpoint reached **vs heuristic 0.52** (greedy) / 0.4975 (sampled) at update 120, with
**vs random 0.9675** — strictly better than the baseline and with the plateau broken. 105 tests
passed. The session also produced (by request) a **handoff prompt for multiprocessing** the rollouts.

---

## 2026-06-16 (12:52) — Parallelize PPO rollouts with multiprocessing

**Session:** `parallelize-ppo-rollouts-with-multiprocessing`. The 64 matches in an update are
independent given the current weights — embarrassingly parallel. Profiling: rollout was 77% of
runtime. The box had `os.cpu_count() = 10`.

### Design (in `truco/rl/ppo.py`)

- **Hermetic matches** — each match reseeds `torch.manual_seed(spec["seed"])`, making results
  **independent of worker count** (a hard requirement for reproducibility). Opponents are described
  by **picklable descriptors** via the new `League.sample_descriptor()` (`{"kind":"self"}`,
  `{"kind":"snapshot","index":i}`, `{"kind":"heuristic",...}`), which consumes the RNG identically
  to `sample_opponent`.
- **Persistent spawn pool** — `multiprocessing.get_context("spawn").Pool(workers,
  initializer=_worker_init)` created once; `_worker_init` calls `torch.set_num_threads(1)` to stop N
  workers oversubscribing cores. CPU `state_dict`s (~380 KB) shipped per update.
- **Order-preserving contiguous chunks** (`_contiguous_chunks`) so the pooled transition order
  equals the serial order → the PPO batch is identical regardless of worker count.
- New config field **`num_workers: int = 0`** (0 = auto `max(1, cpu_count-1)`, capped at
  `matches_per_update`; ≤1 = in-process serial path, which the tests use). `_mixed_setup` removed.

### Bugs & results

- The **only** real failure was a benchmark-script bug: `/tmp/bench_rollout.py` lacked an
  `if __name__ == "__main__":` guard, so under `spawn` every worker re-ran `bench()` → recursive
  `train()` → recursive `Pool`. (The real `truco-train` CLI was already guarded.)
- A separate, near-empty session, `debug-multiprocessing-semaphore-leak`, captured the
  `resource_tracker: There appear to be 6 leaked semaphore objects` warning — a known torch +
  multiprocessing-pool artifact at shutdown, addressed in the main session by `close()/join()` in a
  `finally` block.
- A new test `test_rollout_is_worker_count_independent` proves **bitwise-identical weights** across
  `num_workers=1` vs `3` after 2 updates (`torch.equal` on every parameter). Suite → **106 passed**.
- **Steady-state speedup: 5.09×** per rollout (2.07s → 0.41s per update, 9 workers); the default
  end-to-end run hit a new peak **vs heuristic 0.560** at update 100.

---

## 2026-06-16 (13:25) — Debugging the learning plateau (the breakthrough recipe)

**Session:** `debug-model-learning-plateau-in-reinforcement-training`. The user pasted a 200-update
log showing vs-heuristic peaking ~0.49 and decaying, and set a **hard goal via a Stop hook**:
iterate hyperparameters until **vs heuristic > 0.75**.

### Root causes identified

1. **Dense reward annealed away** — `shaping_coef_end=0.15` (and earlier 0) killed the only signal
   teaching hand-level skill; the peak-then-decay matched the anneal schedule exactly.
2. **Learning rate too low** — `lr=5e-5` is ~5× below standard PPO; "the policy crawls."
3. **Self-play dilution** — 40–60% self/snapshot opponents; "optimizing 'beat myself' doesn't
   transfer to 'beat the heuristic.'" (The agent's *partner* is itself — one shared net drives both
   team seats — and unlike the heuristic pair it doesn't signal informatively, a real handicap.)

### The experiments (`tmp/exp.py` harness, config overrides only — no source edits)

A diagnostic sweep at 30 updates each plateaued knob-tuning at ~0.40–0.45:

| Run | Change | best vs heuristic |
|---|---|---|
| A_base | defaults | 0.380 |
| B_lr | lr=2.5e-4 | 0.430 |
| C_lr_shape | lr=2.5e-4, shaping_coef_end=1.0 | 0.430 |
| D_full | C + 80% mix, 85% heuristic | 0.450 |

The breakthrough was direction **E — train against 100% heuristic opponents** (`opponent_mix_prob=1.0`,
`league_heuristic_prob=1.0`, lr 2.5e-4, **constant shaping 1.0, no anneal**):

- `E_pureheur` (50 updates): climbed to **0.590** and *was still rising*.
- `E2` (150 updates): vs heuristic **0.760 @ u120, 0.820 @ u135, 0.76/0.77 @ u150** — **target
  crossed**. vs random stayed ~0.90–0.98.

**Winning recipe: lr 2.5e-4 + constant per-hand shaping (coef 1.0) + 100% heuristic opponents**,
taking the agent from the ~0.44–0.49 plateau to **0.82 vs heuristic**. The session ended with the
user's next ask — "increase model size and make it train against itself once it has beaten the
heuristic" — which seeded the next, much larger phase.

(A short separate session, `define-agent-reward-function`, documented the reward design at this point:
zero-sum ±1 on `MatchWon` plus the annealed per-hand shaping, with the floor rationale tied to the
`gamma`-discount-over-long-horizon problem.)

---

## 2026-06-17 — The big rebuild: transformer, Rust engine, and scale

**Session:** `apply-cuda-optimization-across-codebase` (10:39 → ~21:09). This is the largest session
by far (~18,500 transcript lines) and the one where the project changed shape. By this point the code
had already been substantially rewritten between sessions into the form it has today: the policy is a
**decoder-only transformer (TrucoGPT)** trained as a **3-stage pipeline** — supervised next-token
**pretraining** on a tokenized corpus of expert games, then **PPO** fine-tuning, then a **self-play
league** — and the work now ran on a Linux box with an **NVIDIA RTX 4080 (16 GB), CUDA 13.2, PyTorch
2.12.0+cu130**, 32 CPU cores.

### Starting architecture

`truco/rl/model.py` `TrucoGPT`: a nanoGPT-style decoder-only transformer. Config (`truco/rl/config.py`):
`block_size=512`, `d_model=225`, `n_layer=5`, `n_head=5`, `dropout=0.05`. A **188-token vocabulary**
(`vocab.py`: `VOCAB_SIZE=188`, `MOVE_BASE=93`); the **94 action tokens are a contiguous slice
`[MOVE_BASE : MOVE_BASE+ACTION_SIZE]` of the LM head**, so pretraining the language-model head warms
the policy directly. Two heads on a shared trunk: LM/policy head + value head (value head unused —
REINFORCE uses a batch-mean baseline). Games are turned into **per-seat redacted token streams** by
`truco/rl/tokenizer.py` (`MatchTokenizer`), so a frozen snapshot can act from any seat exactly like
the learner.

### Goal 1: "Is our code using CUDA? Apply it everywhere."

Investigation found NN training already ran on the GPU (`pick_device()` resolves mps→cuda→cpu). The
real bottleneck was elsewhere: the user had a **tournament** running — `truco.tournament --configs
50000 --games 30` — saturating all 32 cores at ~88% while the **GPU sat idle at 7%**. The tournament
is the branchy Python game state machine over **1.5M best-of-3 matches**; that is task-parallel
divergent control flow, which wants CPU cores, not GPU SIMT lanes. Offered the choice of a faster
engine, the user said: **"Implement it in Rust."**

That long tournament finished and then **crashed** at corpus generation on `AttributeError: 'Config'
object has no attribute 'corpus_matches'` (a field had been commented out). Fix: restore
`corpus_matches`. The ranking it produced: top config wr=0.836; **textbook config ranked 187/50000,
wr=0.742**.

### The Rust fast engine (`truco_fast/`)

Built with **PyO3 0.29 + maturin 1.14** (Rust 1.95). `src/engine.rs` (~640 lines) reimplements the
deal/resolve/decide logic, the truco ladder, and the `ConfigurableBot` (14-knob rule-breaking bot),
using a dependency-free **splitmix64** RNG and **rayon** (`par_iter().fold().reduce()` with
commutative integer adds, so results are deterministic regardless of thread scheduling).
`src/lib.rs` exposes `run_tournament(...)` (releasing the GIL via `py.detach`) and `play_match(...)`.

- **Bug:** PyO3 0.29 renamed GIL release `allow_threads → detach`; fixed.
- **Equivalence (not bit-exact — the Python bot's signaling RNG is already non-deterministic across
  processes, so the bar is *statistical*):** a 40k-match cross-engine comparison gave per-config
  Pearson **r = 0.823**, which a variance decomposition showed **sits exactly at the noise ceiling
  (0.830)** for two independent-pairing runs — i.e. zero detectable drift.
- **Speedup (the major milestone):** Python ≈ 3,351 matches/s; **Rust ≈ 2,340,062 matches/s**, and
  **1.5M matches in 0.49s ≈ 3,035,237 matches/s** — roughly **900× over the fresh Python backend**
  (≈2,200× over the historical run). `run_tournament` in `truco/tournament/run.py` got a `backend`
  param ("auto" uses Rust if importable, else the Python pool). All 6 tournament tests still passed.

### Corpus → pretrain → PPO pipeline

The Rust engine accelerates *ranking*, not *tokenization* (which must stay Python to emit streams).
A first corpus run produced **112,000 streams / 22,340,052 tokens** (target >20M). The
opponent pool (`build_opponent_pool.py` → `opponent_pool.json`) kept **9,840 / 20,000 configs**
(wr>0.5); textbook ranked 102/20000 (wr 0.766). Pretrain reached `val_token_acc ≈ 0.738`.

The first PPO run vs the pool (`train_vs_pool_ppo.py`, clip 0.2, epochs 4, lr 3e-5) went **start
0.380 → best 0.550** vs pool. Diagnosis: `rl_batch_matches=8` made advantage noise dominate; PPO was
barely engaging (clip≈0, kl≈0); constant LR walked off.

**Tuning, "to make an unbeatable model"** — applied: `pretrain_steps` 3000 → 6000, `rl_lr` 3e-5 →
6e-5, `lr_decay` constant → cosine, `rl_batch_matches` 8 → 48, `rl_minibatch` 8 → 16, `eval_matches`
200 → 500; the model was also bumped to `d_model=250`. A gated `run_pipeline.sh` (with a hard
>20M-token guard before pretraining) produced a 23.4M-token corpus, pretrain `val_token_acc 0.749`,
and a PPO run that climbed cleanly to **best 0.588 vs pool** (start 0.480) at iter 275 — no walk-off.

### Self-play league

Asked for the next lever, the user chose **self-play league**. `truco/rl/rollout.py` was extended so
`_MatchRunner` can take an `opponent_net`; `advance()` now tags each decision `learner`/`opp` and
records a `Decision` only for learner seats, and `_play_group` runs **two separate batched forward
passes** per round (learner weights and opponent weights can't share a batch). `reinforce.py`'s
`reinforce(...)` gained an `opponent_sampler(it)`; `scripts/train_selfplay.py` warm-starts from the
0.588 PPO model and samples **60% self / 25% pool / 15% default** once the league has snapshots
(`LEAGUE_SIZE=8`, `SNAPSHOT_EVERY=20`). 12 tests passed.

**Result:** the self-play run climbed **start 0.564 → best 0.646 vs pool** (peak 0.646 @ iter 200),
and the new champion **beat the prior PPO champion 603/1000 = 0.603 head-to-head** (≈6.6σ above
coin-flip). Champion → `checkpoints/rl_selfplay.pt`. (A model-download attempt to a Mac was
hard-blocked by the harness's data-exfiltration classifier and abandoned.)

### "Go big" — bigger model + far more diverse data

The user pushed the two levers that "ACTUALLY make a BIG difference: model and training size,"
asking for a bigger architecture (~"343, 7, 7") and a corpus that faces **all kinds of foes and many
kinds of partners**. A key data-diversity bug was found: `match.py` assigned configs **per team**, so
teammates were always homogeneous. Fixes:

- **Architecture** bumped to `d_model=343` (= 7×49), `n_layer=7`, `n_head=7` → **10,219,343 params
  (10.2M)**; `pretrain_steps` 6000 → 20,000, `pretrain_batch` 16 → 32. The 250-dim champions were
  backed up as `*_250.pt`.
- **`generate_diverse_corpus`** + `_run_seats` + a `--mixed-partners` flag seat **four independently
  sampled configs per match**, so teammates vary.
- **`run_big.sh`** ran the full pipeline: a 20,000-config tournament (textbook ranked 435/20000,
  wr=0.651), a mixed-partners corpus of **1,000,000 streams / 211,846,172 tokens (≈427 MB)**, a
  20,000-step pretrain of the 10.2M model, pool-PPO, and a self-play league.

**Result — the biggest jump yet:** the big champion reached **0.734 vs pool** (vs default 0.660) and
**beat the small (250/5/5) champion 659/1000 = 0.659 head-to-head**. New champion →
`checkpoints/rl_selfplay.pt`.

### GPU & corpus optimizations (end of session)

`pretrain.py` got `torch.autocast(bfloat16)` + `torch.compile` (pretrain **41 → ~10 min, ~4×**);
`corpus.py` parallelized tokenization with `ProcessPoolExecutor` (**20 min → ~1.4 min, ~14×**). Two
god-mode tools were added: `scripts/sim_vs_default.py` (TrucoGPT vs HeuristicAgent with all four
hands revealed) and a stdlib-only **`scripts/web_playground.py` + `playground.html`** that lets you
edit the token stream and watch the model's move distribution change live (served at
`http://localhost:8000`). Suite → **18 passed**.

---

## 2026-06-17 (20:16) — The legality penalty

**Session:** `retrain-model-with-illegal-move-penalties`. The user asked: can we retrain (warm
start) while **penalizing the model for choosing illegal moves**?

### The discovery: a mask-dependent policy

Legality is enforced everywhere by a **hard mask** — `model.action_logits()` slices the 94-action
block and does `masked_fill(mask==0, -1e9)` before softmax, used for rollout sampling, eval argmax,
the gradient update, and inference. So the model is *structurally* prevented from playing illegally,
and a naive "penalty for choosing illegal" would never fire. But pretraining only ever pushed legal
next-tokens *up* and RL only ever reinforced *masked* picks — **nothing ever pushed illegal logits
down.** The mask was doing nearly all the legality work.

Measuring the champion's **unmasked** policy (`scripts/measure_illegal.py`, 200 matches, 6,584
decisions):

- **mean illegal probability mass = 0.5476**
- **argmax-illegal rate = 0.4793 (47.9%)** — i.e. without the mask, ~48% of the champion's top moves
  would be illegal
- worst-case mass = 1.0000

### The fix: an auxiliary legality loss

Chosen by the user (over an unmasked-reward-penalty approach), keeping the mask as a safety net at
inference. Added `Config.legality_coef: float = 0.0` (0.0 = off, byte-identical path). A helper
`reinforce._illegal_mass(lm_logits, masks)` computes the **softmax probability the *unmasked* policy
puts on illegal actions** — `((1 - masks) * softmax(raw)).sum()`, in [0,1] per row — and is added to
the loss in **both** the REINFORCE and PPO update paths when `legality_coef > 0`. Its gradient
self-limits to 0 as the illegal mass shrinks. A new `scripts/train_legality.py` warm-starts from the
champion and writes a *separate* `checkpoints/rl_legality.pt` (champion untouched).

### Results & a saved-checkpoint gotcha

A smoke run took illegal mass 0.47 → ~0.27 in 12 iters with win-rate held. The first 50-iter
characterization run dropped illegal mass **0.55 → 0.084 (~85%)** while win-rate-vs-pool held
0.68–0.72 — **but the checkpoint was never saved**: `reinforce()` only checkpoints when win-rate
*beats* the warm-start baseline, and legality isn't in the selection metric. Fix: capture the
returned net and save it **unconditionally** at the end with full legality meta.

The final saved run (50 iters, `legality_coef=0.5`, cosine LR) produced `rl_legality.pt`:

| Metric | Champion baseline | rl_legality.pt | Change |
|---|---|---|---|
| argmax-illegal (mask-free) | **47.9%** | **7.6%** | ~6.3× fewer |
| mean illegal mass | 0.55 | 0.094 | ~83% lower |
| win-rate vs pool | 0.730 | 0.690 | −4 pts |
| win-rate vs default | 0.660 | 0.668 | +0.8 pt |

So **legality is cheap** — for a ~2–4 point win-rate cost the underlying policy went from ~52% legal
to ~92% legal at argmax, much less reliant on the mask crutch. With the mask on (the user's choice),
actual play stays 100% legal. Most of the drop landed by iter 30, then plateaued ~0.08–0.10 as cosine
LR decayed; pushing below ~0.05 would need a higher `legality_coef` and/or more iters. The new model
was **not promoted to champion** — left as a separate checkpoint for a head-to-head decision.

---

## Final state / outcome

**The agent.** TrucoGPT is a **decoder-only transformer** (nanoGPT-style), not a DQN. The final
architecture (`truco/rl/config.py`):

- `d_model=343` (= 7×49), `n_layer=7`, `n_head=7`, `block_size=512`, `dropout=0.05` →
  **~10.2M parameters**.
- A **188-token vocabulary** (`vocab.py`), with the **94 action tokens** living as a contiguous
  slice `[MOVE_BASE=93 : 93+94]` of the LM head, so supervised pretraining directly warms the policy.
- Two heads on a shared trunk: LM/policy head + value head; legality enforced by a **hard `-1e9`
  mask** over the action block.

**The training pipeline (3 stages).** (1) **Corpus** — run a Rust tournament over 20,000 randomized
`ConfigurableBot` configs, take the top configs, and tokenize their games (with **mixed partners**)
into per-seat redacted streams — the champion corpus was **1,000,000 streams / 211,846,172 tokens**.
(2) **Supervised pretrain** — 20,000 steps of next-token prediction (bf16 + `torch.compile`),
reaching ~0.73–0.75 token accuracy. (3) **RL fine-tune** — PPO vs the strong-config opponent pool
(`opponent_pool.json`, 9,840 configs with wr>0.5), then a **self-play league** (60% self / 25% pool /
15% default, up to 8 frozen snapshots).

**The Rust engine.** `truco_fast/` (PyO3 + maturin + rayon) reimplements the engine and bot at
**~3,000,000 best-of-3 matches/s** (≈900× the Python backend), statistically equivalent to the
Python engine (cross-engine r=0.823, at the noise ceiling). This is what made the 211M-token corpus
and the large opponent pool cheap.

**The numbers.** The win-rate-vs-pool arc was **0.550 → 0.588 → 0.646 → 0.734**. The final 10.2M
champion (`rl_selfplay.pt` in the 06-17 session) scored **0.734 vs the pool, 0.660 vs the default
heuristic**, and beat the previous 250-dim champion **65.9% head-to-head**. The earlier MLP line had
been driven from a ~0.45 plateau to **0.82 vs the heuristic** by the lr-2.5e-4 / constant-shaping /
100%-heuristic recipe. The legality work cut the unmasked illegal-move rate from **47.9% to 7.6%**
for a ~4-point win-rate cost, producing `rl_legality.pt`.

**Repository layout (as of the export).** `truco/` (the pure engine: `cards/state/actions/engine/
observation/env`, plus `cli/` and `agents/`), `truco/rl/` (the transformer stack: `model.py`,
`config.py`, `vocab.py`, `tokenizer.py`, `action_space.py`, `rollout.py`, `reinforce.py`,
`pretrain.py`, `eval.py`, `opponents.py`, `data.py`), `truco/tournament/` (corpus & ranking),
`truco_fast/` (Rust engine), `scripts/` (`train_vs_pool_ppo.py`, `train_selfplay.py`,
`train_legality.py`, `measure_illegal.py`, `sim_vs_default.py`, `web_playground.py` +
`playground.html`, …), and the pipeline runners `run_big.sh`, `run_selfplay.sh`, `run_legality.sh`.
The live `checkpoints/` holds `corpus.npz` (≈427 MB), `pretrain.pt`, `opponent_pool.json`, and
`rl_legality.pt` (the larger self-play checkpoints from the 06-17 session having since been pruned).

**Open threads.** Pushing unmasked illegal moves below ~5% would need a higher `legality_coef`; and the
champion still loses ~27–40% of games to the strongest configs in the pool — "unbeatable" remains
aspirational.

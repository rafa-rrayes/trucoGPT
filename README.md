# TrucoGPT

A small GPT-style transformer that learns to play **Truco** — the Brazilian bar card game of
bluffing, partner signalling, and escalating bets. The agent is trained in three stages:
imitation **pretraining** on a corpus of strong tournament play, **reinforcement learning** against
a population of opponents and itself, and a final **legality fine-tune** that teaches the raw policy
to stop proposing illegal moves.

> The full rules of the game are in [`docs/TRUCO.md`](docs/TRUCO.md).
> The blow-by-blow story of how this project was built is in
> [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md).

*(The directory is historically named `trucoDQN`, but the final method is a transformer policy
trained with REINFORCE/PPO — not a DQN.)*

---

## The champion

The shipped model is [`checkpoints/rl_legality.pt`](checkpoints) — a **10.2M-parameter** transformer
(`d_model=343`, `7` layers, `7` heads, context `512`). Measured over held-out evaluation matches:

| Metric | Value |
|---|---|
| Win-rate vs the strong opponent **pool** (9,840 configs) | **0.69–0.73** |
| Win-rate vs the **textbook** heuristic | **0.67** |
| Unmasked illegal-move mass (before → after legality fine-tune) | **51% → 9%** |
| Greedy (argmax) illegal rate after fine-tune | **7.6%** |

The opponent pool is the top half (win-rate > 0.5) of 20,000 hand-design-sampled rule-bending bots;
its strongest member wins 90% of its games, and a perfectly "by-the-book" bot only ranks ~102nd —
confirming that good Truco needs measured bluffing and signalling, not textbook play.

---

## How it works

```
                 tournament                pretrain                 RL fine-tune
  20k bot         (Rust engine)    211M-token    (next-token)   PPO vs pool  →  self-play league  →  legality
  configs  ───►  top configs play  ───► corpus  ───────────►   transformer  ───────────────────►   fine-tune
  (LHS)          ~250k matches            .npz                  (imitation)   (0.72)      (0.73)     (champion)
```

1. **Tournament & corpus.** Thousands of `BotConfig`s (a fully-parameterised rule-bending bot) are
   Latin-hypercube-sampled and play a large round-robin. The top configs' games are tokenised into a
   **corpus of 1,000,000 streams / ~212M tokens**. A Rust engine (`truco_fast`) makes this ~10–50×
   faster than the pure-Python engine.
2. **Pretraining.** The transformer learns to imitate that strong play as a next-token task
   (20k steps, batch 32) — reaching ~0.74 validation token-accuracy.
3. **RL.** Warm-started from the pretrained weights, the policy is fine-tuned with PPO, first against
   the frozen opponent pool (→ 0.72 vs pool), then in a **self-play league** of frozen snapshots of
   itself (→ 0.73 vs pool).
4. **Legality fine-tune.** A final pass adds an auxiliary penalty (`legality_coef`) on the
   *unmasked* policy's illegal probability mass, cutting raw illegal moves from ~51% to ~9% while
   keeping win-rate — yielding the champion `rl_legality.pt`.

---

## Repository layout

```
truco/                 The game + the model (the "harness")
  cards.py state.py engine.py actions.py observation.py   pure game engine
  env.py                                                  RL environment wrapper
  agents/                heuristic, configurable, policy, human, random players
  cli/                   play a match in your terminal
  tournament/            config sampling, round-robin, corpus generation
  rl/                    tokenizer, model (nanoGPT), pretrain, REINFORCE/PPO, rollout, eval
truco_fast/            Optional Rust reimplementation of the engine (PyO3), for fast corpus gen
scripts/               Training drivers + simulation / evaluation / web-playground tools
tests/                 Engine, tokenizer, model, and RL unit tests
checkpoints/           rl_legality.pt (the champion) + opponent_pool.json  [big artifacts gitignored]
docs/                  TRUCO.md (rules), PLAN.md (design notes)
run_big.sh             Full training pipeline: corpus → pretrain → PPO → self-play
run_selfplay.sh        Self-play league stage (standalone) + head-to-head check
run_legality.sh        Final legality fine-tune → the champion
```

---

## Setup

Requires Python ≥ 3.12. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The Rust fast-engine is **optional** — everything works on the pure-Python engine; the Rust backend
just speeds up corpus generation. To build it:

```bash
cd truco_fast && uv run maturin develop --release
```

---

## Play against the champion

In your terminal (you are seat 0, partnered with seat 2):

```bash
uv run python -m truco --policy checkpoints/rl_legality.pt     # face three copies of the model
uv run python -m truco                                         # face the heuristic bots instead
```

In your browser (a visual board that also shows the model's move probabilities):

```bash
uv run python scripts/web_playground.py        # then open http://localhost:8000
```

Watch the model play itself or the heuristics, with full move-by-move narration:

```bash
uv run python scripts/sim_best_model.py        # champion vs the textbook heuristic
uv run python scripts/sim_match.py             # champion vs the #1 pool config
```

---

## Reproduce the training

The whole pipeline is GPU-friendly and gated stage-by-stage. From scratch:

```bash
./run_big.sh        # corpus (≈212M tokens) → pretrain (10.2M params) → PPO vs pool → self-play league
./run_legality.sh   # legality fine-tune of the self-play champion → checkpoints/rl_legality.pt
```

`run_big.sh` regenerates the large artifacts (`checkpoints/corpus.npz`, intermediate checkpoints)
that are intentionally **not** committed to git. Individual stages can also be run on their own —
see `scripts/train_vs_pool_ppo.py`, `scripts/train_selfplay.py`, and `scripts/train_legality.py`.
Hyperparameters live in [`truco/rl/config.py`](truco/rl/config.py).

---

## Tests

```bash
uv run pytest -q
```

---

See [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md) for the full development narrative — the dead
ends (the multiprocessing semaphore leak, the learning plateau, the 48%-illegal raw policy), the key
decisions (REINFORCE → PPO, the self-play league, the Rust engine), and the benchmark numbers along
the way.

"""Self-play league fine-tuning — push TrucoGPT past the pool-trained 0.588 ceiling.

Warm-starts from the strongest checkpoint (`rl_vs_pool_ppo.pt`) and PPO-fine-tunes against a
*league*: a sliding window of frozen snapshots of the model itself, mixed with the strong-config
pool and the default heuristic. Beating fixed heuristics plateaus; having to beat your own
ever-improving play does not — that's the route toward a genuinely unbeatable policy.

Each iteration faces ONE opponent drawn from the league, so the whole rollout batch shares the
opponent's weights and stays batched; over many iterations the learner sees the entire league.
Every `SNAPSHOT_EVERY` iters the current net is frozen and appended (capped at `LEAGUE_SIZE`
most-recent snapshots).

The selection metric stays the **pool** win-rate (so it's directly comparable to the prior runs),
with vs-`default` printed alongside. Best checkpoint → `checkpoints/rl_selfplay.pt`.

Run: `uv run python scripts/train_selfplay.py [ITERS] [POOL.json]`
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truco.agents import BotConfig
from truco.rl.config import Config
from truco.rl.model import TrucoGPT, pick_device
from truco.rl.reinforce import reinforce

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
POOL_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    Path(__file__).resolve().parent.parent / "checkpoints" / "opponent_pool.json")

LEAGUE_SIZE = 8        # frozen snapshots kept resident (sliding window of the most recent)
SNAPSHOT_EVERY = 20    # iters between adding a snapshot of the current net to the league

_FIELDS = {f.name for f in dataclasses.fields(BotConfig)}


def load_pool(path: Path) -> tuple[list[BotConfig], dict]:
    data = json.loads(path.read_text())
    pool = [BotConfig(**{k: v for k, v in e["knobs"].items() if k in _FIELDS})
            for e in data["pool"]]
    return pool, data["meta"]


def snapshot(net: TrucoGPT, device: str) -> TrucoGPT:
    """A frozen, eval-mode deep copy of the net — an opponent that never trains."""
    snap = copy.deepcopy(net).to(device)
    snap.eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap


def main() -> int:
    pool, meta = load_pool(POOL_PATH)
    device = pick_device()
    print(f"device: {device}")
    print(f"opponent pool: {len(pool)} configs (wr>{meta['wr_threshold']} of "
          f"{meta['n_configs']}, best-of-{meta['best_of']}); top wr {meta['top_win_rate']:.3f}")
    print(f"self-play league: up to {LEAGUE_SIZE} snapshots, one added every {SNAPSHOT_EVERY} iters")

    cfg = Config(
        rl_algo="ppo", ppo_clip=0.2, ppo_epochs=4,
        rl_lr=6e-5, lr_decay="cosine", entropy_coef=0.01,
        rl_batch_matches=48, rl_minibatch=16,
        rl_iters=ITERS, eval_every=25, eval_matches=500,
        # warm-start from our strongest model so far, not the raw pretrain
        pretrain_ckpt="checkpoints/rl_vs_pool_ppo.pt",
        rl_ckpt="checkpoints/rl_selfplay.pt",
    )

    league: list[TrucoGPT] = []
    rng = random.Random(cfg.seed)

    def opponent_sampler(it: int) -> dict:
        """Mostly self-play once snapshots exist; keep pool/default in the mix for breadth."""
        r = rng.random()
        if league:
            if r < 0.60:
                return {"net": rng.choice(league)}        # beat a past self
            if r < 0.85:
                return {"configs": pool}                  # keep beating the strong configs
            return {}                                     # and the canonical heuristic
        return {"configs": pool} if r < 0.7 else {}       # warm-up before any snapshot exists

    def on_iter(it: int, net: TrucoGPT) -> None:
        if it % SNAPSHOT_EVERY == 0:
            league.append(snapshot(net, device))
            if len(league) > LEAGUE_SIZE:
                league.pop(0)                             # drop the oldest snapshot
            print(f"    + league snapshot @ iter {it} (league size {len(league)})")

    reinforce(cfg, device=device, opponent_configs=pool,
              opponent_sampler=opponent_sampler, on_iter=on_iter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

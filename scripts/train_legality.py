"""Warm-start fine-tune that adds a legality objective on top of the current champion.

Same proven self-play-league recipe that produced `rl_selfplay.pt` (PPO, sliding league of frozen
snapshots, opponent mix self/pool/default, cosine LR), but:

  * it **warm-starts from the champion** (`rl_selfplay.pt`) instead of the pool-trained model, and
  * it turns on `legality_coef` so the loss penalises the *unmasked* policy's illegal probability
    mass (see `reinforce._illegal_mass`).

Masked play is unchanged — every move the net actually makes is still legal — so the win-rate
objective and stability are untouched; the new term just teaches the raw policy to stand on its own,
turning the legality mask into a pure safety net. Selection stays the **pool** win-rate (comparable
to all prior runs); the per-iter log gains an `illeg …` column (mean unmasked illegal mass).

Run:  uv run python scripts/train_legality.py [ITERS] [LEGALITY_COEF] [POOL.json]
Then verify with:  uv run python scripts/measure_illegal.py checkpoints/rl_legality.pt 500
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

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
LEGALITY_COEF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
EVAL_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 25   # iters between 500-match win-rate evals
POOL_PATH = Path(sys.argv[4]) if len(sys.argv) > 4 else (
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
    print(f"legality penalty: legality_coef={LEGALITY_COEF} (warm-start from rl_selfplay.pt)")

    cfg = Config(
        rl_algo="ppo", ppo_clip=0.2, ppo_epochs=4,
        rl_lr=6e-5, lr_decay="cosine", entropy_coef=0.01,
        legality_coef=LEGALITY_COEF,
        rl_batch_matches=48, rl_minibatch=16,
        rl_iters=ITERS, eval_every=EVAL_EVERY, eval_matches=500,
        # warm-start from the current champion, not the pool-trained model
        pretrain_ckpt="checkpoints/rl_selfplay.pt",
        rl_ckpt="checkpoints/rl_legality.pt",
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

    net = reinforce(cfg, device=device, opponent_configs=pool,
                    opponent_sampler=opponent_sampler, on_iter=on_iter)

    # `reinforce` only checkpoints when pool win-rate beats the warm-start baseline; a legality run
    # optimises *legality*, not win-rate, so the best-by-wr gate can leave nothing saved. Persist the
    # FINAL net unconditionally (it has the lowest illegal mass) and stamp its measured legality +
    # win-rates into the meta so the trade-off is recorded with the checkpoint.
    from measure_illegal import measure

    from truco.rl.eval import win_rate
    from truco.rl.model import save

    leg = measure(net, cfg, 500, device, opponent_configs=pool)
    wr_pool = win_rate(net, cfg, 500, device=device, opponent_configs=pool)
    wr_def = win_rate(net, cfg, 500, device=device)
    save(net, cfg.rl_ckpt, meta={"stage": "legality", "iter": ITERS, "legality_coef": LEGALITY_COEF,
                                 "win_rate": wr_pool, "win_rate_vs_default": wr_def,
                                 "mean_illegal_mass": leg["mean_illegal_mass"],
                                 "argmax_illegal_rate": leg["argmax_illegal_rate"]})
    print(f"saved final net → {cfg.rl_ckpt}: wr_pool={wr_pool:.3f} wr_def={wr_def:.3f}  "
          f"illegal_mass={leg['mean_illegal_mass']:.4f} argmax_illegal={leg['argmax_illegal_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

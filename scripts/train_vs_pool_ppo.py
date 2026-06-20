"""PPO-fine-tune the pretrained TrucoGPT against the strong-config opponent pool.

Phase-1 of the "make the agent unbeatable" plan: same opponent pool, warm-start, and known-good
RL hyperparameters as `train_vs_pool.py` (the REINFORCE baseline that peaked at 0.545 vs pool then
walked off), but with the PPO-clipped, K-epoch update (`rl_algo="ppo"`). Holding everything else
fixed isolates PPO's effect on the *peak* and the *stability* of the win-rate curve.

The value baseline is OFF by default (`--value` turns it on): with it off the advantage is exactly
the REINFORCE advantage (batch-standardised whitened returns), so the only change from the baseline
run is the clip + reusing each rollout batch for `ppo_epochs` gradient steps.

Best checkpoint → `checkpoints/rl_vs_pool_ppo.pt` (the 0.545 `rl_vs_pool.pt` is left untouched for
comparison).

Run: `uv run python scripts/train_vs_pool_ppo.py [ITERS] [--value] [POOL.json]`
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truco.agents import BotConfig
from truco.rl.config import Config
from truco.rl.model import pick_device
from truco.rl.reinforce import reinforce

_args = [a for a in sys.argv[1:] if a != "--value"]
USE_VALUE = "--value" in sys.argv[1:]
ITERS = int(_args[0]) if _args else 150
POOL_PATH = Path(_args[1]) if len(_args) > 1 else (
    Path(__file__).resolve().parent.parent / "checkpoints" / "opponent_pool.json")

_FIELDS = {f.name for f in dataclasses.fields(BotConfig)}


def load_pool(path: Path) -> tuple[list[BotConfig], dict]:
    data = json.loads(path.read_text())
    pool = [BotConfig(**{k: v for k, v in entry["knobs"].items() if k in _FIELDS})
            for entry in data["pool"]]
    return pool, data["meta"]


def main() -> int:
    pool, meta = load_pool(POOL_PATH)
    device = pick_device()
    print(f"device: {device}")
    print(f"opponent pool: {len(pool)} configs (wr>{meta['wr_threshold']} of "
          f"{meta['n_configs']}, best-of-{meta['best_of']}); top wr {meta['top_win_rate']:.3f}")
    print(f"PPO: clip 0.2, epochs 4, value_baseline={USE_VALUE}")

    cfg = Config(
        rl_algo="ppo",
        ppo_clip=0.2,
        ppo_epochs=4,
        use_value_baseline=USE_VALUE,
        value_coef=0.5,
        rl_lr=6e-5,                 # was 3e-5 — the larger rollout batch tolerates a bigger step
        lr_decay="cosine",          # was "constant" — anneal LR→0 so the policy settles at its peak
        entropy_coef=0.01,          # same as the baseline run
        rl_batch_matches=48,        # was 8 — ~2.5× less advantage noise per update
        rl_minibatch=16,            # was 8 — pure GPU speed knob (grad-accumulated; update unchanged)
        rl_iters=ITERS,
        eval_every=25,
        eval_matches=500,           # was 200 — ±0.022 at p≈0.5, so "best" selection isn't noise
        # distinct file per variant so the two PPO runs (and the 0.545 REINFORCE ckpt) never clobber
        rl_ckpt=("checkpoints/rl_vs_pool_ppo_value.pt" if USE_VALUE
                 else "checkpoints/rl_vs_pool_ppo.pt"),
    )
    reinforce(cfg, device=device, opponent_configs=pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

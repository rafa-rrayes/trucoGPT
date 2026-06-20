"""RL-fine-tune the pretrained TrucoGPT against the strong-config opponent pool.

Loads the pool isolated by `build_opponent_pool.py` (configs whose best-of-3 tournament win-rate
was > 0.5), warm-starts from `checkpoints/pretrain.pt`, and runs REINFORCE where every training
match seats a `ConfigurableBot` team drawn from that pool. Selection metric is the **pool**
win-rate (the actual training distribution); the vs-`default` win-rate is printed alongside for
comparison with prior runs. Best checkpoint → `checkpoints/rl_vs_pool.pt`.

Hyperparameters reuse the sweep's best-known RL settings (constant lr 3e-5, entropy 10× the old
default, a bigger rollout batch — see the ablation-sweep findings).

Run: `uv run python scripts/train_vs_pool.py [ITERS] [POOL.json]`
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

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
POOL_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    Path(__file__).resolve().parent.parent / "checkpoints" / "opponent_pool.json")

# Only the BotConfig fields (drop the JSON bookkeeping keys) when rebuilding configs.
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

    cfg = Config(
        rl_lr=3e-5,                 # sweep's sweet spot — constant
        lr_decay="constant",
        entropy_coef=0.01,          # 10× the old default; lifts the ceiling (ablation finding)
        rl_batch_matches=8,         # bigger rollout batch — also lifts the ceiling
        rl_minibatch=8,
        rl_iters=ITERS,
        eval_every=25,
        eval_matches=200,           # firmer eval (±0.07 at p≈0.5)
        rl_ckpt="checkpoints/rl_vs_pool.pt",
    )
    reinforce(cfg, device=device, opponent_configs=pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

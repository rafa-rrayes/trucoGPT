#!/usr/bin/env bash
# Self-play league fine-tune, then a direct net-vs-net head-to-head against the current champion.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

echo "============================================================"
echo "Self-play league training (300 iters)"
echo "============================================================"
uv run python scripts/train_selfplay.py 300

echo
echo "============================================================"
echo "Head-to-head: rl_selfplay (learner) vs rl_vs_pool_ppo (opponent net), argmax"
echo "============================================================"
uv run python - <<'PY'
import os
from truco.rl.config import Config
from truco.rl.model import load, pick_device
from truco.rl.rollout import rollout_matches

if not os.path.exists("checkpoints/rl_selfplay.pt"):
    raise SystemExit("rl_selfplay.pt was never saved — self-play did not beat the 0.588 "
                     "vs-pool baseline. Champion (rl_vs_pool_ppo.pt) is unchanged.")

device = pick_device()
new, nm = load("checkpoints/rl_selfplay.pt", map_location=device); new.to(device).eval()
old, _ = load("checkpoints/rl_vs_pool_ppo.pt", map_location=device); old.to(device).eval()
cfg = Config()
N = 1000
res = rollout_matches(new, cfg, seeds=range(3_000_000, 3_000_000 + N), learner_seats=(0, 2),
                      sample=False, dealers=[i % 4 for i in range(N)], reward_shaping=False,
                      collect=False, device=device, opponent_net=old)
wins = sum(r.learner_won for r in res)
print(f"rl_selfplay best vs-pool = {nm.get('win_rate')}  (vs default {nm.get('win_rate_vs_default')})")
print(f"head-to-head over {N} matches (argmax): rl_selfplay beats rl_vs_pool_ppo "
      f"{wins}/{N} = {wins / N:.3f}")
PY
echo "============================================================"
echo "SELF-PLAY PIPELINE DONE"
echo "============================================================"

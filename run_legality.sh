#!/usr/bin/env bash
# Legality fine-tune: measure the champion's unmasked illegality, retrain with the legality
# penalty (warm-started from the champion), then measure again and head-to-head vs the champion.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

ITERS="${1:-200}"
COEF="${2:-0.5}"

echo "============================================================"
echo "BEFORE: unmasked legality of the champion (rl_selfplay.pt)"
echo "============================================================"
uv run python scripts/measure_illegal.py checkpoints/rl_selfplay.pt 500

echo
echo "============================================================"
echo "Legality fine-tune ($ITERS iters, legality_coef=$COEF)"
echo "============================================================"
uv run python scripts/train_legality.py "$ITERS" "$COEF"

echo
echo "============================================================"
echo "AFTER: unmasked legality of the retrained model (rl_legality.pt)"
echo "============================================================"
uv run python scripts/measure_illegal.py checkpoints/rl_legality.pt 500

echo
echo "============================================================"
echo "Head-to-head: rl_legality (learner) vs rl_selfplay champion, argmax (masked play)"
echo "============================================================"
uv run python - <<'PY'
import os
from truco.rl.config import Config
from truco.rl.model import load, pick_device
from truco.rl.rollout import rollout_matches

if not os.path.exists("checkpoints/rl_legality.pt"):
    raise SystemExit("rl_legality.pt was never saved — the legality run did not beat the "
                     "vs-pool baseline. Champion (rl_selfplay.pt) is unchanged.")

device = pick_device()
new, nm = load("checkpoints/rl_legality.pt", map_location=device); new.to(device).eval()
old, _ = load("checkpoints/rl_selfplay.pt", map_location=device); old.to(device).eval()
cfg = Config()
N = 1000
res = rollout_matches(new, cfg, seeds=range(3_000_000, 3_000_000 + N), learner_seats=(0, 2),
                      sample=False, dealers=[i % 4 for i in range(N)], reward_shaping=False,
                      collect=False, device=device, opponent_net=old)
wins = sum(r.learner_won for r in res)
print(f"rl_legality best vs-pool = {nm.get('win_rate')}  (vs default {nm.get('win_rate_vs_default')})")
print(f"head-to-head over {N} matches (argmax): rl_legality beats rl_selfplay "
      f"{wins}/{N} = {wins / N:.3f}")
PY
echo "============================================================"
echo "LEGALITY PIPELINE DONE"
echo "============================================================"

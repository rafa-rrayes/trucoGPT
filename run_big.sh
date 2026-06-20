#!/usr/bin/env bash
# BIG run: 10.2M-param model (343/7/7) + a large, diverse, mixed-partner corpus, then the full
# RL chain (pool PPO -> self-play league). Champions from the 250-dim run are preserved as *_250.pt.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

echo "############################################################"
echo "STAGE 1/4 — big diverse corpus (mixed partners, broad config pool)"
echo "############################################################"
# 20k-config tournament -> top 25% (5000 configs) as the style pool; 250k matches, 4 random
# configs per match (varied teammates AND opponents) -> ~200M tokens. Writes checkpoints/corpus.npz.
uv run python -m truco.tournament \
    --configs 20000 --top-frac 0.25 --mixed-partners \
    --corpus-matches 250000 --show 0

echo
echo "--- verifying corpus is large (>100M tokens) ---"
uv run python - <<'PY'
import sys
from truco.rl.config import Config
from truco.rl.data import load_corpus
streams = load_corpus(Config().corpus_path)
tokens = sum(len(s) for s in streams)
print(f"corpus: {len(streams):,} streams, {tokens:,} tokens ({tokens/max(len(streams),1):.0f} avg)")
if tokens < 100_000_000:
    sys.exit(f"ABORT: only {tokens:,} tokens (<100M) — not pretraining.")
print("OK: corpus is large.")
PY

echo
echo "############################################################"
echo "STAGE 2/4 — pretrain the 10.2M-param model (20000 steps, batch 32)"
echo "############################################################"
uv run python -m truco.rl.pretrain

echo
echo "############################################################"
echo "STAGE 3/4 — pool PPO (warm from pretrain) -> rl_vs_pool_ppo.pt"
echo "############################################################"
uv run python scripts/train_vs_pool_ppo.py 300

echo
echo "############################################################"
echo "STAGE 4/4 — self-play league (warm from pool champion) -> rl_selfplay.pt"
echo "############################################################"
uv run python scripts/train_selfplay.py 300

echo
echo "############################################################"
echo "HEAD-TO-HEAD — big champion vs the preserved 250-dim champion (argmax)"
echo "############################################################"
uv run python - <<'PY'
import os
from truco.rl.config import Config
from truco.rl.model import load, pick_device
from truco.rl.rollout import rollout_matches

device = pick_device()
big, bm = load("checkpoints/rl_selfplay.pt", map_location=device); big.to(device).eval()
print(f"big champion best vs-pool = {bm.get('win_rate')}  (vs default {bm.get('win_rate_vs_default')})")
if not os.path.exists("checkpoints/rl_selfplay_250.pt"):
    raise SystemExit("(no 250-dim champion to compare against)")
small, _ = load("checkpoints/rl_selfplay_250.pt", map_location=device); small.to(device).eval()
cfg = Config()  # big arch; block/move_base/action_size identical across archs (shared vocab)
N = 1000
res = rollout_matches(big, cfg, seeds=range(4_000_000, 4_000_000 + N), learner_seats=(0, 2),
                      sample=False, dealers=[i % 4 for i in range(N)], reward_shaping=False,
                      collect=False, device=device, opponent_net=small)
wins = sum(r.learner_won for r in res)
print(f"head-to-head over {N} matches (argmax): BIG (343/7/7) beats SMALL (250/5/5) "
      f"{wins}/{N} = {wins/N:.3f}")
PY
echo "############################################################"
echo "BIG PIPELINE DONE"
echo "############################################################"

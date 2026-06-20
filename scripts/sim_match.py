"""Narrate ONE example match: a trained TrucoGPT checkpoint (seats 0 & 2) vs one config from the
strong-opponent pool (seats 1 & 3).

Reuses the exact engine loop, illegal-token fallback, and seat-0-POV narration from
`sim_pretrain_vs_configs.py` — only the checkpoint and the single opponent change. The opponent is
picked by its rank in `checkpoints/opponent_pool.json` (rank 1 = strongest by tournament win-rate).

Run: `uv run python scripts/sim_match.py [CKPT] [POOL.json] [RANK] [SEED]`
     (defaults: checkpoints/rl_legality.pt, checkpoints/opponent_pool.json, rank 1, seed 2_000_000)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truco.agents import BotConfig
from truco.rl.model import load
from truco.state import team_of

from sim_pretrain_vs_configs import LEARNER, play_and_narrate

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/rl_legality.pt"
POOL = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("checkpoints/opponent_pool.json")
RANK = int(sys.argv[3]) if len(sys.argv) > 3 else 1            # 1 = strongest pool config
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 2_000_001

_FIELDS = {f.name for f in __import__("dataclasses").fields(BotConfig)}


def main() -> int:
    device = "cpu"
    net, meta = load(CKPT, map_location=device)
    net = net.to(device).eval()
    cfg = net.cfg

    data = json.loads(POOL.read_text())
    entry = next(e for e in data["pool"] if e["rank"] == RANK)
    config = BotConfig(**{k: v for k, v in entry["knobs"].items() if k in _FIELDS})
    nz = {k: v for k, v in entry["knobs"].items() if v}

    print("=" * 78)
    print(f"Trained TrucoGPT  ({Path(CKPT).name})   vs   pool opponent rank {RANK}")
    print(f"checkpoint meta: {meta}")
    print(f"opponent: tournament win-rate {entry['win_rate']:.3f} "
          f"({entry['wins']}/{entry['games']} best-of-{data['meta']['best_of']}), "
          f"{'TEXTBOOK' if entry['is_textbook'] else 'tuned'}")
    print(f"opponent nonzero knobs: {nz}")
    print("  Legend:  Us = trained DL model (seats 0 & 2)   Them = pool ConfigurableBot (seats 1 & 3)")
    print('           "You"+"Partner" = the two DL agents · "Opp-1"+"Opp-2" = the two heuristic agents')
    print("           ⚠️ marks a turn where the model fell through an illegal top token.")
    print("=" * 78)

    dealer = SEED % 4
    winner, scores, lines, hands, stats = play_and_narrate(net, cfg, config, SEED, dealer, device)
    print(f"\n  ── Match (seed {SEED}, dealer seat {dealer}) ──")
    print("\n".join(lines))

    dl_won = winner == team_of(LEARNER[0])
    us, them = scores[0], scores[1]
    il = (f"; model used the illegal-token fallback on {stats.illegal_top}/{stats.decisions} moves "
          f"({stats.tokens_skipped} tokens skipped)" if stats.illegal_top else
          f"; all {stats.decisions} model moves were its top token (no illegal fallback)")
    print(f"\n  ── RESULT: {'DL WINS' if dl_won else 'OPPONENT WINS'}  "
          f"(Us {us} – {them} Them, over {hands} hands{il})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

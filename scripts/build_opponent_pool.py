"""Rank 20,000 sampled BotConfigs in a best-of-3 tournament and isolate the winning ones.

Per the request: sample 20k configs, play a best-of-3 (sets_to_win=2) tournament with 20
guaranteed games per config, then keep every config whose tournament win-rate is **> 0.5** —
the above-the-field, genuinely-winning styles. That filtered set becomes the RL opponent pool.

The result is saved to `checkpoints/opponent_pool.json` as a list of full BotConfig knob dicts
(every field, so loading is just `BotConfig(**d)`), plus each config's win-rate / wins / games and
a metadata header. Reproducible: fixed seed, pure-engine matches fanned out across all cores.

Run: `uv run python scripts/build_opponent_pool.py [N_CONFIGS] [GAMES] [SETS] [OUT.json]`
     (defaults: 20000 configs, 20 games, best-of-3 sets, checkpoints/opponent_pool.json)
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truco.tournament import run_tournament
from truco.tournament.configs import KNOB_SPEC

N_CONFIGS = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SETS = int(sys.argv[3]) if len(sys.argv) > 3 else 2          # best-of-(2*SETS-1) = best-of-3
WR_THRESHOLD = 0.5
SEED = 0
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else (
    Path(__file__).resolve().parent.parent / "checkpoints" / "opponent_pool.json")


def _knobs(cfg) -> dict:
    """Every BotConfig field as a plain dict (round-trips via BotConfig(**d))."""
    return dataclasses.asdict(cfg)


def main() -> int:
    t0 = time.time()
    print(f"Ranking {N_CONFIGS} configs · {GAMES} games each · best-of-{2 * SETS - 1} sets · "
          f"seed {SEED}", flush=True)
    result = run_tournament(N_CONFIGS, GAMES, seed=SEED, sets_to_win=SETS, progress=True)

    # Isolate the winning configs: tournament win-rate strictly above 0.5.
    kept_idx = [i for i in result.ranking if result.win_rate[i] > WR_THRESHOLD]
    pool = [{
        "rank": rank + 1,
        "win_rate": result.win_rate[i],
        "wins": result.wins[i],
        "games": result.games[i],
        "is_textbook": (i == 0),
        "knobs": _knobs(result.configs[i]),
    } for rank, i in enumerate(result.ranking) if result.win_rate[i] > WR_THRESHOLD]

    n = len(result.configs)
    textbook_rank = result.ranking.index(0) + 1
    wrs = [result.win_rate[i] for i in result.ranking]
    payload = {
        "meta": {
            "n_configs": n,
            "games_per_config": GAMES,
            "sets_to_win": SETS,
            "best_of": 2 * SETS - 1,
            "seed": SEED,
            "wr_threshold": WR_THRESHOLD,
            "n_kept": len(kept_idx),
            "kept_fraction": len(kept_idx) / n,
            "top_win_rate": wrs[0],
            "median_win_rate": wrs[n // 2],
            "textbook_rank": textbook_rank,
            "textbook_win_rate": result.win_rate[0],
            "knob_fields": list(KNOB_SPEC),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "pool": pool,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"\n=== done in {payload['meta']['elapsed_sec']:.0f}s ===", flush=True)
    print(f"kept {len(kept_idx)}/{n} configs with win-rate > {WR_THRESHOLD} "
          f"({len(kept_idx) / n:.1%} of the field)")
    print(f"top wr {wrs[0]:.3f} · median wr {wrs[n // 2]:.3f} · "
          f"textbook ranked {textbook_rank}/{n} (wr {result.win_rate[0]:.3f})")
    print(f"pool → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

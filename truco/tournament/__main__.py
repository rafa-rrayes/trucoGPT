"""Run the tournament and (re)generate the pretraining corpus from the top configs.

    uv run python -m truco.tournament                 # defaults: rank + write corpus.npz
    uv run python -m truco.tournament --configs 300 --games 60 --topk 20 --corpus-matches 3000
    uv run python -m truco.tournament --rank-only     # just rank configs, don't touch the corpus

The corpus is written to `Config.corpus_path` (checkpoints/corpus.npz) in the format the existing
pretrainer loads, so the only change downstream is *what* the model imitates: strong, varied play.
"""

from __future__ import annotations

import argparse

from ..agents import BotConfig
from ..rl.config import Config
from ..rl.data import save_corpus
from .configs import KNOB_SPEC
from .corpus import generate_diverse_corpus, generate_topk_corpus
from .run import run_tournament

_DEFAULT = BotConfig()


def _describe(cfg: BotConfig) -> str:
    """Show only the knobs that differ from the textbook config (empty = textbook)."""
    diffs = [f"{name}={getattr(cfg, name):g}"
             for name in KNOB_SPEC if getattr(cfg, name) != getattr(_DEFAULT, name)]
    return ", ".join(diffs) if diffs else "(textbook — all rules obeyed)"


def _print_ranking(result, show: int) -> None:
    n = len(result.configs)
    print(f"\n=== ranking ({n} configs) ===")
    for rank, idx in enumerate(result.ranking[:show], 1):
        tag = "  <- textbook" if idx == 0 else ""
        print(f"{rank:>3}. wr={result.win_rate[idx]:.3f} "
              f"({result.wins[idx]}/{result.games[idx]})  [{_describe(result.configs[idx])}]{tag}")
    textbook_rank = result.ranking.index(0) + 1
    print(f"\ntextbook config ranked {textbook_rank}/{n} "
          f"(wr={result.win_rate[0]:.3f}) — configs above it out-play the textbook line.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Config tournament → top-config corpus.")
    ap.add_argument("--configs", type=int, default=200, help="number of configs to sample")
    ap.add_argument("--games", type=int, default=40, help="guaranteed matches per config")
    ap.add_argument("--sets", type=int, default=1, help="sets to win each match (2 = best-of-3)")
    ap.add_argument("--topk", type=int, default=15, help="top configs to draw the corpus from")
    ap.add_argument("--top-frac", type=float, default=None,
                    help="take this fraction of configs instead of --topk (e.g. 0.3 = top 30%%)")
    ap.add_argument("--winners-only", action="store_true",
                    help="keep only the winning team's streams from each corpus match")
    ap.add_argument("--mixed-partners", action="store_true",
                    help="seat 4 independently-sampled configs per match (varied teammates) for a "
                         "more diverse corpus, instead of one config per team")
    ap.add_argument("--corpus-matches", type=int, default=None,
                    help="corpus matches to generate (default: Config.corpus_matches)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None, help="parallel workers (default: all cores)")
    ap.add_argument("--out", type=str, default=None, help="corpus path (default: Config.corpus_path)")
    ap.add_argument("--show", type=int, default=20, help="how many ranked configs to print")
    ap.add_argument("--rank-only", action="store_true", help="rank configs but don't write a corpus")
    args = ap.parse_args()

    cfg = Config()
    result = run_tournament(args.configs, args.games, seed=args.seed, jobs=args.jobs,
                            sets_to_win=args.sets)
    _print_ranking(result, args.show)

    if args.rank_only:
        print("\n--rank-only: skipping corpus generation.")
        return 0

    n_matches = args.corpus_matches if args.corpus_matches is not None else cfg.corpus_matches
    out = args.out or cfg.corpus_path
    topk = max(1, round(args.top_frac * len(result.configs))) if args.top_frac else args.topk
    top = result.top(topk)
    kept = "winning-team streams only" if args.winners_only else "all 4 seats"
    style = "mixed partners (4 configs/match)" if args.mixed_partners else "one config per team"
    print(f"\ngenerating corpus: {n_matches} matches from top-{topk} configs "
          f"({kept}, {style}) → {out}…")
    if args.mixed_partners:
        streams = generate_diverse_corpus(top, n_matches, seed=args.seed,
                                          sets_to_win=args.sets, winners_only=args.winners_only)
    else:
        streams = generate_topk_corpus(top, n_matches, seed=args.seed,
                                       sets_to_win=args.sets, winners_only=args.winners_only)
    save_corpus(streams, out)
    tokens = sum(len(s) for s in streams)
    print(f"wrote {out}: {len(streams)} streams, {tokens} tokens ({tokens / len(streams):.0f} avg).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 3 — the headline metric: win-rate of the deterministic policy vs the `default` heuristic.

A learned policy is only as good as the games it wins. This evaluator plays the net (argmax, no
sampling) as one team against the canonical heuristic bot on the other, over many matches, and
reports the fraction the learner team wins. It is what we quote before vs after RL fine-tuning.

Two fairness details:
  * the dealer rotates each match (`seed % 4`), so neither team gets a fixed positional edge;
  * shaping is irrelevant here — we read only `MatchResult.learner_won` — so we leave the env's
    default reward shaping on and never look at the reward timeline.

Run: `uv run python -m truco.rl.eval [checkpoint]` (defaults to `cfg.rl_ckpt`, falling back to
`cfg.pretrain_ckpt`).
"""

from __future__ import annotations

import os

import torch

from .config import Config
from .model import TrucoGPT
from .rollout import rollout_matches


@torch.no_grad()
def win_rate(net: TrucoGPT, cfg: Config, n_matches: int | None = None, *,
             seed: int = 1_000_000, device: str = "cpu", learner_seats=None,
             opponent_configs=None) -> float:
    """Fraction of `n_matches` the (argmax) learner team wins.

    Against the `default` heuristic by default; against the strong-config pool when
    `opponent_configs` is given (each match draws a config team per its own seed).
    """
    n_matches = n_matches if n_matches is not None else cfg.eval_matches
    learner_seats = tuple(learner_seats) if learner_seats is not None else cfg.learner_seats
    was_training = net.training
    net.eval()
    results = rollout_matches(net, cfg, [seed + i for i in range(n_matches)],
                              learner_seats=learner_seats, sample=False,
                              dealers=[i % 4 for i in range(n_matches)],
                              collect=False, device=device,
                              opponent_configs=opponent_configs)
    wins = sum(int(r.learner_won) for r in results)
    if was_training:
        net.train()
    return wins / max(n_matches, 1)


def _main() -> int:
    import sys

    cfg = Config()
    path = sys.argv[1] if len(sys.argv) > 1 else (
        cfg.rl_ckpt if os.path.exists(cfg.rl_ckpt) else cfg.pretrain_ckpt)
    from .model import load, pick_device

    device = pick_device()
    net, meta = load(path)
    net = net.to(device)
    n = cfg.eval_matches
    print(f"device: {device}; evaluating {path} (meta={meta}) over {n} matches "
          f"vs `default` heuristic…")
    wr = win_rate(net, cfg, n, device=device)
    print(f"win-rate: {wr:.3f} ({int(wr * n)}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

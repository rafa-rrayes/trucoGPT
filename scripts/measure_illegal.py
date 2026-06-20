"""Diagnostic: how illegal is a checkpoint's *unmasked* policy?

Play always stays legal — `action_logits` masks illegal moves to ~-inf, so the model can never
actually commit one. This script asks the counterfactual the legality penalty cares about: at the
states the policy visits, if we lifted the mask, how much probability would the *raw* policy put on
illegal actions, and how often would its top (argmax) move be illegal?

It rolls out matches (sampling, learner vs the `default` heuristic by default), collects the
learner's decision states, then re-reads the raw action-block logits at each and reports:

  * mean illegal mass   — Σ over illegal actions of softmax(raw)  (0 ⇒ the raw policy is fully legal)
  * argmax-illegal rate — fraction of decisions whose unmasked top move is illegal (the rate at
                          which a mask-free deployment would actually play an illegal move)

Run:  uv run python scripts/measure_illegal.py [checkpoint] [n_matches]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truco.rl.config import Config
from truco.rl.model import load, pick_device
from truco.rl.rollout import batch_last_outputs, rollout_matches


@torch.no_grad()
def measure(net, cfg: Config, n_matches: int, device: str, opponent_configs=None) -> dict:
    """Mean unmasked illegal mass and argmax-illegal rate over the policy's own decision states."""
    net.eval()
    results = rollout_matches(net, cfg, [4_000_000 + i for i in range(n_matches)],
                              learner_seats=cfg.learner_seats, sample=True,
                              dealers=[i % 4 for i in range(n_matches)],
                              collect=True, device=device, opponent_configs=opponent_configs)
    decisions = [d for r in results for d in r.decisions]
    if not decisions:
        return {"decisions": 0, "mean_illegal_mass": 0.0, "argmax_illegal_rate": 0.0, "max_illegal_mass": 0.0}

    mass_sum = argmax_illegal = max_mass = 0.0
    bs = 256                                              # cap the forward batch like the rollout does
    for i in range(0, len(decisions), bs):
        chunk = decisions[i:i + bs]
        lm, _ = batch_last_outputs(net, [d.stream for d in chunk], cfg.block_size, device)
        masks = torch.tensor([d.mask for d in chunk], dtype=torch.float32, device=device)
        raw = lm[..., cfg.move_base:cfg.move_base + cfg.action_size]      # unmasked action logits
        probs = torch.softmax(raw, dim=-1)
        illegal = ((1.0 - masks) * probs).sum(dim=-1)                     # [chunk]
        top_illegal = masks.gather(1, raw.argmax(dim=-1, keepdim=True)).squeeze(1) == 0
        mass_sum += float(illegal.sum().item())
        argmax_illegal += float(top_illegal.float().sum().item())
        max_mass = max(max_mass, float(illegal.max().item()))

    n = len(decisions)
    return {"decisions": n, "mean_illegal_mass": mass_sum / n,
            "argmax_illegal_rate": argmax_illegal / n, "max_illegal_mass": max_mass}


def _main() -> int:
    cfg = Config()
    path = sys.argv[1] if len(sys.argv) > 1 else cfg.rl_ckpt
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    device = pick_device()
    net, meta = load(path, map_location=device)
    net = net.to(device)
    print(f"device: {device}; measuring unmasked legality of {path} (meta={meta}) "
          f"over {n} matches vs `default`…")
    stats = measure(net, cfg, n, device)
    print(f"  decisions sampled : {stats['decisions']}")
    print(f"  mean illegal mass : {stats['mean_illegal_mass']:.4f}   (0 ⇒ raw policy fully legal)")
    print(f"  argmax-illegal    : {stats['argmax_illegal_rate']:.4f}   (mask-free play would pick "
          f"illegal at this rate)")
    print(f"  worst-case mass   : {stats['max_illegal_mass']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

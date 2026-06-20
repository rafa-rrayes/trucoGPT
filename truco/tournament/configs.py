"""Sample `BotConfig`s across the knob space with a Latin hypercube.

A full grid over 14 knobs is astronomically large (9·6·9 styles × 11^11 rule-breaks ≈ 10^14), so we
sample instead. Plain uniform sampling wastes draws — many configs end up similar. A *Latin
hypercube* spreads the samples so that, for every single knob, its levels are each hit about
N/|levels| times (a balanced marginal), while the combinations across knobs stay random. That
gives broad, low-variance coverage from only a handful of configs per level.

The grid is fine-grained (0.05/0.1/0.25 steps), so even thousands of sampled configs come out
effectively all-distinct. Rule-break knobs are deliberately capped at 0.5 — a bot that breaks a
rule more than half the time is just noise, not an interestingly-flawed style worth imitating.

`sample_configs(n)` always returns the textbook config (every rule obeyed, no bluff) as index 0,
so the tournament has a fixed reference point to rank everything else against.
"""

from __future__ import annotations

import random

from ..agents import BotConfig

# Discrete levels per knob. Style knobs span timid→reckless / silent→chatty / honest→big-bluffer;
# each rule-break knob runs 0 (always obey) → 0.5 (break half the time). Rule-breaks stay in [0,0.5].
_BREAK_LEVELS = [round(0.05 * i, 2) for i in range(11)]  # 0.0, 0.05, …, 0.50
KNOB_SPEC: dict[str, list[float]] = {
    # --- continuous style ---
    "bluff": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "signal_prob": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "aggression": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
    # --- rule-break probabilities (all capped at 0.5) ---
    "p_lie": _BREAK_LEVELS,
    "p_signal_late": _BREAK_LEVELS,
    "p_signal_partial": _BREAK_LEVELS,
    "p_contradict": _BREAK_LEVELS,
    "p_ignore_partner": _BREAK_LEVELS,
    "p_overtrump": _BREAK_LEVELS,
    "p_burn": _BREAK_LEVELS,
    "p_fold_won": _BREAK_LEVELS,
    "p_overkill": _BREAK_LEVELS,
    "p_bad_facedown": _BREAK_LEVELS,
    "p_bad_eleven": _BREAK_LEVELS,
}


def _lhs_column(n: int, levels: list[float], rng: random.Random) -> list[float]:
    """One Latin-hypercube column: a balanced, shuffled spread of `levels` over `n` samples."""
    strata = list(range(n))
    rng.shuffle(strata)                       # independent permutation per knob
    col = []
    for k in strata:
        u = (k + rng.random()) / n            # jittered centre of stratum k, in [0, 1)
        col.append(levels[min(int(u * len(levels)), len(levels) - 1)])
    return col


def sample_configs(n: int, seed: int = 0) -> list[BotConfig]:
    """`n` Latin-hypercube-sampled configs; index 0 is always the textbook reference config."""
    if n < 1:
        return []
    rng = random.Random(seed)
    n_sampled = n - 1                         # reserve slot 0 for the textbook config
    columns = {name: _lhs_column(n_sampled, levels, rng)
               for name, levels in KNOB_SPEC.items()}
    sampled = [
        BotConfig(**{name: columns[name][i] for name in KNOB_SPEC})
        for i in range(n_sampled)
    ]
    return [BotConfig()] + sampled            # textbook first

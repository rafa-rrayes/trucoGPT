"""A uniform-random legal-move agent — the weakest baseline and a fuzz tester."""

from __future__ import annotations

import random

from ..observation import Observation


class RandomAgent:
    def __init__(self, rng: random.Random | None = None, seed: int | None = None):
        self.rng = rng or random.Random(seed)

    def act(self, obs: Observation):
        return self.rng.choice(list(obs.legal_actions))

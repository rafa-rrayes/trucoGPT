"""TrucoEnv — a thin, PettingZoo-AEC-shaped wrapper over the engine.

One agent (= one seat) acts at a time at `agent_selection`. This is intentionally close
to PettingZoo's Agent-Environment-Cycle API so a future trainer can adapt it (or subclass
into a real `AECEnv`) without touching the engine. It adds nothing to the rules — only
turn iteration, observation plumbing, and a reward signal.

Rewards (per seat, from the last step):
  - terminal: +1 to the winning team's seats, -1 to the losers, on MatchWon.
  - optional shaping (`reward_shaping=True`): each resolved hand pays its point value
    (normalized by the set target) to the winners and the negative to the losers, giving
    a denser learning signal between the sparse match outcomes.
"""

from __future__ import annotations

import random

from . import engine, observation
from .state import SET_TARGET, team_of


class TrucoEnv:
    """Sequential multi-agent Truco environment. Seats 0..3 are the four agents."""

    agents = [0, 1, 2, 3]

    def __init__(self, seed: int | None = None, dealer: int = 0, reward_shaping: bool = False):
        self._seed = seed
        self._dealer = dealer
        self._shaping = reward_shaping
        self.reset(seed)

    # ---------------------------------------------------------------- lifecycle
    def reset(self, seed: int | None = None) -> observation.Observation:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self.state = engine.new_match(self._rng, dealer=self._dealer)
        self.rewards = {s: 0.0 for s in self.agents}            # reward from the last step
        self.cumulative_rewards = {s: 0.0 for s in self.agents}
        return self.observe(self.agent_selection)

    # ---------------------------------------------------------------- accessors
    @property
    def agent_selection(self) -> int | None:
        """The seat that must act now (None only when the match is over)."""
        return self.state.current_actor

    @property
    def done(self) -> bool:
        return self.state.done

    @property
    def terminations(self) -> dict[int, bool]:
        return {s: self.state.done for s in self.agents}

    @property
    def truncations(self) -> dict[int, bool]:
        return {s: False for s in self.agents}

    def observe(self, seat: int) -> observation.Observation:
        return observation.observe(self.state, seat)

    def encoded(self, seat: int) -> list[float]:
        return observation.encode(self.observe(seat))

    def legal_actions(self, seat: int | None = None):
        """Legal actions for the acting seat (others have none); `seat` is accepted for parity."""
        if seat is not None and seat != self.agent_selection:
            return []
        return engine.legal_actions(self.state)

    # ---------------------------------------------------------------- stepping
    def step(self, action):
        """Apply `action` for the current agent; updates rewards. Returns the events list."""
        actor = self.agent_selection
        self.state, events = engine.apply_action(self.state, action, self._rng)
        self._apply_rewards(events)
        return events

    def _apply_rewards(self, events) -> None:
        self.rewards = {s: 0.0 for s in self.agents}
        for ev in events:
            if isinstance(ev, engine.HandResolved) and self._shaping:
                self._pay(ev.winner_team, ev.value / SET_TARGET)
            elif isinstance(ev, engine.MatchWon):
                self._pay(ev.team, 1.0)
        for s in self.agents:
            self.cumulative_rewards[s] += self.rewards[s]

    def _pay(self, team: int, amount: float) -> None:
        for s in self.agents:
            self.rewards[s] += amount if team_of(s) == team else -amount

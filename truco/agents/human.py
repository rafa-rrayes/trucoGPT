"""The interactive agent — the only one that reads from a console.

It presents the observation's legal actions as a numbered menu (in a friendlier order
than the raw engine list: cards first, then stake calls, then signals) and returns the
one the player picks. I/O is injected so the CLI can wire in its own renderer and tests
can drive it without a real terminal.
"""

from __future__ import annotations

from typing import Callable

from ..actions import (
    AcceptTruco,
    CallTruco,
    ElevenChoice,
    PlayCard,
    ReraiseTruco,
    RunTruco,
    Signal,
    describe,
)
from ..cards import card_strength
from ..observation import Observation


def _menu_order(legal, obs: Observation):
    """Order actions for a human: cards (strong->weak, face-up before face-down),
    then stake/eleven decisions, then signals last."""

    def key(a):
        if isinstance(a, PlayCard):
            return (0, a.face_down, -card_strength(a.card, obs.manilha))
        if isinstance(a, (CallTruco, ReraiseTruco, AcceptTruco, RunTruco, ElevenChoice)):
            return (1, 0, 0)
        if isinstance(a, Signal):
            return (2, int(a.code), 0)
        return (3, 0, 0)

    return sorted(legal, key=key)


class HumanAgent:
    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ):
        self._input = input_fn
        self._out = output_fn

    def act(self, obs: Observation):
        options = _menu_order(obs.legal_actions, obs)
        if len(options) == 1:
            self._out(f"  (only option: {describe(options[0])})")
            return options[0]

        self._out("\n  Your move:")
        for i, action in enumerate(options, 1):
            self._out(f"    {i}. {describe(action)}")

        while True:
            raw = self._input("  > ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
            self._out(f"  Please enter a number between 1 and {len(options)}.")

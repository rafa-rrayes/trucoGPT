"""The action space.

Every legal move an agent can make is one of these frozen dataclasses. The engine
exposes the legal subset via `engine.legal_actions(state)`, and a policy/agent picks
one. Keeping actions as small value objects (hashable, comparable) makes them easy to
enumerate, log, encode for RL, and compare in tests.

Action set by phase:
  PLAY           -> PlayCard | Signal | CallTruco
  RESPOND        -> AcceptTruco | RunTruco | ReraiseTruco
  ELEVEN_CHOICE  -> ElevenChoice(play=True/False)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .cards import Card


class SignalCode(IntEnum):
    """The 8-symbol signal vocabulary.

    To the engine these are opaque ids on a *private partner channel*; the names just
    give the CLI human-readable labels mirroring traditional table signals. Agents may
    signal anything (bluffing is allowed — the engine never checks truthfulness).
    """

    WEAK = 0          # "I've got nothing"
    STRONG = 1        # "I'm loaded"
    HAS_ZAP = 2       # clubs manilha
    HAS_HEARTS = 3    # hearts manilha (copas)
    HAS_SPADES = 4    # spades manilha (espadilha)
    HAS_DIAMONDS = 5  # diamonds manilha (ouros)
    HAS_THREE = 6     # holding a 3
    HAS_TWO = 7       # holding a 2


SIGNAL_LABEL: dict[int, str] = {
    SignalCode.WEAK: "I'm weak",
    SignalCode.STRONG: "I'm strong",
    SignalCode.HAS_ZAP: "I have the Zap",
    SignalCode.HAS_HEARTS: "I have the hearts manilha",
    SignalCode.HAS_SPADES: "I have the spades manilha",
    SignalCode.HAS_DIAMONDS: "I have the diamonds manilha",
    SignalCode.HAS_THREE: "I have a 3",
    SignalCode.HAS_TWO: "I have a 2",
}


@dataclass(frozen=True)
class PlayCard:
    """Play a card. `face_down` hides its identity from everyone (illegal in round 1)."""

    card: Card
    face_down: bool = False


@dataclass(frozen=True)
class Signal:
    """Flash a secret signal to your partner. Free action (turn stays with you); once per hand."""

    code: SignalCode


@dataclass(frozen=True)
class CallTruco:
    """Open/raise the stakes one rung up the ladder (Truco -> Six -> Nine -> Twelve)."""


@dataclass(frozen=True)
class AcceptTruco:
    """Accept the outstanding raise; the hand is now worth the called value."""


@dataclass(frozen=True)
class RunTruco:
    """Fold to the outstanding raise; opponents score the previous rung's value."""


@dataclass(frozen=True)
class ReraiseTruco:
    """Throw it back harder — escalate to the next rung instead of accepting."""


@dataclass(frozen=True)
class ElevenChoice:
    """Hand-of-Eleven decision: play the hand (worth 3) or skip it (opponents score 1)."""

    play: bool


# Convenience union for type hints / isinstance tuples.
Action = (
    PlayCard | Signal | CallTruco | AcceptTruco | RunTruco | ReraiseTruco | ElevenChoice
)


def describe(action) -> str:
    """A short human-readable label for an action (used by the CLI and the human agent)."""
    if isinstance(action, PlayCard):
        return f"Play {action.card}" + (" (face-down)" if action.face_down else "")
    if isinstance(action, Signal):
        return f"Signal to partner: \"{SIGNAL_LABEL[action.code]}\""
    if isinstance(action, CallTruco):
        return "Raise the stakes (Truco / Six / Nine / Twelve)"
    if isinstance(action, AcceptTruco):
        return "Accept"
    if isinstance(action, RunTruco):
        return "Run (fold the hand)"
    if isinstance(action, ReraiseTruco):
        return "Re-raise (throw it back harder)"
    if isinstance(action, ElevenChoice):
        return "Play this hand" if action.play else "Skip this hand"
    return repr(action)

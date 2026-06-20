"""Game state.

State is held in plain mutable dataclasses, but `engine.apply_action` deep-copies
before mutating, so transitions are pure at the API boundary (callers' states are never
modified). This keeps the engine functional for self-play while avoiding the verbosity
of rebuilding deeply-nested frozen structures by hand.

Seating (fixed, anti-clockwise turn order 0->1->2->3->0):
    teams are {0, 2} and {1, 3}; your partner sits across the table (seat ^ ... +2),
    your opponents on either side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .cards import Card

# Hand values up the raise ladder: index = accepted stake level.
#   level 0 = no call (1) · Truco (3) · Six (6) · Nine (9) · Twelve (12)
VALUES: tuple[int, ...] = (1, 3, 6, 9, 12)
RAISE_NAMES: tuple[str, ...] = ("none", "Truco", "Six", "Nine", "Twelve")

SET_TARGET = 12      # points to win a set
SETS_TO_WIN = 1      # sets to win the match
ELEVEN = 11          # set score that triggers the Hand of Eleven


def team_of(seat: int) -> int:
    """Team 0 = seats {0, 2}; team 1 = seats {1, 3}."""
    return seat % 2


def partner_of(seat: int) -> int:
    return (seat + 2) % 4


def next_seat(seat: int) -> int:
    """The next player to act, anti-clockwise."""
    return (seat + 1) % 4


def opponents_of(seat: int) -> tuple[int, int]:
    return ((seat + 1) % 4, (seat + 3) % 4)


class Phase(IntEnum):
    ELEVEN_CHOICE = 0   # an at-11 team must choose to play or skip the hand
    PLAY = 1            # waiting for the current player to play/signal/call
    RESPOND = 2         # waiting for a team to answer an outstanding raise
    HAND_DONE = 3       # hand scored; transient (engine deals the next hand)
    MATCH_DONE = 4      # someone won 2 sets


@dataclass
class PlayedCard:
    """One card laid on the table this round. `face_down` hides identity from everyone."""

    seat: int
    card: Card
    face_down: bool = False


@dataclass
class PendingRaise:
    """An outstanding (un-answered) raise on the ladder."""

    proposed_level: int   # the rung being offered (1..4); value = VALUES[proposed_level]
    responder: int        # seat that must answer
    origin: int           # seat whose turn it was when the chain started (plays after accept)
    caller_team: int      # team that made the latest call (winner if the responder runs)


@dataclass
class HandState:
    """Everything about the hand currently in progress."""

    dealer: int
    vira: Card
    manilha: str
    hands: list[list[Card]]                       # remaining cards per seat (4 lists)
    lead: int                                     # who leads the current round
    turn: int                                     # whose turn to play (PLAY phase)
    current_round: int = 0                        # 0-based round index (0,1,2)
    plays: list[PlayedCard] = field(default_factory=list)   # plays in the current round
    round_results: list[int] = field(default_factory=list)  # per finished round: -1 tie / 0 / 1
    stake_level: int = 0                          # accepted rung (index into VALUES)
    last_raise_team: int | None = None            # team that owns the last call (can't re-raise)
    pending: PendingRaise | None = None           # set during RESPOND
    truco_locked: bool = False                    # True during a Hand of Eleven (no raises)
    base_value: int = 1                           # 1 normally; 3 for an Eleven hand played
    # Per seat, the signal codes flashed to one's partner *this hand*, in order. A seat may
    # send several (its whole informative holding) — the engine only forbids exact repeats.
    signals: list[list[int]] = field(default_factory=lambda: [[], [], [], []])

    def value(self) -> int:
        """Points this hand is currently worth if won by normal play."""
        return self.base_value if self.truco_locked else VALUES[self.stake_level]


@dataclass
class MatchState:
    """The whole match: first to `sets_to_win` sets, each a race to 12."""

    dealer: int
    set_scores: list[int] = field(default_factory=lambda: [0, 0])
    match_sets: list[int] = field(default_factory=lambda: [0, 0])
    sets_to_win: int = SETS_TO_WIN                # sets needed to win this match (best-of 2*N-1)
    hand: HandState | None = None
    phase: Phase = Phase.PLAY
    eleven_decider: int | None = None             # seat deciding during ELEVEN_CHOICE
    winner: int | None = None                     # winning team once MATCH_DONE
    hand_number: int = 0                          # hands dealt so far (record/debug)

    @property
    def current_actor(self) -> int | None:
        """Whose turn it is to submit an action, or None if the match is between actions."""
        if self.phase is Phase.PLAY and self.hand is not None:
            return self.hand.turn
        if self.phase is Phase.RESPOND and self.hand is not None and self.hand.pending is not None:
            return self.hand.pending.responder
        if self.phase is Phase.ELEVEN_CHOICE:
            return self.eleven_decider
        return None

    @property
    def done(self) -> bool:
        return self.phase is Phase.MATCH_DONE
"""Per-seat observations — the imperfect-information view a single player legally has.

This is the boundary a policy network (or a human) sees. It deliberately hides what a
seat must not know:
  - opponents' and partner's hands,
  - the identity of any face-down card you did not play yourself,
  - any signal not made by you or your partner (the partner channel is private).

`observe(state, seat)` builds that view; `encode(obs)` flattens it to a fixed-width
list[float] (one-hots for categoricals) so it can feed a network later. The two are kept
separate so the CLI can use the rich `Observation` while a trainer uses `encode`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import Card, Suit
from .state import (
    MatchState,
    Phase,
    VALUES,
    partner_of,
    team_of,
)

# Card-slot codes used by the encoder.
_CARD_HIDDEN = 40   # a face-down card someone else played
_CARD_EMPTY = 41    # an empty slot (card not played / hand position unused)
_CARD_SLOT = 42     # one-hot width for a card slot


@dataclass(frozen=True)
class VisiblePlay:
    """A card on the table as `seat` sees it (card is None when hidden face-down)."""

    seat: int
    card: Card | None
    face_down: bool


@dataclass(frozen=True)
class PendingView:
    proposed_level: int
    responder: int
    caller_team: int


@dataclass(frozen=True)
class Observation:
    """Everything a single seat legally knows right now."""

    seat: int
    team: int
    phase: Phase

    my_hand: tuple[Card, ...]
    vira: Card
    manilha: str

    dealer: int
    lead: int
    current_actor: int | None
    current_round: int
    table: tuple[VisiblePlay, ...]          # plays so far in the current round
    round_results: tuple[int, ...]          # finished rounds: -1 tie / 0 / 1

    set_scores: tuple[int, int]
    match_sets: tuple[int, int]

    stake_level: int
    value: int
    truco_locked: bool
    last_raise_team: int | None
    pending: PendingView | None

    my_signals: tuple[int, ...]             # every code you have flashed this hand, in order
    partner_signals: tuple[int, ...]        # private channel: only your partner's signals
    partner_hand: tuple[Card, ...] | None   # revealed only to the at-11 team during the choice

    legal_actions: tuple                     # non-empty only when it's your turn


def observe(state: MatchState, seat: int) -> Observation:
    """Build `seat`'s view of the match."""
    from . import engine  # local import to avoid a cycle (engine imports nothing here)

    hand = state.hand
    team = team_of(seat)

    if hand is None:  # match finished, no hand on the table
        return Observation(
            seat=seat, team=team, phase=state.phase, my_hand=(), vira=Card("4", Suit.DIAMONDS),
            manilha="4", dealer=state.dealer, lead=0, current_actor=None, current_round=0,
            table=(), round_results=(), set_scores=tuple(state.set_scores),  # type: ignore[arg-type]
            match_sets=tuple(state.match_sets), stake_level=0, value=0,  # type: ignore[arg-type]
            truco_locked=False, last_raise_team=None, pending=None, my_signals=(),
            partner_signals=(), partner_hand=None, legal_actions=(),
        )

    table = tuple(
        VisiblePlay(
            seat=pc.seat,
            card=None if (pc.face_down and pc.seat != seat) else pc.card,
            face_down=pc.face_down,
        )
        for pc in hand.plays
    )

    pending = None
    if hand.pending is not None:
        pending = PendingView(
            hand.pending.proposed_level, hand.pending.responder, hand.pending.caller_team
        )

    # The at-11 team shows each other their cards during the choice.
    partner_hand = None
    if state.phase is Phase.ELEVEN_CHOICE and state.eleven_decider is not None:
        if team_of(state.eleven_decider) == team:
            partner_hand = tuple(hand.hands[partner_of(seat)])

    legal = tuple(engine.legal_actions(state)) if state.current_actor == seat else ()

    return Observation(
        seat=seat,
        team=team,
        phase=state.phase,
        my_hand=tuple(hand.hands[seat]),
        vira=hand.vira,
        manilha=hand.manilha,
        dealer=hand.dealer,
        lead=hand.lead,
        current_actor=state.current_actor,
        current_round=hand.current_round,
        table=table,
        round_results=tuple(hand.round_results),
        set_scores=tuple(state.set_scores),  # type: ignore[arg-type]
        match_sets=tuple(state.match_sets),  # type: ignore[arg-type]
        stake_level=hand.stake_level,
        value=hand.value(),
        truco_locked=hand.truco_locked,
        last_raise_team=hand.last_raise_team,
        pending=pending,
        my_signals=tuple(hand.signals[seat]),
        partner_signals=tuple(hand.signals[partner_of(seat)]),
        partner_hand=partner_hand,
        legal_actions=legal,
    )


def observe_all(state: MatchState) -> dict[int, Observation]:
    return {seat: observe(state, seat) for seat in range(4)}


# ----------------------------------------------------------------------- RL encoding


def _one_hot(value: int, n: int) -> list[float]:
    vec = [0.0] * n
    if 0 <= value < n:
        vec[value] = 1.0
    return vec


def _multi_hot(values, n: int) -> list[float]:
    """A length-`n` 0/1 vector with a bit set for each value in `values` (set semantics)."""
    vec = [0.0] * n
    for v in values:
        if 0 <= v < n:
            vec[v] = 1.0
    return vec


def _card_slot(card: Card | None, hidden: bool = False) -> list[float]:
    """42-way one-hot: 0..39 card id, 40 hidden face-down, 41 empty."""
    if card is None:
        return _one_hot(_CARD_HIDDEN if hidden else _CARD_EMPTY, _CARD_SLOT)
    return _one_hot(card.id, _CARD_SLOT)


def _rel_round_result(result: int, my_team: int) -> int:
    # 0 not-played, 1 my team won, 2 opponents won, 3 tie
    if result == -1:
        return 3
    return 1 if result == my_team else 2


def encode(obs: Observation) -> list[float]:
    """Flatten an observation to a fixed-width feature vector (see ENCODED_SIZE).

    Spatial fields (the table, signals, scores) are encoded *relative to the observing
    seat* — index 0 is always "me", then around the table — so a single policy can play
    any seat without re-learning orientation.
    """
    f: list[float] = []

    # My hand: 3 slots (padded with empty).
    hand = list(obs.my_hand)
    for i in range(3):
        f += _card_slot(hand[i] if i < len(hand) else None)

    # The vira (always known).
    f += _card_slot(obs.vira)

    # Table, relative to me: pos 0 = me, 1 = next, 2 = partner, 3 = previous.
    by_seat = {vp.seat: vp for vp in obs.table}
    for pos in range(4):
        s = (obs.seat + pos) % 4
        vp = by_seat.get(s)
        if vp is None:
            f += _card_slot(None)                       # not yet played -> empty
        elif vp.card is None:
            f += _card_slot(None, hidden=True)          # hidden face-down
        else:
            f += _card_slot(vp.card)

    # Round results, relative to my team (3 rounds).
    for i in range(3):
        r = obs.round_results[i] if i < len(obs.round_results) else 0
        coded = _rel_round_result(r, obs.team) if i < len(obs.round_results) else 0
        f += _one_hot(coded, 4)

    # Scores (mine, opponents), normalized.
    f.append(obs.set_scores[obs.team] / 12.0)
    f.append(obs.set_scores[1 - obs.team] / 12.0)
    f.append(obs.match_sets[obs.team] / 2.0)
    f.append(obs.match_sets[1 - obs.team] / 2.0)

    # Stake / value / phase / lock.
    f += _one_hot(obs.stake_level, len(VALUES))
    f.append(obs.value / 12.0)
    f += _one_hot(int(obs.phase), len(Phase))
    f.append(1.0 if obs.truco_locked else 0.0)

    # Pending raise (proposed level one-hot; responder relative position; caller is my team?).
    f += _one_hot(obs.pending.proposed_level if obs.pending else 0, len(VALUES))
    resp_rel = ((obs.pending.responder - obs.seat) % 4) if obs.pending else 0
    f += _one_hot(resp_rel, 4)
    f.append(1.0 if (obs.pending and obs.pending.caller_team == obs.team) else 0.0)

    # Signals (mine, partner's) — 8-way multi-hot: one bit per code flashed this hand
    # (a seat may now send several), all-zero = silent.
    f += _multi_hot(obs.my_signals, 8)
    f += _multi_hot(obs.partner_signals, 8)

    # Partner hand revealed during the Eleven choice (3 slots, empty otherwise).
    ph = list(obs.partner_hand) if obs.partner_hand else []
    for i in range(3):
        f += _card_slot(ph[i] if i < len(ph) else None)

    # Is it my turn?
    f.append(1.0 if obs.current_actor == obs.seat else 0.0)

    return f


# Computed once so callers (and a test) can assert the schema width is stable.
ENCODED_SIZE = (
    3 * _CARD_SLOT          # my hand
    + _CARD_SLOT            # vira
    + 4 * _CARD_SLOT        # table
    + 3 * 4                 # round results
    + 4                     # scores
    + len(VALUES)           # stake one-hot
    + 1                     # value
    + len(Phase)            # phase one-hot
    + 1                     # truco_locked
    + len(VALUES)           # pending proposed level
    + 4                     # pending responder rel
    + 1                     # pending caller is my team
    + 8 + 8                 # signals (mine, partner) — multi-hot
    + 3 * _CARD_SLOT        # partner hand (eleven)
    + 1                     # my turn
)
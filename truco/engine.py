"""The rules engine: legal actions + pure state transitions.

Public API:
    new_match(rng, dealer=0) -> MatchState
    legal_actions(state)     -> list[Action]
    apply_action(state, action, rng) -> (MatchState, list[Event])

`apply_action` deep-copies the input state, so it never mutates the caller's state, and
fully advances to the next *actionable* state (dealing the next hand after a hand ends,
etc.). All randomness comes from the injected `rng`, so a fixed seed + fixed action
sequence reproduces a match exactly — a hard requirement for RL self-play.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from .actions import (
    AcceptTruco,
    CallTruco,
    ElevenChoice,
    PlayCard,
    ReraiseTruco,
    RunTruco,
    Signal,
    SignalCode,
)
from .cards import Card, FACE_DOWN_STRENGTH, card_strength, manilha_rank, shuffled_deck
from .state import (
    ELEVEN,
    SET_TARGET,
    SETS_TO_WIN,
    VALUES,
    HandState,
    MatchState,
    PendingRaise,
    Phase,
    PlayedCard,
    next_seat,
    team_of,
)

MAX_LEVEL = len(VALUES) - 1  # 4 == "Twelve"; ladder can't climb past this
REDEAL = -2                  # _decide_hand sentinel: all rounds tied, wash and re-deal


# --------------------------------------------------------------------------- events
# Lightweight records of what happened in a transition. The CLI narrates them and a
# trainer can shape rewards from them; the engine itself never needs to read them back.


@dataclass(frozen=True)
class Event:
    pass


@dataclass(frozen=True)
class HandDealt(Event):
    dealer: int
    vira: Card
    manilha: str
    hand_number: int


@dataclass(frozen=True)
class ElevenStarted(Event):
    team: int
    decider: int


@dataclass(frozen=True)
class ElevenDecided(Event):
    team: int
    play: bool


@dataclass(frozen=True)
class SignalSent(Event):
    seat: int
    code: int


@dataclass(frozen=True)
class CardPlayed(Event):
    seat: int
    card: Card
    face_down: bool


@dataclass(frozen=True)
class TrucoCalled(Event):
    seat: int
    proposed_level: int


@dataclass(frozen=True)
class TrucoAnswered(Event):
    seat: int
    response: str  # "accept" | "run" | "raise"
    level: int


@dataclass(frozen=True)
class RoundResolved(Event):
    round_index: int
    result_team: int       # -1 for a tie
    winning_seat: int | None


@dataclass(frozen=True)
class HandResolved(Event):
    winner_team: int
    value: int
    reason: str  # "rounds" | "run" | "eleven_skip"


@dataclass(frozen=True)
class Redeal(Event):
    pass


@dataclass(frozen=True)
class SetWon(Event):
    team: int


@dataclass(frozen=True)
class MatchWon(Event):
    team: int


# ----------------------------------------------------------------------- match setup


def new_match(rng: random.Random, dealer: int = 0, sets_to_win: int = SETS_TO_WIN) -> MatchState:
    """Create a fresh match and deal the first hand. `sets_to_win` makes it best-of-(2N-1)."""
    state = MatchState(dealer=dealer, sets_to_win=sets_to_win)
    _deal_hand(state, rng, [])
    return state


def _deal_hand(state: MatchState, rng: random.Random, events: list[Event]) -> None:
    """Deal a new hand into `state`, flip the vira, and set the opening phase."""
    state.hand_number += 1
    dealer = state.dealer
    deck = shuffled_deck(rng)
    hands = [deck[3 * s : 3 * s + 3] for s in range(4)]
    vira = deck[12]
    lead = next_seat(dealer)  # player to the dealer's right leads

    hand = HandState(
        dealer=dealer,
        vira=vira,
        manilha=manilha_rank(vira),
        hands=hands,
        lead=lead,
        turn=lead,
    )
    state.hand = hand
    state.eleven_decider = None
    events.append(HandDealt(dealer, vira, hand.manilha, state.hand_number))

    team0_at_11 = state.set_scores[0] == ELEVEN
    team1_at_11 = state.set_scores[1] == ELEVEN

    if team0_at_11 and team1_at_11:
        # Both at 11 ("hand of iron"): forced play at value 1, no raises (doc-silent default).
        hand.truco_locked = True
        hand.base_value = 1
        state.phase = Phase.PLAY
    elif team0_at_11 or team1_at_11:
        team = 0 if team0_at_11 else 1
        hand.truco_locked = True
        hand.base_value = 3  # if played, an Eleven hand is worth 3
        decider = lead if team_of(lead) == team else next_seat(lead)
        state.eleven_decider = decider
        state.phase = Phase.ELEVEN_CHOICE
        events.append(ElevenStarted(team, decider))
    else:
        state.phase = Phase.PLAY


# -------------------------------------------------------------------- round/hand math


def _effective_strength(pc: PlayedCard, manilha: str) -> int:
    """A face-down card can never win; otherwise its normal manilha-aware strength."""
    return FACE_DOWN_STRENGTH if pc.face_down else card_strength(pc.card, manilha)


def _resolve_round(hand: HandState) -> tuple[int, int | None]:
    """Resolve the 4 cards on the table -> (team, winning_seat), or (-1, None) for a tie.

    A tie happens only when the top strength is shared across *both* teams (two equal
    normal cards from opponents, or an all-face-down round). Manilha strengths are
    unique by suit, so manilha-vs-manilha never ties.
    """
    strengths = [(pc.seat, _effective_strength(pc, hand.manilha)) for pc in hand.plays]
    top = max(s for _, s in strengths)
    top_seats = [seat for seat, s in strengths if s == top]
    teams_at_top = {team_of(seat) for seat in top_seats}
    if len(teams_at_top) > 1:
        return -1, None
    winning_seat = top_seats[0]  # earliest player at the top strength leads next
    return team_of(winning_seat), winning_seat


def _decide_hand(results: list[int]) -> int | None:
    """Best-of-3 with the §9 tie rules. Returns winning team, REDEAL, or None (play on).

    results entries: -1 tie / 0 team0 / 1 team1.
    """
    n = len(results)
    if n < 2:
        return None
    r1, r2 = results[0], results[1]
    if n == 2:
        if r1 == -1 and r2 == -1:
            return None                     # both tied -> need round 3
        if r1 == -1:
            return r2                        # R1 tie -> R2 winner takes it
        if r2 == -1:
            return r1                        # R2 tie -> R1 winner takes it
        return r1 if r1 == r2 else None      # 2-0 decides; 1-1 -> round 3
    # n == 3
    r3 = results[2]
    if r3 != -1:
        return r3                            # round 3 has a winner
    if r1 != -1:
        return r1                            # R3 tie -> R1 winner takes it
    return REDEAL                            # all three tied -> wash and re-deal


# ------------------------------------------------------------------------ legal moves


def _can_call_truco(hand: HandState, actor: int) -> bool:
    if hand.truco_locked or hand.stake_level >= MAX_LEVEL:
        return False
    # The team holding the last call can't raise again until the other team does.
    return hand.last_raise_team is None or hand.last_raise_team != team_of(actor)


def legal_actions(state: MatchState):
    """The exact set of legal actions for whoever must act now (the action mask)."""
    if state.phase is Phase.MATCH_DONE or state.hand is None:
        return []
    if state.phase is Phase.ELEVEN_CHOICE:
        return [ElevenChoice(True), ElevenChoice(False)]

    hand = state.hand
    if state.phase is Phase.RESPOND:
        assert hand.pending is not None
        acts: list = [AcceptTruco(), RunTruco()]
        if hand.pending.proposed_level < MAX_LEVEL:
            acts.append(ReraiseTruco())
        return acts

    # Phase.PLAY
    actor = hand.turn
    acts = []
    sent = set(hand.signals[actor])  # a seat may keep signalling, but never repeat a code
    acts.extend(Signal(code) for code in SignalCode if int(code) not in sent)
    for card in hand.hands[actor]:
        acts.append(PlayCard(card, face_down=False))
        if hand.current_round > 0:  # face-down is illegal in round 1
            acts.append(PlayCard(card, face_down=True))
    if _can_call_truco(hand, actor):
        acts.append(CallTruco())
    return acts


# ----------------------------------------------------------------------- transitions


def apply_action(state: MatchState, action, rng: random.Random):
    """Apply `action`, returning a fresh (state, events). Raises ValueError if illegal."""
    if action not in legal_actions(state):
        raise ValueError(f"illegal action {action!r} in phase {state.phase.name}")

    state = copy.deepcopy(state)
    events: list[Event] = []
    hand = state.hand
    assert hand is not None

    if state.phase is Phase.ELEVEN_CHOICE:
        _apply_eleven_choice(state, action, events)
    elif state.phase is Phase.RESPOND:
        _apply_respond(state, action, events)
    else:  # Phase.PLAY
        _apply_play(state, action, rng, events)

    # A hand that just ended (run / skip / scored rounds) deals the next one.
    if state.phase is Phase.HAND_DONE:
        _advance_after_hand(state, rng, events)

    return state, events


def _apply_eleven_choice(state, action: ElevenChoice, events) -> None:
    team = team_of(state.eleven_decider)
    events.append(ElevenDecided(team, action.play))
    state.eleven_decider = None
    if action.play:
        state.phase = Phase.PLAY  # base_value=3 / truco_locked already set at deal
    else:
        _score_hand(state, 1 - team, 1, "eleven_skip", events)


def _apply_respond(state, action, events) -> None:
    hand = state.hand
    pend = hand.pending
    assert pend is not None
    if isinstance(action, AcceptTruco):
        hand.stake_level = pend.proposed_level
        hand.last_raise_team = pend.caller_team
        hand.turn = pend.origin
        hand.pending = None
        state.phase = Phase.PLAY
        events.append(TrucoAnswered(pend.responder, "accept", pend.proposed_level))
    elif isinstance(action, RunTruco):
        # Running concedes at the previous rung's value to the team that called.
        value = VALUES[pend.proposed_level - 1]
        events.append(TrucoAnswered(pend.responder, "run", pend.proposed_level))
        hand.pending = None
        _score_hand(state, pend.caller_team, value, "run", events)
    else:  # ReraiseTruco
        events.append(TrucoCalled(pend.responder, pend.proposed_level + 1))
        pend.caller_team = team_of(pend.responder)
        pend.responder = next_seat(pend.responder)
        pend.proposed_level += 1
        # phase stays RESPOND; the other team now answers


def _apply_play(state, action, rng, events) -> None:
    hand = state.hand
    actor = hand.turn

    if isinstance(action, Signal):
        hand.signals[actor].append(int(action.code))
        events.append(SignalSent(actor, int(action.code)))
        return  # free action: same player still acts

    if isinstance(action, CallTruco):
        proposed = hand.stake_level + 1
        hand.pending = PendingRaise(
            proposed_level=proposed,
            responder=next_seat(actor),
            origin=actor,
            caller_team=team_of(actor),
        )
        state.phase = Phase.RESPOND
        events.append(TrucoCalled(actor, proposed))
        return

    # PlayCard
    hand.hands[actor].remove(action.card)
    hand.plays.append(PlayedCard(actor, action.card, action.face_down))
    events.append(CardPlayed(actor, action.card, action.face_down))

    if len(hand.plays) == 4:
        _after_round(state, rng, events)
    else:
        hand.turn = next_seat(actor)


def _after_round(state, rng, events) -> None:
    hand = state.hand
    result, win_seat = _resolve_round(hand)
    hand.round_results.append(result)
    events.append(RoundResolved(hand.current_round, result, win_seat))

    decision = _decide_hand(hand.round_results)
    if decision is None:
        # Start the next round. A tie keeps the same leader; else the winner leads.
        hand.current_round += 1
        hand.plays = []
        hand.lead = win_seat if result != -1 else hand.lead
        hand.turn = hand.lead
    elif decision == REDEAL:
        # All three rounds tied: no score, same dealer re-deals immediately.
        events.append(Redeal())
        _deal_hand(state, rng, events)
    else:
        _score_hand(state, decision, hand.value(), "rounds", events)


def _score_hand(state, winner_team: int, value: int, reason: str, events) -> None:
    """Credit a hand's points and resolve set/match completion."""
    state.set_scores[winner_team] += value
    events.append(HandResolved(winner_team, value, reason))

    if state.set_scores[winner_team] >= SET_TARGET:
        state.match_sets[winner_team] += 1
        events.append(SetWon(winner_team))
        state.set_scores = [0, 0]
        if state.match_sets[winner_team] >= state.sets_to_win:
            state.winner = winner_team
            state.phase = Phase.MATCH_DONE
            events.append(MatchWon(winner_team))
            return
    state.phase = Phase.HAND_DONE


def _advance_after_hand(state, rng, events) -> None:
    """After a scored hand, rotate the dealer and deal the next one."""
    state.dealer = next_seat(state.dealer)
    _deal_hand(state, rng, events)

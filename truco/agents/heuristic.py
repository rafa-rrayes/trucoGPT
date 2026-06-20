"""A rule-based bot: the CLI opponents and the RL baseline to beat.

It is intentionally simple and readable, not optimal. It estimates its hand strength
from the (manilha-aware) card values, plays the cheapest card that does the job, raises
and answers the stakes on strong hands with a small seeded bluff rate, and flashes an
honest-ish signal to its partner now and then. Every decision is made over the
observation's `legal_actions`, so it can only ever return a legal move — the same
guarantee a future policy network must honour.
"""

from __future__ import annotations

import random

from ..actions import (
    AcceptTruco,
    CallTruco,
    ElevenChoice,
    PlayCard,
    ReraiseTruco,
    RunTruco,
    Signal,
    SignalCode,
)
from ..cards import MANILHA_BASE, Suit, card_strength, is_manilha
from ..observation import Observation
from ..state import Phase, partner_of, team_of


def _card_score(card, manilha: str) -> float:
    """A card's standalone strength on a 0..1 scale (manilhas land near the top)."""
    s = card_strength(card, manilha)
    if s >= MANILHA_BASE:                 # 100..103 -> 0.86..0.95 (zap strongest)
        return 0.86 + 0.03 * (s - MANILHA_BASE)
    return (s / 9.0) * 0.78               # normal 0..9 -> 0..0.78


def _hand_score(cards, manilha: str) -> float:
    """Mean card score across a set of cards (0 for an empty set)."""
    if not cards:
        return 0.0
    return sum(_card_score(c, manilha) for c in cards) / len(cards)


def hand_strength(obs: Observation) -> float:
    """How good `obs.my_hand` looks right now, 0..1."""
    return _hand_score(obs.my_hand, obs.manilha)


# Map a manilha's suit to the signal that announces it.
_SUIT_SIGNAL = {
    Suit.CLUBS: SignalCode.HAS_ZAP,
    Suit.HEARTS: SignalCode.HAS_HEARTS,
    Suit.SPADES: SignalCode.HAS_SPADES,
    Suit.DIAMONDS: SignalCode.HAS_DIAMONDS,
}


class HeuristicAgent:
    def __init__(
        self,
        rng: random.Random | None = None,
        seed: int | None = None,
        bluff: float = 0.12,
        signal_prob: float = 0.30,
        aggression: float = 0.0,
    ):
        self.rng = rng or random.Random(seed)
        self.bluff = bluff
        self.signal_prob = signal_prob
        self.aggression = aggression
        # Positive aggression lowers every call/accept/reraise gate (riskier); 0.0 == default.
        self._risk_shift = 0.15 * aggression

    # ----------------------------------------------------------------- public API
    def act(self, obs: Observation):
        if obs.phase is Phase.ELEVEN_CHOICE:
            return self._eleven(obs)
        if obs.phase is Phase.RESPOND:
            return self._respond(obs)
        return self._play(obs)

    # ------------------------------------------------------------- PLAY decisions
    def _play(self, obs: Observation):
        legal = obs.legal_actions

        # 1. Maybe flash a signal first (a free action; we'll be asked again to play).
        #    The canonical bot sends at most one signal per hand — `not obs.my_signals`
        #    keeps it byte-identical now that the engine permits several (see ConfigurableBot).
        signals = [a for a in legal if isinstance(a, Signal)]
        if signals and not obs.my_signals and self.rng.random() < self.signal_prob:
            return self._pick_signal(obs, signals)

        # 2. Maybe raise the stakes.
        call = next((a for a in legal if isinstance(a, CallTruco)), None)
        if call is not None and self._should_call(obs):
            return call

        # 3. Otherwise commit a card.
        return self._choose_card(obs, legal)

    def _choose_card(self, obs: Observation, legal):
        faceup = sorted(
            (a for a in legal if isinstance(a, PlayCard) and not a.face_down),
            key=lambda a: card_strength(a.card, obs.manilha),
        )
        facedown = [a for a in legal if isinstance(a, PlayCard) and a.face_down]
        partner_winning, opp_best, on_table = _table_status(obs)

        if on_table == 0:                                # we lead the round
            return faceup[len(faceup) // 2]              # lead a middling card
        if partner_winning:                              # partner has it won — save power
            return self._dump(faceup, facedown)
        winners = [a for a in faceup if card_strength(a.card, obs.manilha) > opp_best]
        if winners:                                      # win as cheaply as possible
            return min(winners, key=lambda a: card_strength(a.card, obs.manilha))
        return self._dump(faceup, facedown)              # can't win — throw the trash

    def _dump(self, faceup, facedown):
        """Discard the weakest card, face-down (to hide it) when that's allowed."""
        weakest = faceup[0]
        for a in facedown:
            if a.card == weakest.card:
                return a
        return weakest

    def _pick_signal(self, obs: Observation, signals):
        codes = {SignalCode(a.code): a for a in signals}
        manilhas = [c for c in obs.my_hand if is_manilha(c, obs.manilha)]
        if manilhas:
            best = max(manilhas, key=lambda c: int(c.suit))
            code = _SUIT_SIGNAL[best.suit]
            if code in codes:
                return codes[code]
        if any(c.rank == "3" for c in obs.my_hand) and SignalCode.HAS_THREE in codes:
            return codes[SignalCode.HAS_THREE]
        if any(c.rank == "2" for c in obs.my_hand) and SignalCode.HAS_TWO in codes:
            return codes[SignalCode.HAS_TWO]
        code = SignalCode.STRONG if hand_strength(obs) > 0.55 else SignalCode.WEAK
        if self.rng.random() < self.bluff:              # occasionally lie about it
            code = SignalCode.WEAK if code is SignalCode.STRONG else SignalCode.STRONG
        return codes.get(code, signals[0])

    def _should_call(self, obs: Observation) -> bool:
        s = hand_strength(obs)
        won_round = any(r == obs.team for r in obs.round_results)
        d = self._risk_shift
        prob = 0.6 if s > 0.72 - d else 0.25 if s > 0.55 - d else 0.0
        if won_round:
            prob += 0.2
        if s < 0.40:
            prob += self.bluff                          # rare stone-cold bluff
        return self.rng.random() < prob

    # ---------------------------------------------------------- RESPOND decisions
    def _respond(self, obs: Observation):
        legal = obs.legal_actions
        s = hand_strength(obs)
        if any(r == obs.team for r in obs.round_results):
            s += 0.10
        d = self._risk_shift
        reraise = next((a for a in legal if isinstance(a, ReraiseTruco)), None)
        if reraise is not None and (s > 0.80 - d or (s > 0.55 - d and self.rng.random() < self.bluff)):
            return reraise
        if s > 0.42 - d or self.rng.random() < self.bluff:
            return next(a for a in legal if isinstance(a, AcceptTruco))
        return next(a for a in legal if isinstance(a, RunTruco))

    # ----------------------------------------------- ELEVEN_CHOICE decision (3 pts)
    def _eleven(self, obs: Observation):
        combined = list(obs.my_hand) + list(obs.partner_hand or ())
        play = _hand_score(combined, obs.manilha) > 0.45 - self._risk_shift  # risk 3 vs. concede 1
        return next(a for a in obs.legal_actions if isinstance(a, ElevenChoice) and a.play is play)


def _table_status(obs: Observation) -> tuple[bool, int, int]:
    """Read the current round's table -> (partner_is_winning, best_opponent_strength, cards_played)."""
    me_team = obs.team
    partner = partner_of(obs.seat)
    best_seat, best_str, opp_best = None, -2, -2
    for vp in obs.table:
        # A face-down or hidden card can never win, so it counts as the lowest strength.
        st = -1 if (vp.face_down or vp.card is None) else card_strength(vp.card, obs.manilha)
        if team_of(vp.seat) != me_team:
            opp_best = max(opp_best, st)
        if st > best_str:
            best_str, best_seat = st, vp.seat
    partner_winning = best_seat == partner and best_str > opp_best
    return partner_winning, opp_best, len(obs.table)

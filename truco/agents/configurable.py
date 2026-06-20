"""A fully-tunable Truco bot: every rule of good play is a break-probability knob.

`HeuristicAgent` bakes in one fixed style. `ConfigurableBot` instead encodes the *canonical
good-play policy* — honest up-front signalling, partner-coordinated betting, cheap wins, safe
discards, correct Hand-of-Eleven math — and exposes each of those rules as a probability of
*breaking* it. At p_break = 0 the bot plays the textbook line; at p_break = 1 it always does the
wrong thing. Sweeping these knobs spans a huge space of styles whose quality is then decided not
by our assumptions but by a win-rate tournament (see `truco.tournament`).

The rules, each the zero-point of a break-knob:

  Signalling   R1 truthful · R2 up-front · R3 whole holding · R4 no self-contradiction
  Coordination R5 bet on partner's señal · R6 don't overtrump a winning partner
               R7 concede cheaply rather than burn a usable card
  Cards/bets   R8 don't fold a won position · R9 win as cheaply as possible
               R10 discard face-down only when it can't cost the round · R11 respect Eleven math

Bluffing (calling/accepting on garbage) and chattiness/aggression stay continuous *style* knobs,
not rule breaks — bluffing is good Truco, so it is swept 0 → high rather than penalised.

Every decision is taken over `obs.legal_actions`, so the bot can only ever return a legal move.
Signalling choices are drawn from a per-hand RNG (`_hand_rng`) so they stay consistent across the
several `act` calls a single free-signal-then-play turn produces; in-the-moment break rolls use
the instance RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

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
from ..cards import card_strength, is_manilha
from ..observation import Observation
from ..state import Phase
from .heuristic import _SUIT_SIGNAL, _hand_score, _table_status, hand_strength

# Codes that tell the partner "I'm loaded" (everything except an explicit WEAK).
_POSITIVE_CODES = frozenset(
    int(c) for c in SignalCode if c is not SignalCode.WEAK
)


@dataclass(frozen=True)
class BotConfig:
    """All knobs. Defaults = the textbook good-play bot (every rule obeyed, no bluffing)."""

    # --- continuous style (not rule breaks) ---
    bluff: float = 0.0          # call/accept/reraise on garbage; soul of Truco, swept 0→high
    signal_prob: float = 1.0    # chattiness: chance the bot bothers to signal at all this hand
    aggression: float = 0.0     # shifts every call/accept/reraise gate (−1 timid … +1 reckless)

    # --- rule-break probabilities (0 = always obey the good rule) ---
    p_lie: float = 0.0          # R1: signal the opposite of the truth (sandbag / bluff-signal)
    p_signal_late: float = 0.0  # R2: don't signal up-front (defer past round 0)
    p_signal_partial: float = 0.0  # R3: reveal only the single strongest holding, not all of it
    p_contradict: float = 0.0   # R4: flash both STRONG and WEAK (self-contradiction)
    p_ignore_partner: float = 0.0  # R5: don't factor the partner's señal into betting
    p_overtrump: float = 0.0    # R6: waste a winner over a partner who already has the round
    p_burn: float = 0.0         # R7: throw a usable card when you can't win, instead of conceding
    p_fold_won: float = 0.0     # R8: run from a position you've effectively won
    p_overkill: float = 0.0     # R9: win with a needlessly strong card
    p_bad_facedown: float = 0.0  # R10: misuse face-down (hide a winner / reveal a discard)
    p_bad_eleven: float = 0.0   # R11: make the wrong Hand-of-Eleven call


class ConfigurableBot:
    """The canonical good-play bot, each rule breakable with its configured probability."""

    def __init__(self, config: BotConfig, rng: random.Random | None = None,
                 seed: int | None = None):
        self.cfg = config
        self.rng = rng or random.Random(seed)
        self._risk_shift = 0.15 * config.aggression  # +aggression lowers every betting gate
        self._salt = self.rng.randrange(1 << 30)      # stable per-instance hand-RNG salt

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
        sig = self._maybe_signal(obs, legal)
        if sig is not None:
            return sig
        call = next((a for a in legal if isinstance(a, CallTruco)), None)
        if call is not None and self._should_call(obs):
            return call
        return self._choose_card(obs, legal)

    # ----------------------------------------------------------- signalling (R1–R4)
    def _hand_rng(self, obs: Observation) -> random.Random:
        """A per-hand RNG: stable across the multiple `act` calls one signalling turn makes."""
        key = (tuple(c.id for c in obs.my_hand), obs.manilha, obs.dealer,
               obs.set_scores, obs.match_sets)
        return random.Random(hash(key) ^ self._salt)

    def _honest_codes(self, obs: Observation) -> list[int]:
        """The full informative holding (R3), strongest first: manilhas, 3s, 2s, then a summary."""
        codes: list[int] = []
        manilhas = [c for c in obs.my_hand if is_manilha(c, obs.manilha)]
        for c in sorted(manilhas, key=lambda c: int(c.suit), reverse=True):
            codes.append(int(_SUIT_SIGNAL[c.suit]))
        if any(c.rank == "3" for c in obs.my_hand):
            codes.append(int(SignalCode.HAS_THREE))
        if any(c.rank == "2" for c in obs.my_hand):
            codes.append(int(SignalCode.HAS_TWO))
        codes.append(int(SignalCode.STRONG if hand_strength(obs) > 0.55 else SignalCode.WEAK))
        return codes

    def _signal_decision(self, obs: Observation) -> tuple[list[int], bool]:
        """Decide this hand's signal plan + whether to signal late. Deterministic per hand.

        All five rolls are drawn unconditionally so the RNG sequence is identical on every
        `act` call within the hand, making the plan stable as signals are sent one per call.
        """
        hr = self._hand_rng(obs)
        r_chat, r_partial, r_lie, r_contra, r_late = (hr.random() for _ in range(5))
        if r_chat >= self.cfg.signal_prob:
            return [], False  # silent this hand

        strong, weak = int(SignalCode.STRONG), int(SignalCode.WEAK)
        truth_strong = hand_strength(obs) > 0.55
        codes = self._honest_codes(obs)
        if r_partial < self.cfg.p_signal_partial:      # R3 break: only the strongest tell
            codes = codes[:1]
        if r_lie < self.cfg.p_lie:                      # R1 break: misrepresent strength
            codes = [weak if truth_strong else strong]
        if r_contra < self.cfg.p_contradict:           # R4 break: flash both STRONG and WEAK
            codes = [c for c in codes if c not in (strong, weak)] + [strong, weak]
        return codes, (r_late < self.cfg.p_signal_late)

    def _maybe_signal(self, obs: Observation, legal):
        """Return the next signal to send this turn, or None to move on to betting/playing."""
        sig_actions = {int(a.code): a for a in legal if isinstance(a, Signal)}
        if not sig_actions:
            return None
        codes, late = self._signal_decision(obs)
        already = set(obs.my_signals)
        remaining = [c for c in codes if c not in already and c in sig_actions]
        if not remaining:
            return None
        if late:                                        # R2 break: stay silent in round 0
            if obs.current_round == 0:
                return None
        elif obs.current_round != 0 and not obs.my_signals:
            return None                                 # up-front: never *start* after round 0
        return sig_actions[remaining[0]]

    # ----------------------------------------------------------- card choice (R6,R7,R9,R10)
    def _choose_card(self, obs: Observation, legal):
        faceup = sorted(
            (a for a in legal if isinstance(a, PlayCard) and not a.face_down),
            key=lambda a: card_strength(a.card, obs.manilha),
        )
        facedown = [a for a in legal if isinstance(a, PlayCard) and a.face_down]
        partner_winning, opp_best, on_table = _table_status(obs)

        if on_table == 0:                               # we lead the round
            return faceup[len(faceup) // 2]             # lead a middling card

        if partner_winning:                             # R6: don't overtrump a winning partner
            if self.rng.random() < self.cfg.p_overtrump:
                winners = self._winners(faceup, opp_best, obs)
                if winners:
                    cheapest = min(winners, key=lambda a: card_strength(a.card, obs.manilha))
                    return self._maybe_facedown(cheapest, facedown, safe=False)
            return self._dump(obs, faceup, facedown)

        winners = self._winners(faceup, opp_best, obs)
        if winners:                                     # R9: win as cheaply as possible
            pick = max if self.rng.random() < self.cfg.p_overkill else min
            chosen = pick(winners, key=lambda a: card_strength(a.card, obs.manilha))
            return self._maybe_facedown(chosen, facedown, safe=False)

        if self.rng.random() < self.cfg.p_burn:         # R7: burn a usable card instead of folding low
            return self._maybe_facedown(faceup[-1], facedown, safe=True)
        return self._dump(obs, faceup, facedown)

    @staticmethod
    def _winners(faceup, opp_best, obs):
        return [a for a in faceup if card_strength(a.card, obs.manilha) > opp_best]

    def _dump(self, obs: Observation, faceup, facedown):
        """Discard the weakest card; hide it face-down when that can't cost the round (R10)."""
        return self._maybe_facedown(faceup[0], facedown, safe=True)

    def _maybe_facedown(self, faceup_action: PlayCard, facedown, safe: bool):
        """Pick the face-up/face-down variant. Obey R10: hide iff `safe`; break: invert it."""
        want_facedown = safe
        if self.rng.random() < self.cfg.p_bad_facedown:
            want_facedown = not want_facedown
        if want_facedown:
            twin = next((a for a in facedown if a.card == faceup_action.card), None)
            if twin is not None:
                return twin
        return faceup_action

    # ------------------------------------------------------------- betting (R5, R8)
    def _partner_boost(self, obs: Observation) -> float:
        """Honest read of the partner's señal: a risk shift added to hand strength (R5)."""
        sigs = set(obs.partner_signals)
        boost = 0.0
        if sigs & _POSITIVE_CODES:
            boost += 0.15
        if int(SignalCode.WEAK) in sigs:
            boost -= 0.15
        return boost

    def _should_call(self, obs: Observation) -> bool:
        s = hand_strength(obs)
        if self.rng.random() >= self.cfg.p_ignore_partner:
            s += self._partner_boost(obs)
        won_round = any(r == obs.team for r in obs.round_results)
        d = self._risk_shift
        prob = 0.6 if s > 0.72 - d else 0.25 if s > 0.55 - d else 0.0
        if won_round:
            prob += 0.2
        if s < 0.40:
            prob += self.cfg.bluff
        return self.rng.random() < prob

    def _respond(self, obs: Observation):
        legal = obs.legal_actions
        s = hand_strength(obs)
        won_round = any(r == obs.team for r in obs.round_results)
        if won_round:
            s += 0.10
        if self.rng.random() >= self.cfg.p_ignore_partner:
            s += self._partner_boost(obs)
        d = self._risk_shift
        reraise = next((a for a in legal if isinstance(a, ReraiseTruco)), None)
        if reraise is not None and (s > 0.80 - d or (s > 0.55 - d and self.rng.random() < self.cfg.bluff)):
            return reraise
        if s > 0.42 - d or self.rng.random() < self.cfg.bluff:
            return next(a for a in legal if isinstance(a, AcceptTruco))
        # Would fold — but R8: don't fold a position you've effectively won.
        if won_round and s > 0.5 and self.rng.random() >= self.cfg.p_fold_won:
            return next(a for a in legal if isinstance(a, AcceptTruco))
        return next(a for a in legal if isinstance(a, RunTruco))

    # ----------------------------------------------------- ELEVEN_CHOICE (R11)
    def _eleven(self, obs: Observation):
        combined = list(obs.my_hand) + list(obs.partner_hand or ())
        good = _hand_score(combined, obs.manilha) > 0.45 - self._risk_shift
        if self.rng.random() < self.cfg.p_bad_eleven:
            good = not good
        return next(a for a in obs.legal_actions if isinstance(a, ElevenChoice) and a.play is good)

"""Turn a match into four per-seat token streams — the imperfect-information transcripts.

`MatchTokenizer` is a structured sibling of `cli.render.narrate`: where `narrate` redacts a
transition's events into prose for one human seat, this redacts them into `vocab` tokens for
*all four* seats at once. It maintains one growing stream per seat and, on every transition,
appends each event to each seat's stream under that seat's privacy rules:

  * a face-down card is revealed only to the player who played it; to everyone else it is
    `PLAY_HIDDEN` (identity withheld);
  * a signal reaches only its sender and that sender's partner — opponents see nothing at all;
  * a hand is dealt face-up only into the holder's own stream; the partner's hand is added
    only to the deciding team's streams during a Hand of Eleven.

The invariant this guarantees — *a seat's stream contains exactly what that seat may legally
know* — is the whole point, and is asserted independently in `tests/test_tokenizer.py`.

Driving it (one match):

    mt = MatchTokenizer()
    mt.start(state)                          # BOS + the opening deal
    while not state.done:
        actor = state.current_actor
        mt.begin_turn(actor)                 # TO_ACT prompt into the actor's stream
        state, events = engine.apply_action(state, agent.act(...), rng)
        mt.ingest(state, events)             # redacted tokens into every stream
    stream = mt.streams[seat]                # that seat's full transcript

`begin_turn` appends the `TO_ACT` prompt; the move itself is reconstructed from the resulting
event in `ingest`, so a stream stays correct under any driver (heuristic match, self-play, or
a learner) without the caller having to report which action it chose.
"""

from __future__ import annotations

from ..actions import (
    AcceptTruco,
    CallTruco,
    ElevenChoice,
    PlayCard,
    RunTruco,
    Signal,
    SignalCode,
)
from ..engine import (
    CardPlayed,
    ElevenDecided,
    ElevenStarted,
    HandDealt,
    HandResolved,
    MatchWon,
    Redeal,
    RoundResolved,
    SetWon,
    SignalSent,
    TrucoAnswered,
    TrucoCalled,
)
from ..state import MatchState, partner_of, team_of
from . import vocab
from .action_space import action_index

_SEATS = range(4)


class MatchTokenizer:
    """Accumulates one redacted token stream per seat over the course of a match."""

    def __init__(self) -> None:
        self.streams: dict[int, list[int]] = {s: [] for s in _SEATS}
        self._decider: int | None = None  # remembered across an Eleven start→decision

    # ------------------------------------------------------------------- lifecycle
    def start(self, state: MatchState) -> None:
        """Open every stream with BOS and the first hand (dealt by `new_match`, eventless)."""
        self.streams = {s: [vocab.BOS] for s in _SEATS}
        assert state.hand is not None, "start() needs a freshly dealt match"
        h = state.hand
        self._emit_hand_dealt(state, h.dealer, h.vira.id, h.manilha)

    def begin_turn(self, seat: int) -> None:
        """Mark that `seat` must act now: append the TO_ACT prompt to its stream."""
        self.streams[seat].append(vocab.TO_ACT)

    def ingest(self, state: MatchState, events) -> None:
        """Append every event in a transition to every seat's stream, redacted per seat."""
        for ev in events:
            self._emit(state, ev)

    # ------------------------------------------------------------------- per-event
    def _emit(self, state: MatchState, ev) -> None:
        if isinstance(ev, CardPlayed):
            self._emit_move(ev.seat, PlayCard(ev.card, ev.face_down), hidden=ev.face_down)
        elif isinstance(ev, SignalSent):
            self._emit_signal(ev.seat, ev.code)
        elif isinstance(ev, TrucoCalled):
            # A re-raise also surfaces as TrucoCalled; both read publicly as "stakes raised".
            self._emit_move(ev.seat, CallTruco())
        elif isinstance(ev, TrucoAnswered):
            answer = AcceptTruco() if ev.response == "accept" else RunTruco()
            self._emit_move(ev.seat, answer)
        elif isinstance(ev, HandDealt):
            self._emit_hand_dealt(state, ev.dealer, ev.vira.id, ev.manilha)
        elif isinstance(ev, ElevenStarted):
            self._emit_eleven_start(state, ev.team, ev.decider)
        elif isinstance(ev, ElevenDecided):
            self._emit_eleven_decided(ev.play)
        elif isinstance(ev, RoundResolved):
            self._broadcast_result(vocab.ROUND_END, ev.result_team)
        elif isinstance(ev, HandResolved):
            self._emit_hand_resolved(ev.winner_team, ev.value, ev.reason)
        elif isinstance(ev, Redeal):
            for s in _SEATS:
                self.streams[s].append(vocab.REDEAL)
        elif isinstance(ev, SetWon):
            self._broadcast_result(vocab.SET_END, ev.team)
        elif isinstance(ev, MatchWon):
            self._broadcast_result(vocab.MATCH_END, ev.team)
            for s in _SEATS:
                self.streams[s].append(vocab.EOS)

    # ------------------------------------------------------------------- helpers
    def _emit_move(self, actor: int, action, hidden: bool = False) -> None:
        """A move by `actor`: a bare move token to the actor, an actor-tagged one to others.

        `hidden` (a face-down play) withholds the card's identity from everyone but the actor.
        """
        token = vocab.move_token(action_index(action))
        for s in _SEATS:
            rel = (actor - s) % 4
            if rel == 0:                                    # the actor: prompted by TO_ACT
                self.streams[s].append(token)
            elif hidden:                                    # others can't see a face-down card
                self.streams[s].extend((vocab.actor_token(rel), vocab.PLAY_HIDDEN))
            else:
                self.streams[s].extend((vocab.actor_token(rel), token))

    def _emit_signal(self, actor: int, code: int) -> None:
        """A signal is private to its sender and that sender's partner; opponents see nothing."""
        token = vocab.move_token(action_index(Signal(SignalCode(code))))
        partner = partner_of(actor)
        for s in _SEATS:
            if s == actor:
                self.streams[s].append(token)              # own signal (after TO_ACT)
            elif s == partner:
                self.streams[s].extend((vocab.BY_PARTNER, token))

    def _emit_hand_dealt(self, state: MatchState, dealer: int, vira_id: int, manilha: str) -> None:
        assert state.hand is not None
        for s in _SEATS:
            stream = self.streams[s]
            stream.extend((
                vocab.HAND_START,
                vocab.seatrel_token((dealer - s) % 4),
                vocab.card_token(vira_id),
                vocab.manilha_token(manilha),
            ))
            stream.extend(self._score_tokens(state, s))
            stream.append(vocab.MY_HAND)
            stream.extend(vocab.card_token(c.id) for c in state.hand.hands[s])

    def _emit_eleven_start(self, state: MatchState, team: int, decider: int) -> None:
        assert state.hand is not None
        self._decider = decider
        for s in _SEATS:
            stream = self.streams[s]
            stream.extend((
                vocab.ELEVEN_START,
                self._result_token(team, s),
                vocab.seatrel_token((decider - s) % 4),
            ))
            if team_of(s) == team:                          # the deciding team sees both hands
                stream.append(vocab.PARTNER_HAND)
                stream.extend(vocab.card_token(c.id) for c in state.hand.hands[partner_of(s)])

    def _emit_eleven_decided(self, play: bool) -> None:
        assert self._decider is not None, "ElevenDecided without a remembered decider"
        decider = self._decider
        self._decider = None
        token = vocab.move_token(action_index(ElevenChoice(play)))
        outcome = vocab.DECIDED_PLAY if play else vocab.DECIDED_SKIP
        for s in _SEATS:
            if s == decider:                                # the decider's own move (after TO_ACT)
                self.streams[s].append(token)
            else:
                self.streams[s].extend((vocab.ELEVEN_DECISION, outcome))

    def _emit_hand_resolved(self, winner_team: int, value: int, reason: str) -> None:
        reason_token = {
            "rounds": vocab.REASON_ROUNDS,
            "run": vocab.REASON_RUN,
            "eleven_skip": vocab.REASON_ELEVEN_SKIP,
        }[reason]
        for s in _SEATS:
            self.streams[s].extend((
                vocab.HAND_END,
                self._result_token(winner_team, s),
                vocab.num_token(value),
                reason_token,
            ))

    def _broadcast_result(self, marker: int, team: int) -> None:
        """A marker (ROUND/SET/MATCH end) plus a team result, written to every seat."""
        for s in _SEATS:
            self.streams[s].extend((marker, self._result_token(team, s)))

    def _score_tokens(self, state: MatchState, seat: int) -> tuple[int, ...]:
        us, them = team_of(seat), 1 - team_of(seat)
        return (
            vocab.SCORES,
            vocab.num_token(state.set_scores[us]),
            vocab.num_token(state.set_scores[them]),
            vocab.num_token(state.match_sets[us]),
            vocab.num_token(state.match_sets[them]),
        )

    @staticmethod
    def _result_token(team: int, seat: int) -> int:
        if team == -1:
            return vocab.RESULT_TIE
        return vocab.RESULT_US if team == team_of(seat) else vocab.RESULT_THEM

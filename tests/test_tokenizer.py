"""Property test: a seat's token stream contains exactly what that seat may legally know.

This is the guarantee module 1 exists to provide. For many seeded full matches we drive the
real engine, build the four per-seat streams with `MatchTokenizer`, and independently derive —
straight from `observation.observe`, the canonical definition of a seat's legal view — what
each seat is allowed to see. Then we assert the stream never exceeds it:

  * every card a stream reveals was, at some point, legally visible to that seat;
  * a face-down card played by *someone else* appears only as `PLAY_HIDDEN`, never by identity,
    and the count of such hidden plays matches reality;
  * a signal appears only in its sender's and that sender's partner's streams — never an
    opponent's.

Run directly (`uv run python tests/test_tokenizer.py`) for a summary, or under pytest.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import truco`

from truco import engine
from truco.agents import HeuristicAgent
from truco.observation import observe
from truco.rl import vocab
from truco.rl.tokenizer import MatchTokenizer
from truco.state import partner_of, team_of

SEATS = range(4)


class _MatchRecord:
    """Everything one driven match produced: the streams, plus ground truth from `observe`."""

    def __init__(self) -> None:
        self.tokenizer = MatchTokenizer()
        self.visible: dict[int, set[int]] = {s: set() for s in SEATS}   # card ids a seat saw
        self.events: list = []                                          # every engine event


def _absorb_observations(rec: _MatchRecord, state) -> None:
    """Fold each seat's current legal view into its accumulated set of visible card ids.

    Covers the subtle, snapshot-only reveals: your own hand, the public vira, and the
    partner's hand shown to the deciding team during a Hand of Eleven. Plays are handled
    from events (`_absorb_events`) instead, because a round's fourth card is swept off the
    table inside the same transition that plays it and so never appears in any `observe`.
    """
    for s in SEATS:
        obs = observe(state, s)
        seen = rec.visible[s]
        seen.add(obs.vira.id)                                   # the vira is public
        seen.update(c.id for c in obs.my_hand)                  # my own hand
        if obs.partner_hand is not None:                        # Eleven reveal to my team
            seen.update(c.id for c in obs.partner_hand)


def _absorb_events(rec: _MatchRecord, events) -> None:
    """Record cards made visible by plays: a face-up card is public; a face-down card is
    seen only by the player who played it."""
    for e in events:
        if isinstance(e, engine.CardPlayed):
            if e.face_down:
                rec.visible[e.seat].add(e.card.id)
            else:
                for s in SEATS:
                    rec.visible[s].add(e.card.id)


def play_match(seed: int) -> _MatchRecord:
    """Drive one full heuristic-vs-heuristic match, tokenizing and recording ground truth."""
    rng = random.Random(seed)
    state = engine.new_match(rng, dealer=seed % 4)
    agents = {s: HeuristicAgent(rng=random.Random(rng.random())) for s in SEATS}

    rec = _MatchRecord()
    rec.tokenizer.start(state)
    _absorb_observations(rec, state)

    while not state.done:
        actor = state.current_actor
        rec.tokenizer.begin_turn(actor)
        action = agents[actor].act(observe(state, actor))
        state, events = engine.apply_action(state, action, rng)
        rec.events.extend(events)
        rec.tokenizer.ingest(state, events)
        _absorb_events(rec, events)
        _absorb_observations(rec, state)

    return rec


def _check_match(rec: _MatchRecord) -> None:
    streams = rec.tokenizer.streams
    for s in SEATS:
        stream = streams[s]

        # Structural sanity: a real, framed transcript.
        assert stream and stream[0] == vocab.BOS, f"seat {s}: stream must open with BOS"
        assert stream[-1] == vocab.EOS, f"seat {s}: a finished match must close with EOS"
        assert len(stream) > 20, f"seat {s}: suspiciously short stream ({len(stream)} tokens)"
        assert all(0 <= t < vocab.VOCAB_SIZE for t in stream), f"seat {s}: token out of vocab"

        # (1) No illegal card ever surfaces.
        for t in stream:
            cid = vocab.revealed_card_id(t)
            if cid is not None and cid not in rec.visible[s]:
                raise AssertionError(
                    f"seat {s}: stream reveals card {cid} ({vocab.token_name(t)}) it never "
                    f"legally saw"
                )

        # (2) Others' face-down plays are hidden — by identity and by count.
        hidden_in_stream = stream.count(vocab.PLAY_HIDDEN)
        hidden_expected = sum(
            1 for e in rec.events
            if isinstance(e, engine.CardPlayed) and e.face_down and e.seat != s
        )
        assert hidden_in_stream == hidden_expected, (
            f"seat {s}: {hidden_in_stream} hidden plays in stream, expected {hidden_expected}"
        )

        # (3) Signals reach only sender and partner — never an opponent.
        signals_in_stream = sum(1 for t in stream if vocab.is_signal_move(t))
        signals_expected = sum(
            1 for e in rec.events
            if isinstance(e, engine.SignalSent) and (e.seat == s or partner_of(e.seat) == s)
        )
        assert signals_in_stream == signals_expected, (
            f"seat {s}: {signals_in_stream} signals in stream, expected {signals_expected} "
            f"(an opponent's signal must never appear)"
        )


def test_streams_never_leak_illegal_information() -> None:
    for seed in range(200):
        _check_match(play_match(seed))


def test_face_down_plays_are_hidden_from_every_non_actor() -> None:
    """A sound, non-vacuous guard. Card identity repeats across hands (the deck is re-dealt
    every hand), so a card a face-down play hides can legitimately appear elsewhere in a
    seat's stream — checking a card id against the whole match would false-positive. The
    leak-proof identity is structural instead: each face-down play must surface as exactly
    one `PLAY_HIDDEN` token in each of the three non-actors' streams, and the real card only
    in the actor's. Were the tokenizer ever to reveal a face-down card to a non-actor, that
    seat's `PLAY_HIDDEN` count would fall short, breaking the 3:1 total below."""
    total_face_down = total_hidden_tokens = 0
    for seed in range(60):
        rec = play_match(seed)
        total_face_down += sum(
            1 for e in rec.events if isinstance(e, engine.CardPlayed) and e.face_down
        )
        total_hidden_tokens += sum(
            st.count(vocab.PLAY_HIDDEN) for st in rec.tokenizer.streams.values()
        )
    assert total_face_down > 0, "no face-down plays occurred — guard would be vacuous"
    assert total_hidden_tokens == 3 * total_face_down, (
        f"{total_hidden_tokens} hidden tokens for {total_face_down} face-down plays; "
        f"expected exactly 3 per play (one per non-actor)"
    )


def _main() -> int:
    seeds = 200
    total_tokens = hidden = signals = 0
    for seed in range(seeds):
        rec = play_match(seed)
        _check_match(rec)
        total_tokens += sum(len(st) for st in rec.tokenizer.streams.values())
        hidden += sum(1 for e in rec.events if isinstance(e, engine.CardPlayed) and e.face_down)
        signals += sum(1 for e in rec.events if isinstance(e, engine.SignalSent))
    print(f"PASS — {seeds} matches, redaction invariant holds for all 4 seats each.")
    print(f"  avg stream length : {total_tokens / (seeds * 4):.1f} tokens/seat")
    print(f"  face-down plays   : {hidden} (each hidden from non-actors)")
    print(f"  signals sent      : {signals} (each seen only by sender + partner)")
    print(f"  vocabulary size   : {vocab.VOCAB_SIZE} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

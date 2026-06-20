"""Turn engine state into text for the human — strictly from one seat's point of view.

`render_board(obs)` draws the scoreboard, vira/manilha, the current table and the
human's hand. `narrate(events, human_seat)` turns a transition's events into lines,
honouring privacy: an opponent's signal is never printed, and a face-down card played by
someone else is shown only as "a card face-down".
"""

from __future__ import annotations

from ..actions import SIGNAL_LABEL
from ..cards import is_zap
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
from ..observation import Observation
from ..state import RAISE_NAMES, VALUES, partner_of, team_of

SEAT_NAME = {0: "You", 1: "Opp-1", 2: "Partner", 3: "Opp-2"}
_WIDTH = 60


def _seat(seat: int) -> str:
    return SEAT_NAME.get(seat, f"Seat {seat}")


def _team(team: int, human_seat: int) -> str:
    return "Us" if team == team_of(human_seat) else "Them"


def _round_tag(result: int, human_seat: int) -> str:
    if result == -1:
        return "tie"
    return _team(result, human_seat)


def render_board(obs: Observation) -> str:
    us, them = obs.team, 1 - obs.team
    bar = "=" * _WIDTH
    rule = "-" * _WIDTH
    lines = [bar]

    # Scoreboard.
    lines.append(
        f"  Us  {obs.set_scores[us]:>2} pts / {obs.match_sets[us]} sets"
        f"        Them  {obs.set_scores[them]:>2} pts / {obs.match_sets[them]} sets"
    )

    # Vira / manilha / stakes.
    stake = RAISE_NAMES[obs.stake_level] if obs.stake_level else "—"
    lines.append(
        f"  Vira {obs.vira}  →  manilha is any {obs.manilha}"
        f"     (worth {obs.value}, stake: {stake})"
    )
    lines.append(rule)

    # The table this round.
    lines.append(f"  Table — round {obs.current_round + 1}:")
    played = {vp.seat: vp for vp in obs.table}
    for seat in range(4):
        vp = played.get(seat)
        if vp is None:
            shown = "—"
        elif vp.card is None:                       # someone else's hidden card
            shown = "(face-down)"
        elif vp.face_down:                          # our own face-down card
            shown = f"{vp.card} (your face-down)"
        else:
            shown = str(vp.card)
        marker = " <" if seat == obs.current_actor else ""
        lines.append(f"    {_seat(seat):<9}: {shown}{marker}")

    # Round history + the partner's private signal (if any).
    if obs.round_results:
        tags = ", ".join(
            f"R{i + 1} {_round_tag(r, obs.seat)}" for i, r in enumerate(obs.round_results)
        )
        lines.append(f"  Rounds so far: {tags}")
    if obs.partner_signals:
        flashed = ", ".join(f"\"{SIGNAL_LABEL[c]}\"" for c in obs.partner_signals)
        lines.append(f"  ✋ Partner's signal(s): {flashed}")
    if obs.my_signals:
        flashed = ", ".join(f"\"{SIGNAL_LABEL[c]}\"" for c in obs.my_signals)
        lines.append(f"     (you signalled: {flashed})")

    # Partner's hand is visible only while your team decides a Hand of Eleven.
    if obs.partner_hand is not None:
        lines.append(f"  Partner's hand (Eleven): {_cards(obs.partner_hand, obs.manilha)}")

    lines.append(rule)
    lines.append(f"  Your hand: {_cards(obs.my_hand, obs.manilha)}")
    lines.append(bar)
    return "\n".join(lines)


def _cards(cards, manilha: str) -> str:
    if not cards:
        return "(none)"
    return "  ".join(str(c) + (" ⚡" if is_zap(c, manilha) else "") for c in cards)


def narrate(events, human_seat: int) -> list[str]:
    """Public, privacy-safe lines describing a transition, from the human's seat."""
    out: list[str] = []
    for e in events:
        if isinstance(e, HandDealt):
            out.append(
                f"\n— Hand #{e.hand_number} dealt · dealer {_seat(e.dealer)} · "
                f"vira {e.vira} → manilha {e.manilha}."
            )
        elif isinstance(e, ElevenStarted):
            out.append(
                f"  Hand of Eleven for {_team(e.team, human_seat)}: "
                f"{_seat(e.decider)} chooses play (worth 3) or skip (opponents +1)."
            )
        elif isinstance(e, ElevenDecided):
            out.append(
                f"  {_team(e.team, human_seat)} chose to "
                f"{'PLAY' if e.play else 'SKIP'} the Hand of Eleven."
            )
        elif isinstance(e, SignalSent):
            # Private partner channel: only surfaces if it reaches the human.
            if e.seat == human_seat or partner_of(e.seat) == human_seat:
                who = "You" if e.seat == human_seat else "Partner"
                out.append(f"  ✋ {who} signalled: \"{SIGNAL_LABEL[e.code]}\"")
        elif isinstance(e, CardPlayed):
            if e.face_down:
                what = f"{e.card} face-down" if e.seat == human_seat else "a card face-down"
            else:
                what = str(e.card)
            plays = "play" if e.seat == human_seat else "plays"
            out.append(f"  {_seat(e.seat)} {plays} {what}.")
        elif isinstance(e, TrucoCalled):
            calls = "call" if e.seat == human_seat else "calls"
            out.append(
                f"  🔥 {_seat(e.seat)} {calls} {RAISE_NAMES[e.proposed_level]}! "
                f"(stake → {VALUES[e.proposed_level]})"
            )
        elif isinstance(e, TrucoAnswered):
            you = e.seat == human_seat
            verb = {
                "accept": "accept" if you else "accepts",
                "run": "run" if you else "runs",
                "raise": "re-raise" if you else "re-raises",
            }[e.response]
            out.append(f"  {_seat(e.seat)} {verb}.")
        elif isinstance(e, RoundResolved):
            if e.result_team == -1:
                out.append(f"  Round {e.round_index + 1}: tied.")
            else:
                out.append(
                    f"  Round {e.round_index + 1}: {_team(e.result_team, human_seat)} win "
                    f"(led by {_seat(e.winning_seat)})."
                )
        elif isinstance(e, HandResolved):
            tail = {"rounds": "", "run": " — they ran", "eleven_skip": " — Eleven skipped"}[e.reason]
            out.append(f"  ➜ {_team(e.winner_team, human_seat)} take the hand: +{e.value}{tail}.")
        elif isinstance(e, Redeal):
            out.append("  All three rounds tied — no score; re-dealing.")
        elif isinstance(e, SetWon):
            out.append(f"  🏆 {_team(e.team, human_seat)} win the SET!")
        elif isinstance(e, MatchWon):
            out.append(f"  🎉 {_team(e.team, human_seat)} WIN THE MATCH! 🎉")
    return out

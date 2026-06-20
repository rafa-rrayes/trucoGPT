"""Serialise the engine's per-seat view into JSON for the browser.

Everything here is built from :func:`truco.observation.observe`, so a seat only ever
receives what it may legally know — opponents' hands and the identity of others'
face-down cards are never serialised. The room layer wraps the game view returned by
:func:`game_view` with lobby data (seat occupants, the narration log, etc.).
"""

from __future__ import annotations

from ..actions import (
    AcceptTruco,
    CallTruco,
    ElevenChoice,
    PlayCard,
    ReraiseTruco,
    RunTruco,
    Signal,
    SIGNAL_LABEL,
)
from ..cards import SUIT_NAME, SUIT_SYMBOL, Card, is_zap
from ..observation import Observation
from ..rl.action_space import action_from_index, action_index
from ..state import Phase, RAISE_NAMES, VALUES, team_of

_RED_SUITS = {"diamonds", "hearts"}


def card_json(card: Card, manilha: str) -> dict:
    """A single card: id, rank, suit glyph/colour, and its role this hand."""
    suit = SUIT_NAME[card.suit]
    return {
        "id": card.id,
        "rank": card.rank,
        "suit": suit,
        "symbol": SUIT_SYMBOL[card.suit],
        "color": "red" if suit in _RED_SUITS else "black",
        "str": str(card),
        "manilha": card.rank == manilha,
        "zap": is_zap(card, manilha),
    }


def _action_label(action) -> str:
    if isinstance(action, PlayCard):
        return f"Play {action.card}" + (" face-down" if action.face_down else "")
    if isinstance(action, Signal):
        return f'Signal: "{SIGNAL_LABEL[int(action.code)]}"'
    if isinstance(action, CallTruco):
        return "Truco / raise"
    if isinstance(action, AcceptTruco):
        return "Accept"
    if isinstance(action, RunTruco):
        return "Run (fold)"
    if isinstance(action, ReraiseTruco):
        return "Re-raise"
    if isinstance(action, ElevenChoice):
        return "Play this hand" if action.play else "Skip (give +1)"
    return str(action)


def action_json(index: int, manilha: str) -> dict:
    """Describe a legal action index for the UI to render and echo back."""
    a = action_from_index(index)
    d: dict = {"index": index, "label": _action_label(a)}
    if isinstance(a, PlayCard):
        d["kind"] = "play"
        d["cardId"] = a.card.id
        d["faceDown"] = a.face_down
    elif isinstance(a, Signal):
        d["kind"] = "signal"
        d["signalCode"] = int(a.code)
    elif isinstance(a, CallTruco):
        d["kind"] = "truco"
    elif isinstance(a, AcceptTruco):
        d["kind"] = "accept"
    elif isinstance(a, RunTruco):
        d["kind"] = "run"
    elif isinstance(a, ReraiseTruco):
        d["kind"] = "reraise"
    elif isinstance(a, ElevenChoice):
        d["kind"] = "eleven"
        d["play"] = a.play
    return d


_PHASE_NAME = {
    Phase.ELEVEN_CHOICE: "eleven_choice",
    Phase.PLAY: "play",
    Phase.RESPOND: "respond",
    Phase.HAND_DONE: "hand_done",
    Phase.MATCH_DONE: "match_done",
}


def _result_word(result: int, my_team: int) -> str:
    if result == -1:
        return "tie"
    return "us" if result == my_team else "them"


def game_view(state, seat: int) -> dict:
    """The full game state as `seat` may legally see it, ready to JSON-encode."""
    obs: Observation = _observe(state, seat)
    manilha = obs.manilha
    us, them = obs.team, 1 - obs.team

    table = []
    for vp in obs.table:
        hidden = vp.card is None and vp.face_down
        table.append({
            "seat": vp.seat,
            "card": card_json(vp.card, manilha) if vp.card is not None else None,
            "faceDown": vp.face_down,
            "hidden": hidden,
            "isYou": vp.seat == seat,
        })

    pending = None
    if obs.pending is not None:
        pending = {
            "level": obs.pending.proposed_level,
            "value": VALUES[obs.pending.proposed_level],
            "name": RAISE_NAMES[obs.pending.proposed_level],
            "callerTeam": "us" if obs.pending.caller_team == obs.team else "them",
            "responder": obs.pending.responder,
            "youRespond": obs.pending.responder == seat,
        }

    your_turn = obs.current_actor == seat
    legal = [action_json(action_index(a), manilha) for a in obs.legal_actions] if your_turn else []

    # Index the play actions by card so the hand can render its own buttons directly.
    play_up = {a["cardId"]: a["index"] for a in legal if a.get("kind") == "play" and not a["faceDown"]}
    play_down = {a["cardId"]: a["index"] for a in legal if a.get("kind") == "play" and a["faceDown"]}
    hand = []
    for c in obs.my_hand:
        cj = card_json(c, manilha)
        cj["playIndex"] = play_up.get(c.id)
        cj["faceDownIndex"] = play_down.get(c.id)
        cj["playable"] = cj["playIndex"] is not None
        hand.append(cj)

    return {
        "seat": seat,
        "team": obs.team,
        "phase": _PHASE_NAME[obs.phase],
        "yourTurn": your_turn,
        "currentActor": obs.current_actor,
        "dealer": obs.dealer,
        "lead": obs.lead,
        "vira": card_json(obs.vira, manilha),
        "manilha": manilha,
        "round": obs.current_round + 1,
        "roundResults": [_result_word(r, obs.team) for r in obs.round_results],
        "scores": {
            "us": obs.set_scores[us], "them": obs.set_scores[them],
            "usSets": obs.match_sets[us], "themSets": obs.match_sets[them],
        },
        "stake": {
            "level": obs.stake_level,
            "value": obs.value,
            "name": RAISE_NAMES[obs.stake_level],
            "locked": obs.truco_locked,
        },
        "pending": pending,
        "table": table,
        "hand": hand,
        "legalActions": legal,
        "mySignals": [SIGNAL_LABEL[c] for c in obs.my_signals],
        "partnerSignals": [SIGNAL_LABEL[c] for c in obs.partner_signals],
        "partnerHand": (
            [card_json(c, manilha) for c in obs.partner_hand]
            if obs.partner_hand is not None else None
        ),
    }


def public_view(state) -> dict:
    """A fully public view for spectators: no hands, all face-down cards anonymous."""
    v = game_view(state, 0)
    v["hand"] = []
    v["legalActions"] = []
    v["partnerHand"] = None
    v["mySignals"] = []
    v["partnerSignals"] = []
    v["yourTurn"] = False
    for cell in v["table"]:
        if cell["faceDown"]:
            cell["card"] = None
            cell["hidden"] = True
            cell["isYou"] = False
    return v


def _observe(state, seat: int) -> Observation:
    from ..observation import observe
    return observe(state, seat)

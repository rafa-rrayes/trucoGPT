"""A stable, fixed-size action index space and the legal-action mask.

The engine enumerates a *variable* set of legal `Action` value-objects per state. A policy
network needs the opposite: a fixed-width output where index `i` always means the same move,
regardless of hand or phase. This module is that bijection.

The encoding is **absolute / card-id based**, not hand-slot based: playing a given card maps
to the same index whatever position it holds in the hand. This mirrors the observation
encoder (which one-hots cards by `Card.id`), so the network sees a card at a fixed feature
slot *and* acts on it at a fixed action slot — a clean correspondence to learn — and it makes
`action_from_index` independent of any observation.

Layout (ACTION_SIZE = 94):
    0..39    PlayCard(card_from_id(i), face_down=False)
    40..79   PlayCard(card_from_id(i-40), face_down=True)
    80..87   Signal(SignalCode(i-80))
    88       CallTruco()
    89       AcceptTruco()
    90       RunTruco()
    91       ReraiseTruco()
    92       ElevenChoice(play=True)
    93       ElevenChoice(play=False)

A decoded action is *value-equal* to the one the engine enumerates (actions are frozen
dataclasses; `card_from_id` reconstructs the same `Card`), so `engine.apply_action` accepts
it. Callers only ever sample indices where `legal_mask` is 1, so every emitted action is
legal by construction.
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
    SignalCode,
)
from ..cards import card_from_id
from ..observation import Observation

# --- index block boundaries (kept as constants so the layout is stated once) -----------
_PLAY_FACEUP = 0      # 0..39
_PLAY_FACEDOWN = 40   # 40..79
_SIGNAL = 80          # 80..87
_CALL_TRUCO = 88
_ACCEPT = 89
_RUN = 90
_RERAISE = 91
_ELEVEN_PLAY = 92
_ELEVEN_SKIP = 93

ACTION_SIZE = 94


def action_index(action) -> int:
    """Map an `Action` value-object to its stable index in `0..ACTION_SIZE`."""
    if isinstance(action, PlayCard):
        base = _PLAY_FACEDOWN if action.face_down else _PLAY_FACEUP
        return base + action.card.id
    if isinstance(action, Signal):
        return _SIGNAL + int(action.code)
    if isinstance(action, CallTruco):
        return _CALL_TRUCO
    if isinstance(action, AcceptTruco):
        return _ACCEPT
    if isinstance(action, RunTruco):
        return _RUN
    if isinstance(action, ReraiseTruco):
        return _RERAISE
    if isinstance(action, ElevenChoice):
        return _ELEVEN_PLAY if action.play else _ELEVEN_SKIP
    raise TypeError(f"not an encodable action: {action!r}")


def action_from_index(index: int):
    """Inverse of `action_index`. Independent of any observation."""
    if not 0 <= index < ACTION_SIZE:
        raise ValueError(f"action index out of range: {index}")
    if index < _SIGNAL:
        face_down = index >= _PLAY_FACEDOWN
        card_id = index - (_PLAY_FACEDOWN if face_down else _PLAY_FACEUP)
        return PlayCard(card_from_id(card_id), face_down=face_down)
    if index < _CALL_TRUCO:
        return Signal(SignalCode(index - _SIGNAL))
    if index == _CALL_TRUCO:
        return CallTruco()
    if index == _ACCEPT:
        return AcceptTruco()
    if index == _RUN:
        return RunTruco()
    if index == _RERAISE:
        return ReraiseTruco()
    if index == _ELEVEN_PLAY:
        return ElevenChoice(play=True)
    return ElevenChoice(play=False)  # _ELEVEN_SKIP


def legal_mask(obs: Observation) -> list[float]:
    """A length-`ACTION_SIZE` 0/1 mask of the actions legal for the observing seat.

    All-zero when it is not this seat's turn (`obs.legal_actions` is empty then).
    """
    mask = [0.0] * ACTION_SIZE
    for action in obs.legal_actions:
        mask[action_index(action)] = 1.0
    return mask

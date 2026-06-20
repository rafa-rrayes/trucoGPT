"""The token vocabulary: a factored, seat-relative language for a whole match.

A future sequence model reads a *per-seat* token stream — the entire history legally
visible to one player — and predicts the next move. This module is that fixed alphabet:
a bijection between small integers and the structural markers, redacted card reveals, and
moves that make up a transcript. `tokenizer.MatchTokenizer` produces the streams; this file
only defines what the symbols *are*.

Two design choices matter:

  * **Seat-relative.** Every actor/team/seat is named relative to the reader (me / right /
    partner / left, and "us" / "them"), so one model plays any seat without re-learning
    orientation — mirroring `observation.encode`.

  * **Moves mirror `action_space`.** The 94 moves an agent can emit occupy one contiguous
    block, `MOVE_BASE + i`, where `i` is exactly the `action_space` index. So a predicted
    move token decodes straight back to an engine `Action`, and the existing `legal_mask`
    projects onto this block unchanged. The one extra symbol, `PLAY_HIDDEN`, is the *only*
    move-like token an agent can never emit: it is how a face-down card played by someone
    else appears to a reader who is not allowed to know its identity.

Card identity reaches a stream two ways, both of which the redaction test treats as a
"reveal": a `CARD_*` token (your dealt hand, the public vira, a partner's hand shown during
a Hand of Eleven) and a face-up/own-face-down move token (which carries the card id inside
the `action_space` index). A `PLAY_HIDDEN` token reveals nothing.
"""

from __future__ import annotations

from ..actions import PlayCard, Signal
from ..cards import RANK_INDEX, RANK_ORDER
from .action_space import ACTION_SIZE, action_from_index

# --- single-symbol tokens ---------------------------------------------------------------
PAD = 0          # padding (never emitted into a stream; reserved for batching)
BOS = 1          # beginning of a match
EOS = 2          # end of a match (after MATCH_END)

TO_ACT = 3       # "you act now" — the prompt that precedes the reader's own move
HAND_START = 4   # a hand was dealt (followed by dealer, vira, manilha, scores, my hand)
MY_HAND = 5      # precedes the reader's three dealt cards
PARTNER_HAND = 6 # precedes the partner's three cards (Eleven reveal, deciding team only)
SCORES = 7       # precedes four NUM tokens: us pts, them pts, us sets, them sets
ROUND_END = 8    # a round resolved (followed by a RESULT token)
HAND_END = 9     # a hand resolved (followed by RESULT, NUM value, REASON)
SET_END = 10     # a set was won (followed by a RESULT token)
MATCH_END = 11   # the match was won (followed by a RESULT token, then EOS)
REDEAL = 12      # all three rounds tied — no score, re-deal
ELEVEN_START = 13  # a Hand of Eleven began (followed by RESULT team, SEATREL decider)
ELEVEN_DECISION = 14  # an observer's view of the decider's choice (followed by DECIDED_*)

# Actor of a non-self event, relative to the reader. Self is marked by TO_ACT, never here.
BY_RIGHT = 15    # the player to my right (next seat)
BY_PARTNER = 16  # my partner (across the table)
BY_LEFT = 17     # the player to my left (previous seat)

# Team/result, relative to the reader's team.
RESULT_US = 18
RESULT_THEM = 19
RESULT_TIE = 20

# Why a hand ended.
REASON_ROUNDS = 21      # decided by winning rounds
REASON_RUN = 22         # the other team folded a raise
REASON_ELEVEN_SKIP = 23 # an at-eleven team declined to play

# An observer's view of a Hand-of-Eleven decision (the decider emits a move token instead).
DECIDED_PLAY = 24
DECIDED_SKIP = 25

# --- block tokens (base + offset) -------------------------------------------------------
MANILHA_BASE = 26   # + RANK_INDEX[rank]  (0..9)   -> which rank is the manilha this hand
NUM_BASE = 36       # + n (0..12)                  -> a small integer (scores, hand value)
SEATREL_BASE = 49   # + rel (0..3)                 -> a seat, relative to the reader
CARD_BASE = 53      # + card_id (0..39)            -> a known card's identity (hand/vira)
MOVE_BASE = 93      # + action_index (0..93)       -> a move, == action_space index
PLAY_HIDDEN = MOVE_BASE + ACTION_SIZE  # 187: a face-down card played by someone else

VOCAB_SIZE = PLAY_HIDDEN + 1  # 188

_NUM_MAX = 12       # largest integer a NUM token represents (hand value caps at 12)


# --- builders ---------------------------------------------------------------------------
def manilha_token(rank: str) -> int:
    return MANILHA_BASE + RANK_INDEX[rank]


def num_token(n: int) -> int:
    if not 0 <= n <= _NUM_MAX:
        raise ValueError(f"NUM token out of range: {n}")
    return NUM_BASE + n


def seatrel_token(rel: int) -> int:
    if not 0 <= rel < 4:
        raise ValueError(f"seat-relative position out of range: {rel}")
    return SEATREL_BASE + rel


def card_token(card_id: int) -> int:
    if not 0 <= card_id < 40:
        raise ValueError(f"card id out of range: {card_id}")
    return CARD_BASE + card_id


def move_token(action_index: int) -> int:
    if not 0 <= action_index < ACTION_SIZE:
        raise ValueError(f"action index out of range: {action_index}")
    return MOVE_BASE + action_index


# Relative actor (rel 1/2/3); rel 0 (self) is never an actor token — TO_ACT marks self.
_ACTOR_BY_REL = {1: BY_RIGHT, 2: BY_PARTNER, 3: BY_LEFT}


def actor_token(rel: int) -> int:
    if rel not in _ACTOR_BY_REL:
        raise ValueError(f"no actor token for relative position {rel} (0 is self)")
    return _ACTOR_BY_REL[rel]


# --- decoders (used by the redaction test and by logging) -------------------------------
def move_action_index(token: int) -> int | None:
    """The `action_space` index a move token carries, or None if it isn't a move token."""
    if MOVE_BASE <= token < MOVE_BASE + ACTION_SIZE:
        return token - MOVE_BASE
    return None


def revealed_card_id(token: int) -> int | None:
    """The card id a token discloses, or None if it discloses no specific card.

    Covers both ways an identity reaches a stream: a `CARD_*` token, and a face-up or
    own-face-down move token (whose `action_space` index embeds the card). `PLAY_HIDDEN`
    and every non-card token return None.
    """
    if CARD_BASE <= token < CARD_BASE + 40:
        return token - CARD_BASE
    idx = move_action_index(token)
    if idx is not None:
        action = action_from_index(idx)
        if isinstance(action, PlayCard):
            return action.card.id
    return None


def is_signal_move(token: int) -> bool:
    """True if the token is a (private) signal move — the reader's own or their partner's."""
    idx = move_action_index(token)
    return idx is not None and isinstance(action_from_index(idx), Signal)


def token_name(token: int) -> str:
    """A short human-readable label for a token (debugging / test failure messages)."""
    singles = {
        PAD: "PAD", BOS: "BOS", EOS: "EOS", TO_ACT: "TO_ACT", HAND_START: "HAND_START",
        MY_HAND: "MY_HAND", PARTNER_HAND: "PARTNER_HAND", SCORES: "SCORES",
        ROUND_END: "ROUND_END", HAND_END: "HAND_END", SET_END: "SET_END",
        MATCH_END: "MATCH_END", REDEAL: "REDEAL", ELEVEN_START: "ELEVEN_START",
        ELEVEN_DECISION: "ELEVEN_DECISION", BY_RIGHT: "BY_RIGHT", BY_PARTNER: "BY_PARTNER",
        BY_LEFT: "BY_LEFT", RESULT_US: "RESULT_US", RESULT_THEM: "RESULT_THEM",
        RESULT_TIE: "RESULT_TIE", REASON_ROUNDS: "REASON_ROUNDS", REASON_RUN: "REASON_RUN",
        REASON_ELEVEN_SKIP: "REASON_ELEVEN_SKIP", DECIDED_PLAY: "DECIDED_PLAY",
        DECIDED_SKIP: "DECIDED_SKIP", PLAY_HIDDEN: "PLAY_HIDDEN",
    }
    if token in singles:
        return singles[token]
    if MANILHA_BASE <= token < MANILHA_BASE + 10:
        return f"MANILHA_{RANK_ORDER[token - MANILHA_BASE]}"
    if NUM_BASE <= token <= NUM_BASE + _NUM_MAX:
        return f"NUM_{token - NUM_BASE}"
    if SEATREL_BASE <= token < SEATREL_BASE + 4:
        return f"SEATREL_{token - SEATREL_BASE}"
    if CARD_BASE <= token < CARD_BASE + 40:
        return f"CARD_{token - CARD_BASE}"
    idx = move_action_index(token)
    if idx is not None:
        return f"MOVE[{action_from_index(idx)!r}]"
    return f"<{token}>"

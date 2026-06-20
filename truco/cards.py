"""Cards, the deck, and Truco's context-dependent card strength.

A card's strength is **not** intrinsic: it depends on the *vira* (the flipped card)
which fixes the hand's manilha. So `Card` has no natural ordering; instead callers
pass the manilha rank to `card_strength` / `compare`.

Ranking (worst -> best):   4 5 6 7 Q J K A 2 3
Manilha: the rank one above the vira (wraps 3 -> 4); the four manilhas beat every
normal card. Among manilhas the suit breaks ties (worst -> best): diamonds, spades,
hearts, clubs. The clubs manilha is the Zap, the single strongest card.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum

# Rank order, worst -> best. Index in this list is a normal card's base strength (0..9).
RANK_ORDER: tuple[str, ...] = ("4", "5", "6", "7", "Q", "J", "K", "A", "2", "3")
RANK_INDEX: dict[str, int] = {r: i for i, r in enumerate(RANK_ORDER)}


class Suit(IntEnum):
    """Suits, ordered worst -> best for manilha tie-breaks (the IntEnum value *is* that order)."""

    DIAMONDS = 0
    SPADES = 1
    HEARTS = 2
    CLUBS = 3


SUIT_SYMBOL: dict[Suit, str] = {
    Suit.DIAMONDS: "♦",  # ♦
    Suit.SPADES: "♠",    # ♠
    Suit.HEARTS: "♥",    # ♥
    Suit.CLUBS: "♣",     # ♣
}
SUIT_NAME: dict[Suit, str] = {
    Suit.DIAMONDS: "diamonds",
    Suit.SPADES: "spades",
    Suit.HEARTS: "hearts",
    Suit.CLUBS: "clubs",
}

# Strength offset that lifts every manilha above every normal card (max normal strength is 9).
MANILHA_BASE = 100
# A face-down card cannot win a round: it sits below the weakest real card.
FACE_DOWN_STRENGTH = -1


@dataclass(frozen=True)
class Card:
    """A single card. Frozen + hashable so it can live in sets and be deep-copied cheaply."""

    rank: str
    suit: Suit

    def __post_init__(self) -> None:
        if self.rank not in RANK_INDEX:
            raise ValueError(f"unknown rank: {self.rank!r}")

    def __str__(self) -> str:
        return f"{self.rank}{SUIT_SYMBOL[self.suit]}"

    @property
    def id(self) -> int:
        """A stable 0..39 id (rank-major). Useful for one-hot encoding for RL."""
        return RANK_INDEX[self.rank] * 4 + int(self.suit)


def card_from_id(card_id: int) -> Card:
    """Inverse of `Card.id`."""
    if not 0 <= card_id < 40:
        raise ValueError(f"card id out of range: {card_id}")
    return Card(RANK_ORDER[card_id // 4], Suit(card_id % 4))


def build_deck() -> list[Card]:
    """The 40-card Truco deck (8s, 9s, 10s and Jokers already excluded by RANK_ORDER)."""
    return [Card(rank, suit) for rank in RANK_ORDER for suit in Suit]


def shuffled_deck(rng: random.Random) -> list[Card]:
    """A freshly shuffled deck. Randomness is injected for reproducibility."""
    deck = build_deck()
    rng.shuffle(deck)
    return deck


def manilha_rank(vira: Card) -> str:
    """The manilha rank for a hand: one rank above the vira, wrapping 3 -> 4."""
    return RANK_ORDER[(RANK_INDEX[vira.rank] + 1) % len(RANK_ORDER)]


def is_manilha(card: Card, manilha: str) -> bool:
    return card.rank == manilha


def is_zap(card: Card, manilha: str) -> bool:
    """The Zap: the clubs manilha, strongest card in the game."""
    return card.rank == manilha and card.suit is Suit.CLUBS


def card_strength(card: Card, manilha: str) -> int:
    """Total order over real cards given the hand's manilha.

    Normal cards: 0..9 by rank. Manilhas: 100 + suit (100..103), so any manilha
    beats any normal card and the suit (diamonds<spades<hearts<clubs) breaks
    manilha-vs-manilha. Strengths are unique among manilhas, so two manilhas never tie.
    """
    if card.rank == manilha:
        return MANILHA_BASE + int(card.suit)
    return RANK_INDEX[card.rank]


def compare(a: Card, b: Card, manilha: str) -> int:
    """-1 if a < b, 0 if equal strength (only possible for two equal normal cards), 1 if a > b."""
    sa, sb = card_strength(a, manilha), card_strength(b, manilha)
    return (sa > sb) - (sa < sb)

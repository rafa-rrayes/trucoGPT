"""Run one config-vs-config match. Pure engine — no torch, so it parallelises across processes.

Team 0 (seats 0 and 2) is played by one `BotConfig`, team 1 (seats 1 and 3) by another. Both
seats on a team share the config but get independent RNGs, mirroring how `data.play_heuristic_match`
seats the canonical bot. A single core loop drives the match; pass a `MatchTokenizer` to also
build the four per-seat streams (used when regenerating the corpus).
"""

from __future__ import annotations

import random

from .. import engine
from ..agents import BotConfig, ConfigurableBot
from ..observation import observe
from ..rl.tokenizer import MatchTokenizer
from ..state import team_of


def _run_seats(seat_cfgs: list[BotConfig], seed: int, dealer: int,
               tok: MatchTokenizer | None, sets_to_win: int = 1) -> int:
    """Drive a full match with one config per seat (`seat_cfgs[0..3]`); return the winning team.

    Per-seat configs let teammates differ (varied partners), not just the two teams. Tokenizes
    into `tok` if given. The shared-config team layout is just `[cfg0, cfg1, cfg0, cfg1]`.
    """
    assert len(seat_cfgs) == 4, "need exactly one config per seat"
    rng = random.Random(seed)
    state = engine.new_match(rng, dealer=dealer, sets_to_win=sets_to_win)
    agents = {s: ConfigurableBot(seat_cfgs[s], rng=random.Random(rng.random()))
              for s in range(4)}
    if tok is not None:
        tok.start(state)
    while not state.done:
        actor = state.current_actor
        if tok is not None:
            tok.begin_turn(actor)
        action = agents[actor].act(observe(state, actor))
        state, events = engine.apply_action(state, action, rng)
        if tok is not None:
            tok.ingest(state, events)
    return state.winner


def _run(cfg0: BotConfig, cfg1: BotConfig, seed: int, dealer: int,
         tok: MatchTokenizer | None, sets_to_win: int = 1) -> int:
    """Drive a full match, team 0 = cfg0 (seats 0,2), team 1 = cfg1 (seats 1,3)."""
    return _run_seats([cfg0, cfg1, cfg0, cfg1], seed, dealer, tok, sets_to_win)


def play_config_match(cfg0: BotConfig, cfg1: BotConfig, seed: int, dealer: int = 0,
                      sets_to_win: int = 1) -> int:
    """Play cfg0 (seats 0,2) vs cfg1 (seats 1,3); return the winning team (0 or 1)."""
    return _run(cfg0, cfg1, seed, dealer, tok=None, sets_to_win=sets_to_win)


def tokenize_config_match(cfg0: BotConfig, cfg1: BotConfig, seed: int, dealer: int = 0,
                          sets_to_win: int = 1) -> tuple[MatchTokenizer, int]:
    """Play the match; return (tokenizer with four per-seat streams, winning team 0/1)."""
    tok = MatchTokenizer()
    winner = _run(cfg0, cfg1, seed, dealer, tok=tok, sets_to_win=sets_to_win)
    return tok, winner


def tokenize_config_match_seats(seat_cfgs: list[BotConfig], seed: int, dealer: int = 0,
                                sets_to_win: int = 1) -> tuple[MatchTokenizer, int]:
    """Like `tokenize_config_match`, but with an independent config per seat (varied partners)."""
    tok = MatchTokenizer()
    winner = _run_seats(seat_cfgs, seed, dealer, tok=tok, sets_to_win=sets_to_win)
    return tok, winner

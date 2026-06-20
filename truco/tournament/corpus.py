"""Regenerate the pretraining corpus from the tournament's top configs.

Both teams in every corpus match are drawn (independently) from the strongest configs, so the
seats the model learns to imitate play near the top of the field. This replaces the old
single-style self-play corpus with a stronger, more varied one. Streams are saved in the exact
flat `.npz` format `rl.data.load_corpus` expects, so the pretrainer is unchanged.

With `winners_only=True` only the two seats of the *winning* team are kept from each match, so the
model imitates play that actually won — never the losing side of a strong-vs-strong game.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..agents import BotConfig
from ..state import team_of
from .match import tokenize_config_match, tokenize_config_match_seats

# ---- parallel helpers (top-level so ProcessPoolExecutor can pickle them) ----

_POOL_CFGS: list[BotConfig] | None = None
_SETS_TO_WIN: int = 1


def _init_worker_corpus(pool: list[BotConfig], sets_to_win: int) -> None:
    global _POOL_CFGS, _SETS_TO_WIN
    _POOL_CFGS = pool
    _SETS_TO_WIN = sets_to_win


def _play_topk_job(args: tuple[int, int]) -> list[np.ndarray]:
    i, seed = args
    assert _POOL_CFGS is not None
    rng = random.Random(seed)
    cfg0 = rng.choice(_POOL_CFGS)
    cfg1 = rng.choice(_POOL_CFGS)
    mt, winner = tokenize_config_match(cfg0, cfg1, rng.randrange(1 << 30), dealer=i % 4,
                                       sets_to_win=_SETS_TO_WIN)
    return [np.asarray(mt.streams[s], dtype=np.int16) for s in range(4)]


def _play_diverse_job(args: tuple[int, int, bool]) -> list[np.ndarray]:
    i, seed, winners_only = args
    assert _POOL_CFGS is not None
    rng = random.Random(seed)
    seat_cfgs = [rng.choice(_POOL_CFGS) for _ in range(4)]
    mt, winner = tokenize_config_match_seats(seat_cfgs, rng.randrange(1 << 30), dealer=i % 4,
                                             sets_to_win=_SETS_TO_WIN)
    seats = [s for s in range(4) if team_of(s) == winner] if winners_only else range(4)
    return [np.asarray(mt.streams[s], dtype=np.int16) for s in seats]


def generate_topk_corpus(top_configs: list[BotConfig], n_matches: int, seed: int = 0,
                         sets_to_win: int = 1, winners_only: bool = False,
                         jobs: int | None = None) -> list[np.ndarray]:
    """Play `n_matches` between random top config pairs; return per-seat int16 streams.

    4 streams per match by default, or just the winning team's 2 when `winners_only`.
    Parallelises across CPU cores via ProcessPoolExecutor.
    """
    if not top_configs:
        raise ValueError("need at least one top config to generate a corpus")
    workers = jobs if jobs is not None else (os.cpu_count() or 1)
    rng = random.Random(seed)
    job_args = [(i, rng.randrange(1 << 30)) for i in range(n_matches)]
    chunk = max(1, n_matches // (workers * 8))
    streams: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker_corpus,
                             initargs=(top_configs, sets_to_win)) as ex:
        for batch in ex.map(_play_topk_job, job_args, chunksize=chunk):
            if winners_only:
                streams.extend(batch[:2])
            else:
                streams.extend(batch)
    return streams


def generate_diverse_corpus(pool: list[BotConfig], n_matches: int, seed: int = 0,
                            sets_to_win: int = 1, winners_only: bool = False,
                            jobs: int | None = None) -> list[np.ndarray]:
    """Play `n_matches` where every match seats FOUR independently-sampled configs from `pool`.

    Unlike `generate_topk_corpus` (one config per *team*), this varies *teammates* too: seat 0 and
    seat 2 can be different styles. Combined with a broad `pool`, every transcript is a fresh mix
    of partners and opponents. Parallelises across CPU cores via ProcessPoolExecutor.
    """
    if not pool:
        raise ValueError("need at least one config to generate a corpus")
    workers = jobs if jobs is not None else (os.cpu_count() or 1)
    rng = random.Random(seed)
    job_args = [(i, rng.randrange(1 << 30), winners_only) for i in range(n_matches)]
    chunk = max(1, n_matches // (workers * 8))
    streams: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker_corpus,
                             initargs=(pool, sets_to_win)) as ex:
        for batch in ex.map(_play_diverse_job, job_args, chunksize=chunk):
            streams.extend(batch)
    return streams

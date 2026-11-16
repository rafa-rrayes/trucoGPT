"""Play a sampled config-vs-config tournament and rank every config by win-rate.

Each config is guaranteed `games_per_config` matches against random opponents, alternating which
team's seats it occupies so seat/dealer position can't bias its score. Matches are independent and
torch-free, so they fan out across processes; the parent just tallies wins. The win-rate ranking
is the whole point: it, not our assumptions, decides which configs are worth imitating.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

from ..agents import BotConfig
from .configs import sample_configs
from .match import play_config_match

# A match job: (team-0 config index, team-1 config index, match seed, dealer).
_Job = tuple[int, int, int, int]

_WORKER_CONFIGS: list[BotConfig] | None = None  # set per worker process by the pool initializer
_WORKER_SETS: int = 1                            # sets-to-win for every match this run

# Knob order the Rust engine's `BotConfig::from_slice` expects (truco_fast/src/engine.rs). This is
# the dataclass field order; pinned explicitly so a field reorder can't silently desync the layout.
_ROW_FIELDS = (
    "bluff", "signal_prob", "aggression", "p_lie", "p_signal_late", "p_signal_partial",
    "p_contradict", "p_ignore_partner", "p_overtrump", "p_burn", "p_fold_won", "p_overkill",
    "p_bad_facedown", "p_bad_eleven",
)


def _config_to_row(cfg: BotConfig) -> list[float]:
    """Flatten a BotConfig to the 14-float row the Rust engine consumes."""
    return [float(getattr(cfg, name)) for name in _ROW_FIELDS]


def _rust_backend():
    """Return the compiled `truco_fast` module if importable, else None (use Python fallback)."""
    try:
        import truco_fast
    except ImportError:
        return None
    # The repo's `truco_fast/` crate directory is importable as an empty namespace package
    # when the extension isn't built, so check for a real symbol.
    return truco_fast if hasattr(truco_fast, "run_tournament") else None


@dataclass
class TournamentResult:
    configs: list[BotConfig]
    wins: list[int]
    games: list[int]
    win_rate: list[float]
    ranking: list[int]            # config indices, best win-rate first

    def top(self, k: int) -> list[BotConfig]:
        return [self.configs[i] for i in self.ranking[:k]]


def build_schedule(n: int, games_per_config: int, seed: int) -> list[_Job]:
    """Random balanced pairings: each config gets `games_per_config` matches, seats alternated."""
    rng = random.Random(seed ^ 0x5CED)
    sched: list[_Job] = []
    ms = seed
    for i in range(n):
        for g in range(games_per_config):
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1                              # any opponent but itself
            a, b = (i, j) if g % 2 == 0 else (j, i)  # alternate the team-0 (seats 0,2) occupant
            sched.append((a, b, ms, ms % 4))
            ms += 1
    return sched


def _init_worker(configs: list[BotConfig], sets_to_win: int = 1) -> None:
    global _WORKER_CONFIGS, _WORKER_SETS
    _WORKER_CONFIGS = configs
    _WORKER_SETS = sets_to_win


def _play_job(job: _Job) -> tuple[int, int, int]:
    """Run one scheduled match; return (team0 config idx, team1 config idx, winning team)."""
    a, b, seed, dealer = job
    assert _WORKER_CONFIGS is not None
    winner = play_config_match(_WORKER_CONFIGS[a], _WORKER_CONFIGS[b], seed, dealer, _WORKER_SETS)
    return a, b, winner


def _execute(configs: list[BotConfig], sched: list[_Job], jobs: int, progress: bool,
             sets_to_win: int):
    """Yield (a, b, winner) for every scheduled match, serially or across a process pool."""
    if jobs == 1:
        _init_worker(configs, sets_to_win)
        for n, job in enumerate(sched, 1):
            yield _play_job(job)
            if progress and n % max(1, len(sched) // 20) == 0:
                print(f"  …{n}/{len(sched)} matches", flush=True)
        return
    from concurrent.futures import ProcessPoolExecutor

    chunk = max(1, len(sched) // (jobs * 8))
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                             initargs=(configs, sets_to_win)) as ex:
        done = 0
        for res in ex.map(_play_job, sched, chunksize=chunk):
            done += 1
            if progress and done % max(1, len(sched) // 20) == 0:
                print(f"  …{done}/{len(sched)} matches", flush=True)
            yield res


def run_tournament(n_configs: int, games_per_config: int, *, seed: int = 0,
                   jobs: int | None = None, progress: bool = True,
                   sets_to_win: int = 1, backend: str = "auto") -> TournamentResult:
    """Sample `n_configs`, play the tournament, and return configs ranked by win-rate.

    `backend`: "auto" uses the Rust `truco_fast` engine when importable (≈10-50× faster per core)
    and falls back to the Python process pool otherwise; "rust"/"python" force one path. Both are
    deterministic given the same seed; the two are statistically — not bit — equivalent (the Rust
    engine uses its own PRNG, so individual pairings differ but per-config win-rates agree).
    """
    configs = sample_configs(n_configs, seed)
    n = len(configs)

    fast = _rust_backend() if backend in ("auto", "rust") else None
    if backend == "rust" and fast is None:
        raise RuntimeError("backend='rust' requested but the truco_fast extension is not built")

    if fast is not None:
        total = n * games_per_config
        if progress:
            sets_note = f", best-of-{2 * sets_to_win - 1} sets" if sets_to_win > 1 else ""
            print(f"tournament: {n} configs × {games_per_config} games = {total} matches "
                  f"on the Rust engine (all cores){sets_note}…", flush=True)
        rows = [_config_to_row(c) for c in configs]
        wins_t, games_t = fast.run_tournament(rows, games_per_config, seed, sets_to_win)
        wins, games = list(wins_t), list(games_t)
    else:
        sched = build_schedule(n, games_per_config, seed)
        jobs = jobs if jobs is not None else (os.cpu_count() or 1)
        if progress:
            sets_note = f", best-of-{2 * sets_to_win - 1} sets" if sets_to_win > 1 else ""
            print(f"tournament: {n} configs × {games_per_config} games = {len(sched)} matches "
                  f"on {jobs} job(s){sets_note}…", flush=True)
        wins = [0] * n
        games = [0] * n
        for a, b, winner in _execute(configs, sched, jobs, progress, sets_to_win):
            wins[a if winner == 0 else b] += 1
            games[a] += 1
            games[b] += 1

    win_rate = [wins[k] / games[k] if games[k] else 0.0 for k in range(n)]
    ranking = sorted(range(n), key=lambda k: (win_rate[k], wins[k]), reverse=True)
    return TournamentResult(configs, wins, games, win_rate, ranking)

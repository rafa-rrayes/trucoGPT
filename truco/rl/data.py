"""Generate the supervised pretraining corpus: per-seat token streams of heuristic matches.

Stage 1 of training is plain next-token prediction — "a GPT that predicts the next token". Its
fuel is a pile of legal, sensible games, which we get by letting the canonical heuristic bot play
itself many times and tokenizing each match into its four per-seat streams (module 1). Every
stream is a self-contained transcript (`BOS … EOS`) of exactly what one seat legally saw.

Streams are stored flat (one concatenated int array + per-stream lengths) so the corpus is a
single compact `.npz`.
"""

from __future__ import annotations

import random

import numpy as np

from .. import engine
from ..agents import HeuristicAgent
from ..observation import observe
from .tokenizer import MatchTokenizer


def play_heuristic_match(seed: int, dealer: int = 0) -> MatchTokenizer:
    """Drive one full canonical-heuristic self-play match and return the populated tokenizer."""
    rng = random.Random(seed)
    state = engine.new_match(rng, dealer=dealer)
    agents = {s: HeuristicAgent(rng=random.Random(rng.random())) for s in range(4)}

    mt = MatchTokenizer()
    mt.start(state)
    while not state.done:
        actor = state.current_actor
        mt.begin_turn(actor)
        action = agents[actor].act(observe(state, actor))
        state, events = engine.apply_action(state, action, rng)
        mt.ingest(state, events)
    return mt


def generate_corpus(n_matches: int, seed: int = 0) -> list[np.ndarray]:
    """Play `n_matches` heuristic matches; return 4 per-seat streams each as an int16 array."""
    rng = random.Random(seed)
    streams: list[np.ndarray] = []
    for i in range(n_matches):
        mt = play_heuristic_match(rng.randrange(1 << 30), dealer=i % 4)
        for s in range(4):
            streams.append(np.asarray(mt.streams[s], dtype=np.int16))
    return streams


def save_corpus(streams: list[np.ndarray], path: str) -> None:
    """Persist streams flat: one concatenated array plus the per-stream lengths."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat = np.concatenate(streams) if streams else np.zeros(0, dtype=np.int16)
    lengths = np.asarray([len(s) for s in streams], dtype=np.int32)
    np.savez(path, flat=flat, lengths=lengths)


def load_corpus(path: str) -> list[np.ndarray]:
    """Inverse of `save_corpus`: split the flat array back into per-stream arrays."""
    data = np.load(path)
    flat, lengths = data["flat"], data["lengths"]
    out, off = [], 0
    for n in lengths:
        out.append(flat[off:off + n])
        off += int(n)
    return out


def _main() -> int:
    from .config import Config

    cfg = Config()
    print(f"generating {cfg.corpus_matches} heuristic matches "
          f"({cfg.corpus_matches * 4} streams)…")
    streams = generate_corpus(cfg.corpus_matches, seed=cfg.seed)
    save_corpus(streams, cfg.corpus_path)
    tokens = sum(len(s) for s in streams)
    print(f"wrote {cfg.corpus_path}: {len(streams)} streams, {tokens} tokens "
          f"({tokens / len(streams):.0f} avg).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

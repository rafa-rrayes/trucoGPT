"""Dump the per-seat token sequences of one real heuristic-vs-heuristic match to a file.

Drives the real engine for a single seeded match, tokenizes it with `MatchTokenizer`,
and writes every seat's stream to a text file in three views:

  * the raw token ids (exactly what a model would receive),
  * a decoded, one-token-per-line listing (id -> human-readable name),
  * a compact, whitespace-grouped rendering for eyeballing the flow.

Run: `uv run python scripts/dump_match.py [SEED] [OUT_PATH]`
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

SEATS = range(4)
SEAT_LABEL = {0: "seat 0 (you)", 1: "seat 1 (right)", 2: "seat 2 (partner)", 3: "seat 3 (left)"}

# Tokens that open a logical group — start a fresh line before them when rendering compactly.
_BREAK_BEFORE = {
    vocab.TO_ACT, vocab.HAND_START, vocab.ROUND_END, vocab.HAND_END, vocab.SET_END,
    vocab.MATCH_END, vocab.REDEAL, vocab.ELEVEN_START, vocab.ELEVEN_DECISION,
    vocab.BY_RIGHT, vocab.BY_PARTNER, vocab.BY_LEFT, vocab.EOS,
}


def play_match(seed: int) -> MatchTokenizer:
    """Drive one full heuristic-vs-heuristic match, returning the populated tokenizer."""
    rng = random.Random(seed)
    state = engine.new_match(rng, dealer=seed % 4)
    agents = {s: HeuristicAgent(rng=random.Random(rng.random())) for s in SEATS}

    mt = MatchTokenizer()
    mt.start(state)
    while not state.done:
        actor = state.current_actor
        mt.begin_turn(actor)
        action = agents[actor].act(observe(state, actor))
        state, events = engine.apply_action(state, action, rng)
        mt.ingest(state, events)
    return mt


def _compact(stream: list[int]) -> str:
    """Group a stream onto readable lines, breaking before each logical-group opener."""
    lines: list[list[str]] = [[]]
    for t in stream:
        if t in _BREAK_BEFORE and lines[-1]:
            lines.append([])
        lines[-1].append(vocab.token_name(t))
    return "\n".join("  " + " ".join(line) for line in lines if line)


def dump(seed: int, out_path: Path) -> None:
    mt = play_match(seed)
    parts: list[str] = []
    parts.append(f"Token sequences of one real match (seed={seed}, "
                 f"dealer=seat {seed % 4}), heuristic vs heuristic.")
    parts.append(f"Vocabulary size: {vocab.VOCAB_SIZE} tokens. "
                 f"Stream lengths: " + ", ".join(
                     f"{SEAT_LABEL[s]}={len(mt.streams[s])}" for s in SEATS) + ".")
    parts.append("")

    for s in SEATS:
        stream = mt.streams[s]
        parts.append("=" * 78)
        parts.append(f"{SEAT_LABEL[s]} — {len(stream)} tokens")
        parts.append("=" * 78)
        parts.append("")
        parts.append("--- raw token ids ---")
        parts.append(" ".join(str(t) for t in stream))
        parts.append("")
        parts.append("--- decoded (grouped) ---")
        parts.append(_compact(stream))
        parts.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path}  ({sum(len(mt.streams[s]) for s in SEATS)} tokens across 4 seats)")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent.parent / "samples" / f"match_seed{seed}_tokens.txt")
    dump(seed, out)

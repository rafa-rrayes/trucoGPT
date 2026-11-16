"""The shipped checkpoint can drive a full match through the CLI loop."""

import random
from pathlib import Path

import pytest

from truco import engine
from truco.agents import PolicyAgent
from truco.cli.play import play_match

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "rl_legality.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoint not present")
def test_policy_agent_plays_full_match():
    rng = random.Random(3)
    bot = PolicyAgent(CKPT)
    state = engine.new_match(rng, dealer=0)
    winner = play_match(state, {s: bot for s in range(4)}, rng, out=lambda *_: None)
    assert winner in (0, 1)

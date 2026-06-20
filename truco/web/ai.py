"""AI players for the web platform.

Two kinds of computer player can fill a seat:

  * ``"model"``     — the champion ``TrucoGPT`` transformer, driven from a per-seat token
                      stream produced by :class:`~truco.rl.tokenizer.MatchTokenizer`.
  * ``"heuristic"`` — the rule-based :class:`~truco.agents.HeuristicAgent`.

The transformer is loaded **once** at process start and shared across every room; its
forward passes are batch-1 and cheap, but we still serialise them with a lock so several
rooms hitting the model at the same time is safe. Heuristic players are tiny per-seat
objects created by each room.

Both paths ultimately return a stable action-space *index* (0..93, see
:mod:`truco.rl.action_space`), which the engine layer turns back into an ``Action``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch

from ..rl import vocab
from ..rl.action_space import ACTION_SIZE, action_index
from ..rl.model import load
from ..rl.rollout import batch_last_outputs

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CKPT = _REPO_ROOT / "checkpoints" / "rl_legality.pt"

# AI difficulty / personality presets exposed in the UI.
AI_KINDS = ("model", "heuristic")


class ModelEngine:
    """The shared, process-wide transformer policy.

    ``pick(stream, legal)`` returns the strongest *legal* action index by reading the
    next-token logits at the end of ``stream`` (which must end at the actor's ``TO_ACT``
    prompt) and restricting the argmax to the legal action block.
    """

    def __init__(self, ckpt_path: str | Path | None = None, device: str = "cpu") -> None:
        ckpt_path = str(ckpt_path or _DEFAULT_CKPT)
        self.device = device
        self.net, self.meta = load(ckpt_path, map_location=device)
        self.net = self.net.to(device).eval()
        self.block_size = self.net.cfg.block_size
        self.name = Path(ckpt_path).name
        self._lock = threading.Lock()

    @torch.no_grad()
    def pick(self, stream: list[int], legal: list[int]) -> int:
        """Greedy legal move for the actor whose stream ends in TO_ACT."""
        if not legal:
            raise ValueError("ModelEngine.pick called with no legal actions")
        legal_set = set(legal)
        with self._lock:
            lm, _ = batch_last_outputs(self.net, [stream], self.block_size, self.device)
            order = torch.argsort(lm[0], descending=True).tolist()
        for tok in order:
            a = tok - vocab.MOVE_BASE
            if 0 <= a < ACTION_SIZE and a in legal_set:
                return a
        return legal[0]  # nothing in the action block was legal (shouldn't happen)


# --------------------------------------------------------------------------- singleton

_ENGINE: ModelEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_model_engine(ckpt_path: str | Path | None = None, device: str = "cpu") -> ModelEngine:
    """Return the shared ModelEngine, loading it on first use."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = ModelEngine(ckpt_path, device)
    return _ENGINE


def heuristic_action_index(agent, obs) -> int:
    """Run a HeuristicAgent on an observation and return its action-space index."""
    return action_index(agent.act(obs))

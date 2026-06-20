"""Unit tests for TrucoGPT: output shapes, action-masking, and checkpoint round-trip."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import truco`

import torch

from truco.rl import vocab
from truco.rl.action_space import ACTION_SIZE
from truco.rl.config import Config
from truco.rl.model import TrucoGPT, action_logits, load, save


def _tiny_cfg() -> Config:
    return Config(block_size=64, d_model=32, n_layer=2, n_head=2)


def test_forward_shapes() -> None:
    cfg = _tiny_cfg()
    model = TrucoGPT(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (3, 20))
    lm_logits, value = model(tokens)
    assert lm_logits.shape == (3, 20, vocab.VOCAB_SIZE)
    assert value.shape == (3, 20)
    assert torch.isfinite(lm_logits).all() and torch.isfinite(value).all()


def test_context_is_left_truncated() -> None:
    cfg = _tiny_cfg()
    model = TrucoGPT(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 50))
    lm_logits, value = model(tokens)
    # The model keeps only the most recent block_size positions.
    assert lm_logits.shape == (1, cfg.block_size, vocab.VOCAB_SIZE)
    assert value.shape == (1, cfg.block_size)


def test_action_logits_masks_illegal_actions() -> None:
    cfg = _tiny_cfg()
    model = TrucoGPT(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 10))
    lm_logits, _ = model(tokens)
    last = lm_logits[:, -1, :]                          # [1, vocab] at the decision position

    mask = torch.zeros(1, ACTION_SIZE)
    legal = [0, 5, 88, 93]                              # an arbitrary legal subset
    for i in legal:
        mask[0, i] = 1.0

    a = action_logits(last, mask, cfg.move_base, cfg.action_size)
    assert a.shape == (1, ACTION_SIZE)

    # Illegal actions are crushed; legal ones keep their real (finite, larger) value.
    illegal = [i for i in range(ACTION_SIZE) if i not in legal]
    assert (a[0, illegal] <= -1e8).all()
    assert (a[0, legal] > -1e8).all()

    # The resulting distribution is valid and supported only on legal actions.
    probs = torch.softmax(a, dim=-1)
    assert torch.isfinite(probs).all()
    assert abs(probs.sum().item() - 1.0) < 1e-5
    assert probs[0, illegal].sum().item() < 1e-6

    # Entropy must stay finite (the reason we mask with a large finite value, not -inf).
    dist = torch.distributions.Categorical(logits=a)
    assert torch.isfinite(dist.entropy()).all()


def test_save_load_round_trip() -> None:
    cfg = _tiny_cfg()
    model = TrucoGPT(cfg)
    model.eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 15))
    with torch.no_grad():
        before, _ = model(tokens)

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "m.pt")
        save(model, path, meta={"note": "test"})
        reloaded, meta = load(path)

    assert meta == {"note": "test"}
    assert reloaded.cfg.arch() == cfg.arch()
    with torch.no_grad():
        after, _ = reloaded(tokens)
    assert torch.allclose(before, after, atol=1e-6)


if __name__ == "__main__":
    test_forward_shapes()
    test_context_is_left_truncated()
    test_action_logits_masks_illegal_actions()
    test_save_load_round_trip()
    print("PASS — model shapes, masking, and save/load round-trip all hold.")

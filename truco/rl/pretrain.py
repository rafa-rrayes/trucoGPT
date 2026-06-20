"""Stage 1 — supervised next-token pretraining of TrucoGPT on the heuristic corpus.

Standard causal-LM training: sample windows of token streams, predict each next token, minimise
cross-entropy over the full 188-token vocabulary. Because the action tokens are a slice of that
vocabulary, this single objective simultaneously teaches the model the *grammar* of legal Truco
(structural tokens, deals, results) and a first, heuristic-like *policy* at decision points — the
warm start the RL stage fine-tunes.

Padding uses `vocab.PAD` (id 0, which never occurs in a real stream) with `ignore_index`, so
padded positions contribute no loss. Right-padding is safe under causal attention: real positions
never attend to trailing pad.

Run: `uv run python -m truco.rl.pretrain` (expects a corpus from `truco.rl.data`).
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from . import vocab
from .config import Config
from .data import load_corpus
from .model import TrucoGPT, pick_device, release_memory, save


def make_batch(streams: list[np.ndarray], batch: int, block_size: int,
               rng: random.Random, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch` stream windows, right-pad to a common length, split into (inputs, targets)."""
    window = block_size + 1
    seqs = []
    for _ in range(batch):
        s = np.asarray(streams[rng.randrange(len(streams))], dtype=np.int64)
        if len(s) > window:                                    # random window of long streams
            start = rng.randrange(0, len(s) - window + 1)
            s = s[start:start + window]
        seqs.append(s)
    maxlen = max(len(s) for s in seqs)
    arr = np.full((batch, maxlen), vocab.PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = s
    t = torch.from_numpy(arr).to(device)
    return t[:, :-1], t[:, 1:]                                 # inputs, next-token targets


@torch.no_grad()
def token_accuracy(net: TrucoGPT, streams: list[np.ndarray], cfg: Config,
                   device: str, n_batches: int = 20) -> float:
    """Fraction of next-token predictions that match, over non-pad targets in sampled batches."""
    net.eval()
    rng = random.Random(12345)
    correct = total = 0
    _amp = device == "cuda"
    for _ in range(n_batches):
        x, y = make_batch(streams, cfg.pretrain_batch, cfg.block_size, rng, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=_amp):
            logits, _ = net(x)
        pred = logits.argmax(dim=-1)
        keep = y != vocab.PAD
        correct += int(((pred == y) & keep).sum().item())
        total += int(keep.sum().item())
    net.train()
    return correct / max(total, 1)


def _split(streams: list[np.ndarray], val_frac: float, seed: int):
    idx = list(range(len(streams)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(streams) * val_frac))
    val = [streams[i] for i in idx[:n_val]]
    train = [streams[i] for i in idx[n_val:]]
    return train, val


def pretrain(cfg: Config, streams: list[np.ndarray] | None = None,
             device: str = "cpu", log_every: int = 100) -> TrucoGPT:
    """Train a fresh TrucoGPT by next-token prediction; checkpoint to `cfg.pretrain_ckpt`."""
    if streams is None:
        streams = load_corpus(cfg.corpus_path)
    train, val = _split(streams, cfg.pretrain_val_frac, cfg.seed)
    print(f"pretrain: {len(train)} train / {len(val)} val streams; "
          f"model d_model={cfg.d_model} n_layer={cfg.n_layer}")

    net = TrucoGPT(cfg).to(device)
    if device == "cuda":
        net = torch.compile(net)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.pretrain_lr)
    rng = random.Random(cfg.seed)
    _amp = device == "cuda"

    for step in range(1, cfg.pretrain_steps + 1):
        x, y = make_batch(train, cfg.pretrain_batch, cfg.block_size, rng, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=_amp):
            logits, _ = net(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1),
                                   ignore_index=vocab.PAD)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 1 or step % log_every == 0:
            acc = token_accuracy(net, val, cfg, device)
            release_memory(device)                        # release cached blocks (matters on MPS)
            print(f"  step {step:5d}/{cfg.pretrain_steps}  loss {loss.item():.4f}  "
                  f"val_token_acc {acc:.3f}")

    save(net, cfg.pretrain_ckpt, meta={"stage": "pretrain", "steps": cfg.pretrain_steps,
                                       "val_token_acc": token_accuracy(net, val, cfg, device)})
    print(f"saved {cfg.pretrain_ckpt}")
    return net


def _main() -> int:
    cfg = Config()
    device = pick_device()
    print(f"device: {device}")
    pretrain(cfg, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

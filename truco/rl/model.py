"""TrucoGPT — a small decoder-only transformer over a seat's token stream.

It reads a per-seat token stream (`tokenizer.MatchTokenizer`) and, at every position, predicts a
distribution over the next token. Two heads sit on the shared trunk:

  * the **LM / policy head** — next-token logits over the whole 188-token vocabulary. Trained by
    supervised cross-entropy during pretraining; at a `TO_ACT` decision point its restriction to
    the 94-token action block (sliced out by `action_logits`) *is* the policy.
  * the **value head** — a scalar per position. Currently unused by training: REINFORCE uses a
    batch-mean baseline (see `reinforce.py`), not a learned value. Kept on the module so existing
    checkpoints round-trip and a critic can be reintroduced without an arch change.

The crucial design fact (see `vocab.py`): the action tokens are a contiguous slice of the
vocabulary, `[MOVE_BASE : MOVE_BASE + ACTION_SIZE]`. So the policy is literally a slice of the
next-token head — pretraining the LM head directly warms up the policy, and RL only has to learn
the value head from scratch.

This module is the one place (besides the trainers) that imports torch; the base game stays
torch-free.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

_MASK_FILL = -1e9  # finite (not -inf) so masked entropy/logsumexp never produce NaNs


def pick_device(prefer: str | None = None) -> str:
    """Resolve a torch device string: explicit arg > `$TRUCO_DEVICE` > mps (Apple GPU) > cuda > cpu.

    Auto-selects Apple's Metal backend on a Mac. Set `TRUCO_DEVICE=cpu` to override per stage
    (the batch-1 rollout/eval can be faster on CPU than MPS), or pass `prefer` explicitly.
    """
    import os

    choice = prefer or os.environ.get("TRUCO_DEVICE")
    if choice:
        return choice
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def release_memory(device: str) -> None:
    """Hand cached blocks back to the OS. MPS/CUDA allocators hold freed buffers otherwise.

    Important on MPS specifically: it uses *unified* (system) memory and its caching allocator
    keeps a separate pool per allocation size. With variable-length batches that pool set grows,
    so the resident high-water mark climbs across iterations unless we empty it.
    """
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


class _CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)  # [B, nh, T, hd]
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class _Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = _CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TrucoGPT(nn.Module):
    """Decoder-only transformer; `forward(tokens) -> (lm_logits, value)`."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.value_head = nn.Linear(cfg.d_model, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Shared trunk: tokens [B, T] -> final hidden states [B, T, d_model] (post final LN).

        Split out so callers that only need a few positions (the RL/eval decision points) can
        apply the heads to just those, instead of materialising logits over the whole sequence.
        """
        if tokens.size(1) > self.cfg.block_size:          # keep the most recent context
            tokens = tokens[:, -self.cfg.block_size:]
        T = tokens.size(1)
        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`tokens`: LongTensor [B, T] -> (lm_logits [B, T, vocab], value [B, T])."""
        x = self.features(tokens)
        return self.lm_head(x), self.value_head(x).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def action_logits(lm_logits: torch.Tensor, mask: torch.Tensor,
                  move_base: int, action_size: int) -> torch.Tensor:
    """Slice the next-token logits down to the action block and apply the legality mask.

    `lm_logits` is `[..., vocab]` (typically read at a single TO_ACT position) and `mask` is
    `[..., action_size]` of 0/1. Illegal actions are pushed to a large finite negative so they
    get ~zero probability while keeping softmax/entropy NaN-free.
    """
    a = lm_logits[..., move_base:move_base + action_size]
    return a.masked_fill(mask == 0, _MASK_FILL)


def save(model: TrucoGPT, path: str, meta: dict | None = None) -> None:
    """Persist arch + weights (+ optional metadata) so `load` can rebuild the exact model."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"arch": model.cfg.arch(), "state_dict": model.state_dict(),
                "meta": meta or {}}, path)


def load(path: str, map_location: str = "cpu") -> tuple[TrucoGPT, dict]:
    """Inverse of `save`: returns `(model, meta)`, with the model rebuilt at the saved shape."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = Config(**ckpt["arch"])
    model = TrucoGPT(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})

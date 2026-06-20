"""Hyperparameters for the TrucoGPT model and its two training stages.

One `Config` carries everything: the architecture (saved inside each checkpoint so a model
reloads at the right shape), the supervised next-token pretraining knobs, and the REINFORCE
fine-tuning knobs. The I/O dimensions are pinned to the tokenizer's vocabulary so the model and
the streams can never silently disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import vocab
from .action_space import ACTION_SIZE

# Fields that describe the *network shape*; only these need to match for a checkpoint to load.
ARCH_FIELDS = ("vocab_size", "move_base", "action_size", "block_size",
               "d_model", "n_layer", "n_head", "dropout")


@dataclass
class Config:
    # --- I/O, fixed by the tokenizer (do not change without rebuilding the vocab) ---
    vocab_size: int = vocab.VOCAB_SIZE        # 188 — full token alphabet
    move_base: int = vocab.MOVE_BASE          # 93 — first action token; policy = LM logits here
    action_size: int = ACTION_SIZE            # 94 — width of the action / legal-mask block

    # --- architecture (a small nanoGPT) ---
    block_size: int = 512                    # context length; longer streams are left-truncated
    d_model: int = 343                       # 343 = 7×49 (divisible by n_head); ~10M params
    n_layer: int = 7
    n_head: int = 7
    dropout: float = 0.05

    # --- supervised next-token pretraining ---
    corpus_matches: int = 20_000                # matches to generate as the corpus (top-config self-play)
    pretrain_steps: int = 20_000                # big-arch run on a ~150M-token diverse corpus
    pretrain_batch: int = 32
    pretrain_lr: float = 3e-4
    pretrain_val_frac: float = 0.05           # held-out streams for token-accuracy

    # --- REINFORCE fine-tuning ---
    gamma: float = 0.99                       # discount over a match's reward timeline
    rl_lr: float = 1e-4                       # policy now gets the whole grad step (no value loss)
    rl_iters: int = 4000
    rl_batch_matches: int = 2                # matches rolled out per gradient step
    rl_minibatch: int = 1                    # decisions per forward chunk (grad-accumulated;
    #                                           pure memory/speed knob — does not change the update)
    entropy_coef: float = 0.001               # exploration bonus
    legality_coef: float = 0.0                # penalty on the UNMASKED policy's illegal prob. mass
    #                                           (0.0 ⇒ off, path byte-identical to before; >0 teaches
    #                                            the raw policy to be legal while masked play stays legal)
    reward_shaping: bool = True               # dense per-hand reward on top of terminal +-1
    lr_decay: str = "constant"                # "constant" | "cosine": anneal rl_lr → 0 over rl_iters

    # --- RL algorithm selector (default "reinforce" leaves the path above byte-identical) ---
    rl_algo: str = "reinforce"                # "reinforce" | "ppo"
    ppo_clip: float = 0.2                     # PPO surrogate clip range, 1±clip
    ppo_epochs: int = 4                       # optimiser passes over each rollout batch
    value_coef: float = 0.5                   # weight on the value-head MSE (PPO + use_value_baseline)
    use_value_baseline: bool = False          # subtract a learned V(s) in the advantage (else batch-mean)
    learner_seats: tuple[int, ...] = (0, 2)   # the team the learner controls (seat-relative net)
    eval_every: int = 25                      # RL iters between win-rate evals

    # --- evaluation ---
    eval_matches: int = 100                   # matches per win-rate measurement

    # --- bookkeeping ---
    seed: int = 0
    ckpt_dir: str = "checkpoints"
    pretrain_ckpt: str = "checkpoints/pretrain.pt"
    rl_ckpt: str = "checkpoints/rl.pt"
    corpus_path: str = "checkpoints/corpus.npz"

    def arch(self) -> dict:
        """Just the shape-defining fields — what a checkpoint must round-trip to reload."""
        return {k: getattr(self, k) for k in ARCH_FIELDS}


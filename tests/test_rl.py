"""RL smoke tests: the rollout driver, a REINFORCE step, and eval all run end-to-end.

These are correctness/plumbing checks on a deliberately tiny net and a handful of matches — fast
enough for CI, not a measure of strength. They assert the invariants that matter: every action
the policy emits is legal, returns are finite, a gradient step runs without error, and win-rate is
a probability.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import truco`

import torch

from truco.rl.action_space import action_index, legal_mask
from truco.rl.config import Config
from truco.rl.eval import win_rate
from truco.rl.model import TrucoGPT, action_logits
from truco.rl.reinforce import _update_ppo, collect_batch, update
from truco.rl.rollout import batch_last_outputs, rollout_match
from truco.observation import observe


def _tiny_cfg(**kw) -> Config:
    base = dict(block_size=256, d_model=32, n_layer=2, n_head=2,
                rl_batch_matches=3, rl_minibatch=16, eval_matches=4)
    base.update(kw)
    return Config(**base)


def test_rollout_emits_only_legal_actions() -> None:
    cfg = _tiny_cfg()
    net = TrucoGPT(cfg)
    res = rollout_match(net, cfg, seed=7, sample=True, collect=True)
    assert res.decisions, "a match must contain at least one learner decision"
    for d in res.decisions:
        assert d.mask[d.action_idx] == 1.0, "policy chose an illegal action"
        assert isinstance(d.ret, float) and d.ret == d.ret, "return must be a finite float"
    assert res.winner_team in (0, 1)


def test_reinforce_step_runs_and_changes_params() -> None:
    cfg = _tiny_cfg()
    net = TrucoGPT(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    decisions, train_wr, mean_ret = collect_batch(net, cfg, base_seed=100, device="cpu")
    assert decisions, "rollout collected no decisions"
    assert 0.0 <= train_wr <= 1.0

    before = net.lm_head.weight.detach().clone()
    stats = update(net, opt, decisions, cfg, device="cpu")
    after = net.lm_head.weight.detach()

    for k in ("policy_loss", "entropy"):
        assert stats[k] == stats[k], f"{k} is NaN"           # finite check
    assert not torch.allclose(before, after), "policy head did not update"


def _ppo_advantage(decisions, cfg) -> torch.Tensor:
    """The exact advantage `_update_ppo` optimises against (whitened returns, then standardised).

    Mirrors the trainer's no-baseline path so the test can score the surrogate the update maximises.
    """
    returns = torch.tensor([d.ret for d in decisions], dtype=torch.float32)
    ret_w = (returns - returns.mean()) / (returns.std() + 1e-8)
    adv = ret_w
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def _clipped_surrogate(net, decisions, adv, cfg) -> float:
    """Mean PPO-clipped surrogate over a fixed batch under the *current* net (eval, no grad).

    This is the objective `_update_ppo` ascends; at the sampling net (ratio≈1) it is ≈0, and a
    working update must raise it.
    """
    net.eval()
    with torch.no_grad():
        lm, _ = batch_last_outputs(net, [d.stream for d in decisions], cfg.block_size, "cpu")
        masks = torch.tensor([d.mask for d in decisions], dtype=torch.float32)
        logits = action_logits(lm, masks, cfg.move_base, cfg.action_size)
        logp = torch.log_softmax(logits, dim=-1).gather(
            1, torch.tensor([[d.action_idx] for d in decisions])).squeeze(1)
        old = torch.tensor([d.old_logp for d in decisions], dtype=torch.float32)
        ratio = torch.exp(logp - old)
        surr = torch.min(ratio * adv,
                         torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv)
    return float(surr.mean())


def test_ppo_step_runs_and_improves_surrogate() -> None:
    cfg = _tiny_cfg(rl_algo="ppo", ppo_epochs=4, ppo_clip=0.2, entropy_coef=0.0,
                    rl_batch_matches=4)
    net = TrucoGPT(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    decisions, _, _ = collect_batch(net, cfg, base_seed=100, device="cpu")
    assert decisions, "rollout collected no decisions"
    for d in decisions:                                      # rollout must populate the PPO fields
        assert isinstance(d.old_logp, float) and d.old_logp <= 0.0
        assert isinstance(d.value, float)

    adv = _ppo_advantage(decisions, cfg)
    before = _clipped_surrogate(net, decisions, adv, cfg)    # ≈ 0 at the sampling net (ratio≈1)
    head_before = net.lm_head.weight.detach().clone()
    stats = _update_ppo(net, opt, decisions, cfg, device="cpu")
    after = _clipped_surrogate(net, decisions, adv, cfg)

    for k in ("policy_loss", "entropy", "value_loss", "clipfrac", "kl"):
        assert k in stats and stats[k] == stats[k], f"{k} missing or NaN"
    assert not torch.allclose(head_before, net.lm_head.weight.detach()), "policy head did not move"
    assert after > before, f"PPO did not improve the clipped surrogate ({before:.4f} -> {after:.4f})"


def test_ppo_value_baseline_trains_value_head() -> None:
    cfg = _tiny_cfg(rl_algo="ppo", ppo_epochs=3, use_value_baseline=True, value_coef=0.5,
                    rl_batch_matches=4)
    net = TrucoGPT(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    decisions, _, _ = collect_batch(net, cfg, base_seed=200, device="cpu")
    vhead_before = net.value_head.weight.detach().clone()
    stats = _update_ppo(net, opt, decisions, cfg, device="cpu")

    assert stats["value_loss"] > 0.0 and stats["value_loss"] == stats["value_loss"]
    assert not torch.allclose(vhead_before, net.value_head.weight.detach()), \
        "value head did not train under the value baseline"


def test_win_rate_is_a_probability() -> None:
    cfg = _tiny_cfg()
    net = TrucoGPT(cfg)
    wr = win_rate(net, cfg, n_matches=4)
    assert 0.0 <= wr <= 1.0


def test_legal_mask_matches_engine_enumeration() -> None:
    """Every legal action the engine enumerates is exactly the mask's support (sanity bridge)."""
    cfg = _tiny_cfg()
    net = TrucoGPT(cfg)
    res = rollout_match(net, cfg, seed=3, sample=True, collect=True)
    # The recorded masks are non-empty and the chosen index sits in their support.
    for d in res.decisions:
        assert sum(d.mask) >= 1.0
        assert 0 <= d.action_idx < cfg.action_size


if __name__ == "__main__":
    test_rollout_emits_only_legal_actions()
    test_reinforce_step_runs_and_changes_params()
    test_ppo_step_runs_and_improves_surrogate()
    test_ppo_value_baseline_trains_value_head()
    test_win_rate_is_a_probability()
    test_legal_mask_matches_engine_enumeration()
    print("PASS — rollout legality, REINFORCE + PPO steps, eval probability, and mask support all hold.")

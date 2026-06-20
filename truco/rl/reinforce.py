"""Stage 2 — REINFORCE (policy gradient with a value baseline) fine-tuning of TrucoGPT.

The pretrained net already speaks legal Truco and imitates the heuristic. Now we push it to
*win*. Each iteration:

  1. **Roll out** a batch of full matches with the net controlling one team (seats `learner_seats`)
     and the `default` heuristic the other, *sampling* actions from the legal-masked policy. The
     rollout driver records one `Decision` per learner turn and the discounted return-to-go `G`
     for each (the AEC→single-agent credit assignment, see `rollout.py`).
  2. **Update** once. For every decision we recompute, *with gradients*, the policy log-prob at
     that TO_ACT position by re-running the net on the recorded stream:

         policy loss   =  -A.detach() · logπ(a)               (REINFORCE with a baseline)
         entropy bonus =  -H(π)                               (discourage early collapse)

     The advantage `A` is the discounted return-to-go `G` standardised across the batch — i.e. a
     *constant* (batch-mean) baseline. A constant baseline adds zero variance to the score-function
     estimator; it is detached and only scales logπ, never flowing gradient. There is no learned
     value baseline (a value head sharing the trunk could not fit the noisy discounted return, so it
     injected variance into the advantage *and* its MSE swamped the policy gradient), no PPO clip,
     and no GAE: this is plain REINFORCE, as chosen.

Decisions are processed in `rl_minibatch`-sized chunks whose gradients accumulate into a single
optimiser step per iteration — true batch-REINFORCE semantics at bounded memory. The recompute runs
in `eval()` (dropout off) so the log-probs come from the *same* policy that sampled the actions.

Run: `uv run python -m truco.rl.reinforce` (warm-starts from `cfg.pretrain_ckpt` if present).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import Config
from .eval import win_rate
from .model import TrucoGPT, action_logits, load, pick_device, release_memory, save
from .rollout import Decision, batch_last_outputs, rollout_matches


def _init_net(cfg: Config, device: str) -> TrucoGPT:
    """Warm-start from the pretrain checkpoint when present; else a fresh net."""
    if os.path.exists(cfg.pretrain_ckpt):
        net, meta = load(cfg.pretrain_ckpt, map_location=device)
        if net.cfg.arch() != cfg.arch():
            raise ValueError(f"pretrain arch {net.cfg.arch()} != config arch {cfg.arch()}")
        print(f"warm-start from {cfg.pretrain_ckpt} "
              f"(val_token_acc={meta.get('val_token_acc'):.3f})"
              if "val_token_acc" in meta else f"warm-start from {cfg.pretrain_ckpt}")
        return net.to(device)
    print("no pretrain checkpoint — training the policy from random init")
    return TrucoGPT(cfg).to(device)


def collect_batch(net: TrucoGPT, cfg: Config, base_seed: int,
                  device: str, opponent_configs=None,
                  opponent_net: TrucoGPT | None = None) -> tuple[list[Decision], float, float]:
    """Roll out `rl_batch_matches` sampled matches; return (decisions, win_rate, mean_return).

    The opposing team is `opponent_net` (a frozen self-play snapshot) if given, else drawn from
    `opponent_configs` (the strong-config pool), else the `default` heuristic.
    """
    net.eval()                                            # sampling forward (no dropout noise)
    n = cfg.rl_batch_matches
    results = rollout_matches(net, cfg, [base_seed + m for m in range(n)],
                              learner_seats=cfg.learner_seats, sample=True,
                              dealers=[m % 4 for m in range(n)],
                              reward_shaping=cfg.reward_shaping, collect=True, device=device,
                              opponent_configs=opponent_configs, opponent_net=opponent_net)
    decisions: list[Decision] = []
    wins = 0
    for res in results:
        decisions.extend(res.decisions)
        wins += int(res.learner_won)
    mean_ret = sum(d.ret for d in decisions) / max(len(decisions), 1)
    return decisions, wins / cfg.rl_batch_matches, mean_ret


def _illegal_mass(lm_logits: torch.Tensor, masks: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Summed probability the *unmasked* policy assigns to illegal actions, over the rows of a chunk.

    `action_logits` (used for sampling and the policy term) pushes illegal moves to ~-inf, so every
    move actually *played* is legal — the model can't pick an illegal one. This looks instead at the
    *raw*, unmasked action-block logits and returns how much probability mass would sit on illegal
    actions if the mask were lifted. Adding `cfg.legality_coef ·` this to the loss trains the raw
    policy to be legal on its own (so the mask becomes a pure safety net), while masked play keeps
    the win-rate objective untouched. The mass is in [0, 1] per row and its gradient fades to ~0 as
    the illegal mass does — gentle, self-limiting pressure that won't fight a confident legal policy.
    """
    raw = lm_logits[..., cfg.move_base:cfg.move_base + cfg.action_size]   # unmasked action logits
    return ((1.0 - masks) * torch.softmax(raw, dim=-1)).sum()


def update(net: TrucoGPT, opt: torch.optim.Optimizer, decisions: list[Decision],
           cfg: Config, device: str) -> dict:
    """One policy update over all `decisions`. Dispatches on `cfg.rl_algo`.

    Default `"reinforce"` runs the original single-step batch-REINFORCE below, untouched.
    `"ppo"` runs the clipped-surrogate, K-epoch update in `_update_ppo`.
    """
    if cfg.rl_algo == "ppo":
        return _update_ppo(net, opt, decisions, cfg, device)

    n = len(decisions)
    if n == 0:
        return {"policy_loss": 0.0, "entropy": 0.0}

    # Bucket by stream length: each minibatch then pads only to its own (often short) max instead
    # of up to block_size, slashing wasted compute/activation memory. REINFORCE sums over all
    # decisions and standardises the advantage globally, so reordering changes nothing numerically.
    decisions = sorted(decisions, key=lambda d: len(d.stream))
    returns = torch.tensor([d.ret for d in decisions], dtype=torch.float32, device=device)
    mb = cfg.rl_minibatch

    # Advantage = the return-to-go standardised across the batch: a *constant* (batch-mean) baseline.
    # A constant baseline adds zero variance to the score-function estimator, so this is strictly a
    # safe variance reduction. (We deliberately do not subtract a learned value head: it could not
    # fit the noisy discounted return — adding variance to the advantage — and its trunk-shared MSE
    # dominated the gradient, leaving the policy barely trained.)
    adv = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Recompute logπ per chunk (with grad); accumulate the scaled policy loss into one opt step.
    # Stay in eval() so dropout is OFF — the log-probs must come from the same policy that sampled
    # the actions during rollout (also eval()), else the score function is differentiated through a
    # stochastically different network.
    net.eval()
    opt.zero_grad()
    tot_pl = tot_ent = tot_illegal = 0.0
    for i in range(0, n, mb):
        chunk = decisions[i:i + mb]
        lm, _ = batch_last_outputs(net, [d.stream for d in chunk], cfg.block_size, device)
        masks = torch.tensor([d.mask for d in chunk], dtype=torch.float32, device=device)
        dist = Categorical(logits=action_logits(lm, masks, cfg.move_base, cfg.action_size))
        acts = torch.tensor([d.action_idx for d in chunk], dtype=torch.long, device=device)

        logp = dist.log_prob(acts)
        ent = dist.entropy()
        policy_loss = -(adv[i:i + len(chunk)] * logp).sum()
        illegal = _illegal_mass(lm, masks, cfg)           # diagnostic always; penalised only if coef>0
        loss = policy_loss - cfg.entropy_coef * ent.sum()
        if cfg.legality_coef:
            loss = loss + cfg.legality_coef * illegal
        (loss / n).backward()                             # graph for this chunk is freed here
        tot_pl += float(policy_loss.item())
        tot_ent += float(ent.sum().item())
        tot_illegal += float(illegal.item())
        release_memory(device)                            # keep peak ≈ one chunk, not the whole batch

    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return {"policy_loss": tot_pl / n, "entropy": tot_ent / n, "illegal_mass": tot_illegal / n}


def _update_ppo(net: TrucoGPT, opt: torch.optim.Optimizer, decisions: list[Decision],
                cfg: Config, device: str) -> dict:
    """PPO-clipped update: `ppo_epochs` passes over the batch, each a clipped surrogate against
    the rollout-time log-probs, with a standardised advantage and an optional learned value baseline.

    Why this over plain REINFORCE: the clip lets us reuse each (expensive-to-collect) batch for K
    gradient steps without the policy walking off — the ratio clamp caps how far any step moves the
    action probabilities, so high-variance Truco returns stop yanking the policy past its peak.

    Spaces (kept consistent so the value MSE and the policy term are comparably scaled):
      * `ret_w` — the return-to-go whitened across the batch. This is BOTH the value-head target and,
        when there is no learned baseline, the advantage itself (a zero-mean constant baseline,
        identical in spirit to the REINFORCE path).
      * advantage — `ret_w - V_old(s)` when `use_value_baseline` (the recorded no-grad value, already
        in whitened scale because it is trained on `ret_w`), else `ret_w`; then standardised.

    `old_logp`/`value` were captured at sampling time (also under `eval()`), so on epoch 0 the ratio
    is ≈1 and the first pass mirrors a REINFORCE step; later epochs diverge and the clip engages.
    Gradients accumulate across length-bucketed minibatches into one optimiser step per epoch —
    the same bounded-memory pattern as the REINFORCE path.
    """
    n = len(decisions)
    zero = {"policy_loss": 0.0, "entropy": 0.0, "value_loss": 0.0, "clipfrac": 0.0, "kl": 0.0}
    if n == 0:
        return zero

    decisions = sorted(decisions, key=lambda d: len(d.stream))   # pad to per-minibatch max (cheap)
    returns = torch.tensor([d.ret for d in decisions], dtype=torch.float32, device=device)
    old_logp = torch.tensor([d.old_logp for d in decisions], dtype=torch.float32, device=device)

    ret_w = (returns - returns.mean()) / (returns.std() + 1e-8)
    if cfg.use_value_baseline:
        values_old = torch.tensor([d.value for d in decisions], dtype=torch.float32, device=device)
        adv = ret_w - values_old
    else:
        adv = ret_w
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    mb, clip = cfg.rl_minibatch, cfg.ppo_clip
    net.eval()                                            # dropout OFF — match the sampling policy
    tot_pl = tot_ent = tot_vl = tot_clip = tot_kl = tot_illegal = 0.0
    for _ in range(cfg.ppo_epochs):
        opt.zero_grad()
        for i in range(0, n, mb):
            chunk = decisions[i:i + mb]
            sl = slice(i, i + len(chunk))
            lm, val = batch_last_outputs(net, [d.stream for d in chunk], cfg.block_size, device)
            masks = torch.tensor([d.mask for d in chunk], dtype=torch.float32, device=device)
            dist = Categorical(logits=action_logits(lm, masks, cfg.move_base, cfg.action_size))
            acts = torch.tensor([d.action_idx for d in chunk], dtype=torch.long, device=device)

            logp = dist.log_prob(acts)
            ent = dist.entropy()
            ratio = torch.exp(logp - old_logp[sl])
            surr = torch.min(ratio * adv[sl],
                             torch.clamp(ratio, 1 - clip, 1 + clip) * adv[sl])
            policy_loss = -surr.sum()

            illegal = _illegal_mass(lm, masks, cfg)       # diagnostic always; penalised only if coef>0
            loss = policy_loss - cfg.entropy_coef * ent.sum()
            if cfg.legality_coef:
                loss = loss + cfg.legality_coef * illegal
            if cfg.use_value_baseline:
                value_loss = F.mse_loss(val, ret_w[sl], reduction="sum")
                loss = loss + cfg.value_coef * value_loss
                tot_vl += float(value_loss.item())
            (loss / n).backward()                         # graph for this chunk freed here

            tot_pl += float(policy_loss.item())
            tot_ent += float(ent.sum().item())
            tot_illegal += float(illegal.item())
            with torch.no_grad():                         # diagnostics only
                tot_clip += float(((ratio - 1.0).abs() > clip).float().sum().item())
                tot_kl += float((old_logp[sl] - logp).sum().item())   # approx KL(old‖new)
            release_memory(device)

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    denom = n * cfg.ppo_epochs
    return {"policy_loss": tot_pl / denom, "entropy": tot_ent / denom,
            "value_loss": tot_vl / denom, "clipfrac": tot_clip / denom, "kl": tot_kl / denom,
            "illegal_mass": tot_illegal / denom}


def reinforce(cfg: Config, device: str = "cpu", eval_matches: int | None = None,
              opponent_configs=None, opponent_sampler=None, on_iter=None) -> TrucoGPT:
    """Fine-tune the policy by REINFORCE; checkpoint the best (by eval win-rate) to `cfg.rl_ckpt`.

    With `opponent_configs`, the learner trains against the strong-config pool and the *pool*
    win-rate is the selection metric; the vs-`default` win-rate is still printed for comparison.

    `opponent_sampler(it) -> dict` overrides the *training* opponent each iteration (eval is left
    on `opponent_configs` so the selection metric stays comparable): return `{"net": frozen_net}`
    for self-play, `{"configs": pool}` for the config pool, or `{}` for the default heuristic.
    `on_iter(it, net)` is called after every iteration — used by the self-play league to snapshot.
    """
    torch.manual_seed(cfg.seed)
    net = _init_net(cfg, device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.rl_lr)
    # Optional cosine anneal of the LR → 0 over the run. Phase-2 showed the policy peaks ~iter 120
    # then walks off it: a constant LR + a per-batch-standardised advantage (which never shrinks near
    # the optimum) keeps taking full-size noisy steps. Decaying the step size lets it settle at the peak.
    sched = (CosineAnnealingLR(opt, T_max=cfg.rl_iters, eta_min=0.0)
             if cfg.lr_decay == "cosine" else None)
    eval_n = eval_matches if eval_matches is not None else cfg.eval_matches
    vs_pool = opponent_configs is not None
    target = "pool" if vs_pool else "default"

    def evaluate() -> tuple[float, float]:
        """(selection win-rate, vs-default win-rate). They coincide when training vs default."""
        sel = win_rate(net, cfg, eval_n, device=device, opponent_configs=opponent_configs)
        ref = win_rate(net, cfg, eval_n, device=device) if vs_pool else sel
        return sel, ref

    base, base_ref = evaluate()
    extra = f" (vs default = {base_ref:.3f})" if vs_pool else ""
    print(f"start: win-rate vs {target} = {base:.3f}{extra} over {eval_n} matches")
    best = base

    for it in range(1, cfg.rl_iters + 1):
        seed = 2_000_000 + it * cfg.rl_batch_matches      # disjoint from eval seeds (1_000_000+)
        if opponent_sampler is not None:
            spec = opponent_sampler(it)                   # {"net":…} | {"configs":…} | {}
            opp_net, opp_cfgs = spec.get("net"), spec.get("configs")
        else:
            opp_net, opp_cfgs = None, opponent_configs
        otag = "self" if opp_net is not None else "pool" if opp_cfgs else "dflt"
        decisions, train_wr, mean_ret = collect_batch(net, cfg, seed, device,
                                                       opponent_configs=opp_cfgs,
                                                       opponent_net=opp_net)
        stats = update(net, opt, decisions, cfg, device)
        if sched is not None:
            sched.step()
        release_memory(device)                            # bound resident memory across iterations
        ppo_extra = (f"  vl {stats.get('value_loss', 0.0):+.4f}  "
                     f"clip {stats.get('clipfrac', 0.0):.2f}  kl {stats.get('kl', 0.0):+.4f}"
                     if cfg.rl_algo == "ppo" else "")
        otag_str = f"[{otag}] " if opponent_sampler is not None else ""
        illeg = f"  illeg {stats.get('illegal_mass', 0.0):.4f}" if cfg.legality_coef else ""
        print(f"  iter {it:4d}/{cfg.rl_iters}  {otag_str}train_wr {train_wr:.2f}  ret {mean_ret:+.3f}  "
              f"pl {stats['policy_loss']:+.4f}  H {stats['entropy']:.3f}{ppo_extra}{illeg}  "
              f"({len(decisions)} decisions)")

        if it % cfg.eval_every == 0 or it == cfg.rl_iters:
            wr, ref = evaluate()
            tag = ""
            if wr > best:
                best = wr
                save(net, cfg.rl_ckpt, meta={"stage": "reinforce", "iter": it, "win_rate": wr,
                                             "target": target, "win_rate_vs_default": ref})
                tag = "  *saved best*"
            extra = f"  (vs default {ref:.3f})" if vs_pool else ""
            print(f"  ── eval @ iter {it}: win-rate vs {target} = {wr:.3f}{extra}{tag}"
                  f"  (lr={opt.param_groups[0]['lr']:.2e})")

        if on_iter is not None:
            on_iter(it, net)

    print(f"done: best win-rate vs {target} = {best:.3f} (start was {base:.3f})")
    return net


def _main() -> int:
    cfg = Config()
    device = pick_device()
    print(f"device: {device}")
    reinforce(cfg, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

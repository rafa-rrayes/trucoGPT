"""The shared rollout driver: play matches, tokenizing as they go.

This is the single engine↔tokenizer loop used by *both* the REINFORCE trainer and the evaluator.
It runs full matches in which the learner net controls some seats (a whole team) and supplied
opponents control the rest, while a `MatchTokenizer` builds the learner seats' streams. At each
learner turn it feeds that seat's stream (ending in `TO_ACT`) to the net, reads the next-token
logits at the final position, restricts them to the legal action block, and picks a move
(sampling for RL, argmax for eval).

**Batched rollouts.** The net forward is the cost; for this tiny model at batch 1 it is pure
per-call overhead. So `rollout_matches` plays many independent matches *concurrently*: every
round it advances each live match through its opponent turns until that match is either finished
or waiting on a learner decision, then runs **one** batched net forward over all the waiting
matches at once (`_batched_pick`). Because the matches are independent and each keeps its own
`begin_turn → step → ingest` order, the per-match transcripts are identical to playing them one
at a time — and because attention is causal, the right-padding in the batch cannot reach a
stream's decision position, so each row's logits match the batch-1 result exactly. The only
difference from sequential play is the *order* in which the global RNG is consumed when sampling,
so sampled games differ run-to-run (argmax/eval is bit-for-bit identical). `rollout_match` is a
thin single-match wrapper over the same machinery.

Action *selection* runs under `no_grad`; the trainer recomputes log-probs/values with gradients
by re-running the net on the recorded `Decision.stream`. That separation keeps rollout cheap and
lets the trainer batch decisions across many matches.

Credit assignment (the AEC→single-agent reduction): rewards land at hand/match boundaries via
`TrucoEnv`. We record the learner team's immediate reward after every env step into one timeline,
then each decision's return-to-go is the discounted sum of that timeline from the decision's step
onward — i.e. a seat's "next state" is its next decision, several env steps later.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from ..agents import BotConfig, ConfigurableBot, HeuristicAgent
from ..env import TrucoEnv
from ..state import team_of
from . import vocab
from .action_space import action_from_index, legal_mask
from .config import Config
from .model import TrucoGPT, action_logits, release_memory
from .tokenizer import MatchTokenizer

# Default cap on matches played concurrently in one batched group. Bounds the peak forward batch
# (and thus activation memory) while still turning thousands of batch-1 calls into a handful of
# batched ones. Eval (hundreds of matches) is chunked into groups of this size automatically.
DEFAULT_MAX_CONCURRENT = 64


@dataclass
class Decision:
    """One learner move: enough to recompute its log-prob/value/entropy under gradient later."""
    seat: int
    stream: list[int]          # the seat's token prefix, ending in TO_ACT
    action_idx: int            # chosen action_space index (the move the net committed to)
    mask: list[float]          # length-ACTION_SIZE legality mask at the decision
    ret: float = 0.0           # discounted return-to-go (filled in after the match ends)
    old_logp: float = 0.0      # logπ(a|s) under the sampling policy, no grad — PPO ratio denominator
    value: float = 0.0         # V(s) under the sampling policy, no grad — learned value baseline


@dataclass
class MatchResult:
    winner_team: int
    learner_won: bool
    decisions: list[Decision] = field(default_factory=list)


def heuristic_opponents(rng: random.Random, seats) -> dict:
    """A `default` (canonical) heuristic bot for each given seat."""
    return {s: HeuristicAgent(rng=random.Random(rng.random())) for s in seats}


def config_opponents(rng: random.Random, seats, pool: list[BotConfig]) -> dict:
    """Build opponents for `seats` by sampling ONE config from `pool` (the opposing *team* shares
    a style, mirroring the tournament) and giving each seat its own seeded `ConfigurableBot`.

    Draws on the passed `rng` so the choice is deterministic per match (the runner seeds `rng`
    from the match seed), keeping rollouts reproducible.
    """
    cfg = rng.choice(pool)
    return {s: ConfigurableBot(cfg, rng=random.Random(rng.random())) for s in seats}


@torch.no_grad()
def _batched_pick(net: TrucoGPT, streams: list[list[int]], masks: list[list[float]],
                  cfg: Config, device: str, sample: bool) -> tuple[list[int], list[float], list[float]]:
    """Pick one action index per stream in a single batched forward.

    Returns `(indices, log_probs, values)`: the chosen action index, its log-prob under the
    (legal-masked) policy, and the value-head estimate V(s) — all detached, recorded onto each
    `Decision` so the PPO update can form the importance ratio and the value baseline without a
    second sampling pass. Computing them here is free (the forward already ran) and consumes no
    RNG after the `multinomial` draw, so sampled rollouts stay bit-identical to before.

    Equivalent, per row, to running the net on each stream alone and reading the TO_ACT-position
    logits: `batch_last_outputs` clips+pads and gathers each stream's true last position, and the
    causal mask keeps padding out of that position. Illegal actions are pushed to ~-inf by
    `action_logits`, so argmax/multinomial only ever return a legal index.
    """
    lm, val = batch_last_outputs(net, streams, cfg.block_size, device)    # [B, vocab], [B]
    m = torch.tensor(masks, dtype=torch.float32, device=device)           # [B, action_size]
    a = action_logits(lm, m, cfg.move_base, cfg.action_size)              # [B, action_size]
    if sample:
        idx = torch.multinomial(torch.softmax(a, dim=-1), 1).squeeze(1)   # [B]
    else:
        idx = torch.argmax(a, dim=-1)                                      # [B]
    logp = torch.log_softmax(a, dim=-1).gather(1, idx[:, None]).squeeze(1)  # [B]
    return idx.tolist(), logp.tolist(), val.tolist()


class _MatchRunner:
    """Drives one match up to each learner decision, deferring the net forward to the caller.

    `advance()` steps through any opponent turns and stops at the next learner decision (returning
    its stream+mask) or at match end (returning None). The caller then computes the action — alone
    or, crucially, batched across many runners — and hands it back via `commit()`. This split is
    what lets `rollout_matches` collapse thousands of batch-1 forwards into a few batched ones.
    """

    def __init__(self, cfg: Config, seed: int, learner_seats: tuple[int, ...], dealer: int,
                 reward_shaping: bool, collect: bool, opponents: dict | None = None,
                 opponent_configs: list[BotConfig] | None = None,
                 opponent_net: bool = False) -> None:
        self.learner_team = team_of(learner_seats[0])
        assert all(team_of(s) == self.learner_team for s in learner_seats), \
            "learner must control exactly one team (shared seat-relative policy)"
        self.cfg = cfg
        self.learner_seats = learner_seats
        self.collect = collect
        # When True the opposing team is driven by a frozen snapshot net the *caller* forwards
        # (deferred + batched, exactly like the learner), not by a synchronous config/heuristic bot.
        self.opponent_net = opponent_net

        rng = random.Random(seed)
        if opponent_net:
            opponents = {}                                # net opponents are picked by the driver
        elif opponents is None:
            other = [s for s in range(4) if s not in learner_seats]
            opponents = (config_opponents(rng, other, opponent_configs) if opponent_configs
                         else heuristic_opponents(rng, other))
        self.opponents = opponents

        self.env = TrucoEnv(seed=seed, dealer=dealer, reward_shaping=reward_shaping)
        self.tok = MatchTokenizer()
        self.tok.start(self.env.state)
        self.decisions: list[Decision] = []
        self.team_rewards: list[float] = []   # learner-team immediate reward after each env step
        self._pending_actor: int | None = None

    def advance(self) -> tuple[str, list[int], list[float]] | None:
        """Advance to the next *deferred* decision, returning `(role, stream, mask)` or None at end.

        `role` is "learner" for a learner seat (always deferred to the caller's net) or "opp" for an
        opponent seat when `opponent_net` is set (deferred to the caller's frozen snapshot).
        Synchronous config/heuristic opponents are still played inline here.
        """
        env, tok = self.env, self.tok
        while not env.done:
            actor = env.agent_selection
            tok.begin_turn(actor)
            if actor in self.learner_seats:
                self._pending_actor = actor
                return "learner", tok.streams[actor], legal_mask(env.observe(actor))
            if self.opponent_net:                         # opponent seat, net-controlled: defer it
                self._pending_actor = actor
                return "opp", tok.streams[actor], legal_mask(env.observe(actor))
            events = env.step(self.opponents[actor].act(env.observe(actor)))
            tok.ingest(env.state, events)
            self.team_rewards.append(env.rewards[self.learner_seats[0]])
        return None

    def commit(self, action_idx: int, mask: list[float],
               old_logp: float = 0.0, value: float = 0.0) -> None:
        """Apply a deferred action (learner or net-opponent): step, tokenize, log reward.

        A `Decision` is recorded only for learner seats — opponent-net turns advance the match but
        are never trained on. Either way one learner-team reward lands in the timeline per env step.
        """
        actor = self._pending_actor
        if self.collect and actor in self.learner_seats:
            self.decisions.append(Decision(seat=actor, stream=list(self.tok.streams[actor]),
                                            action_idx=action_idx, mask=mask,
                                            old_logp=old_logp, value=value,
                                            ret=float(len(self.team_rewards))))  # temp: step pointer
        events = self.env.step(action_from_index(action_idx))
        self.tok.ingest(self.env.state, events)
        self.team_rewards.append(self.env.rewards[self.learner_seats[0]])
        self._pending_actor = None

    def finish(self) -> MatchResult:
        _fill_returns(self.decisions, self.team_rewards, self.cfg.gamma)
        winner_team = self.env.state.winner
        return MatchResult(winner_team=winner_team,
                           learner_won=(winner_team == self.learner_team),
                           decisions=self.decisions)


def _play_group(net: TrucoGPT, cfg: Config, runners: list[_MatchRunner], sample: bool,
                device: str, opp_net: TrucoGPT | None = None) -> None:
    """Play a group of runners to completion in lockstep, batching net forwards per round.

    Each round every live runner advances to its next deferred decision. Learner decisions are
    picked (and recorded) with `net`; opponent decisions — present only when `opp_net` is given —
    are picked with the frozen snapshot `opp_net` in a *separate* batched forward (distinct weights
    can't share a batch). A runner stays live as long as it produced a turn this round.
    """
    live = runners
    while live:
        learner_pend: list[tuple[_MatchRunner, list[int], list[float]]] = []
        opp_pend: list[tuple[_MatchRunner, list[int], list[float]]] = []
        for r in live:
            res = r.advance()
            if res is None:
                continue                                  # match finished — drop from `live`
            role, stream, mask = res
            (learner_pend if role == "learner" else opp_pend).append((r, stream, mask))
        if not learner_pend and not opp_pend:
            break
        if learner_pend:                                  # the trained policy: record logp/value
            idxs, logps, vals = _batched_pick(
                net, [s for _, s, _ in learner_pend], [m for _, _, m in learner_pend],
                cfg, device, sample)
            for (r, _, mask), a_idx, lp, v in zip(learner_pend, idxs, logps, vals):
                r.commit(a_idx, mask, lp, v)
        if opp_pend:                                      # frozen snapshot: pick only, nothing recorded
            idxs, _, _ = _batched_pick(
                opp_net, [s for _, s, _ in opp_pend], [m for _, _, m in opp_pend],
                cfg, device, sample)
            for (r, _, mask), a_idx in zip(opp_pend, idxs):
                r.commit(a_idx, mask)
        live = [r for r, _, _ in learner_pend] + [r for r, _, _ in opp_pend]
    release_memory(device)                                # hand activation buffers back (MPS/CUDA)


def rollout_matches(net: TrucoGPT, cfg: Config, seeds, *, learner_seats=(0, 2),
                    sample: bool = True, dealers=None, reward_shaping: bool = True,
                    collect: bool = True, device: str = "cpu",
                    opponent_configs: list[BotConfig] | None = None,
                    opponent_net: TrucoGPT | None = None,
                    max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> list[MatchResult]:
    """Play one match per seed concurrently (batched net forwards); return results in seed order.

    Matches are played in groups of at most `max_concurrent` to bound the peak forward batch.
    Behaviourally identical to calling `rollout_match` once per seed (argmax: bit-for-bit; sampling:
    same per-decision distributions, different global-RNG draw order).

    The opposing team is, in order of precedence: `opponent_net` (a frozen snapshot — self-play),
    else a `ConfigurableBot` team drawn per match from `opponent_configs` (the strong-config pool),
    else the canonical `default` heuristic. `opponent_net` and `opponent_configs` are mutually
    exclusive.
    """
    assert not (opponent_configs and opponent_net is not None), \
        "pass either opponent_configs or opponent_net, not both"
    seeds = list(seeds)
    learner_seats = tuple(learner_seats)
    dealers = list(dealers) if dealers is not None else [0] * len(seeds)
    group = max_concurrent if max_concurrent and max_concurrent > 0 else len(seeds)

    results: list[MatchResult] = []
    for i in range(0, len(seeds), group):
        runners = [_MatchRunner(cfg, s, learner_seats, d, reward_shaping, collect,
                                opponent_configs=opponent_configs,
                                opponent_net=(opponent_net is not None))
                   for s, d in zip(seeds[i:i + group], dealers[i:i + group])]
        _play_group(net, cfg, runners, sample, device, opp_net=opponent_net)
        results.extend(r.finish() for r in runners)
    return results


def rollout_match(net: TrucoGPT, cfg: Config, seed: int, *, learner_seats=(0, 2),
                  opponents: dict | None = None, sample: bool = True, dealer: int = 0,
                  reward_shaping: bool = True, collect: bool = True,
                  device: str = "cpu") -> MatchResult:
    """Play one match; return its outcome and (optionally) the learner's recorded decisions.

    A thin single-match wrapper over the same runner the batched driver uses. Kept for tests,
    scripts, and the custom-`opponents` path; the trainer/evaluator use `rollout_matches`.
    """
    runner = _MatchRunner(cfg, seed, tuple(learner_seats), dealer, reward_shaping, collect,
                          opponents=opponents)
    while True:
        res = runner.advance()
        if res is None:
            break
        _role, stream, mask = res
        idxs, logps, vals = _batched_pick(net, [stream], [mask], cfg, device, sample)
        runner.commit(idxs[0], mask, logps[0], vals[0])
    return runner.finish()


def batch_last_outputs(net: TrucoGPT, streams: list[list[int]], block_size: int,
                       device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the net on a batch of streams; return (lm_logits, value) at each stream's last token.

    Each stream is first clipped to its final `block_size` tokens (matching the net's own
    left-truncation) so the gathered "last position" is always the genuine decision point — the
    TO_ACT that ends the stream. Used by the rollout picker, the trainer (with grad), and metrics.
    """
    clipped = [s[-block_size:] for s in streams]
    lengths = [len(s) for s in clipped]
    maxlen = max(lengths)
    arr = np.full((len(clipped), maxlen), vocab.PAD, dtype=np.int64)
    for i, s in enumerate(clipped):
        arr[i, :len(s)] = s
    x = torch.from_numpy(arr).to(device)
    feats = net.features(x)                                # [B, T, d_model]
    rows = torch.arange(len(clipped), device=device)
    cols = torch.tensor([n - 1 for n in lengths], device=device)
    h = feats[rows, cols]                                  # [B, d_model] — only decision positions
    return net.lm_head(h), net.value_head(h).squeeze(-1)   # apply heads to B rows, not B×T


def _fill_returns(decisions: list[Decision], team_rewards: list[float], gamma: float) -> None:
    """Replace each decision's step-pointer with its discounted return-to-go.

    `G[j] = sum_{k>=j} gamma^(k-j) * team_rewards[k]` — the discounted future reward from the
    step the decision was taken at, onward.
    """
    n = len(team_rewards)
    suffix = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        suffix[k] = team_rewards[k] + gamma * suffix[k + 1]
    for d in decisions:
        d.ret = suffix[int(d.ret)]

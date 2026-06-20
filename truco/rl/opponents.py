"""Rollout opponents: a spectrum of heuristic styles and a self-play league.

Training purely against the current policy gives a single, narrow opponent distribution — the
learner overfits to its own quirks and never has to actually beat anything. This module supplies
the *diversity*: a fixed set of risk-tuned `HeuristicAgent` styles (including the exact default
bot we evaluate against), plus a `League` that also keeps frozen snapshots of past learner
policies. Mixed matches (see `ppo._rollout_match`) draw an opponent from this pool, so the policy
is pushed to be robust rather than cyclic.

Everything is seeded off an injected `random.Random` so a run stays reproducible. The torch-only
pieces (snapshots wrap a `PolicyNet`) are imported lazily, keeping `make_heuristic_variants`
itself dependency-light.
"""

from __future__ import annotations

import random

from ..agents.heuristic import HeuristicAgent


def _named(name: str, agent):
    """Tag an agent with a human-readable style name (handy for logging / tests)."""
    agent.name = name
    return agent


def make_heuristic_variants(rng: random.Random) -> list:
    """A spectrum of distinct, named `HeuristicAgent` styles for opponent mixing.

    Each variant draws its own seeded RNG from `rng` (so the set is reproducible) and only ever
    returns a legal move. Styles span the risk axis: `aggression` shifts the call/accept/reraise
    gates, `bluff` controls lying (signals + stone-cold raises), `signal_prob` how chatty it is.
    The `default` variant is byte-identical to the canonical `HeuristicAgent` we evaluate against.
    """
    def _rng() -> random.Random:
        return random.Random(rng.random())

    return [
        _named("default", HeuristicAgent(rng=_rng())),
        _named("tight-passive", HeuristicAgent(rng=_rng(), aggression=-1.0, bluff=0.0, signal_prob=0.15)),
        _named("loose-aggressive", HeuristicAgent(rng=_rng(), aggression=1.0, bluff=0.25, signal_prob=0.45)),
        _named("big-bluffer", HeuristicAgent(rng=_rng(), aggression=0.4, bluff=0.5, signal_prob=0.5)),
        _named("honest-rock", HeuristicAgent(rng=_rng(), aggression=-0.3, bluff=0.0, signal_prob=0.3)),
    ]


class League:
    """The pool of rollout opponents: fixed heuristic variants + frozen learner snapshots.

    A mixed match samples one opponent *source* per a configurable distribution
    (`league_self_prob` / `league_heuristic_prob` / `league_snapshot_prob`):
      - "self"     — a stochastic copy of the *current* learner (live net), like self-play but
                     crediting only the learner's seats;
      - "heuristic"— one of `make_heuristic_variants`;
      - "snapshot" — a FIFO-bounded frozen clone of a past learner policy, never updated.
    Snapshots guard against strategy cycling / overfitting to a single opponent.
    """

    def __init__(self, config, rng: random.Random):
        self.config = config
        self.rng = rng
        self.heuristics = make_heuristic_variants(rng)
        self.snapshots: list = []  # PolicyAgents, oldest first, bounded by league_max_snapshots

    def add_snapshot(self, net) -> None:
        """Freeze `net`'s current weights into a stochastic opponent and add it (FIFO-bounded)."""
        from ..agents.policy_agent import PolicyAgent
        from .policy import PolicyNet

        frozen = PolicyNet(
            encoded_size=net.encoded_size, action_size=net.action_size, hidden=net.hidden
        )
        frozen.load_state_dict({k: v.detach().clone().cpu() for k, v in net.state_dict().items()})
        frozen.eval()
        agent = PolicyAgent(net=frozen, deterministic=False, temperature=self.config.rollout_temperature)
        agent.name = f"snapshot-{len(self.snapshots)}"
        self.snapshots.append(agent)
        if len(self.snapshots) > self.config.league_max_snapshots:
            self.snapshots.pop(0)  # FIFO: drop the oldest

    def sample_opponent(self, net):
        """Pick a mixed-match opponent per the league distribution; returns an `act(obs)` agent."""
        from ..agents.policy_agent import PolicyAgent

        sources = ["self", "heuristic"]
        weights = [self.config.league_self_prob, self.config.league_heuristic_prob]
        if self.snapshots:  # only an option once we've taken at least one snapshot
            sources.append("snapshot")
            weights.append(self.config.league_snapshot_prob)
        choice = self.rng.choices(sources, weights=weights)[0]
        if choice == "self":
            return PolicyAgent(net=net, deterministic=False, temperature=self.config.rollout_temperature)
        if choice == "snapshot":
            return self.rng.choice(self.snapshots)
        return self.rng.choice(self.heuristics)

    def sample_descriptor(self) -> dict:
        """Picklable mirror of `sample_opponent`: pick a source, return it as serializable data.

        Used by the parallel trainer to choose each mixed match's opponent in the main process and
        ship it to a worker, which rebuilds the agent (`ppo._build_opponents`). It consumes
        `self.rng` *identically* to `sample_opponent` — same source draw, and `randrange(n)` /
        `choice(seq-of-n)` both draw one `_randbelow(n)` — so the two are interchangeable without
        perturbing the reproducible sampling stream:
          - {"kind": "self"}                              → rebuilt from the shipped learner weights
          - {"kind": "snapshot", "index": i}             → rebuilt from shipped snapshot weights[i]
          - {"kind": "heuristic", aggression/bluff/...}  → rebuilt as a seeded HeuristicAgent
        """
        sources = ["self", "heuristic"]
        weights = [self.config.league_self_prob, self.config.league_heuristic_prob]
        if self.snapshots:
            sources.append("snapshot")
            weights.append(self.config.league_snapshot_prob)
        choice = self.rng.choices(sources, weights=weights)[0]
        if choice == "self":
            return {"kind": "self"}
        if choice == "snapshot":
            return {"kind": "snapshot", "index": self.rng.randrange(len(self.snapshots))}
        h = self.rng.choice(self.heuristics)
        return {"kind": "heuristic", "aggression": h.aggression,
                "bluff": h.bluff, "signal_prob": h.signal_prob}

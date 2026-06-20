"""Reinforcement-learning layer for Truco.

A self-play PPO trainer that learns a single shared policy mapping an encoded, seat-relative
observation (`observation.encode`) to a *masked* action distribution over a fixed action
index space (`action_space`). The learned policy is exposed to the rest of the codebase as
`agents.PolicyAgent`, which honours the same `act(obs)` interface as every other agent and
so drops straight into the CLI.

Nothing here touches the engine, env, or observation modules — the RL code is a pure
consumer of their public APIs.
"""

from .action_space import ACTION_SIZE, action_from_index, action_index, legal_mask

__all__ = ["ACTION_SIZE", "action_index", "action_from_index", "legal_mask"]

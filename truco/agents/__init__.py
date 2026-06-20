"""Policies that turn an Observation into an action.

All agents share one tiny interface (`base.Agent`): `act(observation) -> action`. The
observation already carries the legal-action mask, so an agent only ever returns a legal
move. `HumanAgent` is the only agent that performs console I/O; the rest are pure.

`PolicyAgent` (the learned RL policy) is exported lazily via `__getattr__`: importing it
pulls in torch, so the base game and CLI stay torch-free unless a policy is actually asked
for.
"""

from .base import Agent
from .configurable import BotConfig, ConfigurableBot
from .heuristic import HeuristicAgent
from .human import HumanAgent
from .random_agent import RandomAgent

__all__ = ["Agent", "RandomAgent", "HeuristicAgent", "HumanAgent", "PolicyAgent",
           "BotConfig", "ConfigurableBot"]


def __getattr__(name):
    if name == "PolicyAgent":
        from .policy_agent import PolicyAgent

        return PolicyAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

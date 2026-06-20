"""The agent interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..observation import Observation


@runtime_checkable
class Agent(Protocol):
    """Anything that can choose a legal action from an observation.

    The observation carries `legal_actions`; an implementation MUST return one of them.
    A future RL policy implements this same method, so it drops straight into the CLI.
    """

    def act(self, obs: Observation): ...

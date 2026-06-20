"""A learned-policy agent — the inference face of the RL trainer.

Wraps a trained `PolicyNet` behind the standard `act(obs)` interface, so it drops straight
into the CLI and the evaluation driver alongside `RandomAgent`/`HeuristicAgent`. It encodes
the observation, masks illegal actions, and (by default) plays greedily — the single
strongest legal move. Pass `deterministic=False` (optionally with a `temperature`) to sample
instead — useful for less-exploitable play and for frozen league opponents. Because it only
ever picks an index whose mask bit is set, the action it returns is always one of
`obs.legal_actions`.

Torch is imported lazily (only when this agent is actually constructed), so the base game
and CLI stay torch-free unless someone asks for a policy.
"""

from __future__ import annotations

from ..observation import Observation, encode
from ..rl.action_space import action_from_index, legal_mask


class PolicyAgent:
    """Plays a trained policy. Pass a checkpoint path, or an in-memory net (used by eval)."""

    def __init__(
        self,
        checkpoint_path=None,
        net=None,
        deterministic: bool = True,
        temperature: float = 1.0,
    ):
        from ..rl import policy as policy_mod  # lazy: pulls in torch

        if net is None:
            if checkpoint_path is None:
                raise ValueError("PolicyAgent needs either checkpoint_path or net")
            net, _ = policy_mod.load(checkpoint_path)
        net.eval()
        self.net = net
        self.deterministic = deterministic
        self.temperature = temperature

    def act(self, obs: Observation):
        import torch

        vec = torch.tensor(encode(obs), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(legal_mask(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(vec, mask)
            if self.deterministic:
                index = int(torch.argmax(logits, dim=-1).item())
            else:
                from torch.distributions import Categorical

                # Temperature scales the spread; the -1e9 mask stays hugely negative after /T.
                index = int(Categorical(logits=logits / self.temperature).sample().item())
        return action_from_index(index)

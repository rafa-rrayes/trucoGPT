"""A learned-policy agent: the trained TrucoGPT transformer behind the `act` interface.

The transformer reads a per-seat token stream (built by `truco.rl.tokenizer.MatchTokenizer`),
not a flat observation vector. The driver therefore tokenizes the match and passes the acting
seat's stream to `act(obs, stream)`; the stream must already end in the `TO_ACT` prompt.
Illegal actions are masked out, so the returned action is always in `obs.legal_actions`.

By default the agent plays greedily (argmax). Pass `deterministic=False`, optionally with a
`temperature`, to sample instead for less predictable play.

Torch is imported lazily (only when this agent is constructed), so the base game and CLI stay
torch-free unless a policy is requested.
"""

from __future__ import annotations

from ..observation import Observation
from ..rl.action_space import action_from_index, legal_mask


class PolicyAgent:
    """Plays a trained policy. Pass a checkpoint path, or an in-memory net."""

    needs_stream = True  # drivers must supply the seat's token stream to `act`

    def __init__(self, checkpoint_path=None, net=None, deterministic: bool = True,
                 temperature: float = 1.0, device: str = "cpu"):
        from ..rl.model import load  # lazy: pulls in torch

        if net is None:
            if checkpoint_path is None:
                raise ValueError("PolicyAgent needs either checkpoint_path or net")
            net, _ = load(str(checkpoint_path), map_location=device)
        self.net = net.to(device).eval()
        self.device = device
        self.deterministic = deterministic
        self.temperature = temperature

    def act(self, obs: Observation, stream: list[int]):
        import torch

        from ..rl.model import action_logits
        from ..rl.rollout import batch_last_outputs

        cfg = self.net.cfg
        with torch.no_grad():
            lm, _ = batch_last_outputs(self.net, [stream], cfg.block_size, self.device)
            mask = torch.tensor([legal_mask(obs)], dtype=torch.float32, device=self.device)
            logits = action_logits(lm, mask, cfg.move_base, cfg.action_size)
            if self.deterministic:
                index = int(torch.argmax(logits, dim=-1).item())
            else:
                # Masked entries are ~-inf and stay so after dividing by the temperature.
                probs = torch.softmax(logits / self.temperature, dim=-1)
                index = int(torch.multinomial(probs, 1).item())
        return action_from_index(index)

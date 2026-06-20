"""Truco — a 2v2 Brazilian Truco engine, environment, and CLI.

The package is layered so the same core drives both the CLI and a future RL trainer:

    cards / state / actions / engine / observation / env   <- pure, I/O-free core
    agents/                                                <- policies (human does I/O)
    cli/                                                   <- interactive frontend

The core never prints, reads input, or touches a global RNG: all randomness is an
injected ``random.Random`` so self-play is reproducible.
"""

__all__ = ["cards", "state", "actions", "engine", "observation", "env"]

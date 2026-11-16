"""Interactive match loop: you (seat 0) versus three heuristic bots.

Builds a match, then drives it action-by-action through the same pure engine the trainer
will use. The human sees the board before each of their decisions; bot moves are narrated
in between. No game logic lives here — only presentation and turn dispatch.
"""

from __future__ import annotations

import argparse
import random

from .. import engine
from ..agents import HeuristicAgent, HumanAgent
from ..observation import observe
from ..state import team_of
from .render import narrate, render_board

HUMAN_SEAT = 0
_INTRO = """
╔════════════════════════════════════════════════════════════╗
║                     T R U C O  (CLI)                         ║
╠════════════════════════════════════════════════════════════╣
║  You are seat 0; your partner is seat 2 (Partner).           ║
║  Opponents are seats 1 & 3. First team to 12 wins a set;     ║
║  first to 2 sets wins the match.                             ║
║                                                              ║
║  Ranks (worst→best):  4 5 6 7 Q J K A 2 3                    ║
║  The vira fixes the manilha (the rank just above it); the    ║
║  four manilhas beat everything — ♣ (Zap ⚡) is the strongest.║
║                                                              ║
║  On your turn you may play a card, raise the stakes          ║
║  (Truco→Six→Nine→Twelve), or flash a private signal to your  ║
║  partner. From round 2 on you may play a card face-down      ║
║  (hidden, but it can't win the round).                       ║
╚════════════════════════════════════════════════════════════╝
"""


def build_agents(rng: random.Random, policy_path: str | None = None, temperature: float = 1.0):
    """Seat 0 is the human; the other three seats are bots.

    By default the bots are independent heuristics; pass `policy_path` to face three copies
    of a trained RL policy loaded from that checkpoint instead. The policy plays greedily unless
    `temperature != 1.0`, in which case it samples at that temperature (less predictable play).
    """
    if policy_path is not None:
        from ..agents import PolicyAgent  # lazy: only this branch needs torch

        greedy = temperature == 1.0
        bot = PolicyAgent(policy_path, deterministic=greedy, temperature=temperature)
        bots = {1: bot, 2: bot, 3: bot}
    else:
        bots = {seat: HeuristicAgent(rng=random.Random(rng.random())) for seat in (1, 2, 3)}
    return {HUMAN_SEAT: HumanAgent(), **bots}


def play_match(state, agents, rng, out=print) -> int:
    """Run a match to completion, rendering for the human. Returns the winning team."""
    # Transformer bots read token streams, so tokenize the match only when one is seated.
    tok = None
    if any(getattr(a, "needs_stream", False) for a in agents.values()):
        from ..rl.tokenizer import MatchTokenizer  # lazy: keeps the base CLI torch-free

        tok = MatchTokenizer()
        tok.start(state)
    while not state.done:
        actor = state.current_actor
        if actor == HUMAN_SEAT:
            out(render_board(observe(state, HUMAN_SEAT)))
        if tok is not None:
            tok.begin_turn(actor)
        agent = agents[actor]
        obs = observe(state, actor)
        action = agent.act(obs, tok.streams[actor]) if getattr(agent, "needs_stream", False) \
            else agent.act(obs)
        state, events = engine.apply_action(state, action, rng)
        if tok is not None:
            tok.ingest(state, events)
        for line in narrate(events, HUMAN_SEAT):
            out(line)
    return state.winner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="truco", description="Play Truco against three bots.")
    parser.add_argument("--seed", type=int, default=None, help="seed for a reproducible match")
    parser.add_argument("--dealer", type=int, default=0, choices=range(4),
                        help="seat that deals the first hand (default 0)")
    parser.add_argument("--policy", default=None, metavar="CKPT",
                        help="face a trained RL policy (checkpoint path) instead of the heuristic bots")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature for --policy bots (1.0 = greedy; >1 = noisier)")
    opts = parser.parse_args(argv)

    rng = random.Random(opts.seed)
    state = engine.new_match(rng, dealer=opts.dealer)
    agents = build_agents(rng, policy_path=opts.policy, temperature=opts.temperature)

    print(_INTRO)
    h = state.hand
    print(f"  First hand dealt. Vira {h.vira} → manilha is any {h.manilha}.\n")

    try:
        winner = play_match(state, agents, rng)
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Game abandoned. Até a próxima!")
        return 1

    outcome = "You and your partner WIN!" if team_of(HUMAN_SEAT) == winner else "The bots win this time."
    print(f"\n  {outcome}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

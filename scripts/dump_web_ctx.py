"""Dump realistic web template contexts to JSON for frontend development/testing.

Produces /tmp/truco_ctx_samples.json with several `ctx` dicts (lobby, mid-play with a
raise pending, finished) exactly as the web render layer feeds them to the templates.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truco import engine
from truco.agents import HeuristicAgent
from truco.rl.action_space import action_from_index, legal_mask
from truco.observation import observe
from truco.web import serialize


def seat_meta(state, you_seat, occupants, host_seat, phase):
    from truco.state import team_of
    cur = state.current_actor if state.hand is not None else None
    dealer = state.dealer
    out = []
    for s in range(4):
        out.append({
            "seat": s, "team": team_of(s),
            "teamName": "us" if team_of(s) == team_of(you_seat) else "them",
            "occupant": occupants[s][0], "name": occupants[s][1],
            "connected": True, "isYou": s == you_seat, "isHost": s == host_seat,
            "isDealer": s == dealer, "isCurrent": s == cur,
        })
    return out


def make_ctx(state, you_seat, occupants, host_seat, phase, log, toast=None, winner=None):
    from truco.state import team_of
    game = serialize.game_view(state, you_seat) if state.hand is not None or phase != "lobby" else None
    return {
        "code": "WXYZ",
        "phase": phase,
        "you": {"pid": "p-rafa", "name": occupants[you_seat][1], "seat": you_seat,
                "isHost": you_seat == host_seat, "isSeated": True, "isSpectator": False},
        "seats": seat_meta(state, you_seat, occupants, host_seat, phase),
        "canStart": phase == "lobby",
        "seatsFilled": 4,
        "settings": {"setsToWin": 1},
        "game": game,
        "log": log,
        "toast": toast,
        "winner": winner,
        "winnerName": ("You & Partner" if winner == "us" else "Opponents") if winner else None,
    }


def main():
    occupants = {
        0: ("human", "Rafa"),
        1: ("model", "TrucoGPT"),
        2: ("human", "Bruno"),
        3: ("heuristic", "Robo"),
    }
    rng = random.Random(7)
    state = engine.new_match(rng, dealer=3)

    samples = {}

    # Lobby sample (no game yet).
    lobby_state = engine.new_match(random.Random(1), dealer=0)
    lobby_ctx = make_ctx(lobby_state, 0, occupants, 0, "lobby",
                         log=["Room WXYZ created.", "Rafa joined.", "Bruno joined."])
    lobby_ctx["game"] = None
    lobby_ctx["canStart"] = True
    samples["lobby"] = lobby_ctx

    # Advance with heuristics until it's seat 0's turn to PLAY (a rich decision point).
    from truco.state import Phase
    bots = {s: HeuristicAgent(rng=random.Random(s + 1)) for s in range(4)}
    play_state = engine.new_match(random.Random(7), dealer=3)  # dealer 3 -> seat 0 leads
    guard = 0
    while not (play_state.current_actor == 0 and play_state.phase is Phase.PLAY) and guard < 200:
        actor = play_state.current_actor
        action = bots[actor].act(observe(play_state, actor))
        play_state, _ = engine.apply_action(play_state, action, rng)
        guard += 1
    samples["playing"] = make_ctx(
        play_state, 0, occupants, 0, "playing",
        log=["— Hand #1 dealt · dealer Opp-2 · vira 5♣ → manilha 6.",
             "  Opp-1 plays K♦.", "  Partner plays 7♥.", "  Opp-2 plays 4♠."],
    )

    # A truco RESPOND decision for seat 0: dealer 0 -> lead 1, so seat 3 acts third and calls.
    from truco.actions import CallTruco, PlayCard
    resp = engine.new_match(random.Random(11), dealer=0)
    # seats 1, 2 play a card each; seat 3 calls Truco -> responder is seat 0.
    for _ in range(2):
        actor = resp.current_actor
        card = next(a for a in engine.legal_actions(resp) if isinstance(a, PlayCard) and not a.face_down)
        resp, _ = engine.apply_action(resp, card, rng)
    resp, _ = engine.apply_action(resp, CallTruco(), rng)  # seat 3 calls; seat 0 must answer
    samples["respond"] = make_ctx(
        resp, 0, occupants, 0, "playing",
        log=["— Hand #1 dealt · dealer You · vira Q♦ → manilha K.",
             "  Opp-1 plays 2♠.", "  Partner plays 4♣.",
             "  🔥 Opp-2 calls Truco! (stake → 3)"],
    )

    # Force a finished-ish snapshot: keep a state with scores for the finished view.
    fin = engine.new_match(random.Random(99), dealer=1)
    fin.match_sets = [1, 0]
    fin.winner = 0
    fin.phase = Phase.MATCH_DONE
    fin_ctx = make_ctx(fin, 0, occupants, 0, "finished",
                       log=["  🎉 Us WIN THE MATCH! 🎉"], winner="us")
    samples["finished"] = fin_ctx

    out = Path("/tmp/truco_ctx_samples.json")
    out.write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    print("\n--- playing.game keys ---")
    print(sorted(samples["playing"]["game"].keys()))
    print("\n--- a hand card ---")
    print(json.dumps(samples["playing"]["game"]["hand"][0], ensure_ascii=False))
    print("\n--- legalActions ---")
    print(json.dumps(samples["playing"]["game"]["legalActions"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

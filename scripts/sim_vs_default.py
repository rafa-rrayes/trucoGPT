"""Simulate one match: TrucoGPT (seats 0 & 2) vs HeuristicAgent (seats 1 & 3).

Shows every card dealt to every seat, every action, every round result, and all scores.
All 4 seats' cards are revealed (god-mode view — no privacy masks).

Run: `uv run python scripts/sim_vs_default.py [SEED]`
     (default seed 42)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from truco import engine
from truco.agents import HeuristicAgent
from truco.cards import card_strength, is_zap, manilha_rank
from truco.engine import (
    CardPlayed, ElevenDecided, ElevenStarted, HandDealt, HandResolved,
    MatchWon, Redeal, RoundResolved, SetWon, SignalSent, TrucoAnswered, TrucoCalled,
)
from truco.observation import observe
from truco.rl.action_space import action_from_index, legal_mask
from truco.rl.model import load, pick_device
from truco.rl.rollout import batch_last_outputs
from truco.rl.tokenizer import MatchTokenizer
from truco.state import RAISE_NAMES, VALUES, team_of
from truco.actions import SIGNAL_LABEL

CKPT = "checkpoints/rl_legality.pt"
LEARNER_SEATS = (0, 2)
HEURISTIC_SEATS = (1, 3)
SEAT_LABEL = {0: "Model-0  ", 1: "Heuristic-1", 2: "Model-2  ", 3: "Heuristic-3"}
TEAM_LABEL = {0: "MODEL (seats 0,2)", 1: "HEURISTIC (seats 1,3)"}
W = 70


def _bar(c="="):  return c * W
def _rule():      return "-" * W
def _team(t):     return TEAM_LABEL[t]
def _seat(s):     return SEAT_LABEL[s]


def _card_list(cards, manilha: str) -> str:
    parts = []
    for c in cards:
        s = str(c)
        if is_zap(c, manilha):
            s += " (ZAP⚡)"
        elif c.rank == manilha:
            s += " (manilha)"
        parts.append(s)
    return "  ".join(parts) if parts else "(none)"


@torch.no_grad()
def _model_pick(net, stream, mask, cfg, device):
    lm, _ = batch_last_outputs(net, [stream], cfg.block_size, device)
    order = torch.argsort(lm[0], descending=True).tolist()
    legal = {i for i, m in enumerate(mask) if m > 0}
    for tok in order:
        a_idx = tok - cfg.move_base
        if 0 <= a_idx < cfg.action_size and a_idx in legal:
            return a_idx
    raise RuntimeError("no legal action found")


def simulate(seed: int) -> None:
    device = pick_device()
    net, meta = load(CKPT, map_location=device)
    net = net.to(device).eval()
    cfg = net.cfg

    rng = random.Random(seed)
    heuristics = {s: HeuristicAgent(rng=random.Random(rng.random())) for s in HEURISTIC_SEATS}
    state = engine.new_match(rng, dealer=seed % 4)
    tok = MatchTokenizer()
    tok.start(state)

    print(_bar())
    print(f"  TrucoGPT ({Path(CKPT).name})  vs  HeuristicAgent")
    print(f"  checkpoint: win_rate={meta.get('win_rate')}, "
          f"win_rate_vs_default={meta.get('win_rate_vs_default')}")
    print(f"  seed={seed}  |  first dealer: {_seat(seed % 4)}")
    print(_bar())

    set_scores = [0, 0]         # running points this set
    match_sets = [0, 0]
    hand_num = 0

    def _print_hand_header(h, hand_n):
        mrank = manilha_rank(h.vira)
        print()
        print(_bar("="))
        print(f"  HAND #{hand_n}   dealer: {_seat(h.dealer)}   "
              f"vira: {h.vira}  →  manilha rank: {mrank}")
        print(f"  Score so far:  "
              f"Model {set_scores[0]}–{set_scores[1]} Heuristic   "
              f"(sets: Model {match_sets[0]}–{match_sets[1]} Heuristic)")
        print(_rule())
        print("  Cards dealt:")
        for s in range(4):
            cards = h.hands[s]
            strength_order = sorted(cards, key=lambda c: card_strength(c, mrank), reverse=True)
            print(f"    {_seat(s)}: {_card_list(strength_order, mrank)}")
        print(_bar("="))

    # Print the first hand — already dealt by new_match, no event fires for it
    hand_num += 1
    _print_hand_header(state.hand, hand_num)

    while not state.done:
        actor = state.current_actor
        tok.begin_turn(actor)
        obs = observe(state, actor)

        if actor in LEARNER_SEATS:
            stream = tok.streams[actor]
            mask = legal_mask(obs)
            a_idx = _model_pick(net, stream, mask, cfg, device)
            action = action_from_index(a_idx)
        else:
            action = heuristics[actor].act(obs)

        state, events = engine.apply_action(state, action, rng)
        tok.ingest(state, events)

        for ev in events:
            # ---------- hand dealt ----------
            if isinstance(ev, HandDealt):
                hand_num += 1
                _print_hand_header(state.hand, hand_num)

            # ---------- eleven ----------
            elif isinstance(ev, ElevenStarted):
                print(f"  *** HAND OF ELEVEN — {_team(ev.team)} ({_seat(ev.decider)}) decides ***")
            elif isinstance(ev, ElevenDecided):
                print(f"  {_team(ev.team)} chose to {'PLAY' if ev.play else 'SKIP'} the Hand of Eleven.")

            # ---------- signal ----------
            elif isinstance(ev, SignalSent):
                label = SIGNAL_LABEL.get(ev.code, f"code {ev.code}")
                print(f"  ✋ {_seat(ev.seat)} signals: \"{label}\"")

            # ---------- card played ----------
            elif isinstance(ev, CardPlayed):
                fd = " (face-down)" if ev.face_down else ""
                print(f"  {_seat(ev.seat)} plays {ev.card}{fd}")

            # ---------- truco / raise ----------
            elif isinstance(ev, TrucoCalled):
                print(f"  🔥 {_seat(ev.seat)} calls {RAISE_NAMES[ev.proposed_level]}! "
                      f"(stake → {VALUES[ev.proposed_level]} pts)")
            elif isinstance(ev, TrucoAnswered):
                verb = {"accept": "ACCEPTS", "run": "RUNS", "raise": "RE-RAISES"}[ev.response]
                print(f"  {_seat(ev.seat)} {verb}.")

            # ---------- round result ----------
            elif isinstance(ev, RoundResolved):
                if ev.result_team == -1:
                    print(f"  >> Round {ev.round_index+1}: TIE")
                else:
                    print(f"  >> Round {ev.round_index+1}: {_team(ev.result_team)} wins "
                          f"(led by {_seat(ev.winning_seat)})")

            # ---------- hand result ----------
            elif isinstance(ev, HandResolved):
                reason_map = {"rounds": "", "run": " (opponent ran)",
                              "eleven_skip": " (Eleven skipped)"}
                print(_rule())
                print(f"  ➜ HAND RESULT: {_team(ev.winner_team)} +{ev.value} pts"
                      f"{reason_map.get(ev.reason, '')}")
                set_scores[ev.winner_team] += ev.value
                print(f"  Running set score:  Model {set_scores[0]} – {set_scores[1]} Heuristic")

            # ---------- redeal ----------
            elif isinstance(ev, Redeal):
                print("  All three rounds tied — RE-DEAL (no score)")

            # ---------- set won ----------
            elif isinstance(ev, SetWon):
                match_sets[ev.team] += 1
                set_scores = [0, 0]
                print()
                print(_bar("*"))
                print(f"  🏆 SET WON by {_team(ev.team)}!  "
                      f"Match sets: Model {match_sets[0]}–{match_sets[1]} Heuristic")
                print(_bar("*"))

            # ---------- match won ----------
            elif isinstance(ev, MatchWon):
                print()
                print(_bar())
                print(f"  🎉 MATCH WON by {_team(ev.team)}")
                print(f"  Final sets: Model {match_sets[0]}–{match_sets[1]} Heuristic")
                print(_bar())


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    simulate(seed)

"""Simulate N full matches of a trained TrucoGPT (seats 0 & 2) vs the `default` heuristic
(seats 1 & 3) — a 2-DL-vs-2-heuristic team game — and narrate every move.

Faithful to evaluation: the model plays **argmax** (greedy, no sampling) and we drive the real
engine through the exact same `begin_turn → act → step → ingest` loop the rollout/eval driver
uses, seeding env + opponents identically. By default we replay the first N games of the firm
eval set (seeds 1_000_000+i, dealer i%4), so these ARE the games behind the reported win-rate.
A cross-check asserts our per-match winners equal `rollout_matches(..., sample=False)`.

Narration is from seat 0's point of view (`truco.cli.render.narrate`), so **"Us" = the DL team**
(seats 0,2) and **"Them" = the heuristic team** (seats 1,3). In the lines, "You" and "Partner"
are the two DL agents (seats 0 and 2); "Opp-1"/"Opp-2" are the heuristic agents (seats 1 and 3).

Run: `uv run python scripts/sim_best_model.py [CKPT] [N] [OUT.txt]`
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import truco`

from truco.cli.render import narrate
from truco.engine import HandResolved, SetWon
from truco.env import TrucoEnv
from truco.rl.action_space import action_from_index, legal_mask
from truco.rl.model import load
from truco.rl.rollout import _batched_pick, heuristic_opponents, rollout_matches
from truco.rl.tokenizer import MatchTokenizer
from truco.state import team_of

EVAL_SEED0 = 1_000_000           # the firm-eval seed base — so these are the real eval games
LEARNER = (0, 2)                 # the DL team


def play_and_narrate(net, cfg, seed: int, dealer: int, device: str = "cpu"):
    """Drive one match (DL on LEARNER seats, heuristic elsewhere); return (winner, scores, lines).

    Mirrors `rollout._MatchRunner` exactly (same seeding, same begin_turn/act/step/ingest order),
    so with sample=False the game is bit-for-bit the eval game — we just also collect narration.
    """
    learner_team = team_of(LEARNER[0])
    rng = random.Random(seed)
    other = [s for s in range(4) if s not in LEARNER]
    opponents = heuristic_opponents(rng, other)        # identical RNG draws to _MatchRunner
    env = TrucoEnv(seed=seed, dealer=dealer, reward_shaping=True)
    tok = MatchTokenizer()
    tok.start(env.state)

    # The engine zeroes set_scores the instant a set is won, so reconstruct the deciding-set
    # score from HandResolved values and snapshot it at SetWon (before the reset).
    set_pts = [0, 0]
    final_set_pts = [0, 0]

    lines: list[str] = []
    while not env.done:
        actor = env.agent_selection
        tok.begin_turn(actor)
        if actor in LEARNER:
            stream = tok.streams[actor]
            mask = legal_mask(env.observe(actor))
            a_idx = _batched_pick(net, [stream], [mask], cfg, device, sample=False)[0][0]
            action = action_from_index(a_idx)
        else:
            action = opponents[actor].act(env.observe(actor))
        events = env.step(action)
        tok.ingest(env.state, events)
        lines += narrate(events, human_seat=0)         # POV seat 0 → "Us" = DL team
        for e in events:
            if isinstance(e, HandResolved):
                set_pts[e.winner_team] += e.value
            elif isinstance(e, SetWon):
                final_set_pts = list(set_pts)
                set_pts = [0, 0]

    st = env.state
    return st.winner, final_set_pts, lines, st.hand_number


LEGEND = (
    "Legend:  Us = DL model (seats 0 & 2)   Them = `default` heuristic (seats 1 & 3)\n"
    "         \"You\"+\"Partner\" = the two DL agents · \"Opp-1\"+\"Opp-2\" = the two heuristic agents\n"
    "         Model plays greedily (argmax), exactly as in evaluation."
)


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/rl_legality.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        Path(__file__).resolve().parent.parent / "best_model_10games.txt")
    device = "cpu"

    net, meta = load(ckpt, map_location=device)
    net = net.to(device).eval()
    cfg = net.cfg
    seeds = [EVAL_SEED0 + i for i in range(n)]
    dealers = [i % 4 for i in range(n)]

    # Play each game with narration.
    games = [play_and_narrate(net, cfg, s, d, device) for s, d in zip(seeds, dealers)]

    # Fidelity cross-check: winners must equal the batched eval driver (argmax = bit-for-bit).
    ref = rollout_matches(net, cfg, seeds, learner_seats=LEARNER, sample=False,
                          dealers=dealers, collect=False, device=device)
    ref_won = [r.learner_won for r in ref]
    mine_won = [w == team_of(LEARNER[0]) for (w, _, _, _) in games]
    cross_ok = ref_won == mine_won

    full: list[str] = []
    full.append("=" * 70)
    full.append(f"TrucoGPT (best model: {Path(ckpt).name})  vs  `default` heuristic — 2 DL vs 2 Heuristic")
    full.append(f"checkpoint meta: {meta}")
    full.append(LEGEND)
    full.append("=" * 70)

    wins = 0
    rows = []
    for i, (winner, scores, lines, hands) in enumerate(games):
        dl_won = winner == team_of(LEARNER[0])
        wins += int(dl_won)
        us, them = scores[0], scores[1]
        rows.append((i + 1, seeds[i], dealers[i], dl_won, us, them, hands))
        full.append("")
        full.append("#" * 70)
        full.append(f"# GAME {i + 1}/{n}  (seed {seeds[i]}, dealer seat {dealers[i]})")
        full.append("#" * 70)
        full.extend(lines)
        full.append(f"  ── RESULT: {'DL WINS' if dl_won else 'HEURISTIC WINS'}  "
                    f"(Us {us} – {them} Them, over {hands} hands)")

    # Write the complete transcript of all games to disk.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(full))

    # ---- console summary (compact) ----
    print(LEGEND)
    print()
    print(f"{'Game':>4}  {'seed':>9}  {'dealer':>6}  {'winner':>9}  {'score (Us–Them)':>16}  {'hands':>5}")
    print("-" * 62)
    for (g, seed, dealer, dl_won, us, them, hands) in rows:
        print(f"{g:>4}  {seed:>9}  seat {dealer}  {'DL' if dl_won else 'Heuristic':>9}  "
              f"{us:>7} – {them:<6}  {hands:>5}")
    print("-" * 62)
    avg_us = sum(r[4] for r in rows) / len(rows)
    avg_them = sum(r[5] for r in rows) / len(rows)
    print(f"DL record: {wins}–{n - wins}  (win-rate {wins / n:.0%})   "
          f"avg points: Us {avg_us:.1f} – {avg_them:.1f} Them")
    print(f"fidelity cross-check vs rollout_matches(sample=False): "
          f"{'PASS — identical winners' if cross_ok else 'FAIL — winners differ!'}")
    print(f"full move-by-move transcript of all {n} games → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compete the **pretrained** TrucoGPT (seats 0 & 2) against a set of tuned `ConfigurableBot`
heuristics (seats 1 & 3) — one 2-DL-vs-2-heuristic match per config — and narrate every move.

This drives the real engine through the exact `begin_turn → act → step → ingest` loop the
rollout/eval driver uses, so each match is a faithful game. Two differences from the eval driver:

  * **Opponents.** Instead of the canonical `default` heuristic, seats 1 & 3 are a `ConfigurableBot`
    built from one of the configs in `CONFIGS` below (the tuned-bot knob dicts the user supplied).

  * **Illegal-move fallback (as requested).** The model picks its move *greedily over the full
    next-token distribution*: it takes its single most-likely token and, if that token is not a
    legal move (a non-move token, or a move that isn't legal right now), it falls through to the
    next-most-likely token, and so on, until it lands on a legal move. (The eval driver instead
    masks the logits before argmax; for a legal pick the two agree, but this version also reports
    how often the model's top choice was an illegal move — informative for a *pretrained* net that
    has never been RL-tuned to keep its probability mass on legal actions.)

Narration is from seat 0's POV (`truco.cli.render.narrate`): **"Us" = the DL team** (seats 0,2),
**"Them" = the heuristic** (seats 1,3). "You"/"Partner" are the two DL agents; "Opp-1"/"Opp-2"
are the two heuristic agents.

Run: `uv run python scripts/sim_pretrain_vs_configs.py [CKPT] [N_PER_CONFIG] [OUT.txt]`
     (defaults: checkpoints/pretrain.pt, 1 match per config, samples/pretrain_vs_configs.txt)
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import truco`

import torch

from truco.agents import BotConfig, ConfigurableBot
from truco.cli.render import narrate
from truco.engine import HandResolved, SetWon
from truco.env import TrucoEnv
from truco.rl.action_space import action_from_index, legal_mask
from truco.rl.model import load
from truco.rl.rollout import batch_last_outputs
from truco.rl.tokenizer import MatchTokenizer
from truco.rl.vocab import token_name
from truco.state import team_of

LEARNER = (0, 2)            # the DL (pretrained-model) team
SEED0 = 2_000_000          # base seed for these exhibition matches (distinct from the eval set)

# The tuned heuristic configs to compete against, in the order given. `wr` is each config's
# win-rate *in the ranking tournament* (vs the textbook bot) — its label here, NOT its result
# against the model. Any knob omitted from a dict keeps its BotConfig default.
CONFIGS: list[tuple[str, dict]] = [
    ("wr=1.000 (18/18)", dict(bluff=0.05, signal_prob=0.4, p_lie=0.3, p_signal_late=0.05,
        p_signal_partial=0.05, p_contradict=0.5, p_ignore_partner=0.35, p_overtrump=0.25,
        p_burn=0.35, p_fold_won=0.45, p_overkill=0.15, p_bad_facedown=0.05, p_bad_eleven=0.25)),
    ("wr=0.952 (20/21)", dict(bluff=0.15, signal_prob=0.4, aggression=0.25, p_signal_late=0.3,
        p_contradict=0.5, p_ignore_partner=0.1, p_overtrump=0.2, p_fold_won=0.1, p_overkill=0.2,
        p_bad_facedown=0.05, p_bad_eleven=0.4)),
    ("wr=0.952 (20/21)", dict(bluff=0.3, signal_prob=0.0, aggression=0.25, p_lie=0.25,
        p_signal_late=0.5, p_signal_partial=0.3, p_ignore_partner=0.1, p_overtrump=0.15,
        p_burn=0.05, p_fold_won=0.1, p_bad_facedown=0.1, p_bad_eleven=0.35)),
    ("wr=0.944 (17/18)", dict(bluff=0.05, signal_prob=0.6, aggression=0.25, p_lie=0.1,
        p_signal_late=0.35, p_signal_partial=0.4, p_contradict=0.4, p_ignore_partner=0.05,
        p_overtrump=0.15, p_burn=0.5, p_fold_won=0.25, p_overkill=0.25, p_bad_eleven=0.1)),
    ("wr=0.944 (17/18)", dict(bluff=0.2, signal_prob=0.2, aggression=0.5, p_lie=0.5,
        p_signal_late=0.5, p_signal_partial=0.15, p_contradict=0.3, p_ignore_partner=0.25,
        p_overtrump=0.2, p_burn=0.2, p_overkill=0.05, p_bad_facedown=0.15, p_bad_eleven=0.05)),
    ("wr=0.944 (17/18)", dict(bluff=0.3, signal_prob=0.8, aggression=0.5, p_lie=0.35,
        p_signal_late=0.15, p_signal_partial=0.15, p_contradict=0.35, p_ignore_partner=0.25,
        p_overtrump=0.1, p_burn=0.3, p_fold_won=0.2, p_overkill=0.4, p_bad_eleven=0.45)),
    ("wr=0.941 (16/17)", dict(bluff=0.3, signal_prob=0.6, p_lie=0.1, p_signal_late=0.05,
        p_signal_partial=0.15, p_ignore_partner=0.35, p_overtrump=0.2, p_burn=0.1, p_fold_won=0.35,
        p_overkill=0.05, p_bad_facedown=0.15, p_bad_eleven=0.15)),
    ("wr=0.941 (16/17)", dict(bluff=0.5, signal_prob=0.4, aggression=-0.25, p_lie=0.25,
        p_signal_partial=0.05, p_contradict=0.1, p_ignore_partner=0.3, p_burn=0.1, p_fold_won=0.3,
        p_overkill=0.2, p_bad_eleven=0.45)),
    ("wr=0.938 (15/16)", dict(bluff=0.2, signal_prob=0.6, aggression=0.5, p_lie=0.4,
        p_signal_late=0.3, p_signal_partial=0.3, p_contradict=0.4, p_ignore_partner=0.25,
        p_overtrump=0.25, p_burn=0.1, p_fold_won=0.35, p_overkill=0.15, p_bad_facedown=0.1,
        p_bad_eleven=0.25)),
    ("wr=0.938 (15/16)", dict(bluff=0.4, signal_prob=0.8, aggression=0.5, p_lie=0.35,
        p_contradict=0.3, p_ignore_partner=0.4, p_overtrump=0.1, p_burn=0.15, p_overkill=0.45,
        p_bad_eleven=0.2)),
]


@dataclass
class PickStats:
    decisions: int = 0          # learner moves made
    illegal_top: int = 0        # decisions whose most-likely token was NOT a legal move
    tokens_skipped: int = 0     # total higher-prob tokens stepped over to reach a legal move


@torch.no_grad()
def greedy_legal_pick(net, stream, mask, cfg, device, stats: PickStats) -> tuple[int, str | None]:
    """The model's move: its most-likely next token, falling through illegal tokens to the next.

    Reads the full next-token logits at the decision (TO_ACT) position, ranks the *entire*
    vocabulary by probability, and returns the first token that is a legal move. Returns
    `(action_index, note)` where `note` is None if the top token was already that legal move,
    else a short string naming the illegal top token that was skipped over.
    """
    lm, _ = batch_last_outputs(net, [stream], cfg.block_size, device)   # [1, vocab]
    order = torch.argsort(lm[0], descending=True).tolist()             # full-vocab ranking
    legal = {i for i, m in enumerate(mask) if m > 0}

    stats.decisions += 1
    skipped: list[int] = []
    for tok in order:
        a_idx = tok - cfg.move_base
        if 0 <= a_idx < cfg.action_size and a_idx in legal:
            if skipped:
                stats.illegal_top += 1
                stats.tokens_skipped += len(skipped)
                note = (f"model's top token was {token_name(skipped[0])} (illegal) — "
                        f"fell through {len(skipped)} token(s) to next legal move")
                return a_idx, note
            return a_idx, None
        skipped.append(tok)
    raise RuntimeError("no legal move token found in the vocabulary (should be impossible)")


def play_and_narrate(net, cfg, config: BotConfig, seed: int, dealer: int, device: str = "cpu"):
    """Drive one match (pretrained model on LEARNER seats, ConfigurableBot elsewhere)."""
    learner_team = team_of(LEARNER[0])
    rng = random.Random(seed)
    other = [s for s in range(4) if s not in LEARNER]
    opponents = {s: ConfigurableBot(config, rng=random.Random(rng.random())) for s in other}

    env = TrucoEnv(seed=seed, dealer=dealer, reward_shaping=True)
    tok = MatchTokenizer()
    tok.start(env.state)

    set_pts = [0, 0]
    final_set_pts = [0, 0]
    stats = PickStats()
    lines: list[str] = []
    while not env.done:
        actor = env.agent_selection
        tok.begin_turn(actor)
        if actor in LEARNER:
            stream = tok.streams[actor]
            mask = legal_mask(env.observe(actor))
            a_idx, note = greedy_legal_pick(net, stream, mask, cfg, device, stats)
            action = action_from_index(a_idx)
            events = env.step(action)
            tok.ingest(env.state, events)
            new_lines = narrate(events, human_seat=0)
            if note and new_lines:
                new_lines[-1] += f"   ⚠️ [{note}]"
            lines += new_lines
        else:
            action = opponents[actor].act(env.observe(actor))
            events = env.step(action)
            tok.ingest(env.state, events)
            lines += narrate(events, human_seat=0)
        for e in events:
            if isinstance(e, HandResolved):
                set_pts[e.winner_team] += e.value
            elif isinstance(e, SetWon):
                final_set_pts = list(set_pts)
                set_pts = [0, 0]

    st = env.state
    return st.winner, final_set_pts, lines, st.hand_number, stats


LEGEND = (
    "Legend:  Us = pretrained DL model (seats 0 & 2)   Them = tuned ConfigurableBot (seats 1 & 3)\n"
    "         \"You\"+\"Partner\" = the two DL agents · \"Opp-1\"+\"Opp-2\" = the two heuristic agents\n"
    "         Model plays its most-likely next token; if illegal, it falls through to the next\n"
    "         legal move (⚠️ marks a turn where that fallback fired)."
)


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/pretrain.pt"
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        Path(__file__).resolve().parent.parent / "samples" / "pretrain_vs_configs.txt")
    device = "cpu"

    net, meta = load(ckpt, map_location=device)
    net = net.to(device).eval()
    cfg = net.cfg

    full: list[str] = []
    full.append("=" * 78)
    full.append(f"Pretrained TrucoGPT ({Path(ckpt).name})  vs  10 tuned ConfigurableBot heuristics")
    full.append(f"checkpoint meta: {meta}")
    full.append(LEGEND)
    full.append("=" * 78)

    rows = []
    match_no = 0
    for ci, (label, knobs) in enumerate(CONFIGS):
        config = BotConfig(**knobs)
        full.append("")
        full.append("#" * 78)
        full.append(f"# HEURISTIC {ci + 1}/10   [tournament {label}]")
        full.append(f"#   knobs: {knobs}")
        full.append("#" * 78)
        for j in range(n_per):
            match_no += 1
            seed = SEED0 + ci * 1000 + j
            dealer = match_no % 4
            winner, scores, lines, hands, stats = play_and_narrate(
                net, cfg, config, seed, dealer, device)
            dl_won = winner == team_of(LEARNER[0])
            us, them = scores[0], scores[1]
            rows.append((ci + 1, label, j + 1, seed, dealer, dl_won, us, them, hands, stats))
            full.append("")
            full.append(f"  ── Match {j + 1}/{n_per}  (seed {seed}, dealer seat {dealer}) ──")
            full.extend(lines)
            il = (f"; model played {stats.illegal_top}/{stats.decisions} moves via "
                  f"illegal-token fallback ({stats.tokens_skipped} tokens skipped)"
                  if stats.illegal_top else
                  f"; all {stats.decisions} model moves were its top token (no fallback)")
            full.append(f"  ── RESULT: {'DL WINS' if dl_won else 'HEURISTIC WINS'}  "
                        f"(Us {us} – {them} Them, over {hands} hands{il})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(full))

    # ---- console output: full transcripts + summary ----
    print("\n".join(full))
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(LEGEND)
    print()
    hdr = (f"{'#':>2}  {'tournament wr':>16}  {'M':>2}  {'winner':>9}  "
           f"{'score(Us-Them)':>14}  {'hands':>5}  {'illegal-top':>11}")
    print(hdr)
    print("-" * len(hdr))
    wins = 0
    for (ci, label, j, seed, dealer, dl_won, us, them, hands, stats) in rows:
        wins += int(dl_won)
        ill = f"{stats.illegal_top}/{stats.decisions}"
        print(f"{ci:>2}  {label:>16}  {j:>2}  {'DL' if dl_won else 'Heuristic':>9}  "
              f"{us:>6} – {them:<5}  {hands:>5}  {ill:>11}")
    print("-" * len(hdr))
    n = len(rows)
    tot_dec = sum(r[9].decisions for r in rows)
    tot_ill = sum(r[9].illegal_top for r in rows)
    print(f"Pretrained-model record: {wins}–{n - wins}  (win-rate {wins / n:.0%})")
    print(f"Illegal-top-token fallback fired on {tot_ill}/{tot_dec} model decisions "
          f"({tot_ill / max(tot_dec, 1):.1%}) across all matches.")
    print(f"full move-by-move transcript → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

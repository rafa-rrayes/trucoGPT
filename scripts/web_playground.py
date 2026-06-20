"""Interactive web playground for TrucoGPT.

Watch a real match between the trained model (seats 0 & 2) and the heuristic bot (seats 1 & 3),
then pick any of the model's decision points, EDIT the token stream the model reads, and watch its
predicted move distribution change live. The model is seat-relative, so you can also inspect what it
*would* do sitting in a heuristic seat.

Pure standard-library HTTP server (no FastAPI/Flask). Model inference runs in-process on CPU.

Run:  uv run python scripts/web_playground.py [--ckpt PATH] [--port 8000]
Then open http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from truco import engine
from truco.actions import SIGNAL_LABEL, AcceptTruco, CallTruco, ElevenChoice, PlayCard, ReraiseTruco, RunTruco, Signal
from truco.agents import HeuristicAgent
from truco.cards import card_from_id, is_zap, manilha_rank
from truco.observation import observe
from truco.rl import vocab
from truco.rl.action_space import ACTION_SIZE, action_from_index, legal_mask
from truco.rl.model import load, pick_device
from truco.rl.rollout import batch_last_outputs
from truco.rl.tokenizer import MatchTokenizer
from truco.state import team_of

HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "playground.html"

# globals set in main()
NET = None
CFG = None
META: dict = {}
DEVICE = "cpu"
CKPT_NAME = ""

MODEL_SEATS = (0, 2)
SEAT_NAMES = {0: "Model-0", 1: "Heuristic-1", 2: "Model-2", 3: "Heuristic-3"}


# --------------------------------------------------------------------------- token labels

def _action_label(idx: int) -> str:
    """Friendly description of an action-space index (0..93)."""
    a = action_from_index(idx)
    if isinstance(a, PlayCard):
        return f"play {card_from_id(a.card.id)}" + (" (face-down)" if a.face_down else "")
    if isinstance(a, Signal):
        return f'signal: "{SIGNAL_LABEL[int(a.code)]}"'
    if isinstance(a, CallTruco):
        return "call Truco / raise"
    if isinstance(a, AcceptTruco):
        return "accept"
    if isinstance(a, RunTruco):
        return "run / fold"
    if isinstance(a, ReraiseTruco):
        return "re-raise"
    if isinstance(a, ElevenChoice):
        return "Eleven: play" if a.play else "Eleven: skip"
    return f"action {idx}"


_FRIENDLY_SINGLE = {
    vocab.PAD: "·pad", vocab.BOS: "▶ match start", vocab.EOS: "■ match end",
    vocab.TO_ACT: "❓ YOU ACT", vocab.HAND_START: "hand dealt", vocab.MY_HAND: "my hand:",
    vocab.PARTNER_HAND: "partner hand:", vocab.SCORES: "scores:", vocab.ROUND_END: "round end",
    vocab.HAND_END: "hand end", vocab.SET_END: "set end", vocab.MATCH_END: "match end",
    vocab.REDEAL: "redeal", vocab.ELEVEN_START: "Eleven start", vocab.ELEVEN_DECISION: "Eleven decision",
    vocab.BY_RIGHT: "→ right plays", vocab.BY_PARTNER: "↑ partner plays", vocab.BY_LEFT: "← left plays",
    vocab.RESULT_US: "result: US", vocab.RESULT_THEM: "result: THEM", vocab.RESULT_TIE: "result: tie",
    vocab.REASON_ROUNDS: "by rounds", vocab.REASON_RUN: "by fold", vocab.REASON_ELEVEN_SKIP: "Eleven skipped",
    vocab.DECIDED_PLAY: "decided: play", vocab.DECIDED_SKIP: "decided: skip",
    vocab.PLAY_HIDDEN: "plays face-down (hidden)",
}
_SEATREL_NAME = {0: "me", 1: "right", 2: "partner", 3: "left"}


def token_label(tid: int) -> str:
    if tid in _FRIENDLY_SINGLE:
        return _FRIENDLY_SINGLE[tid]
    if vocab.MANILHA_BASE <= tid < vocab.MANILHA_BASE + 10:
        from truco.cards import RANK_ORDER
        return f"manilha = {RANK_ORDER[tid - vocab.MANILHA_BASE]}"
    if vocab.NUM_BASE <= tid <= vocab.NUM_BASE + 12:
        return f"#{tid - vocab.NUM_BASE}"
    if vocab.SEATREL_BASE <= tid < vocab.SEATREL_BASE + 4:
        return f"seat: {_SEATREL_NAME[tid - vocab.SEATREL_BASE]}"
    if vocab.CARD_BASE <= tid < vocab.CARD_BASE + 40:
        return f"card {card_from_id(tid - vocab.CARD_BASE)}"
    idx = vocab.move_action_index(tid)
    if idx is not None:
        return _action_label(idx)
    return f"<{tid}>"


def token_group(tid: int) -> str:
    if tid in (vocab.BOS, vocab.EOS, vocab.TO_ACT, vocab.HAND_START, vocab.MY_HAND,
               vocab.PARTNER_HAND, vocab.SCORES, vocab.ROUND_END, vocab.HAND_END,
               vocab.SET_END, vocab.MATCH_END, vocab.REDEAL, vocab.ELEVEN_START,
               vocab.ELEVEN_DECISION, vocab.PAD):
        return "structural"
    if tid in (vocab.BY_RIGHT, vocab.BY_PARTNER, vocab.BY_LEFT):
        return "actor"
    if tid in (vocab.RESULT_US, vocab.RESULT_THEM, vocab.RESULT_TIE,
               vocab.REASON_ROUNDS, vocab.REASON_RUN, vocab.REASON_ELEVEN_SKIP,
               vocab.DECIDED_PLAY, vocab.DECIDED_SKIP):
        return "result"
    if vocab.MANILHA_BASE <= tid < vocab.MANILHA_BASE + 10:
        return "manilha"
    if vocab.NUM_BASE <= tid <= vocab.NUM_BASE + 12:
        return "number"
    if vocab.SEATREL_BASE <= tid < vocab.SEATREL_BASE + 4:
        return "seat"
    if vocab.CARD_BASE <= tid < vocab.CARD_BASE + 40:
        return "card"
    if tid == vocab.PLAY_HIDDEN:
        return "move"
    if vocab.move_action_index(tid) is not None:
        return "move"
    return "other"


def vocab_payload() -> dict:
    tokens = [{"id": t, "name": vocab.token_name(t), "label": token_label(t), "group": token_group(t)}
              for t in range(vocab.VOCAB_SIZE)]
    return {
        "tokens": tokens,
        "MOVE_BASE": vocab.MOVE_BASE,
        "ACTION_SIZE": ACTION_SIZE,
        "VOCAB_SIZE": vocab.VOCAB_SIZE,
        "TO_ACT": vocab.TO_ACT,
        "actions": [{"index": i, "token": vocab.MOVE_BASE + i, "label": _action_label(i)}
                    for i in range(ACTION_SIZE)],
    }


# --------------------------------------------------------------------------- prediction

@torch.no_grad()
def predict(stream: list[int], legal_mask_arr: list[float] | None = None) -> dict:
    if not stream:
        return {"empty": True}
    lm, value = batch_last_outputs(NET, [stream], CFG.block_size, DEVICE)
    logits = lm[0]                                   # [vocab]
    probs = torch.softmax(logits.float(), dim=-1)

    # top-k over the FULL vocabulary
    k = 12
    tp, ti = torch.topk(probs, k)
    full_topk = [{"token": int(ti[j]), "name": vocab.token_name(int(ti[j])),
                  "label": token_label(int(ti[j])), "prob": float(tp[j])}
                 for j in range(k)]

    # the action block, renormalised so the move-only distribution sums to 1
    a_logits = logits[vocab.MOVE_BASE:vocab.MOVE_BASE + ACTION_SIZE].float()
    a_probs = torch.softmax(a_logits, dim=-1)
    actions = []
    for i in range(ACTION_SIZE):
        legal = None if legal_mask_arr is None else bool(legal_mask_arr[i] > 0)
        actions.append({"index": i, "label": _action_label(i), "prob": float(a_probs[i]), "legal": legal})
    actions.sort(key=lambda d: d["prob"], reverse=True)

    # masked argmax (what the model would legally play), if a mask is supplied
    legal_pick = None
    if legal_mask_arr is not None and any(m > 0 for m in legal_mask_arr):
        masked = a_logits.clone()
        for i, m in enumerate(legal_mask_arr):
            if m <= 0:
                masked[i] = float("-inf")
        legal_pick = int(torch.argmax(masked))

    return {
        "value": float(value[0]),
        "full_topk": full_topk,
        "actions": actions,
        "argmax_action": int(torch.argmax(a_probs)),
        "legal_pick": legal_pick,
        "prob_mass_on_moves": float(probs[vocab.MOVE_BASE:vocab.MOVE_BASE + ACTION_SIZE].sum()),
        "stream_len": len(stream),
    }


# --------------------------------------------------------------------------- match replay

def _card_obj(card) -> dict:
    return {"id": card.id, "str": str(card)}


def _board_snapshot(obs, manilha_rk: str, hand_number: int) -> dict:
    table = []
    played = {vp.seat: vp for vp in obs.table}
    for s in range(4):
        vp = played.get(s)
        if vp is None:
            cell = {"seat": s, "name": SEAT_NAMES[s], "card": None, "face_down": False}
        elif vp.card is None:
            cell = {"seat": s, "name": SEAT_NAMES[s], "card": "face-down", "face_down": True}
        else:
            cell = {"seat": s, "name": SEAT_NAMES[s], "card": str(vp.card), "face_down": vp.face_down}
        table.append(cell)
    my_hand = [{"str": str(c), "id": c.id, "manilha": c.rank == manilha_rk, "zap": is_zap(c, manilha_rk)}
               for c in obs.my_hand]
    us, them = obs.team, 1 - obs.team
    return {
        "hand_number": hand_number,
        "seat": obs.seat,
        "seat_name": SEAT_NAMES[obs.seat],
        "dealer": obs.dealer,
        "dealer_name": SEAT_NAMES[obs.dealer],
        "vira": _card_obj(obs.vira),
        "manilha": manilha_rk,
        "round": obs.current_round + 1,
        "scores": {"us": obs.set_scores[us], "them": obs.set_scores[them],
                   "us_sets": obs.match_sets[us], "them_sets": obs.match_sets[them]},
        "stake_value": obs.value,
        "table": table,
        "round_results": [("us" if r == obs.team else "them" if r in (0, 1) else "tie")
                          for r in obs.round_results],
        "my_hand": my_hand,
        "my_signals": [SIGNAL_LABEL[c] for c in obs.my_signals],
        "partner_signals": [SIGNAL_LABEL[c] for c in obs.partner_signals],
    }


@torch.no_grad()
def _model_action(stream: list[int], mask: list[float]) -> int:
    lm, _ = batch_last_outputs(NET, [stream], CFG.block_size, DEVICE)
    order = torch.argsort(lm[0], descending=True).tolist()
    legal = {i for i, m in enumerate(mask) if m > 0}
    for tok in order:
        a = tok - vocab.MOVE_BASE
        if 0 <= a < ACTION_SIZE and a in legal:
            return a
    return next(iter(legal))


def play_match(seed: int, focus_seat: int) -> dict:
    rng = random.Random(seed)
    heuristics = {s: HeuristicAgent(rng=random.Random(rng.random()))
                  for s in range(4) if s not in MODEL_SEATS}
    state = engine.new_match(rng, dealer=seed % 4)
    tok = MatchTokenizer()
    tok.start(state)

    decisions = []
    hand_number = 1
    while not state.done:
        actor = state.current_actor
        tok.begin_turn(actor)
        obs = observe(state, actor)
        mask = legal_mask(obs)

        if actor in MODEL_SEATS:
            a_idx = _model_action(tok.streams[actor], mask)
            action = action_from_index(a_idx)
            controller = "model"
        else:
            action = heuristics[actor].act(obs)
            a_idx = None
            controller = "heuristic"

        if actor == focus_seat:
            # snapshot the stream EXACTLY as the model would read it to choose now (ends in TO_ACT)
            mrank = manilha_rank(state.hand.vira) if state.hand else obs.manilha
            decisions.append({
                "index": len(decisions),
                "stream": list(tok.streams[actor]),
                "legal_mask": mask,
                "controller": controller,
                "board": _board_snapshot(obs, mrank, hand_number),
            })

        state, events = engine.apply_action(state, action, rng)
        tok.ingest(state, events)

        # record the actual action that was taken for the focus decision just captured
        if actor == focus_seat and decisions:
            taken = a_idx if a_idx is not None else _resolve_taken_index(action)
            decisions[-1]["actual_action"] = taken

        if any(isinstance(e, engine.HandDealt) for e in events):
            hand_number += 1   # next hand begins for subsequent decisions

    return {
        "seed": seed,
        "focus_seat": focus_seat,
        "focus_seat_name": SEAT_NAMES[focus_seat],
        "dealer": seed % 4,
        "winner_team": state.winner,
        "winner_label": "MODEL" if state.winner == team_of(MODEL_SEATS[0]) else "HEURISTIC",
        "meta": {"ckpt": CKPT_NAME, "win_rate": META.get("win_rate"),
                 "win_rate_vs_default": META.get("win_rate_vs_default")},
        "decisions": decisions,
    }


def _resolve_taken_index(action) -> int:
    from truco.rl.action_space import action_index
    try:
        return action_index(action)
    except Exception:
        return -1


# --------------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            html = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if u.path == "/api/vocab":
            self._json(vocab_payload())
            return
        if u.path == "/api/match":
            q = parse_qs(u.query)
            seed = int(q.get("seed", ["42"])[0])
            seat = int(q.get("seat", ["0"])[0]) % 4
            try:
                self._json(play_match(seed, seat))
            except Exception as e:
                self._json({"error": repr(e)}, 500)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/predict":
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n) or b"{}")
            stream = [int(t) for t in data.get("stream", [])]
            mask = data.get("legal_mask")
            try:
                self._json(predict(stream, mask))
            except Exception as e:
                self._json({"error": repr(e)}, 500)
            return
        self._json({"error": "not found"}, 404)


def main() -> int:
    global NET, CFG, META, DEVICE, CKPT_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/rl_legality.pt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    args = ap.parse_args()

    DEVICE = args.device
    CKPT_NAME = Path(args.ckpt).name
    print(f"loading {args.ckpt} on {DEVICE} …")
    NET, META = load(args.ckpt, map_location=DEVICE)
    NET = NET.to(DEVICE).eval()
    CFG = NET.cfg
    print(f"model ready: d_model={CFG.d_model} n_layer={CFG.n_layer}  "
          f"win_rate={META.get('win_rate')}")

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\n  ▶ TrucoGPT playground:  http://localhost:{args.port}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rooms, seating, and the live game loop for the web platform.

A :class:`Room` owns one match. Seats are filled by humans (identified by a ``pid``) or
AI players (``"model"`` / ``"heuristic"``). The room drives play with :meth:`Room._advance`:
it walks the engine forward, letting AI seats act automatically (with a short delay so the
table animates), and pausing whenever a human must act. After every transition it
broadcasts each connected player their own privacy-correct view.

Everything that mutates a room runs under ``Room.lock`` (an ``asyncio.Lock``), so a human
action and the AI auto-advance can never interleave. Model inference is pushed to a worker
thread so it never blocks the event loop.

The narration log is built per seat with :func:`truco.cli.render.narrate`, which already
honours Truco's privacy rules (an opponent's signal is never shown; someone else's
face-down card is "a card face-down"). Spectators get a fully public log.
"""

from __future__ import annotations

import asyncio
import copy
import random
from dataclasses import dataclass, field

from .. import engine
from ..actions import PlayCard
from ..cli.render import narrate
from ..engine import (
    CardPlayed,
    HandDealt,
    HandResolved,
    MatchWon,
    Redeal,
    RoundResolved,
    SetWon,
    TrucoAnswered,
    TrucoCalled,
)
from ..observation import observe
from ..rl.action_space import action_from_index, legal_mask
from ..rl.tokenizer import MatchTokenizer
from ..state import Phase, PlayedCard, RAISE_NAMES, VALUES, team_of
from . import ai, render, serialize

_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no easily-confused chars
_LOG_LIMIT = 400
AI_NAMES = {"model": "TrucoGPT", "heuristic": "Bot"}


@dataclass
class Player:
    pid: str
    name: str
    seat: int | None = None
    connected: bool = False


class Room:
    def __init__(self, code: str, host_pid: str, *, ai_delay: float = 1.2,
                 round_delay: float = 2.0, model_ckpt: str | None = None,
                 model_device: str = "cpu") -> None:
        self.code = code
        self.host_pid = host_pid
        self.members: dict[str, Player] = {}
        self.seats: list[str | None] = [None, None, None, None]   # seat -> human pid
        self.ai_seats: dict[int, str] = {}                        # seat -> "model"|"heuristic"
        self.heuristics: dict[int, object] = {}                   # seat -> HeuristicAgent
        self.connections: dict[str, set] = {}                     # pid -> {WebSocket}
        self.phase = "lobby"                                       # lobby | playing | finished
        self.state = None
        self.tok: MatchTokenizer | None = None
        self.rng = random.Random()
        self.logs: dict[int, list[str]] = {s: [] for s in range(4)}
        self.public_log: list[str] = []
        self.winner: int | None = None
        self.lock = asyncio.Lock()
        self.ai_delay = ai_delay
        self.round_delay = round_delay
        self.model_ckpt = model_ckpt
        self.model_device = model_device
        self._sets_to_win = 1

    # --------------------------------------------------------------- membership
    def ensure_member(self, pid: str, name: str) -> Player:
        p = self.members.get(pid)
        if p is None:
            p = Player(pid=pid, name=name or "Player")
            self.members[pid] = p
        elif name:
            p.name = name
        return p

    def connect(self, pid: str, ws) -> None:
        self.connections.setdefault(pid, set()).add(ws)
        if pid in self.members:
            self.members[pid].connected = True

    def disconnect(self, pid: str, ws) -> None:
        conns = self.connections.get(pid)
        if conns:
            conns.discard(ws)
        if not self.connections.get(pid):
            if pid in self.members:
                self.members[pid].connected = False

    def _seat_kind(self, seat: int) -> str:
        if self.seats[seat] is not None:
            return "human"
        if seat in self.ai_seats:
            return self.ai_seats[seat]
        return "empty"

    def _occupant(self, seat: int):
        """Return (occupant, name, connected) for a seat."""
        pid = self.seats[seat]
        if pid is not None:
            p = self.members.get(pid)
            return "human", (p.name if p else "Player"), bool(p and p.connected)
        if seat in self.ai_seats:
            kind = self.ai_seats[seat]
            return kind, AI_NAMES[kind], True
        return "empty", "", False

    def _seats_filled(self) -> int:
        return sum(1 for s in range(4) if self._seat_kind(s) != "empty")

    # --------------------------------------------------------------- commands
    async def handle_command(self, pid: str, msg: dict) -> None:
        cmd = msg.get("cmd")
        handler = {
            "sit": self._cmd_sit, "stand": self._cmd_stand,
            "add_ai": self._cmd_add_ai, "remove_ai": self._cmd_remove_ai,
            "start": self._cmd_start, "action": self._cmd_action,
            "new_match": self._cmd_new_match, "set_name": self._cmd_set_name,
        }.get(cmd)
        if handler is None:
            return
        await handler(pid, msg)

    async def _cmd_set_name(self, pid, msg):
        name = (msg.get("name") or "").strip()
        if name and pid in self.members:
            self.members[pid].name = name[:24]
            await self.broadcast()

    async def _cmd_sit(self, pid, msg):
        async with self.lock:
            if self.phase != "lobby":
                return await self.send_toast(pid, "Can't change seats mid-match.", "error")
            seat = _as_seat(msg.get("seat"))
            if seat is None or self._seat_kind(seat) != "empty":
                return await self.send_toast(pid, "That seat is taken.", "error")
            self._vacate_human(pid)
            self.seats[seat] = pid
            self.members[pid].seat = seat
        await self.broadcast()

    async def _cmd_stand(self, pid, msg):
        async with self.lock:
            if self.phase != "lobby":
                return await self.send_toast(pid, "Can't stand mid-match.", "error")
            self._vacate_human(pid)
        await self.broadcast()

    async def _cmd_add_ai(self, pid, msg):
        async with self.lock:
            if self.phase != "lobby":
                return await self.send_toast(pid, "Can't add AI mid-match.", "error")
            seat = _as_seat(msg.get("seat"))
            kind = msg.get("kind")
            if seat is None or self._seat_kind(seat) != "empty":
                return await self.send_toast(pid, "That seat is taken.", "error")
            if kind not in ai.AI_KINDS:
                kind = "model"
            self.ai_seats[seat] = kind
        await self.broadcast()

    async def _cmd_remove_ai(self, pid, msg):
        async with self.lock:
            if self.phase != "lobby":
                return
            seat = _as_seat(msg.get("seat"))
            if seat is not None:
                self.ai_seats.pop(seat, None)
        await self.broadcast()

    async def _cmd_start(self, pid, msg):
        async with self.lock:
            if self.phase != "lobby":
                return
            if pid != self.host_pid:
                return await self.send_toast(pid, "Only the host can start.", "error")
            if self._seats_filled() != 4:
                return await self.send_toast(pid, "All four seats must be filled.", "error")
            self._start_match()
            await self._advance()

    async def _cmd_new_match(self, pid, msg):
        async with self.lock:
            if self.phase != "finished":
                return
            self._start_match()
            await self._advance()

    async def _cmd_action(self, pid, msg):
        async with self.lock:
            if self.phase != "playing" or self.state is None:
                return
            seat = self.members[pid].seat if pid in self.members else None
            if seat is None or self.state.current_actor != seat:
                return await self.send_toast(pid, "It's not your turn.", "error")
            try:
                action = action_from_index(int(msg.get("index")))
            except (TypeError, ValueError):
                return await self.send_toast(pid, "Invalid move.", "error")
            if action not in engine.legal_actions(self.state):
                return await self.send_toast(pid, "That move isn't legal.", "error")
            self._apply(action)
            await self._advance()

    # --------------------------------------------------------------- game loop
    def _start_match(self) -> None:
        self.rng = random.Random()
        dealer = self.rng.randint(0, 3)
        self.state = engine.new_match(self.rng, dealer=dealer, sets_to_win=self._sets_to_win)
        self.tok = MatchTokenizer()
        self.tok.start(self.state)
        self.logs = {s: [] for s in range(4)}
        self.public_log = []
        self.winner = None
        self.phase = "playing"
        self.heuristics = {}
        from ..agents import HeuristicAgent
        for seat, kind in self.ai_seats.items():
            if kind == "heuristic":
                self.heuristics[seat] = HeuristicAgent(rng=random.Random(self.rng.random()))
        self._record(f"— Match started · dealer {self._name_of(dealer)}.")

    def _apply(self, action) -> list:
        """Apply one action by the current actor (its TO_ACT was already emitted)."""
        actor = self.state.current_actor
        self.state, events = engine.apply_action(self.state, action, self.rng)
        self.tok.ingest(self.state, events)
        self._log_events(events)
        return events

    async def _advance(self) -> None:
        """Walk the game forward, auto-playing AI seats, until a human must act / game ends."""
        while True:
            if self.state.done:
                self._finish()
                await self.broadcast()
                return
            actor = self.state.current_actor
            self.tok.begin_turn(actor)
            kind = self._seat_kind(actor)
            await self.broadcast()
            if kind in ("human", "empty"):
                return  # wait for the human (an "empty" actor shouldn't happen once started)
            await asyncio.sleep(self.ai_delay)
            try:
                action = await self._ai_action(actor)
            except Exception as e:  # never let a bad AI move kill the room
                action = self._fallback_action()
                self._record(f"(AI error: {e!r} — playing a safe move)")
            # The engine clears the trick the instant the 4th card lands, so snapshot
            # the table *before* applying and, if the round resolved, hold that frozen
            # view (all four cards + the result) on screen before play moves on.
            frozen = copy.deepcopy(self.state) if self._completes_trick(action) else None
            events = self._apply(action)
            if frozen is not None:
                rr = next((e for e in events if isinstance(e, RoundResolved)), None)
                if rr is not None:
                    self._freeze_trick(frozen, actor, action, rr)
                    await self.broadcast(state=frozen)
                    await asyncio.sleep(self.round_delay)

    def _completes_trick(self, action) -> bool:
        """True if `action` is the fourth card of the current round (so it ends the trick)."""
        h = self.state.hand
        return (h is not None and self.state.phase is Phase.PLAY
                and len(h.plays) == 3 and isinstance(action, PlayCard))

    @staticmethod
    def _freeze_trick(snap, actor, action, rr: "RoundResolved") -> None:
        """Mutate a pre-apply state copy into the 'trick just completed' view: lay the
        fourth card on the table, record the round result, and clear the active turn so
        the board reads as a settled trick rather than someone's move."""
        h = snap.hand
        h.plays.append(PlayedCard(actor, action.card, action.face_down))
        h.round_results.append(rr.result_team)
        h.turn = -1  # no one is "to act" while the finished trick is on display

    async def _ai_action(self, actor: int):
        obs = observe(self.state, actor)
        kind = self.ai_seats[actor]
        if kind == "heuristic":
            return self.heuristics[actor].act(obs)
        # model: read the seat's token stream and pick a legal move off the event loop
        mask = legal_mask(obs)
        legal_idx = [i for i, m in enumerate(mask) if m > 0]
        stream = list(self.tok.streams[actor])
        eng = ai.get_model_engine(self.model_ckpt, self.model_device)
        loop = asyncio.get_event_loop()
        idx = await loop.run_in_executor(None, eng.pick, stream, legal_idx)
        return action_from_index(idx)

    def _fallback_action(self):
        return engine.legal_actions(self.state)[0]

    def _finish(self) -> None:
        self.phase = "finished"
        self.winner = self.state.winner

    # --------------------------------------------------------------- logging
    def _log_events(self, events) -> None:
        for s in range(4):
            self.logs[s].extend(narrate(events, s))
            self.logs[s] = self.logs[s][-_LOG_LIMIT:]
        self.public_log.extend(_public_narrate(events))
        self.public_log = self.public_log[-_LOG_LIMIT:]

    def _record(self, line: str) -> None:
        for s in range(4):
            self.logs[s].append(line)
        self.public_log.append(line)

    def _name_of(self, seat: int) -> str:
        return self._occupant(seat)[1] or f"Seat {seat}"

    # --------------------------------------------------------------- views
    def build_ctx(self, pid: str, state=None) -> dict:
        # `state` lets callers render a transient view (e.g. a frozen, just-completed
        # trick) instead of the live state, without mutating the room.
        st = self.state if state is None else state
        me = self.members.get(pid)
        seat = me.seat if me else None
        viewer_team = team_of(seat) if seat is not None else 0
        cur = st.current_actor if (st is not None and st.hand) else None

        seats = []
        for s in range(4):
            occ, name, connected = self._occupant(s)
            seats.append({
                "seat": s, "team": team_of(s),
                "teamName": "us" if team_of(s) == viewer_team else "them",
                "occupant": occ, "name": name, "connected": connected,
                "isYou": (seat == s), "isHost": (self.seats[s] == self.host_pid),
                "isDealer": (st.dealer == s) if st is not None else False,
                "isCurrent": (cur == s),
                "canSit": self.phase == "lobby" and occ == "empty",
                "canRemoveAi": self.phase == "lobby" and occ in ai.AI_KINDS,
            })

        game = None
        if st is not None:
            game = serialize.game_view(st, seat) if seat is not None \
                else serialize.public_view(st)

        log = self.logs[seat] if seat is not None else self.public_log
        return {
            "code": self.code,
            "phase": self.phase,
            "you": {
                "pid": pid, "name": me.name if me else "",
                "seat": seat, "isHost": pid == self.host_pid,
                "isSeated": seat is not None, "isSpectator": seat is None,
            },
            "seats": seats,
            "canStart": self.phase == "lobby" and self._seats_filled() == 4,
            "seatsFilled": self._seats_filled(),
            "settings": {"setsToWin": self._sets_to_win},
            "game": game,
            "log": log,
            "toast": None,
            "winner": self._winner_word(viewer_team),
            "winnerName": self._winner_name(),
        }

    def _winner_word(self, viewer_team: int) -> str | None:
        if self.winner is None:
            return None
        return "us" if self.winner == viewer_team else "them"

    def _winner_name(self) -> str | None:
        if self.winner is None:
            return None
        names = [self._name_of(s) for s in range(4) if team_of(s) == self.winner]
        return " & ".join(n for n in names if n)

    # --------------------------------------------------------------- transport
    async def broadcast(self, state=None) -> None:
        for pid in list(self.connections.keys()):
            if not self.connections.get(pid):
                continue
            html = render.regions(self.build_ctx(pid, state=state))
            await self._send_to(pid, html)

    async def send_toast(self, pid: str, text: str, level: str = "info") -> None:
        ctx = self.build_ctx(pid)
        ctx["toast"] = {"text": text, "level": level}
        await self._send_to(pid, render.partial("_toast.html", ctx))

    async def send_full(self, pid: str) -> None:
        """Push the current state to one player (used right after they connect)."""
        await self._send_to(pid, render.regions(self.build_ctx(pid)))

    async def _send_to(self, pid: str, html: str) -> None:
        dead = []
        for ws in list(self.connections.get(pid, ())):
            try:
                await ws.send_text(html)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.get(pid, set()).discard(ws)

    # --------------------------------------------------------------- helpers
    def _vacate_human(self, pid: str) -> None:
        for s in range(4):
            if self.seats[s] == pid:
                self.seats[s] = None
        if pid in self.members:
            self.members[pid].seat = None


def _as_seat(v) -> int | None:
    try:
        s = int(v)
    except (TypeError, ValueError):
        return None
    return s if 0 <= s < 4 else None


# ---------------------------------------------------------------- public narration

def _team_word(team: int) -> str:
    return "Team A" if team == 0 else "Team B"


def _public_narrate(events) -> list[str]:
    """Privacy-free narration for spectators: no signals, all face-down cards anonymous."""
    out: list[str] = []
    for e in events:
        if isinstance(e, HandDealt):
            out.append(f"\n— Hand #{e.hand_number} dealt · vira {e.vira} → manilha {e.manilha}.")
        elif isinstance(e, CardPlayed):
            what = "a card face-down" if e.face_down else str(e.card)
            out.append(f"  Seat {e.seat} plays {what}.")
        elif isinstance(e, TrucoCalled):
            out.append(f"  🔥 Seat {e.seat} calls {RAISE_NAMES[e.proposed_level]}! "
                       f"(stake → {VALUES[e.proposed_level]})")
        elif isinstance(e, TrucoAnswered):
            out.append(f"  Seat {e.seat} {e.response}s.")
        elif isinstance(e, RoundResolved):
            if e.result_team == -1:
                out.append(f"  Round {e.round_index + 1}: tied.")
            else:
                out.append(f"  Round {e.round_index + 1}: {_team_word(e.result_team)} win.")
        elif isinstance(e, HandResolved):
            out.append(f"  ➜ {_team_word(e.winner_team)} take the hand: +{e.value}.")
        elif isinstance(e, Redeal):
            out.append("  All three rounds tied — re-dealing.")
        elif isinstance(e, SetWon):
            out.append(f"  🏆 {_team_word(e.team)} win the SET!")
        elif isinstance(e, MatchWon):
            out.append(f"  🎉 {_team_word(e.team)} WIN THE MATCH! 🎉")
    return out


# ------------------------------------------------------------------- manager

class RoomManager:
    def __init__(self, *, ai_delay: float = 1.2, round_delay: float = 2.0,
                 model_ckpt: str | None = None, model_device: str = "cpu") -> None:
        self.rooms: dict[str, Room] = {}
        self._rng = random.Random()
        self.ai_delay = ai_delay
        self.round_delay = round_delay
        self.model_ckpt = model_ckpt
        self.model_device = model_device

    def _new_code(self) -> str:
        while True:
            code = "".join(self._rng.choice(_CODE_ALPHABET) for _ in range(4))
            if code not in self.rooms:
                return code

    def create_room(self, host_pid: str, host_name: str) -> Room:
        code = self._new_code()
        room = Room(code, host_pid, ai_delay=self.ai_delay, round_delay=self.round_delay,
                    model_ckpt=self.model_ckpt, model_device=self.model_device)
        room.ensure_member(host_pid, host_name)
        self.rooms[code] = room
        return room

    def get(self, code: str) -> Room | None:
        return self.rooms.get((code or "").upper())

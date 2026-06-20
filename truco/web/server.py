"""FastAPI app: lobby HTTP routes + the per-room game WebSocket.

Identity is a ``truco_pid`` cookie (a random id) plus a ``truco_name`` cookie, so a player
keeps their seat across reloads and reconnects. The lobby (create/join) is plain HTML
forms; everything inside a room is driven over a single WebSocket using htmx's ws
extension — the client sends ``{"cmd": ...}`` messages and the server pushes rendered HTML
region updates back.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import ai, render
from .rooms import RoomManager

_STATIC = Path(__file__).resolve().parent / "static"

_DEVICE = os.environ.get("TRUCO_WEB_DEVICE", "cpu")
_CKPT = os.environ.get("TRUCO_WEB_CKPT") or None
_AI_DELAY = float(os.environ.get("TRUCO_WEB_AI_DELAY", "1.2"))
_ROUND_DELAY = float(os.environ.get("TRUCO_WEB_ROUND_DELAY", "2.0"))

manager = RoomManager(ai_delay=_AI_DELAY, round_delay=_ROUND_DELAY,
                      model_ckpt=_CKPT, model_device=_DEVICE)

_COOKIE_AGE = 60 * 60 * 24 * 30  # 30 days


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the model off-thread so the first AI move isn't slow.
    def _warm():
        try:
            ai.get_model_engine(_CKPT, _DEVICE)
            print("[truco.web] model ready")
        except Exception as e:  # the heuristic AI still works without it
            print(f"[truco.web] model warm-up failed: {e!r}")
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="TrucoGPT", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# ------------------------------------------------------------------ identity

def _identity(request: Request) -> tuple[str, str]:
    pid = request.cookies.get("truco_pid") or f"p-{uuid.uuid4().hex[:12]}"
    name = request.cookies.get("truco_name") or ""
    return pid, name


def _set_identity(resp, pid: str, name: str) -> None:
    resp.set_cookie("truco_pid", pid, max_age=_COOKIE_AGE, httponly=True, samesite="lax")
    if name:
        resp.set_cookie("truco_name", quote(name), max_age=_COOKIE_AGE, samesite="lax")


def _read_name(raw: str) -> str:
    from urllib.parse import unquote
    return unquote(raw or "").strip()[:24]


# ------------------------------------------------------------------ HTTP routes

@app.get("/", response_class=HTMLResponse)
async def lobby(request: Request):
    return HTMLResponse(render.lobby_page())


@app.post("/create")
async def create(request: Request):
    form = await request.form()
    pid, cookie_name = _identity(request)
    name = (form.get("name") or "").strip()[:24] or _read_name(cookie_name) or "Host"
    room = manager.create_room(pid, name)
    resp = RedirectResponse(url=f"/room/{room.code}", status_code=303)
    _set_identity(resp, pid, name)
    return resp


@app.post("/join")
async def join(request: Request):
    form = await request.form()
    pid, cookie_name = _identity(request)
    name = (form.get("name") or "").strip()[:24] or _read_name(cookie_name) or "Player"
    code = (form.get("code") or "").strip().upper()
    room = manager.get(code)
    if room is None:
        return HTMLResponse(render.lobby_page(
            {"text": f"Room “{code}” was not found.", "level": "error"}))
    room.ensure_member(pid, name)
    resp = RedirectResponse(url=f"/room/{room.code}", status_code=303)
    _set_identity(resp, pid, name)
    return resp


@app.get("/room/{code}", response_class=HTMLResponse)
async def room_page(request: Request, code: str):
    room = manager.get(code)
    pid, cookie_name = _identity(request)
    if room is None:
        resp = RedirectResponse(url="/", status_code=303)
        _set_identity(resp, pid, _read_name(cookie_name))
        return resp
    name = _read_name(cookie_name) or "Player"
    room.ensure_member(pid, name)
    resp = HTMLResponse(render.room_page(room.build_ctx(pid)))
    _set_identity(resp, pid, name)
    return resp


# ------------------------------------------------------------------ WebSocket

@app.websocket("/ws/{code}")
async def game_ws(ws: WebSocket, code: str):
    room = manager.get(code)
    await ws.accept()
    if room is None:
        await ws.close(code=4004)
        return
    pid = ws.cookies.get("truco_pid")
    if not pid:
        pid = f"p-{uuid.uuid4().hex[:12]}"
    name = _read_name(ws.cookies.get("truco_name") or "") or "Player"
    room.ensure_member(pid, name)
    room.connect(pid, ws)
    try:
        await room.send_full(pid)
        await room.broadcast()  # let everyone see the new arrival / connection state
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if isinstance(msg, dict):
                await room.handle_command(pid, msg)
    except WebSocketDisconnect:
        pass
    finally:
        room.disconnect(pid, ws)
        try:
            await room.broadcast()
        except Exception:
            pass

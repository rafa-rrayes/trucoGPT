"""Jinja2 rendering: turn a context dict into HTML for the page and the WS pushes.

Templates live in ``templates/`` (built to a fixed contract; see the partial ids in
``rooms.py``). Every template receives the context under the name ``ctx``.

  * :func:`lobby_page` / :func:`room_page` render full pages (HTTP).
  * :func:`regions` renders ``partials/_update.html`` — the six region partials
    concatenated as top-level elements. The htmx ws extension out-of-band-swaps each by
    ``id``, so one push updates the whole board at once.
  * :func:`partial` renders a single partial (used for one-off toasts).
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES = Path(__file__).resolve().parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES)),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
    auto_reload=True,
)


def lobby_page(toast: dict | None = None) -> str:
    ctx = {"phase": "lobby", "code": "", "toast": toast,
           "you": {"name": "", "seat": None}, "game": None, "seats": [], "log": []}
    return _env.get_template("lobby.html").render(ctx=ctx)


def room_page(ctx: dict) -> str:
    return _env.get_template("room.html").render(ctx=ctx)


def regions(ctx: dict) -> str:
    return _env.get_template("partials/_update.html").render(ctx=ctx)


def partial(name: str, ctx: dict) -> str:
    return _env.get_template(f"partials/{name}").render(ctx=ctx)

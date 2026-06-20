"""Run the TrucoGPT web platform.

    uv run python -m truco.web [--host 0.0.0.0] [--port 8000] [--device cpu|cuda]
                              [--ckpt PATH] [--ai-delay 1.2] [--round-delay 2.0] [--reload]

Open http://localhost:8000 and create a room. Share the 4-letter code with friends, sit
down (or fill seats with the TrucoGPT / Bot AIs), and play.
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser(description="TrucoGPT web platform")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=os.environ.get("TRUCO_WEB_DEVICE", "cpu"),
                    help="torch device for the AI (cpu or cuda)")
    ap.add_argument("--ckpt", default=os.environ.get("TRUCO_WEB_CKPT"),
                    help="path to a TrucoGPT checkpoint (defaults to checkpoints/rl_legality.pt)")
    ap.add_argument("--ai-delay", type=float, default=float(os.environ.get("TRUCO_WEB_AI_DELAY", "1.2")),
                    help="seconds an AI 'thinks' before each move (table pacing)")
    ap.add_argument("--round-delay", type=float,
                    default=float(os.environ.get("TRUCO_WEB_ROUND_DELAY", "2.0")),
                    help="seconds the completed trick stays on the table before it clears")
    ap.add_argument("--reload", action="store_true", help="auto-reload for development")
    args = ap.parse_args()

    # Pass config to the app module via the environment (read at import time).
    os.environ["TRUCO_WEB_DEVICE"] = args.device
    if args.ckpt:
        os.environ["TRUCO_WEB_CKPT"] = args.ckpt
    os.environ["TRUCO_WEB_AI_DELAY"] = str(args.ai_delay)
    os.environ["TRUCO_WEB_ROUND_DELAY"] = str(args.round_delay)

    import uvicorn

    print(f"\n  ▶ TrucoGPT web:  http://localhost:{args.port}\n")
    uvicorn.run("truco.web.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

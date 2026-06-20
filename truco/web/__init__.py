"""TrucoGPT web platform — play Truco online against/with the trained AI.

A FastAPI + WebSocket server that hosts multiplayer Truco rooms. Humans and AI players
(the champion transformer or the rule-based heuristic) share the same table. The core
game logic is the pure `truco.engine`; this package only adds rooms, networking and a
browser UI on top of it.

Run:  uv run python -m truco.web   (then open http://localhost:8000)
"""

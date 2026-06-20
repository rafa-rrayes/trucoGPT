"""`python -m truco` → start an interactive match."""

from .cli.play import main

if __name__ == "__main__":
    raise SystemExit(main())

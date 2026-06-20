"""The console frontend: a renderer plus an interactive match loop.

This is the one place in the package that talks to a terminal. Everything it shows is
built from a single seat's `Observation`, so it can never leak hidden information (an
opponent's hand, a private signal, a face-down card's identity) to the screen.
"""

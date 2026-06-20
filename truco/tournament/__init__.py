"""Sampled config-vs-config tournament → a top-config self-play pretraining corpus.

The pretraining corpus should imitate *strong, varied* play, not one hand-tuned style. So we:

  1. sample many `BotConfig`s across the full knob space (`configs.sample_configs`, a Latin
     hypercube so every knob level is well covered with few samples);
  2. play thousands of config-vs-config matches and rank every config by win-rate
     (`run.run_tournament`) — the win-rate filter, not our priors, decides what's worth imitating;
  3. regenerate the corpus from the top-K configs playing each other, with both teams strong
     (`corpus.generate_topk_corpus`).

Everything is seeded for reproducibility and the match runner is pure-engine (no torch), so a
big tournament parallelises cleanly across processes.
"""

from .configs import KNOB_SPEC, sample_configs
from .corpus import generate_topk_corpus
from .match import play_config_match, tokenize_config_match
from .run import TournamentResult, run_tournament

__all__ = [
    "KNOB_SPEC", "sample_configs", "play_config_match", "tokenize_config_match",
    "run_tournament", "TournamentResult", "generate_topk_corpus",
]

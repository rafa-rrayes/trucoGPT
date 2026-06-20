"""Tests for the configurable bot, the widened señal channel, and the tournament pipeline.

The headline guarantee re-checked here is that widening signals to *several per hand* did not
open an information leak: a chatty `ConfigurableBot` that flashes its whole holding must still
produce per-seat streams that contain only what that seat may legally know (mirroring
`test_tokenizer`, but exercising multi-signal hands the canonical heuristic never makes).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import truco`

from truco import engine
from truco.agents import BotConfig, ConfigurableBot
from truco.observation import observe
from truco.rl import vocab
from truco.rl.data import load_corpus, save_corpus
from truco.rl.tokenizer import MatchTokenizer
from truco.state import partner_of
from truco.tournament import (
    KNOB_SPEC,
    generate_topk_corpus,
    run_tournament,
    sample_configs,
)

SEATS = range(4)

# A deliberately chatty, rule-bending config: signals its whole holding every hand (signal_prob=1)
# and otherwise stresses the break paths. signal_partial/lie are off so it sends *several* codes.
_CHATTY = BotConfig(signal_prob=1.0, bluff=0.3, aggression=0.5, p_contradict=0.5,
                    p_overtrump=0.3, p_burn=0.3, p_overkill=0.3, p_bad_facedown=0.3)


def _play(cfgs, seed):
    """Drive one match (cfgs[team] per seat), tokenizing; return (tokenizer, events)."""
    rng = random.Random(seed)
    state = engine.new_match(rng, dealer=seed % 4)
    agents = {s: ConfigurableBot(cfgs[s % 2], rng=random.Random(rng.random())) for s in SEATS}
    tok, events = MatchTokenizer(), []
    tok.start(state)
    while not state.done:
        actor = state.current_actor
        tok.begin_turn(actor)
        action = agents[actor].act(observe(state, actor))
        assert action in engine.legal_actions(state), f"bot returned illegal {action!r}"
        state, evs = engine.apply_action(state, action, rng)
        events.extend(evs)
        tok.ingest(state, evs)
    return tok, events


def test_configurable_bot_only_ever_plays_legal_moves() -> None:
    """Across varied configs the bot never returns an illegal action (asserted inside `_play`)."""
    cfgs = sample_configs(12, seed=3)
    for seed in range(40):
        _play([cfgs[seed % len(cfgs)], cfgs[(seed + 5) % len(cfgs)]], seed)


def test_multi_signal_streams_never_leak() -> None:
    """A chatty multi-signal bot still yields leak-free streams (the widened-channel guarantee)."""
    saw_multi = False
    for seed in range(120):
        tok, events = _play([_CHATTY, _CHATTY], seed)

        # Per-hand signal counts: confirm the bot really sends several in some hand (non-vacuous).
        per_seat = {}
        for e in events:
            if isinstance(e, engine.SignalSent):
                per_seat[e.seat] = per_seat.get(e.seat, 0) + 1
        if any(c >= 2 for c in per_seat.values()):
            saw_multi = True

        for s in SEATS:
            stream = tok.streams[s]
            assert stream[0] == vocab.BOS and stream[-1] == vocab.EOS

            # A signal reaches only its sender and that sender's partner — never an opponent.
            signals_in_stream = sum(1 for t in stream if vocab.is_signal_move(t))
            signals_expected = sum(
                1 for e in events
                if isinstance(e, engine.SignalSent) and (e.seat == s or partner_of(e.seat) == s)
            )
            assert signals_in_stream == signals_expected, (
                f"seat {s}: {signals_in_stream} signals, expected {signals_expected}"
            )
    assert saw_multi, "no hand had a seat signal twice — multi-signal path went untested"


def test_textbook_beats_rules_broken() -> None:
    """Obeying every rule should clearly out-perform breaking every rule (knobs degrade play)."""
    from truco.tournament.match import play_config_match

    textbook = BotConfig()
    broken = BotConfig(**{name: 1.0 for name in KNOB_SPEC if name.startswith("p_")})
    wins = sum(play_config_match(textbook, broken, seed, seed % 4) == 0 for seed in range(200))
    assert wins / 200 > 0.70, f"textbook only won {wins}/200 vs all-rules-broken"


def test_sample_configs_textbook_first_and_balanced() -> None:
    configs = sample_configs(80, seed=0)
    assert len(configs) == 80
    assert configs[0] == BotConfig(), "index 0 must be the textbook reference config"
    # Latin-hypercube marginals: every level of every knob should be represented.
    for name, levels in KNOB_SPEC.items():
        used = {getattr(c, name) for c in configs[1:]}
        assert set(levels) <= used, f"knob {name}: levels {set(levels) - used} never sampled"


def test_tournament_is_deterministic() -> None:
    a = run_tournament(24, 16, seed=7, jobs=1, progress=False)
    b = run_tournament(24, 16, seed=7, jobs=1, progress=False)
    assert a.ranking == b.ranking and a.win_rate == b.win_rate


def test_topk_corpus_roundtrips(tmp_path) -> None:
    result = run_tournament(20, 12, seed=2, jobs=1, progress=False)
    streams = generate_topk_corpus(result.top(5), n_matches=30, seed=2)
    assert len(streams) == 30 * 4
    path = str(tmp_path / "corpus.npz")
    save_corpus(streams, path)
    loaded = load_corpus(path)
    assert len(loaded) == len(streams)
    assert all(st[0] == vocab.BOS and st[-1] == vocab.EOS for st in loaded)
    assert all(0 <= int(st.max()) < vocab.VOCAB_SIZE for st in loaded)

//! A from-scratch Rust port of the Truco engine + `ConfigurableBot`, fused into a single
//! allocation-light match loop. The canonical Python in `truco/` stays the reference; this
//! mirrors its *rules and decision probabilities* (not its exact RNG stream — Python's
//! `_hand_rng` is already per-process-salted, so the system was never bit-reproducible across
//! runs). The contract we hold is statistical: identical rules + identical break-probabilities
//! ⇒ identical expected win-rates, hence the same ranking up to sampling noise.
//!
//! Cards are `u8` ids 0..40 (`rank_index*4 + suit`); a hand's manilha is just a rank index.

// --------------------------------------------------------------------------- constants

const RANK_THREE: u8 = 9; // index of "3" in RANK_ORDER (worst→best: 4 5 6 7 Q J K A 2 3)
const RANK_TWO: u8 = 8; //   index of "2"

const VALUES: [i32; 5] = [1, 3, 6, 9, 12];
const MAX_LEVEL: i32 = 4; // len(VALUES) - 1
const SET_TARGET: i32 = 12;
const ELEVEN: i32 = 11;

// SignalCode
const SIG_WEAK: i32 = 0;
const SIG_STRONG: i32 = 1;

#[inline]
fn team_of(seat: usize) -> usize {
    seat & 1
}
#[inline]
fn next_seat(seat: usize) -> usize {
    (seat + 1) & 3
}
#[inline]
fn partner_of(seat: usize) -> usize {
    (seat + 2) & 3
}

#[inline]
fn rank_of(id: u8) -> u8 {
    id / 4
}
#[inline]
fn suit_of(id: u8) -> u8 {
    id % 4
}

/// `_SUIT_SIGNAL`: clubs(3)→ZAP(2), hearts(2)→3, spades(1)→4, diamonds(0)→5.
#[inline]
fn suit_signal(suit: u8) -> i32 {
    match suit {
        3 => 2,
        2 => 3,
        1 => 4,
        _ => 5,
    }
}

/// Manilha-aware total order. Normal cards: 0..9 by rank. Manilhas: 100+suit.
#[inline]
fn card_strength(id: u8, manilha: u8) -> i32 {
    let r = rank_of(id);
    if r == manilha {
        100 + suit_of(id) as i32
    } else {
        r as i32
    }
}

/// A card's standalone strength on a 0..1 scale (mirrors `heuristic._card_score`).
#[inline]
fn card_score(id: u8, manilha: u8) -> f64 {
    let s = card_strength(id, manilha);
    if s >= 100 {
        0.86 + 0.03 * (s - 100) as f64
    } else {
        (s as f64 / 9.0) * 0.78
    }
}

fn hand_score(cards: &[u8], manilha: u8) -> f64 {
    if cards.is_empty() {
        return 0.0;
    }
    let sum: f64 = cards.iter().map(|&c| card_score(c, manilha)).sum();
    sum / cards.len() as f64
}

// --------------------------------------------------------------------------- RNG

/// splitmix64 — fast, deterministic, ample quality for game-decision draws and shuffles.
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    #[inline]
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform integer in [0, n). (modulo bias is negligible for n ≤ 40)
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// --------------------------------------------------------------------------- config

/// The 14 knobs, in the fixed column order the Python side serialises.
#[derive(Clone)]
pub struct BotConfig {
    pub bluff: f64,
    pub signal_prob: f64,
    pub aggression: f64,
    pub p_lie: f64,
    pub p_signal_late: f64,
    pub p_signal_partial: f64,
    pub p_contradict: f64,
    pub p_ignore_partner: f64,
    pub p_overtrump: f64,
    pub p_burn: f64,
    pub p_fold_won: f64,
    pub p_overkill: f64,
    pub p_bad_facedown: f64,
    pub p_bad_eleven: f64,
}

impl BotConfig {
    pub fn from_slice(v: &[f64]) -> BotConfig {
        BotConfig {
            bluff: v[0],
            signal_prob: v[1],
            aggression: v[2],
            p_lie: v[3],
            p_signal_late: v[4],
            p_signal_partial: v[5],
            p_contradict: v[6],
            p_ignore_partner: v[7],
            p_overtrump: v[8],
            p_burn: v[9],
            p_fold_won: v[10],
            p_overkill: v[11],
            p_bad_facedown: v[12],
            p_bad_eleven: v[13],
        }
    }
}

// --------------------------------------------------------------------------- state

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    ElevenChoice,
    Play,
    Respond,
    HandDone,
    MatchDone,
}

#[derive(Clone, Copy)]
struct PlayedCard {
    seat: usize,
    card: u8,
    face_down: bool,
}

#[derive(Clone, Copy)]
struct PendingRaise {
    proposed_level: i32,
    responder: usize,
    origin: usize,
    caller_team: usize,
}

struct HandState {
    dealer: usize,
    manilha: u8,
    hands: [Vec<u8>; 4],
    lead: usize,
    turn: usize,
    current_round: i32,
    plays: Vec<PlayedCard>,
    round_results: Vec<i8>, // -1 tie / 0 / 1
    stake_level: i32,
    last_raise_team: Option<usize>,
    pending: Option<PendingRaise>,
    truco_locked: bool,
    base_value: i32,
    signals: [Vec<i32>; 4],
}

impl HandState {
    #[inline]
    fn value(&self) -> i32 {
        if self.truco_locked {
            self.base_value
        } else {
            VALUES[self.stake_level as usize]
        }
    }
}

struct MatchState {
    dealer: usize,
    set_scores: [i32; 2],
    match_sets: [i32; 2],
    sets_to_win: i32,
    hand: Option<HandState>,
    phase: Phase,
    eleven_decider: Option<usize>,
    winner: Option<usize>,
    hand_number: i32,
}

impl MatchState {
    #[inline]
    fn current_actor(&self) -> Option<usize> {
        match self.phase {
            Phase::Play => self.hand.as_ref().map(|h| h.turn),
            Phase::Respond => self.hand.as_ref().and_then(|h| h.pending.map(|p| p.responder)),
            Phase::ElevenChoice => self.eleven_decider,
            _ => None,
        }
    }
}

// --------------------------------------------------------------------------- actions

enum Action {
    Signal(i32),
    CallTruco,
    PlayCard { card: u8, face_down: bool },
    AcceptTruco,
    RunTruco,
    ReraiseTruco,
    ElevenChoice(bool),
}

// --------------------------------------------------------------------------- dealing

fn deal_hand(state: &mut MatchState, rng: &mut Rng) {
    state.hand_number += 1;
    let dealer = state.dealer;

    // Fisher–Yates over ids 0..40 (Python builds the deck rank-major == ids in order, then shuffles).
    let mut deck: [u8; 40] = std::array::from_fn(|i| i as u8);
    for i in (1..40).rev() {
        let j = rng.below(i + 1);
        deck.swap(i, j);
    }
    let hands: [Vec<u8>; 4] = [
        deck[0..3].to_vec(),
        deck[3..6].to_vec(),
        deck[6..9].to_vec(),
        deck[9..12].to_vec(),
    ];
    let vira = deck[12];
    let lead = next_seat(dealer);
    let manilha = (rank_of(vira) + 1) % 10;

    let mut hand = HandState {
        dealer,
        manilha,
        hands,
        lead,
        turn: lead,
        current_round: 0,
        plays: Vec::with_capacity(4),
        round_results: Vec::with_capacity(3),
        stake_level: 0,
        last_raise_team: None,
        pending: None,
        truco_locked: false,
        base_value: 1,
        signals: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
    };

    state.eleven_decider = None;
    let t0_11 = state.set_scores[0] == ELEVEN;
    let t1_11 = state.set_scores[1] == ELEVEN;

    if t0_11 && t1_11 {
        hand.truco_locked = true;
        hand.base_value = 1;
        state.hand = Some(hand);
        state.phase = Phase::Play;
    } else if t0_11 || t1_11 {
        let team = if t0_11 { 0 } else { 1 };
        hand.truco_locked = true;
        hand.base_value = 3;
        let decider = if team_of(lead) == team { lead } else { next_seat(lead) };
        state.hand = Some(hand);
        state.eleven_decider = Some(decider);
        state.phase = Phase::ElevenChoice;
    } else {
        state.hand = Some(hand);
        state.phase = Phase::Play;
    }
}

fn new_match(seed: u64, dealer: usize, sets_to_win: i32, rng: &mut Rng) -> MatchState {
    let mut state = MatchState {
        dealer,
        set_scores: [0, 0],
        match_sets: [0, 0],
        sets_to_win,
        hand: None,
        phase: Phase::Play,
        eleven_decider: None,
        winner: None,
        hand_number: 0,
    };
    let _ = seed;
    deal_hand(&mut state, rng);
    state
}

// --------------------------------------------------------------------------- round/hand math

/// Resolve the table → (team, winning_seat) or (-1, None) tie. `winning_seat` is the earliest
/// player at the top strength (first in play order).
fn resolve_round(hand: &HandState) -> (i8, Option<usize>) {
    let manilha = hand.manilha;
    let mut top = i32::MIN;
    for pc in &hand.plays {
        let st = if pc.face_down { -1 } else { card_strength(pc.card, manilha) };
        if st > top {
            top = st;
        }
    }
    let mut win_seat = None;
    let mut teams_at_top = 0u8; // bitmask of teams holding a card at `top`
    for pc in &hand.plays {
        let st = if pc.face_down { -1 } else { card_strength(pc.card, manilha) };
        if st == top {
            teams_at_top |= 1 << team_of(pc.seat);
            if win_seat.is_none() {
                win_seat = Some(pc.seat);
            }
        }
    }
    if teams_at_top == 0b11 {
        (-1, None)
    } else {
        let ws = win_seat.unwrap();
        (team_of(ws) as i8, Some(ws))
    }
}

enum Decision {
    PlayOn,
    Redeal,
    Winner(usize),
}

fn decide_hand(results: &[i8]) -> Decision {
    let n = results.len();
    if n < 2 {
        return Decision::PlayOn;
    }
    let r1 = results[0];
    let r2 = results[1];
    if n == 2 {
        if r1 == -1 && r2 == -1 {
            return Decision::PlayOn;
        }
        if r1 == -1 {
            return Decision::Winner(r2 as usize);
        }
        if r2 == -1 {
            return Decision::Winner(r1 as usize);
        }
        return if r1 == r2 { Decision::Winner(r1 as usize) } else { Decision::PlayOn };
    }
    // n == 3
    let r3 = results[2];
    if r3 != -1 {
        return Decision::Winner(r3 as usize);
    }
    if r1 != -1 {
        return Decision::Winner(r1 as usize);
    }
    Decision::Redeal
}

// --------------------------------------------------------------------------- transitions

#[inline]
fn can_call_truco(hand: &HandState, actor: usize) -> bool {
    if hand.truco_locked || hand.stake_level >= MAX_LEVEL {
        return false;
    }
    match hand.last_raise_team {
        None => true,
        Some(t) => t != team_of(actor),
    }
}

fn score_hand(state: &mut MatchState, winner_team: usize, value: i32) {
    state.set_scores[winner_team] += value;
    if state.set_scores[winner_team] >= SET_TARGET {
        state.match_sets[winner_team] += 1;
        state.set_scores = [0, 0];
        if state.match_sets[winner_team] >= state.sets_to_win {
            state.winner = Some(winner_team);
            state.phase = Phase::MatchDone;
            return;
        }
    }
    state.phase = Phase::HandDone;
}

fn after_round(state: &mut MatchState, rng: &mut Rng) {
    let (result, win_seat) = {
        let hand = state.hand.as_ref().unwrap();
        resolve_round(hand)
    };
    {
        let hand = state.hand.as_mut().unwrap();
        hand.round_results.push(result);
    }
    let decision = decide_hand(&state.hand.as_ref().unwrap().round_results);
    match decision {
        Decision::PlayOn => {
            let hand = state.hand.as_mut().unwrap();
            hand.current_round += 1;
            hand.plays.clear();
            if result != -1 {
                hand.lead = win_seat.unwrap();
            }
            hand.turn = hand.lead;
        }
        Decision::Redeal => {
            deal_hand(state, rng);
        }
        Decision::Winner(team) => {
            let value = state.hand.as_ref().unwrap().value();
            score_hand(state, team, value);
        }
    }
}

fn apply_play(state: &mut MatchState, action: Action, rng: &mut Rng) {
    let actor = state.hand.as_ref().unwrap().turn;
    match action {
        Action::Signal(code) => {
            state.hand.as_mut().unwrap().signals[actor].push(code);
            // free action: same player acts again
        }
        Action::CallTruco => {
            let hand = state.hand.as_mut().unwrap();
            let proposed = hand.stake_level + 1;
            hand.pending = Some(PendingRaise {
                proposed_level: proposed,
                responder: next_seat(actor),
                origin: actor,
                caller_team: team_of(actor),
            });
            state.phase = Phase::Respond;
        }
        Action::PlayCard { card, face_down } => {
            let four = {
                let hand = state.hand.as_mut().unwrap();
                if let Some(pos) = hand.hands[actor].iter().position(|&c| c == card) {
                    hand.hands[actor].remove(pos);
                }
                hand.plays.push(PlayedCard { seat: actor, card, face_down });
                hand.plays.len() == 4
            };
            if four {
                after_round(state, rng);
            } else {
                state.hand.as_mut().unwrap().turn = next_seat(actor);
            }
        }
        _ => unreachable!("non-play action in PLAY phase"),
    }
}

fn apply_respond(state: &mut MatchState, action: Action) {
    let pend = state.hand.as_ref().unwrap().pending.unwrap();
    match action {
        Action::AcceptTruco => {
            let hand = state.hand.as_mut().unwrap();
            hand.stake_level = pend.proposed_level;
            hand.last_raise_team = Some(pend.caller_team);
            hand.turn = pend.origin;
            hand.pending = None;
            state.phase = Phase::Play;
        }
        Action::RunTruco => {
            let value = VALUES[(pend.proposed_level - 1) as usize];
            state.hand.as_mut().unwrap().pending = None;
            score_hand(state, pend.caller_team, value);
        }
        Action::ReraiseTruco => {
            let hand = state.hand.as_mut().unwrap();
            let p = hand.pending.as_mut().unwrap();
            p.caller_team = team_of(p.responder);
            p.responder = next_seat(p.responder);
            p.proposed_level += 1;
            // phase stays Respond
        }
        _ => unreachable!("non-respond action in RESPOND phase"),
    }
}

fn apply_eleven(state: &mut MatchState, action: Action) {
    let decider = state.eleven_decider.unwrap();
    let team = team_of(decider);
    state.eleven_decider = None;
    match action {
        Action::ElevenChoice(true) => {
            state.phase = Phase::Play;
        }
        Action::ElevenChoice(false) => {
            score_hand(state, 1 - team, 1);
        }
        _ => unreachable!("non-eleven action in ELEVEN_CHOICE phase"),
    }
}

fn apply_action(state: &mut MatchState, action: Action, rng: &mut Rng) {
    match state.phase {
        Phase::ElevenChoice => apply_eleven(state, action),
        Phase::Respond => apply_respond(state, action),
        _ => apply_play(state, action, rng),
    }
    if state.phase == Phase::HandDone {
        state.dealer = next_seat(state.dealer);
        deal_hand(state, rng);
    }
}

// --------------------------------------------------------------------------- the bot

struct Bot {
    cfg: BotConfig,
    rng: Rng,
    salt: u64,
    risk_shift: f64,
}

impl Bot {
    fn new(cfg: BotConfig, seed: u64) -> Bot {
        let mut rng = Rng::new(seed);
        let salt = rng.next_u64();
        let risk_shift = 0.15 * cfg.aggression;
        Bot { cfg, rng, salt, risk_shift }
    }

    fn decide(&mut self, state: &MatchState, actor: usize) -> Action {
        match state.phase {
            Phase::ElevenChoice => self.eleven(state, actor),
            Phase::Respond => self.respond(state, actor),
            _ => self.play(state, actor),
        }
    }

    // --- shared reads ---------------------------------------------------------
    fn hand_strength(&self, hand: &HandState, actor: usize) -> f64 {
        hand_score(&hand.hands[actor], hand.manilha)
    }

    /// (partner_winning, opp_best, cards_on_table)
    fn table_status(&self, hand: &HandState, actor: usize) -> (bool, i32, usize) {
        let me_team = team_of(actor);
        let partner = partner_of(actor);
        let (mut best_seat, mut best_str, mut opp_best): (Option<usize>, i32, i32) =
            (None, -2, -2);
        for pc in &hand.plays {
            let st = if pc.face_down { -1 } else { card_strength(pc.card, hand.manilha) };
            if team_of(pc.seat) != me_team {
                opp_best = opp_best.max(st);
            }
            if st > best_str {
                best_str = st;
                best_seat = Some(pc.seat);
            }
        }
        let partner_winning = best_seat == Some(partner) && best_str > opp_best;
        (partner_winning, opp_best, hand.plays.len())
    }

    fn partner_boost(&self, hand: &HandState, actor: usize) -> f64 {
        let sigs = &hand.signals[partner_of(actor)];
        let mut boost = 0.0;
        if sigs.iter().any(|&c| c != SIG_WEAK) {
            boost += 0.15;
        }
        if sigs.iter().any(|&c| c == SIG_WEAK) {
            boost -= 0.15;
        }
        boost
    }

    // --- signalling (R1–R4) ---------------------------------------------------
    /// Per-hand RNG, stable across the several `decide` calls one signalling turn makes.
    /// Keyed on (current hand cards, manilha, dealer, scores) ^ salt — exactly the inputs
    /// Python's `_hand_rng` keys on (signals are deliberately excluded so the plan is stable
    /// as codes are sent one per call).
    fn hand_rng(&self, state: &MatchState, hand: &HandState, actor: usize) -> Rng {
        let mut h: u64 = self.salt;
        let mix = |x: u64, h: &mut u64| {
            *h = (*h ^ x).wrapping_mul(0x100000001B3);
        };
        for &c in &hand.hands[actor] {
            mix(c as u64 + 1, &mut h);
        }
        mix(0xF00D, &mut h); // separator
        mix(hand.manilha as u64, &mut h);
        mix(hand.dealer as u64, &mut h);
        mix(state.set_scores[0] as u64, &mut h);
        mix(state.set_scores[1] as u64, &mut h);
        mix(state.match_sets[0] as u64, &mut h);
        mix(state.match_sets[1] as u64, &mut h);
        Rng::new(h)
    }

    /// Full informative holding, strongest first (manilhas by suit desc, then 3s, 2s, then a
    /// STRONG/WEAK summary).
    fn honest_codes(&self, hand: &HandState, actor: usize) -> Vec<i32> {
        let mut codes: Vec<i32> = Vec::new();
        let mut manilhas: Vec<u8> = hand.hands[actor]
            .iter()
            .copied()
            .filter(|&c| rank_of(c) == hand.manilha)
            .collect();
        manilhas.sort_by(|a, b| suit_of(*b).cmp(&suit_of(*a))); // suit desc
        for c in &manilhas {
            codes.push(suit_signal(suit_of(*c)));
        }
        if hand.hands[actor].iter().any(|&c| rank_of(c) == RANK_THREE) {
            codes.push(6); // HAS_THREE
        }
        if hand.hands[actor].iter().any(|&c| rank_of(c) == RANK_TWO) {
            codes.push(7); // HAS_TWO
        }
        let strong = self.hand_strength(hand, actor) > 0.55;
        codes.push(if strong { SIG_STRONG } else { SIG_WEAK });
        codes
    }

    /// Returns (codes_to_send_in_order, late). Deterministic per hand.
    fn signal_decision(&self, state: &MatchState, hand: &HandState, actor: usize) -> (Vec<i32>, bool) {
        let mut hr = self.hand_rng(state, hand, actor);
        let r_chat = hr.next_f64();
        let r_partial = hr.next_f64();
        let r_lie = hr.next_f64();
        let r_contra = hr.next_f64();
        let r_late = hr.next_f64();
        if r_chat >= self.cfg.signal_prob {
            return (Vec::new(), false);
        }
        let truth_strong = self.hand_strength(hand, actor) > 0.55;
        let mut codes = self.honest_codes(hand, actor);
        if r_partial < self.cfg.p_signal_partial {
            codes.truncate(1);
        }
        if r_lie < self.cfg.p_lie {
            codes = vec![if truth_strong { SIG_WEAK } else { SIG_STRONG }];
        }
        if r_contra < self.cfg.p_contradict {
            let mut c: Vec<i32> = codes
                .into_iter()
                .filter(|&x| x != SIG_STRONG && x != SIG_WEAK)
                .collect();
            c.push(SIG_STRONG);
            c.push(SIG_WEAK);
            codes = c;
        }
        (codes, r_late < self.cfg.p_signal_late)
    }

    fn maybe_signal(&mut self, state: &MatchState, hand: &HandState, actor: usize) -> Option<Action> {
        // legal signals = codes 0..8 not yet sent this hand
        let sent = &hand.signals[actor];
        if sent.len() >= 8 {
            return None; // every code already flashed
        }
        let (codes, late) = self.signal_decision(state, hand, actor);
        let next = codes
            .iter()
            .copied()
            .find(|c| !sent.contains(c) && (0..8).contains(c));
        let code = next?;
        if late {
            if hand.current_round == 0 {
                return None; // R2: stay silent up front
            }
        } else if hand.current_round != 0 && sent.is_empty() {
            return None; // up-front: never start signalling after round 0
        }
        Some(Action::Signal(code))
    }

    // --- PLAY -----------------------------------------------------------------
    fn play(&mut self, state: &MatchState, actor: usize) -> Action {
        let hand = state.hand.as_ref().unwrap();
        if let Some(sig) = self.maybe_signal(state, hand, actor) {
            return sig;
        }
        if can_call_truco(hand, actor) && self.should_call(hand, actor) {
            return Action::CallTruco;
        }
        self.choose_card(hand, actor)
    }

    fn choose_card(&mut self, hand: &HandState, actor: usize) -> Action {
        let manilha = hand.manilha;
        // face-up candidates = the cards in hand, sorted by strength ascending
        let mut faceup: Vec<u8> = hand.hands[actor].clone();
        faceup.sort_by_key(|&c| card_strength(c, manilha));
        let facedown_allowed = hand.current_round > 0;
        let (partner_winning, opp_best, on_table) = self.table_status(hand, actor);

        if on_table == 0 {
            // lead a middling card
            let idx = faceup.len() / 2;
            return Action::PlayCard { card: faceup[idx], face_down: false };
        }

        if partner_winning {
            // R6: don't overtrump a winning partner
            if self.rng.next_f64() < self.cfg.p_overtrump {
                if let Some(&cheapest) = faceup
                    .iter()
                    .filter(|&&c| card_strength(c, manilha) > opp_best)
                    .min_by_key(|&&c| card_strength(c, manilha))
                {
                    return self.maybe_facedown(cheapest, facedown_allowed, false);
                }
            }
            return self.dump(&faceup, facedown_allowed);
        }

        // can we win this round?
        let winners: Vec<u8> = faceup
            .iter()
            .copied()
            .filter(|&c| card_strength(c, manilha) > opp_best)
            .collect();
        if !winners.is_empty() {
            // R9: win cheaply, unless overkill
            let chosen = if self.rng.next_f64() < self.cfg.p_overkill {
                *winners.iter().max_by_key(|&&c| card_strength(c, manilha)).unwrap()
            } else {
                *winners.iter().min_by_key(|&&c| card_strength(c, manilha)).unwrap()
            };
            return self.maybe_facedown(chosen, facedown_allowed, false);
        }

        // can't win
        if self.rng.next_f64() < self.cfg.p_burn {
            // R7: burn a usable (strongest) card instead of conceding low
            let strongest = *faceup.last().unwrap();
            return self.maybe_facedown(strongest, facedown_allowed, true);
        }
        self.dump(&faceup, facedown_allowed)
    }

    fn dump(&mut self, faceup: &[u8], facedown_allowed: bool) -> Action {
        self.maybe_facedown(faceup[0], facedown_allowed, true)
    }

    fn maybe_facedown(&mut self, card: u8, facedown_allowed: bool, safe: bool) -> Action {
        let mut want_facedown = safe;
        if self.rng.next_f64() < self.cfg.p_bad_facedown {
            want_facedown = !want_facedown;
        }
        let face_down = want_facedown && facedown_allowed;
        Action::PlayCard { card, face_down }
    }

    // --- betting (R5, R8) -----------------------------------------------------
    fn should_call(&mut self, hand: &HandState, actor: usize) -> bool {
        let mut s = self.hand_strength(hand, actor);
        if self.rng.next_f64() >= self.cfg.p_ignore_partner {
            s += self.partner_boost(hand, actor);
        }
        let team = team_of(actor) as i8;
        let won_round = hand.round_results.iter().any(|&r| r == team);
        let d = self.risk_shift;
        let mut prob = if s > 0.72 - d {
            0.6
        } else if s > 0.55 - d {
            0.25
        } else {
            0.0
        };
        if won_round {
            prob += 0.2;
        }
        if s < 0.40 {
            prob += self.cfg.bluff;
        }
        self.rng.next_f64() < prob
    }

    fn respond(&mut self, state: &MatchState, actor: usize) -> Action {
        let hand = state.hand.as_ref().unwrap();
        let mut s = self.hand_strength(hand, actor);
        let team = team_of(actor) as i8;
        let won_round = hand.round_results.iter().any(|&r| r == team);
        if won_round {
            s += 0.10;
        }
        if self.rng.next_f64() >= self.cfg.p_ignore_partner {
            s += self.partner_boost(hand, actor);
        }
        let d = self.risk_shift;
        let pend = hand.pending.unwrap();
        let reraise_legal = pend.proposed_level < MAX_LEVEL;
        if reraise_legal && (s > 0.80 - d || (s > 0.55 - d && self.rng.next_f64() < self.cfg.bluff)) {
            return Action::ReraiseTruco;
        }
        if s > 0.42 - d || self.rng.next_f64() < self.cfg.bluff {
            return Action::AcceptTruco;
        }
        // R8: don't fold a position you've effectively won
        if won_round && s > 0.5 && self.rng.next_f64() >= self.cfg.p_fold_won {
            return Action::AcceptTruco;
        }
        Action::RunTruco
    }

    // --- ELEVEN_CHOICE (R11) --------------------------------------------------
    fn eleven(&mut self, state: &MatchState, actor: usize) -> Action {
        let hand = state.hand.as_ref().unwrap();
        let mut combined: Vec<u8> = hand.hands[actor].clone();
        combined.extend_from_slice(&hand.hands[partner_of(actor)]);
        let mut good = hand_score(&combined, hand.manilha) > 0.45 - self.risk_shift;
        if self.rng.next_f64() < self.cfg.p_bad_eleven {
            good = !good;
        }
        Action::ElevenChoice(good)
    }
}

// --------------------------------------------------------------------------- match driver

/// Play a full config-vs-config match; return the winning team (0 or 1).
/// Team 0 = seats {0,2} = `cfg0`; team 1 = seats {1,3} = `cfg1`.
pub fn play_match(
    cfg0: &BotConfig,
    cfg1: &BotConfig,
    seed: u64,
    dealer: usize,
    sets_to_win: i32,
) -> usize {
    let mut match_rng = Rng::new(seed);
    let mut state = new_match(seed, dealer, sets_to_win, &mut match_rng);

    // Per-seat bots, seeded from the match stream (mirrors `random.Random(rng.random())`).
    let mut bots: Vec<Bot> = (0..4)
        .map(|s| {
            let cfg = if team_of(s) == 0 { cfg0.clone() } else { cfg1.clone() };
            Bot::new(cfg, match_rng.next_u64())
        })
        .collect();

    // Safety bound: a real match resolves in well under this many actions.
    for _ in 0..100_000 {
        if state.phase == Phase::MatchDone {
            break;
        }
        let actor = match state.current_actor() {
            Some(a) => a,
            None => break,
        };
        let action = bots[actor].decide(&state, actor);
        apply_action(&mut state, action, &mut match_rng);
    }
    state.winner.unwrap_or(0)
}

/// Build the schedule (mirrors `run.build_schedule`'s structure) and play every match in
/// parallel, tallying (wins, games) per config index. Pure Rust + rayon: deterministic in the
/// result (integer adds commute), independent of thread scheduling.
pub fn run_tournament(
    configs: &[BotConfig],
    games_per_config: usize,
    seed: u64,
    sets_to_win: i32,
) -> (Vec<u64>, Vec<u64>) {
    use rayon::prelude::*;

    let n = configs.len();
    if n < 2 {
        return (vec![0; n], vec![0; n]);
    }

    // Schedule: each config gets `games_per_config` matches vs a random other, seats alternated.
    let mut sched: Vec<(u32, u32, u64, usize)> = Vec::with_capacity(n * games_per_config);
    let mut srng = Rng::new(seed ^ 0x5CED);
    let mut ms = seed;
    for i in 0..n {
        for g in 0..games_per_config {
            let mut j = srng.below(n - 1);
            if j >= i {
                j += 1;
            }
            let (a, b) = if g % 2 == 0 { (i, j) } else { (j, i) };
            sched.push((a as u32, b as u32, ms, (ms % 4) as usize));
            ms += 1;
        }
    }

    let (wins, games) = sched
        .par_iter()
        .fold(
            || (vec![0u64; n], vec![0u64; n]),
            |mut acc, &(a, b, mseed, dealer)| {
                let (a, b) = (a as usize, b as usize);
                let winner_team = play_match(&configs[a], &configs[b], mseed, dealer, sets_to_win);
                let wc = if winner_team == 0 { a } else { b };
                acc.0[wc] += 1;
                acc.1[a] += 1;
                acc.1[b] += 1;
                acc
            },
        )
        .reduce(
            || (vec![0u64; n], vec![0u64; n]),
            |mut x, y| {
                for i in 0..n {
                    x.0[i] += y.0[i];
                    x.1[i] += y.1[i];
                }
                x
            },
        );
    (wins, games)
}

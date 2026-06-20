//! Python bindings for the Rust Truco tournament engine.
//!
//! `truco/tournament/run.py` uses these when the extension is importable; otherwise it falls
//! back to the pure-Python path. Configs cross the boundary as a fixed-order list of 14 floats
//! (see `BotConfig::from_slice` / `run.py:_config_to_row`).

use pyo3::prelude::*;

mod engine;

const N_KNOBS: usize = 14;

fn check_row(row: &[f64]) -> PyResult<()> {
    if row.len() != N_KNOBS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "each config row must have {N_KNOBS} knobs, got {}",
            row.len()
        )));
    }
    Ok(())
}

#[pymodule]
mod truco_fast {
    use super::{check_row, engine};
    use pyo3::prelude::*;

    /// Run the full config tournament in Rust (rayon-parallel) and return `(wins, games)` per
    /// config index. `configs` is a list of 14-float rows. Deterministic given the same args.
    #[pyfunction]
    #[pyo3(signature = (configs, games_per_config, seed, sets_to_win))]
    fn run_tournament(
        py: Python<'_>,
        configs: Vec<Vec<f64>>,
        games_per_config: usize,
        seed: u64,
        sets_to_win: i32,
    ) -> PyResult<(Vec<u64>, Vec<u64>)> {
        for row in &configs {
            check_row(row)?;
        }
        let cfgs: Vec<engine::BotConfig> =
            configs.iter().map(|v| engine::BotConfig::from_slice(v)).collect();
        // Release the GIL: the simulation is pure Rust and touches no Python objects.
        let res = py
            .detach(move || engine::run_tournament(&cfgs, games_per_config, seed, sets_to_win));
        Ok(res)
    }

    /// Play one config-vs-config match; return the winning team (0 = seats {0,2}, 1 = {1,3}).
    /// Exposed mainly for equivalence testing against the Python `play_config_match`.
    #[pyfunction]
    #[pyo3(signature = (cfg0, cfg1, seed, dealer=0, sets_to_win=1))]
    fn play_match(
        cfg0: Vec<f64>,
        cfg1: Vec<f64>,
        seed: u64,
        dealer: usize,
        sets_to_win: i32,
    ) -> PyResult<usize> {
        check_row(&cfg0)?;
        check_row(&cfg1)?;
        Ok(engine::play_match(
            &engine::BotConfig::from_slice(&cfg0),
            &engine::BotConfig::from_slice(&cfg1),
            seed,
            dealer,
            sets_to_win,
        ))
    }
}

# Empirical industrial organization

Three replication-grade implementations written for the PhD Empirical IO sequence at Yale
(ECON 6600, Fall 2025, Profs. Philip Haile and Charles Hodgson).

## `demand/` — BLP demand estimation and merger simulation

Random-coefficients logit demand with optimal instruments, diversion ratios, and merger
counterfactuals, estimated with [pyBLP](https://github.com/jeffgortmaker/pyblp).

- `blp.py` — standalone implementation
- `blp.ipynb` — notebook, `blp.csv` — data
- `BLP_hw_chadim.pdf` — write-up

## `dynamics/` — single-agent dynamics (Rust 1987)

Value-function iteration and structural estimation of the optimal machine-replacement model,
on Rust's original bus-engine data.

- `dp.ipynb` — notebook, `HW1_Rust_data.asc` — data

## `games/` — dynamic entry/exit games (Bajari–Benkard–Levin)

Forward-simulation estimation with conditional choice probabilities, recovering profit
parameters from a dynamic entry/exit game.

- `dg.py` — implementation
- `BBL_hw_chadim.pdf` — write-up

---

Marek Chadim · marek.chadim@yale.edu · [marek-chadim.github.io](https://marek-chadim.github.io)

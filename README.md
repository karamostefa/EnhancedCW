# EnhancedCW – simulators, scripts and data

Companion code of the manuscript *"An Adaptive Compute-and-Wait Proof-of-Work Framework for Energy-Efficient and
Scalable Blockchain Consensus"* (M. Kara, MDPI *Network*).

Two simulators are provided. **The event-driven model is the reference for every quantitative result of the paper.**
The multi-threaded prototype is kept for the protocol logic, the message counts and as a cross-check.

| File | Purpose |
|---|---|
| `ecw_des.py` | Event-driven model of the CW-PoW family (+ PoW, Green-PoW, Proof-of-Experience, PoS, PBFT). All durations are model parameters, nothing is measured on the host, every run is determined by its seed. |
| `ecw_des_experiments.py` | All experiments of Sections 5.5-5.12 (Table 2, comparison with PoW at equal mean block time, fairness, scalability, robustness, mechanism ablation, APT, pipelining, duty cycling). Writes `des_*.csv`. |
| `make_tables.py` | Writes the LaTeX tables of the paper (`tab_*.tex`) from the CSV files and prints the numbers that are quoted in the text. |
| `ecw_figures.py` | Regenerates the figures of the paper (`img/fig_*.pdf`) from the CSV files. |
| `EnhancedCW 5 algo.py` | Multi-threaded prototype (one Python thread per miner, wall-clock measurements). Results depend on the host. |
| `ecw_ablation.py` | Mechanism ablation with the *threaded* prototype (cross-check, 40 repetitions; `ablation_mechanisms_*.csv`). |
| `proto_benchmarks.py` | Host-dependence statements (interpreter-lock contention) and per-block values of the prototype (`prototype_crosscheck.csv`, `prototype_benchmarks.txt`). |

## Protocol rule

All miners start a block together. A round closes when `NbrS` solutions are visible; only the authors of these solutions start the
next round (all at the same instant), and the other miners stand down until the block is final. The winner is the participant of the
last round with the smallest final hash. The model implements this as `Cfg(selective=True)` (default; `selective=False` is the variant in
which every miner completes every round), the threaded prototype as `SELECTIVE = True`. `Cfg.winner_rule='first'` selects the
first finisher of the last round instead (tested and rejected in Section 5.7).

## Reproduce

```bash
pip install numpy matplotlib
python ecw_des_experiments.py 1000 all      # about 30 minutes (the fairness part dominates); writes des_*.csv (deterministic)
python make_tables.py                       # writes tab_*.tex and prints the quoted numbers
python ecw_figures.py                       # writes img/fig_*.pdf
python proto_benchmarks.py                  # host dependent, a few minutes
```

`python ecw_des_experiments.py 1000 validate` runs the validation checks of the model (message counts of the prototype,
hashing energy and quorum wait against an independent Monte Carlo of the round rule, PoW race). Single stages: `equal`, `fair`, `scal`, `ablation`, `apt`, `pipe`, `dc`, `byz`.

## What changed in the prototype (`EnhancedCW 5 algo.py`) in this revision

* `run_cw_pow`: the throughput was computed as `runs / (avg_total_time / runs)`, i.e. it was `runs` times too large. It is now `runs / avg_total_time`.
* Energy per block, time per block, mean rounds per block and duty cycle are returned in addition to the old keys.
  (The old "energy per round" is divided by the *final* `NbrR`; it is kept for compatibility, use the per-block value.)
* `adaptive_tuner`: new flag `APT_LATENCY_PRIORITY` (default `True`). With `False`, the independent rules of the first version are
  reproduced; these cancel every latency-triggered decrease of `NbrR` (see Section 5.10 of the paper).
* `byzantine_withhold_frac` now has an effect: with `reveal_timeout_sec` set, every miner waits for the timeout in every
  round when at least one miner withholds its reveal (reveal-or-forfeit rule). `network_delay_spike_*` delays the visibility of the solution.
* The printed label "Active-Time Fraction (E)" is now "Duty cycle (D = T_compute / (T_compute + T_wait))".

Known simplifications of the prototype accounting (documented in Appendix 1 of the paper): useful-work time is counted as
computing time and also inside the waiting window, the wake-up energy is charged once per block, the sleep power is not used, and
no energy is charged after a thread has delivered its last round or has stood down. This is one reason why the event-driven model is the reference.

The value "Avg fairness (mean F_i)" printed by the prototype is the mean of the ratios of identical miners and is not used for any claim; fairness is evaluated in the event-driven model with different hash powers (Section 5.7 of the paper).

## Data files

`des_*.csv` – outputs of `ecw_des_experiments.py` (means and 95 % confidence half-widths, see the header of each file).
`prototype_*.csv/txt` – outputs of `proto_benchmarks.py`. `*_results.csv`, `ablation_apt_nbrr_nbrs.csv`,
`byzantine_withholding_robustness.csv`, `network_delay_spike_robustness.csv` – per-run outputs of one run of the prototype
(`python "EnhancedCW 5 algo.py"`, see `prototype_output.txt`).

## Limitations

Energy values are proportional estimates in model units (P_active = 100, P_wait = 5, P_sleep = 0.5 units/s, E_wake = 1 unit),
not measurements. Miners are identical in all experiments except the fairness experiment (log-normal hash powers, sigma = 0.5). Forks are not simulated. Green-PoW and Proof-of-Experience are
simplified re-implementations of the mechanisms described in the abstracts of the original papers.

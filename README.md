# Simulation code: On Large-Scale Multiple Testing Over Networks — A Non-Asymptotic Approach

Python simulation code and results for the paper --- exactly the scripts
and result files presented there, nothing else ---

> M. Pournaderi, *On Large-Scale Multiple Testing Over Networks: A
> Non-Asymptotic Approach*, 2026.

The paper develops finite-sample false discovery rate (FDR) control for
distributed multiple testing under communication constraints: cross-fit
greedy aggregation (CFGA) and its inflated variant, a derandomized e-value
aggregation (e-CFGA), and BONuS-GA with a fully implementable per-node
budgeted tier, together with a tight Θ(m^{-1/4} √log m) characterization of
the winner's-curse bias that breaks the original asymptotic guarantee.

## Code map

| File | Role |
|---|---|
| `simulations.py` | method library: baselines (BH, local BH), original greedy, CFGA (+Storey / adaptive ε), Inflated CFGA, BONuS-GA, data generation |
| `ecfga.py` | e-CFGA (Storey and inflated tiers, derandomized over S splits) |
| `bonus_pernode.py` | per-node calibrated and per-node **budgeted** BONuS-GA |
| `sweep_quiet_nodes.py` | Experiment 1 (Local-BH failure as quiet nodes accumulate) |
| `mn_sweep.py` | Experiments 2–6 (signal strength, within-node correlation, hub size, Cauchy, target α) |
| `bonus_sweep.py` | BONuS-GA overlay at the same sweep points/seeds |
| `prov_sweep.py` | provable-tier overlays (inflated e-CFGA, budgeted BONuS-GA) |
| `ablation_fixedbag.py` | Table 1: paired fixed-bag ablation (identical realized bag per arm; paired ±2SE) |
| `rho_hubsonly.py` | hubs-only AR(1) companion run (quiet nulls i.i.d.) |
| `random_labels.py` | random-label companion run (multinomial node sizes, covered by the CFGA theorems) |
| `replot.py` | redraws all six figures from the stored `.npz` results, no recomputation |

## Reproducing the experiments

```bash
pip install -r requirements.txt

python sweep_quiet_nodes.py all          # Experiment 1
python mn_sweep.py mu                    # Experiments 2-6: mu, rho, alpha, n, cauchy
python mn_sweep.py plot mu               # assemble sim_mn_mu.{png,npz}
python bonus_sweep.py all                # BONuS-GA overlay (same seeds)
python prov_sweep.py quiet ecfga_inf     # provable tiers, per axis/method
python prov_sweep.py assemble
python ablation_fixedbag.py              # Table 1 (paired, with 2SE)
python rho_hubsonly.py                   # hubs-only dependence companion
python random_labels.py                  # random-label companion (CFGA certificates)
python replot.py                         # figures from the .npz files
```

Everything runs on a laptop CPU; the full suite completes in well under an
hour. Sweeps are chunked and cache per-point results (with config/source
fingerprints, so stale caches are recomputed), and interrupted runs resume.

## Results

`outputs/` ships with the assembled per-experiment results used for the
paper's figures and tables, so figures can be restyled and numbers
re-analyzed without rerunning anything: `sim_<experiment>.npz` with arrays
`x`, `fdr_<method>`, `pwr_<method>`; per-trial arrays for the paired
ablation (`sim_ablation_fixedbag.npz`) and the hubs-only companion run
(`sim_rho_hubsonly.npz`), and the random-label companion run
(`sim_random_labels.npz`). Rerunning a driver additionally creates per-point
resume caches (`_mn_*`, `_qn_*`, `_prov_*`), which also store 2·SE per
method.

```python
import numpy as np
d = np.load("outputs/sim_mn_n.npz")
print(d["x"], d["fdr_CFGA + Storey + adaptive eps"])
a = np.load("outputs/sim_ablation_fixedbag.npz")   # per-trial arrays
print(a["pwr_pernode_storey"].mean() - a["pwr_global_storey"].mean())
```

## Requirements

Python ≥ 3.9 with `numpy`, `scipy`, `matplotlib` (see `requirements.txt`).

## Acknowledgment

Claude Opus 4.7 and Claude Fable 5 (Anthropic) were used in the research and
preparation of this work; the author retains full responsibility for its
content. See the paper's acknowledgments.

## License

MIT (see `LICENSE`).

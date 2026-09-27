"""Random-label companion run for the CFGA certificates.

The CFGA / Inflated CFGA / e-CFGA theorems are stated under the random-label
model of Assumption 1 (i.i.d. node labels, hence multinomial node sizes),
whereas Experiments 1-6 fix each node's sample size. This script reruns the
provable tiers (and the CFGA+Storey+adaptive reference) with node sizes drawn
as Multinomial(m, q), q proportional to the fixed design sizes, so that the
experimental configuration is exactly covered by the theorems.
Configurations: the base network (m_hub = 1000) and the small-hub network
(m_hub = 250); 2 hubs + 300 quiet nodes (m_quiet = 20 in expectation).
NT = 100 trials each. Writes outputs/sim_random_labels.npz (per-trial arrays).
"""
import sys, time
import numpy as np
from multiprocessing import Pool
import simulations as s
import ecfga
import bonus_pernode as bp
import mn_sweep as M

NT = 100
S_SPLITS = 10
CONFIGS = {'base': 1000, 'smallhub': 250}
METHODS = ['cfga_storey_ada', 'inflated_cfga_ada', 'ecfga_inf', 'bonus_budget']


def _design(m_hub):
    n_hub, n_q = M.N_HUB, M.N_QUIET
    sizes = np.array([m_hub]*n_hub + [M.M_QUIET]*n_q, dtype=float)
    r1 = np.array([M.PI_HUB]*n_hub + [0.0]*n_q)
    mu = [(M.MU_HUB-.5, M.MU_HUB+.5)]*n_hub + [(0., 0.)]*n_q
    return sizes, r1, mu


def _trial(args):
    cfg, t = args
    sizes, r1, mu = _design(CONFIGS[cfg])
    m_total = int(sizes.sum()); alpha = M.ALPHA0
    rng = np.random.default_rng(717171 + 6661*t + CONFIGS[cfg])
    m_per_node = list(rng.multinomial(m_total, sizes/sizes.sum()))   # random labels
    pv, isn = s.generate_pvalues(m_per_node, r1, mu, rng)
    n1 = sum(int(np.sum(~isn[i])) for i in range(len(pv)))
    eps = alpha/np.sqrt(max(m_total, 100)); grid = s._eps_grid_for(m_total, alpha)
    st = rng.bit_generator.state
    out = {}
    for meth in METHODS:
        rng.bit_generator.state = st          # identical stream per method
        if meth == 'cfga_storey_ada':
            V, R = s.method_A_prime_storey_adaptive(pv, isn, alpha, grid, rng)
        elif meth == 'inflated_cfga_ada':
            V, R = s.augmented_cfga_storey_adaptive(pv, isn, alpha, grid, rng)
        elif meth == 'ecfga_inf':
            V, R = ecfga.e_cfga_inflated(pv, isn, alpha, eps, rng, S=S_SPLITS, eps_grid=grid)
        else:
            V, R = bp.bonus_cfga_pernode_budget(pv, isn, alpha, eps, rng)
        out[meth] = (V/max(R, 1), (R-V)/max(n1, 1))
    return out


if __name__ == '__main__':
    res = {}
    for cfg in CONFIGS:
        t0 = time.time()
        with Pool(4) as pool:
            rows = pool.map(_trial, [(cfg, t) for t in range(NT)])
        for meth in METHODS:
            f = np.array([r[meth][0] for r in rows]); p = np.array([r[meth][1] for r in rows])
            res[f'fdp_{cfg}_{meth}'] = f; res[f'pwr_{cfg}_{meth}'] = p
            print(f"{cfg:9s} {meth:18s} FDR={f.mean():.3f}±{2*f.std(ddof=1)/np.sqrt(NT):.3f} "
                  f"pwr={p.mean():.3f}±{2*p.std(ddof=1)/np.sqrt(NT):.3f}")
        print(f"  ({time.time()-t0:.0f}s)")
    np.savez('outputs/sim_random_labels.npz', NT=NT, **res)
    print("saved")

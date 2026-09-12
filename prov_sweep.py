"""Compute the two PROVABLE-TIER methods at the same sweep points (same
configs and per-trial seeds) as the Experiment 1-6 baselines, so they can be
overlaid on the existing figures:

  * e-CFGA (inflated plug-in, S=10 splits, per-fold adaptive eps)
        -- Theorem thm:ecfga, FDR <= alpha + eta
  * per-node budgeted BONuS-GA
        -- Theorem thm:bonus-pernode, FDR <= alpha

Usage (chunked; per-point results accumulate and resume):
  python prov_sweep.py <axis> <method> [n_points]   # run next unfinished points
  python prov_sweep.py assemble                     # write sim_prov_<axis>.npz
  python prov_sweep.py status
axis in {quiet, mu, rho, alpha, n, cauchy};  method in {ecfga_inf, bonus_budget}.

Assembled outputs (the durable artifacts, aligned to each axis grid):
  outputs/sim_prov_<axis>.npz  with arrays
      x, fdr_ecfga_inf, pwr_ecfga_inf, fdr_bonus_budget, pwr_bonus_budget
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
import simulations as s
import ecfga
import bonus_pernode as bp
import mn_sweep as M

OUT = M.OUT
NT = 100          # matches the Experiment 1-6 baselines
S_SPLITS = 10
QUIET = [0, 50, 100, 200, 300, 450, 600]
METHODS = ['ecfga_inf', 'bonus_budget']



def _cache_fingerprint(axis, val, method):
    """Configuration + source fingerprint. Any change to trial counts,
    method defaults, or the implementations of the imported modules
    invalidates previously cached per-point results."""
    import hashlib, inspect
    import simulations as _s, ecfga as _e, bonus_pernode as _bp
    src = "".join(inspect.getsource(mod) for mod in (_s, _e, _bp))
    key = f"{axis}|{val}|{method}|NT={NT}|" + hashlib.sha256(src.encode()).hexdigest()[:16]
    return key

def _point_config(axis, val):
    """(m_per_node, r1, mu_bounds, alpha, rho, dist, seed_fn) at one point,
    reproducing mn_sweep.one_point / sweep_quiet_nodes.one_config exactly."""
    if axis == 'quiet':
        n_null = int(val)
        m_per_node = [1000] * 2 + [20] * n_null
        r1 = np.array([0.30] * 2 + [0.0] * n_null)
        mu_bounds = [(2.5, 3.5)] * 2 + [(0.0, 0.0)] * n_null
        return (m_per_node, r1, mu_bounds, 0.2, 0.0, 'gaussian',
                lambda t: 515151 + 6661 * t + n_null * 17)
    m_per_node, r1, mu_bounds, alpha, rho, dist = M.build(axis, val)
    return (m_per_node, r1, mu_bounds, alpha, rho, dist,
            lambda t: 313131 + 6661 * t + int(round(float(val) * 1000)))


def _one_trial(args):
    axis, val, method, t = args
    m_per_node, r1, mu_bounds, alpha, rho, dist, seed_fn = _point_config(axis, val)
    m_total = sum(m_per_node)
    eps = alpha / np.sqrt(max(m_total, 100))
    rng = np.random.default_rng(seed_fn(t))
    pv, isn = s.generate_pvalues(m_per_node, r1, mu_bounds, rng,
                                 rho=rho, distribution=dist)
    n1 = sum(int(np.sum(~isn[i])) for i in range(len(pv)))
    if method == 'ecfga_inf':
        grid = s._eps_grid_for(m_total, alpha)
        V, R = ecfga.e_cfga_inflated(pv, isn, alpha, eps, rng,
                                     S=S_SPLITS, eps_grid=grid)
    elif method == 'bonus_budget':
        V, R = bp.bonus_cfga_pernode_budget(pv, isn, alpha, eps, rng)
    else:
        raise ValueError(method)
    return V / max(R, 1), (R - V) / max(n1, 1)


def _values(axis):
    return QUIET if axis == 'quiet' else M.AXES[axis]['values']


def _pt_path(axis, val, method):
    tag = str(val).replace('.', 'p').replace('-', 'm')
    return os.path.join(OUT, f'_prov_{axis}_{tag}_{method}.npz')


def run_axis(axis, method, max_points=99):
    done = 0
    for val in _values(axis):
        if done >= max_points:
            break
        path = _pt_path(axis, val, method)
        cfg = _cache_fingerprint(axis, val, method)
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            if len(d['f']) >= NT and str(d.get('cfg', '')) == cfg:
                continue  # cache valid
            # stale cache (config or implementation changed): recompute

        t0 = time.time()
        with Pool(4) as pool:
            res = pool.map(_one_trial,
                           [(axis, val, method, t) for t in range(NT)])
        f = np.array([r[0] for r in res]); p = np.array([r[1] for r in res])
        np.savez(path, f=f, p=p, cfg=_cache_fingerprint(axis, val, method))
        print(f'[{axis}={val} {method}] FDR={f.mean():.3f} pwr={p.mean():.3f} '
              f'({time.time()-t0:.0f}s)')
        done += 1


def assemble():
    for axis in ['quiet', 'mu', 'rho', 'alpha', 'n', 'cauchy']:
        vals = _values(axis)
        out = {'x': np.array(vals, dtype=float)}
        complete = True
        for method in METHODS:
            F, P = [], []
            for val in vals:
                path = _pt_path(axis, val, method)
                if not os.path.exists(path):
                    complete = False
                    break
                d = np.load(path)
                F.append(float(d['f'].mean())); P.append(float(d['p'].mean()))
            if not complete:
                break
            out[f'fdr_{method}'] = np.array(F)
            out[f'pwr_{method}'] = np.array(P)
        if complete:
            np.savez(os.path.join(OUT, f'sim_prov_{axis}.npz'), **out)
            print(f'wrote sim_prov_{axis}.npz')
        else:
            print(f'{axis}: incomplete, skipped')


def status():
    for axis in ['quiet', 'mu', 'rho', 'alpha', 'n', 'cauchy']:
        for method in METHODS:
            n = sum(os.path.exists(_pt_path(axis, v, method))
                    for v in _values(axis))
            print(f'{axis:7s} {method:13s} {n}/{len(_values(axis))} points')


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == 'assemble':
        assemble()
    elif a and a[0] == 'status':
        status()
    else:
        run_axis(a[0], a[1], int(a[2]) if len(a) > 2 else 99)

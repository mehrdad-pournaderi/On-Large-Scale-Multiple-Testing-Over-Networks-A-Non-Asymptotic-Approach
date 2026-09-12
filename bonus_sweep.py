"""Compute ONLY BONuS-CFGA at the same sweep points (same configs/seeds) as
mn_sweep.py / sweep_quiet_nodes.py, so it can be overlaid on the existing
figures without recomputing the 8 baseline methods.

  python bonus_sweep.py <axis> [<axis> ...]   # axis in {mu,rho,alpha,n,cauchy,quiet,all}

Saves sim_bonus_<axis>.npz with arrays x, fdr, pwr (aligned to the axis grid).
"""
import os, sys, time
import numpy as np
import mn_sweep as M
import simulations as s

OUT = M.OUT
NT = M.NT  # 100, matches the baselines


def _axis_point(axis, val):
    """BONuS FDR/power at one (axis,val), reproducing mn_sweep.one_point's
    config + per-trial seed exactly."""
    m_per_node, r1, mu_bounds, alpha, rho, dist = M.build(axis, val)
    N = len(m_per_node)
    m_total = sum(m_per_node)
    eps = alpha / np.sqrt(max(m_total, 100))
    f_acc, p_acc = [], []
    for t in range(NT):
        rng = np.random.default_rng(313131 + 6661 * t + int(round(float(val) * 1000)))
        pv, isn = s.generate_pvalues(m_per_node, r1, mu_bounds, rng,
                                     rho=rho, distribution=dist)
        n1 = sum(int(np.sum(~isn[i])) for i in range(N))
        V, R = s.bonus_cfga(pv, isn, alpha, eps, rng)
        f_acc.append(V / max(R, 1)); p_acc.append((R - V) / max(n1, 1))
    return float(np.mean(f_acc)), float(np.mean(p_acc))


def _quiet_point(n_null):
    """BONuS FDR/power for sweep_quiet_nodes.one_config, same config + seeds."""
    N_RICH, M_RICH, PI_RICH, MU_RICH = 2, 1000, 0.30, 3.0
    M_NULL, ALPHA = 20, 0.2
    N = N_RICH + n_null
    m_per_node = [M_RICH] * N_RICH + [M_NULL] * n_null
    r1 = np.array([PI_RICH] * N_RICH + [0.0] * n_null)
    mu_bounds = [(MU_RICH - 0.5, MU_RICH + 0.5)] * N_RICH + [(0.0, 0.0)] * n_null
    m_total = sum(m_per_node)
    eps = ALPHA / np.sqrt(max(m_total, 100))
    f_acc, p_acc = [], []
    for t in range(NT):
        rng = np.random.default_rng(515151 + 6661 * t + n_null * 17)
        pv, isn = s.generate_pvalues(m_per_node, r1, mu_bounds, rng)
        n1 = sum(int(np.sum(~isn[i])) for i in range(N))
        V, R = s.bonus_cfga(pv, isn, ALPHA, eps, rng)
        f_acc.append(V / max(R, 1)); p_acc.append((R - V) / max(n1, 1))
    return float(np.mean(f_acc)), float(np.mean(p_acc))


def do_axis(axis):
    t0 = time.time()
    if axis == 'quiet':
        xs = [0, 50, 100, 200, 300, 450, 600]
        fdr, pwr = [], []
        for nq in xs:
            f, p = _quiet_point(nq); fdr.append(f); pwr.append(p)
            print(f"  quiet={nq:4d}  FDR={f:.3f}  pwr={p:.3f}")
    else:
        xs = M.AXES[axis]['values']
        fdr, pwr = [], []
        for val in xs:
            f, p = _axis_point(axis, val); fdr.append(f); pwr.append(p)
            print(f"  {axis}={val}  FDR={f:.3f}  pwr={p:.3f}")
    np.savez(os.path.join(OUT, f'sim_bonus_{axis}.npz'),
             x=np.array(xs, dtype=float), fdr=np.array(fdr), pwr=np.array(pwr))
    print(f"  saved sim_bonus_{axis}.npz ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    args = sys.argv[1:] or ['all']
    if args == ['all']:
        args = ['mu', 'rho', 'alpha', 'n', 'cauchy', 'quiet']
    for ax in args:
        do_axis(ax)

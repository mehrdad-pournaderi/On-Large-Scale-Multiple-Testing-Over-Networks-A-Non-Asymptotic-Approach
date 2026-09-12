"""'Monitoring network' sweep: a few signal-rich hub nodes + a growing number of
quiet (pure-null) small nodes. As quiet nodes accumulate, the union of per-node
BH at level alpha (Local BH(alpha)) accumulates false rejections it cannot
suppress within tiny nodes -> its POOLED (global) FDR climbs past alpha. The
global methods (Pooled BH, CFGA, CFGA+Storey) keep global FDR <= alpha.

Saves sim_localbh_break.{png,npz}.
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simulations as s

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)
ALPHA = 0.2
N_RICH, M_RICH, PI_RICH, MU_RICH = 2, 1000, 0.30, 3.0
M_NULL = 20
QUIET = [0, 50, 100, 200, 300, 450, 600]
NT = 100
PLOT = ['Local BH (alpha)', 'Local BH (alpha/N)', 'Pooled BH', 'Original Greedy',
        'CFGA + Storey', 'CFGA + Storey + adaptive eps',
        'Augmented CFGA + Storey', 'Augmented CFGA + Storey + adaptive eps']


def one_config(n_null, n_trials, seed_base=515151):
    N = N_RICH + n_null
    m_per_node = [M_RICH] * N_RICH + [M_NULL] * n_null
    r1 = np.array([PI_RICH] * N_RICH + [0.0] * n_null)
    mu_bounds = ([(MU_RICH - 0.5, MU_RICH + 0.5)] * N_RICH +
                 [(0.0, 0.0)] * n_null)
    m_total = sum(m_per_node)
    eps = ALPHA / np.sqrt(max(m_total, 100))
    eps_grid = s._eps_grid_for(m_total, ALPHA)
    acc = {k: [[], []] for k in PLOT}
    for t in range(n_trials):
        rng = np.random.default_rng(seed_base + 6661 * t + n_null * 17)
        pv, isn = s.generate_pvalues(m_per_node, r1, mu_bounds, rng)
        n1 = sum(int(np.sum(~isn[i])) for i in range(N))
        def stat(V, R): return V / max(R, 1), (R - V) / max(n1, 1)
        pp = np.concatenate(pv); pn = np.concatenate(isn)
        sel, _ = s.greedy_aggregation(pv, ALPHA, eps)
        res = {
            'Pooled BH': s.bh(pp, pn, ALPHA),
            'Local BH (alpha)': s.local_bh(pv, isn, ALPHA, lambda N: ALPHA),
            'Local BH (alpha/N)': s.local_bh(pv, isn, ALPHA, lambda N: ALPHA / N),
            'Original Greedy': s.evaluate_selected(pv, isn, sel),
            'CFGA + Storey': s.method_A_prime_storey(pv, isn, ALPHA, eps, rng),
            'CFGA + Storey + adaptive eps': s.method_A_prime_storey_adaptive(pv, isn, ALPHA, eps_grid, rng),
            'Augmented CFGA + Storey': s.augmented_cfga_storey(pv, isn, ALPHA, eps, rng),
            'Augmented CFGA + Storey + adaptive eps': s.augmented_cfga_storey_adaptive(pv, isn, ALPHA, eps_grid, rng),
        }
        for k, (V, R) in res.items():
            f, p = stat(V, R); acc[k][0].append(f); acc[k][1].append(p)
    fdr = {k: float(np.mean(acc[k][0])) for k in PLOT}
    fse = {k: 2 * float(np.std(acc[k][0], ddof=1) / np.sqrt(n_trials)) for k in PLOT}
    pwr = {k: float(np.mean(acc[k][1])) for k in PLOT}
    return fdr, fse, pwr, m_total, N


def _cfg_path(nq):
    return os.path.join(OUT, f'_qn_{nq}.npz')


def compute_subset(quiet_list):
    """Compute and save per-config results for each nq in quiet_list."""
    import time as _t
    for nq in quiet_list:
        t0 = _t.time()
        fdr, fse, pwr, mtot, N = one_config(nq, NT)
        np.savez(_cfg_path(nq), nq=nq, N=N, mtot=mtot,
                 **{f'fdr_{k}': fdr[k] for k in PLOT},
                 **{f'fse_{k}': fse[k] for k in PLOT},
                 **{f'pwr_{k}': pwr[k] for k in PLOT})
        print(f"quiet={nq:4d} N={N:4d} m_tot={mtot:6d} ({_t.time()-t0:.1f}s) | "
              f"LBH(a)={fdr['Local BH (alpha)']:.3f} "
              f"OrigGrdy={fdr['Original Greedy']:.3f}/{pwr['Original Greedy']:.3f} "
              f"CFGA+Sto={fdr['CFGA + Storey']:.3f}/{pwr['CFGA + Storey']:.3f} "
              f"CFGA+Sto+Ada={fdr['CFGA + Storey + adaptive eps']:.3f}/{pwr['CFGA + Storey + adaptive eps']:.3f} "
              f"Aug+Sto+Ada={fdr['Augmented CFGA + Storey + adaptive eps']:.3f}/{pwr['Augmented CFGA + Storey + adaptive eps']:.3f}")


def assemble_and_plot():
    FDR = {k: [] for k in PLOT}; FSE = {k: [] for k in PLOT}; PWR = {k: [] for k in PLOT}
    Ntot = []; xs = []
    for nq in QUIET:
        p = _cfg_path(nq)
        if not os.path.exists(p):
            print(f"  MISSING {p} -- run: python sweep_quiet_nodes.py {nq}")
            continue
        d = np.load(p, allow_pickle=True)
        xs.append(int(d['nq'])); Ntot.append(int(d['N']))
        for k in PLOT:
            FDR[k].append(float(d[f'fdr_{k}'])); FSE[k].append(float(d[f'fse_{k}']))
            PWR[k].append(float(d[f'pwr_{k}']))
    np.savez(os.path.join(OUT, 'sim_localbh_break.npz'),
             quiet=np.array(xs), Ntot=np.array(Ntot),
             **{f'fdr_{k}': np.array(FDR[k]) for k in PLOT},
             **{f'fdrse_{k}': np.array(FSE[k]) for k in PLOT},
             **{f'pwr_{k}': np.array(PWR[k]) for k in PLOT})

    style = {
        'Local BH (alpha)': dict(color='#d62728', marker='o', ls='-'),
        'Local BH (alpha/N)': dict(color='#ff9896', marker='v', ls='--'),
        'Pooled BH': dict(color='#7f7f7f', marker='s', ls=':'),
        'Original Greedy': dict(color='#8c564b', marker='x', ls='--'),
        'CFGA + Storey': dict(color='#2ca02c', marker='^', ls='-'),
        'CFGA + Storey + adaptive eps': dict(color='#1f77b4', marker='D', ls='-'),
        'Augmented CFGA + Storey': dict(color='#9467bd', marker='P', ls='-.'),
        'Augmented CFGA + Storey + adaptive eps': dict(color='#ff7f0e', marker='*', ls='-.'),
    }
    label = {
        'Local BH (alpha)': r'Local BH ($\alpha$)',
        'Local BH (alpha/N)': r'Local BH ($\alpha/N$)',
        'Pooled BH': 'Pooled BH',
        'Original Greedy': 'Orig. Greedy',
        'CFGA + Storey': 'CFGA+Storey',
        'CFGA + Storey + adaptive eps': r'CFGA+Storey+Ada $\varepsilon$',
        'Augmented CFGA + Storey': 'Inflated CFGA+Storey',
        'Augmented CFGA + Storey + adaptive eps': r'Inflated CFGA+Storey+Ada $\varepsilon$',
    }
    x = np.array(xs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for k in PLOT:
        st = style[k]
        axes[0].plot(x, FDR[k], color=st['color'], marker=st['marker'],
                     linestyle=st['ls'], label=label[k], markersize=6)
        axes[1].plot(x, PWR[k], color=st['color'], marker=st['marker'],
                     linestyle=st['ls'], label=label[k], markersize=6)
    axes[0].axhline(ALPHA, color='black', ls=':', lw=1)
    axes[0].text(x[-1], ALPHA + 0.008, r'target $\alpha=0.2$', ha='right', fontsize=8)
    axes[0].set_ylabel('global FDR'); axes[0].set_title(r'Global FDR vs. number of quiet nodes')
    axes[1].set_ylabel('power'); axes[1].set_title('Power vs. number of quiet nodes')
    for ax in axes:
        ax.set_xlabel('number of quiet (pure-null) nodes')
        ax.grid(alpha=0.3)
    # Legend only on the power (right) panel
    axes[1].legend(loc='best', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'sim_localbh_break.png'), dpi=130, bbox_inches='tight')
    plt.close()
    print('saved sim_localbh_break.png/.npz')


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if not args or args[0] == 'all':
        compute_subset(QUIET); assemble_and_plot()
    elif args[0] == 'plot':
        assemble_and_plot()
    else:
        compute_subset([int(a) for a in args])

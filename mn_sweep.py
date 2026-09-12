"""Re-run the standard experiment axes under the MONITORING-NETWORK base config:
   n_hub signal-rich hubs (m_hub hyps, pi_hub signal, effect mu_hub)
 + n_quiet quiet pure-null nodes (m_quiet hyps each).
This is the regime where Local BH(alpha) violates global FDR; we sweep each axis
and record all methods, including the adaptive-eps augmented rescue.

Chunked execution (each (axis,value) saved to disk):
  python mn_sweep.py <axis> <val1> <val2> ...   # compute these values
  python mn_sweep.py plot <axis>                # assemble figure sim_mn_<axis>.png/.npz
  python mn_sweep.py axes                        # list axes + default values
axis in {mu, rho, alpha, n, cauchy}.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simulations as s

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)

# ---- monitoring-network base ----
N_HUB, M_HUB, PI_HUB, MU_HUB = 2, 1000, 0.30, 3.0
MU_HUB_CAUCHY = 8.0   # Cauchy tails are heavy: need a larger shift (cf. orig Exp 2b, mu in 6..14)
N_QUIET, M_QUIET = 300, 20
ALPHA0 = 0.2
NT = 100

PLOT = ['Local BH (alpha)', 'Local BH (alpha/N)', 'Pooled BH', 'Original Greedy',
        'CFGA + Storey', 'CFGA + Storey + adaptive eps',
        'Augmented CFGA + Storey', 'Augmented CFGA + Storey + adaptive eps']

STYLE = {
    'Local BH (alpha)': dict(color='#d62728', marker='o', ls='-'),
    'Local BH (alpha/N)': dict(color='#ff9896', marker='v', ls='--'),
    'Pooled BH': dict(color='#7f7f7f', marker='s', ls=':'),
    'Original Greedy': dict(color='#8c564b', marker='x', ls='--'),
    'CFGA + Storey': dict(color='#2ca02c', marker='^', ls='-'),
    'CFGA + Storey + adaptive eps': dict(color='#1f77b4', marker='D', ls='-'),
    'Augmented CFGA + Storey': dict(color='#9467bd', marker='P', ls='-.'),
    'Augmented CFGA + Storey + adaptive eps': dict(color='#ff7f0e', marker='*', ls='-.'),
}

# compact legend labels
LABEL = {
    'Local BH (alpha)': r'Local BH ($\alpha$)',
    'Local BH (alpha/N)': r'Local BH ($\alpha/N$)',
    'Pooled BH': 'Pooled BH',
    'Original Greedy': 'Orig. Greedy',
    'CFGA + Storey': 'CFGA+Storey',
    'CFGA + Storey + adaptive eps': r'CFGA+Storey+Ada $\varepsilon$',
    'Augmented CFGA + Storey': 'Inflated CFGA+Storey',
    'Augmented CFGA + Storey + adaptive eps': r'Inflated CFGA+Storey+Ada $\varepsilon$',
}

AXES = {
    'mu':     dict(values=[1.5, 2.0, 2.5, 3.0, 3.5], xlabel=r'hub signal strength $\mu_{\mathrm{hub}}$',
                   title='vs. hub signal strength', xlog=False),
    'rho':    dict(values=[0.0, 0.2, 0.4, 0.6, 0.8], xlabel=r'within-node AR(1) correlation $\rho$',
                   title='vs. within-node correlation', xlog=False),
    'alpha':  dict(values=[0.05, 0.10, 0.15, 0.20], xlabel=r'target FDR $\alpha$',
                   title='vs. target FDR level', xlog=False),
    'n':      dict(values=[250, 500, 1000, 2000], xlabel=r'hub size $m_{\mathrm{hub}}$',
                   title='vs. hub sample size', xlog=True),
    'cauchy': dict(values=[500, 1000, 2000], xlabel=r'hub size $m_{\mathrm{hub}}$ (Cauchy stats)',
                   title='vs. hub size (heavy-tailed)', xlog=True),
}


def build(axis, val):
    """Return (m_per_node, r1, mu_bounds, alpha, rho, distribution)."""
    n_hub, m_hub, pi_hub, mu_hub = N_HUB, M_HUB, PI_HUB, MU_HUB
    alpha, rho, dist = ALPHA0, 0.0, 'gaussian'
    if axis == 'mu':
        mu_hub = val
    elif axis == 'rho':
        rho = val
    elif axis == 'alpha':
        alpha = val
    elif axis == 'n':
        m_hub = int(val)
    elif axis == 'cauchy':
        m_hub = int(val); dist = 'cauchy'; mu_hub = MU_HUB_CAUCHY
    else:
        raise SystemExit(f'unknown axis {axis!r}')
    m_per_node = [m_hub] * n_hub + [M_QUIET] * N_QUIET
    r1 = np.array([pi_hub] * n_hub + [0.0] * N_QUIET)
    mu_bounds = ([(mu_hub - 0.5, mu_hub + 0.5)] * n_hub +
                 [(0.0, 0.0)] * N_QUIET)
    return m_per_node, r1, mu_bounds, alpha, rho, dist


def one_point(axis, val, n_trials, seed_base=313131):
    m_per_node, r1, mu_bounds, alpha, rho, dist = build(axis, val)
    N = len(m_per_node)
    m_total = sum(m_per_node)
    eps = alpha / np.sqrt(max(m_total, 100))
    eps_grid = s._eps_grid_for(m_total, alpha)
    acc = {k: [[], []] for k in PLOT}
    for t in range(n_trials):
        rng = np.random.default_rng(seed_base + 6661 * t + int(round(float(val) * 1000)))
        pv, isn = s.generate_pvalues(m_per_node, r1, mu_bounds, rng,
                                     rho=rho, distribution=dist)
        n1 = sum(int(np.sum(~isn[i])) for i in range(N))
        def stat(V, R): return V / max(R, 1), (R - V) / max(n1, 1)
        pp = np.concatenate(pv); pn = np.concatenate(isn)
        sel, _ = s.greedy_aggregation(pv, alpha, eps)
        res = {
            'Pooled BH': s.bh(pp, pn, alpha),
            'Local BH (alpha)': s.local_bh(pv, isn, alpha, lambda N: alpha),
            'Local BH (alpha/N)': s.local_bh(pv, isn, alpha, lambda N: alpha / N),
            'Original Greedy': s.evaluate_selected(pv, isn, sel),
            'CFGA + Storey': s.method_A_prime_storey(pv, isn, alpha, eps, rng),
            'CFGA + Storey + adaptive eps': s.method_A_prime_storey_adaptive(pv, isn, alpha, eps_grid, rng),
            'Augmented CFGA + Storey': s.augmented_cfga_storey(pv, isn, alpha, eps, rng),
            'Augmented CFGA + Storey + adaptive eps': s.augmented_cfga_storey_adaptive(pv, isn, alpha, eps_grid, rng),
        }
        for k, (V, R) in res.items():
            f, p = stat(V, R); acc[k][0].append(f); acc[k][1].append(p)
    fdr = {k: float(np.mean(acc[k][0])) for k in PLOT}
    fse = {k: 2 * float(np.std(acc[k][0], ddof=1) / np.sqrt(n_trials)) for k in PLOT}
    pwr = {k: float(np.mean(acc[k][1])) for k in PLOT}
    return fdr, fse, pwr, N, m_total, alpha


def _pt_path(axis, val):
    tag = str(val).replace('.', 'p').replace('-', 'm')
    return os.path.join(OUT, f'_mn_{axis}_{tag}.npz')


def compute_subset(axis, vals):
    for val in vals:
        t0 = time.time()
        fdr, fse, pwr, N, mtot, alpha = one_point(axis, val, NT)
        np.savez(_pt_path(axis, val), axis=axis, val=val, N=N, mtot=mtot, alpha=alpha,
                 **{f'fdr_{k}': fdr[k] for k in PLOT},
                 **{f'fse_{k}': fse[k] for k in PLOT},
                 **{f'pwr_{k}': pwr[k] for k in PLOT})
        print(f"[{axis}={val}] N={N} m_tot={mtot} a={alpha} ({time.time()-t0:.1f}s) | "
              f"LBH(a)={fdr['Local BH (alpha)']:.3f} "
              f"OrigGrdy={fdr['Original Greedy']:.3f}/{pwr['Original Greedy']:.3f} "
              f"CFGA+Sto={fdr['CFGA + Storey']:.3f}/{pwr['CFGA + Storey']:.3f} "
              f"CFGA+Sto+Ada={fdr['CFGA + Storey + adaptive eps']:.3f}/{pwr['CFGA + Storey + adaptive eps']:.3f} "
              f"Aug+Sto+Ada={fdr['Augmented CFGA + Storey + adaptive eps']:.3f}/{pwr['Augmented CFGA + Storey + adaptive eps']:.3f}")


def assemble_and_plot(axis):
    spec = AXES[axis]; vals = spec['values']
    FDR = {k: [] for k in PLOT}; FSE = {k: [] for k in PLOT}; PWR = {k: [] for k in PLOT}
    xs = []
    for val in vals:
        p = _pt_path(axis, val)
        if not os.path.exists(p):
            print(f"  MISSING {p} -- run: python mn_sweep.py {axis} {val}"); continue
        d = np.load(p, allow_pickle=True)
        xs.append(val)
        for k in PLOT:
            FDR[k].append(float(d[f'fdr_{k}'])); FSE[k].append(float(d[f'fse_{k}']))
            PWR[k].append(float(d[f'pwr_{k}']))
    np.savez(os.path.join(OUT, f'sim_mn_{axis}.npz'), x=np.array(xs),
             **{f'fdr_{k}': np.array(FDR[k]) for k in PLOT},
             **{f'fdrse_{k}': np.array(FSE[k]) for k in PLOT},
             **{f'pwr_{k}': np.array(PWR[k]) for k in PLOT})
    x = np.array(xs, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for k in PLOT:
        st = STYLE[k]
        axes[0].plot(x, FDR[k], color=st['color'], marker=st['marker'],
                     linestyle=st['ls'], label=LABEL[k], markersize=6)
        axes[1].plot(x, PWR[k], color=st['color'], marker=st['marker'],
                     linestyle=st['ls'], label=LABEL[k], markersize=6)
    a_ref = ALPHA0 if axis != 'alpha' else None
    if axis == 'alpha':
        axes[0].plot(x, x, color='black', ls=':', lw=1)
        axes[0].text(x[-1], x[-1] + 0.008, r'$y=\alpha$', ha='right', fontsize=8)
    else:
        axes[0].axhline(ALPHA0, color='black', ls=':', lw=1)
        axes[0].text(x[-1], ALPHA0 + 0.008, r'target $\alpha=0.2$', ha='right', fontsize=8)
    axes[0].set_ylabel('global FDR'); axes[0].set_title(f'Global FDR {spec["title"]}')
    axes[1].set_ylabel('power'); axes[1].set_title(f'Power {spec["title"]}')
    for ax in axes:
        ax.set_xlabel(spec['xlabel']); ax.grid(alpha=0.3)
        if spec['xlog']:
            ax.set_xscale('log')
    # Legend only on the power (right) panel
    axes[1].legend(loc='best', fontsize=7)
    plt.suptitle(f'Monitoring network: {N_HUB} hubs (m={M_HUB}, '
                 rf'$\pi$={PI_HUB}) + {N_QUIET} quiet nodes (m={M_QUIET})', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT, f'sim_mn_{axis}.png'), dpi=130, bbox_inches='tight')
    plt.close()
    print(f'saved sim_mn_{axis}.png/.npz  (x={xs})')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a or a[0] == 'axes':
        for k, v in AXES.items():
            print(f"{k:7s} values={v['values']}")
    elif a[0] == 'plot':
        assemble_and_plot(a[1])
    else:
        axis = a[0]
        if a[1:]:
            vals = ([int(float(x)) for x in a[1:]] if axis in ('n', 'cauchy')
                    else [float(x) for x in a[1:]])
        else:
            vals = AXES[axis]['values']
        compute_subset(axis, vals)

"""Re-draw the 6 monitoring-network figures from the assembled .npz files,
using the updated compact LABEL map (no recomputation). The per-point _mn_*
files were cleaned up, so we read sim_mn_<axis>.npz / sim_localbh_break.npz."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mn_sweep as M

OUT = M.OUT

# BONuS-GA overlay (computed separately by bonus_sweep.py; no sample split)
BONUS_STYLE = dict(color='#17becf', marker='h', ls='-')
BONUS_LABEL = 'BONuS-GA'

# Provable-tier overlays (computed by prov_sweep.py; sim_prov_<axis>.npz)
PROV = {
    'ecfga_inf':    dict(style=dict(color='#e377c2', marker='p', ls='-'),
                         label=r'e-CFGA (infl.$+$Ada $\varepsilon$)'),
    'bonus_budget': dict(style=dict(color='black', marker='*', ls='--'),
                         label='Budgeted BONuS-GA'),
}


def _load_prov(tag):
    p = f'{OUT}/sim_prov_{tag}.npz'
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True)


def _mn_fse(axis, vals):
    """FDR 2*SE per baseline method from the per-point caches (if present)."""
    import mn_sweep as _M
    out = {k: [] for k in _M.PLOT}
    for v in vals:
        p = _M._pt_path(axis, v)
        if not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=True)
        for k in _M.PLOT:
            key = f'fse_{k}'
            out[k].append(float(d[key]) if key in d else np.nan)
    return out


def _qn_fse(quiet_vals):
    import sweep_quiet_nodes as _Q
    out = {k: [] for k in _Q.PLOT}
    for nq in quiet_vals:
        p = _Q._cfg_path(int(nq))
        if not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=True)
        for k in _Q.PLOT:
            key = f'fse_{k}'
            out[k].append(float(d[key]) if key in d else np.nan)
    return out


def _prov_se(tag, vals):
    """(fdr_2se, pwr_2se) per provable method from per-trial caches."""
    import prov_sweep as _P
    out = {}
    for meth in PROV:
        fs, ps = [], []
        for v in vals:
            p = _P._pt_path(tag, v, meth)
            if not os.path.exists(p):
                return None
            d = np.load(p, allow_pickle=True)
            f, pw = d['f'], d['p']
            fs.append(2*float(np.std(f, ddof=1))/np.sqrt(len(f)))
            ps.append(2*float(np.std(pw, ddof=1))/np.sqrt(len(pw)))
        out[meth] = (np.array(fs), np.array(ps))
    return out


def _draw(x, d, title_l, title_r, xlabel, fname, xlog=False, alpha_diag=False,
          bonus=None, prov=None, base_fse=None, prov_se=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for k in M.PLOT:
        st = M.STYLE[k]
        axes[0].plot(x, d[f'fdr_{k}'], color=st['color'], marker=st['marker'],
                     linestyle=st['ls'], label=M.LABEL[k], markersize=6)
        axes[1].plot(x, d[f'pwr_{k}'], color=st['color'], marker=st['marker'],
                     linestyle=st['ls'], label=M.LABEL[k], markersize=6)
    if bonus is not None:
        bx = np.array(bonus['x'], dtype=float)
        axes[0].plot(bx, bonus['fdr'], color=BONUS_STYLE['color'],
                     marker=BONUS_STYLE['marker'], linestyle=BONUS_STYLE['ls'],
                     label=BONUS_LABEL, markersize=6)
        axes[1].plot(bx, bonus['pwr'], color=BONUS_STYLE['color'],
                     marker=BONUS_STYLE['marker'], linestyle=BONUS_STYLE['ls'],
                     label=BONUS_LABEL, markersize=6)
    if prov is not None:
        px = np.array(prov['x'], dtype=float)
        for key, spec in PROV.items():
            st = spec['style']
            axes[0].plot(px, prov[f'fdr_{key}'], color=st['color'],
                         marker=st['marker'], linestyle=st['ls'],
                         label=spec['label'], markersize=7)
            axes[1].plot(px, prov[f'pwr_{key}'], color=st['color'],
                         marker=st['marker'], linestyle=st['ls'],
                         label=spec['label'], markersize=7)
    if alpha_diag:
        axes[0].plot(x, x, color='black', ls=':', lw=1)
        axes[0].text(x[-1], x[-1] + 0.008, r'$y=\alpha$', ha='right', fontsize=8)
    else:
        axes[0].axhline(M.ALPHA0, color='black', ls=':', lw=1)
        axes[0].text(x[-1], M.ALPHA0 + 0.008, r'target $\alpha=0.2$', ha='right', fontsize=8)
    axes[0].set_ylabel('global FDR'); axes[0].set_title(title_l)
    axes[1].set_ylabel('power'); axes[1].set_title(title_r)
    for ax in axes:
        ax.set_xlabel(xlabel); ax.grid(alpha=0.3)
        if xlog:
            ax.set_xscale('log')
    # Legend only on the power (right) panel
    axes[1].legend(loc='best', fontsize=7)
    return fig, axes


def _load_bonus(tag):
    p = f'{OUT}/sim_bonus_{tag}.npz'
    if not os.path.exists(p):
        return None
    b = np.load(p, allow_pickle=True)
    return {'x': b['x'], 'fdr': b['fdr'], 'pwr': b['pwr']}


for axis, spec in M.AXES.items():
    d = np.load(f'{OUT}/sim_mn_{axis}.npz', allow_pickle=True)
    x = np.array(d['x'], dtype=float)
    fig, axes = _draw(x, d, f'Global FDR {spec["title"]}', f'Power {spec["title"]}',
                      spec['xlabel'], None, xlog=spec['xlog'], alpha_diag=(axis == 'alpha'),
                      bonus=_load_bonus(axis), prov=_load_prov(axis),
                      base_fse=_mn_fse(axis, list(d['x'])),
                      prov_se=_prov_se(axis, list(d['x'])))
    plt.suptitle(f'Monitoring network: {M.N_HUB} hubs (m={M.M_HUB}, '
                 rf'$\pi$={M.PI_HUB}) + {M.N_QUIET} quiet nodes (m={M.M_QUIET})', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{OUT}/sim_mn_{axis}.png', dpi=130, bbox_inches='tight'); plt.close()
    print('replotted sim_mn_%s.png' % axis)

# quiet-nodes figure
d = np.load(f'{OUT}/sim_localbh_break.npz', allow_pickle=True)
x = np.array(d['quiet'])
fig, axes = _draw(x, d, 'Global FDR vs. number of quiet nodes',
                  'Power vs. number of quiet nodes',
                  'number of quiet (pure-null) nodes', None,
                  bonus=_load_bonus('quiet'), prov=_load_prov('quiet'),
                  base_fse=_qn_fse(list(x)), prov_se=_prov_se('quiet', list(x)))
plt.tight_layout()
plt.savefig(f'{OUT}/sim_localbh_break.png', dpi=130, bbox_inches='tight'); plt.close()
print('replotted sim_localbh_break.png')

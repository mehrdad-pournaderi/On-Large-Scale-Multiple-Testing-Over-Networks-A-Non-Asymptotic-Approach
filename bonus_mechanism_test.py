"""Isolate the mechanism behind BONuS-GA's small-hub weakness.

At m_hub=250 (Experiment 4's worst point), test four variants:
  (1) Implementable: \\tilde m^{(i)} \\propto m^{(i)} * \\hat r_0^{(i)}, global ratio.
  (2) Oracle allocation: \\tilde m^{(i)} \\propto m^{(i)} * r_0^{(i)} (truth), global ratio.
  (3) Larger bag (c=4): same allocation, 4x the synthetic budget.
  (4) Per-node calibration: each node uses its own ratio N^{(i)}(A)/Nt^{(i)}(A).

If (2) >> (1): the Storey plug-in in the allocation is the culprit.
If (3) >> (1): the global-ratio noise floor is the culprit.
If (4) >> (1): the global-ratio choice (not its noise) is the culprit.
If all comparable: something else (e.g., greedy ranking dilution).
"""
import numpy as np
import simulations as s
import bonus_pernode as bp

# Experiment 4 config at the small-hub point
N_RICH, M_RICH, PI_RICH, MU_RICH = 2, 250, 0.30, 3.0
N_NULL, M_NULL = 300, 20
ALPHA = 0.20

N = N_RICH + N_NULL
m_per_node = [M_RICH] * N_RICH + [M_NULL] * N_NULL
r1 = np.array([PI_RICH] * N_RICH + [0.0] * N_NULL)
mu_bounds = [(MU_RICH - 0.5, MU_RICH + 0.5)] * N_RICH + [(0.0, 0.0)] * N_NULL
m_total = sum(m_per_node)
eps = ALPHA / np.sqrt(max(m_total, 100))

# True per-node null rate: 1 - r1 (null fraction in the random-effect model)
r0_true = 1.0 - r1


def _make_bag_oracle(pvalues_list, rng, c=1.0):
    """Bag with TRUE r_0^{(i)} weights and i.i.d. categorical labels
    (multinomial per-node sizes), matching Theorem 3 / Algorithm 3 Step 2."""
    n = len(pvalues_list)
    m_i = np.array([len(pvalues_list[i]) for i in range(n)], dtype=float)
    w = m_i * r0_true
    if w.sum() == 0:
        w = m_i
    target_total = int(round(c * float(m_i.sum())))
    sizes = rng.multinomial(target_total, w / max(w.sum(), 1e-12))
    return [rng.random(int(sizes[i])) for i in range(n)]


def bonus_oracle_alloc(pvalues_list, is_null_list, rng, c=1.0):
    """BONuS-GA with the ORACLE (true r_0) allocation, global ratio."""
    synth = _make_bag_oracle(pvalues_list, rng, c=c)
    return s._bonus_accept(pvalues_list, is_null_list, synth, ALPHA, eps, 0.5)


def bonus_bigbag(pvalues_list, is_null_list, rng, c=4.0):
    """BONuS-GA implementable, c=4 (four times the synthetic budget)."""
    return s.bonus_cfga(pvalues_list, is_null_list, ALPHA, eps, rng,
                        c=c, allocation='storey')


def bonus_pernode(pvalues_list, is_null_list, rng, c=1.0):
    """BONuS-GA with per-node calibration (Remark 6), 'global' small-node fallback."""
    return bp.bonus_cfga_pernode(pvalues_list, is_null_list, ALPHA, eps, rng,
                                 c=c, small_node='global')


def bonus_impl(pvalues_list, is_null_list, rng, c=1.0):
    """Baseline: implementable BONuS-GA (Storey allocation, global ratio)."""
    return s.bonus_cfga(pvalues_list, is_null_list, ALPHA, eps, rng,
                        c=c, allocation='storey')


def evaluate(method, name, NT=200):
    fdrs, pwrs = [], []
    for t in range(NT):
        rng = np.random.default_rng(424242 + 6661 * t)
        pv, isn = s.generate_pvalues(m_per_node, r1, mu_bounds, rng)
        n1 = sum(int(np.sum(~isn[i])) for i in range(N))
        V, R = method(pv, isn, rng)
        fdrs.append(V / max(R, 1))
        pwrs.append((R - V) / max(n1, 1))
    print(f'  {name:30s}  FDR={np.mean(fdrs):.3f}  pwr={np.mean(pwrs):.3f}')


print(f'Experiment 4 small-hub config: m_hub={M_RICH}, N_quiet={N_NULL}')
print(f'  alpha={ALPHA}, eps={eps:.4f}, NT=200 trials')
print()
print('Mechanism isolation at m_hub=250:')
evaluate(lambda p, i, r: bonus_impl(p, i, r),         '(1) Implementable [baseline]')
evaluate(lambda p, i, r: bonus_oracle_alloc(p, i, r), '(2) Oracle allocation')
evaluate(lambda p, i, r: bonus_bigbag(p, i, r),       '(3) Bigger bag (c=4)')
evaluate(lambda p, i, r: bonus_pernode(p, i, r),      '(4) Per-node calibration')
evaluate(lambda p, i, r: bp.bonus_cfga_pernode_budget(p, i, ALPHA, eps, r),
                                                      '(5) Per-node budgeted (Thm 5)')
print()
print('Reference (from Experiment 4 sweep, corrected code, NT=100):')
print('  CFGA+Storey+adaptive eps         FDR=0.100  pwr=0.783')

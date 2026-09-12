"""
Shared library for the simulations in "On Large-Scale Multiple Testing Over
Networks: A Non-Asymptotic Approach".

Methods (paper name -> here):
  - Pooled BH, Local BH(alpha), Local BH(alpha/N)   (baselines)
  - Original Greedy        -> greedy_aggregation        (Pournaderi & Xiang 2024)
  - CFGA (+Storey, +adaptive eps)
                           -> method_A_prime_* family   (Algorithm 1)
  - Inflated CFGA          -> augmented_cfga_* family   (Algorithm 2; the code
                              keeps the historical name "Augmented")
  - BONuS-GA               -> bonus_cfga / _make_bag    (Algorithm 3)

The monitoring-network experiments of the paper (Section VII) are driven by:
  sweep_quiet_nodes.py      Experiment 1    (sim_localbh_break.png)
  mn_sweep.py               Experiments 2-6 (sim_mn_{mu,rho,n,cauchy,alpha}.png)
  bonus_sweep.py            BONuS-GA overlay on the same seeds
  bonus_mechanism_test.py   Table I ablation (m_hub = 250)
  replot.py                 re-draw all figures from saved .npz

The __main__ block below runs an older 5-node configuration kept for
reference; it does NOT generate the figures in the paper. method_B / method_C
are legacy explorations and are not part of the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_pvalues(m_per_node, r1_per_node, mu_bounds_per_node, rng,
                     rho=0.0, distribution='gaussian', two_sided=False):
    """
    Generate p-values from per-node test statistics.

    distribution: 'gaussian' or 'cauchy'.
    two_sided: if True, p = 2 * (1 - F(|z|)); else p = 1 - F(z) (one-sided).
    rho: within-node AR(1) correlation, a scalar applied to EVERY node
         (hubs and quiet alike -- this is what the paper's Experiment 3
         reports) or a length-N sequence of per-node values
         (Gaussian-noise AR; preserves stationary
         N(0,1) marginal for Gaussian statistics).
    """
    from scipy.stats import cauchy as cauchy_dist
    n_nodes = len(m_per_node)
    pvalues = [None] * n_nodes
    is_null = [None] * n_nodes
    for i in range(n_nodes):
        m_i = int(m_per_node[i])
        r1_i = r1_per_node[i]
        null_mask = rng.random(m_i) > r1_i

        if distribution == 'gaussian':
            rho_i = float(rho[i]) if np.ndim(rho) > 0 else float(rho)
            if rho_i == 0.0:
                z = rng.standard_normal(m_i)
            else:
                eps = rng.standard_normal(m_i)
                z = np.empty(m_i)
                z[0] = eps[0]
                scale = np.sqrt(1.0 - rho_i * rho_i)
                for k in range(1, m_i):
                    z[k] = rho_i * z[k - 1] + scale * eps[k]
            null_cdf = norm
        elif distribution == 'cauchy':
            # Cauchy is heavy-tailed; AR is not applied (no nice stationary form).
            z = rng.standard_cauchy(m_i)
            null_cdf = cauchy_dist
        else:
            raise ValueError(f"unknown distribution {distribution!r}")

        lo, hi = mu_bounds_per_node[i]
        mu_samples = rng.uniform(lo, hi, size=m_i)
        z[~null_mask] += mu_samples[~null_mask]

        if two_sided:
            pvalues[i] = 2.0 * null_cdf.sf(np.abs(z))
            np.clip(pvalues[i], 0.0, 1.0, out=pvalues[i])
        else:
            pvalues[i] = null_cdf.sf(z)
        is_null[i] = null_mask
    return pvalues, is_null

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def bh(pvalues, is_null, alpha):
    """Standard BH on pooled p-values."""
    m = len(pvalues)
    sort_idx = np.argsort(pvalues)
    sorted_p = pvalues[sort_idx]
    sorted_null = is_null[sort_idx]
    thresholds = alpha * np.arange(1, m + 1) / m
    below = np.where(sorted_p <= thresholds)[0]
    if len(below) == 0:
        return 0, 0
    k_hat = below.max() + 1
    return int(np.sum(sorted_null[:k_hat])), int(k_hat)

def local_bh(pvalues_list, is_null_list, alpha, level_fn):
    """Decentralized BH: each node runs BH at level_fn(N) on its own p-values,
    union the rejections. level_fn(N)=alpha gives per-node (local) FDR control;
    level_fn(N)=alpha/N gives valid global FDR control via Bonferroni."""
    n_nodes = len(pvalues_list)
    node_level = level_fn(n_nodes)
    V_tot, R_tot = 0, 0
    for i in range(n_nodes):
        Vi, Ri = bh(pvalues_list[i], is_null_list[i], node_level)
        V_tot += Vi
        R_tot += Ri
    return V_tot, R_tot

def storey_r0(p, lam=0.5):
    return min((1 + np.sum(p > lam)) / ((1 - lam) * len(p)), 1.0)

# ---------------------------------------------------------------------------
# Helper: enumerate intervals at each node given grid spacing
# ---------------------------------------------------------------------------

def interval_counts(pvalues_list, L):
    """Return list of (i, j, N_ij, lo, hi)."""
    out = []
    for i, p in enumerate(pvalues_list):
        K_i = int(np.floor(1.0 / L[i]))
        # vectorized
        if K_i == 0:
            continue
        bins = np.linspace(0, K_i * L[i], K_i + 1)
        counts, _ = np.histogram(p, bins=bins)
        for j in range(K_i):
            out.append((i, j + 1, int(counts[j]), bins[j], bins[j + 1]))
    return out

# ---------------------------------------------------------------------------
# Original greedy aggregation
# ---------------------------------------------------------------------------

def greedy_aggregation(pvalues_list, alpha, epsilon, r0_fn=storey_r0):
    n_nodes = len(pvalues_list)
    m_per_node = np.array([len(p) for p in pvalues_list])
    m_total = m_per_node.sum()
    q_hat = m_per_node / m_total
    r0_hat = np.array([r0_fn(p) for p in pvalues_list])
    L_hat = epsilon / (q_hat * np.clip(r0_hat, 0.01, 1.0))

    intervals = interval_counts(pvalues_list, L_hat)
    # Sort by N descending (proxy for h_hat descending)
    intervals.sort(key=lambda x: -x[2])

    selected = []
    sum_h = 0.0
    for entry in intervals:
        i, j, N_ij, lo, hi = entry
        if N_ij == 0:
            break
        h_val = N_ij / (epsilon * m_total)
        new_sum = sum_h + h_val
        new_M = len(selected) + 1
        fdp_hat = new_M / new_sum
        if fdp_hat > alpha:
            break
        selected.append(entry)
        sum_h = new_sum
    return selected, L_hat

def evaluate_selected(pvalues_list, is_null_list, selected):
    V, R = 0, 0
    for (i, j, N_ij, lo, hi) in selected:
        in_int = (pvalues_list[i] > lo) & (pvalues_list[i] <= hi)
        R += int(np.sum(in_int))
        V += int(np.sum(in_int & is_null_list[i]))
    return V, R

# ---------------------------------------------------------------------------
# Method A: sample-split greedy aggregation
# ---------------------------------------------------------------------------

def method_A(pvalues_list, is_null_list, alpha, epsilon, rng):
    n_nodes = len(pvalues_list)
    # I.i.d. Bernoulli(1/2) fold assignment (Algorithm 1, Step 1).
    D1_pvalues, D2_pvalues = [None] * n_nodes, [None] * n_nodes
    D1_null, D2_null = [None] * n_nodes, [None] * n_nodes
    for i in range(n_nodes):
        m_i = len(pvalues_list[i])
        z = rng.random(m_i) < 0.5
        D1_pvalues[i] = pvalues_list[i][z]
        D2_pvalues[i] = pvalues_list[i][~z]
        D1_null[i] = is_null_list[i][z]
        D2_null[i] = is_null_list[i][~z]

    # Run greedy on D1, get full ranked list (no stopping)
    m1_total = sum(len(p) for p in D1_pvalues)
    q1 = np.array([len(p) / m1_total for p in D1_pvalues])
    r0_1 = np.array([storey_r0(p) for p in D1_pvalues])
    L1 = epsilon / (q1 * np.clip(r0_1, 0.01, 1.0))
    ranked = interval_counts(D1_pvalues, L1)
    ranked.sort(key=lambda x: -x[2])

    m2_per_node = np.array([len(p) for p in D2_pvalues])
    cum_R2 = 0
    cum_measure = np.zeros(n_nodes)
    best_M = 0
    cum_V2 = 0  # tracked for output

    M = 0
    rejections = []
    for entry in ranked:
        i, j, N_ij, lo, hi = entry
        M += 1
        in_int = (D2_pvalues[i] > lo) & (D2_pvalues[i] <= hi)
        cum_R2 += int(np.sum(in_int))
        cum_V2 += int(np.sum(in_int & D2_null[i]))
        cum_measure[i] += (hi - lo)
        rejections.append((cum_V2, cum_R2))
        bar_V = float(np.sum(m2_per_node * cum_measure))
        fdp_hat = (1 + bar_V) / max(cum_R2, 1)
        if fdp_hat <= alpha:
            best_M = M

    if best_M == 0:
        return 0, 0
    V, R = rejections[best_M - 1]
    return V, R

# ---------------------------------------------------------------------------
# Method B: mirror calibration
# ---------------------------------------------------------------------------

def method_B(pvalues_list, is_null_list, alpha, epsilon):
    """
    Interval-level mirror calibration (Selective SeqStep+ on interval pairs).
    Returns hypothesis-level (V, R) AND interval-level (V_int, R_int) where
    V_int counts rejected fully-null intervals.

    NOTE: only interval-level FDR is guaranteed by the proof. Hypothesis-level
    FDR may exceed alpha when rejected intervals contain a mix of nulls and
    non-nulls; this is documented in the manuscript.
    """
    n_nodes = len(pvalues_list)
    m_per_node = np.array([len(p) for p in pvalues_list])
    m_total = m_per_node.sum()
    q = m_per_node / m_total
    L = epsilon / q

    pairs = []
    for i in range(n_nodes):
        K_i = int(np.floor(1.0 / L[i]))
        for j in range(1, K_i // 2 + 1):
            lo_m, hi_m = (j - 1) * L[i], j * L[i]
            lo_mir, hi_mir = 1 - j * L[i], 1 - (j - 1) * L[i]
            in_m = (pvalues_list[i] > lo_m) & (pvalues_list[i] <= hi_m)
            in_mir = (pvalues_list[i] > lo_mir) & (pvalues_list[i] <= hi_mir)
            N_m = int(np.sum(in_m))
            N_mir = int(np.sum(in_mir))
            V_m = int(np.sum(in_m & is_null_list[i]))
            pairs.append((i, j, N_m, N_mir, V_m, lo_m, hi_m))

    if not pairs:
        return 0, 0, 0, 0

    # Order by |N - N_mirror| descending
    pairs.sort(key=lambda x: -abs(x[2] - x[3]))

    # Interval-level Selective SeqStep+: each interval is one test.
    best_idx = -1
    cum_pos = 0  # count of rejected interval-pairs
    cum_neg = 0  # count of anti-rejected interval-pairs
    for idx, (i, j, Nm, Nmir, Vm, lo, hi) in enumerate(pairs):
        if Nm > Nmir:
            cum_pos += 1
        elif Nm < Nmir:
            cum_neg += 1
        if cum_pos > 0:
            fdp_hat = (1 + cum_neg) / cum_pos
            if fdp_hat <= alpha:
                best_idx = idx

    if best_idx < 0:
        return 0, 0, 0, 0

    V, R = 0, 0          # hypothesis-level counts
    V_int, R_int = 0, 0  # interval-level counts (fully-null int. rejected)
    for k in range(best_idx + 1):
        i, j, Nm, Nmir, Vm, lo, hi = pairs[k]
        if Nm > Nmir:
            R += Nm
            V += Vm
            R_int += 1
            if Vm == Nm:  # fully-null interval rejected
                V_int += 1
    return V, R, V_int, R_int

# ---------------------------------------------------------------------------
# Method C: leave-one-out density e-values + e-BH
# ---------------------------------------------------------------------------

def method_C(pvalues_list, is_null_list, alpha, epsilon):
    n_nodes = len(pvalues_list)
    m_per_node = np.array([len(p) for p in pvalues_list])
    m_total = int(m_per_node.sum())
    q = m_per_node / m_total
    L = epsilon / q

    e_arr = np.empty(m_total)
    null_arr = np.empty(m_total, dtype=bool)
    cursor = 0
    for i in range(n_nodes):
        m_i = len(pvalues_list[i])
        K_i = int(np.floor(1.0 / L[i]))
        p_i = pvalues_list[i]
        # interval index of each p-value
        j_idx = np.ceil(p_i / L[i]).astype(int)
        valid = (j_idx >= 1) & (j_idx <= K_i)
        # count per interval
        bins = np.linspace(0, K_i * L[i], K_i + 1)
        counts, _ = np.histogram(p_i, bins=bins)
        Nj = np.zeros(m_i)
        Nj[valid] = counts[j_idx[valid] - 1]
        # leave-one-out e-value
        e_k = (Nj - 1) / max(m_i - 1, 1) / L[i]
        e_k[~valid] = 0.0
        e_arr[cursor:cursor + m_i] = e_k
        null_arr[cursor:cursor + m_i] = is_null_list[i]
        cursor += m_i

    # e-BH: order descending, find largest K with K * e_(K) >= m_total / alpha
    order = np.argsort(-e_arr)
    e_sorted = e_arr[order]
    null_sorted = null_arr[order]
    threshold = m_total / alpha
    K_candidates = np.where(np.arange(1, m_total + 1) * e_sorted >= threshold)[0]
    if len(K_candidates) == 0:
        return 0, 0
    K_hat = int(K_candidates.max() + 1)
    V = int(np.sum(null_sorted[:K_hat]))
    R = K_hat
    return V, R

# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def _greedy_full_rank(pvalues_subset, epsilon, lambda_=0.5):
    """Helper: run greedy on a p-value subset, return ranked intervals and L."""
    n_nodes = len(pvalues_subset)
    m_total_sub = sum(len(p) for p in pvalues_subset)
    if m_total_sub == 0:
        return [], np.ones(n_nodes)
    q = np.array([len(p) / m_total_sub for p in pvalues_subset])
    r0 = np.array([storey_r0(p, lambda_) if len(p) > 0 else 1.0
                   for p in pvalues_subset])
    # Guard q against empty nodes (possible under the Bernoulli split):
    # L -> huge -> K_i = 0 -> the node simply contributes no intervals.
    L = epsilon / np.maximum(q * np.clip(r0, 0.01, 1.0), 1e-12)
    ranked = interval_counts(pvalues_subset, L)
    ranked.sort(key=lambda x: -x[2])
    return ranked, L


def _ranked_eval_counts(pvalues_eval, null_eval, ranked):
    """Vectorized inner loop for the FDP-path evaluators.

    For each ranked interval (i, j, N, lo, hi), return, in ranked order:
        nodes  : node index of each interval
        r      : # eval p-values in (lo, hi] at that node
        v      : # eval NULL p-values in (lo, hi] at that node
        widths : (hi - lo)
    Counts use searchsorted with 'right' on per-node sorted arrays, i.e.
    #{lo < P <= hi}, identical (a.s., p-values continuous) to the original
    per-interval boolean scan but O(K + m) instead of O(K*m)."""
    K = len(ranked)
    if K == 0:
        z = np.zeros(0)
        return np.zeros(0, dtype=np.intp), z, z, z
    meta = np.array([(e[0], e[3], e[4]) for e in ranked], dtype=float)
    nodes = meta[:, 0].astype(np.intp)
    los = meta[:, 1]
    his = meta[:, 2]
    r = np.zeros(K)
    v = np.zeros(K)
    for i in np.unique(nodes):
        mask = nodes == i
        p_sorted = np.sort(pvalues_eval[i])
        pn_sorted = np.sort(pvalues_eval[i][null_eval[i]])
        lo_i = los[mask]
        hi_i = his[mask]
        r[mask] = (np.searchsorted(p_sorted, hi_i, side='right')
                   - np.searchsorted(p_sorted, lo_i, side='right'))
        v[mask] = (np.searchsorted(pn_sorted, hi_i, side='right')
                   - np.searchsorted(pn_sorted, lo_i, side='right'))
    return nodes, r, v, (his - los)


def _evaluate_fdp_path(pvalues_eval, null_eval, ranked, alpha, m_eval_per_node):
    """Walk `ranked` intervals (greedy order), evaluate the Method-A FDP
    estimator on the eval set, return (best_M, V_best, R_best). Vectorized."""
    nodes, r, v, widths = _ranked_eval_counts(pvalues_eval, null_eval, ranked)
    if nodes.size == 0:
        return 0, 0, 0
    cumR = np.cumsum(r)
    cumV = np.cumsum(v)
    bar_V = np.cumsum(np.asarray(m_eval_per_node, dtype=float)[nodes] * widths)
    fdp = (1.0 + bar_V) / np.maximum(cumR, 1.0)
    ok = np.nonzero(fdp <= alpha)[0]
    if ok.size == 0:
        return 0, 0, 0
    best = int(ok[-1])
    return best + 1, int(cumV[best]), int(cumR[best])


def method_A_with_epsilon(D1_pvalues, D1_null, D2_pvalues, D2_null, alpha, epsilon):
    """Single-split fold of CFGA (no Storey) with explicit splits and given epsilon."""
    ranked, _ = _greedy_full_rank(D1_pvalues, epsilon)
    m2_per_node = np.array([len(p) for p in D2_pvalues])
    _, V2, R2 = _evaluate_fdp_path(D2_pvalues, D2_null, ranked, alpha, m2_per_node)
    return V2, R2


def _evaluate_fdp_path_storey(pvalues_eval, null_eval, ranked, alpha, lambda_,
                              epsilon):
    """
    Storey-refined acceptance step (heuristic plug-in). Only intervals
    contained in [0, lambda_] are considered. By the Storey grid design
    L_i = epsilon/(qhat_i rhat0_i), every interval carries the same predictable
    null mass m_eval_total * epsilon, so the null-count numerator collapses to
        bar_V(M) = m_eval_total * epsilon * M,
    where M is the number of selected intervals. This is exactly the original
    greedy FDP estimator (1 + m eps M)/R of [PournaderiXiang2024], here scored
    cross-fit on the held-out half.
    """
    # Restrict to intervals in [0, lambda_]
    ranked_R = [r for r in ranked if r[4] <= lambda_]
    if not ranked_R:
        return 0, 0, 0
    nodes, r, v, widths = _ranked_eval_counts(pvalues_eval, null_eval, ranked_R)
    m_eval_total = sum(len(p) for p in pvalues_eval)
    cumR = np.cumsum(r)
    cumV = np.cumsum(v)
    M_idx = np.arange(1, len(ranked_R) + 1, dtype=float)
    bar_V = m_eval_total * epsilon * M_idx
    fdp = (1.0 + bar_V) / np.maximum(cumR, 1.0)
    ok = np.nonzero(fdp <= alpha)[0]
    if ok.size == 0:
        return 0, 0, 0
    best = int(ok[-1])
    return best + 1, int(cumV[best]), int(cumR[best])


def method_A_storey_with_epsilon(D1_pvalues, D1_null, D2_pvalues, D2_null,
                                 alpha, epsilon, lambda_=0.5):
    """Storey-refined single-split fold of CFGA."""
    ranked, _ = _greedy_full_rank(D1_pvalues, epsilon)
    _, V2, R2 = _evaluate_fdp_path_storey(D2_pvalues, D2_null, ranked, alpha,
                                          lambda_, epsilon)
    return V2, R2


def _evaluate_fdp_path_augmented(pvalues_eval, null_eval, ranked, alpha,
                                 m_eval_total, weights):
    """Lifted-space acceptance. Predictable null mass uses a single (q- or
    pi-)weighted region measure:
        bar_V(M) = m_eval_total * sum_i weights[i] * nu_i(M).
    `weights` is a length-n_nodes array (inflated qhat or pihat)."""
    nodes, r, v, widths = _ranked_eval_counts(pvalues_eval, null_eval, ranked)
    if nodes.size == 0:
        return 0, 0, 0
    cumR = np.cumsum(r)
    cumV = np.cumsum(v)
    bar_V = m_eval_total * np.cumsum(np.asarray(weights, dtype=float)[nodes] * widths)
    fdp = (1.0 + bar_V) / np.maximum(cumR, 1.0)
    ok = np.nonzero(fdp <= alpha)[0]
    if ok.size == 0:
        return 0, 0, 0
    best = int(ok[-1])
    return best + 1, int(cumV[best]), int(cumR[best])


def _dkw_margin(m1, n_nodes, eta):
    """Hoeffding-union inflation sqrt(log(2N/eta) / (2 m1)).

    This is the margin for an UNSCALED proportion estimate (e.g. qhat).
    For the Storey null-rate estimate pihat = X / (m1 (1-lambda)) of eq. (11),
    Theorem 2 requires the margin divided by (1-lambda) -- see eq. (12);
    callers using the Storey estimate must apply that factor themselves."""
    if m1 <= 0:
        return 1.0
    return float(np.sqrt(np.log(2.0 * n_nodes / eta) / (2.0 * m1)))


def augmented_fold(D1_p, D1_n, D2_p, D2_n, alpha, epsilon, eta):
    """One fold of Augmented CFGA: estimate q on the selection half, inflate,
    use the lifted-space null mass on the inference half."""
    n_nodes = len(D1_p)
    m1 = sum(len(p) for p in D1_p)
    m2 = sum(len(p) for p in D2_p)
    if m1 == 0 or m2 == 0:
        return 0, 0
    qhat1 = np.array([len(D1_p[i]) / m1 for i in range(n_nodes)])
    delta = _dkw_margin(m1, n_nodes, eta)
    weights = qhat1 + delta
    ranked, _ = _greedy_full_rank(D1_p, epsilon)
    _, V2, R2 = _evaluate_fdp_path_augmented(D2_p, D2_n, ranked, alpha, m2, weights)
    return V2, R2


def augmented_cfga(pvalues_list, is_null_list, alpha, epsilon, rng, eta=None):
    """Lifted-space CFGA (non-Storey): cross-fit q-estimation + DKW inflation."""
    if eta is None:
        m_total = sum(len(p) for p in pvalues_list)
        eta = 1.0 / max(m_total, 1)
    D1_p, D1_n, D2_p, D2_n = make_split(pvalues_list, is_null_list, rng)
    V1, R1 = augmented_fold(D1_p, D1_n, D2_p, D2_n, alpha / 2, epsilon, eta)
    V2, R2 = augmented_fold(D2_p, D2_n, D1_p, D1_n, alpha / 2, epsilon, eta)
    return V1 + V2, R1 + R2


def augmented_storey_fold(D1_p, D1_n, D2_p, D2_n, alpha, epsilon, eta,
                          lambda_=0.5):
    """One fold of Augmented CFGA+Storey: estimate the per-node NULL RATE
    pi^(i) = q^(i) r_0^(i) on the selection half via a Storey-type count,
    inflate, and use it as the lifted-space null mass on [0, lambda_]."""
    n_nodes = len(D1_p)
    m1 = sum(len(p) for p in D1_p)
    m2 = sum(len(p) for p in D2_p)
    if m1 == 0 or m2 == 0:
        return 0, 0
    pihat1 = np.array([
        float(np.sum(D1_p[i] > lambda_)) / (m1 * (1.0 - lambda_))
        for i in range(n_nodes)
    ])
    # Eq. (12): delta = (1-lambda)^{-1} sqrt(log(2N/eta) / (2 m1)).
    delta = _dkw_margin(m1, n_nodes, eta) / (1.0 - lambda_)
    weights = pihat1 + delta
    ranked, _ = _greedy_full_rank(D1_p, epsilon)
    ranked_R = [r for r in ranked if r[4] <= lambda_]   # restrict to [0, lambda_]
    _, V2, R2 = _evaluate_fdp_path_augmented(D2_p, D2_n, ranked_R, alpha, m2, weights)
    return V2, R2


def augmented_cfga_storey(pvalues_list, is_null_list, alpha, epsilon, rng,
                          eta=None, lambda_=0.5):
    """Lifted-space CFGA + cross-fit Storey null-rate estimation + inflation."""
    if eta is None:
        m_total = sum(len(p) for p in pvalues_list)
        eta = 1.0 / max(m_total, 1)
    D1_p, D1_n, D2_p, D2_n = make_split(pvalues_list, is_null_list, rng)
    V1, R1 = augmented_storey_fold(D1_p, D1_n, D2_p, D2_n, alpha/2, epsilon, eta, lambda_)
    V2, R2 = augmented_storey_fold(D2_p, D2_n, D1_p, D1_n, alpha/2, epsilon, eta, lambda_)
    return V1 + V2, R1 + R2


# ---------------------------------------------------------------------------
# BONuS-CFGA: counting-knockoffs with a bag of uniform synthetic nulls.
#   Avoids sample splitting -- ALL real p-values are used for both selection
#   and inference; the decoupling comes from masking/exchangeability instead.
#   Yang, Lei, Ho & Fithian (2021), instantiated for the exactly-uniform null.
# ---------------------------------------------------------------------------

def _make_bag(pvalues_list, rng, c=1.0, allocation='storey', lambda_=0.5):
    r"""Per-node bag of synthetic U[0,1] nulls.

    The TOTAL synthetic budget is fixed at c*m (m = total real count).
    Per-node sizes follow the chosen allocation rule:

      * 'proportional' : \tilde m^{(i)} \propto m^{(i)}   (simple heuristic).
      * 'storey'       : \tilde m^{(i)} \propto m^{(i)} * \hat r_0^{(i)}
                         (Implementable recipe of Section VI; plug-in to the
                         oracle weight w^{(i)} = pi^{(i)} of Theorem 3).

    Storey is used by default to match the manuscript's recommended
    implementable BONuS-CFGA. Nodes with \hat r_0^{(i)} = 0 (extremely strong
    rejection of the null in [\lambda,1]) get zero synthetic mass under
    'storey'; we floor by 1/m^{(i)} so every nonempty node still contributes
    at least a handful of synthetic points and the ratio N(A)+1)/Nt(A) does
    not blow up at sparse-tail nodes.
    """
    n = len(pvalues_list)
    m_i = np.array([len(pvalues_list[i]) for i in range(n)], dtype=float)
    if m_i.sum() == 0:
        return [rng.random(0) for _ in range(n)]
    if allocation == 'proportional':
        w = m_i
    elif allocation == 'storey':
        r0 = np.array(
            [storey_r0(pvalues_list[i], lam=lambda_) if m_i[i] > 0 else 0.0
             for i in range(n)],
            dtype=float,
        )
        # Floor each node's effective r0 so a Storey estimate of 0 (no p>lambda)
        # does not drop the node's synthetic mass to zero.
        floor = np.where(m_i > 0, 1.0 / np.maximum(m_i, 1.0), 0.0)
        r0 = np.maximum(r0, floor)
        w = m_i * r0
        if w.sum() == 0:
            w = m_i  # safety fallback
    else:
        raise ValueError(f"allocation must be 'storey' or 'proportional', got {allocation!r}")
    # Algorithm 3, Step 2: draw i.i.d. categorical labels for the synthetic
    # points, so the per-node sizes are multinomial(target_total, w/sum(w))
    # rather than deterministic -- this is what makes the bag exchangeable
    # with the real-null lift in Theorem 3.
    target_total = int(round(c * float(m_i.sum())))
    probs = w / max(w.sum(), 1e-12)
    sizes = rng.multinomial(target_total, probs)
    return [rng.random(int(sizes[i])) for i in range(n)]


def _bonus_accept(pvalues_list, is_null_list, synth, alpha, epsilon, lambda_=0.5):
    """Storey-BONuS acceptance for a GIVEN synthetic bag. Pool real+synthetic,
    rank intervals by pooled count (masking-measurable), restrict to [0, lambda_],
    and take the largest region with
        FDP_hat(M) = (N(A)+1)/Nt(A) * (Nt(Gamma_M)+1)/(1 v N(Gamma_M)) <= alpha,
    A=(lambda_,1]. Returns hypothesis-level (V, R) over the REAL data."""
    n_nodes = len(pvalues_list)
    if sum(len(p) for p in pvalues_list) == 0 or sum(len(s) for s in synth) == 0:
        return 0, 0
    synth_null = [np.ones(len(synth[i]), dtype=bool) for i in range(n_nodes)]
    pooled = [np.concatenate([pvalues_list[i], synth[i]]) for i in range(n_nodes)]
    ranked, _ = _greedy_full_rank(pooled, epsilon, lambda_)
    ranked_R = [r for r in ranked if r[4] <= lambda_]
    if not ranked_R:
        return 0, 0
    nodes, N_real, V_real, _ = _ranked_eval_counts(pvalues_list, is_null_list, ranked_R)
    _, N_syn, _, _ = _ranked_eval_counts(synth, synth_null, ranked_R)
    cumN = np.cumsum(N_real); cumV = np.cumsum(V_real); cumNt = np.cumsum(N_syn)
    NA = sum(int(np.sum(pvalues_list[i] > lambda_)) for i in range(n_nodes))
    NtA = sum(int(np.sum(synth[i] > lambda_)) for i in range(n_nodes))
    if NtA == 0:
        # Empty synthetic tail: the Storey-BONuS calibration ratio is
        # undefined (formally +inf), so the procedure makes no rejections.
        return 0, 0
    ratio = (NA + 1.0) / NtA
    fdp = ratio * (cumNt + 1.0) / np.maximum(cumN, 1.0)
    ok = np.nonzero(fdp <= alpha)[0]
    if ok.size == 0:
        return 0, 0
    best = int(ok[-1])
    return int(cumV[best]), int(cumN[best])


def bonus_cfga(pvalues_list, is_null_list, alpha, epsilon, rng,
               lambda_=0.5, c=1.0, allocation='storey'):
    r"""BONuS-style greedy aggregation for the exactly-uniform null (fixed eps).

    Each node draws a bag of synthetic null p-values U[0,1]. The TOTAL bag size
    is c*m; per-node sizes follow `allocation`:
      * 'storey'       : \tilde m^{(i)} ~ m^{(i)} \hat r_0^{(i)}  (Implementable
                         recipe of Section VI; default).
      * 'proportional' : \tilde m^{(i)} ~ m^{(i)}                (simpler
                         heuristic; kept for backward compatibility).

    Real and synthetic points are pooled; the greedy ranks intervals by the
    POOLED count (masking-measurable) confined to [0, lambda_], and the
    Storey-BONuS estimator calibrates the false-rejection count via the
    exchangeability of synthetic and real nulls. No sample splitting: all real
    p-values are used for selection AND inference. Randomized through the bag.
    """
    synth = _make_bag(pvalues_list, rng, c, allocation=allocation, lambda_=lambda_)
    return _bonus_accept(pvalues_list, is_null_list, synth, alpha, epsilon, lambda_)


# NOTE: the per-node BUDGETED variant of Theorem `thm:bonus-pernode`
# (Algorithm alg:bonus-budget in the paper) is implemented in
# bonus_pernode.bonus_cfga_pernode_budget.


def bonus_cfga_holdout_eps(pvalues_list, is_null_list, alpha, eps_grid, rng,
                           gamma=0.15, lambda_=0.5, c=1.0,
                           allocation='storey'):
    """Route 1 (adaptive eps): hold out a fraction gamma of the REAL data to pick
    eps by maximizing rejections, then run BONuS-CFGA on the remaining (1-gamma)
    with a fresh bag. eps_hat is independent of the main data, so the BONuS
    guarantee on the main run holds. Power is over the full non-null count, so
    the held-out slice is a genuine (small) cost. Rejects only in the main part.
    """
    n_nodes = len(pvalues_list)
    sel_p, sel_n, main_p, main_n = [], [], [], []
    for i in range(n_nodes):
        m_i = len(pvalues_list[i])
        idx = rng.permutation(m_i)
        k = int(round(gamma * m_i))
        sel_p.append(pvalues_list[i][idx[:k]]); sel_n.append(is_null_list[i][idx[:k]])
        main_p.append(pvalues_list[i][idx[k:]]); main_n.append(is_null_list[i][idx[k:]])
    # choose eps on the held-out slice (fixed bag, vary eps, maximize rejections)
    best_R, best_eps = -1, eps_grid[0]
    sel_bag = _make_bag(sel_p, rng, c, allocation=allocation, lambda_=lambda_)
    if sum(len(s) for s in sel_bag) > 0 and sum(len(p) for p in sel_p) > 0:
        for eps in eps_grid:
            _, R = _bonus_accept(sel_p, sel_n, sel_bag, alpha, eps, lambda_)
            if R > best_R:
                best_R, best_eps = R, eps
    return bonus_cfga(main_p, main_n, alpha, best_eps, rng, lambda_, c,
                      allocation=allocation)


def bonus_cfga_pooled_eps(pvalues_list, is_null_list, alpha, eps_grid, rng,
                          lambda_=0.5, c=1.0, allocation='storey'):
    """Route 2 (adaptive eps): pick eps from a MASKING-MEASURABLE score of the
    pooled (real+synthetic) p-values -- a standardized chi-square excess over
    uniform on [0, lambda_], which detects resolved signal without using the
    real/synthetic identities. eps_hat is a function of pooled VALUES only, so
    the BONuS guarantee is preserved; the same bag is reused for acceptance.
    Fully split-free.
    """
    n_nodes = len(pvalues_list)
    synth = _make_bag(pvalues_list, rng, c, allocation=allocation, lambda_=lambda_)
    if sum(len(p) for p in pvalues_list) == 0 or sum(len(s) for s in synth) == 0:
        return 0, 0
    pooled = [np.concatenate([pvalues_list[i], synth[i]]) for i in range(n_nodes)]
    m_pool = sum(len(p) for p in pooled)
    q = np.array([len(pooled[i]) / m_pool for i in range(n_nodes)])
    r0 = np.array([storey_r0(pooled[i]) if len(pooled[i]) > 0 else 1.0
                   for i in range(n_nodes)])
    best_score, best_eps = -np.inf, eps_grid[0]
    for eps in eps_grid:
        L = eps / (q * np.clip(r0, 0.01, 1.0))
        score = 0.0
        for i in range(n_nodes):
            Li = L[i]
            Ki = int(np.floor(lambda_ / Li)) if Li > 0 else 0
            if Ki < 2:
                continue
            bins = np.linspace(0.0, Ki * Li, Ki + 1)
            cj, _ = np.histogram(pooled[i], bins=bins)
            n_in = int(cj.sum())
            if n_in == 0:
                continue
            e = n_in / Ki                       # expected per bin under uniform
            chi2 = float(np.sum((cj - e) ** 2) / e)
            score += (chi2 - (Ki - 1)) / np.sqrt(2.0 * (Ki - 1))
        if score > best_score:
            best_score, best_eps = score, eps
    return _bonus_accept(pvalues_list, is_null_list, synth, alpha, best_eps, lambda_)


def make_split(pvalues_list, is_null_list, rng):
    """I.i.d. Bernoulli(1/2) fold split (Algorithm 1, Step 1).

    Each hypothesis is assigned to fold 1 or 2 by an independent fair coin,
    drawn independently of the data. This i.i.d. assignment is what Lemma 1
    requires: conditional on one fold, the other fold is i.i.d. from the
    model. (An exact per-node half-split, used in an earlier draft, makes
    the held-out per-node counts essentially deterministic given the ranking
    half and does not satisfy the lemma's conditions.)
    Returns (D1_pvalues, D1_null, D2_pvalues, D2_null)."""
    n_nodes = len(pvalues_list)
    D1_pvalues, D2_pvalues = [None] * n_nodes, [None] * n_nodes
    D1_null, D2_null = [None] * n_nodes, [None] * n_nodes
    for i in range(n_nodes):
        m_i = len(pvalues_list[i])
        z = rng.random(m_i) < 0.5
        D1_pvalues[i] = pvalues_list[i][z]
        D2_pvalues[i] = pvalues_list[i][~z]
        D1_null[i] = is_null_list[i][z]
        D2_null[i] = is_null_list[i][~z]
    return D1_pvalues, D1_null, D2_pvalues, D2_null


def adaptive_epsilon_choice(D1_pvalues, D1_null, alpha, eps_grid, rng):
    """
    Sub-split D_1 into D_1a, D_1b. For each candidate eps in eps_grid:
        - run greedy on D_1a
        - apply Method-A acceptance on D_1b at level alpha
        - record the resulting rejection count R_1b
    Return eps_hat = argmax R_1b.
    """
    n_nodes = len(D1_pvalues)
    D1a_p, D1a_n, D1b_p, D1b_n = make_split(D1_pvalues, D1_null, rng)
    m1b_per_node = np.array([len(p) for p in D1b_p])
    best_R = -1
    best_eps = eps_grid[0]
    for eps in eps_grid:
        ranked, _ = _greedy_full_rank(D1a_p, eps)
        _, _, R_eval = _evaluate_fdp_path(D1b_p, D1b_n, ranked, alpha, m1b_per_node)
        if R_eval > best_R:
            best_R = R_eval
            best_eps = eps
    return best_eps


def method_A_adaptive(pvalues_list, is_null_list, alpha, eps_grid, rng):
    """Method A with adaptive epsilon chosen by sub-split CV on D_1."""
    D1_p, D1_n, D2_p, D2_n = make_split(pvalues_list, is_null_list, rng)
    eps_hat = adaptive_epsilon_choice(D1_p, D1_n, alpha, eps_grid, rng)
    return method_A_with_epsilon(D1_p, D1_n, D2_p, D2_n, alpha, eps_hat)


def method_A_prime_with_epsilon(pvalues_list, is_null_list, alpha, epsilon, rng):
    """CFGA: cross-fit greedy aggregation at fixed epsilon, no Storey."""
    D1_p, D1_n, D2_p, D2_n = make_split(pvalues_list, is_null_list, rng)
    V1, R1 = method_A_with_epsilon(D1_p, D1_n, D2_p, D2_n, alpha / 2, epsilon)
    V2, R2 = method_A_with_epsilon(D2_p, D2_n, D1_p, D1_n, alpha / 2, epsilon)
    return V1 + V2, R1 + R2


def method_A_prime_storey(pvalues_list, is_null_list, alpha, epsilon, rng,
                          lambda_=0.5):
    """CFGA + Storey refinement."""
    D1_p, D1_n, D2_p, D2_n = make_split(pvalues_list, is_null_list, rng)
    V1, R1 = method_A_storey_with_epsilon(D1_p, D1_n, D2_p, D2_n,
                                          alpha / 2, epsilon, lambda_)
    V2, R2 = method_A_storey_with_epsilon(D2_p, D2_n, D1_p, D1_n,
                                          alpha / 2, epsilon, lambda_)
    return V1 + V2, R1 + R2


def adaptive_epsilon_choice_storey(D1_pvalues, D1_null, alpha, eps_grid, rng,
                                   lambda_=0.5):
    """Storey-refined sub-split CV to pick epsilon on D_1."""
    D1a_p, D1a_n, D1b_p, D1b_n = make_split(D1_pvalues, D1_null, rng)
    best_R = -1
    best_eps = eps_grid[0]
    for eps in eps_grid:
        ranked, _ = _greedy_full_rank(D1a_p, eps)
        _, _, R_eval = _evaluate_fdp_path_storey(D1b_p, D1b_n, ranked, alpha,
                                                 lambda_, eps)
        if R_eval > best_R:
            best_R = R_eval
            best_eps = eps
    return best_eps


def method_A_prime_storey_adaptive(pvalues_list, is_null_list, alpha,
                                   eps_grid, rng, lambda_=0.5):
    """CFGA + Storey + adaptive epsilon."""
    D1_p, D1_n, D2_p, D2_n = make_split(pvalues_list, is_null_list, rng)
    eps1 = adaptive_epsilon_choice_storey(D1_p, D1_n, alpha / 2, eps_grid, rng, lambda_)
    V1, R1 = method_A_storey_with_epsilon(D1_p, D1_n, D2_p, D2_n,
                                          alpha / 2, eps1, lambda_)
    eps2 = adaptive_epsilon_choice_storey(D2_p, D2_n, alpha / 2, eps_grid, rng, lambda_)
    V2, R2 = method_A_storey_with_epsilon(D2_p, D2_n, D1_p, D1_n,
                                          alpha / 2, eps2, lambda_)
    return V1 + V2, R1 + R2


def method_A_prime_adaptive(pvalues_list, is_null_list, alpha, eps_grid, rng):
    """CFGA + adaptive epsilon (no Storey). Kept for completeness."""
    D1_p, D1_n, D2_p, D2_n = make_split(pvalues_list, is_null_list, rng)
    eps1 = adaptive_epsilon_choice(D1_p, D1_n, alpha / 2, eps_grid, rng)
    V1, R1 = method_A_with_epsilon(D1_p, D1_n, D2_p, D2_n, alpha / 2, eps1)
    eps2 = adaptive_epsilon_choice(D2_p, D2_n, alpha / 2, eps_grid, rng)
    V2, R2 = method_A_with_epsilon(D2_p, D2_n, D1_p, D1_n, alpha / 2, eps2)
    return V1 + V2, R1 + R2


def adaptive_epsilon_choice_augmented(D1_p, D1_n, alpha, eps_grid, rng, eta,
                                      storey=False, lambda_=0.5):
    """Pick epsilon for the augmented (lifted-space) fold by sub-split CV on D_1.
    Sub-split D_1 -> (D_1a selection, D_1b validation). For each candidate eps,
    estimate the lifted-space weights (qhat or pihat) + DKW margin on D_1a, run
    greedy on D_1a, evaluate the augmented acceptance on D_1b, keep the eps with
    the most validation rejections. eps is sigma(D_1)-measurable, so the fold's
    FDR guarantee is preserved."""
    D1a_p, D1a_n, D1b_p, D1b_n = make_split(D1_p, D1_n, rng)
    n_nodes = len(D1a_p)
    m1a = sum(len(p) for p in D1a_p)
    m1b = sum(len(p) for p in D1b_p)
    if m1a == 0 or m1b == 0:
        return eps_grid[0]
    delta = _dkw_margin(m1a, n_nodes, eta)
    if storey:
        # Storey rate estimate -> margin needs the 1/(1-lambda) factor (eq. 12).
        w = np.array([float(np.sum(D1a_p[i] > lambda_)) / (m1a * (1.0 - lambda_))
                      for i in range(n_nodes)]) + delta / (1.0 - lambda_)
    else:
        w = np.array([len(D1a_p[i]) / m1a for i in range(n_nodes)]) + delta
    best_R = -1
    best_eps = eps_grid[0]
    for eps in eps_grid:
        ranked, _ = _greedy_full_rank(D1a_p, eps)
        if storey:
            ranked = [r for r in ranked if r[4] <= lambda_]
        _, _, R_eval = _evaluate_fdp_path_augmented(
            D1b_p, D1b_n, ranked, alpha, m1b, w)
        if R_eval > best_R:
            best_R = R_eval
            best_eps = eps
    return best_eps


def augmented_cfga_adaptive(pvalues_list, is_null_list, alpha, eps_grid, rng,
                            eta=None):
    """Augmented CFGA (non-Storey) with adaptive epsilon chosen per fold."""
    if eta is None:
        m_total = sum(len(p) for p in pvalues_list)
        eta = 1.0 / max(m_total, 1)
    D1_p, D1_n, D2_p, D2_n = make_split(pvalues_list, is_null_list, rng)
    e1 = adaptive_epsilon_choice_augmented(D1_p, D1_n, alpha / 2, eps_grid, rng, eta)
    V1, R1 = augmented_fold(D1_p, D1_n, D2_p, D2_n, alpha / 2, e1, eta)
    e2 = adaptive_epsilon_choice_augmented(D2_p, D2_n, alpha / 2, eps_grid, rng, eta)
    V2, R2 = augmented_fold(D2_p, D2_n, D1_p, D1_n, alpha / 2, e2, eta)
    return V1 + V2, R1 + R2


def augmented_cfga_storey_adaptive(pvalues_list, is_null_list, alpha, eps_grid,
                                   rng, eta=None, lambda_=0.5):
    """Augmented CFGA + Storey with adaptive epsilon chosen per fold."""
    if eta is None:
        m_total = sum(len(p) for p in pvalues_list)
        eta = 1.0 / max(m_total, 1)
    D1_p, D1_n, D2_p, D2_n = make_split(pvalues_list, is_null_list, rng)
    e1 = adaptive_epsilon_choice_augmented(D1_p, D1_n, alpha / 2, eps_grid, rng,
                                           eta, storey=True, lambda_=lambda_)
    V1, R1 = augmented_storey_fold(D1_p, D1_n, D2_p, D2_n, alpha / 2, e1, eta, lambda_)
    e2 = adaptive_epsilon_choice_augmented(D2_p, D2_n, alpha / 2, eps_grid, rng,
                                           eta, storey=True, lambda_=lambda_)
    V2, R2 = augmented_storey_fold(D2_p, D2_n, D1_p, D1_n, alpha / 2, e2, eta, lambda_)
    return V1 + V2, R1 + R2


def method_A_prime(pvalues_list, is_null_list, alpha, epsilon, rng):
    """Legacy alias. The original implementation drew a fresh split for its
    second run and double-counted overlapping rejections (it could report
    more rejections than hypotheses). It now delegates to
    method_A_prime_with_epsilon, the correct swap-the-folds implementation
    used by all paper drivers."""
    return method_A_prime_with_epsilon(pvalues_list, is_null_list, alpha,
                                       epsilon, rng)

def _eps_grid_for(m_total, alpha):
    """Grid of candidate epsilons spanning c * alpha * m^{-1/2} for c in
    {1/16, 1/8, 1/4, 1/2, 1, 2, 4}. Wider grid gives the adaptive procedure
    more room when the default c=1 is far from optimal (small-m regime)."""
    return [
        0.0625 * alpha * m_total ** (-1 / 2),
        0.125 * alpha * m_total ** (-1 / 2),
        0.25 * alpha * m_total ** (-1 / 2),
        0.5  * alpha * m_total ** (-1 / 2),
        1.0  * alpha * m_total ** (-1 / 2),
        2.0  * alpha * m_total ** (-1 / 2),
        4.0  * alpha * m_total ** (-1 / 2),
    ]


def run_trial(m_per_node, r1, mu_bounds, alpha, epsilon, rng,
              rho=0.0, distribution='gaussian', two_sided=False):
    pvalues, is_null = generate_pvalues(
        m_per_node, r1, mu_bounds, rng,
        rho=rho, distribution=distribution, two_sided=two_sided)
    n1_total = sum(int(np.sum(~is_null[i])) for i in range(len(pvalues)))

    def stats(V, R):
        fdp = V / max(R, 1)
        tdp = (R - V) / max(n1_total, 1)
        return fdp, tdp

    out = {}
    pooled_p = np.concatenate(pvalues)
    pooled_n = np.concatenate(is_null)
    out['Pooled BH'] = stats(*bh(pooled_p, pooled_n, alpha))

    selected, L = greedy_aggregation(pvalues, alpha, epsilon)
    out['Original Greedy'] = stats(*evaluate_selected(pvalues, is_null, selected))
    out['Local BH (alpha)'] = stats(*local_bh(pvalues, is_null, alpha, lambda N: alpha))
    out['Local BH (alpha/N)'] = stats(*local_bh(pvalues, is_null, alpha, lambda N: alpha / N))

    out['CFGA'] = stats(*method_A_prime_with_epsilon(
        pvalues, is_null, alpha, epsilon, rng))
    out['Augmented CFGA'] = stats(*augmented_cfga(pvalues, is_null, alpha, epsilon, rng))
    out['CFGA + Storey'] = stats(*method_A_prime_storey(
        pvalues, is_null, alpha, epsilon, rng))
    out['Augmented CFGA + Storey'] = stats(*augmented_cfga_storey(pvalues, is_null, alpha, epsilon, rng))
    m_total = sum(m_per_node)
    eps_grid = _eps_grid_for(m_total, alpha)
    out['CFGA + Storey + adaptive eps'] = stats(*method_A_prime_storey_adaptive(
        pvalues, is_null, alpha, eps_grid, rng))

    return out

def _aggregate(trials_dict):
    """Convert per-trial arrays into mean and 2*SE for each method."""
    means = {}
    bands = {}
    for k, arr in trials_dict.items():
        a = np.asarray(arr)
        means[k] = float(np.mean(a))
        bands[k] = 2.0 * float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return means, bands


def simulation_vary_n(n_values, n_trials=200, alpha=0.2, seed_base=12345):
    n_nodes = 5
    r1 = np.array([0.5 - 0.1 * i for i in range(n_nodes)])
    mu_base = np.array([1.25 * (i + 1) for i in range(n_nodes)])
    mu_bounds = [(mu_base[i] - 0.5, mu_base[i] + 0.5) for i in range(n_nodes)]

    fdr = {m: [] for m in METHODS}; pwr = {m: [] for m in METHODS}
    fdr_se = {m: [] for m in METHODS}; pwr_se = {m: [] for m in METHODS}

    for n in n_values:
        print(f"  n = {n}")
        m_per_node = [int((1 - 0.2 * i) * n) for i in range(n_nodes)]
        m_total = sum(m_per_node)
        epsilon = alpha / np.sqrt(max(m_total, 100))
        fdr_trial = {m: [] for m in METHODS}; pwr_trial = {m: [] for m in METHODS}
        for trial in range(n_trials):
            rng = np.random.default_rng(seed_base + trial * 9973 + n)
            stats = run_trial(m_per_node, r1, mu_bounds, alpha, epsilon, rng)
            for m_name, (fdp, tdp) in stats.items():
                fdr_trial[m_name].append(fdp)
                pwr_trial[m_name].append(tdp)
        mF, sF = _aggregate(fdr_trial); mP, sP = _aggregate(pwr_trial)
        for m_name in METHODS:
            fdr[m_name].append(mF[m_name]); fdr_se[m_name].append(sF[m_name])
            pwr[m_name].append(mP[m_name]); pwr_se[m_name].append(sP[m_name])
    return fdr, pwr, fdr_se, pwr_se


def simulation_vary_mu(mu_values, n_per_node_base=1000, n_trials=200, alpha=0.2, seed_base=98765):
    n_nodes = 5
    r1 = np.array([0.5 - 0.1 * i for i in range(n_nodes)])
    m_per_node = [int((1 - 0.2 * i) * n_per_node_base) for i in range(n_nodes)]
    m_total = sum(m_per_node)
    epsilon = alpha / np.sqrt(m_total)

    fdr = {m: [] for m in METHODS}; pwr = {m: [] for m in METHODS}
    fdr_se = {m: [] for m in METHODS}; pwr_se = {m: [] for m in METHODS}

    for mu in mu_values:
        print(f"  mu = {mu}")
        mu_bounds = [(mu * (i + 1) - 0.5, mu * (i + 1) + 0.5) for i in range(n_nodes)]
        fdr_trial = {m: [] for m in METHODS}; pwr_trial = {m: [] for m in METHODS}
        for trial in range(n_trials):
            rng = np.random.default_rng(seed_base + trial * 7919 + int(mu * 1000))
            stats = run_trial(m_per_node, r1, mu_bounds, alpha, epsilon, rng)
            for m_name, (fdp, tdp) in stats.items():
                fdr_trial[m_name].append(fdp)
                pwr_trial[m_name].append(tdp)
        mF, sF = _aggregate(fdr_trial); mP, sP = _aggregate(pwr_trial)
        for m_name in METHODS:
            fdr[m_name].append(mF[m_name]); fdr_se[m_name].append(sF[m_name])
            pwr[m_name].append(mP[m_name]); pwr_se[m_name].append(sP[m_name])
    return fdr, pwr, fdr_se, pwr_se


def simulation_cauchy(n_values, n_trials=80, alpha=0.2, seed_base=77777):
    """Experiment 5: Cauchy (heavy-tailed) test statistics, vary n.

    Cauchy p-values are less concentrated than Gaussian under the same
    location shift, so we use the larger shifts mu in {2, 4, 6, 8, 10}
    matching the original paper's Experiment 2(b).
    """
    n_nodes = 5
    r1 = np.array([0.5 - 0.1 * i for i in range(n_nodes)])
    mu_base = np.array([6.0 + 2.0 * i for i in range(n_nodes)])  # 6, 8, 10, 12, 14
    mu_bounds = [(mu_base[i] - 0.5, mu_base[i] + 0.5) for i in range(n_nodes)]

    fdr = {m: [] for m in METHODS}; pwr = {m: [] for m in METHODS}
    fdr_se = {m: [] for m in METHODS}; pwr_se = {m: [] for m in METHODS}

    for n in n_values:
        print(f"  n = {n}")
        m_per_node = [int((1 - 0.2 * i) * n) for i in range(n_nodes)]
        m_total = sum(m_per_node)
        epsilon = alpha / np.sqrt(max(m_total, 100))
        fdr_trial = {m: [] for m in METHODS}; pwr_trial = {m: [] for m in METHODS}
        for trial in range(n_trials):
            rng = np.random.default_rng(seed_base + trial * 2417 + n)
            stats = run_trial(m_per_node, r1, mu_bounds, alpha, epsilon, rng,
                              distribution='cauchy')
            for m_name, (fdp, tdp) in stats.items():
                fdr_trial[m_name].append(fdp)
                pwr_trial[m_name].append(tdp)
        mF, sF = _aggregate(fdr_trial); mP, sP = _aggregate(pwr_trial)
        for m_name in METHODS:
            fdr[m_name].append(mF[m_name]); fdr_se[m_name].append(sF[m_name])
            pwr[m_name].append(mP[m_name]); pwr_se[m_name].append(sP[m_name])
    return fdr, pwr, fdr_se, pwr_se


def simulation_vary_N(N_values, m_total_target=10000, n_trials=80, alpha=0.2,
                      seed_base=55555):
    """
    Experiment 6: vary network size N at approximately fixed total m.
    Each node gets m_total_target / N p-values; r_1 spans 0.5 to 0.1; mu_base
    spans 1.25 to 6.25 linearly across nodes.
    """
    fdr = {m: [] for m in METHODS}; pwr = {m: [] for m in METHODS}
    fdr_se = {m: [] for m in METHODS}; pwr_se = {m: [] for m in METHODS}

    for N in N_values:
        print(f"  N = {N}")
        n_per_node = max(int(m_total_target / N), 10)
        m_per_node = [n_per_node] * N
        m_total = sum(m_per_node)
        # Heterogeneity: r_1 linearly from 0.5 down to 0.1
        if N == 1:
            r1 = np.array([0.3])
            mu_base = np.array([3.75])
        else:
            r1 = np.array([0.5 - 0.4 * i / (N - 1) for i in range(N)])
            mu_base = np.array([1.25 + 5.0 * i / (N - 1) for i in range(N)])
        mu_bounds = [(mu_base[i] - 0.5, mu_base[i] + 0.5) for i in range(N)]
        epsilon = alpha / np.sqrt(max(m_total, 100))
        fdr_trial = {m: [] for m in METHODS}; pwr_trial = {m: [] for m in METHODS}
        for trial in range(n_trials):
            rng = np.random.default_rng(seed_base + trial * 3733 + N)
            stats = run_trial(m_per_node, r1, mu_bounds, alpha, epsilon, rng)
            for m_name, (fdp, tdp) in stats.items():
                fdr_trial[m_name].append(fdp)
                pwr_trial[m_name].append(tdp)
        mF, sF = _aggregate(fdr_trial); mP, sP = _aggregate(pwr_trial)
        for m_name in METHODS:
            fdr[m_name].append(mF[m_name]); fdr_se[m_name].append(sF[m_name])
            pwr[m_name].append(mP[m_name]); pwr_se[m_name].append(sP[m_name])
    return fdr, pwr, fdr_se, pwr_se


def simulation_vary_alpha(alpha_values, n=2000, n_trials=80, seed_base=11111):
    """Experiment 7: vary target FDR alpha at fixed n."""
    n_nodes = 5
    r1 = np.array([0.5 - 0.1 * i for i in range(n_nodes)])
    mu_base = np.array([1.25 * (i + 1) for i in range(n_nodes)])
    mu_bounds = [(mu_base[i] - 0.5, mu_base[i] + 0.5) for i in range(n_nodes)]
    m_per_node = [int((1 - 0.2 * i) * n) for i in range(n_nodes)]
    m_total = sum(m_per_node)

    fdr = {m: [] for m in METHODS}; pwr = {m: [] for m in METHODS}
    fdr_se = {m: [] for m in METHODS}; pwr_se = {m: [] for m in METHODS}

    for alpha in alpha_values:
        print(f"  alpha = {alpha}")
        epsilon = alpha / np.sqrt(max(m_total, 100))
        fdr_trial = {m: [] for m in METHODS}; pwr_trial = {m: [] for m in METHODS}
        for trial in range(n_trials):
            rng = np.random.default_rng(seed_base + trial * 1607 + int(alpha * 10000))
            stats = run_trial(m_per_node, r1, mu_bounds, alpha, epsilon, rng)
            for m_name, (fdp, tdp) in stats.items():
                fdr_trial[m_name].append(fdp)
                pwr_trial[m_name].append(tdp)
        mF, sF = _aggregate(fdr_trial); mP, sP = _aggregate(pwr_trial)
        for m_name in METHODS:
            fdr[m_name].append(mF[m_name]); fdr_se[m_name].append(sF[m_name])
            pwr[m_name].append(mP[m_name]); pwr_se[m_name].append(sP[m_name])
    return fdr, pwr, fdr_se, pwr_se


def simulation_vary_rho(rho_values, n=1000, n_trials=80, alpha=0.2, seed_base=33333):
    """Experiment 4: vary within-node AR(1) correlation at fixed n."""
    n_nodes = 5
    r1 = np.array([0.5 - 0.1 * i for i in range(n_nodes)])
    mu_base = np.array([1.25 * (i + 1) for i in range(n_nodes)])
    mu_bounds = [(mu_base[i] - 0.5, mu_base[i] + 0.5) for i in range(n_nodes)]
    m_per_node = [int((1 - 0.2 * i) * n) for i in range(n_nodes)]
    m_total = sum(m_per_node)
    epsilon = alpha / np.sqrt(max(m_total, 100))

    fdr = {m: [] for m in METHODS}; pwr = {m: [] for m in METHODS}
    fdr_se = {m: [] for m in METHODS}; pwr_se = {m: [] for m in METHODS}

    for rho in rho_values:
        print(f"  rho = {rho}")
        fdr_trial = {m: [] for m in METHODS}; pwr_trial = {m: [] for m in METHODS}
        for trial in range(n_trials):
            rng = np.random.default_rng(seed_base + trial * 4517 + int(rho * 1000))
            stats = run_trial(m_per_node, r1, mu_bounds, alpha, epsilon, rng, rho=rho)
            for m_name, (fdp, tdp) in stats.items():
                fdr_trial[m_name].append(fdp)
                pwr_trial[m_name].append(tdp)
        mF, sF = _aggregate(fdr_trial); mP, sP = _aggregate(pwr_trial)
        for m_name in METHODS:
            fdr[m_name].append(mF[m_name]); fdr_se[m_name].append(sF[m_name])
            pwr[m_name].append(mP[m_name]); pwr_se[m_name].append(sP[m_name])
    return fdr, pwr, fdr_se, pwr_se

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STYLE = {
    'Pooled BH':                       {'color': 'tab:blue',   'marker': 'o', 'linestyle': '-'},
    'Local BH (alpha)':                {'color': 'tab:brown',  'marker': 'v', 'linestyle': ':'},
    'Local BH (alpha/N)':              {'color': 'tab:pink',   'marker': '<', 'linestyle': ':'},
    'Original Greedy':                 {'color': 'tab:orange', 'marker': 's', 'linestyle': '-'},
    'CFGA':                            {'color': 'tab:red',    'marker': 'D', 'linestyle': '--'},
    'Augmented CFGA':                  {'color': 'tab:purple', 'marker': 'P', 'linestyle': '-.'},
    'CFGA + Storey':                   {'color': 'tab:green',  'marker': '^', 'linestyle': '--'},
    'Augmented CFGA + Storey':         {'color': 'tab:olive',  'marker': '>', 'linestyle': '-.'},
    'CFGA + Storey + adaptive eps':    {'color': 'tab:cyan',   'marker': 'X', 'linestyle': '-'},
}

def _plot_with_bands(ax, x, mean, se, color, marker, linestyle, label):
    # SE bands removed per user request; argument kept for backward compatibility.
    x_arr = np.asarray(x)
    m = np.asarray(mean)
    ax.plot(x_arr, m, color=color, marker=marker, linestyle=linestyle,
            label=label, markersize=6)


def plot_results(x_values, fdr, pwr, xlabel, title_suffix, alpha, out_path,
                 xlog=False, fdr_se=None, pwr_se=None, methods=None, style=None):
    if methods is None: methods = METHODS
    if style is None:   style = STYLE
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for m_name in methods:
        s = style[m_name]
        fse = fdr_se[m_name] if fdr_se is not None else [0] * len(x_values)
        pse = pwr_se[m_name] if pwr_se is not None else [0] * len(x_values)
        _plot_with_bands(axes[0], x_values, fdr[m_name], fse,
                         s['color'], s['marker'], s['linestyle'], m_name)
        _plot_with_bands(axes[1], x_values, pwr[m_name], pse,
                         s['color'], s['marker'], s['linestyle'], m_name)
    axes[0].axhline(alpha, color='black', linestyle=':', label=f'Target FDR = {alpha}')
    if xlog:
        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
    axes[0].set_xlabel(xlabel)
    axes[1].set_xlabel(xlabel)
    axes[0].set_ylabel('FDR')
    axes[1].set_ylabel('Power')
    axes[0].set_title(f'FDR {title_suffix}')
    axes[1].set_title(f'Power {title_suffix}')
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    axes[0].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[1].legend(loc='best', fontsize=8, framealpha=0.9)
    axes[0].set_ylim(0, max(0.35, alpha * 1.5))
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EPS_METHODS = [
    "A' (eps=alpha m^{-1/2}, baseline)",
    "A' (eps=alpha m^{-1/3}, theory)",
    "A' (oracle-best in grid)",
    "A' (adaptive eps)",
]

EPS_STYLE = {
    "A' (eps=alpha m^{-1/2}, baseline)": {'color': 'tab:red',    'marker': 'D', 'linestyle': '--'},
    "A' (eps=alpha m^{-1/3}, theory)":   {'color': 'tab:olive',  'marker': 'P', 'linestyle': ':'},
    "A' (oracle-best in grid)":          {'color': 'tab:gray',   'marker': '*', 'linestyle': '-.'},
    "A' (adaptive eps)":                 {'color': 'tab:cyan',   'marker': 'X', 'linestyle': '-'},
}


def make_eps_grid(m_total, alpha):
    """Logarithmic grid of candidate epsilons spanning useful range."""
    return [
        0.25 * alpha * m_total ** (-1 / 2),  # 1/4 of baseline
        0.5  * alpha * m_total ** (-1 / 2),  # 1/2 of baseline
        1.0  * alpha * m_total ** (-1 / 2),  # baseline
        2.0  * alpha * m_total ** (-1 / 2),  # 2x baseline
        0.25 * alpha * m_total ** (-1 / 3),  # smaller of m^{-1/3}
    ]


def run_eps_trial(m_per_node, r1, mu_bounds, alpha, rng):
    """Compare epsilon choices for Method A'."""
    pvalues, is_null = generate_pvalues(m_per_node, r1, mu_bounds, rng)
    n1_total = sum(int(np.sum(~is_null[i])) for i in range(len(pvalues)))
    m_total = sum(m_per_node)

    def stats(V, R):
        fdp = V / max(R, 1)
        tdp = (R - V) / max(n1_total, 1)
        return fdp, tdp

    out = {}
    eps_baseline = alpha * m_total ** (-1 / 2)
    eps_theory = alpha * m_total ** (-1 / 3)
    eps_grid = make_eps_grid(m_total, alpha)

    out["A' (eps=alpha m^{-1/2}, baseline)"] = stats(*method_A_prime_with_epsilon(
        pvalues, is_null, alpha, eps_baseline, rng))
    out["A' (eps=alpha m^{-1/3}, theory)"] = stats(*method_A_prime_with_epsilon(
        pvalues, is_null, alpha, eps_theory, rng))

    # Oracle-best: try every eps in the grid, take the one with most rejections
    # (this is an upper bound on adaptive performance, since it uses the true labels).
    best_V, best_R = 0, 0
    best_rejections = -1
    for eps in eps_grid:
        V, R = method_A_prime_with_epsilon(pvalues, is_null, alpha, eps, rng)
        if R - V > best_rejections:  # true power, oracle
            best_rejections = R - V
            best_V, best_R = V, R
    out["A' (oracle-best in grid)"] = stats(best_V, best_R)

    out["A' (adaptive eps)"] = stats(*method_A_prime_adaptive(
        pvalues, is_null, alpha, eps_grid, rng))
    return out


def simulation_eps_comparison(n_values, n_trials=80, alpha=0.2, seed_base=24680):
    n_nodes = 5
    r1 = np.array([0.5 - 0.1 * i for i in range(n_nodes)])
    mu_base = np.array([1.25 * (i + 1) for i in range(n_nodes)])
    mu_bounds = [(mu_base[i] - 0.5, mu_base[i] + 0.5) for i in range(n_nodes)]

    fdr = {m: [] for m in EPS_METHODS}; pwr = {m: [] for m in EPS_METHODS}
    fdr_se = {m: [] for m in EPS_METHODS}; pwr_se = {m: [] for m in EPS_METHODS}

    for n in n_values:
        print(f"  n = {n}")
        m_per_node = [int((1 - 0.2 * i) * n) for i in range(n_nodes)]
        fdr_trial = {m: [] for m in EPS_METHODS}; pwr_trial = {m: [] for m in EPS_METHODS}
        for trial in range(n_trials):
            rng = np.random.default_rng(seed_base + trial * 6151 + n)
            stats = run_eps_trial(m_per_node, r1, mu_bounds, alpha, rng)
            for m_name, (fdp, tdp) in stats.items():
                fdr_trial[m_name].append(fdp)
                pwr_trial[m_name].append(tdp)
        mF, sF = _aggregate(fdr_trial); mP, sP = _aggregate(pwr_trial)
        for m_name in EPS_METHODS:
            fdr[m_name].append(mF[m_name]); fdr_se[m_name].append(sF[m_name])
            pwr[m_name].append(mP[m_name]); pwr_se[m_name].append(sP[m_name])
    return fdr, pwr, fdr_se, pwr_se


def plot_eps_results(x_values, fdr, pwr, alpha, out_path, fdr_se=None, pwr_se=None):
    plot_results(x_values, fdr, pwr,
                 xlabel='$n$', title_suffix='vs. sample size (different $\\varepsilon$)',
                 alpha=alpha, out_path=out_path, xlog=True,
                 fdr_se=fdr_se, pwr_se=pwr_se,
                 methods=EPS_METHODS, style=EPS_STYLE)


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    print('Experiment 1: vary n')
    n_values = [1000, 2000, 5000, 10000]
    fdr_n, pwr_n, fdrse_n, pwrse_n = simulation_vary_n(n_values, n_trials=80)
    plot_results(n_values, fdr_n, pwr_n,
                 xlabel='$n$ (base p-values per heaviest node)',
                 title_suffix='vs. sample size',
                 alpha=0.2, out_path=os.path.join(out_dir, 'sim_vary_n.png'),
                 xlog=True, fdr_se=fdrse_n, pwr_se=pwrse_n)
    np.savez(os.path.join(out_dir, 'sim_vary_n.npz'),
             n=np.array(n_values),
             **{f'fdr_{m}': np.array(v) for m, v in fdr_n.items()},
             **{f'pwr_{m}': np.array(v) for m, v in pwr_n.items()},
             **{f'fdrse_{m}': np.array(v) for m, v in fdrse_n.items()},
             **{f'pwrse_{m}': np.array(v) for m, v in pwrse_n.items()})

    print('Experiment 2: vary mu_base')
    mu_values = [1.0, 1.25, 1.5, 1.75, 2.0]
    fdr_mu, pwr_mu, fdrse_mu, pwrse_mu = simulation_vary_mu(
        mu_values, n_per_node_base=1000, n_trials=80)
    plot_results(mu_values, fdr_mu, pwr_mu,
                 xlabel='signal multiplier $\\eta$ (so $\\mu_{\\mathrm{base}}^{(i)} = \\eta\\cdot i$)',
                 title_suffix='vs. signal strength', alpha=0.2,
                 out_path=os.path.join(out_dir, 'sim_vary_mu.png'),
                 xlog=False, fdr_se=fdrse_mu, pwr_se=pwrse_mu)
    np.savez(os.path.join(out_dir, 'sim_vary_mu.npz'),
             mu=np.array(mu_values),
             **{f'fdr_{m}': np.array(v) for m, v in fdr_mu.items()},
             **{f'pwr_{m}': np.array(v) for m, v in pwr_mu.items()},
             **{f'fdrse_{m}': np.array(v) for m, v in fdrse_mu.items()},
             **{f'pwrse_{m}': np.array(v) for m, v in pwrse_mu.items()})

    print('Experiment 3: epsilon comparison')
    eps_n_values = [500, 1000, 2000, 5000, 10000]
    fdr_eps, pwr_eps, fdrse_eps, pwrse_eps = simulation_eps_comparison(
        eps_n_values, n_trials=60)
    plot_eps_results(eps_n_values, fdr_eps, pwr_eps,
                     alpha=0.2, out_path=os.path.join(out_dir, 'sim_eps.png'),
                     fdr_se=fdrse_eps, pwr_se=pwrse_eps)
    np.savez(os.path.join(out_dir, 'sim_eps.npz'),
             n=np.array(eps_n_values),
             **{f'fdr_{m}': np.array(v) for m, v in fdr_eps.items()},
             **{f'pwr_{m}': np.array(v) for m, v in pwr_eps.items()},
             **{f'fdrse_{m}': np.array(v) for m, v in fdrse_eps.items()},
             **{f'pwrse_{m}': np.array(v) for m, v in pwrse_eps.items()})

    print('Experiment 4: vary AR(1) within-node correlation rho')
    rho_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    fdr_rho, pwr_rho, fdrse_rho, pwrse_rho = simulation_vary_rho(
        rho_values, n=2000, n_trials=80)
    plot_results(rho_values, fdr_rho, pwr_rho,
                 xlabel=r'within-node AR(1) correlation $\rho$',
                 title_suffix=r'vs. within-node correlation ($n=2000$)',
                 alpha=0.2, out_path=os.path.join(out_dir, 'sim_vary_rho.png'),
                 xlog=False, fdr_se=fdrse_rho, pwr_se=pwrse_rho)
    np.savez(os.path.join(out_dir, 'sim_vary_rho.npz'),
             rho=np.array(rho_values),
             **{f'fdr_{m}': np.array(v) for m, v in fdr_rho.items()},
             **{f'pwr_{m}': np.array(v) for m, v in pwr_rho.items()},
             **{f'fdrse_{m}': np.array(v) for m, v in fdrse_rho.items()},
             **{f'pwrse_{m}': np.array(v) for m, v in pwrse_rho.items()})

    print('Experiment 5: Cauchy statistics, vary n')
    cau_n_values = [500, 1000, 2000, 5000, 10000]
    fdr_c, pwr_c, fdrse_c, pwrse_c = simulation_cauchy(cau_n_values, n_trials=80)
    plot_results(cau_n_values, fdr_c, pwr_c,
                 xlabel='$n$ (Cauchy statistics)',
                 title_suffix='vs. sample size (heavy-tailed)',
                 alpha=0.2, out_path=os.path.join(out_dir, 'sim_cauchy.png'),
                 xlog=True, fdr_se=fdrse_c, pwr_se=pwrse_c)
    np.savez(os.path.join(out_dir, 'sim_cauchy.npz'),
             n=np.array(cau_n_values),
             **{f'fdr_{m}': np.array(v) for m, v in fdr_c.items()},
             **{f'pwr_{m}': np.array(v) for m, v in pwr_c.items()})

    print('Experiment 6: vary number of nodes Nodes')
    N_values = [5, 10, 20]
    fdr_N, pwr_N, fdrse_N, pwrse_N = simulation_vary_N(
        N_values, m_total_target=20000, n_trials=80)
    plot_results(N_values, fdr_N, pwr_N,
                 xlabel='number of nodes $N$',
                 title_suffix=r'vs. network size ($m_{\mathrm{total}}\!\approx\!10000$)',
                 alpha=0.2, out_path=os.path.join(out_dir, 'sim_vary_N.png'),
                 xlog=True, fdr_se=fdrse_N, pwr_se=pwrse_N)
    np.savez(os.path.join(out_dir, 'sim_vary_N.npz'),
             N=np.array(N_values),
             **{f'fdr_{m}': np.array(v) for m, v in fdr_N.items()},
             **{f'pwr_{m}': np.array(v) for m, v in pwr_N.items()})

    print('Experiment 7: vary target FDR alpha')
    alpha_values = [0.01, 0.05, 0.10, 0.20]
    fdr_a, pwr_a, fdrse_a, pwrse_a = simulation_vary_alpha(
        alpha_values, n=5000, n_trials=50)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for m_name in METHODS:
        s = STYLE[m_name]
        axes[0].plot(alpha_values, fdr_a[m_name], color=s['color'],
                     marker=s['marker'], linestyle=s['linestyle'],
                     label=m_name, markersize=6)
        axes[1].plot(alpha_values, pwr_a[m_name], color=s['color'],
                     marker=s['marker'], linestyle=s['linestyle'],
                     label=m_name, markersize=6)
    axes[0].plot(alpha_values, alpha_values, color='black', linestyle=':',
                 label=r'$y = \alpha$')
    axes[0].set_xlabel(r'target FDR $\alpha$')
    axes[1].set_xlabel(r'target FDR $\alpha$')
    axes[0].set_ylabel('FDR'); axes[1].set_ylabel('Power')
    axes[0].set_title(r'FDR vs. $\alpha$ ($n=5000$)')
    axes[1].set_title(r'Power vs. $\alpha$ ($n=5000$)')
    axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
    axes[0].legend(loc='best', fontsize=8); axes[1].legend(loc='best', fontsize=8)
    axes[0].set_xscale('log'); axes[1].set_xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sim_vary_alpha.png'), dpi=130, bbox_inches='tight')
    plt.close()
    print('  Saved sim_vary_alpha.png')
    np.savez(os.path.join(out_dir, 'sim_vary_alpha.npz'),
             alpha=np.array(alpha_values),
             **{f'fdr_{m}': np.array(v) for m, v in fdr_a.items()},
             **{f'pwr_{m}': np.array(v) for m, v in pwr_a.items()})

    print('Done.')

"""e-value cross-fit greedy aggregation (e-CFGA): a Bonferroni-free combiner.

Construction (under the random-effect model, Lemma 1):
  Per fold b in {1, 2}:
    * Selection on D_b with the Storey grid (same as method_A_prime_storey).
    * Stop at level ALPHA (not alpha/2) using the cross-fit FDP estimator
        FDP_hat_b(M) = (1 + m_inf * eps * M) / (R_b(M) v 1)
      -> stopping index M_hat^(b) = max{M : FDP_hat_b(M) <= alpha}.
    * Per-hypothesis e-value for k in D_b' (inference half of fold b):
        e_k^(b) = m_inf * 1{k rejected by fold b} / w^(b),
      where w^(b) = 1 + m_inf * eps * M_hat^(b).
    * For k not in D_b'  -> e_k^(b) = 0.
  Combined e-value: e_k = e_k^(1) + e_k^(2)  (only one term is nonzero per k).
  Apply e-BH at level ALPHA: reject {k : e_k >= m / (alpha * R)}.

Sum-bound (Lemma 1): E[sum_{null k in D_b'} e_k^(b) | D_b] <= m_b' for each fold,
so E[sum_{null} e_k] <= m_1 + m_2 = m. Wang-Ramdas e-BH then gives FDR <= alpha
under arbitrary dependence -- no Bonferroni division.

Within each fold all rejected hypotheses receive the same e-value, so the full
e-value vector has at most two distinct nonzero values. e-BH then reduces to:
  - sort folds by e-value descending;
  - reject all-of-fold-high if e_high * alpha * R_high >= m;
  - or reject both folds if e_low * alpha * (R_high + R_low) >= m;
  - take the largest feasible total rejection count.

This file exposes one method:
  e_cfga_storey(pvalues_list, is_null_list, alpha, epsilon, rng, lambda_=0.5)
returning (V, R) matching the other methods' calling convention in mn_sweep.
"""
import numpy as np
import simulations as s


def _fold_evalue(sel_p, inf_p, alpha, epsilon, lambda_):
    """One fold: greedy on sel_p, step-up at level alpha on inf_p with the
    rate-free Storey numerator. Returns (e_weight, admitted_intervals) where
    e_weight = m_inf / (1 + m_inf * eps * M_hat); admitted_intervals is the
    list of (node, lo, hi] making up the rejected region (empty if none)."""
    m_inf = sum(len(p) for p in inf_p)
    if m_inf == 0:
        return 0.0, []
    ranked, _ = s._greedy_full_rank(sel_p, epsilon)
    ranked_R = [r for r in ranked if r[4] <= lambda_]
    if not ranked_R:
        return 0.0, []
    nodes, r_arr, _, _ = s._ranked_eval_counts(
        inf_p, [np.zeros(len(p), dtype=bool) for p in inf_p], ranked_R)
    cumR = np.cumsum(r_arr)
    M_idx = np.arange(1, len(ranked_R) + 1, dtype=float)
    fdp = (1.0 + m_inf * epsilon * M_idx) / np.maximum(cumR, 1.0)
    ok = np.nonzero(fdp <= alpha)[0]
    if ok.size == 0:
        return 0.0, []
    M_hat = int(ok[-1]) + 1
    e_w = m_inf / (1.0 + m_inf * epsilon * M_hat)
    return e_w, [(r[0], r[3], r[4]) for r in ranked_R[:M_hat]]


def _fold_evalue_inflated(sel_p, inf_p, gamma, epsilon, lambda_, delta):
    """One fold of INFLATED e-CFGA: greedy on sel_p, step-up at level gamma on
    inf_p with the inflated lifted-space numerator
        bar_V(M) = m_inf * sum_i (pihat_i + delta) * nu(Gamma_M^(i)),
    pihat_i estimated on the ranking-half tail (eq. pihat). Returns
    (e_weight, admitted_intervals)."""
    n_nodes = len(sel_p)
    m_sel = sum(len(p) for p in sel_p)
    m_inf = sum(len(p) for p in inf_p)
    if m_sel == 0 or m_inf == 0:
        return 0.0, []
    w = np.array([float(np.sum(sel_p[i] > lambda_)) / (m_sel * (1.0 - lambda_))
                  for i in range(n_nodes)]) + delta
    ranked, _ = s._greedy_full_rank(sel_p, epsilon)
    ranked_R = [r for r in ranked if r[4] <= lambda_]
    if not ranked_R:
        return 0.0, []
    nodes, r_arr, _, widths = s._ranked_eval_counts(
        inf_p, [np.zeros(len(p), dtype=bool) for p in inf_p], ranked_R)
    cumR = np.cumsum(r_arr)
    barV = m_inf * np.cumsum(w[nodes] * widths)
    fdp = (1.0 + barV) / np.maximum(cumR, 1.0)
    ok = np.nonzero(fdp <= gamma)[0]
    if ok.size == 0:
        return 0.0, []
    best = int(ok[-1])
    e_w = m_inf / (1.0 + barV[best])
    return e_w, [(r[0], r[3], r[4]) for r in ranked_R[:best + 1]]


def e_cfga_inflated(pvalues_list, is_null_list, alpha, epsilon, rng,
                    S=10, lambda_=0.5, eps_grid=None, gamma=None, eta=None):
    """Derandomized e-CFGA with the INFLATED plug-in (the rigorous tier of
    Theorem thm:ecfga): per fold, the null-mass numerator uses Storey
    null-rate estimates inflated by the Hoeffding margin of eq. (delta) at
    per-split confidence eta/S, so FDR <= alpha + eta finite-sample. eta
    defaults to 1/m. Adaptive epsilon per fold (eps_grid) uses the
    inflated acceptance in the inner CV, sigma(ranking half)-measurable."""
    n_nodes = len(pvalues_list)
    m_total = sum(len(p) for p in pvalues_list)
    if m_total == 0:
        return 0, 0
    if eta is None:
        eta = 1.0 / m_total
    if gamma is None:
        gamma = 0.75 * alpha
    eta_split = eta / S
    ebar = [np.zeros(len(p)) for p in pvalues_list]
    for _ in range(S):
        z = [rng.random(len(p)) < 0.5 for p in pvalues_list]
        for fold in (0, 1):
            sel = [pvalues_list[i][z[i]] if fold == 0 else pvalues_list[i][~z[i]]
                   for i in range(n_nodes)]
            inf_mask = [~z[i] if fold == 0 else z[i] for i in range(n_nodes)]
            inf_p = [pvalues_list[i][inf_mask[i]] for i in range(n_nodes)]
            m_sel = sum(len(p) for p in sel)
            if m_sel == 0:
                continue
            # eq. (delta) with per-split confidence eta/S
            delta = (s._dkw_margin(m_sel, n_nodes, eta_split)
                     / (1.0 - lambda_))
            eps_f = epsilon
            if eps_grid is not None:
                sel_n = [np.zeros(len(x), dtype=bool) for x in sel]
                eps_f = s.adaptive_epsilon_choice_augmented(
                    sel, sel_n, gamma, eps_grid, rng, eta_split,
                    storey=True, lambda_=lambda_)
            e_w, intervals = _fold_evalue_inflated(
                sel, inf_p, gamma, eps_f, lambda_, delta)
            if e_w == 0.0 or not intervals:
                continue
            for (i, lo, hi) in intervals:
                i = int(i)
                hit = inf_mask[i] & (pvalues_list[i] > lo) & (pvalues_list[i] <= hi)
                ebar[i][hit] += e_w
    ebar = np.concatenate(ebar) / float(S)
    null_flat = np.concatenate(is_null_list)
    order = np.argsort(-ebar)
    e_sorted = ebar[order]
    ks = np.arange(1, m_total + 1, dtype=float)
    feas = np.nonzero(ks * e_sorted >= m_total / alpha)[0]
    if feas.size == 0:
        return 0, 0
    k_star = int(feas[-1]) + 1
    V = int(np.sum(null_flat[order[:k_star]]))
    return V, k_star


def e_cfga_derand(pvalues_list, is_null_list, alpha, epsilon, rng,
                  S=10, lambda_=0.5, eps_grid=None, gamma=None):
    """Derandomized e-CFGA: S independent Bernoulli(1/2) splits; per split,
    each fold produces generalized e-values
        e_k = m_inf * 1{k in rejected region} / (1 + m_inf*eps*M_hat)
    for the hypotheses in its inference half (step-up at level ALPHA, no
    Bonferroni); e-values are averaged over splits and e-BH at level alpha
    gives the final rejection set.

    Validity (oracle-mass tier, Theorem in Sec. e-CFGA): per fold,
    Lemma 1 bounds the null e-mass by m_inf, so the two folds of one split
    contribute at most m_1 + m_2 = m, and averaging keeps sum_null E[e] <= m
    under arbitrary dependence across splits; e-BH then gives FDR <= alpha.
    The Storey rate-free numerator used here is the same plug-in heuristic
    as CFGA+Storey. If eps_grid is given, epsilon is chosen per fold by
    sub-split CV on the ranking half (sigma(ranking half)-measurable, so the
    guarantee is preserved fold by fold). NOTE: with the ordinary Storey
    plug-in inside each fold this variant is HEURISTIC, exactly as for
    single-split CFGA+Storey; the finite-sample guarantee holds for the
    oracle and inflated tiers (e_cfga_inflated).
    """
    n_nodes = len(pvalues_list)
    m_total = sum(len(p) for p in pvalues_list)
    if m_total == 0:
        return 0, 0
    # Fold stopping level. Stopping each fold at alpha puts e-BH exactly at
    # its feasibility boundary (k ~ 2R needed vs ~2R*rejprob available); a
    # fixed stricter level gamma ~ alpha*rejprob^2 restores slack. gamma is
    # a constant chosen a priori, so each fold's index remains a backward
    # stopping time and the e-value bound is unaffected.
    if gamma is None:
        gamma = 0.75 * alpha
    ebar = [np.zeros(len(p)) for p in pvalues_list]
    for _ in range(S):
        # i.i.d. Bernoulli(1/2) assignment per hypothesis (Algorithm 1 Step 1)
        z = [rng.random(len(p)) < 0.5 for p in pvalues_list]
        for fold in (0, 1):
            sel = [pvalues_list[i][z[i]] if fold == 0 else pvalues_list[i][~z[i]]
                   for i in range(n_nodes)]
            inf_mask = [~z[i] if fold == 0 else z[i] for i in range(n_nodes)]
            inf_p = [pvalues_list[i][inf_mask[i]] for i in range(n_nodes)]
            eps_f = epsilon
            if eps_grid is not None:
                sel_n = [np.zeros(len(x), dtype=bool) for x in sel]
                eps_f = s.adaptive_epsilon_choice_storey(
                    sel, sel_n, gamma, eps_grid, rng, lambda_)
            e_w, intervals = _fold_evalue(sel, inf_p, gamma, eps_f, lambda_)
            if e_w == 0.0 or not intervals:
                continue
            for (i, lo, hi) in intervals:
                i = int(i)
                hit = inf_mask[i] & (pvalues_list[i] > lo) & (pvalues_list[i] <= hi)
                ebar[i][hit] += e_w
    ebar = np.concatenate(ebar) / float(S)
    null_flat = np.concatenate(is_null_list)
    # e-BH at level alpha: largest k with k * e_(k) >= m / alpha
    order = np.argsort(-ebar)
    e_sorted = ebar[order]
    ks = np.arange(1, m_total + 1, dtype=float)
    feas = np.nonzero(ks * e_sorted >= m_total / alpha)[0]
    if feas.size == 0:
        return 0, 0
    k_star = int(feas[-1]) + 1
    V = int(np.sum(null_flat[order[:k_star]]))
    return V, k_star


def e_cfga_storey(pvalues_list, is_null_list, alpha, epsilon, rng,
                  lambda_=0.5):
    n_nodes = len(pvalues_list)
    # split (uniform random per-node, same as method_A_prime_storey)
    D1_p, D1_n, D2_p, D2_n = s.make_split(pvalues_list, is_null_list, rng)
    m_total = sum(len(p) for p in pvalues_list)
    if m_total == 0:
        return 0, 0

    folds = []  # entries: (R_b, V_b, e_b)
    for sel_p, inf_p, inf_n in [(D1_p, D2_p, D2_n), (D2_p, D1_p, D1_n)]:
        m_inf = sum(len(p) for p in inf_p)
        if m_inf == 0:
            folds.append((0, 0, 0.0))
            continue
        ranked, _ = s._greedy_full_rank(sel_p, epsilon)
        ranked_R = [r for r in ranked if r[4] <= lambda_]
        if not ranked_R:
            folds.append((0, 0, 0.0))
            continue
        nodes, r_arr, v_arr, _ = s._ranked_eval_counts(inf_p, inf_n, ranked_R)
        cumR = np.cumsum(r_arr)
        cumV = np.cumsum(v_arr)
        M_idx = np.arange(1, len(ranked_R) + 1, dtype=float)
        bar_V = m_inf * epsilon * M_idx
        fdp = (1.0 + bar_V) / np.maximum(cumR, 1.0)
        ok = np.nonzero(fdp <= alpha)[0]   # NOTE: level alpha, NOT alpha/2
        if ok.size == 0:
            folds.append((0, 0, 0.0))
            continue
        best = int(ok[-1])
        M_hat = best + 1
        R_b = int(cumR[best]); V_b = int(cumV[best])
        w_b = 1.0 + m_inf * epsilon * M_hat
        e_b = m_inf / w_b
        folds.append((R_b, V_b, e_b))

    # e-BH on the combined e-value vector (at most two distinct nonzero levels).
    folds.sort(key=lambda x: -x[2])    # high e-value first
    R_cum = 0; V_cum = 0
    feasible = []
    for R_b, V_b, e_b in folds:
        if R_b == 0 or e_b == 0.0:
            continue
        R_cum += R_b; V_cum += V_b
        # e-BH feasibility at level R_cum: e_b >= m / (alpha * R_cum)
        if e_b * alpha * R_cum >= m_total:
            feasible.append((R_cum, V_cum))
    if not feasible:
        return 0, 0
    R_best, V_best = max(feasible, key=lambda x: x[0])
    return V_best, R_best

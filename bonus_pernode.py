"""Per-node BONuS-CFGA (Remark 6): replaces the single global tail ratio
        (N(A)+1) / Nt(A)
in eq:bonus-fdp with per-node ratios
        N^(i)(A) / Nt^(i)(A)
that target n_0^(i) * nu(Gamma_M^(i)) for ANY bag, removing the dependence on
the oracle synthetic allocation.

The Storey-BONuS estimator becomes
        FDP_hat(M) = ( 1 + sum_i (N^(i)(A) / Nt^(i)(A)) * Nt^(i)(Gamma_M) )
                     / ( R(M) v 1 ),
keeping exactly ONE +1 (the SeqStep+ offset, outside the sum). No extra "+1"
is inserted in the per-node denominator: doing so would bias the ratio toward
zero (smaller FDP estimate) and threaten FDR control.

For nodes with Nt^(i)(A) == 0 (quiet nodes whose synthetic tail count is
empty) the per-node ratio is undefined. Two policies are exposed:
  - drop  : omit the node's contribution (predictable null mass treated as 0
            for that node). Anti-conservative; we use it to diagnose whether
            small-node noise materially hurts FDR.
  - global: fall back to the GLOBAL tail ratio (N(A)+1)/Nt(A) for that node,
            which preserves the original procedure's behaviour at small nodes
            while keeping per-node calibration where it's well-defined.
"""
import numpy as np
import simulations as s


def _make_bag(pvalues_list, rng, c=1.0):
    return [rng.random(int(round(c * len(pvalues_list[i]))))
            for i in range(len(pvalues_list))]


def bonus_cfga_pernode(pvalues_list, is_null_list, alpha, epsilon, rng,
                       lambda_=0.5, c=1.0, small_node='drop'):
    """Per-node-calibrated BONuS-CFGA (Remark 6).

    small_node in {'drop', 'global'} controls the Nt^(i)(A)=0 handling.
    Returns (V, R) -- hypothesis-level true-null and total rejection counts.
    """
    n_nodes = len(pvalues_list)
    if sum(len(p) for p in pvalues_list) == 0:
        return 0, 0
    synth = _make_bag(pvalues_list, rng, c)
    if sum(len(b) for b in synth) == 0:
        return 0, 0

    # Pool real + synthetic at each node; rank intervals by pooled count.
    synth_null = [np.ones(len(synth[i]), dtype=bool) for i in range(n_nodes)]
    pooled = [np.concatenate([pvalues_list[i], synth[i]]) for i in range(n_nodes)]
    ranked, _ = s._greedy_full_rank(pooled, epsilon)
    ranked_R = [r for r in ranked if r[4] <= lambda_]
    if not ranked_R:
        return 0, 0

    # Per-interval counts on REAL and SYNTHETIC, tagged with node index.
    nodes, N_real, V_real, _ = s._ranked_eval_counts(pvalues_list, is_null_list, ranked_R)
    _, N_syn, _, _ = s._ranked_eval_counts(synth, synth_null, ranked_R)

    # Per-node tail counts at A = (lambda_, 1].
    N_A  = np.array([int(np.sum(pvalues_list[i] > lambda_)) for i in range(n_nodes)],
                    dtype=float)
    Nt_A = np.array([int(np.sum(synth[i]       > lambda_)) for i in range(n_nodes)],
                    dtype=float)
    NA_tot  = float(N_A.sum())
    NtA_tot = float(Nt_A.sum())

    # Per-node ratio. NO "+1" inside (would bias toward 0).
    has_tail = Nt_A > 0
    ratio = np.zeros(n_nodes, dtype=float)
    ratio[has_tail] = N_A[has_tail] / Nt_A[has_tail]

    if small_node == 'drop':
        # nodes with Nt^(i)(A)==0 contribute 0 (anti-conservative; diagnostic).
        pass
    elif small_node == 'global':
        # fall back to global ratio for small nodes (conservative).
        global_ratio = (NA_tot + 1.0) / max(NtA_tot, 1.0)
        ratio[~has_tail] = global_ratio
    else:
        raise ValueError(f"small_node must be 'drop' or 'global', got {small_node!r}")

    # Per-step null-mass contribution from each ranked interval: ratio[node] * 1 syn
    # (we cumulate synthetic counts per node).
    per_step = ratio[nodes] * N_syn  # length K; ratio applied to each interval's syn count
    cumN  = np.cumsum(N_real)
    cumV  = np.cumsum(V_real)
    cumWeighted = np.cumsum(per_step)   # sum_i ratio[i] * Nt^(i)(Gamma_M)

    # SeqStep+ FDP estimator: ONE +1 in the numerator, outside the sum.
    fdp = (1.0 + cumWeighted) / np.maximum(cumN, 1.0)
    ok = np.nonzero(fdp <= alpha)[0]
    if ok.size == 0:
        return 0, 0
    best = int(ok[-1])
    return int(cumV[best]), int(cumN[best])


def bonus_cfga_pernode_budget(pvalues_list, is_null_list, alpha, epsilon, rng,
                              lambda_=0.5, c=1.0, allocation='proportional'):
    """Per-node-BUDGETED BONuS-GA (Theorem 4): the provable per-node variant.

    Each node i gets a fixed budget share alpha_i (masking-measurable: computed
    from POOLED counts only), sum_i alpha_i = alpha. The step-up index is
        M_hat = max{ M : for every node i active at M,
                     (N^(i)(A)+1)/Nt^(i)(A) * (Nt^(i)(Gamma_M)+1)
                         <= alpha_i * (R(M) v 1) },
    where "active" means Gamma_M^(i) is nonempty. Per-node exchangeability of
    (real-null + synthetic) U[0,1] p-values within each node -- which holds for
    ANY bag allocation -- gives E[Q_i] <= 1 per node, and the fixed shares
    turn FDP <= sum_i alpha_i Q_i into FDR <= alpha. Nodes with Nt^(i)(A)=0
    can never be active (their constraint is +infinity).

    Default shares: proportional to each node's pooled EXCESS count over the
    uniform profile across its candidate intervals (a masking-measurable
    signal proxy that concentrates budget on signal-bearing nodes).
    """
    if allocation != 'proportional':
        raise ValueError(
            "bonus_cfga_pernode_budget: Theorem `thm:bonus-pernode' requires a "
            "data-independent bag allocation; allocation='storey' depends on "
            "the observed p-values and voids the guarantee. Use the "
            "per-node-calibrated entry point for heuristic allocations.")
    n_nodes = len(pvalues_list)
    if sum(len(p) for p in pvalues_list) == 0:
        return 0, 0
    synth = s._make_bag(pvalues_list, rng, c, allocation=allocation,
                        lambda_=lambda_)
    if sum(len(b) for b in synth) == 0:
        return 0, 0
    synth_null = [np.ones(len(synth[i]), dtype=bool) for i in range(n_nodes)]
    pooled = [np.concatenate([pvalues_list[i], synth[i]]) for i in range(n_nodes)]
    ranked, L = s._greedy_full_rank(pooled, epsilon)
    ranked_R = [r for r in ranked if r[4] <= lambda_]
    if not ranked_R:
        return 0, 0

    nodes, N_real, V_real, widths = s._ranked_eval_counts(
        pvalues_list, is_null_list, ranked_R)
    _, N_syn, _, _ = s._ranked_eval_counts(synth, synth_null, ranked_R)
    N_pool = N_real + N_syn

    # --- masking-measurable budget shares ---
    # pooled head count per node and uniform per-interval expectation
    head_pool = np.array([float(np.sum(pooled[i] <= lambda_))
                          for i in range(n_nodes)])
    exp_unif = head_pool[nodes] * (widths / lambda_)
    excess = np.maximum(N_pool - exp_unif, 0.0)
    w = np.zeros(n_nodes)
    np.add.at(w, nodes, excess)
    if w.sum() <= 0:
        np.add.at(w, nodes, N_pool.astype(float))  # fallback: pooled mass
    if w.sum() <= 0:
        return 0, 0
    # Masking-measurable family filtering: drop candidate intervals of nodes
    # whose share is negligible (< 5% of the max), so that a noise-admitted
    # interval of a near-zero-budget node cannot block the step-up index.
    keep = w >= 0.05 * w.max()
    if not np.all(keep[nodes]):
        sel = keep[nodes]
        ranked_R = [r for k, r in zip(sel, ranked_R) if k]
        if not ranked_R:
            return 0, 0
        nodes, N_real, V_real, widths = s._ranked_eval_counts(
            pvalues_list, is_null_list, ranked_R)
        _, N_syn, _, _ = s._ranked_eval_counts(synth, synth_null, ranked_R)
        N_pool = N_real + N_syn
    w = np.where(keep, w, 0.0)
    alpha_i = alpha * w / w.sum()

    # per-node tail counts on A=(lambda,1]
    N_A = np.array([float(np.sum(pvalues_list[i] > lambda_))
                    for i in range(n_nodes)])
    Nt_A = np.array([float(np.sum(synth[i] > lambda_)) for i in range(n_nodes)])

    # cumulative per-node synthetic counts along the ranking
    K = len(ranked_R)
    cumNt_i = np.zeros((K, n_nodes))   # Nt^(i)(Gamma_M) for each M (rows)
    onehot = np.zeros((K, n_nodes))
    onehot[np.arange(K), nodes] = N_syn
    np.cumsum(onehot, axis=0, out=cumNt_i)
    active = np.zeros((K, n_nodes), dtype=bool)
    act1 = np.zeros((K, n_nodes), dtype=bool)
    act1[np.arange(K), nodes] = True
    np.logical_or.accumulate(act1, axis=0, out=active)

    cumR = np.cumsum(N_real)
    cumV = np.cumsum(V_real)
    Rv = np.maximum(cumR, 1.0)

    # per-node statistic W_i(M) = (N_A+1)/Nt_A * (Nt_i(Gamma_M)+1); inf if Nt_A=0
    with np.errstate(divide='ignore'):
        ratio_i = np.where(Nt_A > 0, (N_A + 1.0) / np.maximum(Nt_A, 1.0), np.inf)
    W = ratio_i[None, :] * (cumNt_i + 1.0)
    okM = np.all(~active | (W <= alpha_i[None, :] * Rv[:, None]), axis=1)
    idx = np.nonzero(okM)[0]
    if idx.size == 0:
        return 0, 0
    best = int(idx[-1])
    return int(cumV[best]), int(cumR[best])

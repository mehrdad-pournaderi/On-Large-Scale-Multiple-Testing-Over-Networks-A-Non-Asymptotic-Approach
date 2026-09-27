"""Extended BONuS-GA ablation at m_hub=250 with PAIRED arms and Monte Carlo SEs.

Within each trial every arm sees the same data, and arms sharing an
allocation rule see the IDENTICAL realized bag (same RNG stream restored
before each arm), so pairwise contrasts isolate one design choice at a time:
  calibration (global vs per-node), bag held fixed:  (1) vs (4), (6) vs (7)
  allocation (storey vs proportional), calibration fixed: (1) vs (6), (4) vs (7)
  allocation noise (oracle r0 weights): (2) vs (1)
  bag size (c=4): (3) vs (1)
  budgeted certificate vs its heuristic siblings: (5)
  CFGA+Storey+adaptive eps reference, same trials: (8)
Seeds: 424242 + 6661*t, NT=200 (matching the earlier unpaired ablation).
Writes outputs/sim_ablation_fixedbag.npz with per-trial fdp/pwr arrays per arm.
"""
import numpy as np, time
import simulations as s
import bonus_pernode as bp

N_RICH, M_RICH, PI_RICH, MU_RICH = 2, 250, 0.30, 3.0
N_NULL, M_NULL = 300, 20
ALPHA = 0.20
N = N_RICH + N_NULL
m_per_node = [M_RICH]*N_RICH + [M_NULL]*N_NULL
r1 = np.array([PI_RICH]*N_RICH + [0.0]*N_NULL)
mu_bounds = [(MU_RICH-.5, MU_RICH+.5)]*N_RICH + [(0.,0.)]*N_NULL
r0_true = 1.0 - r1
eps = ALPHA/np.sqrt(sum(m_per_node))
NT = 200

def _bag_oracle(pv, rng, c=1.0):
    m_i = np.array([len(p) for p in pv], float)
    w = m_i*r0_true
    sizes = rng.multinomial(int(round(c*m_i.sum())), w/max(w.sum(),1e-12))
    return [rng.random(int(k)) for k in sizes]

ARMS = {
 'global_storey':    lambda pv,isn,rng: s.bonus_cfga(pv,isn,ALPHA,eps,rng,c=1.,allocation='storey'),
 'oracle_alloc':     lambda pv,isn,rng: s._bonus_accept(pv,isn,_bag_oracle(pv,rng),ALPHA,eps,0.5),
 'bigbag_c4':        lambda pv,isn,rng: s.bonus_cfga(pv,isn,ALPHA,eps,rng,c=4.,allocation='storey'),
 'pernode_storey':   lambda pv,isn,rng: bp.bonus_cfga_pernode(pv,isn,ALPHA,eps,rng,c=1.,small_node='global'),
 'budget_prop':      lambda pv,isn,rng: bp.bonus_cfga_pernode_budget(pv,isn,ALPHA,eps,rng,c=1.,allocation='proportional'),
 'global_prop':      lambda pv,isn,rng: s.bonus_cfga(pv,isn,ALPHA,eps,rng,c=1.,allocation='proportional'),
 'pernode_prop':     lambda pv,isn,rng: _pernode_prop(pv,isn,rng),
 'cfga_storey_ada':  lambda pv,isn,rng: s.method_A_prime_storey_adaptive(
                         pv,isn,ALPHA,s._eps_grid_for(sum(m_per_node),ALPHA),rng),
}
def _pernode_prop(pv,isn,rng):
    # per-node calibration on a PROPORTIONAL bag: temporarily swap the
    # default allocation used inside bonus_cfga_pernode.
    import simulations as _s
    orig = bp._make_bag
    bp._make_bag = lambda p,r,c=1.0: _s._make_bag(p,r,c,allocation='proportional')
    try:    return bp.bonus_cfga_pernode(pv,isn,ALPHA,eps,rng,c=1.,small_node='global')
    finally: bp._make_bag = orig

if __name__ == '__main__':
    res = {k: ([],[]) for k in ARMS}
    t0=time.time()
    for t in range(NT):
        rng = np.random.default_rng(424242 + 6661*t)
        pv, isn = s.generate_pvalues(m_per_node, r1, mu_bounds, rng)
        n1 = sum(int(np.sum(~isn[i])) for i in range(N))
        st = rng.bit_generator.state
        for k, fn in ARMS.items():
            rng.bit_generator.state = st          # identical stream per arm
            V,R = fn(pv,isn,rng)
            res[k][0].append(V/max(R,1)); res[k][1].append((R-V)/max(n1,1))
    out = {}
    for k in ARMS:
        f = np.array(res[k][0]); p = np.array(res[k][1])
        out[f'fdp_{k}']=f; out[f'pwr_{k}']=p
        print(f"{k:16s} FDR={f.mean():.3f}±{2*f.std(ddof=1)/np.sqrt(NT):.3f}  "
              f"pwr={p.mean():.3f}±{2*p.std(ddof=1)/np.sqrt(NT):.3f}")
    # paired contrasts (same trials): power differences with paired SE
    def paired(a,b,lab):
        d = np.array(res[a][1])-np.array(res[b][1])
        print(f"  Δpwr {lab:36s} {d.mean():+.3f} ± {2*d.std(ddof=1)/np.sqrt(NT):.3f} (paired 2SE)")
    paired('pernode_storey','global_storey','calibration | storey bag fixed')
    paired('pernode_prop','global_prop',    'calibration | proportional bag fixed')
    paired('global_prop','global_storey',   'allocation  | global calibration')
    paired('pernode_prop','pernode_storey', 'allocation  | per-node calibration')
    paired('oracle_alloc','global_storey',  'oracle weights | global calibration')
    paired('bigbag_c4','global_storey',     'bag size c=4 | global calibration')
    np.savez('outputs/sim_ablation_fixedbag.npz', NT=NT, **out)
    print(f"saved. ({time.time()-t0:.1f}s)")

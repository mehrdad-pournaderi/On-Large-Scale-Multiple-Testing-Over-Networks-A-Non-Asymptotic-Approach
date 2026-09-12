"""Hubs-only AR(1) variant of Experiment 3: correlation applied to the two
hubs only, quiet nulls i.i.d. Same seeds as mn_sweep's rho axis, NT=100.
Verifies the manuscript's claim that this milder perturbation leaves the
un-inflated adaptive variant close to level. Writes outputs/sim_rho_hubsonly.npz."""
import numpy as np, time
import simulations as s
import mn_sweep as M

NT = 100
VALS = [0.0, 0.2, 0.4, 0.6, 0.8]
METHODS = ['CFGA + Storey', 'CFGA + Storey + adaptive eps',
           'Augmented CFGA + Storey + adaptive eps']

if __name__ == '__main__':
    m_per_node, r1, mu_bounds, alpha, _, dist = M.build('rho', 0.0)
    N = len(m_per_node); m_total = sum(m_per_node)
    eps = alpha/np.sqrt(max(m_total,100)); grid = s._eps_grid_for(m_total, alpha)
    out = {}
    for val in VALS:
        rho_vec = [val]*M.N_HUB + [0.0]*M.N_QUIET
        acc = {k: ([],[]) for k in METHODS}
        t0=time.time()
        for t in range(NT):
            rng = np.random.default_rng(313131 + 6661*t + int(round(val*1000)))
            pv, isn = s.generate_pvalues(m_per_node, r1, mu_bounds, rng,
                                         rho=rho_vec, distribution=dist)
            n1 = sum(int(np.sum(~isn[i])) for i in range(N))
            res = {
              'CFGA + Storey': s.method_A_prime_storey(pv, isn, alpha, eps, rng),
              'CFGA + Storey + adaptive eps': s.method_A_prime_storey_adaptive(pv, isn, alpha, grid, rng),
              'Augmented CFGA + Storey + adaptive eps': s.augmented_cfga_storey_adaptive(pv, isn, alpha, grid, rng),
            }
            for k,(V,R) in res.items():
                acc[k][0].append(V/max(R,1)); acc[k][1].append((R-V)/max(n1,1))
        for k in METHODS:
            f=np.array(acc[k][0]); p=np.array(acc[k][1])
            tag=k.replace(' ','_').replace('+','')
            out[f'fdp_{tag}_{val}']=f; out[f'pwr_{tag}_{val}']=p
            print(f"rho_hub={val}  {k:42s} FDR={f.mean():.3f}±{2*f.std(ddof=1)/np.sqrt(NT):.3f} "
                  f"pwr={p.mean():.3f}±{2*p.std(ddof=1)/np.sqrt(NT):.3f}")
        print(f"  ({time.time()-t0:.1f}s)")
    np.savez('outputs/sim_rho_hubsonly.npz', NT=NT, vals=VALS, **out)
    print("saved")

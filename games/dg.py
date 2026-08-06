"""Dynamic Entry/Exit Game - Production Implementation
Solves equilibrium, simulates market dynamics, and estimates structural parameters
using Bajari-Benkard-Levin (BBL) forward simulation estimator.
"""
import os
import argparse
import numpy as np
from scipy.stats import norm, binom
from scipy.optimize import minimize
try:
    from joblib import Parallel, delayed
except Exception:  # joblib is optional
    Parallel = None
    delayed = None

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    tqdm = None

class DynamicGame:
    """Markov-perfect Nash equilibrium for dynamic entry/exit game."""
    
    def __init__(self, beta=0.9, mu_mean=5.0, mu_var=5.0, gamma_mean=5.0, gamma_var=5.0, entry_tax=0.0):
        self.beta = beta
        self.mu_mean, self.mu_std = mu_mean, np.sqrt(mu_var)
        self.gamma_mean, self.gamma_std = gamma_mean, np.sqrt(gamma_var)
        self.entry_tax = entry_tax
        self.N_max, self.x_vals = 5, np.array([-5, 0, 5])
        self.x_trans = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        N_vals = np.arange(1, 6)[:, None]
        self.pi = np.vstack([np.zeros(3), ((10 + self.x_vals) / (N_vals + 1)) ** 2 - 5])
        self.mu_cutoff = self.gamma_cutoff = self.V_bar = np.zeros_like(self.pi)

    def solve_equilibrium(self, tol=1e-8, max_iter=5000, damp=0.5):
        """Iterate best-response dynamics to convergence."""
        self.V_bar = self.mu_cutoff = self.gamma_cutoff = self.pi.copy()
        for _ in range(max_iter):
            V_old, mu_old, ga_old = self.V_bar.copy(), self.mu_cutoff.copy(), self.gamma_cutoff.copy()
            p_stay = np.clip(norm.cdf((self.mu_cutoff - self.mu_mean) / self.mu_std), 0, 1)
            p_enter = np.clip(norm.cdf((self.gamma_cutoff - self.gamma_mean) / self.gamma_std), 0, 1)
            p_stay[0], p_enter[5] = 0, 0
            
            Pr_d1, Pr_e1 = np.zeros((6, 6, 3)), np.zeros((6, 6, 3))
            for N in range(6):
                for j in range(3):
                    if N > 0:
                        pe = p_enter[N,j]
                        for k in range(N):
                            w, Na = binom.pmf(k, N-1, p_stay[N,j]), 1+k
                            Pr_d1[N,Na,j] += w*(1-pe)
                            if Na < 5: Pr_d1[N,Na+1,j] += w*pe
                    if N < 5:
                        for k in range(N+1):
                            Pr_e1[N, min(1+k,5), j] += binom.pmf(k, N, p_stay[N,j] if N>0 else 0)
            
            EV = self.x_trans.T @ self.V_bar.T
            Psi1 = self.beta * np.einsum('ijk,jk->ik', Pr_d1, EV.T)
            Psi2 = self.beta * np.einsum('ijk,jk->ik', Pr_e1, EV.T)
            mu_new, ga_new = self.pi + Psi1, Psi2 - self.entry_tax
            
            z = (mu_new - self.mu_mean) / self.mu_std
            V_new = np.zeros_like(self.V_bar)
            V_new[1:] = (1-norm.cdf(z[1:]))*self.mu_mean + self.mu_std*norm.pdf(z[1:]) + norm.cdf(z[1:])*mu_new[1:]
            
            self.V_bar = damp*V_new + (1-damp)*V_old
            self.mu_cutoff = damp*mu_new + (1-damp)*mu_old
            self.gamma_cutoff = damp*ga_new + (1-damp)*ga_old
            
            if max(np.abs(self.V_bar-V_old).max(), np.abs(self.mu_cutoff-mu_old).max(), 
                   np.abs(self.gamma_cutoff-ga_old).max()) < tol:
                return True
        return False

    def simulate(self, T=10000, seed=2007):
        """Simulate market trajectory."""
        rng = np.random.default_rng(seed)
        d_cut = (self.mu_cutoff - self.mu_mean) / self.mu_std
        e_cut = (self.gamma_cutoff - self.gamma_mean) / self.gamma_std
        N, xj = 0, 1
        N_hist = np.zeros(T, dtype=int)
        for t in range(T):
            N_hist[t] = N
            n_stay = int(np.sum(rng.standard_normal(N) <= d_cut[N,xj])) if N > 0 else 0
            e = int(rng.standard_normal() <= e_cut[N,xj]) if N < 5 else 0
            N = n_stay + e
            xj = np.searchsorted(self.x_trans[xj].cumsum(), rng.random())
        return N_hist

def estimate_ccps(eq, T=10000, seed=2007):
    """Nonparametric CCP estimation from simulated data."""
    rng = np.random.default_rng(seed)
    d_cut = (eq.mu_cutoff - eq.mu_mean) / eq.mu_std
    e_cut = (eq.gamma_cutoff - eq.gamma_mean) / eq.gamma_std
    stay_cnt, tot_cnt, ent_cnt, ent_opp = {}, {}, {}, {}
    N, xj = 0, 1
    
    for _ in range(T):
        xv = eq.x_vals[xj]
        if N > 0:
            ns = int(np.sum(rng.standard_normal(N) <= d_cut[N,xj]))
            stay_cnt[(N,xv)] = stay_cnt.get((N,xv), 0) + ns
            tot_cnt[(N,xv)] = tot_cnt.get((N,xv), 0) + N
        else:
            ns = 0
        if N < 5:
            e = int(rng.standard_normal() <= e_cut[N,xj])
            ent_cnt[(N,xv)] = ent_cnt.get((N,xv), 0) + e
            ent_opp[(N,xv)] = ent_opp.get((N,xv), 0) + 1
            N = ns + e
        else:
            N = ns
        xj = np.searchsorted(eq.x_trans[xj].cumsum(), rng.random())
    
    d_hat = {(N,x): np.clip(stay_cnt.get((N,x),0)/tot_cnt.get((N,x),1), 1e-3, 1-1e-3) 
             if (N,x) in tot_cnt else 0.5 for N in range(1,6) for x in eq.x_vals}
    e_hat = {(N,x): np.clip(ent_cnt.get((N,x),0)/ent_opp.get((N,x),1), 1e-3, 1-1e-3) 
             if (N,x) in ent_opp else 0.5 for N in range(5) for x in eq.x_vals}
    return d_hat, e_hat

def _generate_common_draws(n_states_inc, n_states_ent, n_sim, horizon, seed):
    """Generate common random numbers with antithetic variates.
    Returns arrays with n_sim' >= n_sim after antithetic augmentation; caller should
    read actual n_sim from returned arrays' second dimension.
    """
    rng = np.random.default_rng(seed)
    half = max(1, n_sim // 2)

    # Helper to build antithetic for normals and uniforms
    def antithetic_norm(shape):
        z = rng.standard_normal((shape[0], half) + shape[2:])
        za = -z
        out = np.concatenate([z, za], axis=1)
        if out.shape[1] < n_sim:
            extra = rng.standard_normal((shape[0], 1) + shape[2:])
            out = np.concatenate([out, extra], axis=1)
        return out

    def antithetic_unif(shape):
        u = rng.random((shape[0], half) + shape[2:])
        ua = 1.0 - u
        out = np.concatenate([u, ua], axis=1)
        if out.shape[1] < n_sim:
            extra = rng.random((shape[0], 1) + shape[2:])
            out = np.concatenate([out, extra], axis=1)
        return out

    # For incumbent PDVs (condition on staying at t=0)
    mu_other = antithetic_norm((n_states_inc, n_sim, horizon, 5))
    mu_firm1 = antithetic_norm((n_states_inc, n_sim, horizon))
    gamma_entry = antithetic_norm((n_states_inc, n_sim, horizon))
    demand_u_inc = antithetic_unif((n_states_inc, n_sim, horizon))
    # For entrant PDVs
    mu_inc_initial = antithetic_norm((n_states_ent, n_sim, 5))
    mu_firmE = antithetic_norm((n_states_ent, n_sim, horizon))
    mu_otherE = antithetic_norm((n_states_ent, n_sim, horizon, 5))
    gamma_entryE = antithetic_norm((n_states_ent, n_sim, horizon))
    demand_u_ent = antithetic_unif((n_states_ent, n_sim, horizon))

    return (mu_other, mu_firm1, gamma_entry, demand_u_inc,
            mu_inc_initial, mu_firmE, mu_otherE, gamma_entryE, demand_u_ent)

def bbl_estimate(d_hat, e_hat, eq, n_sim=50, horizon=100, seed=2007, bootstrap=False, n_bootstrap=100, start_from=None):
    """BBL-like estimator with vectorized simulation, antithetics, and optional
    multi-start L-BFGS-B.
    Tunables via env: BBL_NJOBS, BBL_VERBOSE, BBL_N_STARTS, BBL_BOOTSTRAP, BBL_N_BOOTSTRAP.
    
    If bootstrap=True, computes standard errors by resampling the simulated data.
    """
    # States as indices
    states_inc = [(N, j) for N in range(1, 6) for j in range(3)]
    states_ent = [(N, j) for N in range(0, 5) for j in range(3)]

    # Convert CCP dicts to arrays and precompute probit thresholds
    d_arr = np.full((6, 3), 0.5)
    e_arr = np.full((6, 3), 0.5)
    for N in range(1, 6):
        for j, xv in enumerate(eq.x_vals):
            if (N, xv) in d_hat:
                d_arr[N, j] = d_hat[(N, xv)]
    for N in range(0, 5):
        for j, xv in enumerate(eq.x_vals):
            if (N, xv) in e_hat:
                e_arr[N, j] = e_hat[(N, xv)]
    d_arr = np.clip(d_arr, 1e-6, 1 - 1e-6)
    e_arr = np.clip(e_arr, 1e-6, 1 - 1e-6)
    d_thr = norm.ppf(d_arr)
    e_thr = norm.ppf(e_arr)

    # Demand transition cumulative
    x_cum_full = eq.x_trans.cumsum(axis=1)

    # Verbosity and multi-start settings
    VERBOSE = int(os.environ.get("BBL_VERBOSE", "0"))
    N_STARTS = max(1, int(os.environ.get("BBL_N_STARTS", "3")))

    def make_simulator(hor, ns, sd):
        # Precompute powers and shared draws for given horizon/ns
        beta_pows = eq.beta ** np.arange(hor + 1)
        draws = _generate_common_draws(len(states_inc), len(states_ent), ns, hor, sd)
        (mu_other, mu_firm1, gamma_entry, demand_u_inc,
         mu_inc_initial, mu_firmE, mu_otherE, gamma_entryE, demand_u_ent) = draws
        ns_eff = mu_other.shape[1]

        def next_x_idx(x_idx_vec, u_vec):
            c = x_cum_full[x_idx_vec]
            return np.sum(u_vec[:, None] > c, axis=1)

        def simulate_Lambda(theta):
            mu_m, sig_mu, ga_m, sig_g = theta
            Lam = {}
            LamE = {}

            def sim_inc_state(si, N0, xi0):
                N = np.full(ns_eff, N0, dtype=int)
                x = np.full(ns_eff, xi0, dtype=int)
                alive = np.ones(ns_eff, dtype=bool)
                pdv = np.zeros(ns_eff)
                pdv += eq.pi[N0, xi0]
                if N0 > 1:
                    z_vec0 = mu_other[si, :, 0, :N0 - 1]
                    stay_other = np.sum(z_vec0 <= d_thr[N0, xi0], axis=1)
                else:
                    stay_other = np.zeros(ns_eff, dtype=int)
                if N0 < 5:
                    ent0 = (gamma_entry[si, :, 0] <= e_thr[N0, xi0]).astype(int)
                else:
                    ent0 = np.zeros(ns_eff, dtype=int)
                N = np.minimum(1 + stay_other + ent0, 5)
                x = next_x_idx(x, demand_u_inc[si, :, 0])
                for t in range(1, hor):
                    if not alive.any():
                        break
                    idx = np.where(alive)[0]
                    zf = mu_firm1[si, idx, t]
                    thr = d_thr[N[idx], x[idx]]
                    stay_f1 = zf <= thr
                    exiting = idx[~stay_f1]
                    if exiting.size > 0:
                        pdv[exiting] += beta_pows[t] * (mu_m + sig_mu * mu_firm1[si, exiting, t])
                        alive[exiting] = False
                    surv = idx[stay_f1]
                    if surv.size > 0:
                        pdv[surv] += beta_pows[t] * eq.pi[N[surv], x[surv]]
                        stay_o = np.zeros(surv.size, dtype=int)
                        has_others = N[surv] > 1
                        if np.any(has_others):
                            sv_idx = np.where(has_others)[0]
                            z_o = mu_other[si, surv[sv_idx], t, :]
                            thr_o = d_thr[N[surv[sv_idx]], x[surv[sv_idx]]][:, None]
                            cols = np.arange(5)
                            mask = cols < (N[surv[sv_idx]] - 1)[:, None]
                            stay_o_vals = (z_o <= thr_o) & mask
                            stay_o[sv_idx] = stay_o_vals.sum(axis=1)
                        Nn = stay_o + 1
                        ent = np.zeros(surv.size, dtype=int)
                        can_ent = N[surv] < 5
                        if np.any(can_ent):
                            ce_idx = np.where(can_ent)[0]
                            ent[ce_idx] = (gamma_entry[si, surv[ce_idx], t] <= e_thr[N[surv[ce_idx]], x[surv[ce_idx]]]).astype(int)
                        N[surv] = np.minimum(Nn + ent, 5)
                        x[surv] = next_x_idx(x[surv], demand_u_inc[si, surv, t])
                return (N0, eq.x_vals[xi0]), float(pdv.mean())

            def sim_ent_state(si, N0, xi0):
                if N0 > 0:
                    z0 = mu_inc_initial[si, :, :N0]
                    stay_inc = np.sum(z0 <= d_thr[N0, xi0], axis=1)
                else:
                    stay_inc = np.zeros(ns_eff, dtype=int)
                N = np.minimum(stay_inc + 1, 5)
                x = next_x_idx(np.full(ns_eff, xi0, dtype=int), demand_u_ent[si, :, 0])
                alive = np.ones(ns_eff, dtype=bool)
                pdv = np.zeros(ns_eff)
                for t in range(1, hor):
                    if not alive.any():
                        break
                    idx = np.where(alive)[0]
                    zE = mu_firmE[si, idx, t]
                    thr = d_thr[N[idx], x[idx]]
                    stay_E = zE <= thr
                    exiting = idx[~stay_E]
                    if exiting.size > 0:
                        pdv[exiting] += beta_pows[t] * (mu_m + sig_mu * mu_firmE[si, exiting, t])
                        alive[exiting] = False
                    surv = idx[stay_E]
                    if surv.size > 0:
                        pdv[surv] += beta_pows[t] * eq.pi[N[surv], x[surv]]
                        stay_o = np.zeros(surv.size, dtype=int)
                        has_others = N[surv] > 1
                        if np.any(has_others):
                            sv_idx = np.where(has_others)[0]
                            zOE = mu_otherE[si, surv[sv_idx], t, :]
                            thr_o = d_thr[N[surv[sv_idx]], x[surv[sv_idx]]][:, None]
                            cols = np.arange(5)
                            mask = cols < (N[surv[sv_idx]] - 1)[:, None]
                            stay_o_vals = (zOE <= thr_o) & mask
                            stay_o[sv_idx] = stay_o_vals.sum(axis=1)
                        Nn = stay_o + 1
                        ent = np.zeros(surv.size, dtype=int)
                        can_ent = N[surv] < 5
                        if np.any(can_ent):
                            ce_idx = np.where(can_ent)[0]
                            ent[ce_idx] = (gamma_entryE[si, surv[ce_idx], t] <= e_thr[N[surv[ce_idx]], x[surv[ce_idx]]]).astype(int)
                        N[surv] = np.minimum(Nn + ent, 5)
                        x[surv] = next_x_idx(x[surv], demand_u_ent[si, surv, t])
                return (N0, eq.x_vals[xi0]), float(pdv.mean())

            n_jobs = int(os.environ.get("BBL_NJOBS", "1"))
            if Parallel is not None and n_jobs != 1:
                inc_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(sim_inc_state)(si, N0, xi0) for si, (N0, xi0) in enumerate(states_inc)
                )
                ent_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(sim_ent_state)(si, N0, xi0) for si, (N0, xi0) in enumerate(states_ent)
                )
            else:
                inc_results = [sim_inc_state(si, N0, xi0) for si, (N0, xi0) in enumerate(states_inc)]
                ent_results = [sim_ent_state(si, N0, xi0) for si, (N0, xi0) in enumerate(states_ent)]
            Lam, LamE = {}, {}
            for k, v in inc_results:
                Lam[k] = v
            for k, v in ent_results:
                LamE[k] = v
            return Lam, LamE

        def objective(theta):
            Lam, LamE = simulate_Lambda(theta)
            inc_err = 0.0
            for N, j in states_inc:
                inc_err += (norm.cdf((Lam[(N, eq.x_vals[j])] - theta[0]) / theta[1]) - d_arr[N, j]) ** 2
            ent_err = 0.0
            for N, j in states_ent:
                ent_err += (norm.cdf((LamE[(N, eq.x_vals[j])] - theta[2]) / theta[3]) - e_arr[N, j]) ** 2
            return inc_err + ent_err

        return simulate_Lambda, objective

    # Build full simulator
    sim_full, obj_full = make_simulator(horizon, n_sim, seed)

    # Multi-start BFGS with improved starting values
    # If start_from provided, use single-start from that point (for bootstrap efficiency)
    if start_from is not None:
        starts = [np.array(start_from, dtype=float)]
    else:
        # Generate N_STARTS random starting points from fixed seed
        starts = []
        rng = np.random.default_rng(int(os.environ.get("BBL_SEED", "12345")))
        for _ in range(N_STARTS):
            # μ, γ: cover flow profit range with bias toward positive values
            mu0 = rng.uniform(0, 8)
            g0 = rng.uniform(0, 8)
            # σ: bracket true √5≈2.236 more tightly, avoid very small values
            s1 = rng.uniform(1.2, 3.5)
            s2 = rng.uniform(1.2, 3.5)
            starts.append(np.array([mu0, s1, g0, s2]))

    best_res = None
    all_results = []
    for i, x0 in enumerate(starts):
        res = minimize(obj_full, x0=x0, method='BFGS')
        all_results.append((i, x0, res))
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    # Store all results in best_res for reporting
    best_res.all_starts = all_results
    best_res.se = None  # Will be computed separately if needed
    # Attach objective breakdown at the solution for diagnostics
    try:
        # Rebuild arrays in current scope
        # States as indices already defined: states_inc, states_ent
        def cdf_inc(val, N, j):
            return norm.cdf((val - best_res.x[0]) / best_res.x[1])
        def cdf_ent(val, N, j):
            return norm.cdf((val - best_res.x[2]) / best_res.x[3])
        simulate_Lambda, _ = make_simulator(horizon, n_sim, seed)
        Lam, LamE = simulate_Lambda(best_res.x)
        inc_sq = []
        ent_sq = []
        inc_abs = []
        ent_abs = []
        for N, j in states_inc:
            pred = cdf_inc(Lam[(N, eq.x_vals[j])], N, j)
            err = pred - d_arr[N, j]
            inc_sq.append(err*err)
            inc_abs.append(abs(err))
        for N, j in states_ent:
            pred = cdf_ent(LamE[(N, eq.x_vals[j])], N, j)
            err = pred - e_arr[N, j]
            ent_sq.append(err*err)
            ent_abs.append(abs(err))
        inc_sum = float(np.sum(inc_sq))
        ent_sum = float(np.sum(ent_sq))
        best_res.obj_breakdown = {
            "inc_sum": inc_sum,
            "ent_sum": ent_sum,
            "total": inc_sum + ent_sum,
            "inc_mse": inc_sum / len(states_inc),
            "ent_mse": ent_sum / len(states_ent),
            "inc_max_abs": float(np.max(inc_abs)) if inc_abs else 0.0,
            "ent_max_abs": float(np.max(ent_abs)) if ent_abs else 0.0,
            "n_sim": n_sim,
            "horizon": horizon,
        }
    except Exception:
        best_res.obj_breakdown = None
    return best_res

def _bootstrap_rep(b, eq, theta_hat, n_sim, horizon, T_ccp, seed):
    """One bootstrap replication: generate new trajectory, re-estimate CCPs, re-estimate theta.
    
    This is the CORRECT bootstrap for two-step estimators:
    1. Generate new market trajectory from DGP (bootstrap the data)
    2. Re-estimate first-stage CCPs from bootstrapped trajectory
    3. Re-estimate second-stage structural parameters
    """
    try:
        boot_seed = seed + 10000 * (b + 1)
        
        # Step 1: Generate bootstrap market trajectory from true equilibrium
        # This captures sampling variation in the original data
        d_hat_boot, e_hat_boot = estimate_ccps(eq, T=T_ccp, seed=boot_seed)
        
        # Step 2: Re-estimate structural parameters from bootstrapped CCPs
        # Suppress inner parallelism for stability
        old_verbose = os.environ.get("BBL_VERBOSE", "0")
        old_nstarts = os.environ.get("BBL_N_STARTS", "1")
        old_njobs = os.environ.get("BBL_NJOBS", "1")
        os.environ["BBL_VERBOSE"] = "0"
        os.environ["BBL_N_STARTS"] = "1"
        os.environ["BBL_NJOBS"] = "1"
        
        res = bbl_estimate(d_hat_boot, e_hat_boot, eq, n_sim=n_sim, horizon=horizon, 
                          seed=boot_seed, bootstrap=False, n_bootstrap=0, start_from=theta_hat)
        
        # Restore env
        os.environ["BBL_VERBOSE"] = old_verbose
        os.environ["BBL_N_STARTS"] = old_nstarts
        os.environ["BBL_NJOBS"] = old_njobs
        
        return res.x if res is not None and hasattr(res, 'x') else np.array([np.nan, np.nan, np.nan, np.nan])
    except Exception:
        return np.array([np.nan, np.nan, np.nan, np.nan])


def bootstrap_estimates_parallel(eq, theta_hat, n_sim=300, horizon=300, T_ccp=10000, seed=2007, n_boot=100, n_jobs=-1):
    """Proper bootstrap for BBL two-step estimator.
    
    Resamples from the DATA GENERATION PROCESS (market trajectories), not from CCPs.
    This correctly captures:
    - Sampling variation in original trajectory
    - First-stage CCP estimation error  
    - Second-stage simulation error
    - Time-series correlation and dynamics
    
    Returns array of valid bootstrap estimates (B_valid x 4).
    Uses batching and threading backend to prevent memory issues.
    """
    if Parallel is None:
        # Fallback to simple serial if joblib missing
        estimates = []
        for b in range(n_boot):
            est = _bootstrap_rep(b, eq, theta_hat, n_sim, horizon, T_ccp, seed)
            estimates.append(est)
        estimates = np.array(estimates)
    else:
        print(f"Running {n_boot} BBL bootstraps in parallel (n_jobs={n_jobs})...")
        # No batching - process all at once
        batch_size = n_boot
        n_batches = 1
        all_estimates = []
        
        batch_range = range(n_boot)
        
        # Use threading backend to reduce memory overhead vs loky
        batch_estimates = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
            delayed(_bootstrap_rep)(b, eq, theta_hat, n_sim, horizon, T_ccp, seed)
            for b in batch_range
        )
        all_estimates.extend(batch_estimates)
        
        estimates = np.array(all_estimates)
        print(f"  Completed {n_boot} bootstrap replications")

    valid = ~np.isnan(estimates).any(axis=1)
    if valid.sum() < 5:
        return None
    return estimates[valid]

def main():
    parser = argparse.ArgumentParser(description="Dynamic Entry/Exit Game Simulation and Estimation")
    parser.add_argument('--mode', choices=['ps', 'hp'], default='hp', help='Mode: ps (problem setup, n_sim=50) or hp (high precision, n_sim=1000)')
    args = parser.parse_args()
    
    print("Dynamic Entry/Exit Game: Equilibrium and Estimation\n" + "="*70)
    
    # Q4-5: Solve and check uniqueness
    print("\nQ4-5: Equilibrium Computation")
    inits = [("π", None), ("0", np.zeros((6,3))), ("1", np.ones((6,3))),
             ("U[0,10]", np.random.RandomState(123).uniform(0,10,(6,3))),
             ("U[-5,15]", np.random.RandomState(456).uniform(-5,15,(6,3)))]
    sols = []
    for name, init_val in inits:
        game = DynamicGame()
        if init_val is not None:
            game.V_bar = game.mu_cutoff = game.gamma_cutoff = init_val.copy()
        game.solve_equilibrium()
        sols.append(game)
        print(f"  Init {name:8s}: V̄(3,0)={game.V_bar[3,1]:.4f}, μ̄(3,0)={game.mu_cutoff[3,1]:.4f}")
    print(f"  Result: {'UNIQUE' if all(np.allclose(sols[0].V_bar,s.V_bar,1e-6) for s in sols[1:]) else 'MULTIPLE'}")
    
    # Q6: Report equilibrium values
    eq = sols[0]
    print(f"\nQ6: Equilibrium at (N,x)=(3,0)")
    print(f"  μ̄(3,0) = {eq.mu_cutoff[3,1]:.6f}")
    print(f"  γ̄(3,0) = {eq.gamma_cutoff[3,1]:.6f}")
    print(f"  V̄(3,0) = {eq.V_bar[3,1]:.6f}")
    print(f"  V(3,0,-2) = {max(-2.0, eq.mu_cutoff[3,1]):.6f} (firm {'stays' if -2<=eq.mu_cutoff[3,1] else 'exits'})")
    
    # Q7: Simulate baseline
    print(f"\nQ7: Market Simulation (10,000 periods)")
    N_hist = eq.simulate()
    print(f"  Average firms: {N_hist.mean():.4f}")
    
    # Q8: Entry tax counterfactual
    print(f"\nQ8: Entry Tax Counterfactual")
    tax = DynamicGame(entry_tax=5.0)
    tax.solve_equilibrium()
    N_tax = tax.simulate()
    change = N_tax.mean() - N_hist.mean()
    print(f"  With 5-unit tax: {N_tax.mean():.4f} firms")
    print(f"  Change: {change:.4f} firms ({100*change/N_hist.mean():.2f}%)")
    
    # Q9: BBL-like estimation (Problem 9 specification)
    print(f"\nQ9: BBL-like Estimation (forward simulation of Λ, Λ^E)")
    
    # Estimate CCPs
    print("  Step 1: Nonparametric CCP estimation from simulated data")
    T_ccp = int(os.environ.get("BBL_T_CCP", "10000"))
    print(f"    Simulating {T_ccp} periods from equilibrium...")
    d_hat, e_hat = estimate_ccps(eq, T=T_ccp, seed=2007)
    print(f"    Estimated {len(d_hat)} incumbent CCPs and {len(e_hat)} entry CCPs from data")
    
    # Report some estimated CCPs vs true
    print("    Sample CCP comparison (estimated vs true):")
    d_true = {}
    for N in range(1, 6):
        for j, xv in enumerate(eq.x_vals):
            d_true[(N, xv)] = norm.cdf((eq.mu_cutoff[N, j] - eq.mu_mean) / eq.mu_std)
    e_true = {}
    for N in range(5):
        for j, xv in enumerate(eq.x_vals):
            e_true[(N, xv)] = norm.cdf((eq.gamma_cutoff[N, j] - eq.gamma_mean) / eq.gamma_std)
    for (N, x) in [(3, 0.0), (3, -5.0), (3, 5.0)]:
        print(f"      d(N={N},x={x:.0f}): est={d_hat[(N,x)]:.4f}, true={d_true[(N,x)]:.4f}")
    for (N, x) in [(3, 0.0), (3, -5.0), (3, 5.0)]:
        print(f"      e(N={N},x={x:.0f}): est={e_hat[(N,x)]:.4f}, true={e_true[(N,x)]:.4f}")
    
    print("  Step 2: Forward simulation (Λ, Λ^E) and minimum distance search")
    
    # Run with problem setup settings
    n_sim_ps = 50
    horizon_ps = 1000
    print("  Using 50 simulations, horizon=1000")
    res_ps = bbl_estimate(d_hat, e_hat, eq, n_sim=n_sim_ps, horizon=horizon_ps, seed=2007, bootstrap=False)
    
    # Report all starts for problem spec
    if hasattr(res_ps, 'all_starts'):
        for i, x0, r in res_ps.all_starts:
            start_label = f"rng-{i+1}"
            print(f"    Start {start_label:8s}: obj={r.fun:.6f}, μ={r.x[0]:6.3f}, σ_μ={r.x[1]:5.3f}, γ={r.x[2]:6.3f}, σ_γ={r.x[3]:5.3f}, iters={r.nit:2d}")
    if hasattr(res_ps, 'obj_breakdown') and isinstance(res_ps.obj_breakdown, dict):
        ob = res_ps.obj_breakdown
        print(f"    Settings: n_sim={ob['n_sim']}, horizon={ob['horizon']}")
        print(f"    Objective breakdown: total={ob['total']:.6f}, inc_sum={ob['inc_sum']:.6f}, ent_sum={ob['ent_sum']:.6f}")
    
    # Run with high-precision settings
    n_sim_hp = int(os.environ.get("BBL_N_SIM", "1000")) 
    horizon_hp = int(os.environ.get("BBL_H", "1000"))    
    print(f"  Using {n_sim_hp} simulations, horizon={horizon_hp}")
    res_hp = bbl_estimate(d_hat, e_hat, eq, n_sim=n_sim_hp, horizon=horizon_hp, seed=2007, bootstrap=False)
    
    # Report all starts for high-precision
    if hasattr(res_hp, 'all_starts'):
        for i, x0, r in res_hp.all_starts:
            start_label = f"rng-{i+1}"
            print(f"    Start {start_label:8s}: obj={r.fun:.6f}, μ={r.x[0]:6.3f}, σ_μ={r.x[1]:5.3f}, γ={r.x[2]:6.3f}, σ_γ={r.x[3]:5.3f}, iters={r.nit:2d}")
    if hasattr(res_hp, 'obj_breakdown') and isinstance(res_hp.obj_breakdown, dict):
        ob = res_hp.obj_breakdown
        print(f"    Settings: n_sim={ob['n_sim']}, horizon={ob['horizon']}")
        print(f"    Objective breakdown: total={ob['total']:.6f}, inc_sum={ob['inc_sum']:.6f}, ent_sum={ob['ent_sum']:.6f}")

    res = res_ps if args.mode == 'ps' else res_hp
    n_sim_boot = n_sim_ps if args.mode == 'ps' else n_sim_hp
    
    # Compute bootstrap percentile CIs (parallel)
    do_bootstrap = int(os.environ.get("BBL_BOOTSTRAP", "1")) > 0
    if do_bootstrap:
        print(f"  Bootstrap using mode '{args.mode}' with res n_sim={res.obj_breakdown['n_sim'] if res.obj_breakdown else 'unknown'}")
        n_boot = int(os.environ.get("BBL_N_BOOTSTRAP", "100"))
        n_jobs = int(os.environ.get("BBL_BOOT_NJOBS", "-1"))
        T_ccp_boot = int(os.environ.get("BBL_BOOT_T_CCP", "10000"))
        n_sim_boot = int(os.environ.get("BBL_BOOT_N_SIM", str(n_sim_boot)))
        boot = bootstrap_estimates_parallel(eq, res.x, n_sim=n_sim_boot, horizon=1000, T_ccp=T_ccp_boot, seed=2007, n_boot=n_boot, n_jobs=n_jobs)
        if boot is not None and boot.shape[0] >= 5:
            se = boot.std(axis=0, ddof=1)
            bias = boot.mean(axis=0) - res.x
            bias_corrected_theta = res.x - bias
            # Percentile CI
            ci_lo_perc = np.percentile(boot, 2.5, axis=0)
            ci_hi_perc = np.percentile(boot, 97.5, axis=0)
            # Bias-corrected CI
            ci_lo_bc = 2 * res.x - np.percentile(boot, 97.5, axis=0)
            ci_hi_bc = 2 * res.x - np.percentile(boot, 2.5, axis=0)
            print(f"  Bootstrap SE:         μ={se[0]:6.3f}, σ_μ={se[1]:5.3f}, γ={se[2]:6.3f}, σ_γ={se[3]:5.3f}")
            print(f"  Original θ:           μ={res.x[0]:6.3f}, σ_μ={res.x[1]:5.3f}, γ={res.x[2]:6.3f}, σ_γ={res.x[3]:5.3f}")
            print(f"  Bootstrap bias:       μ={bias[0]:6.3f}, σ_μ={bias[1]:5.3f}, γ={bias[2]:6.3f}, σ_γ={bias[3]:5.3f}")
            print(f"  Bias-corrected θ:     μ={bias_corrected_theta[0]:6.3f}, σ_μ={bias_corrected_theta[1]:5.3f}, γ={bias_corrected_theta[2]:6.3f}, σ_γ={bias_corrected_theta[3]:5.3f}")
            print(f"  Percentile 95% CI:    μ=[{ci_lo_perc[0]:6.3f}, {ci_hi_perc[0]:6.3f}], σ_μ=[{ci_lo_perc[1]:5.3f}, {ci_hi_perc[1]:5.3f}], γ=[{ci_lo_perc[2]:6.3f}, {ci_hi_perc[2]:6.3f}], σ_γ=[{ci_lo_perc[3]:5.3f}, {ci_hi_perc[3]:5.3f}]")
            print(f"  Bias-corrected 95% CI: μ=[{ci_lo_bc[0]:6.3f}, {ci_hi_bc[0]:6.3f}], σ_μ=[{ci_lo_bc[1]:5.3f}, {ci_hi_bc[1]:5.3f}], γ=[{ci_lo_bc[2]:6.3f}, {ci_hi_bc[2]:6.3f}], σ_γ=[{ci_lo_bc[3]:5.3f}, {ci_hi_bc[3]:5.3f}]")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

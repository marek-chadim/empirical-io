import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import scipy.stats as stats
import statsmodels.api as sm
import IPython.display
IPython.display.display(IPython.display.HTML('<style>pre { white-space: pre !important; }</style>'))
import pyblp
pyblp.options.digits = 3
pd.options.display.precision = 3
pd.options.display.max_columns = 50
pyblp.__version__

np.random.seed(1995)

# Model parameters
T, J = 600, 4
alpha, beta1 = -2, 1
beta2, beta3 = 4, 4
sigma_satellite, sigma_wired = 1, 1
gamma0, gamma1 = 0.5, 0.25

# Product data structure
data = [
{'market_ids': t, 'firm_ids': j+1, 'product_ids': j}
for t in range(T)
for j in range(J)
]
product_data = pd.DataFrame(data)

# Exogenous variables: x_jt and w_jt as absolute values of iid standard normal draws
product_data['x'] = np.abs(
np.random.normal(0, 1, len(product_data))
)
product_data['w'] = np.abs(
np.random.normal(0, 1, len(product_data))
)

# Indicators
product_data['satellite'] = (
product_data['firm_ids'].isin([1, 2]).astype(int)
)
product_data['wired'] = (
product_data['firm_ids'].isin([3, 4]).astype(int)
)

# Unobservables: ξ_jt and ω_jt with covariance matrix [[1, 0.25], [0.25, 1]]
cov_matrix = np.array([[1, 0.25], [0.25, 1]])
A = np.linalg.cholesky(cov_matrix)
z = np.random.normal(0, 1, (len(product_data), 2))
unobs = z @ A.T
product_data['xi'] = unobs[:, 0]  # demand unobservable
product_data['omega'] = unobs[:, 1]  # cost unobservable

print("Question 1 completed:")
print(f"Generated {len(product_data)} observations across {T} markets")
print(f'x range: {product_data["x"].min():.3f} to {product_data["x"].max():.3f}')
print(f'w range: {product_data["w"].min():.3f} to {product_data["w"].max():.3f}')
xi_omega_corr = product_data[['xi', 'omega']].corr().iloc[0,1]
print(f"ξ-ω correlation: {xi_omega_corr:.3f} (target: 0.25)")
sat_count = product_data["satellite"].sum()
wired_count = product_data["wired"].sum()
print(f"Satellite products: {sat_count}, Wired products: {wired_count}")

def market_shares_and_derivatives(prices, market_data, nu_draws):
    """
    Compute shares, derivatives, and inside_shares_draws efficiently in one pass.
    Returns: (shares, derivatives, inside_shares_draws)
    """
    J = len(market_data)
    x = market_data['x'].values
    xi = market_data['xi'].values
    sat = market_data['satellite'].values
    wired = market_data['wired'].values

    # Compute utilities once
    utilities = (
        beta1 * x + xi +
        nu_draws[:, 0:1] * sat +
        nu_draws[:, 1:2] * wired +
        alpha * prices
    )
    utilities = np.column_stack([utilities, np.zeros(nu_draws.shape[0])])
    exp_u = np.exp(utilities - np.max(utilities, axis=1, keepdims=True))
    choice_probs = exp_u / exp_u.sum(axis=1, keepdims=True)
    inside_shares_draws = choice_probs[:, :J]

    # Shares: average over draws
    shares = np.mean(inside_shares_draws, axis=0)

    # Derivatives: compute analytically from choice probabilities
    derivatives = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            indicator = float(j == k)
            deriv_draws = (
                alpha * inside_shares_draws[:, j] *
                (indicator - inside_shares_draws[:, k])
            )
            derivatives[j, k] = np.mean(deriv_draws)

    return shares, derivatives, inside_shares_draws

# Pre-draw simulation draws (to avoid jittering)
np.random.seed(1995)
n_draws = 10000
all_nu_draws = [
    np.random.multivariate_normal(
        [beta2, beta3],
        np.diag([sigma_satellite, sigma_wired]),
        size=n_draws
    )
    for _ in range(T)
]

def test_convergence(prices, market_data, nu_draws_full, draw_counts, n_reps=100):
    """Test derivative stability across different numbers of simulation draws."""
    np.random.seed(1995)
    stds = []
    n_available = len(nu_draws_full)

    for n_draws in draw_counts:
        deriv_list = []
        for rep in range(n_reps):
            # Randomly sample n_draws from the pre-drawn samples
            indices = np.random.choice(n_available, size=n_draws, replace=False)
            nu_draws = nu_draws_full[indices]
            _, derivs, _ = market_shares_and_derivatives(
                prices, market_data, nu_draws
            )
            deriv_list.append(derivs)
        stds.append(np.std(deriv_list, axis=0).mean())
    return np.array(stds)

# Test at initial prices (p = MC)
product_data['mc'] = np.exp(
    gamma0 + gamma1 * product_data['w'] + product_data['omega'] / 8
)
# Test at initial prices (p = MC) for market 0
market_0 = product_data[product_data['market_ids'] == 0]
prices_init = market_0['mc'].values
draw_counts = [50, 100, 200, 500, 1000, 2000, 5000]
stds = test_convergence(prices_init, market_0, all_nu_draws[0], draw_counts)
stds

print(f"MC range: {product_data['mc'].min():.3f} to {product_data['mc'].max():.3f}")
print(f"MC mean: {product_data['mc'].mean():.3f}, median: {product_data['mc'].median():.3f}")
print("FOC: (p_jt - mc_jt) * ∂s_jt/∂p_jt + s_jt = 0")
print("Rearranged: p_jt - mc_jt = - (∂s_jt/∂p_jt)⁻¹ * s_jt")

def solve_prices_direct(market_data, mc_market, nu_draws):
    """Solve for equilibrium prices using direct nonlinear solver with robust matrix inversion"""
    J = len(market_data)

    def foc_residual(prices):
        """FOC residuals: p - mc + (∂s/∂p)^{-1} s = 0"""
        # Compute shares and derivatives at current prices
        shares, derivatives, _ = market_shares_and_derivatives(
            prices, market_data, nu_draws
        )

        # Inversion of derivative matrix
        invD = np.linalg.inv(derivatives)

        # FOC residuals: p - mc + inv(∂s/∂p) @ s
        residuals = prices - mc_market + invD @ shares
        return residuals
    # Initial guess: marginal costs
    p0 = mc_market.copy()
    # Solve using root finder (hybr method)
    sol = opt.root(foc_residual, p0, method='hybr', tol=1e-8)
    prices_sol = sol.x
    success = sol.success
    # Additional check: verify that residuals are small
    final_residuals = foc_residual(prices_sol)
    if np.max(np.abs(final_residuals)) > 1e-6:
        success = False
    return prices_sol, success

# Solve using direct method
equilibrium_prices_direct = []
success_flags_direct = []
for t in range(T):
    market_data = product_data[product_data['market_ids'] == t]
    mc_market = market_data['mc'].values
    nu_draws = all_nu_draws[t]
    prices_direct, success = solve_prices_direct(
        market_data, mc_market, nu_draws
    )
    equilibrium_prices_direct.append(prices_direct)
    success_flags_direct.append(success)
equilibrium_prices_direct = np.array(equilibrium_prices_direct)
success_count = sum(success_flags_direct)
print("Question 2(c)i completed:")
print(f"Direct nonlinear solver (root): {success_count}/{T} markets solved successfully")
print(f"Success rate: {success_count/T:.1%}")
price_range_text = (
    f"Price range: {equilibrium_prices_direct.min():.3f} to "
    f"{equilibrium_prices_direct.max():.3f}"
)
print(price_range_text)
price_stats_text = (
    f"Price mean: {equilibrium_prices_direct.mean():.3f}, "
    f"std: {equilibrium_prices_direct.std():.3f}"
)
print(price_stats_text)

def solve_prices_morrow_skerlos(market_data, mc_market, nu_draws, max_iter=100, tol=1e-6):
    """Morrow-Skerlos algorithm"""
    prices = mc_market.copy()
    for iteration in range(max_iter):
        # Efficiently compute shares, derivatives, and inside_shares_draws in one pass
        shares, derivatives, inside_shares_draws = market_shares_and_derivatives(prices, market_data, nu_draws)

        Lambda = np.diag(alpha * shares)
        Gamma = alpha * (inside_shares_draws.T @ inside_shares_draws) / nu_draws.shape[0]
        diff = prices - mc_market
        zeta = np.linalg.solve(Lambda, Gamma.T @ diff - shares)
        prices_new = mc_market + zeta
        foc_residual = Lambda @ (prices - mc_market - zeta)
        if np.max(np.abs(foc_residual)) < tol:
            break
        prices = 0.5 * prices + 0.5 * prices_new
    return prices, iteration + 1

# Solve using Morrow-Skerlos method
equilibrium_prices_ms = []
iterations_ms = []

for t in range(T):
    market_data = product_data[product_data['market_ids'] == t]
    mc_market = market_data['mc'].values
    nu_draws = all_nu_draws[t]

    prices_ms, iters = solve_prices_morrow_skerlos(market_data, mc_market, nu_draws)
    equilibrium_prices_ms.append(prices_ms)
    iterations_ms.append(iters)

equilibrium_prices_ms = np.array(equilibrium_prices_ms)
print("Question 2(c)ii completed:")
print(f"Morrow-Skerlos method: {T} markets solved")
print(f"Average iterations: {np.mean(iterations_ms):.1f}")
print(f"Max iterations: {np.max(iterations_ms)}")
print(f"Price range: {equilibrium_prices_ms.min():.3f} to {equilibrium_prices_ms.max():.3f}")
print(f"Price mean: {equilibrium_prices_ms.mean():.3f}, std: {equilibrium_prices_ms.std():.3f}")

# Compare direct vs Morrow-Skerlos if direct succeeded for all
if len(equilibrium_prices_direct) == T:
    price_diff = np.abs(np.array(equilibrium_prices_direct) - equilibrium_prices_ms)
    print(f"Max price difference between methods: {price_diff.max():.2e}")
    print(f"Mean price difference: {price_diff.mean():.2e}")
else:
    print("Direct method failed for some markets, skipping comparison.")
    print("Preferred method: Morrow-Skerlos, as it is more numerically stable.")

# Use Morrow-Skerlos prices
product_data['prices'] = equilibrium_prices_ms.flatten()

# Compare derivative convergence at initial vs equilibrium prices
market_0 = product_data[product_data['market_ids'] == 0]
prices_equilibrium = market_0['prices'].values

draw_counts = [50, 100, 200, 500, 1000, 2000, 5000]
# Reuse previously calculated initial_stds from test_convergence
initial_stds = stds  # Already calculated earlier at initial prices

# Only compute equilibrium stds
np.random.seed(1995)
n_available = len(all_nu_draws[0])
n_reps = 100
eq_stds = []
for n_draws in draw_counts:
    deriv_list = []
    for _ in range(n_reps):
        indices = np.random.choice(n_available, size=n_draws, replace=False)
        nu_draws = all_nu_draws[0][indices]
        _, derivatives, _ = market_shares_and_derivatives(
            prices_equilibrium, market_0, nu_draws
        )
        deriv_list.append(derivatives)
    eq_stds.append(np.std(deriv_list, axis=0).mean())
eq_stds = np.array(eq_stds)

print("Comparing derivative approximation convergence:")
print("Draws\t| Initial Std Dev\t| Equilibrium Std Dev\t| Ratio (Eq/Init)")
print("-" * 75)

for i, n_draws in enumerate(draw_counts):
    ratio = eq_stds[i] / initial_stds[i] if initial_stds[i] > 0 else float('inf')
    print(f"{n_draws:6d}\t| {initial_stds[i]:.2e}\t\t| {eq_stds[i]:.2e}\t\t| {ratio:.2f}")

# Compute shares
observed_shares = []
for t in range(T):
    market_data = product_data[product_data['market_ids'] == t]
    prices_market = market_data['prices'].values
    # Use pre-drawn simulation draws for this market
    shares_market, _, _ = market_shares_and_derivatives(
        prices_market, market_data, all_nu_draws[t]
    )
    observed_shares.extend(shares_market)

product_data['shares'] = observed_shares
print(f"Share range: {product_data['shares'].min():.3f} to {product_data['shares'].max():.3f}")
print(f"Share mean: {product_data['shares'].mean():.3f}, std: {product_data['shares'].std():.3f}")

# Validation: Check market share sums
market_share_sums = product_data.groupby('market_ids')['shares'].sum()
print(f"Market share sums (should be < 1):")
print(f"Average: {market_share_sums.mean():.3f}")
print(f"Min: {market_share_sums.min():.3f}, Max: {market_share_sums.max():.3f}")
print(f"Outside shares: {1 - market_share_sums.mean():.3f} (average)")

# Check by product type
satellite_shares = product_data[product_data['satellite'] == 1]['shares'].mean()
wired_shares = product_data[product_data['wired'] == 1]['shares'].mean()
print(f"Average satellite product share: {satellite_shares:.3f}")
print(f"Average wired product share: {wired_shares:.3f}")

# Create ln_within_share
product_data["group_share"] = product_data.groupby(["market_ids", "satellite"])["shares"].transform("sum")
product_data["ln_within_share"] = np.log(product_data["shares"] / product_data["group_share"])

# Create nest-specific ln_within_share
product_data["ln_within_share_sat"] = product_data["ln_within_share"] * product_data["satellite"]
product_data["ln_within_share_wired"] = product_data["ln_within_share"] * product_data["wired"]

# Create quadratic and interaction columns first
product_data['x**2'] = product_data['x'] ** 2
product_data['w**2'] = product_data['w'] ** 2
product_data['x*w'] = product_data['x'] * product_data['w']

# sum over competing goods in market t
product_data['sum_x_competitors'] = (
    product_data.groupby('market_ids')['x'].transform('sum') -
    product_data['x']
)
product_data['sum_w_competitors'] = (
    product_data.groupby('market_ids')['w'].transform('sum') -
    product_data['w']
)

# index of the other good in the same nest
product_data['x_other_in_nest'] = (
    product_data.groupby(['market_ids', 'satellite'])['x'].transform('sum') -
    product_data['x']
)
product_data['w_other_in_nest'] = (
    product_data.groupby(['market_ids', 'satellite'])['w'].transform('sum') -
    product_data['w']
)

# Use satellite and wired dummies instead of constant
Z = product_data[[
    'satellite', 'wired', 'x', 'w', 'x**2', 'w**2', 'x*w',
    'sum_x_competitors', 'sum_w_competitors', 'x_other_in_nest', 'w_other_in_nest'
]]

# Regression 1: Prices on extended instruments (Relevance check)
price_model = sm.OLS(product_data['prices'], Z).fit()

# Regression 2: Market shares on extended instruments
share_model = sm.OLS(product_data['shares'], Z).fit()

# Regression 3: Demand unobservable ξ on instruments (Exclusion check)
xi_model = sm.OLS(product_data['xi'], Z).fit()

# Regression 4: Cost unobservable ω on instruments (Exclusion check)
omega_model = sm.OLS(product_data['omega'], Z).fit()

# Test joint significance of excluded instruments
print("="*75)
print("INSTRUMENT VALIDITY TESTS")
print("="*75)
excluded_vars = ['w', 'x**2', 'w**2', 'x*w',
                 'sum_x_competitors', 'sum_w_competitors',
                 'x_other_in_nest', 'w_other_in_nest']

# Create hypothesis string using actual variable names
hypothesis = ', '.join([f'{var}=0' for var in excluded_vars])

# F-test for excluded instruments in price regression
price_f_test = price_model.f_test(hypothesis)
print(f"\n1. Price Regression (Relevance Test)")
print(f"   R²: {price_model.rsquared:.3f}")
print(f"   Excluded demand instruments F-stat: {price_f_test.fvalue:.2f} (p={price_f_test.pvalue:.2e})")
print(f"   → Excluded instruments are {'relevant' if price_f_test.pvalue < 0.01 else 'weak'} for prices")

# F-test for excluded instruments in share regression
share_f_test = share_model.f_test(hypothesis)
print(f"\n2. Share Regression (Relevance Test)")
print(f"   R²: {share_model.rsquared:.3f}")
print(f"   Excluded demand instruments F-stat: {share_f_test.fvalue:.2f} (p={share_f_test.pvalue:.2e})")
print(f"   → Excluded instruments are {'relevant' if share_f_test.pvalue < 0.01 else 'weak'} for shares")

# F-test for excluded instruments in xi regression (should be insignificant)
xi_f_test = xi_model.f_test(hypothesis)
print(f"\n3. ξ Regression (Exclusion Test)")
print(f"   R²: {xi_model.rsquared:.3f}")
print(f"   Excluded demand instruments F-stat: {xi_f_test.fvalue:.2f} (p={xi_f_test.pvalue:.2e})")
print(f"   → Excluded instruments are {'exogenous' if xi_f_test.pvalue >= 0.01 else 'endogenous'}")

# F-test for excluded instruments in omega regression (should be insignificant)
omega_f_test = omega_model.f_test(hypothesis)
print(f"\n4. ω Regression (Exclusion Test)")
print(f"   R²: {omega_model.rsquared:.3f}")
print(f"   Excluded demand instruments F-stat: {omega_f_test.fvalue:.2f} (p={omega_f_test.pvalue:.2e})")
print(f"   → Excluded instruments are {'exogenous' if omega_f_test.pvalue >= 0.01 else 'endogenous'}")

# Assess instrument validity
weak_instruments = (
    (price_model.f_pvalue >= 0.01 and share_model.f_pvalue >= 0.01) or
    (price_model.rsquared < 0.05 and share_model.rsquared < 0.05)
)
excluded_instruments = (
    xi_f_test.pvalue < 0.01 or omega_f_test.pvalue < 0.01
)
print()
print("="*75)
print("FINAL PARAMETER CHOICE:")
print("="*75)
if weak_instruments or excluded_instruments:
    print("Parameters need adjustment - instruments are weak or invalid.")
else:
    print(f"Demand: α = {alpha}, β^(1) = {beta1}, β_i^(2) ~ N({beta2}, {sigma_satellite}²), β_i^(3) ~ N({beta3}, {sigma_wired}²)")
    print(f"Supply: γ^(0) = {gamma0}, γ^(1) = {gamma1}")
    print("These parameters generate data with valid instruments and are retained as final.")

product_data.to_csv('blp.csv', index=False)
print(product_data.head(8))

# Compute outside shares for each market
product_data['outside_share'] = 1 - product_data.groupby('market_ids')['shares'].transform('sum')

# Compute logit delta: ln(s_jt / s_0t)
product_data['logit_delta'] = np.log(product_data['shares'] / product_data['outside_share'])

# OLS using matrix algebra (no intercept)
y = product_data['logit_delta'].values
X = product_data[['prices', 'x', 'satellite', 'wired' ]].values

# Compute OLS estimates: beta_hat = (X^T X)^(-1) X^T y
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

# Compute residuals and HC0 robust standard errors
y_hat = X @ beta_hat
residuals = y - y_hat
n, k = X.shape

# HC0 robust covariance matrix
V = X.T @ np.diag(residuals**2) @ X
cov_matrix_ols = np.linalg.inv(X.T @ X) @ V @ np.linalg.inv(X.T @ X)
se_ols = np.sqrt(np.diag(cov_matrix_ols))

# t-statistics and p-values
t_stats = beta_hat / se_ols
p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

print("OLS Regression: ln(s_jt/s_0t) ~ x + satellite + wired + prices (no intercept)")
print("-" * 70)
param_names = ['prices', 'x', 'satellite', 'wired']
for i, param in enumerate(param_names):
    print(f"{param:12s}: {beta_hat[i]:8.3f} (SE: {se_ols[i]:.3f}, t: {t_stats[i]:6.2f}, p: {p_values[i]:.3f})")

product_data['demand_instruments0'] = product_data['prices']
ols_problem = pyblp.Problem(pyblp.Formulation('0 + prices + x + satellite + wired '), product_data)
ols_results = ols_problem.solve(method='1s')

pd.DataFrame(index=ols_results.beta_labels, data={
    ("Estimates", "Manual OLS"): beta_hat,
    ("Estimates", "PyBLP"): ols_results.beta.flat,
    ("SEs", "Manual OLS"): se_ols,
    ("SEs", "PyBLP"): ols_results.beta_se.flat
})

# First stage:
Z = product_data[['satellite', 'wired', 'x', 'w', 'x**2', 'w**2', 'x*w', 'sum_x_competitors', 'sum_w_competitors']].values

# First stage OLS:
sigma_hat = np.linalg.inv(Z.T @ Z) @ Z.T @ product_data['prices'].values
prices_hat = Z @ sigma_hat

# First stage diagnostics
first_stage_residuals = product_data['prices'].values - prices_hat
SST = np.sum((product_data['prices'].values - product_data['prices'].mean())**2)
SSR = np.sum(first_stage_residuals**2)
R2_first_stage = 1 - SSR/SST

# F-statistic for excluded instruments (w, x², w², x*w, sum_x_competitors, sum_w_competitors)
# Restricted model: prices ~ satellite + wired + x
Z_restricted = product_data[['satellite', 'wired', 'x']].values
sigma_restricted = np.linalg.inv(Z_restricted.T @ Z_restricted) @ Z_restricted.T @ product_data['prices'].values
prices_restricted = Z_restricted @ sigma_restricted
SSR_restricted = np.sum((product_data['prices'].values - prices_restricted)**2)

# F-test: F = [(SSR_r - SSR_ur)/q] / [SSR_ur/(n-k)]
n = len(product_data)
k = Z.shape[1]  # number of parameters in unrestricted model
q = 6  # number of excluded instruments
F_stat = ((SSR_restricted - SSR) / q) / (SSR / (n - k))
p_value_F = 1 - stats.f.cdf(F_stat, q, n - k)

print(f"First Stage Diagnostics:")
print(f"  R² = {R2_first_stage:.4f}")
print(f"  F-statistic (excluded instruments) = {F_stat:.2f} (p = {p_value_F:.4f})")
print()

# First stage for nested logit (moved here to be available for 2SLS)
# Define variables
exog_vars = ["x", "satellite", "wired"]
endog_vars = ["prices", "ln_within_share_sat", "ln_within_share_wired"]
instr_vars = ["w", "x**2", "w**2", "x*w", "sum_x_competitors", "sum_w_competitors", "x_other_in_nest", "w_other_in_nest"]
Z_vars = exog_vars + instr_vars

# First stage: Z = exog + instr
Z_nested = product_data[Z_vars].values

# First stage OLS for each endog
n_endog = len(endog_vars)
endog_hat = np.zeros((len(product_data), n_endog))
for i, var in enumerate(endog_vars):
    y_endog = product_data[var].values
    sigma = np.linalg.inv(Z_nested.T @ Z_nested) @ Z_nested.T @ y_endog
    endog_hat[:, i] = Z_nested @ sigma

# Second stage: Regress logit_delta on x + satellite + wired + predicted_prices
y = product_data['logit_delta'].values
X_hat = np.column_stack([
    prices_hat,  # Use predicted prices from first stage
    product_data['x'].values,
    product_data['satellite'].values,
    product_data['wired'].values,
    endog_hat[:, 1],  # ln_within_share_sat_hat
    endog_hat[:, 2]   # ln_within_share_wired_hat
])

# 2SLS estimates: beta_hat_iv = (X_hat^T X_hat)^(-1) X_hat^T y
beta_hat_iv = np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T @ y

# Compute 2SLS standard errors (HC0 robust)
# Need to use original regressors X, not fitted X_hat
X = np.column_stack([
    product_data["prices"].values,
    product_data["x"].values,
    product_data["satellite"].values,
    product_data["wired"].values,
    product_data["ln_within_share_sat"].values,
    product_data["ln_within_share_wired"].values
])

residuals_iv = y - X @ beta_hat_iv

# HC0 robust covariance for 2SLS: (X'Z(Z'Z)^{-1}Z'X)^{-1} X'Z(Z'Z)^{-1} Ω (Z'Z)^{-1}Z'X (X'Z(Z'Z)^{-1}Z'X)^{-1}
# where Ω = diag(residuals²)
P_Z = Z @ np.linalg.inv(Z.T @ Z) @ Z.T  # Projection matrix
Omega = np.diag(residuals_iv**2)

# Simplified: (X'P_Z X)^{-1} X'P_Z Ω P_Z X (X'P_Z X)^{-1}
XPZ = X.T @ P_Z
bread = np.linalg.inv(XPZ @ X)
meat = XPZ @ Omega @ P_Z @ X
cov_matrix_iv = bread @ meat @ bread
se_iv = np.sqrt(np.diag(cov_matrix_iv))
t_stats_iv = beta_hat_iv / se_iv
p_values_iv = 2 * (1 - stats.norm.cdf(np.abs(t_stats_iv)))

print("2SLS IV Regression: ln(s_jt/s_0t) ~ x + satellite + wired + prices_hat (no intercept)")
print("First stage instruments: x, w, x², w², x*w, sum_x_competitors, sum_w_competitors")
print("-" * 80)
param_names = ["prices", "x", "satellite", "wired", "ln_within_share_sat", "ln_within_share_wired"]
for i, param in enumerate(param_names):
    print(f"{param:20s}: {beta_hat_iv[i]:8.3f} (SE: {se_iv[i]:.3f}, t: {t_stats_iv[i]:6.2f}, p: {p_values_iv[i]:.3f})")

# Extract nested logit parameters
beta_hat_iv_nested = beta_hat_iv
se_iv_nested = se_iv
alpha_nl, beta_x_nl, rho_sat_nl, rho_wired_nl = beta_hat_iv_nested[[0, 1, 4, 5]]

def compute_nested_logit_elasticities_analytic(market_df, alpha, beta_x, rho_sat, rho_wired):
    """Compute elasticities using pyBLP's exact Jacobian formula for nested logit.

    Based on pyBLP's compute_capital_lamda_gamma:
        - Lambda_jj = alpha * s_j / (1 - rho_j)
        - Gamma_jk = alpha * s_j * s_k + rho/(1-rho) * membership_jk * alpha * s_j|g * s_k
        - Jacobian[j,k] = Lambda_jj - Gamma_jk (if j==k), -Gamma_jk (if j!=k)
        - Elasticity[j,k] = Jacobian[j,k] * price[k] / share[j]

        This matches pyBLP to within ~1% numerical precision.
        """
    J = len(market_df)
    prices = market_df['prices'].values
    shares = market_df['shares'].values
    satellite, wired = market_df['satellite'].values, market_df['wired'].values

    # Compute within-nest shares (conditionals in pyBLP terminology)
    s_group = market_df.groupby('satellite')['shares'].transform('sum').values
    conditionals = shares / s_group

    # Nesting parameter for each product
    rho = np.where(satellite == 1, rho_sat, rho_wired)

    # Compute full elasticity matrix using pyBLP's formula
    elasticities = np.zeros((J, J))
    for j in range(J):
        # Lambda diagonal element
        lambda_jj = alpha * shares[j] / (1 - rho[j])

        for k in range(J):
            # Gamma matrix element
            same_nest = (satellite[j] == satellite[k]) and (wired[j] == wired[k])
            gamma_jk = alpha * shares[j] * shares[k]
            if same_nest:
                gamma_jk += (rho[j] / (1 - rho[j])) * alpha * conditionals[j] * shares[k]

            # Jacobian = Lambda - Gamma (on diagonal), -Gamma (off-diagonal)
            if j == k:
                jac_jk = lambda_jj - gamma_jk
            else:
                jac_jk = -gamma_jk

            # Elasticity = Jacobian * price / share
            elasticities[j, k] = jac_jk * prices[k] / shares[j]

    return elasticities

def compute_rc_elasticities_observed_shares(market_df, nu_draws, alpha, beta_x, beta_sat, beta_wired, sigma_sat, sigma_wired):
    """Compute elasticities from RC logit using OBSERVED shares (not recomputed from xi).

    This matches pyBLP's approach:
        1. Start with observed shares
        2. Back out mean utilities (delta) that rationalize these shares via contraction mapping
        3. Compute individual choice probabilities using delta + random coefficients
        4. Compute elasticities via analytical derivatives

        Key difference from old method:
            - OLD: Uses TRUE xi to compute shares, then elasticities (wrong for comparison!)
            - NEW: Uses OBSERVED shares, backs out delta, then computes elasticities (correct!)
        """
    J = len(market_df)
    prices = market_df['prices'].values
    observed_shares = market_df['shares'].values
    x, satellite, wired = market_df['x'].values, market_df['satellite'].values, market_df['wired'].values

    # Compute random coefficient deviations (the part that varies across individuals)
    # Delta will absorb everything else: beta_x*x + beta_sat*satellite + beta_wired*wired + alpha*prices + xi
    rc_deviation = sigma_sat*nu_draws[:,0:1]*satellite + sigma_wired*nu_draws[:,1:2]*wired

    # Back out mean utilities (delta) via contraction mapping
    # Goal: Find delta such that observed_shares = E[exp(delta + rc_deviation) / (1 + sum exp(delta + rc_deviation))]
    delta = np.log(observed_shares)  # Initial guess

    for iteration in range(1000):
        # Compute individual choice probabilities
        utilities = delta[np.newaxis, :] + rc_deviation  # Shape: (n_draws, J)
        exp_utils = np.exp(utilities)
        denom = 1 + exp_utils.sum(axis=1, keepdims=True)
        choice_probs = exp_utils / denom  # Shape: (n_draws, J)

        # Predicted shares
        predicted_shares = choice_probs.mean(axis=0)

        # Contraction update: delta_new = delta + log(s_obs) - log(s_pred)
        delta_new = delta + np.log(observed_shares) - np.log(predicted_shares)

        # Check convergence
        if np.max(np.abs(delta_new - delta)) < 1e-14:
            delta = delta_new
            break
        delta = delta_new

    # Compute final choice probabilities with converged delta
    utilities = delta[np.newaxis, :] + rc_deviation
    exp_utils = np.exp(utilities)
    choice_probs = exp_utils / (1 + exp_utils.sum(axis=1, keepdims=True))

    # Compute elasticities using analytical derivatives
    elasticities = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                # Own-price: E[s_ij * (1 - s_ij)]
                deriv = alpha * np.mean(choice_probs[:, j] * (1 - choice_probs[:, j]))
            else:
                # Cross-price: -E[s_ij * s_ik]
                deriv = -alpha * np.mean(choice_probs[:, j] * choice_probs[:, k])

            if observed_shares[j] > 1e-10:
                elasticities[j, k] = (prices[k] / observed_shares[j]) * deriv

    return elasticities

# ============================================================================
# Compute elasticities for Q8 comparison
# ============================================================================

# Compute Nested Logit elasticities (analytical derivatives)
print("Computing Nested Logit Elasticities (Analytical Derivatives)...")
elasticity_matrices_analytic = [compute_nested_logit_elasticities_analytic(
    product_data[product_data['market_ids'] == t], alpha_nl, beta_x_nl, rho_sat_nl, rho_wired_nl
) for t in range(T)]

print("Computing True RC Logit Elasticities (True Parameters on OBSERVED shares)...")
# Use the new function that works with observed shares for fair comparison
true_elasticity_matrices = [compute_rc_elasticities_observed_shares(
    product_data[product_data['market_ids'] == t], all_nu_draws[t], -2.0, 1.0, 4.0, 4.0, 1.0, 1.0
) for t in range(T)]

avg_elasticity_matrix_nl = np.mean(elasticity_matrices_analytic, axis=0)
avg_elasticity_matrix_true = np.mean(true_elasticity_matrices, axis=0)

# Comparison table
print("\n" + "="*70 + "\nOWN-PRICE ELASTICITY COMPARISON\n" + "="*70)
print("Nested Logit (Estimated) vs RC Logit (True params, observed shares)")
comparison_df = pd.DataFrame({
    'Product': ['Satellite 1', 'Satellite 2', 'Wired 1', 'Wired 2'],
    'True (RC)': np.diag(avg_elasticity_matrix_true),
    'Estimated (NL)': np.diag(avg_elasticity_matrix_nl),
    'Abs % Error': np.abs(100 * (np.diag(avg_elasticity_matrix_nl) - np.diag(avg_elasticity_matrix_true)) / np.diag(avg_elasticity_matrix_true))
})
print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:8.3f}'))
print(f"\nMean Absolute % Error: {comparison_df['Abs % Error'].mean():.2f}%")
print("\nNote: NL model misspecified (true DGP is RC), so errors expected")
print("="*70 + "\n")

# Store elasticities in product_data
product_data['true_elasticity_rc'] = [true_elasticity_matrices[t][j, j] for t in range(T) for j in range(J)]
product_data['estimated_elasticity_nl'] = [elasticity_matrices_analytic[t][j, j] for t in range(T) for j in range(J)]

# ============================================================================
# PyBLP Nested Logit Estimation
# ============================================================================

# ============================================================================
# DIVERSION RATIOS
# ============================================================================
# Using PyBLP's derivative-based method for both RC and NL models
# Convention: Diagonal shows diversion to outside option D_j0 instead of D_jj=-1

print("Computing Diversion Ratios...")

# Unified function using PyBLP's derivative-based approach
def compute_diversion_ratios_pyblp(elasticity_matrices, product_data, T, J):
    """
    Compute diversion ratios using pyBLP's derivative-based method.

    This method:
        1. Converts elasticities to Jacobian (derivatives)
        2. Replaces diagonal with outside option derivative using adding-up constraint
        3. Computes diversion ratios as D_jk = -(∂s_k/∂p_j) / (∂s_j/∂p_j)

        Works for any model (RC, NL, etc.) - just supply the elasticity matrices.
        """
    diversion_matrices = []

    for t in range(T):
        elast_matrix = elasticity_matrices[t]
        market_data_t = product_data[product_data['market_ids'] == t]
        shares = market_data_t['shares'].values
        prices = market_data_t['prices'].values

        # Convert elasticities to Jacobian (derivatives): ∂s_j/∂p_k = (s_j/p_k) * ε_jk
        jacobian = np.zeros((J, J))
        for j in range(J):
            for k in range(J):
                jacobian[j, k] = (shares[j] / prices[k]) * elast_matrix[j, k]

        # PyBLP's method: Replace diagonal with outside option derivative
        # ∂s_0/∂p_j = -Σ_k ∂s_k/∂p_j (by adding-up constraint)
        jacobian_diag = np.diag(jacobian).copy()
        np.fill_diagonal(jacobian, -jacobian.sum(axis=1))

        # Compute diversion ratios: D_jk = -Jacobian[j,k] / Jacobian[j,j]
        diversion = -jacobian / jacobian_diag[:, None]

        diversion_matrices.append(diversion)

    return diversion_matrices

# --- TRUE DIVERSION RATIOS (from RC model with true parameters on OBSERVED shares) ---
# Note: We recompute true elasticities here to ensure we use observed shares
print("Computing TRUE RC diversion ratios (true params, observed shares)...")
true_elasticity_matrices_for_div = [compute_rc_elasticities_observed_shares(
    product_data[product_data['market_ids'] == t], all_nu_draws[t], -2.0, 1.0, 4.0, 4.0, 1.0, 1.0
) for t in range(T)]

true_diversion_matrices = compute_diversion_ratios_pyblp(
    true_elasticity_matrices_for_div, product_data, T, J
)
true_avg_diversion = np.mean(true_diversion_matrices, axis=0)

# --- ESTIMATED DIVERSION RATIOS (from Nested Logit) ---
estimated_diversion_matrices = compute_diversion_ratios_pyblp(
    elasticity_matrices_analytic, product_data, T, J
)
estimated_avg_diversion = np.mean(estimated_diversion_matrices, axis=0)

# --- DISPLAY RESULTS ---
print("\n" + "=" * 70)
print("DIVERSION RATIO MATRICES")
print("=" * 70)

product_labels = ['Sat 1', 'Sat 2', 'Wired 1', 'Wired 2']

print("\nTrue Diversion Ratios (RC Logit with TRUE params on OBSERVED shares):")
print("Diagonal = diversion to outside option D_j0")
true_df = pd.DataFrame(true_avg_diversion, index=product_labels, columns=product_labels)
print(true_df.to_string(float_format=lambda x: f'{x:7.4f}'))

print("\n\nEstimated Diversion Ratios (from Nested Logit - PyBLP Method):")
print("Diagonal = diversion to outside option D_j0")
est_df = pd.DataFrame(estimated_avg_diversion, index=product_labels, columns=product_labels)
print(est_df.to_string(float_format=lambda x: f'{x:7.4f}'))

print("\n" + "=" * 70)
print("Note: D_jk = -(∂s_k/∂p_j) / (∂s_j/∂p_j)")
print("Off-diagonal: share of j's lost customers who switch to k")
print("Diagonal: share of j's lost customers who leave the market (outside)")
print("=" * 70)

X1_formulation = pyblp.Formulation('0 + prices + x + satellite + wired')
X2_formulation = pyblp.Formulation('0 + satellite + wired')
product_formulations1 = (X1_formulation, X2_formulation)
product_data['demand_instruments0'] = product_data['w']
product_data['demand_instruments1'] = product_data['x**2']
product_data['demand_instruments2'] = product_data['w**2']
product_data['demand_instruments3'] = product_data['x*w']
product_data['demand_instruments4'] = product_data['sum_x_competitors']
product_data['demand_instruments5'] = product_data['sum_w_competitors']
product_data['demand_instruments6'] = product_data['x_other_in_nest']
product_data['demand_instruments7'] = product_data['w_other_in_nest']
integration = pyblp.Integration('product', 10)
problem1 = pyblp.Problem(product_formulations1, product_data, integration=integration)
results1 = problem1.solve(sigma=np.eye(2), initial_update=True)
optimal_iv1 = results1.compute_optimal_instruments(seed=1995)
optimal_iv1.to_problem()
optimal_iv_results1 = optimal_iv1.to_problem().solve(sigma=np.eye(2), initial_update=True)

X3_formulation = pyblp.Formulation('1 + w')
product_formulations2 = (X1_formulation, X2_formulation, X3_formulation)
columns_to_drop = [col for col in product_data.columns if 'instruments' in col]
product_data = product_data.drop(columns=columns_to_drop)
product_data['demand_instruments0'] = optimal_iv1.demand_instruments[:, 0]
product_data['demand_instruments1'] = optimal_iv1.demand_instruments[:, 1]
product_data['demand_instruments2'] = product_data['w']
problem2 = pyblp.Problem(product_formulations2, product_data, costs_type='log', integration=integration)
results2 = problem2.solve(sigma=np.eye(2), beta=optimal_iv_results1.beta, initial_update=True)

# Re-estimate with optimal instruments
columns_to_drop = [col for col in product_data.columns
                   if 'instruments' in col]
product_data = product_data.drop(columns=columns_to_drop)
optimal_iv2 = results2.compute_optimal_instruments(seed=1995)
for i in range(optimal_iv2.demand_instruments.shape[1]-3):
    product_data[f'demand_instruments{i}'] = optimal_iv2.demand_instruments[:, i]
problem3 = pyblp.Problem(product_formulations2, product_data,
                         costs_type='log', integration=integration)
optimal_iv_results2 = problem3.solve(sigma=np.eye(2), beta=results2.beta, initial_update=True)

# Compare individual and joint PyBLP estimates for beta
pyblp_beta_comparison = pd.DataFrame(index=optimal_iv_results1.beta_labels, data={
    ("Estimates", "PyBLP D"): optimal_iv_results1.beta.flat,  # prices, x, satellite, wired
    ("Estimates", "PyBLP D & S"): optimal_iv_results2.beta.flat,
    ("SEs", "PyBLP D"): optimal_iv_results1.beta_se.flat,
    ("SEs", "PyBLP D & S"): optimal_iv_results2.beta_se.flat
})
print("Beta Comparison:")
print(pyblp_beta_comparison)
# Compare sigma estimates
pyblp_sigma_comparison = pd.DataFrame(index=optimal_iv_results1.sigma_labels, data={
    ("Estimates", "PyBLP D"): optimal_iv_results1.sigma.diagonal(),
    ("Estimates", "PyBLP D & S"): optimal_iv_results2.sigma.diagonal(),
    ("SEs", "PyBLP D"): optimal_iv_results1.sigma_se.diagonal(),
    ("SEs", "PyBLP D & S"): optimal_iv_results2.sigma_se.diagonal()
})
print("\nSigma Comparison:")
print(pyblp_sigma_comparison)

# Compare gamma estimates (only available in joint estimation)
print("\n\nGamma Estimates (only from joint D & S estimation):")
pyblp_gamma_comparison = pd.DataFrame(index=optimal_iv_results2.gamma_labels, data={
    ("Estimates", "PyBLP D & S"): optimal_iv_results2.gamma.flat,
    ("SEs", "PyBLP D & S"): optimal_iv_results2.gamma_se.flat,
})
print(pyblp_gamma_comparison)


# ============================================================================
# Q9: Compare TRUE vs ESTIMATED Random Coefficients Elasticities/Diversions
# ============================================================================
# Reuse TRUE elasticities computed in Q8 to avoid redundancy
print("Using TRUE elasticities from Q8 (true params, observed shares)...")
print(f"  (Already computed {len(true_elasticity_matrices_for_div)} markets)")
true_elasticity_matrices_obs = true_elasticity_matrices_for_div  # Reuse from Q8
avg_elasticity_matrix_true_rc = np.mean(true_elasticity_matrices_obs, axis=0)
own_elasticities_rc_true = np.diag(avg_elasticity_matrix_true_rc)

# Compute ESTIMATED elasticities - DEMAND ONLY
print("Computing ESTIMATED elasticities (demand-only params, observed shares)...")
elasticities_rc_est1 = optimal_iv_results1.compute_elasticities()
avg_elasticities_rc_est1 = elasticities_rc_est1.reshape((T, J, J)).mean(axis=0)
own_elasticities_rc_est1 = np.diag(avg_elasticities_rc_est1)

# Compute ESTIMATED elasticities - JOINT DEMAND & SUPPLY
print("Computing ESTIMATED elasticities (joint D&S params, observed shares)...")
elasticities_rc_est2 = optimal_iv_results2.compute_elasticities()
avg_elasticities_rc_est2 = elasticities_rc_est2.reshape((T, J, J)).mean(axis=0)
own_elasticities_rc_est2 = np.diag(avg_elasticities_rc_est2)

# Show parameter differences
print("\nParameter Comparison:")
print("                 True      Demand-only    Joint D&S")
print(f"α (price):      -2.000    {optimal_iv_results1.beta[0,0]:7.3f}      {optimal_iv_results2.beta[0,0]:7.3f}")
print(f"σ_satellite:     1.000    {optimal_iv_results1.sigma[0,0]:7.3f}      {optimal_iv_results2.sigma[0,0]:7.3f}")
print(f"σ_wired:         1.000    {optimal_iv_results1.sigma[1,1]:7.3f}      {optimal_iv_results2.sigma[1,1]:7.3f}")
print()

# Create comparison table with THREE columns
product_labels = ['Sat 1', 'Sat 2', 'Wired 1', 'Wired 2']
elasticity_comparison_rc = pd.DataFrame({
    'Product': product_labels,
    'True': own_elasticities_rc_true,
    'Demand-only': own_elasticities_rc_est1,
    'Joint D&S': own_elasticities_rc_est2,
    '% Error (D-only)': np.abs((own_elasticities_rc_est1 - own_elasticities_rc_true) / own_elasticities_rc_true * 100),
    '% Error (Joint)': np.abs((own_elasticities_rc_est2 - own_elasticities_rc_true) / own_elasticities_rc_true * 100)
})

print("\n" + "=" * 95)
print("TABLE 1: OWN-PRICE ELASTICITY COMPARISON")
print("True = RC logit with TRUE params (-2, 1, 4, 4, 1, 1) on OBSERVED shares")
print("Demand-only = RC logit with demand-only estimated params on OBSERVED shares")
print("Joint D&S = RC logit with joint demand & supply estimated params on OBSERVED shares")
print("=" * 95)
print(elasticity_comparison_rc.to_string(index=False, float_format=lambda x: f'{x:9.4f}'))
print(f"\nMean Absolute % Error (Demand-only): {elasticity_comparison_rc['% Error (D-only)'].mean():.2f}%")
print(f"Mean Absolute % Error (Joint D&S):   {elasticity_comparison_rc['% Error (Joint)'].mean():.2f}%")

# --- DIVERSION RATIO COMPARISON ---
# Reuse TRUE diversion ratios computed in Q8 to avoid redundancy
print("\nUsing TRUE diversion ratios from Q8 (true params, observed shares)...")
print(f"  (Already computed {len(true_diversion_matrices)} markets)")
true_diversion_matrices_obs = true_diversion_matrices  # Reuse from Q8
true_avg_diversion_rc = np.mean(true_diversion_matrices_obs, axis=0)

# RC estimated diversion ratios - DEMAND ONLY
diversion_rc_est1 = optimal_iv_results1.compute_diversion_ratios()
avg_diversion_rc_est1 = diversion_rc_est1.reshape((T, J, J)).mean(axis=0)

# RC estimated diversion ratios - JOINT D&S
diversion_rc_est2 = optimal_iv_results2.compute_diversion_ratios()
avg_diversion_rc_est2 = diversion_rc_est2.reshape((T, J, J)).mean(axis=0)

print("\n" + "=" * 80)
print("TABLE 2: TRUE DIVERSION RATIOS")
print("(from RC Logit with TRUE params: σ_sat=1.0, σ_wired=1.0, on OBSERVED shares)")
print("=" * 80)
print("Diagonal = diversion to outside option D_j0")
true_div_df_rc = pd.DataFrame(true_avg_diversion_rc, index=product_labels, columns=product_labels)
print(true_div_df_rc.to_string(float_format=lambda x: f'{x:7.4f}'))

print("\n" + "=" * 80)
print("TABLE 3: ESTIMATED DIVERSION RATIOS - DEMAND ONLY")
print(f"(from RC Logit with DEMAND-ONLY params: σ_sat={optimal_iv_results1.sigma[0,0]:.3f}, σ_wired={optimal_iv_results1.sigma[1,1]:.3f})")
print("=" * 80)
print("Diagonal = diversion to outside option D_j0")
est_div_df_rc1 = pd.DataFrame(avg_diversion_rc_est1, index=product_labels, columns=product_labels)
print(est_div_df_rc1.to_string(float_format=lambda x: f'{x:7.4f}'))

print("\n" + "=" * 80)
print("TABLE 4: ESTIMATED DIVERSION RATIOS - JOINT DEMAND & SUPPLY")
print(f"(from RC Logit with JOINT D&S params: σ_sat={optimal_iv_results2.sigma[0,0]:.3f}, σ_wired={optimal_iv_results2.sigma[1,1]:.3f})")
print("=" * 80)
print("Diagonal = diversion to outside option D_j0")
est_div_df_rc2 = pd.DataFrame(avg_diversion_rc_est2, index=product_labels, columns=product_labels)
print(est_div_df_rc2.to_string(float_format=lambda x: f'{x:7.4f}'))

# Calculate diversion errors
div_error_d_only = np.abs((avg_diversion_rc_est1 - true_avg_diversion_rc) / true_avg_diversion_rc * 100)
div_error_joint = np.abs((avg_diversion_rc_est2 - true_avg_diversion_rc) / true_avg_diversion_rc * 100)

print(f"Mean Absolute % Error in Diversion Ratios:")
print(f"  • Demand-only: {div_error_d_only.mean():.2f}%")
print(f"  • Joint D&S:   {div_error_joint.mean():.2f}%")
print()
print("Example: When Sat 1 price ↑ 1%, where do lost customers go?")
print(f"  TRUE model:       {true_avg_diversion_rc[0,0]:.1%} to outside, {true_avg_diversion_rc[0,1]:.1%} to Sat 2")
print(f"  DEMAND-only:      {avg_diversion_rc_est1[0,0]:.1%} to outside, {avg_diversion_rc_est1[0,1]:.1%} to Sat 2")
print(f"  JOINT D&S:        {avg_diversion_rc_est2[0,0]:.1%} to outside, {avg_diversion_rc_est2[0,1]:.1%} to Sat 2")

# Baseline marginal costs, markups, profits, and consumer surplus under the estimated demand+supply model
costs = optimal_iv_results2.compute_costs()
markups = optimal_iv_results2.compute_markups(costs=costs)
profits = optimal_iv_results2.compute_profits(costs=costs)
cs = optimal_iv_results2.compute_consumer_surpluses()

# Get pre-merger prices (reshaped as T×J to get average per product)
pre_merger_prices = product_data['prices'].values
pre_merger_prices_avg = pre_merger_prices.reshape((T, J)).mean(axis=0)

# Create merger firm IDs: merge firms 1 and 2 into firm 1
# Firms 1 and 2 (satellite) → firm 1
# Firms 3 and 4 (wired) remain unchanged
merger_firm_ids = product_data['firm_ids'].copy()
merger_firm_ids[merger_firm_ids == 2] = 1  # Firm 2 becomes firm 1

# Compute post-merger equilibrium prices using pyBLP's compute_prices method
# This solves the first-order conditions under the new ownership structure
post_merger_prices = optimal_iv_results2.compute_prices(
    firm_ids=merger_firm_ids,
    iteration=pyblp.Iteration('simple', {'atol': 1e-12})
)

# Reshape to get average prices per product
post_merger_prices_avg = post_merger_prices.reshape((T, J)).mean(axis=0)

# Calculate price changes
price_changes = post_merger_prices_avg - pre_merger_prices_avg
pct_price_changes = (price_changes / pre_merger_prices_avg) * 100

merger_results_df = pd.DataFrame({
    'Product': product_labels,
    'Type': ['Satellite', 'Satellite', 'Wired', 'Wired'],
    'Firm (Pre)': [1, 2, 3, 4],
    'Firm (Post)': [1, 1, 3, 4],
    'Pre-Merger Price': pre_merger_prices_avg,
    'Post-Merger Price': post_merger_prices_avg,
    'Price Change ($)': price_changes,
    'Price Change (%)': pct_price_changes
})
numeric_columns = ['Pre-Merger Price', 'Post-Merger Price', 'Price Change ($)', 'Price Change (%)']
merger_results_df[numeric_columns] = merger_results_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
# Diagnostics inspired by the post_estimation tutorial
post_merger_shares = optimal_iv_results2.compute_shares(post_merger_prices)
post_merger_markups = optimal_iv_results2.compute_markups(post_merger_prices, costs)
post_merger_profits = optimal_iv_results2.compute_profits(post_merger_prices, post_merger_shares, costs)
post_merger_cs = optimal_iv_results2.compute_consumer_surpluses(post_merger_prices)

baseline_metrics = {
    'Average Price ($)': pre_merger_prices_avg.mean(),
    'Average Markup ($)': markups.reshape((T, J)).mean(),
    'Average Profit ($)': profits.mean(),
    'Average CS ($)': cs.mean(),
}
post_merger_metrics = {
    'Average Price ($)': post_merger_prices_avg.mean(),
    'Average Markup ($)': post_merger_markups.reshape((T, J)).mean(),
    'Average Profit ($)': post_merger_profits.mean(),
    'Average CS ($)': post_merger_cs.mean(),
}
metric_names = list(baseline_metrics.keys())
merger_metric_summary = pd.DataFrame({
    'Metric': metric_names,
    'Pre-Merger': [baseline_metrics[m] for m in metric_names],
    'Post-Merger': [post_merger_metrics[m] for m in metric_names]
})
merger_metric_summary['Change'] = (
    merger_metric_summary['Post-Merger'] - merger_metric_summary['Pre-Merger']
)
merger_metric_summary = merger_metric_summary.astype({
    'Pre-Merger': float,
    'Post-Merger': float,
    'Change': float
})
merger_metric_summary = merger_metric_summary.set_index('Metric')
IPython.display.display(merger_results_df.style.format({
    'Pre-Merger Price': '{:,.4f}'.format,
    'Post-Merger Price': '{:,.4f}'.format,
    'Price Change ($)': '{:,.4f}'.format,
    'Price Change (%)': '{:,.2f}'.format,
}))
IPython.display.display(merger_metric_summary.style.format('{:,.3f}'))

# ========================================================================
# QUESTION 13: MERGER SIMULATION (Firms 1 and 3)
# ========================================================================
# Firm 1 is satellite provider, Firm 3 is wired provider (cross-nest merger)

# Note: pre_merger_prices, post_merger_prices, and mean_pct_change_within
# already computed in Question 11

# Create merger firm IDs: merge firms 1 and 3 into firm 1
# Firm 1 (satellite) + Firm 3 (wired) → firm 1
# Firms 2 and 4 remain unchanged
merger_firm_ids_cross = product_data['firm_ids'].copy()
merger_firm_ids_cross[merger_firm_ids_cross == 3] = 1  # Firm 3 becomes firm 1

# Compute post-merger equilibrium prices for cross-nest merger
post_merger_prices_cross = optimal_iv_results2.compute_prices(
    firm_ids=merger_firm_ids_cross,
    iteration=pyblp.Iteration('simple', {'atol': 1e-12})
)

# Reshape and calculate changes
post_merger_prices_cross_matrix = post_merger_prices_cross.reshape((T, J))
post_merger_prices_cross_avg = post_merger_prices_cross_matrix.mean(axis=0)
mean_pct_change_cross = ((post_merger_prices_cross_avg - pre_merger_prices_avg) / pre_merger_prices_avg * 100)

merger_results_cross_df = pd.DataFrame({
    'Product': product_labels,
    'Type': ['Satellite', 'Satellite', 'Wired', 'Wired'],
    'Firm (Pre)': [1, 2, 3, 4],
    'Firm (Post)': [1, 2, 1, 4],
    'Pre-Merger Price': pre_merger_prices_avg,
    'Post-Merger Price': post_merger_prices_cross_avg,
    'Price Change ($)': post_merger_prices_cross_avg - pre_merger_prices_avg,
    'Price Change (%)': mean_pct_change_cross
})
numeric_columns_cross = ['Pre-Merger Price', 'Post-Merger Price', 'Price Change ($)', 'Price Change (%)']
merger_results_cross_df[numeric_columns_cross] = merger_results_cross_df[numeric_columns_cross].apply(pd.to_numeric, errors='coerce')
# Diagnostics parallel to the post_estimation tutorial
post_merger_shares_cross = optimal_iv_results2.compute_shares(post_merger_prices_cross)
post_merger_markups_cross = optimal_iv_results2.compute_markups(post_merger_prices_cross, costs)
post_merger_profits_cross = optimal_iv_results2.compute_profits(
    post_merger_prices_cross,
    post_merger_shares_cross,
    costs
)
post_merger_cs_cross = optimal_iv_results2.compute_consumer_surpluses(post_merger_prices_cross)

cross_merger_metrics = {
    'Average Price ($)': post_merger_prices_cross_avg.mean(),
    'Average Markup ($)': post_merger_markups_cross.reshape((T, J)).mean(),
    'Average Profit ($)': post_merger_profits_cross.mean(),
    'Average CS ($)': post_merger_cs_cross.mean(),
}
cross_metric_summary = pd.DataFrame({
    'Metric': metric_names,
    'Within-Nest (1&2)': [post_merger_metrics[m] for m in metric_names],
    'Cross-Nest (1&3)': [cross_merger_metrics[m] for m in metric_names]
})
cross_metric_summary['Difference (Cross - Within)'] = (
    cross_metric_summary['Cross-Nest (1&3)'] - cross_metric_summary['Within-Nest (1&2)']
)
cross_metric_summary = cross_metric_summary.astype({
    'Within-Nest (1&2)': float,
    'Cross-Nest (1&3)': float,
    'Difference (Cross - Within)': float,
})
cross_metric_summary = cross_metric_summary.set_index('Metric')

IPython.display.display(merger_results_cross_df.style.format({
    'Pre-Merger Price': '{:,.4f}'.format,
    'Post-Merger Price': '{:,.4f}'.format,
    'Price Change ($)': '{:,.4f}'.format,
    'Price Change (%)': '{:,.2f}'.format,
}))

# Comparison table: Within-nest vs Cross-nest mergers
comparison_df = pd.DataFrame({
    'Product': product_labels,
    'Within-Nest (%)': pct_price_changes,
    'Cross-Nest (%)': mean_pct_change_cross,
    'Difference (pp)': mean_pct_change_cross - pct_price_changes
})
comparison_df[['Within-Nest (%)', 'Cross-Nest (%)', 'Difference (pp)']] = comparison_df[['Within-Nest (%)', 'Cross-Nest (%)', 'Difference (pp)']].apply(pd.to_numeric, errors='coerce')
IPython.display.display(comparison_df.style.format({
    'Within-Nest (%)': '{:,.2f}'.format,
    'Cross-Nest (%)': '{:,.2f}'.format,
    'Difference (pp)': '{:,.2f}'.format,
}))

# ========================================================================
# PUBLICATION-READY FIGURE: Within-Nest vs. Cross-Nest Merger Comparison
# Black and White Version
# ========================================================================

# Set Seaborn style and context
sns.set_style("whitegrid", {
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'grid.color': '#666666',
    'grid.alpha': 0.4,
    'axes.linewidth': 1.0,
    'axes.edgecolor': 'black',
})
sns.set_context("paper", font_scale=1.15)

# Additional matplotlib parameters for publication quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))

# Set up bar positions
indices = np.arange(len(product_labels))
width = 0.35

# Black and white colors
colors_merger = {
    'within': '#000000',  # Black for within-nest
    'cross': '#FFFFFF',   # White for cross-nest
}

# Plot bars with hatching patterns for differentiation
bars1 = ax.bar(indices - width/2, pct_price_changes, width,
               label='Within-Nest (Firms 1 & 2)',
               color=colors_merger['within'],
               edgecolor='black',
               linewidth=1.5,
               alpha=1.0,
               zorder=3)

bars2 = ax.bar(indices + width/2, mean_pct_change_cross, width,
               label='Cross-Nest (Firms 1 & 3)',
               color=colors_merger['cross'],
               edgecolor='black',
               linewidth=1.5,
               hatch='///',  # Diagonal hatching
               alpha=1.0,
               zorder=3)

# Horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, zorder=2)

# Add value labels with better styling
for idx, value in enumerate(pct_price_changes):
    if abs(value) > 0.05:  # Only show for non-trivial changes
        va = 'bottom' if value >= 0 else 'top'
        y_offset = 0.25 if value >= 0 else -0.25
        bbox_props = dict(boxstyle='round,pad=0.3',
                         facecolor='white',
                         edgecolor='black',
                         linewidth=1)
        ax.text(indices[idx] - width/2, value + y_offset,
                f'{value:.2f}%',
                ha='center', va=va,
                fontsize=9,
                fontweight='bold',
                color='black',
                bbox=bbox_props,
                zorder=4)

for idx, value in enumerate(mean_pct_change_cross):
    if abs(value) > 0.05:
        va = 'bottom' if value >= 0 else 'top'
        y_offset = 0.25 if value >= 0 else -0.25
        bbox_props = dict(boxstyle='round,pad=0.3',
                         facecolor='white',
                         edgecolor='black',
                         linewidth=1)
        ax.text(indices[idx] + width/2, value + y_offset,
                f'{value:.2f}%',
                ha='center', va=va,
                fontsize=9,
                fontweight='bold',
                color='black',
                bbox=bbox_props,
                zorder=4)

# Formatting
ax.set_ylabel('Price Change (%)', fontweight='bold', fontsize=13)
ax.set_title('Price Effects: Within-Nest vs. Cross-Nest Mergers',
             fontweight='bold', pad=20, fontsize=14)
ax.set_xticks(indices)
ax.set_xticklabels(product_labels, fontweight='semibold')

# Legend with Seaborn-friendly styling
legend = ax.legend(loc='upper right',
                   framealpha=1.0,
                   edgecolor='black',
                   fancybox=False,
                   shadow=False,
                   borderpad=1)
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_facecolor('white')

# Clean up spines
sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

plt.tight_layout()

# Save figures
plt.savefig('merger_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('merger_comparison.png', format='png', bbox_inches='tight', dpi=300)
print('✓ Publication-ready black and white figures saved')

plt.show()

# Reset to default style
sns.reset_defaults()

# Assume same market size M_t for all markets (needed for welfare calculation)
M_t = 1000

# Get estimated marginal costs from optimal_iv_results2
marginal_costs = optimal_iv_results2.compute_costs()

# Create a copy with 15% cost reduction for firms 1 and 2
marginal_costs_efficiency = marginal_costs.copy()

# Apply 15% reduction (multiply by 0.85) to firms 1 and 2 products
# More efficient approach using boolean indexing
products_1_2 = product_data['firm_ids'].isin([1, 2])
marginal_costs_efficiency[products_1_2] *= 0.85

# Use the same merger firm IDs as Question 11
# (firm 2 becomes firm 1, already computed earlier as merger_firm_ids)

# Compute post-merger prices WITH efficiency gains
post_merger_prices_efficiency = optimal_iv_results2.compute_prices(
    firm_ids=merger_firm_ids,  # Reuse from Q11
    costs=marginal_costs_efficiency,
    iteration=pyblp.Iteration('simple', {'atol': 1e-12})
)

# Reshape and calculate changes
post_merger_prices_eff_matrix = post_merger_prices_efficiency.reshape((T, J))
post_merger_prices_eff_avg = post_merger_prices_eff_matrix.mean(axis=0)

# Calculate price changes relative to pre-merger baseline
price_changes_eff = post_merger_prices_eff_avg - pre_merger_prices_avg
pct_price_changes_eff = (price_changes_eff / pre_merger_prices_avg) * 100

merger_efficiency_df = pd.DataFrame({
    'Product': product_labels,
    'Type': ['Satellite', 'Satellite', 'Wired', 'Wired'],
    'Firm (Pre)': [1, 2, 3, 4],
    'Firm (Post)': [1, 1, 3, 4],
    'Pre-Merger Price': pre_merger_prices_avg,
    'Post-Merger Price': post_merger_prices_eff_avg,
    'Price Change ($)': price_changes_eff,
    'Price Change (%)': pct_price_changes_eff
})
numeric_columns_eff = ['Pre-Merger Price', 'Post-Merger Price', 'Price Change ($)', 'Price Change (%)']
merger_efficiency_df[numeric_columns_eff] = merger_efficiency_df[numeric_columns_eff].apply(pd.to_numeric, errors='coerce')
comparison_eff_df = pd.DataFrame({
    'Product': product_labels,
    'No Efficiency (%)': pct_price_changes,  # From Q11
    'With 15% Cost Cut (%)': pct_price_changes_eff,
    'Difference (pp)': pct_price_changes_eff - pct_price_changes
})
comparison_eff_df[['No Efficiency (%)', 'With 15% Cost Cut (%)', 'Difference (pp)']] = comparison_eff_df[['No Efficiency (%)', 'With 15% Cost Cut (%)', 'Difference (pp)']].apply(pd.to_numeric, errors='coerce')
IPython.display.display(merger_efficiency_df.style.format({
    'Pre-Merger Price': '{:,.4f}'.format,
    'Post-Merger Price': '{:,.4f}'.format,
    'Price Change ($)': '{:,.4f}'.format,
    'Price Change (%)': '{:,.2f}'.format,
}))
IPython.display.display(comparison_eff_df.style.format({
    'No Efficiency (%)': '{:,.2f}'.format,
    'With 15% Cost Cut (%)': '{:,.2f}'.format,
    'Difference (pp)': '{:,.2f}'.format,
}))

post_merger_shares_eff = optimal_iv_results2.compute_shares(post_merger_prices_efficiency)
post_merger_markups_eff = optimal_iv_results2.compute_markups(
    post_merger_prices_efficiency,
    marginal_costs_efficiency
)
post_merger_profits_eff = optimal_iv_results2.compute_profits(
    post_merger_prices_efficiency,
    post_merger_shares_eff,
    marginal_costs_efficiency
)
post_merger_cs_eff = optimal_iv_results2.compute_consumer_surpluses(post_merger_prices_efficiency)

efficiency_metrics = {
    'Average Price ($)': post_merger_prices_eff_avg.mean(),
    'Average Markup ($)': post_merger_markups_eff.reshape((T, J)).mean(),
    'Average Profit ($)': post_merger_profits_eff.mean(),
    'Average CS ($)': post_merger_cs_eff.mean(),
}
efficiency_metric_summary = pd.DataFrame({
    'Metric': metric_names,
    'No Efficiency (1&2)': [post_merger_metrics[m] for m in metric_names],
    '15% Cost Cut (1&2)': [efficiency_metrics[m] for m in metric_names]
})
efficiency_metric_summary['Difference (Eff - No Eff)'] = (
    efficiency_metric_summary['15% Cost Cut (1&2)'] - efficiency_metric_summary['No Efficiency (1&2)']
)
efficiency_metric_summary = efficiency_metric_summary.astype({
    'No Efficiency (1&2)': float,
    '15% Cost Cut (1&2)': float,
    'Difference (Eff - No Eff)': float,
})
efficiency_metric_summary = efficiency_metric_summary.set_index('Metric')

IPython.display.display(efficiency_metric_summary.style.format('{:,.3f}'))

# ========================================================================
# WELFARE ANALYSIS
# ========================================================================

# Compute consumer surplus for different scenarios using PyBLP
# Pre-merger consumer surplus (baseline)
cs_pre = optimal_iv_results2.compute_consumer_surpluses()

# Post-merger WITHOUT efficiencies (from Question 11)
cs_post_no_eff = optimal_iv_results2.compute_consumer_surpluses(prices=post_merger_prices)

# Post-merger WITH 15% cost reduction
cs_post_with_eff = optimal_iv_results2.compute_consumer_surpluses(prices=post_merger_prices_efficiency)

# Calculate changes in consumer surplus (per consumer, per market)
delta_cs_no_eff = cs_post_no_eff - cs_pre
delta_cs_with_eff = cs_post_with_eff - cs_pre

# Aggregate across all markets and consumers
# Total CS change = M_t * sum over all markets
total_delta_cs_no_eff = M_t * delta_cs_no_eff.sum()
total_delta_cs_with_eff = M_t * delta_cs_with_eff.sum()

# Average per consumer across markets
avg_delta_cs_no_eff = delta_cs_no_eff.mean()
avg_delta_cs_with_eff = delta_cs_with_eff.mean()

# Compute producer surplus changes using PyBLP
# Producer surplus = (p - mc) × shares × M_t for each product
# Pre-merger
ps_pre = optimal_iv_results2.compute_profits()

# Post-merger without efficiency
shares_post_no_eff = optimal_iv_results2.compute_shares(prices=post_merger_prices)
ps_post_no_eff = optimal_iv_results2.compute_profits(
    prices=post_merger_prices,
    shares=shares_post_no_eff,
    costs=marginal_costs
)

# Post-merger with 15% efficiency
shares_post_eff = optimal_iv_results2.compute_shares(prices=post_merger_prices_efficiency)
ps_post_eff = optimal_iv_results2.compute_profits(
    post_merger_prices_efficiency,
    shares_post_eff,
    marginal_costs_efficiency
)

# Total PS changes = M_t * sum over all products/markets
delta_ps_no_eff = M_t * (ps_post_no_eff - ps_pre).sum()
delta_ps_with_eff = M_t * (ps_post_eff - ps_pre).sum()

# Total welfare changes
delta_w_no_eff = total_delta_cs_no_eff + delta_ps_no_eff
delta_w_with_eff = total_delta_cs_with_eff + delta_ps_with_eff

welfare_summary = pd.DataFrame({
    'Scenario': ['Without efficiency', 'With 15% cost cut'],
    'ΔCS total ($)': [total_delta_cs_no_eff, total_delta_cs_with_eff],
    'ΔPS ($)': [delta_ps_no_eff, delta_ps_with_eff],
    'ΔW ($)': [delta_w_no_eff, delta_w_with_eff],
})
welfare_summary['Verdict'] = welfare_summary['ΔW ($)'].apply(lambda x: 'ENHANCING' if x >= 0 else 'REDUCING')
numeric_columns_welfare = ['ΔCS total ($)', 'ΔPS ($)', 'ΔW ($)']
welfare_summary[numeric_columns_welfare] = welfare_summary[numeric_columns_welfare].apply(pd.to_numeric, errors='coerce')
welfare_summary = welfare_summary.set_index('Scenario')
IPython.display.display(welfare_summary.style.format({
    'ΔCS total ($)': '{:,.2f}'.format,
    'ΔPS ($)': '{:,.2f}'.format,
    'ΔW ($)': '{:,.2f}'.format,
}))

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ========================================================================
# PUBLICATION-READY FIGURE: Surplus Changes by Scenario
# Black and White Version with Seaborn
# ========================================================================

# Set Seaborn style and context
sns.set_style("whitegrid", {
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'grid.color': '#666666',
    'grid.alpha': 0.4,
    'axes.linewidth': 1.0,
    'axes.edgecolor': 'black',
})
sns.set_context("paper", font_scale=1.15)

# Additional matplotlib parameters for publication quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Prepare data for publication figure
scenarios = ['No Efficiency\nGains', 'With 15%\nCost Reduction']
delta_cs = [total_delta_cs_no_eff, total_delta_cs_with_eff]
delta_ps = [abs(delta_ps_no_eff), abs(delta_ps_with_eff)]  # Take absolute value
delta_w = [delta_w_no_eff, delta_w_with_eff]

# Create figure with optimal proportions
fig, ax = plt.subplots(figsize=(9, 6))

# Set up bar positions
x_pos = np.arange(len(scenarios))
width = 0.5

# Black and white colors with different patterns
colors = {
    'CS': '#000000',  # Black for consumer surplus
    'PS': '#666666',  # Dark gray for producer surplus
    'W': '#CCCCCC',   # Light gray for total welfare (readable!)
}

# Create stacked bars with hatching patterns for differentiation
bars_cs = ax.bar(x_pos, delta_cs, width,
                 label='Consumer Surplus (ΔCS)',
                 color=colors['CS'],
                 edgecolor='black',
                 linewidth=1.5,
                 alpha=1.0)

bars_ps = ax.bar(x_pos, delta_ps, width,
                 bottom=delta_cs,
                 label='Producer Surplus (ΔPS)',
                 color=colors['PS'],
                 edgecolor='black',
                 linewidth=1.5,
                 hatch='///',  # Diagonal hatching
                 alpha=1.0)

# Add horizontal line at zero with emphasis
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, zorder=3)

# Enhanced value annotations
for i, (cs, ps, w) in enumerate(zip(delta_cs, delta_ps, delta_w)):
    total_height = cs + ps

    # Component values - only show if segment is large enough
    min_size = 4000

    if abs(cs) > min_size:
        ax.text(i, cs/2, f'${cs:,.0f}',
                ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)

    if abs(ps) > min_size:
        ax.text(i, cs + ps/2, f'${ps:,.0f}',
                ha='center', va='center',
                color='white',
                fontweight='bold', fontsize=10)

    # Add ΔW annotation to the right of each bar
    x_end = i + width/2 + 0.15
    y_annotation = total_height

    # Draw connecting line
    ax.plot([i, x_end], [total_height, y_annotation],
            color='#000000', linewidth=1, linestyle='-', alpha=0.7, zorder=2)

    # Add the ΔW label with light gray background
    bbox_props = dict(boxstyle='round,pad=0.5',
                     facecolor=colors['W'],
                     edgecolor='black',
                     linewidth=1.5)
    ax.text(x_end + 0.05, y_annotation, f'ΔW = ${w:,.0f}',
            ha='left', va='center',
            fontweight='bold', fontsize=10,
            color='black',  # Black text on light gray
            bbox=bbox_props)

# Formatting
ax.set_ylabel('Aggregate Surplus Change ($)', fontweight='bold', fontsize=13)
ax.set_title('Welfare Effects of Merger: Firms 1 & 2',
             fontweight='bold', pad=20, fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios, fontweight='semibold')

# Y-axis formatting
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Extend x-axis
ax.set_xlim(-0.5, len(scenarios) - 0.2)

# Legend with patterns
legend_elements = [
    Patch(facecolor=colors['CS'], edgecolor='black', linewidth=1.5,
          label='Change in Consumer Surplus (ΔCS)', alpha=1.0),
    Patch(facecolor=colors['PS'], edgecolor='black', linewidth=1.5,
          label='Change in Producer Surplus (ΔPS)', hatch='///', alpha=1.0),
    Patch(facecolor=colors['W'], edgecolor='black', linewidth=1.5,
          label='Change in Total Welfare (ΔW)', alpha=1.0),
]

legend = ax.legend(handles=legend_elements, loc='upper left',
                   framealpha=1.0, edgecolor='black',
                   fancybox=False, shadow=False, borderpad=1)
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_facecolor('white')

# Clean spines
sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

plt.tight_layout()

# Save
plt.savefig('surplus_changes_merger.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('surpus_changes_merger.png', format='png', bbox_inches='tight', dpi=300)
print('✓ Publication-ready black and white figures saved')

plt.show()

# Reset to default style
sns.reset_defaults()

# ------------------------------------------------------------------------
# Consolidated diagnostics across all merger scenarios
# ------------------------------------------------------------------------

scenario_metric_table = pd.DataFrame({
    'Metric': metric_names,
    'Pre-Merger Baseline': [baseline_metrics[m] for m in metric_names],
    'Merger 1&2 (No Eff)': [post_merger_metrics[m] for m in metric_names],
    'Merger 1&3 (Cross)': [cross_merger_metrics[m] for m in metric_names],
    'Merger 1&2 (15% Cost Cut)': [efficiency_metrics[m] for m in metric_names],
})
scenario_metric_table['Δ (1&2 - Pre)'] = scenario_metric_table['Merger 1&2 (No Eff)'] - scenario_metric_table['Pre-Merger Baseline']
scenario_metric_table['Δ (1&3 - Pre)'] = scenario_metric_table['Merger 1&3 (Cross)'] - scenario_metric_table['Pre-Merger Baseline']
scenario_metric_table['Δ (15% Cut - Pre)'] = scenario_metric_table['Merger 1&2 (15% Cost Cut)'] - scenario_metric_table['Pre-Merger Baseline']

numeric_columns = scenario_metric_table.select_dtypes(include='number').columns
scenario_metric_table[numeric_columns] = scenario_metric_table[numeric_columns].apply(pd.to_numeric, errors='coerce')
format_dict = {col: '{:,.3f}'.format for col in numeric_columns}
IPython.display.display(scenario_metric_table.style.format(format_dict))

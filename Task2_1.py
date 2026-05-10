import numpy as np
import gurobipy as gp
from gurobipy import GRB

# -----------------------------
# 1. Generate load profiles
# -----------------------------

np.random.seed(42)

n_profiles = 300
n_minutes = 60

min_load = 220
max_load = 600
max_step = 35

profiles = np.zeros((n_profiles, n_minutes))

for w in range(n_profiles):
    profiles[w, 0] = np.random.uniform(min_load, max_load)

    for m in range(1, n_minutes):
        step = np.random.uniform(-max_step, max_step)
        profiles[w, m] = np.clip(profiles[w, m-1] + step, min_load, max_load)

in_sample = profiles[:100, :]
out_sample = profiles[100:, :]


# -----------------------------
# 2. ALSO-X 
# -----------------------------

def solve_alsox(load_profiles, epsilon=0.10, tol=1e-6, max_iter=100):
    """
    ALSO-X iterative solution for the scalar FCR-D UP bidding problem.

    Since the only decision variable is the reserve bid R, the iterative search
    checks whether a candidate R violates the P90 budget.
    """
    values = load_profiles.flatten()
    n_total = len(values)

    max_violations = int(epsilon * n_total)

    lower = 0.0
    upper = np.max(values)

    for _ in range(max_iter):
        candidate = 0.5 * (lower + upper)

        violations = np.sum(values < candidate)

        if violations <= max_violations:
            lower = candidate
        else:
            upper = candidate

        if upper - lower <= tol:
            break

    R_value = lower
    violations = np.sum(values < R_value)
    violation_rate = violations / n_total

    return R_value, violations, violation_rate

# -----------------------------
# 3. CVaR approximation
# -----------------------------

def solve_cvar_p90(load_profiles, epsilon=0.10):
    n_scenarios, n_minutes = load_profiles.shape
    n_total = n_scenarios * n_minutes

    model = gp.Model("cvar_p90")
    model.Params.OutputFlag = 0

    R = model.addVar(lb=0, ub=600, name="R")
    eta = model.addVar(lb=-GRB.INFINITY, name="eta")
    s = model.addVars(n_scenarios, n_minutes, lb=0, name="s")

    for w in range(n_scenarios):
        for m in range(n_minutes):
            model.addConstr(
                s[w, m] >= R - load_profiles[w, m] - eta,
                name=f"cvar_shortfall_{w}_{m}"
            )

    model.addConstr(
        eta + (1 / (epsilon * n_total))
        * gp.quicksum(s[w, m] for w in range(n_scenarios) for m in range(n_minutes))
        <= 0,
        name="cvar_constraint"
    )

    model.setObjective(R, GRB.MAXIMIZE)
    model.optimize()

    R_value = R.X
    violations = np.sum(load_profiles < R_value)
    violation_rate = violations / n_total

    return R_value, violations, violation_rate


# -----------------------------
# 4. Solve Task 2.1
# -----------------------------

R_alsox, viol_alsox, rate_alsox = solve_alsox(in_sample)
R_cvar, viol_cvar, rate_cvar = solve_cvar_p90(in_sample)

print("Task 2.1 results")
print("----------------")
print(f"ALSO-X bid  : {R_alsox:.2f} kW")
print(f"In-sample shortfalls: {viol_alsox}")
print(f"In-sample shortfall rate: {100 * rate_alsox:.2f}%")
print()
print(f"CVaR bid: {R_cvar:.2f} kW")
print(f"In-sample shortfalls: {viol_cvar}")
print(f"In-sample shortfall rate: {100 * rate_cvar:.2f}%")
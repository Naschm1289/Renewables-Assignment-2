import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

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
        profiles[w, m] = np.clip(profiles[w, m - 1] + step, min_load, max_load)

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


# -----------------------------
# 5. Task 2.2 verification
# -----------------------------

def evaluate_shortfalls(bid_kw, profile_matrix, epsilon=0.10):
    """
    Compares the chosen reserve bid with minute-level consumption profiles
    to evaluate possible shortfalls.
    """
    shortfall_mask = profile_matrix < bid_kw

    minute_shortfalls = shortfall_mask.sum()
    minute_shortfall_rate = shortfall_mask.mean()
    p90_met_minute = minute_shortfall_rate <= epsilon

    profile_shortfalls = shortfall_mask.any(axis=1).sum()
    profile_shortfall_rate = shortfall_mask.any(axis=1).mean()
    p90_met_profile = profile_shortfall_rate <= epsilon

    # NEW: shortfall magnitude metrics
    shortfall_magnitudes = np.maximum(0, bid_kw - profile_matrix)
    ers = np.mean(shortfall_magnitudes)  # Expected Reserve Shortfall
    max_shortfall = np.max(shortfall_magnitudes)  # Worst-case shortfall
    conditional_shortfall = (
        shortfall_magnitudes[shortfall_mask].mean() if minute_shortfalls > 0 else 0
    )

    return (
        minute_shortfalls,
        minute_shortfall_rate,
        p90_met_minute,
        profile_shortfalls,
        profile_shortfall_rate,
        p90_met_profile,
        ers,
        max_shortfall,
        conditional_shortfall
    )


alsox_shortfalls, alsox_minute_rate, alsox_minute_met, alsox_profile_shortfalls, alsox_profile_rate, alsox_profile_met, alsox_ers, alsox_max_shortfall, alsox_conditional_shortfall = evaluate_shortfalls(
    R_alsox, out_sample
)
cvar_shortfalls, cvar_minute_rate, cvar_minute_met, cvar_profile_shortfalls, cvar_profile_rate, cvar_profile_met, cvar_ers, cvar_max_shortfall, cvar_conditional_shortfall = evaluate_shortfalls(
    R_cvar, out_sample
)

print("\n" + "=" * 70)
print("TASK 2.2: Out-of-Sample Verification (200 Profiles)")
print("=" * 70)

print("\n--- ALSO-X Solution ---")
print(f"Chosen Reserve Bid:         {R_alsox:.2f} kW")
print(f"Minute-Level Shortfalls:    {alsox_shortfalls}")
print(f"Minute-Level Shortfall Rate:{alsox_minute_rate:.2%} (Limit: 10.00%)")
print(f"P90 Requirement Met:        {'YES' if alsox_minute_met else 'NO'}")
print(f"ERS (avg shortfall):        {alsox_ers:.2f} kW")
print(f"Max shortfall:              {alsox_max_shortfall:.2f} kW")
print(f"Conditional shortfall:      {alsox_conditional_shortfall:.2f} kW")
print(f"-> (Bonus: Profile-Level Shortfalls: {alsox_profile_shortfalls})")
print(f"-> (Bonus: Profile-Level Shortfalls Rate: {alsox_profile_rate:.2%})")

print("\n--- CVaR Solution ---")
print(f"Chosen Reserve Bid:         {R_cvar:.2f} kW")
print(f"Minute-Level Shortfalls:    {cvar_shortfalls}")
print(f"Minute-Level Shortfall Rate:{cvar_minute_rate:.2%} (Limit: 10.00%)")
print(f"P90 Requirement Met:        {'YES' if cvar_minute_met else 'NO'}")
print(f"ERS (avg shortfall):        {cvar_ers:.2f} kW")
print(f"Max shortfall:              {cvar_max_shortfall:.2f} kW")
print(f"Conditional shortfall:      {cvar_conditional_shortfall:.2f} kW")
print(f"-> (Bonus: Profile-Level Shortfalls: {cvar_profile_shortfalls})")
print(f"-> (Bonus: Profile-Level Shortfalls Rate: {cvar_profile_rate:.2%})")
print("=" * 70)


# Plot target from your report statement
bid_for_plot = 247.96

# Compute minute-level shortfalls vs the fixed bid
shortfalls = np.maximum(0, bid_for_plot - out_sample)

# Pick the exact scenario+minute with the largest shortfall
w_idx, m_idx = np.unravel_index(np.argmax(shortfalls), shortfalls.shape)
max_shortfall_plot = shortfalls[w_idx, m_idx]

if max_shortfall_plot <= 0:
    print(f"No out-of-sample shortfalls found below {bid_for_plot:.2f} kW.")
else:
    minutes = np.arange(out_sample.shape[1])
    load_data = out_sample[w_idx, :]
    min_load_at_worst_minute = out_sample[w_idx, m_idx]

    plt.figure(figsize=(11, 5))

    # Load profile
    plt.plot(minutes, load_data, color="tab:blue", linewidth=2, label="Actual Load Profile")

    # Bid line
    plt.axhline(
        y=bid_for_plot,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Reserve Bid = {bid_for_plot:.2f} kW"
    )

    # Shade shortfall area (continuous at threshold crossings)
    shortfall_mask_profile = load_data < bid_for_plot
    plt.fill_between(
        minutes,
        load_data,
        bid_for_plot,
        where=shortfall_mask_profile,
        interpolate=True,   # fixes visible gaps near crossing points
        color="red",
        alpha=0.30,
        linewidth=0,
        zorder=1,
        label="Reserve Shortfall"
    )

    # Mark the worst minute in this scenario
    plt.scatter(
        [m_idx],
        [min_load_at_worst_minute],
        color="darkred",
        zorder=5,
        label="Worst Minute"
    )

    # Vertical marker showing the shortfall magnitude
    plt.vlines(
        x=m_idx,
        ymin=min_load_at_worst_minute,
        ymax=bid_for_plot,
        color="darkred",
        linewidth=2
    )

    plt.annotate(
        f"Max Shortfall = {max_shortfall_plot:.2f} kW\n"
        f"({bid_for_plot:.2f} - {min_load_at_worst_minute:.2f})",
        xy=(m_idx, (bid_for_plot + min_load_at_worst_minute) / 2),
        xytext=(m_idx + 4, bid_for_plot + 18),
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
        fontsize=10,
        color="darkred",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkred", alpha=0.9)
    )

    plt.title(
        f"Out-of-Sample Profile with Largest Shortfall (Scenario {w_idx}, Minute {m_idx})",
        fontsize=13
    )
    plt.xlabel("Minute")
    plt.ylabel("Consumption (kW)")
    plt.ylim(200, 400)  # Adjust y-axis limits for better visibility
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
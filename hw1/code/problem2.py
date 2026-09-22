import numpy as np
import matplotlib.pyplot as plt

DTYPE = np.float32
EPS_MACH_32 = 1.1920929e-07

a, b = np.float32(0.0), np.float32(1.0)
I_exact = 1.0 - np.exp(-1.0)  # exact value of integral_0^1 exp(-t) dt


def f(x):
    return np.exp(-x, dtype=DTYPE)

def midpoint_rule(f, a, b, N):
    a, b = to32(a), to32(b)
    h = (b - a) / to32(N)
    i = np.arange(N, dtype=DTYPE)
    x_mid = a + (i + to32(0.5)) * h
    fx = f(x_mid)
    total = np.float32(0.0)
    for val in fx:  
        total = total + val
    return total * h

def trapezoid_rule(f, a, b, N):
    a, b = to32(a), to32(b)
    h = (b - a) / to32(N)
    x = a + np.arange(N + 1, dtype=DTYPE) * h
    fx = f(x)
    total = (fx[0] + fx[-1]) / to32(2.0)
    for val in fx[1:-1]:
        total = total + val
    return total * h

def simpson_rule(f, a, b, N):
    if N % 2 != 0:
        N += 1
    a, b = to32(a), to32(b)
    h = (b - a) / to32(N)
    x = a + np.arange(N + 1, dtype=DTYPE) * h
    fx = f(x)
    total = fx[0] + fx[-1]
    for i in range(1, N):
        coeff = to32(4.0) if i % 2 == 1 else to32(2.0)
        total = total + coeff * fx[i]
    return total * h / to32(3.0)

def to32(x):
    return np.float32(x)

rules = {
    "Midpoint": midpoint_rule,
    "Trapezoid": trapezoid_rule,
    "Simpson": simpson_rule,
}

N_values = np.unique(np.round(np.logspace(0.5, 6.3, 120)).astype(int))
N_values = N_values[N_values >= 2]

results = {name: {"N": [], "eps": []} for name in rules}

for N in N_values:
    for name, rule in rules.items():
        I_num = np.float64(rule(f, a, b, N))
        eps = abs((I_num - I_exact) / I_exact)
        if eps == 0:
            continue
        results[name]["N"].append(N)
        results[name]["eps"].append(eps)

fig, ax = plt.subplots(figsize=(7.5, 6))

colors = {"Midpoint": "tab:blue", "Trapezoid": "tab:orange", "Simpson": "tab:green"}
for name in rules:
    N_arr = np.array(results[name]["N"])
    eps_arr = np.array(results[name]["eps"])
    ax.loglog(N_arr, eps_arr, ".", ms=4, color=colors[name], label=name, alpha=0.8)

# reference slopes
N_ref = np.logspace(0.5, 6.3, 50)
ax.loglog(N_ref, 0.5 * (N_ref / 10) ** -2, "k--", lw=1, alpha=0.6, label=r"$N^{-2}$")
ax.loglog(N_ref, 1e-4 * (N_ref / 10) ** -4, "k:", lw=1, alpha=0.6, label=r"$N^{-4}$")
ax.loglog(N_ref, EPS_MACH_32 * np.sqrt(N_ref), "k-.", lw=1, alpha=0.6, label=r"$\sqrt{N}\,\epsilon_{mach}$")

ax.set_xlabel("number of bins N")
ax.set_ylabel(r"relative error $\epsilon = |D_{num} - D_{exact}| / D_{exact}$")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("../figures/problem2.png", dpi=150)

print(f"\n{'Rule':<10} {'N_opt':<10} {'min eps':<12}")
for name in rules:
    N_arr = np.array(results[name]["N"])
    eps_arr = np.array(results[name]["eps"])
    i_min = np.argmin(eps_arr)
    print(f"{name:<10} {N_arr[i_min]:<10d} {eps_arr[i_min]:<12.3e}")
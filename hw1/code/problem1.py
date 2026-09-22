import numpy as np
import matplotlib.pyplot as plt

DTYPE = np.float32

def to32(x):
    return np.float32(x)

# Functions and their analytic derivatives
functions = {
    "cos(x)": {
        "f": lambda x: np.cos(x, dtype=DTYPE),
        "fprime": lambda x: -np.sin(x, dtype=DTYPE),
    },
    "exp(x)": {
        "f": lambda x: np.exp(x, dtype=DTYPE),
        "fprime": lambda x: np.exp(x, dtype=DTYPE),
    },
}

x_values = [to32(0.1), to32(10.0)]

# Step sizes: logarithmically spaced from very small to O(1), in single precision.
h_values = np.logspace(-8, 0, 200).astype(DTYPE)

def forward_diff(f, x, h):
    x = to32(x)
    h = to32(h)
    return (f(x + h) - f(x)) / h


def central_diff(f, x, h):
    x = to32(x)
    h = to32(h)
    return (f(x + h) - f(x - h)) / (to32(2.0) * h)


def extrapolated_diff(f, x, h):
    x = to32(x)
    h = to32(h)
    Dc_h = central_diff(f, x, h)
    Dc_h2 = central_diff(f, x, h / to32(2.0))
    return (to32(4.0) * Dc_h2 - Dc_h) / to32(3.0)

algorithms = {
    "Forward": forward_diff,
    "Central": central_diff,
    "Extrapolated": extrapolated_diff,
}

def relative_error(D_num, D_exact):
    D_num = np.float64(D_num)      
    D_exact = np.float64(D_exact)  
    return np.abs((D_num - D_exact) / D_exact)

results = {}  

for fname, fdict in functions.items():
    f = fdict["f"]
    fprime = fdict["fprime"]
    results[fname] = {}
    for x in x_values:
        D_exact = fprime(x)
        results[fname][round(float(x), 6)] = {}
        for aname, algo in algorithms.items():
            eps = np.empty_like(h_values, dtype=np.float64)
            for i, h in enumerate(h_values):
                D_num = algo(f, x, h)
                eps[i] = relative_error(D_num, D_exact)
            results[fname][round(float(x), 6)][aname] = (np.float64(h_values), eps)

eps_mach = np.float32(1.0).view(np.uint32)
EPS_MACH_32 = 1.1920929e-07

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

panel_specs = [
    ("cos(x)", 0.1, axes[0, 0]),
    ("cos(x)", 10.0, axes[0, 1]),
    ("exp(x)", 0.1, axes[1, 0]),
    ("exp(x)", 10.0, axes[1, 1]),
]

colors = {"Forward": "tab:blue", "Central": "tab:orange", "Extrapolated": "tab:green"}

for fname, xval, ax in panel_specs:
    for aname in algorithms:
        h_arr, eps_arr = results[fname][xval][aname]
        ax.loglog(h_arr, eps_arr, label=aname, color=colors[aname], lw=1.5)

    h_ref = np.logspace(-8, 0, 50)
    h0 = 1e-2
    ax.loglog(h_ref, (h_ref / h0) * 1e-2 * 0.5, "k:", lw=1, alpha=0.6)      # ~h
    ax.loglog(h_ref, (h_ref / h0) ** 2 * 1e-4 / 6, "k--", lw=1, alpha=0.6)  # ~h^2
    ax.loglog(h_ref, EPS_MACH_32 / h_ref, "k-.", lw=1, alpha=0.6)          # ~1/h roundoff

    ax.set_title(f"{fname}, x = {xval}")
    ax.set_xlabel("step size h")
    ax.set_ylabel("relative error  |eps|")
    ax.set_ylim(1e-9, 10)
    ax.legend(fontsize=8, loc="lower left")

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("../figures/problem1.png", dpi=150)

print(f"\n{'Function':<8} {'x':<6} {'Algorithm':<13} {'h_opt':<12} {'min eps':<12} {'sig. digits':<10}")
for fname, xval, ax in panel_specs:
    for aname in algorithms:
        h_arr, eps_arr = results[fname][xval][aname]
        i_min = np.argmin(eps_arr)
        sig_digits = -np.log10(eps_arr[i_min]) if eps_arr[i_min] > 0 else np.inf
        print(f"{fname:<8} {xval:<6} {aname:<13} {h_arr[i_min]:<12.3e} {eps_arr[i_min]:<12.3e} {sig_digits:<10.2f}")
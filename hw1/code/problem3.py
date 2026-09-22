import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
import matplotlib.pyplot as plt

data = np.loadtxt("../data/lcdm_z0.matter_pk")
k_tab = data[:, 0]
P_tab = data[:, 1]

print(f"Loaded {len(k_tab)} rows. k range: [{k_tab.min():.3e}, {k_tab.max():.3e}] h/Mpc")

logk_tab = np.log10(k_tab)
logP_tab = np.log10(P_tab)
spline_logP = CubicSpline(logk_tab, logP_tab, extrapolate=False)

def P_of_k(k):
    k = np.atleast_1d(k).astype(float)
    logk = np.log10(np.clip(k, k_tab.min(), k_tab.max()))
    logP = spline_logP(logk)
    P = 10.0 ** logP
    P[(k < k_tab.min()) | (k > k_tab.max())] = 0.0
    return P

def xi_of_r(r, k_min=None, k_max=None, n_k=200000):
    if k_min is None:
        k_min = k_tab.min()
    if k_max is None:
        k_max = k_tab.max()

    logk = np.linspace(np.log10(k_min), np.log10(k_max), n_k)
    k = 10.0 ** logk
    Pk = P_of_k(k)

    kr = k * r
    sinc = np.sinc(kr / np.pi)  # numpy sinc is normalized: sinc(x)=sin(pi x)/(pi x)

    integrand_dlnk = (k ** 3) * Pk * sinc / (2.0 * np.pi ** 2)  # extra k from dk = k dlnk
    # integrate over ln(k) = ln(10) * log10(k)
    val = simpson(integrand_dlnk, x=logk) * np.log(10.0)
    return val

r_test = 100.0
k_max_table = k_tab.max()
test_kmax = [1.0, 5.0, 10.0, 50.0, k_max_table]
print("\nConvergence check of xi(r=100) vs upper k-limit:")
for kmx in test_kmax:
    val = xi_of_r(r_test, k_max=kmx)
    print(f"  k_max = {kmx:>10.3g}  ->  xi(100) = {val:.6e}")

r_values = np.linspace(50.0, 120.0, 400)
xi_values = np.array([xi_of_r(r) for r in r_values])

r2xi = r_values ** 2 * xi_values

trough_mask = (r_values > 70.0) & (r_values < 95.0)
i_trough = np.where(trough_mask)[0][np.argmin(r2xi[trough_mask])]

search_mask = r_values > r_values[i_trough]
i_peak = np.where(search_mask)[0][np.argmax(r2xi[search_mask])]

print(f"\nBroadband trough located at r = {r_values[i_trough]:.2f} Mpc/h")

i1 = int(np.clip(i_peak, 1, len(r_values) - 2))
i0, i2 = i1 - 1, i1 + 1
r0, r1, r2v = r_values[i0], r_values[i1], r_values[i2]
y0, y1, y2 = r2xi[i0], r2xi[i1], r2xi[i2]
denom = (y0 - 2 * y1 + y2)
if denom != 0:
    r_peak = r1 + 0.5 * (y0 - y2) / denom * (r1 - r0)
else:
    r_peak = r1
peak_val = np.interp(r_peak, r_values, r2xi)

print(f"\nBAO peak (grid maximum):     r = {r_values[i_peak]:.2f} Mpc/h")
print(f"BAO peak (parabolic refine): r = {r_peak:.2f} Mpc/h")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(r_values, r2xi, color="tab:blue", lw=1.8)
ax.axvline(r_peak, color="tab:red", ls="--", lw=1.3,
           label=f"BAO peak: r = {r_peak:.1f} Mpc/h")
ax.plot([r_peak], [peak_val], "ro", ms=6)
ax.set_xlabel(r"$r$  [Mpc/h]")
ax.set_ylabel(r"$r^2\, \xi(r)$  [(Mpc/h)$^2$]")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("../figures/problem3.png", dpi=150)
print("\nSaved figures/problem3.png")

# out = np.column_stack([r_values, xi_values, r2xi])
# np.savetxt("../figures/xi_r_output.txt", out, header="r[Mpc/h]  xi(r)  r^2*xi(r)", fmt="%.6e")
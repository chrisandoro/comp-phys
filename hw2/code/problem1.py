import numpy as np

c = 2.0
f  = lambda x: 1 - np.exp(-c * x)
fp = lambda x: c * np.exp(-c * x)          


def relax(f, fp, x, omega=0.0, tol=1e-6):
    n = 0
    while True:
        xnew = (1 + omega) * f(x) - omega * x
        gp = (1 + omega) * fp(x) - omega
        err = (x - xnew) / (1 - 1 / gp)
        x = xnew
        n += 1
        if abs(err) < tol:
            return x, n


x, n = relax(f, fp, 1.0, 0.0)
print(f"relaxation (w=0):  x = {x:.6f}, iterations = {n}")

print("overrelaxation:")
for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    x, n = relax(f, fp, 1.0, w)
    print(f"    w = {w:.1f}: x = {x:.6f}, iterations = {n}")

g  = lambda x: np.exp(-x)
gd = lambda x: -np.exp(-x)
print("(d) x = exp(-x):")
for w in [0.0, -0.2, -0.36, -0.5]:
    x, n = relax(g, gd, 1.0, w)
    print(f"w = {w:5.2f}: x = {x:.6f}, iterations = {n}")
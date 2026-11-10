import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

a, b = 1.0, 3.0
delta = 1e-10          
nmax = 8               

def f(r):
    x, y = r
    return np.array([1 - (b + 1) * x + a * x**2 * y,
                     b * x - a * x**2 * y])

def modified_midpoint(r, H, n):
    h = H / n
    k = r + 0.5 * h * f(r)
    rr = r + h * f(k)
    for _ in range(n - 1):
        k = k + h * f(rr)
        rr = rr + h * f(k)
    return 0.5 * (k + rr + 0.5 * h * f(rr))

times, points = [0.0], [np.array([0.0, 0.0])]  
nsteps_used = []

def bs_step(r, t, H):
    R = [[modified_midpoint(r, H, 1)]]
    for n in range(2, nmax + 1):
        row = [modified_midpoint(r, H, n)]
        for m in range(1, n):
            eps = (row[m - 1] - R[n - 2][m - 1]) / ((n / (n - 1.0))**(2 * m) - 1)
            row.append(row[m - 1] + eps)
        R.append(row)
        err = np.abs(eps)              
        if np.all(err < H * delta):     
            rnew = row[-1]
            times.append(t + H); points.append(rnew); nsteps_used.append(n)
            return rnew
    r1 = bs_step(r, t, H / 2)
    return bs_step(r1, t + H / 2, H / 2)

r0 = np.array([0.0, 0.0])
bs_step(r0, 0.0, 20.0)

T = np.array(times); P = np.array(points)
print("number of intervals:", len(T) - 1)
print("smallest / largest interval: %.3e / %.3e" % (np.diff(T).min(), np.diff(T).max()))
print("x(20), y(20) =", P[-1])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(T, P[:, 0], "o-", ms=4, lw=1, label="$x$")
ax.plot(T, P[:, 1], "o-", ms=4, lw=1, label="$y$")
ax.set_xlabel("time $t$"); 
ax.set_ylabel("concentration"); 
ax.legend()
fig.tight_layout(); 
fig.savefig("brusselator.png", dpi=200)
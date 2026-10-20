import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

y = np.loadtxt("../data/dow.txt")
N = len(y)
c = np.fft.rfft(y)
print("N =", N, " len(c) =", len(c))

def smooth(frac):
    c2 = c.copy()
    keep = int(frac * len(c))
    c2[keep:] = 0
    return np.fft.irfft(c2, n=N), keep

fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
ax[0].plot(y, lw=0.8); ax[0].set_ylabel("Dow Jones")
for a, frac in zip(ax[1:], (0.10, 0.02)):
    s, keep = smooth(frac)
    a.plot(y, color="tab:blue", lw=0.8, label="data")
    a.plot(s, color="tab:red", lw=1.5, label="first %d%% of coefficients (%d kept)" % (round(frac*100), keep))
    a.set_ylabel("Dow Jones"); a.legend()
    print(frac, keep, "rms residual = %.1f" % np.sqrt(np.mean((y - s) ** 2)))
ax[2].set_xlabel("business day")
plt.tight_layout(); plt.savefig("fig_dow.pdf")
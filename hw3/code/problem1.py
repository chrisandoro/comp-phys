import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.loadtxt("../data/sunspots.txt")
month, spots = data[:, 0], data[:, 1]
N = len(spots)

plt.figure(figsize=(7, 3.5))
plt.plot(month, spots, lw=0.7)
plt.xlabel("month since January 1749"); plt.ylabel("sunspot number")
plt.tight_layout(); plt.savefig("fig_sunspots_series.pdf"); plt.close()

w = 24
smooth = np.convolve(spots, np.ones(w) / w, mode="same")
peaks = [i for i in range(w, N - w) if smooth[i] == smooth[i - w:i + w + 1].max()]
print("peaks (month):", peaks)
print("mean spacing between peaks: %.1f months" % np.mean(np.diff(peaks)))

c = np.fft.rfft(spots)
power = np.abs(c) ** 2
k = np.arange(len(c))
plt.figure(figsize=(7, 3.5))
plt.plot(k[1:300], power[1:300], lw=0.8)
plt.xlabel("$k$"); plt.ylabel("$|c_k|^2$"); plt.yscale("log")
plt.tight_layout(); plt.savefig("fig_sunspots_power.pdf"); plt.close()

kpk = 1 + np.argmax(power[1:])
print("N =", N, " peak k =", kpk, " period = %.2f months = %.2f years" % (N / kpk, N / kpk / 12))
y0, y1, y2 = np.log(power[kpk - 1:kpk + 2])
d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
print("refined k = %.2f, period = %.2f months" % (kpk + d, N / (kpk + d)))
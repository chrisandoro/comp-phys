import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sigma, eps = 25.0, 1e-3
b = np.loadtxt("../data/blur.txt")
K, L = b.shape

x = np.arange(L); y = np.arange(K)
x = np.minimum(x, L - x); y = np.minimum(y, K - y)
X, Y = np.meshgrid(x, y)
f = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

B = np.fft.rfft2(b)
F = np.fft.rfft2(f)
print("min |F| = %.3e, max |F| = %.3e" % (np.abs(F).min(), np.abs(F).max()))
mask = np.abs(F) >= eps
print("fraction of coefficients divided: %.3f" % mask.mean())
A = B.copy()
A[mask] = B[mask] / F[mask]
a = np.fft.irfft2(A, s=b.shape)

def show(ax, img, title):
    ax.imshow(img, cmap="gray"); ax.set_title(title); ax.axis("off")
fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
show(ax[0], b, "(a) blurred photo")
show(ax[1], f, "(b) point spread function")
show(ax[2], a, r"(c) deblurred, $\epsilon=10^{-3}$")
plt.tight_layout(); plt.savefig("fig_deblur.pdf"); plt.close()

fig, ax = plt.subplots(1, 4, figsize=(15, 4))
for axi, e in zip(ax, (1e-1, 1e-2, 1e-3, 1e-5)):
    m = np.abs(F) >= e
    A2 = B.copy(); A2[m] = B[m] / F[m]
    show(axi, np.fft.irfft2(A2, s=b.shape), r"$\epsilon=%g$" % e)
plt.tight_layout(); plt.savefig("fig_deblur_eps.pdf"); plt.close()
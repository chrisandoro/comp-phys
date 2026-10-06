import numpy as np
import matplotlib.pyplot as plt

def num_grad(f, p, h=1e-6):
    g = np.zeros_like(p)
    for i in range(len(p)):
        d = h * max(1.0, abs(p[i]))
        e = np.zeros_like(p); e[i] = d
        g[i] = (f(p + e) - f(p - e)) / (2 * d)
    return g

def gradient_descent(f, p0, lr0=1.0, tol=1e-10, maxit=100000, max_step=0.5):
    p = np.array(p0, float); fp = f(p)
    P, Fh = [p.copy()], [fp]
    lr = lr0
    for it in range(maxit):
        g = num_grad(f, p)
        gg = g @ g
        if gg == 0: break
        lr = min(2 * lr, max_step / np.sqrt(gg))
        while f(p - lr * g) > fp - 1e-4 * lr * gg:
            lr *= 0.5
            if lr < 1e-18: break
        pn = p - lr * g; fn = f(pn)
        P.append(pn.copy()); Fh.append(fn)
        done = (fp - fn) < tol * max(abs(fp), 1e-30) and it > 10
        p, fp = pn, fn
        if done: break
    return p, np.array(P), np.array(Fh)

logM, n, err = np.loadtxt('../data/smf_cosmos.dat', unpack=True)
M = 10 ** logM

def schechter(M, phi, Ms, alpha):
    x = M / Ms
    return phi * x ** (alpha + 1) * np.exp(-x) * np.log(10)

def unpack(q): return 10 ** q[0], 10 ** q[1], q[2]

def chi2(q):
    return np.sum(((n - schechter(M, *unpack(q))) / err) ** 2)

starts = {"A": [-3.0, 10.5, -1.0], "B": [-2.0, 11.5, -0.5],
          "C": [-4.0, 10.0, -1.5], "D": [-2.5, 11.0, 0.0]}
results = {}
for name, q0 in starts.items():
    q, Q, C = gradient_descent(chi2, q0, lr0=1e-3, tol=1e-12)
    results[name] = (q, C)
    phi, Ms, al = unpack(q)
    print(f"start {name} {q0}: initial chi2 = {C[0]:.1f}, steps = {len(C)-1}, "
          f"chi2 = {C[-1]:.4f}, phi* = {phi:.4e}, M* = {Ms:.4e}, alpha = {al:.4f}")

best = min(results, key=lambda k: results[k][1][-1])
qb = results[best][0]
phi_b, Ms_b, al_b = unpack(qb)
print(f"Best fit: phi* = {phi_b:.4e}, M* = {Ms_b:.4e} (log M* = {np.log10(Ms_b):.4f}), "
      f"alpha = {al_b:.4f}, chi2 = {chi2(qb):.3f}, dof = {len(n) - 3}")

plt.figure(figsize=(6, 4.5))
for name, (q, C) in results.items():
    plt.loglog(np.arange(len(C)) + 1, C, label=f'start {name}')
plt.xlabel('step i'); plt.ylabel(r'$\chi^2$'); plt.legend()
plt.tight_layout(); plt.savefig('chi2.png', dpi=150); plt.close()

plt.figure(figsize=(6, 4.5))
plt.errorbar(M, n, yerr=err, fmt='o', capsize=3, label='COSMOS data')
Mg = np.logspace(9.4, 12.0, 300)
plt.loglog(Mg, schechter(Mg, phi_b, Ms_b, al_b), 'r-', label='best-fit Schechter')
plt.xlabel(r'$M_{\rm gal}$')
plt.ylabel(r'$n(M_{\rm gal})$')
plt.legend(); plt.tight_layout(); plt.savefig('fit.png', dpi=150)
import numpy as np

h  = 6.62607015e-34     # Planck constant [J s]
c  = 2.99792458e8       # speed of light [m/s]
kB = 1.380649e-23       # Boltzmann constant [J/K]

F = lambda x: 5 * np.exp(-x) + x - 5

a, b_ = 1.0, 10.0
n = 0
while b_ - a > 1e-6:
    m = 0.5 * (a + b_)
    if F(m) * F(a) > 0:     
        a = m
    else:                  
        b_ = m
    n += 1
x = 0.5 * (a + b_)
print(f"x = {x:.6f} ({n} bisections)")

b = h * c / (kB * x)
print(f"b = {b:.5e} m K")

lam = 502e-9
print(f"T_sun = {b / lam:.0f} K")
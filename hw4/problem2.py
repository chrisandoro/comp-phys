
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.integrate import solve_ivp

G = 1.0
M_BH = 1.0
r_s = 1e-7
r_stop = 1e-3
TIME_UNIT_MYR = 1.49

def smbh_derivs(r, v, A, B):

    x, y = r
    vx, vy = v

    r_mag = np.hypot(x, y)
    v_mag = np.hypot(vx, vy)
    ax_grav = -(G * M_BH / (4.0 * r_mag**3)) * x
    ay_grav = -(G * M_BH / (4.0 * r_mag**3)) * y
    df_denom = v_mag**3 + B
    df_coeff = A / df_denom if df_denom > 0 else 0.0
    ax_df = -df_coeff * vx
    ay_df = -df_coeff * vy

    return (
        np.array([vx, vy]),
        np.array([ax_grav + ax_df, ay_grav + ay_df])
    )

def rk4_step(r, v, dt, A, B):

    dr1, dv1 = smbh_derivs(r, v, A, B)

    r2 = r + 0.5 * dt * dr1
    v2 = v + 0.5 * dt * dv1

    dr2, dv2 = smbh_derivs(r2, v2, A, B)

    r3 = r + 0.5 * dt * dr2
    v3 = v + 0.5 * dt * dv2

    dr3, dv3 = smbh_derivs(r3, v3, A, B)

    r4 = r + dt * dr3
    v4 = v + dt * dv3

    dr4, dv4 = smbh_derivs(r4, v4, A, B)

    r_next = r + (dt / 6.0) * (
        dr1 + 2 * dr2 + 2 * dr3 + dr4
    )

    v_next = v + (dt / 6.0) * (
        dv1 + 2 * dv2 + 2 * dv3 + dv4
    )

    return r_next, v_next

def adaptive_smbh_integrate(A, B, tol=1e-9):

    r = np.array([1.0, 0.0])

    v_circ = np.sqrt(G * M_BH / (4.0 * np.linalg.norm(r)))
    v = np.array([0.0, 0.8 * v_circ])

    t = 0.0
    dt = 0.01

    t_hist = [t]
    r_hist = [np.linalg.norm(r)]
    x_hist = [r[0]]
    y_hist = [r[1]]

    while np.linalg.norm(r) > r_s and t < 5000.0:

        r1, v1 = rk4_step(r, v, dt, A, B)

        r_mid, v_mid = rk4_step(r, v, dt * 0.5, A, B)
        r2, v2 = rk4_step(r_mid, v_mid, dt * 0.5, A, B)

        error = np.linalg.norm(r2 - r1) / 15.0

        if error < tol or dt < 1e-9:

            r, v = r2, v2
            t += dt

            t_hist.append(t)

            r_norm = np.linalg.norm(r)

            r_hist.append(r_norm)
            x_hist.append(r[0])
            y_hist.append(r[1])

        if error > 0:

            dt = 0.9 * dt * max(
                0.1,
                min(2.0, (tol / error)**0.25)
            )

        else:

            dt *= 1.1

    return (
        np.array(t_hist),
        np.array(r_hist),
        np.array(x_hist),
        np.array(y_hist)
    )

start = time.time()
t_arr, r_arr, x_arr, y_arr = adaptive_smbh_integrate(1.0, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x_arr, y_arr, color='#1f77b4')
axes[0].set_title('SMBH Binary Orbital Inspiral ($A=B=1$)')
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$y$')
axes[0].grid(True)
axes[0].set_aspect('equal', adjustable='datalim')

axes[1].plot(t_arr, np.log10(r_arr), color='#1f77b4')
axes[1].set_title('$\\log_{10}(r)$ vs Time')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('$\\log_{10}(r)$')
axes[1].grid(True)

plt.tight_layout()
plt.show()

def smbh_derivs_ivp(t, state, A, B):

    x, y, vx, vy = state

    r_mag = np.hypot(x, y)
    v_mag = np.hypot(vx, vy)

    ax_grav = -(G * M_BH / (4.0 * r_mag**3)) * x
    ay_grav = -(G * M_BH / (4.0 * r_mag**3)) * y

    df_coeff = A / (v_mag**3 + B)
    ax_df = -df_coeff * vx
    ay_df = -df_coeff * vy

    return [
        vx,
        vy,
        ax_grav + ax_df,
        ay_grav + ay_df
    ]


def adaptive_smbh_integrate_ivp(A, B, tol=1e-5):

    r = np.array([1.0, 0.0])
    v_circ = np.sqrt(G * M_BH / (4.0 * np.linalg.norm(r)))
    v = np.array([0.0, 0.8 * v_circ])

    initial_state = np.concatenate((r, v))

    def collision_event(t, state, A, B):
        return np.hypot(state[0], state[1]) - r_stop

    collision_event.terminal = True
    collision_event.direction = -1

    solution = solve_ivp(
        smbh_derivs_ivp,
        (0.0, 10000.0),
        initial_state,
        args=(A, B),
        method='DOP853',
        rtol=tol,
        atol=tol * 1e-2,
        events=collision_event,
        max_step=0.05
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    x_hist = solution.y[0]
    y_hist = solution.y[1]

    r_hist = np.hypot(x_hist, y_hist)

    return solution.t, r_hist, x_hist, y_hist

start = time.time()
t_arr, r_arr, x_arr, y_arr = adaptive_smbh_integrate_ivp(1.0, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))


a_vals = np.linspace(0.5, 10.0, 3)
b_vals = np.linspace(0.5, 10.0, 3)

ba_ratios = []
t_inspiral_myr = []

start = time.time()

for A_val in a_vals:

    for B_val in b_vals:

        t_h, r_h, _, _ = adaptive_smbh_integrate_ivp(
            A_val,
            B_val
        )

        ba_ratios.append(B_val / A_val)

        t_inspiral_myr.append(
            t_h[-1] * TIME_UNIT_MYR
        )

plt.figure(figsize=(7, 5))

plt.scatter(
    ba_ratios,
    t_inspiral_myr,
    color='#1f77b4',
    alpha=0.7
)

plt.xlabel('$B/A$')
plt.ylabel('Time')
plt.tight_layout()
plt.show()

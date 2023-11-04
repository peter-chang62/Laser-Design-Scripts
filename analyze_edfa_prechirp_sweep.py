import numpy as np
import matplotlib.pyplot as plt
import clipboard
import pandas as pd
from scipy.constants import c
import pynlo

f_r = 200e6
n = 256
v_min = c / 1750e-9
v_max = c / 1400e-9
v0 = c / 1560e-9
e_p = 1e-3 / f_r

t_fwhm = 2e-12
min_time_window = 20e-12
pulse = pynlo.light.Pulse.Sech(
    n,
    v_min,
    v_max,
    v0,
    e_p,
    t_fwhm,
    min_time_window,
    alias=2,
)
dv_dl = pulse.v_grid**2 / c  # J / Hz -> J / m

length_pm1550 = np.arange(0, 5.25, 0.25)

path = "sim_output/11-03-2023_1.5Pfwd_1.5Pbck_pre-chirp_sweep/"
# path = "sim_output/11-03-2023_0.75Pfwd_1.5Pbck_pre-chirp_sweep/"
P_V = np.zeros((length_pm1550.size, pulse.n), dtype=float)
P_T = np.zeros((length_pm1550.size, pulse.n), dtype=float)
E_P = np.zeros(length_pm1550.size, dtype=float)
V_W = np.zeros(length_pm1550.size, dtype=float)
T_W = np.zeros(length_pm1550.size, dtype=float)
for n, i in enumerate(length_pm1550):
    a_v = np.load(path + f"2.0_normal_edf_{i}_pm1550.npy")
    pulse.a_v[:] = a_v
    P_V[n] = pulse.p_v
    P_T[n] = pulse.p_t
    E_P[n] = pulse.e_p
    twidth = pulse.t_width(200)
    vwidth = pulse.v_width(200)
    V_W[n] = vwidth.eqv
    T_W[n] = twidth.eqv

P_WL = P_V * dv_dl

# %% -------------- plotting --------------------------------------------------
fig, ax = plt.subplots(1, 2)
(idx_wl,) = np.logical_and(
    1500 * 1e-9 < pulse.wl_grid, pulse.wl_grid < 1675 * 1e-9
).nonzero()
(idx_t,) = np.logical_and(-5 * 1e-12 < pulse.t_grid, pulse.t_grid < 5 * 1e-12).nonzero()

ax[0].pcolormesh(
    pulse.wl_grid[idx_wl] * 1e9, length_pm1550, P_WL[:, idx_wl], cmap="CMRmap_r_t"
)
ax[0].set_xlabel("wavelength (nm)")
ax[0].set_ylabel("length of pm1550 pre-chirp")
ax[1].pcolormesh(
    pulse.t_grid[idx_t] * 1e12, length_pm1550, P_T[:, idx_t], cmap="CMRmap_r_t"
)
ax[1].set_xlabel("time (ps)")
ax[1].set_ylabel("length of pm1550 pre-chirp")
fig.tight_layout()

fig, ax = plt.subplots(1, 1)
(l1,) = ax.plot(length_pm1550, c / v0**2 * V_W * 1e9, "o", label="frequency width")
ax2 = ax.twinx()
(l2,) = ax2.plot(length_pm1550, T_W * 1e12, "o", color="C1", label="temporal width")
lns = [l1, l2]
labels = [i.get_label() for i in lns]
ax.legend(lns, labels, loc="best")
ax.set_ylabel("frequency width (nm)")
ax2.set_ylabel("temporal width (ps)")
ax.set_xlabel("length of pm1550 pre-chirp (m)")
fig.tight_layout()

fig, ax = plt.subplots(1, 2, figsize=np.array([6.4, 8.03]))
(idx_wl,) = np.logical_and(
    1500 * 1e-9 < pulse.wl_grid, pulse.wl_grid < 1675 * 1e-9
).nonzero()
(idx_t,) = np.logical_and(-5 * 1e-12 < pulse.t_grid, pulse.t_grid < 5 * 1e-12).nonzero()
[
    ax[0].plot(pulse.wl_grid[idx_wl] * 1e9, i[idx_wl] / i.max() + n, "C0", linewidth=2)
    for n, i in enumerate(P_WL)
]
[
    ax[1].plot(pulse.t_grid[idx_t] * 1e12, i[idx_t] / i.max() + n, "C0", linewidth=2)
    for n, i in enumerate(P_T)
]
ax[0].set_xlabel("wavelength (nm)")
ax[1].set_xlabel("time (ps)")
ax[0].get_yaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
fig.tight_layout()

loss_ins = 10 ** (-0.7 / 10)
loss_spl = 10 ** (-0.7 / 10)
fig, ax = plt.subplots(1, 1)
ax.plot(length_pm1550, E_P * f_r * loss_spl * loss_ins, "o")
ax.set_xlabel("length of pm1550 pre-chirp (m)")
ax.set_ylabel("output power (W)")
fig.tight_layout()

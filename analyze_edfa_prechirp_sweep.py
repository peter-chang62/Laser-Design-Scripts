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

length_pm1550 = np.arange(0, 3.25, 0.25)

# path = "sim_output/11-03-2023_1.5Pfwd_1.5Pbck_pre-chirp_sweep/"
path = "sim_output/11-03-2023_0.75Pfwd_1.5Pbck_pre-chirp_sweep/"
P_V = np.zeros((13, pulse.n), dtype=float)
P_T = np.zeros((13, pulse.n), dtype=float)
E_P = np.zeros(13, dtype=float)
for n, i in enumerate(length_pm1550):
    a_v = np.load(path + f"2.0_normal_edf_{i}_pm1550.npy")
    pulse.a_v[:] = a_v
    P_V[n] = pulse.p_v
    P_T[n] = pulse.p_t
    E_P[n] = pulse.e_p

P_WL = P_V * dv_dl

# %% -------------- plotting --------------------------------------------------
fig, ax = plt.subplots(1, 2)
ax[0].pcolormesh(pulse.wl_grid * 1e9, length_pm1550, P_WL, cmap="CMRmap_r_t")
ax[0].set_xlabel("wavelength (nm)")
ax[0].set_ylabel("length of pm1550 pre-chirp")
ax[1].pcolormesh(pulse.t_grid * 1e12, length_pm1550, P_T, cmap="CMRmap_r_t")
ax[1].set_xlim(-2.5, 2.5)
ax[1].set_xlabel("wavelength (nm)")
ax[1].set_ylabel("length of pm1550 pre-chirp")
fig.tight_layout()

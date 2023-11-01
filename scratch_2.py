# %% ----- imports
from scipy.constants import c
import pandas as pd
import clipboard
from re_nlse_joint_5level_wASE import EDF
import pynlo
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import edfa_wASE as edfa
import time

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

# %% -------------- load absorption coefficients from NLight ------------------
sigma = pd.read_excel("NLight_provided/Erbium Cross Section - nlight_pump+signal.xlsx")
sigma = sigma.to_numpy()[1:].astype(float)[:, [0, 2, 3]]
a = sigma[:, :2]
e = sigma[:, [0, 2]]

spl_sigma_a = InterpolatedUnivariateSpline(
    c / a[:, 0][::-1], a[:, 1][::-1], ext="zeros"
)

spl_sigma_e = InterpolatedUnivariateSpline(
    c / e[:, 0][::-1], e[:, 1][::-1], ext="zeros"
)

# %% -------------- load dispersion coefficients ------------------------------
frame = pd.read_excel(
    "NLight_provided/nLIGHT Er80-4_125-HD-PM simulated fiber dispersion.xlsx"
)
# frame = pd.read_excel(
#     "NLight_provided/nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx"
# )
gvd = frame.to_numpy()[:, :2][1:].astype(float)

wl = gvd[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = 2 * np.pi * c / 1560e-9
polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
polyfit = polyfit[::-1]  # lowest order first

# %% ------------- pulse ------------------------------------------------------
loss_ins = 10 ** (-0.7 / 10)
loss_spl = 10 ** (-0.2 / 10)
f_r = 1e9
e_p = 25e-3 / f_r * loss_ins * loss_spl

n = 256
v_min = c / 1700e-9
v_max = c / 1400e-9
v0 = c / 1550e-9
t_fwhm = 250e-15
min_time_window = 10e-12
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
# %% ------------ active fiber ------------------------------------------------
tau = 9 * ms
r_eff = 3.06 * um / 2
a_eff = np.pi * r_eff**2
n_ion = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)

sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p = spl_sigma_a(c / 980e-9)

edf = EDF(
    f_r=f_r,
    overlap_p=1.0,
    overlap_s=1.0,
    n_ion=n_ion,
    a_eff=a_eff,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e,
)
edf.set_beta_from_beta_n(v0, polyfit)  # only gdd
gamma_edf = 0
edf.gamma = gamma_edf / (W * km)

# %% ----------- edfa ---------------------------------------------------------
model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
    p_fwd=pulse,
    p_bck=None,
    edf=edf,
    length=2,
    Pp_fwd=0 * loss_ins * loss_spl,
    Pp_bck=2 * loss_ins * loss_spl,
    n_records=100,
)

# model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
#     p_fwd=pulse,
#     p_bck=None,
#     edf=edf,
#     length=2,
#     Pp_fwd=2 * loss_ins * loss_spl,
#     Pp_bck=0 * loss_ins * loss_spl,
#     n_records=100,
# )

# %% ----- plot results
sim = sim_fwd
sol_Pp = sim.Pp
sol_Ps = np.sum(sim.p_v * pulse.dv * f_r, axis=1)
z = sim.z
n1 = sim.n1_n
n2 = sim.n2_n
n3 = sim.n3_n
n4 = sim.n4_n
n5 = sim.n5_n

fig = plt.figure(
    num="5-level rate equation for 250 fs pulse", figsize=np.array([11.16, 5.21])
)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(z, sol_Pp, label="pump", linewidth=2)
ax1.plot(z, sol_Ps * loss_ins * loss_spl, label="signal", linewidth=2)
ax1_2 = ax1.twinx()
ax1_2.plot(z, np.sum(sim_fwd.P_v_ase, axis=1) * loss_ins * loss_spl * 1e3, "C2--", label="ASE")
ax1_2.set_ylabel("ASE power (mW)")
ax1_2.legend(loc="lower right")
ax1.grid()
ax1.legend(loc="upper left")
ax1.set_xlabel("position (m)")
ax1.set_ylabel("power (W)")

ax2.plot(z, n1, label="n1", linewidth=2)
ax2.plot(z, n2, label="n2", linewidth=2)
ax2.plot(z, n3, label="n3", linewidth=2)
ax2.plot(z, n4, label="n4", linewidth=2)
ax2.plot(z, n5, label="n5", linewidth=2)
ax2.grid()
ax2.legend(loc="best")
ax2.set_xlabel("position (m)")
ax2.set_ylabel("population inversion")

fig.tight_layout()

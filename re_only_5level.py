"""
this scratch file is to re-create just the rate equations (not PyNLO) but this
time including pump excited state absorption (ESA).
"""

# %% ----- imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import pynlo
import clipboard
import pandas as pd
from scipy.constants import c
from scipy.integrate import odeint
from five_level_ss_eqns import (
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
    dPp_dz,
    dPs_dz,
    n1_func,
    n2_func,
    n3_func,
    n4_func,
    n5_func,
)

# selectively turn off effects:
# eps_s = 0  # signal esa
# eps_p = 0  # pump esa
# xi_p = 0  # pump stimulated emission

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


# %% ------------- pulse ------------------------------------------------------
f_r = 1e9
e_p = 1e-3 / f_r

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


# %% -------------------------------------------------------------------------
sigma_p = spl_sigma_a(c / 980e-9)
sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)

n_ion = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)  # dB/m absorption at 1530 nm
r_eff = 3.06 * um / 2
a_eff = np.pi * r_eff**2
nu_p = c / 980e-9


def func(
    X,
    z,
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
    output="deriv",
):
    P_p = X[0]
    P_s = X[1:]
    args = [
        n_ion,
        a_eff,
        1,
        1,
        nu_p,
        P_p,
        pulse.v_grid,
        P_s,
        sigma_p,
        sigma_a,
        sigma_e,
        eps_p,
        xi_p,
        eps_s,
        tau_21,
        tau_32,
        tau_43,
        tau_54,
    ]

    dPpdz = dPp_dz(*args)

    dPsdz = dPs_dz(*args)

    if output == "deriv":
        return np.hstack((dPpdz, dPsdz))
    elif output == "n":
        return (
            n1_func(*args),
            n2_func(*args),
            n3_func(*args),
            n4_func(*args),
            n5_func(*args),
        )


# %% -------------------------------------------------------------------------
Pp_0 = 2
Pv_0 = pulse.p_v.copy() * pulse.dv * f_r
length = 4

X_0 = np.hstack([Pp_0, Pv_0])
z = np.linspace(0, length, 1000)
args = (
    eps_p,
    xi_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
)
sol = odeint(func, X_0, z, args=args)

sol_Pp = sol[:, 0]
sol_Pv = sol[:, 1:]
sol_Ps = np.sum(sol_Pv, axis=1)

n1 = np.zeros(z.size)
n2 = np.zeros(z.size)
n3 = np.zeros(z.size)
n4 = np.zeros(z.size)
n5 = np.zeros(z.size)
for n, (pp, pv) in enumerate(zip(sol_Pp, sol_Pv)):
    inversion = func(np.hstack((pp, pv)), None, *args, output="n")
    n1[n] = inversion[0]
    n2[n] = inversion[1]
    n3[n] = inversion[2]
    n4[n] = inversion[3]
    n5[n] = inversion[4]

# %% ----------------------------- plot results! ------------------------------
fig = plt.figure(
    num="5-level rate equation for 250 fs pulse", figsize=np.array([11.16, 5.21])
)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
(line_11,) = ax1.plot(z, sol_Pp, label="pump")
(line_12,) = ax1.plot(z, sol_Ps, label="signal")
ax1.grid()
ax1.legend(loc="best")
ax1.set_xlabel("position (m)")
ax1.set_ylabel("power (W)")

(line_21,) = ax2.plot(z, n1 / n_ion, label="n1")
(line_22,) = ax2.plot(z, n2 / n_ion, label="n2")
(line_23,) = ax2.plot(z, n3 / n_ion, label="n3")
(line_24,) = ax2.plot(z, n4 / n_ion, label="n4")
(line_25,) = ax2.plot(z, n5 / n_ion, label="n5")
ax2.grid()
ax2.legend(loc="best")
ax2.set_xlabel("position (m)")
ax2.set_ylabel("population inversion")

fig.tight_layout()

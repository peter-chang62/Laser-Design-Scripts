"""
NO DISPERSION, NO NONLINEARITY, JUST THE RATE EQUATION CALCULATED OVER
BROADBAND!
"""

# %% ----- imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.constants import c, h
import pynlo
from scipy.integrate import odeint
import clipboard
import pandas as pd


ps = 1e-12
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0


def n2_over_n(
    overlap_p,
    overlap_s,
    a_eff,
    nu_p,
    Pp,
    nu_v,
    P_v,
    sigma_p,
    sigma_a,
    sigma_e,
    tau,
):
    # terms generally go as overlap * sigma * power / (h * nu * A)
    pump_term = overlap_p * sigma_p * Pp / (h * nu_p * a_eff)
    signal_num = overlap_s * sigma_a * P_v / (h * nu_v * a_eff)
    signal_denom = overlap_s * (sigma_a + sigma_e) * P_v / (h * nu_v * a_eff)

    num = signal_num + pump_term
    denom = signal_denom + pump_term + 1 / tau
    return num / denom


def _n2_over_n(
    overlap_p,
    overlap_s,
    a_eff,
    nu_p,
    Pp,
    sigma_p,
    sum_a,
    sum_e,
    tau,
):
    # terms generally go as overlap * sigma * power / (h * nu * A)
    pump_term = overlap_p * sigma_p * Pp / (h * nu_p * a_eff)
    # sum_a = sigma_a * P_v / (h * nu_v * a_eff)
    # sum_e = sigma_e * P_v / h * nu_v * a_eff
    signal_num = overlap_s * sum_a
    signal_denom = overlap_s * (sum_a + sum_e)

    num = signal_num + pump_term
    denom = signal_denom + pump_term + 1 / tau
    return num / denom


def dpdz(n2_n, n, overlap, sigma_a, sigma_e, p):
    n2 = n2_n * n
    n1 = n - n2

    # terms should go as overlap * sigma * n * P
    emission = overlap * sigma_e * n2 * p
    absorption = overlap * sigma_a * n1 * p
    return emission - absorption


def gain(n2_n, n, overlap, sigma_a, sigma_e):
    n2 = n2_n * n
    n1 = n - n2

    # terms should go as overlap * sigma * n * P
    emission = overlap * sigma_e * n2
    absorption = overlap * sigma_a * n1
    return emission - absorption


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
sigma_p = spl_sigma_a(c / 980e-9)

# %% ------------- pulse ------------------------------------------------------
n = 256
v_min = c / 1700e-9
v_max = c / 1400e-9
v0 = c / 1550e-9
e_p = 5e-3 / 100e6
t_fwhm = 300e-15
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

# %% ------------- edfa parameters --------------------------------------------
# doping and fiber info from ansys
tau = 10e-3
r_eff = 3.06e-6 / 2
a_eff = np.pi * r_eff**2
n_ion = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)

# other parameters that I input / assume
overlap_p = 1
overlap_s = 1
nu_p = c / 980e-9

sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)


# %% ------------- edfa ------------------------------------------------------
def calc_joint_broadband(X, z, f_r):
    Pp = X[0]
    p_v = X[1:]

    sum_a = p_v * f_r * sigma_a / (h * pulse.v_grid * a_eff)
    sum_e = p_v * f_r * sigma_e / (h * pulse.v_grid * a_eff)
    sum_a = np.sum(sum_a * pulse.dv)
    sum_e = np.sum(sum_e * pulse.dv)

    n2_n = _n2_over_n(
        overlap_p,
        overlap_s,
        a_eff,
        nu_p,
        Pp,
        sigma_p,
        sum_a,
        sum_e,
        tau,
    )

    dPp_dz = dpdz(n2_n, n_ion, overlap_p, sigma_p, 0, Pp)
    dp_v_dz = dpdz(n2_n, n_ion, overlap_s, sigma_a, sigma_e, p_v)

    return np.hstack([dPp_dz, dp_v_dz])


def calc_joint_CW(X, z, nu_s):
    Pp, P_v = X
    n2_n = n2_over_n(
        overlap_p,
        overlap_s,
        a_eff,
        nu_p,
        Pp,
        nu_s,
        P_v,
        sigma_p,
        spl_sigma_a(nu_s),
        spl_sigma_e(nu_s),
        tau,
    )
    dPp_dz = dpdz(n2_n, n_ion, overlap_p, sigma_p, 0, Pp)
    dP_v_dz = dpdz(n2_n, n_ion, overlap_s, spl_sigma_a(nu_s), spl_sigma_e(nu_s), P_v)
    return dPp_dz, dP_v_dz


# %% ------------------ broadband calculation ---------------------------------
Pp_0 = 1
length = 2.0
f_r = 100e6
X_0 = np.hstack([Pp_0, pulse.p_v.copy()])
z = np.linspace(0, length, 1000)
sol = odeint(calc_joint_broadband, X_0, z, args=(f_r,))

Pp = sol[:, 0]
p_v = sol[:, 1:]
e_p_z = np.sum(p_v * pulse.dv, axis=1)
p_v = np.where(p_v < 0, 0, p_v)
a_v_out = p_v[-1] ** 0.5
p_out = pulse.copy()
p_out.a_v[:] = a_v_out

sum_a = p_v * f_r * sigma_a / (h * pulse.v_grid * a_eff)
sum_e = p_v * f_r * sigma_e / (h * pulse.v_grid * a_eff)
sum_a = np.sum(sum_a * pulse.dv, axis=1)
sum_e = np.sum(sum_e * pulse.dv, axis=1)
n2_n = _n2_over_n(
    overlap_p,
    overlap_s,
    a_eff,
    nu_p,
    Pp,
    sigma_p,
    sum_a,
    sum_e,
    tau,
)

fig, ax = plt.subplots(1, 1)
ax.plot(z, e_p_z * 100e6, label="signal", linewidth=2)
ax.plot(z, Pp, label="pump", linewidth=2)
ax.grid()
ax.legend(loc="best")
ax_2 = ax.twinx()
ax_2.plot(z, n2_n, color="C2", linestyle="--", linewidth=2)
ax.set_ylabel("power (W)")
ax.set_xlabel("position (m)")
ax.set_title("broadband amplification from RE")
fig.tight_layout()

# %% ------------------ CW calculation ---------------------------------
P_v_0 = pulse.e_p * 100e6
X_0 = np.array([Pp_0, P_v_0])
sol_CW = odeint(calc_joint_CW, X_0, z, args=(c / 1550e-9,))

Pp_CW = sol_CW[:, 0]
P_v_CW = sol_CW[:, 1]
n2_n_CW = n2_over_n(
    overlap_p,
    overlap_s,
    a_eff,
    nu_p,
    Pp_CW,
    c / 1550e-9,
    P_v_CW,
    sigma_p,
    spl_sigma_a(c / 1550e-9),
    spl_sigma_e(c / 1550e-9),
    tau,
)

fig, ax = plt.subplots(1, 1)
ax.plot(z, P_v_CW, label="signal", linewidth=2)
ax.plot(z, Pp_CW, label="pump", linewidth=2)
ax.legend(loc="best")
ax2 = ax.twinx()
ax2.plot(z, n2_n_CW, color="C2", linestyle="--", linewidth=2)
ax.set_ylabel("power (W)")
ax.set_xlabel("position (m)")
ax.set_title("CW amplification @ 1550 nm")
ax.grid()
fig.tight_layout()

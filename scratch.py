"""
diagnose by solving the CW case! should have tried this earlier!

I caught the error! compared to the correct iteration, you were doing something
in between in all the previous cases, where you updated some variables but not
others. That leads to undefined behavior!

It's clever, you can use method of iterations! It's a lot slower than including
it in the RK4, but it prevents you from having to do any additional work...

In addition, Lindberg's case means no iteration is needed for co-propagating
pump, but for multidirectional beams, he also needs to iterate. You just need
to change the conditions on your iteration to move on!
"""

# %% ----- imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import clipboard
from scipy.constants import c, h
import pandas as pd
import pynlo
from scipy.integrate import RK45, odeint
from tqdm import tqdm

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


# %% -------------- load absorption coefficients ------------------------------
sigma = np.genfromtxt("Ansys/er_cross_section_fig_6_1.txt")
a = sigma[3:][:, :2]
e = sigma[3:][:, [0, 2]]

sigma_p = sigma[0, 1]

spl_sigma_a = InterpolatedUnivariateSpline(
    c / a[:, 0][::-1], a[:, 1][::-1], ext="zeros"
)
spl_sigma_e = InterpolatedUnivariateSpline(
    c / e[:, 0][::-1], e[:, 1][::-1], ext="zeros"
)

# %% ------------- edfa parameters --------------------------------------------
# doping and fiber info from ansys
tau = 10e-3
r_eff = 3.06e-6 / 2
a_eff = np.pi * r_eff**2
n_ion = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)  # dB/m absorption at 1530 nm

# other parameters that I input / assume
overlap_p = 1.0
overlap_s = 1.0


def calc_joint(Pp_0, Ps_0, nu_s, length):
    nu_p = c / 980e-9
    sigma_a = spl_sigma_a(nu_s)
    sigma_e = spl_sigma_e(nu_s)

    # %% ------------- joint solver -------------------------------------------
    def func(X, z):
        Pp, Ps = X
        n2_n = n2_over_n(
            overlap_p,
            overlap_s,
            a_eff,
            nu_p,
            Pp,
            nu_s,
            Ps,
            sigma_p,
            sigma_a,
            sigma_e,
            tau,
        )
        dPp_dz = dpdz(n2_n, n_ion, overlap_p, sigma_p, 0, Pp)
        dPs_dz = dpdz(n2_n, n_ion, overlap_s, sigma_a, sigma_e, Ps)
        return dPp_dz, dPs_dz

    z = np.linspace(0, length, 1000)
    sol = odeint(func, np.array([Pp_0, Ps_0]), z)

    return sol


def calc_iter(Pp_0, Ps_0, nu_s, length, thresh=1e-3):
    nu_p = c / 980e-9
    sigma_a = spl_sigma_a(nu_s)
    sigma_e = spl_sigma_e(nu_s)

    z = np.linspace(0, length, 1000)

    done = False
    old = None
    spl_Ps = lambda z: 0
    spl_Pp = lambda z: 0
    while not done:
        # when solving for pump, dpdz is calculated using the previous signal
        # power
        def func(Pp, z):
            n2_n = n2_over_n(
                overlap_p,
                overlap_s,
                a_eff,
                nu_p,
                Pp,
                nu_s,
                spl_Ps(z),
                sigma_p,
                sigma_a,
                sigma_e,
                tau,
            )
            return dpdz(n2_n, n_ion, overlap_p, sigma_p, 0, Pp)

        sol_Pp = odeint(func, np.array([Pp_0]), z)
        spl_Pp = InterpolatedUnivariateSpline(z, sol_Pp)

        # when solving for signal, dpdz is calculates using the previous pump
        def func(Ps, z):
            n2_n = n2_over_n(
                overlap_p,
                overlap_s,
                a_eff,
                nu_p,
                spl_Pp(z),
                nu_s,
                Ps,
                sigma_p,
                sigma_a,
                sigma_e,
                tau,
            )
            return dpdz(n2_n, n_ion, overlap_s, sigma_a, sigma_e, Ps)

        sol_Ps = odeint(func, np.array([Ps_0]), z)
        spl_Ps = InterpolatedUnivariateSpline(z, sol_Ps)

        if old is None:
            old = sol_Ps[-1]
        else:
            new = sol_Ps[-1]
            error = abs(new - old) / old
            old = new
            if error < thresh:
                done = True
            print(error, new)
    return sol_Pp, sol_Ps


Pp_0 = 1
Ps_0 = 5e-3
length = 4
sol1 = calc_joint(Pp_0, Ps_0, c / 1550e-9, length)
sol2 = calc_iter(Pp_0, Ps_0, c / 1550e-9, length, thresh=1e-3)

plt.figure()
z = np.linspace(0, length, 1000)
plt.plot(z, sol1[:, 0])
plt.plot(z, sol1[:, 1])
plt.plot(z, sol2[0])
plt.plot(z, sol2[1])

Pp = sol1[:, 0]
Ps = sol1[:, 1]
sigma_a = spl_sigma_a(c / 1550e-9)
sigma_e = spl_sigma_e(c / 1550e-9)
n2_n = n2_over_n(
    overlap_p,
    overlap_s,
    a_eff,
    c / 980e-9,
    Pp,
    c / 1550e-9,
    Ps,
    sigma_p,
    sigma_a,
    sigma_e,
    tau,
)

plt.gca().twinx().plot(z, n2_n, color="C4", linestyle="--")
# fig, ax = plt.subplots(1, 1)
# ax.plot(z, Pp)
# ax.plot(z, Ps)
# ax.grid()
# ax2 = ax.twinx()
# ax2.plot(z, n2_n, color="C2", linestyle="--")

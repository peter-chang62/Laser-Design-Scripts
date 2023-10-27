from scipy.constants import c
import pandas as pd
import clipboard
from re_nlse_joint_5level import Mode, EDF, NLSE
from five_level_ss_eqns import (
    xi_p,
    eps_p,
    eps_s,
    tau_21,
    tau_32,
    tau_43,
    tau_54,
)
import pynlo
import numpy as np
import collections
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from pynlo.utility.misc import SettableArrayProperty
from scipy.integrate import odeint

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

output = collections.namedtuple("output", ["model", "sim"])


def dPp_dz(
    n1,
    n3,
    P_p,
    sigma_p,
    sigma_a,
    overlap_p,
    eps_p,
    xi_p,
):
    return (
        (-sigma_p * n1 + sigma_p * xi_p * n3 - sigma_p * eps_p * n3) * overlap_p * P_p
    )


def amplify(fiber, pulse, length, Pp_fwd, Pp_bck, p_v_prev=None, Pp_prev=None):
    fiber: EDF

    model, dz = fiber.generate_model(
        pulse,
        t_shock="auto",
        raman_on=True,
        Pp_fwd=Pp_fwd,
        Pp_bck=Pp_bck,
        p_v_prev=p_v_prev,
        Pp_prev=Pp_prev,
    )

    sim = model.simulate(length, dz=dz, n_records=100)

    return output(model=model, sim=sim)


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
f_r = 200e6
e_p = 1e-3 / f_r

n = 256
v_min = c / 1700e-9
v_max = c / 1400e-9
v0 = c / 1550e-9
t_fwhm = 250e-15
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
gamma_edf = 6.5
edf.gamma = gamma_edf / (W * km)

# %% ----- solve amplifier using shooting method ------------------------------
length = 1.5
Pp_0 = 1.0
# to start off: forward propagation with all backward components set to 0
model, sim = amplify(
    edf,
    pulse,
    length,
    Pp_fwd=0,
    Pp_bck=0,
)

# ----- iteration the general rule is: each propagation solves for everything
#       in that direction, and does not touch anything from the solution of
#       the previous direction
done = False
loop_count = 0
tol = 1e-3
while not done:
    # solve the backward propagation based on the the previous forward
    # propagation
    n1 = InterpolatedUnivariateSpline(
        sim.z, sim.n1_n[::-1] * model.mode.n_ion, ext="zeros"
    )
    n3 = InterpolatedUnivariateSpline(
        sim.z, sim.n3_n[::-1] * model.mode.n_ion, ext="zeros"
    )

    def func(P_p, z):
        return dPp_dz(
            n1(z),
            n3(z),
            P_p,
            sigma_p,
            sigma_a,
            overlap_p=1,
            eps_p=eps_p,
            xi_p=xi_p,
        )

    sol_Pp_bck = odeint(func, np.array([Pp_0]), sim.z)
    Pp_bck = InterpolatedUnivariateSpline(sim.z, sol_Pp_bck[::-1])

    # solve the forward propagation based on the previous backward propagation
    model, sim = amplify(
        edf,
        pulse,
        length,
        Pp_fwd=0,
        Pp_bck=0,
        Pp_prev=Pp_bck,
    )

    # book keeping
    if loop_count == 0:
        e_p_old = sim.pulse_out.e_p
        Pp_old = sol_Pp_bck[-1]
        loop_count += 1
        continue

    e_p_new = sim.pulse_out.e_p
    Pp_new = sol_Pp_bck[-1]

    error_e_p = abs(e_p_new - e_p_old) / e_p_new
    error_Pp = abs(Pp_new - Pp_old) / Pp_new
    if error_e_p < tol and error_Pp < tol:
        done = True

    e_p_old = e_p_new
    Pp_old = Pp_new

    print(loop_count, error_e_p, error_Pp)
    loop_count += 1

# ------ look at results, IT MATCHES!!
num = "bck"
fig, ax = plt.subplots(1, 1)
ax.plot(sim.z, np.sum(sim.p_v * pulse.dv, axis=1) * 100e6, label="signal", linewidth=2)
ax.plot(sim.z, sim.Pp, label="pump", linewidth=2)
ax.set_ylabel("power (W)")
ax.set_xlabel("position (m)")
ax.set_title("broadband amplification with PyNLO")
ax.grid()
fig.tight_layout()

sim.plot("wvl", num=num)

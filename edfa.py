from scipy.constants import c
import pandas as pd
import clipboard
from re_nlse_joint_5level import EDF
import pynlo
import numpy as np
import collections
from scipy.interpolate import make_interp_spline, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

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


def propagate_amp(length, edf, pulse, Pp, p_v_prev=None, Pp_prev=None, n_records=None):
    edf: EDF

    model, dz = edf.generate_model(
        pulse,
        t_shock="auto",
        raman_on=True,
        Pp_fwd=Pp,
        Pp_bck=0,
        p_v_prev=p_v_prev,
        Pp_prev=Pp_prev,
    )
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


def amplify(length, edf, p_fwd, p_bck=None, Pp_fwd=0.0, Pp_bck=0.0, n_records=100):
    if p_bck is None:
        if Pp_bck == 0:  # forward seeding + forward pumping
            return propagate_amp(length, edf, p_fwd, Pp_fwd, n_records=n_records)

    print("setting n_records to 100")
    n_records = 100

    # forward propagation with no backward info
    model_fwd, sim_fwd = propagate_amp(
        length, edf, p_fwd, Pp_fwd, p_v_prev=None, Pp_prev=None, n_records=n_records
    )

    done = False
    loop_count = 0
    tol = 1e-3
    while not done:
        # solve the backward propagation using forward info
        p_v = make_interp_spline(sim_fwd.z, sim_fwd.p_v[::-1])
        Pp = InterpolatedUnivariateSpline(sim_fwd.z, sim_fwd.Pp[::-1])
        model_bck, sim_bck = propagate_amp(
            length, edf, p_bck, Pp_bck, p_v_prev=p_v, Pp_prev=Pp, n_records=n_records
        )

        # solve the forward propagation using backward info
        p_v = make_interp_spline(sim_bck.z, sim_bck.p_v[::-1])
        Pp = InterpolatedUnivariateSpline(sim_bck.z, sim_bck.Pp[::-1])
        model_fwd, sim_fwd = propagate_amp(
            length, edf, p_fwd, Pp_fwd, p_v_prev=p_v, Pp_prev=Pp, n_records=n_records
        )

        # book keeping
        if loop_count == 0:
            e_p_fwd_old = sim_fwd.pulse_out.e_p
            e_p_bck_old = sim_bck.pulse_out.e_p
            Pp_old = sim_fwd.Pp[-1]

            loop_count += 1
            continue

        e_p_fwd_new = sim_fwd.pulse_out.e_p
        e_p_bck_new = sim_bck.pulse_out.e_p
        Pp_new = sim_fwd.Pp[-1]

        error_e_p_fwd = abs(e_p_fwd_new - e_p_fwd_old) / e_p_fwd_new
        error_e_p_bck = abs(e_p_bck_new - e_p_bck_old) / e_p_bck_new
        error_Pp = abs(Pp_new - Pp_old) / Pp_new

        if np.all(np.array([error_e_p_fwd, error_e_p_bck, error_Pp]) < tol):
            done = True

        e_p_fwd_old = e_p_fwd_new
        e_p_bck_old = e_p_bck_new
        Pp_old = Pp_new

        print(loop_count, error_e_p_fwd, error_e_p_bck, error_Pp)
        loop_count += 1

    return output(model=model_fwd, sim=sim_fwd), output(model=model_bck, sim=sim_bck)


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

fwd, bck = amplify(1.5, edf, pulse, p_bck=pulse, Pp_fwd=1, Pp_bck=1, n_records=100)

# %% ----- imports
from scipy.constants import c
import pandas as pd
import clipboard
from re_nlse_joint_5level import EDF
import pynlo
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import edfa
import time
import collections

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

output = collections.namedtuple("output", ["model", "sim"])
# save_path = r"sim_output/11-03-2023_1.5Pfwd_1.5Pbck_pre-chirp_sweep/"
save_path = (
    r"sim_output/20231012-200MHz-beforepreamp-withsplitter/gamma_10/"
    + "11-03-2023_0.9mEDF_1.2Pfwd_1.2Pbck_pre-chirp_sweep/"
)


def propagate(fiber, pulse, length, n_records=None):
    """
    propagates a given pulse through fiber of given length

    Args:
        fiber (instance of SilicaFiber): Fiber
        pulse (instance of Pulse): Pulse
        length (float): fiber elngth

    Returns:
        output: model, sim
    """
    fiber: pynlo.materials.SilicaFiber
    model = fiber.generate_model(pulse)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records)
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
frame_n = pd.read_excel(
    "NLight_provided/nLIGHT Er80-4_125-HD-PM simulated fiber dispersion.xlsx"
)
frame_a = pd.read_excel(
    "NLight_provided/nLIGHT_Er80-8_125-PM_simulated_GVD_dispersion.xlsx"
)

gvd_n = frame_n.to_numpy()[:, :2][1:].astype(float)
wl = gvd_n[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = 2 * np.pi * c / 1560e-9
polyfit_n = np.polyfit(omega - omega0, gvd_n[:, 1], deg=3)
polyfit_n = polyfit_n[::-1]  # lowest order first

gvd_a = frame_a.to_numpy()[:, :2][1:].astype(float)
wl = gvd_a[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = 2 * np.pi * c / 1560e-9
polyfit_a = np.polyfit(omega - omega0, gvd_a[:, 1], deg=3)
polyfit_a = polyfit_a[::-1]  # lowest order first

gamma_n = 10 / (W * km)
gamma_a = 1.2 / (W * km)

# %% ------------- pulse ------------------------------------------------------
loss_ins = 10 ** (-0.7 / 10)
loss_spl = 10 ** (-0.7 / 10)

f_r = 200e6
n = 256
v_min = c / 1750e-9
v_max = c / 1400e-9
v0 = c / 1560e-9
e_p = 35e-3 / 2 / f_r

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

# a_v = np.load("sim_output/200MHz_6psnmkm_40cm_totaledf_400mW_pump.npy")
# v_grid = np.load("sim_output/v_grid.npy")
# phi_v = np.unwrap(np.angle(a_v))  # unwrap for fitting
# p_v = abs(a_v) ** 2
# pulse.import_p_v(v_grid, p_v, phi_v=phi_v)

spec = np.genfromtxt(
    "20231012-200MHz-beforepreamp-nosplitter.CSV", delimiter=",", skip_header=44
)
spec[:, 0] = c / (spec[:, 0] * 1e-9)
spec[:, 1] = 10 ** (spec[:, 1] / 10)
spec[::] = spec[::-1]
dl_dv = c / spec[:, 0] ** 2  # J / m -> J / Hz (could be off by an overall scale)
spec[:, 1] *= dl_dv
pulse.import_p_v(spec[:, 0], spec[:, 1], phi_v=None)

# %% ---------- optional passive fiber ----------------------------------------
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = 1.2 / (W * km)

for length_pm1550 in np.arange(0.0, 3.01, 0.01):
    # ignore numpy error if length = 0.0, it occurs when n_records is not None and
    # propagation length is 0, the output pulse is still correct
    model_pm1550, sim_pm1550 = propagate(pm1550, pulse, length_pm1550)
    pulse_pm1550 = sim_pm1550.pulse_out

    # %% ------------ active fiber ------------------------------------------------
    tau = 9 * ms
    r_eff_n = 3.06 * um / 2
    r_eff_a = 8.05 * um / 2
    a_eff_n = np.pi * r_eff_n**2
    a_eff_a = np.pi * r_eff_a**2
    n_ion_n = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)
    n_ion_a = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)

    sigma_a = spl_sigma_a(pulse.v_grid)
    sigma_e = spl_sigma_e(pulse.v_grid)
    sigma_p = spl_sigma_a(c / 980e-9)

    length = 0.9

    edf = EDF(
        f_r=f_r,
        overlap_p=1.0,
        overlap_s=1.0,
        n_ion=n_ion_n,
        a_eff=a_eff_n,
        sigma_p=sigma_p,
        sigma_a=sigma_a,
        sigma_e=sigma_e,
    )
    edf.set_beta_from_beta_n(v0, polyfit_n)
    beta_n = edf._beta(pulse.v_grid)
    edf.gamma = gamma_n

    # %% ----------- edfa ---------------------------------------------------------
    model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
        p_fwd=pulse_pm1550,
        p_bck=None,
        edf=edf,
        length=length,
        Pp_fwd=1.2 * loss_ins * loss_ins * loss_spl,
        Pp_bck=1.2 * loss_ins * loss_ins * loss_spl,
        n_records=100,
    )
    sim = sim_fwd

    # %% ------------ save results ------------------------------------------------
    np.save(
        save_path + f"{length}_normal_edf_{np.round(length_pm1550, 2)}_pm1550.npy", sim.pulse_out.a_v
    )

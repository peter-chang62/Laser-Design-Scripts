# %% ----- imports
from scipy.constants import c
import pandas as pd
import clipboard
from re_nlse_joint_5level_wsplice import EDF
import pynlo
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import edfa_wsplice as edfa
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
save_path = r"sim_output/11-03-2023_0.75Pfwd_1.5Pbck_pre-chirp_sweep/"


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
    model = fiber.generate_model(pulse, t_shock=None, raman_on=False)
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

gamma_n = 6.5 / (W * km)
gamma_a = 1.2 / (W * km)

# %% ------------- pulse ------------------------------------------------------
loss_ins = 10 ** (-0.7 / 10)
loss_spl = 10 ** (-0.2 / 10)

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

# a_v = np.load("sim_output/200MHz_6psnmkm_40cm_totaledf_400mW_pump.npy")
# v_grid = np.load("sim_output/v_grid.npy")
# phi_v = np.unwrap(np.angle(a_v))  # unwrap for fitting
# p_v = abs(a_v) ** 2
# pulse.import_p_v(v_grid, p_v, phi_v=phi_v)

spec = np.genfromtxt(
    "20231012-200MHz-beforepreamp-withsplitter.CSV", delimiter=",", skip_header=44
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

length_pm1550 = 3.0
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

length = 2.0
z_spl = 3

edf = EDF(
    f_r=f_r,
    overlap_p=1.0,
    overlap_s=1.0,
    n_ion_1=n_ion_n,
    n_ion_2=n_ion_a,
    z_spl=z_spl,
    loss_spl=10 ** (-0.7 / 10),
    a_eff_1=a_eff_n,
    a_eff_2=a_eff_a,
    gamma_1=gamma_n,
    gamma_2=gamma_a,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e,
)
edf.set_beta_from_beta_n(v0, polyfit_n)
beta_n = edf._beta(pulse.v_grid)
edf.set_beta_from_beta_n(v0, polyfit_a)
beta_a = edf.beta(pulse.v_grid)

# %% ----------- edfa ---------------------------------------------------------
model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
    p_fwd=pulse_pm1550,
    p_bck=None,
    beta_1=beta_n,
    beta_2=beta_a,
    edf=edf,
    length=length,
    Pp_fwd=0.75 * loss_ins * loss_ins,
    Pp_bck=1.5 * loss_ins * loss_ins,
    n_records=100,
)
sim = sim_fwd

# %% ----------- plot results -------------------------------------------------
# sol_Pp = sim.Pp
# sol_Ps = np.sum(sim.p_v * pulse.dv * f_r, axis=1)
# z = sim.z
# n1 = sim.n1_n
# n2 = sim.n2_n
# n3 = sim.n3_n
# n4 = sim.n4_n
# n5 = sim.n5_n

# fig = plt.figure(
#     num=f"power evolution for {length} normal edf and {length_pm1550} pm1550 pre-chirp",
#     figsize=np.array([11.16, 5.21]),
# )
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# ax1.plot(z, sol_Pp, label="pump", linewidth=2)
# ax1.plot(z, sol_Ps * loss_ins * loss_spl, label="signal", linewidth=2)
# ax1.grid()
# ax1.legend(loc="upper left")
# ax1.set_xlabel("position (m)")
# ax1.set_ylabel("power (W)")

# ax2.plot(z, n1, label="n1", linewidth=2)
# ax2.plot(z, n2, label="n2", linewidth=2)
# ax2.plot(z, n3, label="n3", linewidth=2)
# ax2.plot(z, n4, label="n4", linewidth=2)
# ax2.plot(z, n5, label="n5", linewidth=2)
# ax2.grid()
# ax2.legend(loc="best")
# ax2.set_xlabel("position (m)")
# ax2.set_ylabel("population inversion")
# fig.tight_layout()

# sim.plot(
#     "wvl",
#     num=f"spectral evolution for {length} normal edf and {length_pm1550} pm1550 pre-chirp",
# )

# fig, ax = plt.subplots(
#     1, 2, num=f"output for {length} normal edf and {length_pm1550} pm1550 pre-chirp"
# )
# p_wl = sim.p_v * dv_dl
# ax[0].plot(pulse.wl_grid * 1e9, p_wl[0] / p_wl[0].max())
# ax[0].plot(pulse.wl_grid * 1e9, p_wl[-1] / p_wl[-1].max())
# ax[1].plot(pulse.t_grid * 1e12, sim.p_t[0] / sim.p_t[0].max())
# ax[1].plot(pulse.t_grid * 1e12, sim.p_t[-1] / sim.p_t[-1].max())

# %% ------------ save results ------------------------------------------------
np.save(
    save_path + f"{length}_normal_edf_{length_pm1550}_pm1550.npy", sim.pulse_out.a_v
)

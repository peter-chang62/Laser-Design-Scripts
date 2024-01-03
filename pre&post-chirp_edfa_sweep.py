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
from tqdm import tqdm
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


def propagate(fiber, pulse, length, n_records=None, plot=None):
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
    sim = model.simulate(length, dz=dz, n_records=n_records, plot=plot)
    return output(model=model, sim=sim)


# %% -------------- save paths ------------------------------------------------
save_path = (
    r"sim_output/20231012-200MHz-beforepreamp-withsplitter/gamma_6.5/"
    # + "11-03-2023_1.5mEDF_1.2Pfwd_1.2Pbck_pre-chirp_sweep/"
    + "11-03-2023_1.5mEDF_1.2Pfwd_1.2Pbck_pre-chirp_sweep_mat_Pploss/"
)
# save_path = (
#     r"sim_output/Matt_100MHz_Menlo/gamma_6.5/1.5mEDF_2Wfwd_2W_bck_pre-chirp_sweep/"
# )

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
loss_spl = 10 ** (-0.7 / 10)
loss_mat = 10 ** (-1 / 10)

f_r = 200e6
n = 256
v_min = c / 1750e-9
v_max = c / 1400e-9
v0 = c / 1560e-9
e_p = 35e-3 / 2 / f_r * loss_ins * loss_spl  # mating sleeve mainly affects pump

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
    "Sichong/20231012-200MHz-beforepreamp-nosplitter.CSV", delimiter=",", skip_header=44
)
# spec = np.genfromtxt("Matt/bottomAmpInput.CSV", delimiter=",", skip_header=39)
spec[:, 0] = c / (spec[:, 0] * 1e-9)
spec[:, 1] = 10 ** (spec[:, 1] / 10)
spec[::] = spec[::-1]
dl_dv = c / spec[:, 0] ** 2  # J / m -> J / Hz (could be off by an overall scale)
spec[:, 1] *= dl_dv
pulse.import_p_v(spec[:, 0], spec[:, 1], phi_v=None)

# %% ---------- passive fiber -------------------------------------------------
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = gamma_a / (W * km)

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

length = 1.5

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

# %% ----- pre-chirp sweep ----------------------------------------------------
# for _, pre_chirp in enumerate(tqdm(np.arange(3.0, 4.01, 0.01))):
#     # ignore numpy error if length = 0.0, it occurs when n_records is not None and
#     # propagation length is 0, the output pulse is still correct
#     model_pm1550, sim_pm1550 = propagate(pm1550, pulse, pre_chirp)
#     pulse_pm1550 = sim_pm1550.pulse_out

#     # %% ----- edfa
#     model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
#         p_fwd=pulse_pm1550,
#         p_bck=None,
#         edf=edf,
#         length=length,
#         Pp_fwd=2 * loss_ins * loss_ins * loss_spl * loss_mat,
#         Pp_bck=2 * loss_ins * loss_ins * loss_spl * loss_mat,
#         n_records=100,
#     )
#     sim = sim_fwd

#     # %% ----- save results
#     np.save(
#         save_path + f"{length}_normal_edf_{np.round(pre_chirp, 2)}_pm1550.npy",
#         sim.pulse_out.a_v,
#     )

# %% ------- post chirp sweep -------------------------------------------------
save_path_post_chirp = save_path + "post_chirp_sweep/"

length_edf = 1.5
pre_chirp = np.round(np.arange(0.0, 4.01, 0.01), 2)
post_chirp = np.arange(0, 3.01, 0.01)

P_V = np.zeros((pre_chirp.size, post_chirp.size, pulse.n), dtype=float)
P_T = np.zeros((pre_chirp.size, post_chirp.size, pulse.n), dtype=float)
E_P = np.zeros((pre_chirp.size, post_chirp.size), dtype=float)
V_W = np.zeros((pre_chirp.size, post_chirp.size), dtype=float)
T_W = np.zeros((pre_chirp.size, post_chirp.size), dtype=float)
ERR = np.zeros((pre_chirp.size, post_chirp.size), dtype=float)

# %% ------------- experimental edfa output data ------------------------------
spec = np.genfromtxt(
    "Sichong/20231103-200MHz-preamp4A-withsplitter-front16inches.CSV",
    delimiter=",",
    skip_header=44,
)
# spec = np.genfromtxt("Matt/bottomAmpSpectrum110923.CSV", delimiter=",", skip_header=39)
spec[:, 0] = c / (spec[:, 0] * 1e-9)
spec[:, 1] = 10 ** (spec[:, 1] / 10) * spec[:, 0] ** 2 / c
p_data = pulse.copy()
p_data.import_p_v(v_grid=spec[:, 0], p_v=spec[:, 1], phi_v=None)

# %% --------------------------------------------------------------------------
for n, i in enumerate(tqdm(pre_chirp)):
    a_v = np.load(save_path + f"{length_edf}_normal_edf_{i}_pm1550.npy")
    pulse.a_v[:] = a_v

    # pm1550 after edfa
    pulse.e_p *= loss_spl * loss_ins
    for m, j in enumerate(post_chirp):
        model_pm1550, sim_pm1550 = propagate(
            fiber=pm1550,
            pulse=pulse,
            length=j,
            n_records=None,
            plot=None,
        )
        p_calc = sim_pm1550.pulse_out

        # ------ temporal and frequency bandwidth
        P_V[n, m] = p_calc.p_v
        P_T[n, m] = p_calc.p_t
        E_P[n, m] = p_calc.e_p
        twidth = p_calc.t_width(200)
        vwidth = p_calc.v_width(200)
        V_W[n, m] = vwidth.eqv
        T_W[n, m] = twidth.eqv

        # ------ error calculation
        p_data.e_p = p_calc.e_p
        ERR[n, m] = np.mean((p_data.p_v - p_calc.p_v) ** 2) ** 0.5

P_WL = P_V * dv_dl

# %% ----- save results -------------------------------------------------------
np.save(save_path_post_chirp + "P_V.npy", P_V)
np.save(save_path_post_chirp + "P_T.npy", P_T)
np.save(save_path_post_chirp + "V_W.npy", V_W)
np.save(save_path_post_chirp + "T_W.npy", T_W)

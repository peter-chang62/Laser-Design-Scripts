"""
takes the output files of pre-chirp_edfa_sweep.py and computes a post-chirp
sweep. So, the final result is a 2D sweep of pre-chirp and post-chirp.
"""

# %% --- imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import pynlo
import collections
from tqdm import tqdm
import clipboard


ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0
output = collections.namedtuple("output", ["model", "sim"])

loss_ins = 10 ** (-0.7 / 10)
loss_spl = 10 ** (-0.7 / 10)


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


# %% ------------- pulse ------------------------------------------------------
f_r = 100e6
n = 256
v_min = c / 1750e-9
v_max = c / 1400e-9
v0 = c / 1560e-9
e_p = 5e-3 / f_r

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

pulse_hnlf = pynlo.light.Pulse.Sech(
    n,
    v_min=c / 3000e-9,
    v_max=c / 850e-9,
    v0=c / 1560e-9,
    e_p=e_p,
    t_fwhm=200e-15,
    min_time_window=20e-12,
    alias=2,
)

# %% ----------- fibers -------------------------------------------------------
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = 1.2 / (W * km)

adhnlf = pynlo.materials.SilicaFiber()
adhnlf.load_fiber_from_dict(pynlo.materials.hnlf_5p7)


# %% ------------- experimental data ------------------------------------------
# spec = np.genfromtxt(
#     "Sichong/20231103-200MHz-preamp4A-withsplitter-front16inches.CSV",
#     delimiter=",",
#     skip_header=44,
# )
spec = np.genfromtxt("Matt/bottomAmpSpectrum110923.CSV", delimiter=",", skip_header=39)
spec[:, 0] = c / (spec[:, 0] * 1e-9)
spec[:, 1] = 10 ** (spec[:, 1] / 10) * spec[:, 0] ** 2 / c
p_data = pulse.copy()
p_data.import_p_v(v_grid=spec[:, 0], p_v=spec[:, 1], phi_v=None)

# %% --------------------------------------------------------------------------
length_edf = 1.5

# pre_chirp = np.round(np.arange(0, 3.01, 0.01), 2)
# post_chirp_length = np.arange(0, 3.01, 0.01)

pre_chirp = np.round(np.arange(0.0, 15.05, 0.05), 2)
post_chirp_length = np.arange(0, 3.01, 0.01)

# path = "sim_output/11-03-2023_1.5Pfwd_1.5Pbck_pre-chirp_sweep/"
# path = (
#     r"sim_output/20231012-200MHz-beforepreamp-withsplitter/gamma_6.5/"
#     # + f"11-03-2023_{length_edf}mEDF_1.2Pfwd_1.2Pbck_pre-chirp_sweep/"
#     + f"11-03-2023_{length_edf}mEDF_1.2Pfwd_1.2Pbck_pre-chirp_sweep_mat_Pploss/"
# )
path = r"sim_output/Matt_100MHz_Menlo/gamma_6.5/1.5mEDF_2Wfwd_2W_bck_pre-chirp_sweep/"

P_V = np.zeros((pre_chirp.size, post_chirp_length.size, pulse.n), dtype=float)
P_T = np.zeros((pre_chirp.size, post_chirp_length.size, pulse.n), dtype=float)
E_P = np.zeros((pre_chirp.size, post_chirp_length.size), dtype=float)
V_W = np.zeros((pre_chirp.size, post_chirp_length.size), dtype=float)
T_W = np.zeros((pre_chirp.size, post_chirp_length.size), dtype=float)
ERR = np.zeros((pre_chirp.size, post_chirp_length.size), dtype=float)
for n, i in enumerate(tqdm(pre_chirp)):
    a_v = np.load(path + f"{length_edf}_normal_edf_{i}_pm1550.npy")
    pulse.a_v[:] = a_v

    # pm1550 after edfa
    pulse.e_p *= loss_spl * loss_ins
    for m, j in enumerate(post_chirp_length):
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
np.save(path + "post_chirp_sweep/P_V_0.0to3.0_.01step.npy", P_V)
np.save(path + "post_chirp_sweep/P_T_0.0to3.0_.01step.npy", P_T)
np.save(path + "post_chirp_sweep/V_W_0.0to3.0_.01step.npy", V_W)
np.save(path + "post_chirp_sweep/T_W_0.0to3.0_.01step.npy", T_W)

# %% -------------- plotting --------------------------------------------------
fig, ax = plt.subplots(
    1,
    1,
    num=f"spectral bandwidth for {length_edf} m EDF",
    figsize=np.array([4.02, 3.12]),
)
img = ax.pcolormesh(
    post_chirp_length, pre_chirp, c * 1e9 / v0**2 * V_W, cmap="RdBu_r"
)
clb = plt.colorbar(mappable=img)
clb.set_label("spectral bandwidth (nm)")
ax.set_xlabel("post-chirp length (m)")
ax.set_ylabel("pre-chirp length (m)")
fig.tight_layout()

fig, ax = plt.subplots(
    1, 1, num=f"pulse duration for {length_edf} m EDF", figsize=np.array([4.02, 3.12])
)
img = ax.pcolormesh(post_chirp_length, pre_chirp, T_W * 1e12, cmap="RdBu_r")
clb = plt.colorbar(mappable=img)
clb.set_label("pulse duration (ps)")
ax.set_xlabel("post-chirp length (m)")
ax.set_ylabel("pre-chirp length (m)")
fig.tight_layout()

fig, ax = plt.subplots(
    1,
    1,
    num=f"time bandwidth product for {length_edf} m EDF",
    figsize=np.array([4.02, 3.12]),
)
img = ax.pcolormesh(
    post_chirp_length, pre_chirp, V_W * 1e-12 / (T_W * 1e12), cmap="RdBu_r"
)
clb = plt.colorbar(mappable=img)
clb.set_label("$\\mathrm{\\Delta \\nu / \\Delta t}$ (THz / ps)")
ax.set_xlabel("post-chirp length (m)")
ax.set_ylabel("pre-chirp length (m)")
fig.tight_layout()

fig, ax = plt.subplots(
    1,
    1,
    num=f"error for {length_edf} m EDF",
    figsize=np.array([4.02, 3.12]),
)
img = ax.pcolormesh(post_chirp_length, pre_chirp, ERR, cmap="RdBu_r")
clb = plt.colorbar(mappable=img)
clb.set_label("$\\mathrm{\\Delta \\nu / \\Delta t}$ (THz / ps)")
ax.set_xlabel("post-chirp length (m)")
ax.set_ylabel("pre-chirp length (m)")
fig.tight_layout()

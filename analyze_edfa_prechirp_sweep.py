# %% --- imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
import pandas as pd
from scipy.constants import c
import pynlo
import collections
import copy

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


# %% --------------------------------------------------------------------------
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

pulse_hnlf = pynlo.light.Pulse.Sech(
    n,
    v_min=c / 3000e-9,
    v_max=c / 850e-9,
    v0=c / 1560e-9,
    e_p=1e-3 / f_r,
    t_fwhm=200e-15,
    min_time_window=20e-12,
    alias=2,
)

# %% --------------------------------------------------------------------------
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = 1.2 / (W * km)

adhnlf = pynlo.materials.SilicaFiber()
adhnlf.load_fiber_from_dict(pynlo.materials.hnlf_5p7)


# %% --------------------------------------------------------------------------
spec = np.genfromtxt(
    "20231103-200MHz-preamp4A-withsplitter-front16inches.CSV",
    delimiter=",",
    skip_header=44,
)
spec[:, 0] = c / (spec[:, 0] * 1e-9)
spec[:, 1] = 10 ** (spec[:, 1] / 10) * spec[:, 0] ** 2 / c
p_data = pulse.copy()
p_data.import_p_v(v_grid=spec[:, 0], p_v=spec[:, 1], phi_v=None)

# %% --------------------------------------------------------------------------
length_edf = 0.9
length_pm1550 = np.round(np.arange(0, 3.01, 0.01), 2)
post_chirp_length = 1.4

# path = "sim_output/11-03-2023_1.5Pfwd_1.5Pbck_pre-chirp_sweep/"
path = (
    r"sim_output/20231012-200MHz-beforepreamp-withsplitter/gamma_5/"
    + f"11-03-2023_{length_edf}mEDF_1.2Pfwd_1.2Pbck_pre-chirp_sweep/"
)
P_V = np.zeros((length_pm1550.size, pulse.n), dtype=float)
P_T = np.zeros((length_pm1550.size, pulse.n), dtype=float)
E_P = np.zeros(length_pm1550.size, dtype=float)
V_W = np.zeros(length_pm1550.size, dtype=float)
T_W = np.zeros(length_pm1550.size, dtype=float)
SIM_HNLF = []
SIM_PM1550 = []
P_V_pm1550 = np.zeros((length_pm1550.size, pulse.n), dtype=float)
P_V_hnlf = np.zeros((length_pm1550.size, pulse_hnlf.n), dtype=float)
for n, i in enumerate(length_pm1550):
    a_v = np.load(path + f"{length_edf}_normal_edf_{i}_pm1550.npy")
    pulse.a_v[:] = a_v

    # pm1550 after edfa
    pulse.e_p *= loss_spl * loss_ins
    model_pm1550, sim_pm1550 = propagate(
        fiber=pm1550,
        pulse=pulse,
        length=post_chirp_length,
        n_records=None,
        plot=None,
    )
    P_V_pm1550[n] = sim_pm1550.p_v[-1]
    SIM_PM1550.append(sim_pm1550)
    pulse = sim_pm1550.pulse_out

    # ------ temporal and frequency bandwidth
    P_V[n] = pulse.p_v
    P_T[n] = pulse.p_t
    E_P[n] = pulse.e_p
    twidth = pulse.t_width(200)
    vwidth = pulse.v_width(200)
    V_W[n] = vwidth.eqv
    T_W[n] = twidth.eqv

    # ----- hnlf simulation
    # pulse_hnlf.import_p_v(
    #     v_grid=pulse.v_grid,
    #     p_v=pulse.p_v,
    #     phi_v=np.unwrap(pulse.phi_v),
    # )
    # pulse_hnlf.e_p = pulse.e_p * loss_spl
    # model_hnlf, sim_hnlf = propagate(
    #     fiber=adhnlf,
    #     pulse=pulse_hnlf,
    #     length=1.0,
    #     n_records=100,
    #     plot="wvl",
    # )
    # t_width = np.zeros(100)
    # for m, j in enumerate(sim_hnlf.a_v):  # hnlf fission point
    #     pulse_hnlf.a_v[:] = j[:]
    #     t_width[m] = pulse_hnlf.t_width(200).eqv
    # sim_hnlf.pulse_out.a_v[:] = sim_hnlf.a_v[t_width.argmin()]  # set to fission point
    # P_V_hnlf[n] = sim_hnlf.p_v[t_width.argmin()]
    # SIM_HNLF.append(sim_hnlf)

    # ------- compare to experimental data
    p_data.e_p = pulse.e_p
    plt.figure()
    (idx_wl,) = np.logical_and(
        1500 * 1e-9 < pulse.wl_grid, pulse.wl_grid < 1675 * 1e-9
    ).nonzero()
    plt.plot(
        pulse.wl_grid[idx_wl] * 1e9,
        pulse.p_v[idx_wl] * dv_dl[idx_wl] * f_r * 1e-9 * 1e3,
        label="simulated",
    )
    plt.plot(
        pulse.wl_grid[idx_wl] * 1e9,
        p_data.p_v[idx_wl] * dv_dl[idx_wl] * f_r * 1e-9 * 1e3,
        label="experiment",
    )
    plt.legend(loc="best")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("power (mW / nm)")
    plt.title(f"{length_pm1550[n]}m pre-chirp & {post_chirp_length}m post-chirp")
    plt.tight_layout()


P_WL = P_V * dv_dl

# %% -------------- plotting --------------------------------------------------
fig, ax = plt.subplots(1, 2, num=f"{post_chirp_length}m post chirp")
(idx_wl,) = np.logical_and(
    1500 * 1e-9 < pulse.wl_grid, pulse.wl_grid < 1675 * 1e-9
).nonzero()
(idx_t,) = np.logical_and(-5 * 1e-12 < pulse.t_grid, pulse.t_grid < 5 * 1e-12).nonzero()

ax[0].pcolormesh(
    pulse.wl_grid[idx_wl] * 1e9, length_pm1550, P_WL[:, idx_wl], cmap="CMRmap_r_t"
)
ax[0].set_xlabel("wavelength (nm)")
ax[0].set_ylabel("length of pm1550 pre-chirp")
ax[1].pcolormesh(
    pulse.t_grid[idx_t] * 1e12, length_pm1550, P_T[:, idx_t], cmap="CMRmap_r_t"
)
ax[1].set_xlabel("time (ps)")
ax[1].set_ylabel("length of pm1550 pre-chirp")
fig.tight_layout()

fig, ax = plt.subplots(1, 1, num=f"{post_chirp_length}m post chirp ")
(l1,) = ax.plot(length_pm1550, c / v0**2 * V_W * 1e9, "o", label="frequency width")
ax2 = ax.twinx()
(l2,) = ax2.plot(length_pm1550, T_W * 1e12, "o", color="C1", label="temporal width")
lns = [l1, l2]
labels = [i.get_label() for i in lns]
ax.legend(lns, labels, loc="best")
ax.set_ylabel("frequency width (nm)")
ax2.set_ylabel("temporal width (ps)")
ax.set_xlabel("length of pm1550 pre-chirp (m)")
fig.tight_layout()

fig, ax = plt.subplots(
    1, 2, figsize=np.array([6.4, 8.03]), num=f"{post_chirp_length}m post chirp  "
)
(idx_wl,) = np.logical_and(
    1500 * 1e-9 < pulse.wl_grid, pulse.wl_grid < 1675 * 1e-9
).nonzero()
(idx_t,) = np.logical_and(-5 * 1e-12 < pulse.t_grid, pulse.t_grid < 5 * 1e-12).nonzero()
[
    ax[0].plot(pulse.wl_grid[idx_wl] * 1e9, i[idx_wl] / i.max() + n, "C0", linewidth=2)
    for n, i in enumerate(P_WL)
]
[
    ax[1].plot(pulse.t_grid[idx_t] * 1e12, i[idx_t] / i.max() + n, "C0", linewidth=2)
    for n, i in enumerate(P_T)
]
ax[0].set_xlabel("wavelength (nm)")
ax[1].set_xlabel("time (ps)")
ax[0].get_yaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
fig.tight_layout()

fig, ax = plt.subplots(1, 1, num=f"{post_chirp_length}m post chirp   ")
ax.plot(length_pm1550, E_P * f_r, "o")
ax.set_xlabel("length of pm1550 pre-chirp (m)")
ax.set_ylabel("output power (W)")
fig.tight_layout()

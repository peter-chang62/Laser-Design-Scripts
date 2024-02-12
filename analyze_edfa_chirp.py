# %% --- imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
import pandas as pd
from scipy.constants import c
import pynlo
import collections
import copy
from pynlo.utility import resample_v, resample_t
from tqdm import tqdm
import blit
import time

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

output = collections.namedtuple("output", ["model", "sim"])

PowerSpectralWidth = collections.namedtuple(
    "PowerSpectralWidth", ["fwhm", "rms", "eqv"]
)


PowerEnvelopeWidth = collections.namedtuple(
    "PowerEnvelopeWidth", ["fwhm", "rms", "eqv"]
)


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


def v_width(v_grid, p_v, m=None):
    """
    Calculate the width of the pulse in the frequency domain.

    Set `m` to optionally resample the number of points and change the
    frequency resolution.

    Parameters
    ----------
    m : float, optional
        The multiplicative number of points at which to resample the power
        spectrum. The default is to not resample.

    Returns
    -------
    fwhm : float
        The full width at half maximum of the power spectrum.
    rms : float
        The full root-mean-square width of the power spectrum.
    eqv : float
        The equivalent width of the power spectrum.

    """
    # ---- Power
    p_v = p_v

    # ---- Resample
    if m is None:
        n = v_grid.size
        dv = v_grid[1] - v_grid[0]
    else:
        assert m > 0, "The point multiplier must be greater than 0."
        n = round(m * v_grid.size)
        resampled = resample_v(v_grid, p_v, n)
        # resample_v will return a complex array, but the imaginary
        # components just fluctuate about 0 if resampling a real array
        p_v = resampled.f_v.real
        v_grid = resampled.v_grid
        dv = resampled.dv

    # ---- FWHM
    p_max = p_v.max()
    v_selector = v_grid[p_v >= 0.5 * p_max]
    v_fwhm = dv + (v_selector.max() - v_selector.min())

    # ---- RMS
    p_norm = np.sum(p_v * dv)
    v_avg = np.sum(v_grid * p_v * dv) / p_norm
    v_var = np.sum((v_grid - v_avg) ** 2 * p_v * dv) / p_norm
    v_rms = 2 * v_var**0.5

    # ---- Equivalent
    v_eqv = 1 / np.sum((p_v / p_norm) ** 2 * dv)

    # ---- Construct PowerSpectralWidth
    v_widths = PowerSpectralWidth(fwhm=v_fwhm, rms=v_rms, eqv=v_eqv)
    return v_widths


def t_width(t_grid, p_t, m=None):
    """
    Calculate the width of the pulse in the time domain.

    Set `m` to optionally resample the number of points and change the
    time resolution.

    Parameters
    ----------
    m : float, optional
        The multiplicative number of points at which to resample the power
        envelope. The default is to not resample.

    Returns
    -------
    fwhm : float
        The full width at half maximum of the power envelope.
    rms : float
        The full root-mean-square width of the power envelope.
    eqv : float
        The equivalent width of the power envelope.

    """
    # ---- Power
    p_t = p_t

    # ---- Resample
    if m is None:
        n = t_grid.size
        dt = t_grid[1] - t_grid[0]
    else:
        assert m > 0, "The point multiplier must be greater than 0."
        n = round(m * t_grid.size)
        resampled = resample_t(t_grid, p_t, n)
        p_t = resampled.f_t
        t_grid = resampled.t_grid
        dt = resampled.dt

    # ---- FWHM
    p_max = p_t.max()
    t_selector = t_grid[p_t >= 0.5 * p_max]
    t_fwhm = dt + (t_selector.max() - t_selector.min())

    # ---- RMS
    p_norm = np.sum(p_t * dt)
    t_avg = np.sum(t_grid * p_t * dt) / p_norm
    t_var = np.sum((t_grid - t_avg) ** 2 * p_t * dt) / p_norm
    t_rms = 2 * t_var**0.5

    # ---- Equivalent
    t_eqv = 1 / np.sum((p_t / p_norm) ** 2 * dt)

    # ---- Construct PowerEnvelopeWidth
    t_widths = PowerEnvelopeWidth(fwhm=t_fwhm, rms=t_rms, eqv=t_eqv)
    return t_widths


# %% --------------------------------------------------------------------------
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

# %% ------------- experimental data ------------------------------------------
spec = np.genfromtxt(
    "Sichong/20231103-200MHz-preamp4A-withsplitter-front16inches.CSV",
    delimiter=",",
    skip_header=44,
)
# spec = np.genfromtxt("Matt/bottomAmpSpectrum110923.CSV", delimiter=",", skip_header=39)
spec[:, 0] = c / (spec[:, 0] * 1e-9)
spec[:, 1] = 10 ** (spec[:, 1] / 10) * spec[:, 0] ** 2 / c
spec[:, 1] *= c / spec[:, 0] ** 2  # c / v_grid**2
p_data = pulse.copy()
p_data.import_p_v(v_grid=spec[:, 0], p_v=spec[:, 1], phi_v=None)

# %% --------------------------------------------------------------------------
path = (  # Sichong's stuff
    "sim_output/20231012-200MHz-beforepreamp-withsplitter/gamma_6.5/"
    # + "11-03-2023_0.9mEDF_1.2Pfwd_1.2Pbck_pre-chirp_sweep/"
    # + "11-03-2023_1.5mEDF_1.2Pfwd_1.2Pbck_pre-chirp_sweep/"
    + "11-03-2023_1.5mEDF_1.2Pfwd_1.2Pbck_pre-chirp_sweep_mat_Pploss/"
    + "post_chirp_sweep/"
)
# path = (  # Matt's stuff
#     r"sim_output/Matt_100MHz_Menlo/gamma_6.5/"
#     + "1.5mEDF_2Wfwd_2W_bck_pre-chirp_sweep/post_chirp_sweep/"
# )

pre_chirp = np.arange(0.0, 4.01, 0.01)
post_chirp = np.arange(0.0, 3.01, 0.01)

# %% --------------------------------------------------------------------------
# P_V = np.load(path + "P_V_0.0to3.0_.01step.npy")
# P_WL = P_V * dv_dl
# P_T = np.load(path + "P_T_0.0to3.0_.01step.npy")
# V_W = np.load(path + "V_W_0.0to3.0_.01step.npy")
# T_W = np.load(path + "T_W_0.0to3.0_.01step.npy")

P_V = np.load(path + "P_V.npy")
P_WL = P_V * dv_dl
P_T = np.load(path + "P_T.npy")
V_W = np.load(path + "V_W.npy")
T_W = np.load(path + "T_W.npy")

E_P = np.sum(P_V * pulse.dv, axis=-1)
p_data.e_p = E_P.mean()

# ERR = np.zeros(E_P.shape)
# for n, i in enumerate(P_WL):
#     for m, j in enumerate(i):
#         ERR[n, m] = np.mean((p_data.p_v / p_data.p_v.max() - j / j.max()) ** 2) ** 0.5

# %% --------------------------------------------------------------------------
# instead of crazy plotting (300) figures, animate it with blit instead
# p_wl = P_WL[:, 50]

# fig, ax = plt.subplots(1, 1)
# (l1,) = ax.plot(
#     pulse.wl_grid * 1e9,
#     p_data.p_v * p_data.v_grid**2 / c,
#     "k--",
#     linewidth=2,
#     animated=True,
# )
# (l2,) = ax.plot(
#     pulse.wl_grid * 1e9,
#     p_wl[0],
#     linewidth=2,
#     animated=True,
# )
# fig.tight_layout()

# bm = blit.BlitManager(fig.canvas, [l1, l2])
# bm.update()
# for i in p_wl[1:]:
#     l2.set_ydata(i)
#     bm.update()
#     time.sleep(.01)

# %% --------------------------------------------------------------------------
# I think it's only the ones with a lot of pre-chirp that match okay?
# p_wl = P_WL[330]
# p_t = P_T[330]

# fig, ax = plt.subplots(1, 1)
# [ax.plot(pulse.wl_grid * 1e9, i) for i in p_wl]
# ax.plot(p_data.wl_grid * 1e9, p_data.p_v * p_data.v_grid**2 / c, "k--", linewidth=2)

# fig, ax = plt.subplots(1, 1)
# icorr = np.asarray([np.convolve(i, i[::-1], mode="same") * pulse.dt for i in p_t])
# [ax.plot(pulse.t_grid * 1e15, i) for i in icorr]

import numpy as np
import matplotlib.pyplot as plt
import clipboard
from scipy.interpolate import InterpolatedUnivariateSpline
from re_nlse_joint import EDFA
from scipy.constants import c
import pandas as pd
import pynlo

ps = 1e-12
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

# %% -------------- load absorption coefficients ------------------------------
sigma = np.genfromtxt("Ansys/er_cross_section_fig_6_1.txt")
a = sigma[3:][:, :2]
e = sigma[3:][:, [0, 2]]

sigma_pump = sigma[0, 1]

spl_sigma_a = InterpolatedUnivariateSpline(
    c / a[:, 0][::-1], a[:, 1][::-1], ext="zeros"
)
spl_sigma_e = InterpolatedUnivariateSpline(
    c / e[:, 0][::-1], e[:, 1][::-1], ext="zeros"
)

# %% -------------- load dispersion coefficients ------------------------------
frame = pd.read_excel("nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx")
gvd = frame.to_numpy()[:, :2][1:].astype(float)

wl = gvd[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = 2 * np.pi * c / 1550e-9
polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
polyfit = polyfit[::-1]  # lowest order first

# %% ------------- pulse ------------------------------------------------------
n = 256
v_min = c / 2000e-9
v_max = c / 1000e-9
v0 = c / 1550e-9
e_p = 10e-12
t_fwhm_short = 250e-15
t_fwhm_long = 1.5e-12
min_time_window = 20e-12
pulse_short = pynlo.light.Pulse.Sech(
    n,
    v_min,
    v_max,
    v0,
    e_p,
    t_fwhm_short,
    min_time_window,
    alias=2,
)

pulse_long = pynlo.light.Pulse.Sech(
    n,
    v_min,
    v_max,
    v0,
    e_p,
    t_fwhm_long,
    min_time_window,
    alias=2,
)

# %% ------------- fiber ------------------------------------------------------
fiber = pynlo.materials.SilicaFiber()
fiber.set_beta_from_beta_n(v0, polyfit)
fiber.gamma = 2 / (W * km)

# %% ------------- edfa -------------------------------------------------------
f_r = 100e6
tau = 10.2e-3
r_eff = 1.05e-6
n_ion = 7e24
edfa = EDFA(f_r, tau, r_eff, n_ion)

Pp_fwd = 50e-3
Pp_bck = 50e-3
length = 5

amp = edfa.amplify(
    pulse_short,
    pulse_long,
    fiber,
    Pp_fwd,
    Pp_bck,
    length,
    sigma_pump,
    spl_sigma_a(pulse_short.v_grid),
    spl_sigma_e(pulse_short.v_grid),
    error=1e-3,
)

# %% --------- look at results! -----------------------------------------------
amp.sim_fwd.plot("wvl", num="forward")
amp.sim_bck.plot("wvl", num="backward")

fig, ax = plt.subplots(1, 1)
ax.plot(amp.sim_fwd.z, amp.Pp)
ax_2 = ax.twinx()
ax_2.plot(amp.sim_fwd.z, amp.n2_n, "C1")
# ax_2.set_ylim(ymax=1)

fig, ax = plt.subplots(1, 2, figsize=np.array([9.99, 4.8]))
ax[0].pcolormesh(pulse_short.wl_grid * 1e6, amp.sim_fwd.z, amp.g_fwd, cmap="jet")
ax[0].set_title("forward gain")
ax[1].pcolormesh(pulse_short.wl_grid * 1e6, amp.sim_bck.z, amp.g_bck, cmap="jet")
ax[1].set_title("backward gain")
[i.set_xlabel("wavelength ($\\mathrm{\\mu m}$)") for i in ax]
[i.set_ylabel("position (m)") for i in ax]
fig.tight_layout()

# %% --- imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
import pynlo
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import pandas as pd
from scipy.constants import c


ps = 1e-12
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

# %% ------------------------- load dispersion and gain data ------------------
# gvd data already in s^2/m
gvd_er110 = (
    pd.read_excel("nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx")
    .to_numpy()[:, :2][1:]
    .astype(float)
)
gvd_er80 = (
    pd.read_excel("nLIGHT_Er80-8_125-PM_simulated_GVD_dispersion.xlsx")
    .to_numpy()[:, :2][1:]
    .astype(float)
)

# Erbium Doped Fiber Amplifier,
# https://optics.ansys.com/hc/en-us/articles/360042819353-Erbium-doped-fiber-amplifier.
# units: 10^-25 m^2
sigma = np.genfromtxt("er_cross_section_fig_6_1.txt")
a = np.c_[sigma[:, 0], sigma[:, 1]]
e = np.c_[sigma[:, 0], sigma[:, 2]]

spl_a = UnivariateSpline(
    c / a[:, 0][::-1],
    a[:, 1][::-1],
    k=3,
    s=0.0,
)
spl_e = UnivariateSpline(
    c / e[:, 0][::-1],
    e[:, 1][::-1],
    k=3,
    s=0.0,
)

# %% ------------------------- fiber, pulse, and model ------------------------
n = 2
v_min = c / 1650e-9
v_max = c / 1450e-9
v0 = c / 1550e-9
e_p = 50e-12
t_fwhm = 250e-15
min_time_window = 10e-12
pulse = pynlo.light.Pulse.Sech(
    n, v_min, v_max, v0, e_p, t_fwhm, min_time_window, alias=2
)

# er110 dispersion coefficients
wl = gvd_er110[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = v0 * 2 * np.pi
polyfit_gvd_er110 = np.polyfit(omega - omega0, gvd_er110[:, 1], deg=3)
polyfit_gvd_er110 = polyfit_gvd_er110[::-1]  # lowest order first

# er80 dispersion coefficients
wl = gvd_er80[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = v0 * 2 * np.pi
polyfit_gvd_er80 = np.polyfit(omega - omega0, gvd_er80[:, 1], deg=3)
polyfit_gvd_er80 = polyfit_gvd_er80[::-1]  # lowest order first

fiber = pynlo.materials.SilicaFiber()
fiber.set_beta_from_beta_n(v0, polyfit_gvd_er110)
fiber.gamma = 4 / (W * km)


# %% ------------------ option to make gain z-dependent -----------------------
e_sat = 1e-9


# %% ------------------------- simulation -------------------------------------
model = fiber.generate_model(
    pulse,
    t_shock="auto",
    raman_on=True,
    alpha=spl_e(pulse.v_grid),
    method="nlse",
)
dz = model.estimate_step_size(local_error=1e-6)
sim = model.simulate(
    z_grid=0.05,
    dz=dz,
    local_error=1e-6,
    n_records=100,
    plot=None,
)
print(sim.pulse_out.e_p * 100e6 * 1e3, "mW")

# %% ------------------ plotting dispersion and gain data ---------------------
fig, ax = plt.subplots(1, 1)
wl = np.linspace(1000, 2000, 1000)
ax.plot(gvd_er80[:, 0], gvd_er80[:, 1], ".")
ax.plot(gvd_er110[:, 0], gvd_er110[:, 1], ".")

fig, ax = plt.subplots(1, 1)
d = np.arange(-1, 1.2, 0.2)
wl = np.linspace(*a[:, 0][[0, -1]], 1000)
a_spl = spl_a(c / wl)
e_spl = spl_e(c / wl)
for i in range(d.shape[0]):
    ax.plot(wl, e_spl * (1 + d[i]) - a_spl * (1 - d[i]))

a_orig = np.genfromtxt("absorption.csv", delimiter=",")
e_orig = np.genfromtxt("emission.csv", delimiter=",")
e_orig -= e_orig.min()

# ax.plot(a_orig[:, 0], -2 * a_orig[:, 1], 'k--', linewidth=2)
# ax.plot(e_orig[:, 0], 2 * e_orig[:, 1], 'k--', linewidth=2)

d = np.linspace(-1, 1, 100)
G = np.zeros((d.size, e_spl.size))
for i in range(G.shape[0]):
    G[i] = e_spl * (1 + d[i]) - a_spl * (1 - d[i])

fig, ax = plt.subplots(1, 1)
ax.pcolormesh(wl, d, G, cmap="jet")
ax.set_xlabel("wavlength (nm)")
ax.set_ylabel("pumping")
fig.tight_layout()

# %% ------------------ plotting sim results ----------------------------------
sim.plot("wvl")

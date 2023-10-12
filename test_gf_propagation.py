# %% --- imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
import pynlo
from scipy.interpolate import InterpolatedUnivariateSpline
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

# Erbium Doped Fiber Amplifier,
# https://optics.ansys.com/hc/en-us/articles/360042819353-Erbium-doped-fiber-amplifier.
# units: 10^-25 m^2
system = "ansys"
# system = "optiwave"
if system.lower() == "ansys":
    sigma = np.genfromtxt("Ansys/er_cross_section_fig_6_1.txt")
    sigma = sigma[3:]
elif system.lower() == "optiwave":
    sigma = np.genfromtxt("Optiwave/Erbium.dat")
    sigma[:, 0] *= 1e-9
a = np.c_[sigma[:, 0], sigma[:, 1]]
e = np.c_[sigma[:, 0], sigma[:, 2]]

factor = 11 * np.log(10) / a[:, 1].max()
a[:, 1] *= factor
e[:, 1] *= factor

spl_a = InterpolatedUnivariateSpline(
    c / a[:, 0][::-1],
    a[:, 1][::-1],
    ext="zeros",
)
spl_e = InterpolatedUnivariateSpline(
    c / e[:, 0][::-1],
    e[:, 1][::-1],
    ext="zeros",
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
    z_grid=.3,
    dz=dz,
    local_error=1e-6,
    n_records=100,
    plot=None,
)
print(sim.pulse_out.e_p * 100e6 * 1e3, "mW")

# %% ------------------ plotting dispersion and gain data ---------------------
fig, ax = plt.subplots(1, 1)
wl = np.linspace(1000, 2000, 1000)
ax.plot(gvd_er110[:, 0], gvd_er110[:, 1])
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("GVD ($\\mathrm{s^2/m}$)")
fig.tight_layout()

fig, ax = plt.subplots(1, 1)
d = np.arange(-1, 1.2, 0.2)
wl = np.linspace(*a[:, 0][[0, -1]], 1000)
a_spl = spl_a(c / wl)
e_spl = spl_e(c / wl)
for i in range(1, d.shape[0] - 1):
    ax.plot(wl * 1e9, e_spl / 2 * (1 + d[i]) - a_spl / 2 * (1 - d[i]))
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("gain ($\\mathrm{m^{-1}}$)")
fig.tight_layout()

d = np.linspace(-1, 1, 100)
G = np.zeros((d.size, pulse.n))
for i in range(G.shape[0]):
    G[i] = spl_e(pulse.v_grid) * (1 + d[i]) - spl_a(pulse.v_grid) * (1 - d[i])

fig, ax = plt.subplots(1, 1)
ax.pcolormesh(pulse.wl_grid * 1e9, d, G, cmap="jet")
ax.set_xlabel("wavlength (nm)")
ax.set_ylabel("relative inversion")
fig.tight_layout()

# %% ------------------ plotting sim results ----------------------------------
sim.plot("wvl")

# %% ---- pulse width
p = pulse.copy()
rms_v = np.zeros(sim.z.size)
fwhm_v = np.zeros(sim.z.size)
eqv_v = np.zeros(sim.z.size)
rms_t = np.zeros(sim.z.size)
fwhm_t = np.zeros(sim.z.size)
eqv_t = np.zeros(sim.z.size)
e_p = np.zeros(sim.z.size)
for n, a_t in enumerate(sim.a_t):
    p.a_t[:] = a_t[:]
    v_width = p.v_width(200)
    rms_v[n] = v_width.rms
    fwhm_v[n] = v_width.fwhm
    eqv_v[n] = v_width.eqv

    t_width = p.t_width(200)
    rms_t[n] = t_width.rms
    fwhm_t[n] = t_width.fwhm
    eqv_t[n] = t_width.eqv

    e_p[n] = p.e_p

# %% -----
fig, ax = plt.subplots(1, 3, figsize=np.array([11.95, 4.8]))
ax[0].plot(sim.z, eqv_v * 1e-12)
ax[1].plot(sim.z, c / v0**2 * eqv_v * 1e9)
ax[2].plot(sim.z, e_p * 100e6 * 1e3)
[i.set_xlabel("z") for i in ax]
ax[0].set_ylabel("bandwidth (THz)")
ax[1].set_ylabel("bandwidth (nm)")
ax[2].set_ylabel("power (mW)")
fig.tight_layout()

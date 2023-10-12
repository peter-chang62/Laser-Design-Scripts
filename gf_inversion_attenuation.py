# %% ----- imports
import numpy as np
import clipboard
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
import pynlo
from scipy.integrate import odeint


ps = 1e-12
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0
c = 299792458.0

# %% -------------- load absorption coefficients ------------------------------
sigma = np.genfromtxt("Ansys/er_cross_section_fig_6_1.txt")
a = sigma[3:][:, :2]
e = sigma[3:][:, [0, 2]]

factor = 9 / a[:, 1].max()
a[:, 1] *= factor
e[:, 1] *= factor
sigma_pump = sigma[0, 1] * factor

spl_a = InterpolatedUnivariateSpline(c / a[:, 0][::-1], a[:, 1][::-1], ext="zeros")
spl_e = InterpolatedUnivariateSpline(c / e[:, 0][::-1], e[:, 1][::-1], ext="zeros")

# %% -------------- load dispersion coefficients ------------------------------
frame = pd.read_excel("nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx")
gvd = frame.to_numpy()[:, :2][1:].astype(float)

wl = gvd[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = 2 * np.pi * c / 1550e-9
polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
polyfit = polyfit[::-1]  # lowest order first

# %% ------------- pulse ------------------------------------------------------¬
n = 256
v_min = c / 2000e-9
v_max = c / 1000e-9
v0 = c / 1550e-9
e_p = 50e-12
t_fwhm = 250e-15
min_time_window = 10e-12
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

# %% ------------- fiber ------------------------------------------------------
fiber = pynlo.materials.SilicaFiber()
fiber.set_beta_from_beta_n(v0, polyfit)
fiber.gamma = 4 / (W * km)


# %% ------------- calculate gain ---------------------------------------------
def dpdz(p, z, alpha_p, p_sat):
    return -alpha_p * p / (1 + p / p_sat)


length = 10
z = np.linspace(0, length, 1000)
p_0 = 1
p_sat = p_0 / 20
alpha_p = 14.5

sol = np.squeeze(
    odeint(
        dpdz,
        np.array([p_0]),
        z,
        args=(alpha_p, p_sat),
    )
)

eta = spl_e(c / 1550e-9) / spl_a(c / 1550e-9)
p_thresh = p_sat / eta

spl_inversion = InterpolatedUnivariateSpline(z, sol - p_thresh)
max_gain = spl_e(pulse.v_grid)
max_absorption = spl_a(pulse.v_grid)


e_p_test = []


def alpha(z, p_v):
    if p_v is not None:
        dv = pulse.dv
        e_p_test.append([z, np.sum(p_v * dv)])
    inv = spl_inversion(z)
    return (max_gain * (1 + inv) - max_absorption * (1 - inv)) / 2


# %% ------------- model ------------------------------------------------------
model = fiber.generate_model(
    pulse,
    t_shock="auto",
    raman_on=True,
    alpha=alpha,
    method="nlse",
)

# %% ------------- sim --------------------------------------------------------
dz = model.estimate_step_size()
sim = model.simulate(z_grid=length, dz=dz, n_records=100)

# %% ------------- plot -------------------------------------------------------
# ----- sim results
sim.plot("wvl")
print(sim.pulse_out.e_p * 100e6 * 1e3, "mW")

# %% ----- population inversion
fig, ax = plt.subplots(1, 1)
ax.plot(z, spl_inversion(z))
ax.set_xlabel("z")
ax.set_ylabel("population inversion")
fig.tight_layout()

# %% ----- pulse energy
fig, ax = plt.subplots(1, 1)
Z = np.linspace(0, length, 100)
G = np.zeros((Z.size, pulse.n))
for n, i in enumerate(Z):
    G[n] = alpha(i, None)
ax.pcolormesh(pulse.wl_grid * 1e9, Z, G, cmap="jet")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("z")
fig.tight_layout()

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

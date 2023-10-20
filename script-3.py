# %% ----- imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import pynlo
import clipboard
import pandas as pd
from scipy.constants import c
from re_nlse_joint import EDFA
from scipy.optimize import minimize, Bounds

ps = 1e-12
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

plot = "wvl"

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
sigma_p = spl_sigma_a(c / 980e-9)


# %% -------------- load dispersion coefficients ------------------------------
# frame = pd.read_excel(
#     "NLight_provided/nLIGHT Er80-4_125-HD-PM simulated fiber dispersion.xlsx"
# )
frame = pd.read_excel(
    "NLight_provided/nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx"
)
gvd = frame.to_numpy()[:, :2][1:].astype(float)

wl = gvd[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = 2 * np.pi * c / 1550e-9
polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
polyfit = polyfit[::-1]  # lowest order first

# %% ------------- pulse ------------------------------------------------------
n = 256
v_min = c / 1700e-9
v_max = c / 1400e-9
v0 = c / 1550e-9
e_p = 30e-3 / 100e6
t_fwhm = 300e-15
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


# %% ------------- EDF --------------------------------------------------------
tau = 9e-3
r_eff = 3.06e-6 / 2
a_eff = np.pi * r_eff**2
n_ion = 110 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)  # dB/m absorption at 1530 nm

gamma = 1

fiber = EDFA(
    f_r=100e6,
    overlap_p=1.0,
    overlap_s=1.0,
    n_ion=n_ion,
    a_eff=a_eff,
    sigma_p=sigma_p,
    sigma_a=spl_sigma_a(pulse.v_grid),
    sigma_e=spl_sigma_e(pulse.v_grid),
    tau=tau,
)
fiber.set_beta_from_beta_n(v0, polyfit)
fiber.gamma = gamma / (W * km)


# %% ------------- forward pumped EDFA ----------------------------------------
Pp_0 = 850e-3

model, dz = fiber.generate_model(
    pulse,
    t_shock="auto",
    raman_on=False,
    Pp_fwd=Pp_0,
)
length = 1.5
sim_fwd = model.simulate(length, dz=dz, n_records=100, plot=plot)


# %% ------------- backward pumped EDFA ---------------------------------------
def func(Pp_bck):
    (Pp_bck,) = Pp_bck
    model, dz = fiber.generate_model(
        pulse,
        t_shock="auto",
        raman_on=False,
        Pp_bck=Pp_bck,
    )
    global sim_bck
    sim_bck = model.simulate(length, dz=dz, n_records=100, plot=plot)
    return abs(sim_bck.Pp[-1] - Pp_0) ** 2


guess = 1e-6
res = minimize(func, np.array([guess]), bounds=Bounds(lb=0, ub=np.inf), tol=1e-7)


# %% ------------- forward and backward pumped EDFA ---------------------------
def func(Pp_bck):
    (Pp_bck,) = Pp_bck
    model, dz = fiber.generate_model(
        pulse,
        t_shock="auto",
        raman_on=False,
        Pp_fwd=Pp_0,
        Pp_bck=Pp_bck,
    )
    global sim_both
    sim_both = model.simulate(length, dz=dz, n_records=100, plot=plot)
    return abs(sim_both.Pp[-1] - Pp_0) ** 2


guess = 1e-6
res_both = minimize(func, np.array([guess]), bounds=Bounds(lb=0, ub=np.inf), tol=1e-7)


# %% ------------- look at results! -------------------------------------------
sim = sim_fwd
fig, ax = plt.subplots(1, 1)
ax.plot(sim.z, np.sum(sim.p_v * pulse.dv, axis=1) * 100e6, label="signal", linewidth=2)
ax.plot(sim.z, sim.Pp, label="pump", linewidth=2)
ax2 = ax.twinx()
ax2.plot(sim.z, sim.n2_n, color="C2", linewidth=2, linestyle="--")
ax.set_ylabel("power (W)")
ax.set_xlabel("position (m)")
ax.set_title("broadband amplification with PyNLO")
ax.grid()
fig.tight_layout()

fig, ax = plt.subplots(1, 1)
img = ax.pcolormesh(pulse.wl_grid * 1e9, sim.z, sim.g_v, cmap="jet")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("position (m)")
fig.colorbar(img)
ax.set_title("gain forward pumping")
fig.tight_layout()

sim.plot("wvl", num="forward pumping")

# %% -----
sim = sim_bck
fig, ax = plt.subplots(1, 1)
ax.plot(sim.z, np.sum(sim.p_v * pulse.dv, axis=1) * 100e6, label="signal", linewidth=2)
ax.plot(sim.z, sim.Pp, label="pump", linewidth=2)
ax2 = ax.twinx()
ax2.plot(sim.z, sim.n2_n, color="C2", linewidth=2, linestyle="--")
ax.set_ylabel("power (W)")
ax.set_xlabel("position (m)")
ax.set_title("broadband amplification with PyNLO")
ax.grid()
fig.tight_layout()

fig, ax = plt.subplots(1, 1)
img = ax.pcolormesh(pulse.wl_grid * 1e9, sim.z, sim.g_v, cmap="jet")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("position (m)")
fig.colorbar(img)
ax.set_title("gain backward pumping")
fig.tight_layout()

sim.plot("wvl", num="backward pumping")

# %% -----
sim = sim_both
fig, ax = plt.subplots(1, 1)
ax.plot(sim.z, np.sum(sim.p_v * pulse.dv, axis=1) * 100e6, label="signal", linewidth=2)
ax.plot(sim.z, sim.Pp, label="pump", linewidth=2)
ax2 = ax.twinx()
ax2.plot(sim.z, sim.n2_n, color="C2", linewidth=2, linestyle="--")
ax.set_ylabel("power (W)")
ax.set_xlabel("position (m)")
ax.set_title("broadband amplification with PyNLO")
ax.grid()
fig.tight_layout()

fig, ax = plt.subplots(1, 1)
img = ax.pcolormesh(pulse.wl_grid * 1e9, sim.z, sim.g_v, cmap="jet")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("position (m)")
fig.colorbar(img)
ax.set_title("gain forward and backward pumping")
fig.tight_layout()

sim.plot("wvl", num="forwad and backward pumping")

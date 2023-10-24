# %% ----- imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import pynlo
import clipboard
import pandas as pd
from scipy.constants import c
from re_nlse_joint_5level import EDF
from scipy.optimize import minimize, Bounds
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


def propagate(fiber, pulse, length):
    fiber: pynlo.materials.SilicaFiber
    model = fiber.generate_model(pulse)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=100)
    return output(model=model, sim=sim)


class BackwardSim:
    def __init__(self, fiber, pulse, length, Pp_fwd, Pp_bck):
        self.fiber = fiber
        self.pulse = pulse
        self.length = length
        self.Pp_fwd = Pp_fwd
        self.Pp_bck = Pp_bck

    def func(self, Pp_bck):
        (Pp_bck,) = Pp_bck
        self.model, dz = self.fiber.generate_model(
            self.pulse,
            Pp_fwd=self.Pp_fwd,
            Pp_bck=Pp_bck,
        )
        self.sim = self.model.simulate(self.length, dz=dz, n_records=100)
        return abs(self.sim.Pp[-1] - self.Pp_bck) ** 2


def amplify(fiber, pulse, length, Pp_fwd, Pp_bck):
    fiber: EDF
    if Pp_bck == 0:
        model, dz = fiber.generate_model(pulse, Pp_fwd=Pp_fwd)
        sim = model.simulate(length, dz=dz, n_records=100)

    else:
        backward_sim = BackwardSim(fiber, pulse, length, Pp_fwd, Pp_bck)
        guess = 1e-6
        minimize(
            backward_sim.func,
            np.array([guess]),
            bounds=Bounds(lb=0, ub=np.inf),
            tol=1e-7,
        )
        sim = backward_sim.sim
        model = backward_sim.model

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
frame = pd.read_excel(
    "NLight_provided/nLIGHT Er80-4_125-HD-PM simulated fiber dispersion.xlsx"
)
# frame = pd.read_excel(
#     "NLight_provided/nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx"
# )
gvd = frame.to_numpy()[:, :2][1:].astype(float)

wl = gvd[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = 2 * np.pi * c / 1550e-9
polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
polyfit = polyfit[::-1]  # lowest order first

# %% ------------- pulse ------------------------------------------------------
f_r = 200e6
e_p = 1e-3 / f_r

n = 256
v_min = c / 1700e-9
v_max = c / 1400e-9
v0 = c / 1550e-9
t_fwhm = 250e-15
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

# %% --------- passive fibers -------------------------------------------------
pm1550_info = pynlo.materials.pm1550
pm1550_info["D slope slow axis"] = 0
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pm1550_info)

edf_passive = pynlo.materials.SilicaFiber()
edf_passive.set_beta_from_beta_n(v0, polyfit[:1])
edf_passive.gamma = 1 / (W * km)

# %% ------------ active fiber ------------------------------------------------
tau = 9 * ms
r_eff = 3.06 * um / 2
a_eff = np.pi * r_eff**2
n_ion = 80 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)

sigma_a = spl_sigma_a(pulse.v_grid)
sigma_e = spl_sigma_e(pulse.v_grid)
sigma_p = spl_sigma_a(c / 980e-9)

edf = EDF(
    f_r=f_r,
    overlap_p=1.0,
    overlap_s=1.0,
    n_ion=n_ion,
    a_eff=a_eff,
    sigma_p=sigma_p,
    sigma_a=sigma_a,
    sigma_e=sigma_e,
)
edf.set_beta_from_beta_n(v0, polyfit)
edf.gamma = 1 / (W * km)

# %% ------ quick test
model_fwd, sim_fwd = amplify(edf, pulse, 3, 1, 0)
model_bck, sim_bck = amplify(edf, pulse, 3, 0, 1)
model_both, sim_both = amplify(edf, pulse, 3, 1, 1)

# %% ------------- look at results! -------------------------------------------
for sim, num in zip([sim_fwd, sim_bck, sim_both], ["fwd", "bck", "both"]):
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        sim.z, np.sum(sim.p_v * pulse.dv, axis=1) * 100e6, label="signal", linewidth=2
    )
    ax.plot(sim.z, sim.Pp, label="pump", linewidth=2)
    ax.set_ylabel("power (W)")
    ax.set_xlabel("position (m)")
    ax.set_title("broadband amplification with PyNLO")
    ax.grid()
    fig.tight_layout()

    sim.plot("wvl", num=num)

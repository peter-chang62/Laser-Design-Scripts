# %% ----- imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import pynlo
import clipboard
import pandas as pd
from scipy.constants import c
from re_nlse_joint_5level import EDF
import collections
from scipy.optimize import minimize, Bounds

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

output = collections.namedtuple("output", ["model", "sim"])
n_records = None


def propagate(fiber, pulse, length):
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
    model = fiber.generate_model(pulse, t_shock=None, raman_on=False)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


class _BackwardSim:
    def __init__(self, pulse, fiber, length, Pp_fwd, Pp_bck):
        self.pulse = pulse
        self.fiber = fiber
        self.Pp_fwd = Pp_fwd
        self.Pp_bck = Pp_bck
        self.length = length

        self.pulse: pynlo.light.Pulse
        self.fiber: EDF

    def func(self, Pp_bck):
        (Pp_bck,) = Pp_bck
        self.model, dz = self.fiber.generate_model(
            self.pulse,
            t_shock=None,
            raman_on=False,
            Pp_fwd=self.Pp_fwd,
            Pp_bck=Pp_bck,
        )
        self.sim = self.model.simulate(self.length, dz=dz, n_records=n_records)
        return abs(self.sim.Pp[-1] - self.Pp_bck) ** 2


def amplify(fiber, pulse, length, Pp_fwd, Pp_bck=0.0):
    """
    amplifies a given pulse through edf fiber of given length and pump power

    Args:
        fiber (instance of EDF): erbium doped fiber
        pulse (instance of Pulse): Pulse
        length (float): edf length
        Pp_fwd (float): forward pump power
        Pp_bck (float): backwards pump power

    Returns:
        output: model, sim
    """
    fiber: EDF

    if Pp_bck == 0:
        model, dz = fiber.generate_model(
            pulse, t_shock=None, raman_on=False, Pp_fwd=Pp_fwd
        )
        sim = model.simulate(length, dz=dz, n_records=n_records)
        return output(model=model, sim=sim)

    else:
        backwards_sim = _BackwardSim(pulse, fiber, length, Pp_fwd, Pp_bck)
        minimize(
            backwards_sim.func,
            np.array([1e-6]),
            bounds=Bounds(lb=0, ub=np.inf),
            tol=1e-7,
        )
        sim = backwards_sim.sim
        model = backwards_sim.model
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
# frame = pd.read_excel(
#     "NLight_provided/nLIGHT Er80-4_125-HD-PM simulated fiber dispersion.xlsx"
# )
frame = pd.read_excel(
    "NLight_provided/nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx"
)
gvd = frame.to_numpy()[:, :2][1:].astype(float)

wl = gvd[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = 2 * np.pi * c / 1560e-9
polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
polyfit = polyfit[::-1]  # lowest order first

# %% ------------- pulse ------------------------------------------------------
f_r = 200e6
n = 256
v_min = c / 1800e-9
v_max = c / 1400e-9
v0 = c / 1560e-9
e_p = 1e-3 / f_r

t_fwhm = 2e-12
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

# %% --------- passive fibers -------------------------------------------------
gamma_pm1550 = 1
gamma_edf = 6

pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = gamma_pm1550 / (W * km)

# %% ------------ active fiber ------------------------------------------------
tau = 9 * ms
r_eff = 3.06 * um / 2
a_eff = np.pi * r_eff**2
n_ion = 110 / 10 * np.log(10) / spl_sigma_a(c / 1530e-9)

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
edf.set_beta_from_beta_n(v0, polyfit)  # only gdd
edf.gamma = gamma_edf / (W * km)

# %% ------- figure 9 laser cavity --------------------------------------------
beta2_g = polyfit[0]
D_g = -2 * np.pi * c / 1560e-9**2 * beta2_g / ps * nm * km
D_p = 18
l_t = c / 1.5 / f_r  # total cavity length

# target round trip dispersion in the loop
# D_l = 7
# l_p_s = 0.11  # shortest straight section I can do
# l_g = (D_l - D_p) * (l_t - 2 * l_p_s) / (D_g - D_p)
# l_p_l = (D_g - D_l) * (l_t - 2 * l_p_s) / (D_g - D_p)

# target total round trip dispersion: D_l -> D_rt
D_rt = 7
l_p_s = 0.13  # length of straight section
l_g = -l_t * (D_p - D_rt) / (D_g - D_p)
l_p = l_t - l_g  # passive fiber length
l_p_l = l_p - l_p_s * 2  # passive fiber in loop

assert np.all(np.array([l_g, l_p_s, l_p_l]) >= 0)

p_gf = pulse.copy()  # gain first
p_pf = pulse.copy()  # passive first
p_s = pulse.copy()  # straight section
p_out = pulse.copy()

# parameters
Pp = 500 * 1e-3
phi = np.pi / 2
include_loss = True
do_backward_pump = False
loss = 0.8

# set up plot
fig, ax = plt.subplots(2, 2)

loop_count = 0
done = False
while not done:
    # ------------- start at splitter --------------------------
    p_gf.a_t[:] = p_s.a_t[:] / 2  # straight / 2
    p_pf.a_t[:] = p_s.a_t[:] / 2  # straight / 2

    # ------------- gain fiber first --------------------------
    # gain section
    if include_loss:
        # splice from splitter to gain
        p_gf.p_v[:] *= loss

    # gain fiber
    p_gf.a_t[:] = amplify(edf, p_gf, l_g, Pp).sim.pulse_out.a_t[:]

    if include_loss:
        # splice from gain to phase bias
        p_gf.p_v[:] *= loss
        # phase bias insertion loss
        p_gf.p_v[:] *= loss

    # passive fiber
    p_gf.a_t[:] = propagate(pm1550, p_gf, l_p_l).sim.pulse_out.a_t[:]

    # ------------- passive fiber first --------------------------
    # passive fiber
    p_pf.a_t[:] = propagate(pm1550, p_pf, l_p_l).sim.pulse_out.a_t[:]

    if include_loss:
        # phase bias insertion loss
        p_pf.p_v[:] *= loss
        # splice from phase bias to gain
        p_pf.p_v[:] *= loss

    # gain fiber
    p_pf.a_t[:] = amplify(
        edf,
        p_pf,
        l_g,
        0 if do_backward_pump else Pp,
        Pp if do_backward_pump else 0,
    ).sim.pulse_out.a_t[:]

    if include_loss:
        # splice from gain to splitter
        p_pf.p_v[:] *= loss

    # ------------- back to splitter --------------------------
    p_s.a_t[:] = p_gf.a_t[:] * np.exp(1j * phi) + p_pf.a_t[:]
    p_out.a_t[:] = p_gf.a_t[:] * np.exp(1j * phi) - p_pf.a_t[:]

    # ------------- straight section --------------------------
    if include_loss:
        # splitter insertion loss
        p_s.p_v[:] *= loss

    p_s.a_t[:] = -propagate(pm1550, p_s, l_p_s).sim.pulse_out.a_t[:]

    if include_loss:
        # micro mirror / piezo insertion loss
        p_s.p_v[:] *= loss

    p_s.a_t[:] = propagate(pm1550, p_s, l_p_s).sim.pulse_out.a_t[:]

    if include_loss:
        # splitter insertion loss
        p_s.p_v[:] *= loss

    center = pulse.n // 2
    p_s.a_t[:] = np.roll(p_s.a_t, center - p_s.p_t.argmax())

    # update plot
    if loop_count == 0:
        (l1,) = ax[0, 0].plot(p_out.wl_grid * 1e9, p_out.p_v / p_out.p_v.max())
        (l2,) = ax[0, 1].plot(p_out.t_grid * 1e12, p_out.p_t / p_out.p_t.max())
        (l3,) = ax[1, 0].plot(p_s.wl_grid * 1e9, p_s.p_v / p_s.p_v.max())
        (l4,) = ax[1, 1].plot(p_s.t_grid * 1e12, p_s.p_t / p_s.p_t.max())
        plt.pause(0.01)
    else:
        l1.set_ydata(p_out.p_v / p_out.p_v.max())
        l2.set_ydata(p_out.p_t / p_out.p_t.max())
        l3.set_ydata(p_s.p_v / p_s.p_v.max())
        l4.set_ydata(p_s.p_t / p_s.p_t.max())
        plt.pause(0.01)

    if loop_count == 150:
        done = True

    loop_count += 1
    print(
        loop_count, np.round(p_out.e_p * f_r * 1e3, 4), np.round(p_s.e_p * f_r * 1e3, 4)
    )

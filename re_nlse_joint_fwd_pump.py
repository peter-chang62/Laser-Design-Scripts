# %% ----- imports
import numpy as np
import clipboard
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
import pynlo
from scipy.constants import h, c
from scipy.integrate import odeint
import collections
from tqdm import tqdm

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
e_p = 10e-12
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

# %% ------------- fiber ------------------------------------------------------
fiber = pynlo.materials.SilicaFiber()
fiber.set_beta_from_beta_n(v0, polyfit)
fiber.gamma = 4 / (W * km)

# %% ------------- population inversion and gain ------------------------------
f_r = 100e6

# parameters are from Ansys:
# https://optics.ansys.com/hc/en-us/articles/360042819353-Erbium-doped-fiber-amplifier
tau = 10.2e-3
b_eff = 1.05e-6
a_eff = np.pi * b_eff**2
n_ion = 7e24


def _term(overlap, sigma, P, nu):
    return overlap * P * sigma / (a_eff * h * nu)


def n2_over_n(Pp, p_v, sigm_p, sigm_a, sigm_e):
    pump_term = _term(1, sigm_p, Pp, c / 980e-9)
    num_sum = _term(1, sigm_a, f_r * p_v, pulse.v_grid)
    denom_sum = _term(1, sigm_a + sigm_e, f_r * p_v, pulse.v_grid)
    num = pump_term + np.sum(num_sum * pulse.dv)
    denom = 1 / tau + pump_term + np.sum(denom_sum * pulse.dv)
    return num / denom


# calculate gain for a given inversion and absorption and emission cross-section
# this is a general equation for both signal and pump
def dpdz(overlap, n2_n, p, sigm_a, sigm_e, alph):
    n2 = n2_n * n_ion
    n1 = n_ion - n2
    add = overlap * sigm_e * n2 * p
    subtract = overlap * sigm_a * n1 * p + alph * p
    return add - subtract


def alpha(z, p_v, Pp, overlap, sigm_p, sigm_a, sigm_e):
    n2_n = n2_over_n(Pp, p_v, sigm_p, sigm_a, sigm_e)
    n2 = n2_n * n_ion
    n1 = n_ion - n2
    return overlap * n2 * sigm_e - overlap * n1 * sigm_a


# %% ------------- starting pump and gain profile -----------------------------
def amplify(Pp_0, length):
    # starting pump profile is just a decaying exponential
    # Pp_0 = 100
    # length = 5.0
    z_pump_grid = np.linspace(0, length, 1000)
    func = lambda p, z_pump_grid: dpdz(1, 0, p, sigma_pump, 0, 0)
    sol = odeint(func, np.array([Pp_0]), z_pump_grid)
    spl_Pp = InterpolatedUnivariateSpline(z_pump_grid, sol, ext="zeros")

    # %% ------------- model --------------------------------------------------
    sigma_a = spl_a(pulse.v_grid)
    sigma_e = spl_e(pulse.v_grid)
    model = fiber.generate_model(
        pulse,
        t_shock="auto",
        raman_on=True,
        alpha=lambda z, p_v: alpha(z, p_v, spl_Pp(z), 1, sigma_pump, sigma_a, sigma_e),
        method="nlse",
    )

    # %% ------------- sim ----------------------------------------------------
    dz = model.estimate_step_size()
    sim = model.simulate(z_grid=length, dz=dz, n_records=250)
    p_out = sim.pulse_out

    # %% ------------- iterate! -----------------------------------------------
    REL_ERROR = []
    rel_error = 100
    while rel_error > 1e-3:
        # calculate n2_n and grid it
        n2_n = np.zeros(sim.z.size)
        for n, z in enumerate(sim.z):
            n2_n[n] = n2_over_n(spl_Pp(z), sim.p_v[n], sigma_pump, sigma_a, sigma_e)
        spl_n2_n = InterpolatedUnivariateSpline(sim.z, n2_n, ext="const")

        # use n2_n to calculate the updated pump profile
        func = lambda p, z: dpdz(1, spl_n2_n(z), p, sigma_pump, 0, 0)
        sol = odeint(func, np.array([Pp_0]), z_pump_grid)
        spl_Pp = InterpolatedUnivariateSpline(z_pump_grid, sol, ext="zeros")

        # use the updated pump profile to re-propagate the pulse
        model = fiber.generate_model(
            pulse,
            t_shock="auto",
            raman_on=True,
            alpha=lambda z, p_v: alpha(
                z, p_v, spl_Pp(z), 1, sigma_pump, sigma_a, sigma_e
            ),
            method="nlse",
        )

        dz = model.estimate_step_size()
        sim = model.simulate(z_grid=length, dz=dz, n_records=250)

        rel_error = abs((p_out.e_p - sim.pulse_out.e_p) / sim.pulse_out.e_p)
        REL_ERROR.append(rel_error)
        print(rel_error)

        p_out = sim.pulse_out

    gain_dB = 10 * np.log10(p_out.e_p / pulse.e_p)
    print(f"{gain_dB} dB gain")
    return sim, p_out, gain_dB


length = 5
start = 1e-3
stop = 100e-3
step = 1e-3
Pp = np.arange(start, stop + step, step)
AMP = []
for n, pp in enumerate(tqdm(Pp)):
    res = amplify(pp, length)
    amp = collections.namedtuple("amp", ["sim", "pulse", "g_dB"])
    amp.sim = res[0]
    amp.pulse = res[1]
    amp.g_dB = res[2]
    AMP.append(amp)

x = np.asarray([i.g_dB for i in AMP])

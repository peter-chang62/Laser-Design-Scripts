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


# calculate power z-derivative for a given inversion and absorption and
# emission cross-section. This is a general equation that can be applied to
# both signal and pump, although I only use it for the pump
def dpdz(overlap, n2_n, p, sigm_a, sigm_e, alph):
    n2 = n2_n * n_ion
    n1 = n_ion - n2
    add = overlap * sigm_e * n2 * p
    subtract = overlap * sigm_a * n1 * p + alph * p
    return add - subtract


# calculate the gain coefficient
def alpha(z, p_v, Pp, overlap, sigm_p, sigm_a, sigm_e):
    n2_n = n2_over_n(Pp, p_v, sigm_p, sigm_a, sigm_e)
    n2 = n2_n * n_ion
    n1 = n_ion - n2
    return overlap * n2 * sigm_e - overlap * n1 * sigm_a


# %% ------------- jointly solve RE and NLSE ----------------------------------
def amplify(Pp_0_f, Pp_0_b, length, error=1e-3):
    if Pp_0_f > 0:
        fwd = True
    else:
        fwd = False
    if Pp_0_b > 0:
        bck = True
    else:
        bck = False
    assert fwd or bck, "set at least either a forward or backward pump"

    # The starting pump profile is just a decaying exponential. We "shoot"
    # solutions from both ends if forward and backward pumped. It's true that
    # solving independently will lead to incorrect results where the pumps
    # overlap, but this is really just an approximation that will be made
    # better later by iterating!
    z_pump_grid = np.linspace(0, length, 1000)
    func = lambda p, z_pump_grid: dpdz(1, 0, p, sigma_pump, 0, 0)
    if fwd:
        sol_f = odeint(func, np.array([Pp_0_f]), z_pump_grid)
    else:
        sol_f = 0
    if bck:
        sol_b = odeint(func, np.array([Pp_0_b]), z_pump_grid)[::-1]
    else:
        sol_b = 0
    spl_Pp = InterpolatedUnivariateSpline(z_pump_grid, sol_f + sol_b, ext="zeros")

    # ------------- model --------------------------------------------------
    sigma_a = spl_a(pulse.v_grid)
    sigma_e = spl_e(pulse.v_grid)
    model = fiber.generate_model(
        pulse,
        t_shock="auto",
        raman_on=True,
        alpha=lambda z, p_v: alpha(z, p_v, spl_Pp(z), 1, sigma_pump, sigma_a, sigma_e),
        method="nlse",
    )

    # ------------- sim ----------------------------------------------------
    dz = model.estimate_step_size()
    sim = model.simulate(z_grid=length, dz=dz, n_records=250)
    p_out = sim.pulse_out

    # ------------- iterate! -----------------------------------------------
    REL_ERROR = []
    rel_error = 100
    while rel_error > error:
        # calculate n2_n and grid it
        n2_n = np.zeros(sim.z.size)
        for n, z in enumerate(sim.z):
            n2_n[n] = n2_over_n(spl_Pp(z), sim.p_v[n], sigma_pump, sigma_a, sigma_e)
        if fwd:
            spl_n2_n_f = InterpolatedUnivariateSpline(sim.z, n2_n, ext="const")
        if bck:
            spl_n2_n_b = InterpolatedUnivariateSpline(sim.z, n2_n[::-1], ext="const")

        # use n2_n to calculate the updated pump profile
        if fwd:
            func_f = lambda p, z: dpdz(1, spl_n2_n_f(z), p, sigma_pump, 0, 0)
            sol_f = odeint(func_f, np.array([Pp_0_f]), z_pump_grid)
        else:
            sol_f = 0
        if bck:
            func_b = lambda p, z: dpdz(1, spl_n2_n_b(z), p, sigma_pump, 0, 0)
            sol_b = odeint(func_b, np.array([Pp_0_b]), z_pump_grid)[::-1]
        else:
            sol_b = 0
        spl_Pp = InterpolatedUnivariateSpline(z_pump_grid, sol_f + sol_b, ext="zeros")

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
    return sim, p_out, gain_dB, n2_n, spl_Pp(sim.z)


# %% ------------------------- parameter sweep --------------------------------
length = 5
start = 1e-3
stop = 50e-3
step = 1e-3 / 2.0
Pp = np.arange(start, stop + step, step)
AMP = []
for n, pp in enumerate(tqdm(Pp)):
    res = amplify(pp, pp, length)
    amp = collections.namedtuple("amp", ["sim", "pulse", "g_dB", "n2_n", "Pp"])
    amp.sim = res[0]
    amp.pulse = res[1]
    amp.g_dB = res[2]
    amp.n2_n = res[3]
    amp.Pp = res[4]
    AMP.append(amp)

# %% ------------------------- look at results! -------------------------------
g_dB = np.asarray([i.g_dB for i in AMP])

fig, ax = plt.subplots(1, 1)
ax.plot(Pp * 1e3, g_dB)
ax.set_xlabel("pump power (mW)")
ax.set_ylabel("signal gain (dB)")
fig.tight_layout()

fig, ax = plt.subplots(3, 1, figsize=np.array([4.67, 8.52]))
ax[:] = ax[::-1]
idx_min = g_dB.argmin()
idx_half = abs(g_dB - g_dB.max() / 2).argmin()
idx_max = g_dB.argmax()

ax[0].plot(AMP[idx_min].sim.z, AMP[idx_min].n2_n)
ax[0].set_ylabel("$\\mathrm{n_2/n_1}$")
ax_2 = ax[0].twinx()
ax_2.plot(AMP[idx_min].sim.z, AMP[idx_min].Pp * 1e3, "C1")
ax_2.set_ylabel("pump power (mW)")
ax[0].set_xlabel("position (m)")
ax[0].set_ylim(ymax=1)

ax[1].plot(AMP[idx_half].sim.z, AMP[idx_half].n2_n)
ax[1].set_ylabel("$\\mathrm{n_2/n_1}$")
ax_2 = ax[1].twinx()
ax_2.plot(AMP[idx_half].sim.z, AMP[idx_half].Pp * 1e3, "C1")
ax_2.set_ylabel("pump power (mW)")
ax[1].set_xlabel("position (m)")
ax[1].set_ylim(ymax=1)

ax[2].plot(AMP[idx_max].sim.z, AMP[idx_max].n2_n)
ax[2].set_ylabel("$\\mathrm{n_2/n_1}$")
ax_2 = ax[2].twinx()
ax_2.plot(AMP[idx_max].sim.z, AMP[idx_max].Pp * 1e3, "C1")
ax_2.set_ylabel("pump power (mW)")
ax[2].set_xlabel("position (m)")
ax[2].set_ylim(ymax=1)

fig.tight_layout()

AMP[idx_min].sim.plot("wvl", num="minimum")
AMP[idx_half].sim.plot("wvl", num="half")
AMP[idx_max].sim.plot("wvl", num="max")

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


def n2_over_n(pulse, Pp, p_v, sigm_p, sigm_a, sigm_e):
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
def alpha(pulse, z, p_v, Pp, overlap, sigm_p, sigm_a, sigm_e):
    n2_n = n2_over_n(pulse, Pp, p_v, sigm_p, sigm_a, sigm_e)
    n2 = n2_n * n_ion
    n1 = n_ion - n2
    return overlap * n2 * sigm_e - overlap * n1 * sigm_a


# %% ------------- jointly solve RE and NLSE ----------------------------------
def amplify(
    pulse_fwd,
    pulse_bck,
    fiber,
    Pp_0_f,
    Pp_0_b,
    length,
    sigma_pump,
    sigma_a,
    sigma_e,
    error=1e-3,
):
    if Pp_0_f > 0:
        fwd = True
    else:
        fwd = False
    if Pp_0_b > 0:
        bck = True
    else:
        bck = False
    assert fwd or bck, "set at least either a forward or backward pump"

    if pulse_bck is not None:
        seed_bck = True
    else:
        seed_bck = False
    assert pulse_fwd is not None, "there must be a forward propagating pulse"

    # The starting pump profile is just a decaying exponential. We "shoot"
    # solutions from both ends if forward and backward pumped.
    z_pump_grid = np.linspace(0, length, 1000)
    func = lambda p, z_pump_grid: dpdz(1, 0, p, sigma_pump, 0, 0)
    if fwd:
        sol_f = odeint(func, np.array([Pp_0_f]), z_pump_grid)
    else:
        sol_f = np.array([0])
    if bck:
        sol_b = odeint(func, np.array([Pp_0_b]), z_pump_grid)[::-1]
    else:
        sol_b = np.array([0])
    spl_Pp_fwd = InterpolatedUnivariateSpline(z_pump_grid, sol_f + sol_b, ext="zeros")
    if seed_bck:
        spl_Pp_bck = InterpolatedUnivariateSpline(
            z_pump_grid,
            sol_f[::-1] + sol_b[::-1],
            ext="zeros",
        )

    # ------------- model and sim ---------------------------------------------
    model_fwd = fiber.generate_model(
        pulse_fwd,
        t_shock="auto",
        raman_on=True,
        alpha=lambda z, p_v: alpha(
            pulse_fwd,
            z,
            p_v,
            spl_Pp_fwd(z),
            1,
            sigma_pump,
            sigma_a,
            sigma_e,
        ),
        method="nlse",
    )

    dz = model_fwd.estimate_step_size()
    sim_fwd = model_fwd.simulate(
        z_grid=length, dz=dz, n_records=int(np.round(length / dz / 2.0))
    )
    p_out_fwd = sim_fwd.pulse_out

    if seed_bck:
        model_bck = fiber.generate_model(
            pulse_bck,
            t_shock="auto",
            raman_on=True,
            alpha=lambda z, p_v: alpha(
                pulse_bck,
                z,
                p_v,
                spl_Pp_bck(z),
                1,
                sigma_pump,
                sigma_a,
                sigma_e,
            ),
            method="nlse",
        )

        dz = model_bck.estimate_step_size()
        sim_bck = model_bck.simulate(
            z_grid=length, dz=dz, n_records=int(np.round(length / dz / 2.0))
        )
        p_out_bck = sim_bck.pulse_out
    else:
        p_out_bck = None
        sim_bck = None
        gain_dB_bck = None

    # ------------- iterate! -----------------------------------------------
    REL_ERROR_FWD = []
    REL_ERROR_BCK = []
    rel_error_fwd = None
    rel_error_bck = None
    done_fwd = False
    done_bck = False
    while not done_fwd or not done_bck:
        # calculate n2_n and grid it
        n2_n = np.zeros(sim_fwd.z.size)
        p_v = sim_fwd.p_v.copy()
        if seed_bck:
            p_v += sim_bck.p_v[::-1]
        for n, z in enumerate(sim_fwd.z):
            n2_n[n] = n2_over_n(
                pulse_fwd, spl_Pp_fwd(z), p_v[n], sigma_pump, sigma_a, sigma_e
            )

        # use n2_n to calculate the updated pump profile
        if fwd:
            spl_n2_n_f = InterpolatedUnivariateSpline(sim_fwd.z, n2_n, ext="const")
            func_f = lambda p, z: dpdz(1, spl_n2_n_f(z), p, sigma_pump, 0, 0)
            sol_f = odeint(func_f, np.array([Pp_0_f]), z_pump_grid)
        else:
            sol_f = np.array([0])
        if bck:
            spl_n2_n_b = InterpolatedUnivariateSpline(
                sim_fwd.z, n2_n[::-1], ext="const"
            )
            func_b = lambda p, z: dpdz(1, spl_n2_n_b(z), p, sigma_pump, 0, 0)
            sol_b = odeint(func_b, np.array([Pp_0_b]), z_pump_grid)[::-1]
        else:
            sol_b = np.array([0])
        spl_Pp_fwd = InterpolatedUnivariateSpline(
            z_pump_grid,
            sol_f + sol_b,
            ext="zeros",
        )
        if seed_bck:
            spl_Pp_bck = InterpolatedUnivariateSpline(
                z_pump_grid,
                sol_f[::-1] + sol_b[::-1],
                ext="zeros",
            )

        # use the updated pump profile to re-propagate the pulse
        model_fwd = fiber.generate_model(
            pulse_fwd,
            t_shock="auto",
            raman_on=True,
            alpha=lambda z, p_v: alpha(
                pulse_fwd,
                z,
                p_v,
                spl_Pp_fwd(z),
                1,
                sigma_pump,
                sigma_a,
                sigma_e,
            ),
            method="nlse",
        )

        dz = model_fwd.estimate_step_size()
        sim_fwd = model_fwd.simulate(
            z_grid=length, dz=dz, n_records=int(np.round(length / dz / 2.0))
        )

        if seed_bck:
            model_bck = fiber.generate_model(
                pulse_bck,
                t_shock="auto",
                raman_on=True,
                alpha=lambda z, p_v: alpha(
                    pulse_bck,
                    z,
                    p_v,
                    spl_Pp_bck(z),
                    1,
                    sigma_pump,
                    sigma_a,
                    sigma_e,
                ),
                method="nlse",
            )

            dz = model_bck.estimate_step_size()
            sim_bck = model_bck.simulate(
                z_grid=length, dz=dz, n_records=int(np.round(length / dz / 2.0))
            )

        rel_error_fwd = abs(
            (p_out_fwd.e_p - sim_fwd.pulse_out.e_p) / sim_fwd.pulse_out.e_p
        )
        REL_ERROR_FWD.append(rel_error_fwd)
        p_out_fwd = sim_fwd.pulse_out
        gain_dB_fwd = 10 * np.log10(p_out_fwd.e_p / pulse_fwd.e_p)

        if seed_bck:
            rel_error_bck = abs(
                (p_out_bck.e_p - sim_bck.pulse_out.e_p) / sim_bck.pulse_out.e_p
            )
            REL_ERROR_BCK.append(rel_error_bck)
            p_out_bck = sim_bck.pulse_out
            gain_dB_bck = 10 * np.log10(p_out_bck.e_p / pulse_bck.e_p)

        print(rel_error_fwd, rel_error_bck)

        if rel_error_fwd < error:
            done_fwd = True
        if not seed_bck:
            done_bck = True
        else:
            if rel_error_bck < error:
                done_bck = True

    print(f"{gain_dB_fwd} dB forward gain")
    if seed_bck:
        print(f"{gain_dB_bck} dB backward gain")

    amp = collections.namedtuple(
        "amp",
        [
            "sim_fwd",
            "sim_bck",
            "pulse_fwd",
            "pulse_bck",
            "g_dB_fwd",
            "g_dB_bck",
            "n2_n",
            "Pp",
        ],
    )
    amp.sim_fwd = sim_fwd
    amp.pulse_fwd = p_out_fwd
    amp.sim_bck = sim_bck
    amp.pulse_bck = p_out_bck
    amp.g_dB_fwd = gain_dB_fwd
    amp.g_dB_bck = gain_dB_bck
    amp.n2_n = n2_n
    amp.Pp = spl_Pp_fwd(sim_fwd.z)
    return amp


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

# %% ------------- pulse ------------------------------------------------------
n = 256
v_min = c / 2000e-9
v_max = c / 1000e-9
v0 = c / 1550e-9
e_p = 10e-12
t_fwhm_short = 250e-15
t_fwhm_long = 2e-12
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
fiber.gamma = 4 / (W * km)

# %% ------------- edfa -------------------------------------------------------
amp = amplify(
    pulse_short,
    pulse_short,
    fiber,
    0,
    50e-3,
    5,
    sigma_pump,
    spl_a(pulse_short.v_grid),
    spl_e(pulse_short.v_grid),
    error=1e-3,
)

# %% --------- look at results! -----------------------------------------------
amp.sim_fwd.plot("wvl", num="forward")
amp.sim_bck.plot("wvl", num="backward")

fig, ax = plt.subplots(1, 1)
ax.plot(amp.sim_fwd.z, amp.Pp)
ax_2 = ax.twinx()
ax_2.plot(amp.sim_fwd.z, amp.n2_n, "C1")
ax_2.set_ylim(ymax=1)

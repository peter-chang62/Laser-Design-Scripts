# %% ----- imports
import numpy as np
import matplotlib.pyplot as plt
import pynlo
import clipboard
import pandas as pd
from scipy.constants import c
from re_nlse_joint_5level import EDF
import edfa
import collections
from scipy.interpolate import InterpolatedUnivariateSpline
import blit

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
v_min = c / 1750e-9
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
dv_dl = pulse.v_grid**2 / c

# %% --------- passive fibers -------------------------------------------------
gamma_pm1550 = 1.2
gamma_edf = 6.5

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

# ----- target round trip dispersion in the loop
# D_l = 2
# l_p_s = 0.15  # shortest straight section I can do
# l_g = (D_l - D_p) * (l_t - 2 * l_p_s) / (D_g - D_p)
# l_p_l = (D_g - D_l) * (l_t - 2 * l_p_s) / (D_g - D_p)

# ----- target total round trip dispersion: D_l -> D_rt
D_rt = 4
l_p_s = 0.11  # length of straight section
l_g = -l_t * (D_p - D_rt) / (D_g - D_p)
l_p = l_t - l_g  # passive fiber length
l_p_l = l_p - l_p_s * 2  # passive fiber in loop

assert np.all(np.array([l_g, l_p_s, l_p_l]) >= 0)

p_gf = pulse.copy()  # gain first
p_pf = pulse.copy()  # passive first
p_s = pulse.copy()  # straight section
p_out = pulse.copy()

# parameters
Pp = 400 * 1e-3
phi = np.pi / 2
loss = 10 ** -(0.7 / 10)

# set up plot
fig, ax = plt.subplots(2, 2, num=f"{D_rt} ps/nm/km")
ax[0, 0].set_xlabel("wavelength (nm)")
ax[1, 0].set_xlabel("wavelength (nm)")
ax[0, 1].set_xlabel("time (ps)")
ax[1, 1].set_xlabel("time (ps)")

loop_count = 0
include_loss = True
done = False
tol = 1e-3
while not done:
    # ------------- start at splitter --------------------------
    p_gf.a_t[:] = p_s.a_t[:] / 2  # straight / 2
    p_pf.a_t[:] = p_s.a_t[:] / 2  # straight / 2

    # ------------- gain fiber first --------------------------
    # gain section
    if include_loss:
        p_gf.p_v[:] *= loss  # splice from splitter to gain

    # ------------- passive fiber first --------------------------
    # passive fiber
    p_pf.a_t[:] = propagate(pm1550, p_pf, l_p_l).sim.pulse_out.a_t[:]

    if include_loss:
        p_pf.p_v[:] *= loss  # phase bias insertion loss
        p_pf.p_v[:] *= loss  # splice from phase bias to gain

    # ----------- gain section ---------------------------------
    model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
        p_fwd=p_gf,
        p_bck=p_pf,
        edf=edf,
        length=l_g,
        Pp_fwd=Pp,
        Pp_bck=0.0,
        n_records=n_records,
    )
    p_gf.a_t[:] = sim_fwd.pulse_out.a_t[:]
    p_pf.a_t[:] = sim_bck.pulse_out.a_t[:]

    if include_loss:
        p_gf.p_v[:] *= loss  # splice from gain to phase bias
        p_gf.p_v[:] *= loss  # phase bias insertion loss

    # passive fiber
    p_gf.a_t[:] = propagate(pm1550, p_gf, l_p_l).sim.pulse_out.a_t[:]

    if include_loss:
        p_pf.p_v[:] *= loss  # splice from gain to splitter

    # ------------- back to splitter --------------------------
    p_s.a_t[:] = p_gf.a_t[:] * np.exp(1j * phi) + p_pf.a_t[:]
    p_out.a_t[:] = p_gf.a_t[:] * np.exp(1j * phi) - p_pf.a_t[:]

    oc_percent = np.round(p_out.e_p / (p_s.e_p + p_out.e_p), 4)

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
        (l1,) = ax[0, 0].plot(
            p_out.wl_grid * 1e9,
            p_out.p_v / p_out.p_v.max() * dv_dl,
            animated=True,
        )
        (l2,) = ax[0, 1].plot(
            p_out.t_grid * 1e12,
            p_out.p_t / p_out.p_t.max(),
            animated=True,
        )
        (l3,) = ax[1, 0].plot(
            p_s.wl_grid * 1e9,
            p_s.p_v / p_s.p_v.max() * dv_dl,
            animated=True,
        )
        (l4,) = ax[1, 1].plot(
            p_s.t_grid * 1e12,
            p_s.p_t / p_s.p_t.max(),
            animated=True,
        )
        fr_number = ax[0, 0].annotate(
            "0",
            (0, 1),
            xycoords="axes fraction",
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            animated=True,
        )
        fig.tight_layout()
        bm = blit.BlitManager(fig.canvas, [l1, l2, l3, l4, fr_number])
        bm.update()
    else:
        l1.set_ydata(p_out.p_v / p_out.p_v.max() * dv_dl)
        l2.set_ydata(p_out.p_t / p_out.p_t.max())
        l3.set_ydata(p_s.p_v / p_s.p_v.max() * dv_dl)
        l4.set_ydata(p_s.p_t / p_s.p_t.max())
        fr_number.set_text(f"loop #: {loop_count}")
        bm.update()

    if loop_count == 500:
        done = True

    if loop_count == 0:
        p_v_old = p_out.p_v / p_out.p_v.max()
        error = None
    elif loop_count > 50:
        p_v_new = p_out.p_v / p_out.p_v.max()
        error = np.mean((p_v_new - p_v_old) ** 2) ** 0.5
        if error < tol:
            done = True
        else:
            p_v_old = p_v_new

    loop_count += 1
    print(
        loop_count,
        np.round(p_out.e_p * f_r * 1e3, 4) * loss**2,
        oc_percent,
        error,
    )

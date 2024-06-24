"""
First attempt at simulating additive pulse mode-locking in a figure of 8 laser cavity
"""

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
from numpy.linalg import inv

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

output = collections.namedtuple("output", ["model", "sim"])
n_records = 100
loss = 10 ** -(0.7 / 10)


def propagate(fiber, pulse, length):
    fiber: pynlo.materials.SilicaFiber
    model = fiber.generate_model(pulse, t_shock=None, raman_on=False)
    dz = model.estimate_step_size()
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


# rotation matrix
rot = lambda theta: np.array(
    [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
)

# quarter wave plate jones matrix
M_qwp = np.exp(1j * np.pi / 4) * np.array(
    [
        [1, 0],
        [0, -1j],
    ]
)

# half wave plate jones matrix
M_hwp = np.array(
    [
        [1, 0],
        [0, -1],
    ]
)

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
# frame_normal = pd.read_excel(
#     "NLight_provided/nLIGHT Er80-4_125-HD-PM simulated fiber dispersion.xlsx"
# )
frame_normal = pd.read_excel(
    "NLight_provided/nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx"
)
gvd_n = frame_normal.to_numpy()[:, :2][1:].astype(float)

wl = gvd_n[:, 0] * 1e-9
omega = 2 * np.pi * c / wl
omega0 = 2 * np.pi * c / 1560e-9
polyfit = np.polyfit(omega - omega0, gvd_n[:, 1], deg=3)
polyfit = polyfit[::-1]  # lowest order first

# %% ------------- pulse ------------------------------------------------------
f_r = 100e6
n = 256
v_min = c / 1750e-9
v_max = c / 1400e-9
v0 = c / 1560e-9
e_p = 1e-3 / f_r  # 1 mW

t_fwhm = 2e-12  # 2 ps pulse
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


# %% ----- NALM cavity fiber lengths ------------------------------------------
beta2_g = polyfit[0]
D_g = -2 * np.pi * c / 1560e-9**2 * beta2_g / ps * nm * km
D_p = 18

# total fiber length to hit rep-rate, accounting for free space section in the
# linear arm
l_free_space = 0.00
l_t = (c - 2 * f_r * l_free_space) / (f_r * 1.5)

# want l_g_1, l_g_2, and l_p of the nalm
D_rt = 2.0
l_p_s = 0.25  # length of straight section
l_g = -l_t * (D_p - D_rt) / (D_g - D_p)
l_p = l_t - l_g  # passive fiber length
l_p_l = l_p - l_p_s * 2  # passive fiber in loop

assert np.all(np.array([l_g, l_p]) >= 0)
print(f"{np.round(l_g,3)} gain, {np.round(l_p, 3)} passive")

# %% --------- passive fibers -------------------------------------------------
gamma_pm1550 = 1.2
gamma_edf = 6.5

pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550)
pm1550.gamma = gamma_pm1550 / (W * km)

# %% ------------ active fiber ------------------------------------------------
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


# %% ----- simulate! ----------------------------------------------------------
# start with circular / elliptical polarization
qwp_theta = (45 + 45 / 2) * np.pi / 180
hwp_theta = (45) * np.pi / 180 / 2  # factor of 2 for half wave plate

# angles we can't change
cross_spl_theta = 90 * np.pi / 180 / 2  # factor of 2 for half wave plate

qwp = rot(qwp_theta) @ M_qwp @ inv(rot(qwp_theta))
hwp = rot(hwp_theta) @ M_hwp @ inv(rot(hwp_theta))
cross = rot(cross_spl_theta) @ M_hwp @ inv(rot(cross_spl_theta))
qwp_inv = inv(qwp)
hwp_inv = inv(hwp)

comp = 1  # output port of PBS

Pp = 50 * 1e-3

loss_package = 10 ** -(1 / 10)  # give some leeway
loss_splice = 10 ** -(0.7 / 10)
loss_pbc = 10 ** -(0.7 / 10)
loss_wdm = 10 ** -(0.7 / 10)

# set up plot
fig, ax = plt.subplots(3, 2)
ax[0, 0].set_xlabel("wavelength (nm)")
ax[1, 0].set_xlabel("wavelength (nm)")
ax[2, 0].set_xlabel("wavelength (nm)")
ax[0, 1].set_xlabel("time (ps)")
ax[1, 1].set_xlabel("time (ps)")
ax[2, 1].set_xlabel("time (ps)")

ax[0, 0].set_title("CW")
ax[0, 1].set_title("CW")
ax[1, 0].set_title("CCW")
ax[1, 1].set_title("CCW")
ax[2, 0].set_title("out")
ax[2, 1].set_title("out")

loop_count = 0
done = False
tol = 1e-3

p_fw = pulse.copy()
p_bw = pulse.copy()
p_out = pulse.copy()
X = np.zeros((2, pulse.n), dtype=complex)
while not done:
    # ------------------ nalm -------------------------------------------------
    # forward pulse goes through passive fiber before gain
    p_fw.a_t[:] = propagate(pm1550, p_fw, l_p_l).sim.pulse_out.a_t[:]

    # wdm insertion loss
    p_fw.e_p *= loss_wdm

    # splice from passive to gain
    p_bw.e_p *= loss_splice
    p_fw.e_p *= loss_splice

    # gain section
    model_fwd, sim_fwd, model_bck, sim_bck = edfa.amplify(
        p_fwd=p_fw,
        p_bck=p_bw,
        edf=edf,
        length=l_g,
        Pp_fwd=Pp,
        Pp_bck=0.0,
        t_shock=None,
        raman_on=False,
        n_records=n_records,
        tolerance=1e-3,
    )
    p_fw.a_t[:] = sim_fwd.pulse_out.a_t[:]
    p_bw.a_t[:] = sim_bck.pulse_out.a_t[:]

    # splice from gain to passive
    p_bw.e_p *= loss_splice
    p_fw.e_p *= loss_splice

    # wdm insertion loss
    p_bw.e_p *= loss_wdm

    # backward pulse goes through passive fiber after gain
    p_bw.a_t[:] = propagate(pm1550, p_bw, l_p_l).sim.pulse_out.a_t[:]

    # PBC insertion loss
    p_bw.e_p *= loss_pbc
    p_fw.e_p *= loss_pbc

    # ---------- straight section ---------------------------------------------
    p_bw.a_t[:] = propagate(pm1550, p_bw, l_p_s).sim.pulse_out.a_t[:]
    p_fw.a_t[:] = propagate(pm1550, p_fw, l_p_s).sim.pulse_out.a_t[:]

    # ---------- free space package -------------------------------------------
    X[0] = p_fw.a_t[:]
    X[1] = p_bw.a_t[:]

    X = cross @ X  # cross splice

    X = qwp_inv @ X  # back through quarter wave plate
    X = hwp_inv @ X  # back through half wave plate

    # PBS
    p_out.a_t[:] = X[comp]
    X[comp] = 0

    X = hwp @ X  # through half wave plate
    X = qwp @ X  # through quarter wave plate

    # ---------- setup next round trip ----------------------------------------
    p_fw.a_t[:] = X[0]
    p_bw.a_t[:] = X[1]

    p_fw.p_t = np.roll(p_fw.p_t, p_fw.n // 2 - p_fw.p_t.argmax())
    p_bw.p_t = np.roll(p_bw.p_t, p_bw.n // 2 - p_bw.p_t.argmax())

    oc_ratio = p_out.e_p / (p_fw.e_p + p_bw.e_p + p_out.e_p)

    # free space package insertion loss
    p_bw.e_p *= loss_package
    p_fw.e_p *= loss_package

    # ---------- straight section ---------------------------------------------
    p_bw.a_t[:] = propagate(pm1550, p_bw, l_p_s).sim.pulse_out.a_t[:]
    p_fw.a_t[:] = propagate(pm1550, p_fw, l_p_s).sim.pulse_out.a_t[:]

    # PBC insertion loss
    p_bw.e_p *= loss_pbc
    p_fw.e_p *= loss_pbc

    # ---------------- plotting -----------------------------------------------
    if loop_count == 0:
        (l1,) = ax[0, 0].plot(
            p_fw.wl_grid * 1e9,
            p_fw.p_v / p_fw.p_v.max() * dv_dl / dv_dl.max(),
            animated=True,
        )
        (l2,) = ax[0, 1].plot(
            p_fw.t_grid * 1e12,
            p_fw.p_t / p_fw.p_t.max(),
            animated=True,
        )

        (l3,) = ax[1, 0].plot(
            p_bw.wl_grid * 1e9,
            p_bw.p_v / p_bw.p_v.max() * dv_dl / dv_dl.max(),
            animated=True,
        )
        (l4,) = ax[1, 1].plot(
            p_bw.t_grid * 1e12,
            p_bw.p_t / p_bw.p_t.max(),
            animated=True,
        )

        (l5,) = ax[2, 0].plot(
            p_bw.wl_grid * 1e9,
            p_bw.p_v / p_bw.p_v.max() * dv_dl / dv_dl.max(),
            animated=True,
        )
        (l6,) = ax[2, 1].plot(
            p_bw.t_grid * 1e12,
            p_bw.p_t / p_bw.p_t.max(),
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
        bm = blit.BlitManager(fig.canvas, [l1, l2, l3, l4, l5, l6, fr_number])
        bm.update()

    else:
        l1.set_ydata(p_fw.p_v / p_fw.p_v.max() * dv_dl / dv_dl.max())
        l2.set_ydata(p_fw.p_t / p_fw.p_t.max())
        l3.set_ydata(p_bw.p_v / p_bw.p_v.max() * dv_dl / dv_dl.max())
        l4.set_ydata(p_bw.p_t / p_bw.p_t.max())
        l5.set_ydata(p_out.p_v / p_out.p_v.max() * dv_dl / dv_dl.max())
        l6.set_ydata(p_out.p_t / p_out.p_t.max())
        fr_number.set_text(f"loop #: {loop_count}")
        bm.update()

    if loop_count == 300:
        done = True

    loop_count += 1
    print(loop_count, np.round(oc_ratio, 3), np.round(p_out.e_p * f_r * 1e3, 3))

    # save figure if you want to
    file = str(loop_count)
    file = "0" * (3 - len(file)) + file
    file = "fig/" + file + ".png"
    plt.savefig(file, transparent=True, dpi=300)

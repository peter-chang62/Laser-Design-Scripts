# %% ----- imports
from scipy.constants import c
import pandas as pd
import clipboard
from re_nlse_joint_5level import EDF
import pynlo
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import time

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

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
omega0 = 2 * np.pi * c / 1560e-9
polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
polyfit = polyfit[::-1]  # lowest order first

# %% ------------- pulse ------------------------------------------------------
loss_ins = 10 ** (-0.7 / 10)
loss_spl = 10 ** (-0.2 / 10)
f_r = 1e9
e_p = 25e-3 / f_r * loss_ins * loss_spl

n = 256
v_min = c / 1700e-9
v_max = c / 1400e-9
v0 = c / 1550e-9
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
edf.set_beta_from_beta_n(v0, polyfit)  # only gdd
gamma_edf = 0
edf.gamma = gamma_edf / (W * km)

# %% ----- forward pumped base case
model_fwd = edf.generate_model(pulse, Pp_fwd=2.0)
sim_fwd = model_fwd.simulate(2.0, n_records=100)

# %% ----- solve via iteration ------------------------------------------------
Pp = 2.0 * loss_ins * loss_spl
length = 2.0

direction = 1

p_1 = pulse.copy()
p_2 = pulse.copy()
p_2.p_t[:] = 0

done = False
loop_count = 0
sum_a_prev = lambda z: 0
sum_e_prev = lambda z: 0
Pp_prev = lambda z: 0
while not done:
    model_1 = edf.generate_model(
        p_1,
        Pp_fwd=0,
        sum_a_prev=sum_a_prev,
        sum_e_prev=sum_e_prev,
        Pp_prev=Pp_prev,
    )
    sim_1 = model_1.simulate(length, n_records=100)

    sum_a_prev = InterpolatedUnivariateSpline(
        model_1.z_record,
        model_1.sum_a_record[::direction],
    )
    sum_e_prev = InterpolatedUnivariateSpline(
        model_1.z_record,
        model_1.sum_e_record[::direction],
    )
    Pp_prev = InterpolatedUnivariateSpline(
        model_1.z_record,
        model_1.Pp_record[::direction],
    )

    model_2 = edf.generate_model(
        p_2,
        Pp_fwd=Pp,
        sum_a_prev=sum_a_prev,
        sum_e_prev=sum_e_prev,
        Pp_prev=Pp_prev,
    )

    rk45 = model_2.mode.rk45_Pp
    t = [rk45.t]
    y = [rk45.y[0]]
    while rk45.t < length:
        model_2.mode.z = rk45.t  # update z dependent parameters
        rk45.step()
        t.append(rk45.t)
        y.append(rk45.y[0])
    t = np.asarray(t)
    y = np.asarray(y[::direction])

    sum_a_prev = lambda z: 0
    sum_e_prev = lambda z: 0
    Pp_prev = InterpolatedUnivariateSpline(t, y)

    if loop_count == 0:
        e_p_old = sim_1.pulse_out.e_p
        loop_count += 1
        continue

    e_p_new = sim_1.pulse_out.e_p
    error = abs(e_p_new - e_p_old) / e_p_new
    e_p_old = e_p_new

    if error < 1e-3:
        done = True
    print(loop_count, error)

    loop_count += 1

# %% ------------------- look at results --------------------------------------
sim = sim_fwd
sol_Pp = sim.Pp
sol_Ps = np.sum(sim.p_v * pulse.dv * f_r, axis=1) * loss_ins * loss_spl
z = sim.z
n1 = sim.n1_n
n2 = sim.n2_n
n3 = sim.n3_n
n4 = sim.n4_n
n5 = sim.n5_n

fig = plt.figure(num="forward pump", figsize=np.array([11.16, 5.21]))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
(line_11,) = ax1.plot(z, sol_Pp, label="pump")
(line_12,) = ax1.plot(z, sol_Ps, label="signal")
ax1.grid()
ax1.legend(loc="best")
ax1.set_xlabel("position (m)")
ax1.set_ylabel("power (W)")

(line_21,) = ax2.plot(z, n1, label="n1")
(line_22,) = ax2.plot(z, n2, label="n2")
(line_23,) = ax2.plot(z, n3, label="n3")
(line_24,) = ax2.plot(z, n4, label="n4")
(line_25,) = ax2.plot(z, n5, label="n5")
ax2.grid()
ax2.legend(loc="best")
ax2.set_xlabel("position (m)")
ax2.set_ylabel("population inversion")

fig.tight_layout()

sim = sim_1
sol_Pp = Pp_prev(sim_1.z)
sol_Ps = np.sum(sim.p_v * pulse.dv * f_r, axis=1)
z = sim.z
n1 = sim.n1_n
n2 = sim.n2_n
n3 = sim.n3_n
n4 = sim.n4_n
n5 = sim.n5_n

fig = plt.figure(num="iterative backward", figsize=np.array([11.16, 5.21]))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
(line_11,) = ax1.plot(z, sol_Pp, label="pump")
(line_12,) = ax1.plot(z, sol_Ps, label="signal")
ax1.grid()
ax1.legend(loc="best")
ax1.set_xlabel("position (m)")
ax1.set_ylabel("power (W)")

(line_21,) = ax2.plot(z, n1, label="n1")
(line_22,) = ax2.plot(z, n2, label="n2")
(line_23,) = ax2.plot(z, n3, label="n3")
(line_24,) = ax2.plot(z, n4, label="n4")
(line_25,) = ax2.plot(z, n5, label="n5")
ax2.grid()
ax2.legend(loc="best")
ax2.set_xlabel("position (m)")
ax2.set_ylabel("population inversion")

fig.tight_layout()

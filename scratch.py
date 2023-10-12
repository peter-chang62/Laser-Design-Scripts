# what is the pump saturation power at 980 nm?
# what is the correct absorption coefficient

# what is the saturation power for the signal wavelength?

# %% ----
import numpy as np
import clipboard
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.constants import c, h
import pynlo

# %% --------------------------------------------------------------------------
# def func(p, z, alpha, psat):
#     return -alpha * p / (1 + p / psat)


# z = np.linspace(0, 1.5, 1000)
# p_0 = 1
# alpha = 15
# psat = p_0 / 20
# sol = np.squeeze(
#     odeint(
#         func,
#         np.array([p_0]),
#         z,
#         args=(alpha, psat),
#     )
# )

# # p_thresh = psat / 1.3444263903366238
# fig, ax = plt.subplots(1, 1)
# ax.plot(z, sol)
# ax.grid()
# fig.tight_layout()

# %% ------------------------ moved from re_nlse_joint_try1.py ----------------
# test
# Pp = 1
# p_v = pulse.p_v
# E_P = np.linspace(50e-12, 30e-9, 1000)
# N2_N = np.zeros(E_P.size)
# for n, e_p in enumerate(E_P):
#     pulse.e_p = e_p
#     N2_N[n] = n2_n(Pp, pulse.p_v)

# fig, ax = plt.subplots(1, 1)
# ax.plot(E_P * 100e6, N2_N)

def dpdz(overlap, n2_n, p, sigm_a, sigm_e, alpha):
    n2 = n2_n * n_ion
    n1 = n_ion - n2
    return overlap * sigm_e * n2 * p - overlap * sigm_a * n1 * p - alpha * p

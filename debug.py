import blit
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import pynlo
import clipboard
import time


path = (
    r"/Users/peterchang/Library/CloudStorage/OneDrive-UCB-O365/"
    + "sim_output/200_MHz_ER_110_ER_80_35cm_350mW_pump/"
)
# path = (
#     r"/Users/peterchang/Library/CloudStorage/OneDrive-UCB-O365/"
#     + "sim_output/200_MHz_ER_110_ER_80_40cm_400mW_pump/"
# )
# path = (
#     r"/Users/peterchang/Library/CloudStorage/OneDrive-UCB-O365/"
#     + "sim_output/200_MHz_ER_110_500_mW_pump/"
# )
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

# %% -----
fig, ax = plt.subplots(1, 1)
for i in np.arange(4.5, 11.5, 0.5):
    p_t = np.load(path + f"p_t_{i}_psnmkm_200MHz_ls_11cm.npy")
    p_v = np.load(path + f"p_v_{i}_psnmkm_200MHz_ls_11cm.npy")
    p_wl = p_v * dv_dl * f_r * 1e-9 * 1e3
    p_wl_dB = 10 * np.log10(p_wl)

    ax.plot(pulse.wl_grid * 1e9, p_wl_dB[-1], label=str(i))

ax.legend(loc="best")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("power spectral density (mW / nm)")
ax.set_ylim(-40, 5)
ax.set_xlim(1450, 1650)
fig.tight_layout()

# %% -----
D = 10.0
p_v = np.load(path + f"p_v_{float(D)}_psnmkm_200MHz_ls_11cm.npy")
p_t = np.load(path + f"p_t_{float(D)}_psnmkm_200MHz_ls_11cm.npy")
p_wl_plot = p_v * dv_dl * f_r * 1e-9 * 1e3
p_wl_plot = 10 * np.log10(p_wl_plot)

# fig, ax = plt.subplots(1, 1)
# (line,) = ax.plot(pulse.wl_grid * 1e9, p_wl_plot[0])
# ax.set_ylim(p_wl_plot.max() - 40, ymax=p_wl_plot.max() + 1)
# ax.set_xlim(1450, 1650)
# ax.set_xlabel("wavelength (nm)")
# ax.set_ylabel("power spectral density (mW / nm)")
# fr_number = ax.annotate(
#     "loop count: 0",
#     (0, 1),
#     xycoords="axes fraction",
#     xytext=(10, -10),
#     textcoords="offset points",
#     ha="left",
#     va="top",
#     animated=True,
# )
# fig.tight_layout()
# bm = blit.BlitManager(fig.canvas, [line, fr_number])
# bm.update()
# save = False
# for n, i in enumerate(p_wl_plot):
#     line.set_ydata(i)
#     fr_number.set_text(f"loop count: {n}")
#     bm.update()
#     if save:
#         plt.savefig(f"fig/{n}.png", dpi=300, transparent=True)
#     else:
#         time.sleep(0.05)

fig, ax = plt.subplots(1, 1)
p_wl_dB = p_wl_plot - p_wl_plot.max()
ax.pcolormesh(
    pulse.wl_grid * 1e9,
    np.arange(p_wl_dB.shape[0]),
    p_wl_dB,
    vmin=-40,
    vmax=0,
    cmap="CMRmap_r_t",
)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("loop count")
ax.set_title(f"{float(D)} ps/nm/km")
fig.tight_layout()

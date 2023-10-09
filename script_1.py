# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
from figure9_disp import LaserCavity
import clipboard
from scipy.constants import c


# units
ps = 1e-12
fs = 1e-15
nm = 1e-9
km = 1e3

# %% -----
# get arrays to plot
fr_ll_MHz = 150 + 15
fr_ul_MHz = 150 - 15
fr_target = 150 * 1e6
D_ll_ps_nmkm = 1.0
D_ul_ps_nmkm = 1.2
lc = LaserCavity(18, 20)
x_p, x_g, fr, D, mask = lc.params(
    fr_ll_MHz,
    fr_ul_MHz,
    D_ll_ps_nmkm,
    D_ul_ps_nmkm,
)

# %% ----- plotting!
# mask arrays to show allowed values for round trip dispersion
fr_masked = fr.copy()
fr_masked[mask] = np.nan

D_masked = D.copy()
D_masked[mask] = np.nan

# get mask for 200 MHz region
target_fr_window = np.zeros(fr.shape)
width = 0.5e6
mask_2 = (abs(fr - fr_target) < width).nonzero()
not_mask_2 = (abs(fr - fr_target) >= width).nonzero()
target_fr_window[mask_2] = 1
target_fr_window[not_mask_2] = 0

# ----- D entire parameter space
# fig, ax = plt.subplots(1, 1, figsize=np.array([5.25, 3.85]))
# img = ax.pcolormesh(x_p * 1e2 / 8, x_g * 1e2, D / (ps / nm / km), cmap="jet")
# ax.set_xlabel("pigtail lengths (cm)")
# ax.set_ylabel("gain fiber length (cm)")
# plt.colorbar(img, label="$\\mathrm{D_{RT}}$ (ps/nmkm)")
# fig.suptitle(
#     f"round trip dispersion for $f_r$ b/w {fr_ll_MHz} MHz and {fr_ul_MHz } MHz"
# )
# fig.tight_layout()

# ----- D only valid parameter space
fig, ax = plt.subplots(1, 1, figsize=np.array([5.25, 3.85]))
img = ax.pcolormesh(x_p * 1e2 / 8, x_g * 1e2, D_masked / (ps / nm / km), cmap="jet")
img_ = ax.pcolormesh(x_p * 1e2 / 8, x_g * 1e2, target_fr_window * 1e-6, cmap="binary_t")
ax.set_xlabel("pigtail lengths (cm)")
ax.set_ylabel("gain fiber length (cm)")
plt.colorbar(img, label="$\\mathrm{D_{RT}}$ (ps/nmkm)")
fig.suptitle(
    f"round trip dispersion for $f_r$ b/w {fr_ll_MHz} MHz and {fr_ul_MHz } MHz"
)
fig.tight_layout()

# ----- fr entire parameter space
# fig, ax = plt.subplots(1, 1, figsize=np.array([5.25, 3.85]))
# img = ax.pcolormesh(x_p * 1e2 / 8, x_g * 1e2, fr * 1e-6, cmap="jet")
# # img_ = ax.pcolormesh(x_p * 1e2 / 8, x_g * 1e2, target_fr_window * 1e-6, cmap="binary_t")
# ax.set_xlabel("pigtail lengths (cm)")
# ax.set_ylabel("gain fiber length (cm)")
# plt.colorbar(img, label="repetition rate (MHz)")
# fig.suptitle(f"valid fiber lengths for $f_r$ b/w {fr_ll_MHz} MHz and {fr_ul_MHz } MHz")
# fig.tight_layout()

# ----- fr only valid parameter space
fig, ax = plt.subplots(1, 1, figsize=np.array([5.25, 3.85]))
img = ax.pcolormesh(x_p * 1e2 / 8, x_g * 1e2, fr_masked * 1e-6, cmap="jet")
img_ = ax.pcolormesh(x_p * 1e2 / 8, x_g * 1e2, target_fr_window * 1e-6, cmap="binary_t")
ax.set_xlabel("pigtail lengths (cm)")
ax.set_ylabel("gain fiber length (cm)")
plt.colorbar(img, label="repetition rate (MHz)")
fig.suptitle(f"valid fiber lengths for $f_r$ b/w {fr_ll_MHz} MHz and {fr_ul_MHz } MHz")
fig.tight_layout()

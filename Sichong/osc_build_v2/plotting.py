import numpy as np
import matplotlib.pyplot as plt
import clipboard
from scipy.constants import c


osc_500 = np.genfromtxt("osc_500.CSV", delimiter=",", skip_header=44)
osc_600 = np.genfromtxt("osc_600.CSV", delimiter=",", skip_header=44)
sc_fb = np.genfromtxt("sc.CSV", delimiter=",", skip_header=44)
sc_b = np.genfromtxt("sc_b.CSV", delimiter=",", skip_header=44)

nu_THz = c / (osc_500[:, 0] * 1e-9) * 1e-12
wl_nm = osc_500[:, 0]

osc_500[:, 1] = 10 ** (osc_500[:, 1] / 10)
osc_600[:, 1] = 10 ** (osc_600[:, 1] / 10)
sc_fb[:, 1] = 10 ** (sc_fb[:, 1] / 10)
sc_b[:, 1] = 10 ** (sc_b[:, 1] / 10)

osc_500[:, 1] /= osc_500[:, 1].max()
osc_600[:, 1] /= osc_600[:, 1].max()
sc_fb[:, 1] /= sc_fb[:, 1].max()
sc_b[:, 1] /= sc_b[:, 1].max()

fig, ax = plt.subplots(1, 1, figsize=np.array([4.48, 3.16]))
ax.semilogy(wl_nm, osc_500[:, 1])
# ax.semilogy(wl_nm, osc_600[:, 1])
ax.set_ylim(3.5197151678059057e-06, 1.7166318361817632)
ax.set_xlabel("wavelength (nm)")
fig.tight_layout()

fig, ax = plt.subplots(1, 1, figsize=np.array([8.31, 3.32]))
ax.semilogy(wl_nm, sc_b[:, 1])
ax.set_ylim(0.0001181692680425291, 1.648907114779569)
ax.set_xlabel("wavelength (nm)")
fig.tight_layout()

fig, ax = plt.subplots(1, 1, figsize=np.array([8.31, 3.32]))
ax.semilogy(wl_nm, sc_fb[:, 1], "C1")
ax.set_ylim(0.0001181692680425291, 1.648907114779569)
ax.set_xlabel("wavelength (nm)")
fig.tight_layout()

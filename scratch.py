"""
First attempt at simulating additive pulse mode-locking in a figure of 8 laser cavity
"""

# %% ----- imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
from numpy.linalg import inv

# rotation matrix
rot = lambda theta: np.array(
    [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
)

ret = lambda phi: np.array(
    [
        [np.exp(1j * phi / 2), 0],
        [0, np.exp(-1j * phi / 2)],
    ]
)

# quarter wave plate jones matrix
M_qwp = ret(np.pi / 2)

# half wave plate jones matrix
M_hwp = ret(np.pi)

cross = rot(np.pi / 2 / 2) @ M_hwp @ inv(rot(np.pi / 2 / 2))
comp = 1


# start with circular / elliptical polarization
def freespace(X, theta_qwp, theta_hwp):
    qwp = rot(theta_qwp) @ M_qwp @ inv(rot(theta_qwp))
    hwp = rot(theta_hwp) @ M_hwp @ inv(rot(theta_hwp))

    X = cross @ X  # cross splice

    X = inv(qwp) @ X  # back through quarter wave plate
    X = inv(hwp) @ X  # back through half wave plate

    # PBS & mirror
    X[comp] = 0

    X = hwp @ X  # through half wave plate
    X = qwp @ X  # through quarter wave plate
    return X


# %% ----- fixed free space package, vary input polarization ------------------
# start out with equal projection on slow and fast and no phase delay
x = np.array([1, 1]) / np.sqrt(2)
phi = np.linspace(0, 2 * np.pi, 100)

out = np.zeros((phi.size, 2), dtype=complex)
for n in range(phi.size):
    out[n] = freespace(
        ret(phi[n]) @ x,
        theta_qwp=(45 + 45 / 2) * np.pi / 180,
        theta_hwp=(45) * np.pi / 180 / 2,
    )

t = np.sum(abs(out) ** 2, axis=1)

fig, ax = plt.subplots(1, 1)
ax.plot(phi * 180 / np.pi, t)
ax.set_xlabel("$\\mathrm{\\phi_{nl}}$")
ax.set_ylabel("transmission")
fig.tight_layout()

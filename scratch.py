# what is the pump saturation power at 980 nm?
# what is the correct absorption coefficient

# what is the saturation power for the signal wavelength?

import numpy as np
import clipboard
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def func(p, z, alpha, psat):
    return -alpha * p / (1 + p / psat)


z = np.linspace(0, 1.5, 1000)
p_0 = 1
alpha = 15
psat = p_0 / 20
sol = np.squeeze(
    odeint(
        func,
        np.array([p_0]),
        z,
        args=(alpha, psat),
    )
)

# p_thresh = psat / 1.3444263903366238
fig, ax = plt.subplots(1, 1)
ax.plot(z, sol)
ax.grid()
fig.tight_layout()

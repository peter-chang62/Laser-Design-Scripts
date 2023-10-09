import numpy as np
import matplotlib.pyplot as plt
import clipboard

fr_113_MHz = np.genfromtxt(
    "data/09-22-2023/113mhz_75cm_er80.CSV", skip_header=44, delimiter=","
)
fr_115_MHz = np.genfromtxt(
    "data/09-22-2023/115mhz_75cm_er80.CSV", skip_header=44, delimiter=","
)
fr_117_MHz = np.genfromtxt(
    "data/09-22-2023/117mhz_75cm_er80.CSV", skip_header=44, delimiter=","
)

# the peak is at ~1570 nm ...
fig, ax = plt.subplots(1, 3, figsize=np.array([13.69, 4.8]))
ax[0].plot(fr_113_MHz[:, 0], fr_113_MHz[:, 1])
ax[1].plot(fr_115_MHz[:, 0], fr_115_MHz[:, 1])
ax[2].plot(fr_117_MHz[:, 0], fr_117_MHz[:, 1])
[i.set_xlabel("wavelength (nm)") for i in ax]
ax[0].set_title("113 MHz, 75 cm gain fiber")
ax[1].set_title("115 MHz, 75 cm gain fiber")
ax[2].set_title("117 MHz, 75 cm gain fiber")
fig.tight_layout()

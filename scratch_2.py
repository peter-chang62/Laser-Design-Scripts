import numpy as np
import matplotlib.pyplot as plt
import clipboard
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.constants import c

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

frame = pd.read_excel(
    "NLight_provided/nLIGHT Er80-4_125-HD-PM simulated fiber dispersion.xlsx"
)
gvd = frame.to_numpy()[:, :2][1:].astype(float)

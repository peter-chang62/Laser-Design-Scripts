import numpy as np
import matplotlib.pyplot as plt
import clipboard
from figure9_disp import LaserCavity
from scipy.constants import c


ps = 1e-12
nm = 1e-9
km = 1e3

fr_current = 145.6e6
x_g_current = 0.635

lc = LaserCavity(18, 26)
d_current = lc.calc_d_from_fr_x_g(fr_current, x_g_current)
x_p_current = c / 1.5 / fr_current - x_g_current

x_p_target = lc.calc_x_p_from_x_g_D(0.47, .3 * ps / nm / km)
x_p_cut = x_p_current - x_p_target

print("cut off this much pm-1550:", x_p_cut * 1e2, "cm")

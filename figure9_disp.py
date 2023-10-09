"""
The following is for a laser built only using two types of fiber: gain fiber
with normal dispersion parameter |Dg|, and passive fiber with anomalous
dispersion parameter |Dp|
"""

# %% -----
__all__ = ["LaserCavity"]

# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps


# units
ps = 1e-12
fs = 1e-15
nm = 1e-9
km = 1e3

# global variables
# Dg = 20 * ps / (nm * km)  # ps / nmkm -> s / m^2

ncolors = 256
color_array = plt.get_cmap("binary")(range(ncolors))

# change alpha values
color_array[0][-1] = 0  # just send the white values to transparent!
color_array[-1][-1] = 0.5  # just send the white values to transparent!

# create a colormap object
map_object = LinearSegmentedColormap.from_list(name="binary_t", colors=color_array)

# register this new colormap with matplotlib
colormaps.register(cmap=map_object)


class LaserCavity:
    def __init__(self, Dp_ps_nmkm=18, Dg_ps_nmkm=20):
        self.Dp = Dp_ps_nmkm * ps / nm / km
        self.Dg = Dg_ps_nmkm * ps / nm / km

    def calc_x_g_from_fr_D(self, fr, Drt):
        """
        calculate the amount of gain fiber needed to hit a target round trip
        dispersion Drt at a target repetition rate

        Args:
            fr (array / float):
                target repetition rate
            Drt (array / float):
                round trip dispersion, doesn't need to be mks because the scale
                divides out
        Returns:
            array / float: gain fiber length(s)
        """
        x_t = c / 1.5 / fr
        x_p = x_t * (Drt + self.Dg) / (self.Dp + self.Dg)
        x_g = x_t - x_p
        return x_g

    def calc_x_p_from_x_g_D(self, x_g, Drt):
        """
        calculate the passive fiber length needed to hit a target round trip
        dispersion with a given gain fiber length.

        Typically you know the amount of gain fiber you have, and you would like to
        know if you have too much or too little passive fiber. So, this calculates
        the total passive fiber length given the current amount of gain fiber and
        target round trip dispersion

        Args:
            x_g (float):
                length of gain fiber
            Drt (float):
                target round trip dispersion

        Returns:
            float:
                required total length of passive fiber
        """
        term = (Drt + self.Dg) / (self.Dp + self.Dg)
        return x_g * term / (1 - term)

    def calc_d_from_fr_x_g(self, fr, x_g):
        """
        calculate the round trip dispersion given the current repetition rate and
        length of gain fiber. This is again convenient to have when splicing,
        because the quantities easier to measure are the current length of gain
        fiber, and the measured rep-rate

        Args:
            fr (float):
                repetition rate
            x_g (float):
                length of gain fiber in the loop

        Returns:
            float:
                calculated round trip dispersion
        """
        x_t = c / 1.5 / fr
        x_p = x_t - x_g
        return (x_p * self.Dp - x_g * self.Dg) / x_t

    def calc_x_g_lims_from_fr_D(self, fr_ll_MHz, fr_ul_MHz, D_ll_ps_nmkm, D_ul_ps_nmkm):
        """
        calculate the gain fiber length limits needed to map out the paramter space
        fr: ll -> ul, and Drt: ll -> ul

        Args:
            fr_ll_MHz (float):
                lowest target rep-rate
            fr_ul_MHz (float):
                highest target rep-rate
            D_ll_ps_nmkm (float):
                lowest round trip dispersion
            D_ul_ps_nmkm (float):
                highest round trip dispersion

        Returns: tuple:
            tuple of minimum and maximum gain fiber lengths allowed in this
            parameters space
        """

        # really I just have to evaluate the limits, but I'll just calculate it over
        # the grid. it would let me plot it later on a 2D plot if i wanted to, and
        # also avoids mistakes about choosing the wrong limits ...
        fr = np.linspace(fr_ll_MHz, fr_ul_MHz, 500) * 1e6
        D = np.linspace(D_ll_ps_nmkm, D_ul_ps_nmkm, 500) * ps / nm / km
        fr_2D, D_2D = np.meshgrid(fr, D)
        x_g_2D = self.calc_x_g_from_fr_D(fr_2D, D_2D)
        return x_g_2D.min(), x_g_2D.max()

    def params(self, fr_ll_MHz, fr_ul_MHz, D_ll_ps_nmkm, D_ul_ps_nmkm, plot=False):
        """
        get the parameter space for a given range of rep-rates and round trip dispersion

        Args:
            fr_ll_MHz (float):
                minimum rep-rate
            fr_ul_MHz (float):
                maximum rep-rate
            D_ll_ps_nmkm (float):
                minimum round trip dispersion
            D_ul_ps_nmkm (float):
                maximum round trip dispersion
        """

        # calculate the limits of gain fiber lengths in play
        x_g_ll, x_g_ul = self.calc_x_g_lims_from_fr_D(
            fr_ll_MHz,
            fr_ul_MHz,
            D_ll_ps_nmkm,
            D_ul_ps_nmkm,
        )

        # generate grid of fiber lengths from round trip lengths and gain fiber
        # lengths
        x_t_ll, x_t_ul = c / 1.5 / fr_ul_MHz / 1e6, c / 1.5 / fr_ll_MHz / 1e6
        x_t = np.linspace(x_t_ll, x_t_ul, 500)
        x_g = np.linspace(x_g_ll, x_g_ul, 500)
        x_p = x_t - x_g

        # calculate rep-rates over this 2D grid
        x_p_2D, x_g_2D = np.meshgrid(x_p, x_g)
        x_t_2D = x_p_2D + x_g_2D
        fr_2D = c / 1.5 / x_t_2D

        # calculate the dispersion over this 2D grid
        D_2D = (x_p_2D * self.Dp - x_g_2D * self.Dg) / x_t_2D
        D_2D /= ps / nm / km
        idx = np.logical_or(D_2D < D_ll_ps_nmkm, D_2D > D_ul_ps_nmkm).nonzero()
        fr_2D_all = fr_2D.copy()
        fr_2D[idx] = np.nan

        if plot:
            # ----- plot the results! -----
            fig, ax = plt.subplots(1, 1)
            img = ax.pcolormesh(x_p * 1e2 / 8, x_g * 1e2, D_2D, cmap="jet")
            ax.set_xlabel("pigtail lengths (cm)")
            ax.set_ylabel("gain fiber length (cm)")
            plt.colorbar(img, label="$\\mathrm{D_{RT}}$ (ps/nmkm)")
            fig.suptitle(
                f"round trip dispersion for $f_r$ b/w {fr_ll_MHz} MHz and {fr_ul_MHz } MHz"
            )
            fig.tight_layout()

            fig, ax = plt.subplots(1, 1)
            img = ax.pcolormesh(x_p * 1e2 / 8, x_g * 1e2, fr_2D * 1e-6, cmap="jet")
            ax.set_xlabel("pigtail lengths (cm)")
            ax.set_ylabel("gain fiber length (cm)")
            plt.colorbar(img, label="repetition rate (MHz)")
            fig.suptitle(
                f"valid fiber lengths for $f_r$ b/w {fr_ll_MHz} MHz and {fr_ul_MHz } MHz"
            )
            fig.tight_layout()

        return x_p, x_g, fr_2D_all, D_2D * ps / nm / km, idx

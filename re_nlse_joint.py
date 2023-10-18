"""
let's just solve the forward pumped case first. the backward pumped case is an
iteration from using the forward pumped case anyways, as far as i understand
it anyways.
"""

# %% ----- imports
import numpy as np
from scipy.constants import c, h
import pynlo
from scipy.integrate import RK45
import collections


ps = 1e-12
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

SimulationResult = collections.namedtuple(
    "SimulationResult", ["pulse", "z", "a_t", "a_v", "Pp", "n2_n", "g_v"]
)


def package_sim_output(simulate):
    def wrapper(self, *args, **kwargs):
        (pulse_out, z, a_t, a_v, Pp, n2_n, g_v) = simulate(self, *args, **kwargs)
        model = self

        class result:
            def __init__(self):
                self.pulse_out = pulse_out.copy()
                self.z = z
                self.a_t = a_t
                self.a_v = a_v
                self.p_t = abs(a_t) ** 2
                self.p_v = abs(a_v) ** 2
                self.phi_v = np.angle(a_v)
                self.phi_t = np.angle(a_t)
                self.Pp = Pp
                self.n2_n = n2_n
                self.model = model
                self.g_v = g_v

            def animate(self, plot, save=False, p_ref=None):
                pynlo.utility.misc.animate(
                    self.pulse_out,
                    self.model,
                    self.z,
                    self.a_t,
                    self.a_v,
                    plot=plot,
                    save=save,
                    p_ref=p_ref,
                )

            def plot(self, plot, num="Simulation Results"):
                return pynlo.utility.misc.plot_results(
                    self.pulse_out,
                    self.z,
                    self.a_t,
                    self.a_v,
                    plot=plot,
                    num=num,
                )

            def save(self, path, filename):
                assert path != "" and isinstance(path, str), "give a save path"
                assert filename != "" and isinstance(filename, str)

                path = path + "/" if path[-1] != "" else path
                np.save(path + filename + "_t_grid.npy", self.pulse_out.t_grid)
                np.save(path + filename + "_v_grid.npy", self.pulse_out.v_grid)
                np.save(path + filename + "_z.npy", self.z)
                np.save(path + filename + "_amp_t.npy", abs(self.pulse_out.a_t))
                np.save(path + filename + "_amp_v.npy", abs(self.pulse_out.a_v))
                np.save(path + filename + "_phi_t.npy", np.angle(self.pulse_out.a_t))
                np.save(path + filename + "_phi_v.npy", np.angle(self.pulse_out.a_v))

        return result()

    return wrapper


def n2_over_n(
    overlap_p,
    overlap_s,
    a_eff,
    nu_p,
    Pp,
    nu_v,
    P_v,
    sigma_p,
    sigma_a,
    sigma_e,
    tau,
):
    # terms generally go as overlap * sigma * power / (h * nu * A)
    pump_term = overlap_p * sigma_p * Pp / (h * nu_p * a_eff)
    signal_num = overlap_s * sigma_a * P_v / (h * nu_v * a_eff)
    signal_denom = overlap_s * (sigma_a + sigma_e) * P_v / (h * nu_v * a_eff)

    num = signal_num + pump_term
    denom = signal_denom + pump_term + 1 / tau
    return num / denom


def _n2_over_n(
    overlap_p,
    overlap_s,
    a_eff,
    nu_p,
    Pp,
    sigma_p,
    sum_a,
    sum_e,
    tau,
):
    # terms generally go as overlap * sigma * power / (h * nu * A)
    pump_term = overlap_p * sigma_p * Pp / (h * nu_p * a_eff)
    # sum_a = sigma_a * P_v / (h * nu_v * a_eff)
    # sum_e = sigma_e * P_v / h * nu_v * a_eff
    signal_num = overlap_s * sum_a
    signal_denom = overlap_s * (sum_a + sum_e)

    num = signal_num + pump_term
    denom = signal_denom + pump_term + 1 / tau
    return num / denom


def dpdz(n2_n, n, overlap, sigma_a, sigma_e, p):
    n2 = n2_n * n
    n1 = n - n2

    # terms should go as overlap * sigma * n * P
    emission = overlap * sigma_e * n2 * p
    absorption = overlap * sigma_a * n1 * p
    return emission - absorption


def gain(n2_n, n, overlap, sigma_a, sigma_e):
    n2 = n2_n * n
    n1 = n - n2

    # terms should go as overlap * sigma * n * P
    emission = overlap * sigma_e * n2
    absorption = overlap * sigma_a * n1
    return emission - absorption


class Mode(pynlo.media.Mode):
    def __init__(
        self,
        v_grid,
        beta,
        g2=None,
        g2_inv=None,
        g3=None,
        rv_grid=None,
        r3=None,
        z=0.0,
        # --------- parameters for EDFA --------
        p_v=None,
        f_r=100e6,
        overlap_p=1.0,
        overlap_s=1.0,
        n_ion=7e24,
        a_eff=3.14e-12,
        sigma_p=None,
        sigma_a=None,
        sigma_e=None,
        tau=1e-3,
        Pp_fwd=None,
        direction=1.0,
    ):
        assert direction == 1 or direction == -1
        assert isinstance(p_v, (np.ndarray, pynlo.utility.misc.ArrayWrapper))
        assert p_v.size == v_grid.size
        assert Pp_fwd is not None, "input a pump power!"
        assert sigma_e is not None, "input absorption cross-section at 980 nm"
        assert isinstance(sigma_a, np.ndarray), "provide absorption cross-section"
        assert isinstance(sigma_e, np.ndarray), "provide emission cross-section"
        assert (
            sigma_a.size == v_grid.size
        ), "absorption cross section grid must match frequency grid"
        assert (
            sigma_e.size == v_grid.size
        ), "emission cross section grid must match frequency grid"

        self.f_r = f_r
        self.overlap_p = overlap_p
        self.overlap_s = overlap_s
        self.n_ion = n_ion
        self.a_eff = a_eff
        self.sigma_p = sigma_p
        self.sigma_a = sigma_a
        self.sigma_e = sigma_e
        self.tau = tau
        self._Pp = Pp_fwd
        self._z_start = z
        self.direction = direction

        # alpha = lambda z, p_v: self.gain
        super().__init__(v_grid, beta, self.gain, g2, g2_inv, g3, rv_grid, r3, z)
        # self.v_grid is not defined until after the __init__ call
        # __init__ sets _p_v to None, so assign this after the __init__ call
        self.p_v = p_v
        self.dv = self.v_grid[1] - self.v_grid[0]

    @property
    def nu_p(self):
        return c / 980e-9

    @property
    def n2_n(self):
        sum_a = self.f_r * self.p_v * self.sigma_a / (h * self.v_grid * self.a_eff)
        sum_e = self.f_r * self.p_v * self.sigma_e / (h * self.v_grid * self.a_eff)
        sum_a = np.sum(sum_a * self.dv)
        sum_e = np.sum(sum_e * self.dv)
        n2_n = _n2_over_n(
            self.overlap_p,
            self.overlap_s,
            self.a_eff,
            self.nu_p,
            self.Pp,
            self.sigma_p,
            sum_a,
            sum_e,
            self.tau,
        )
        return n2_n

    def gain(self, z, p_v):
        return gain(self.n2_n, self.n_ion, self.overlap_s, self.sigma_a, self.sigma_e)

    def _dPp_dz(self, z, Pp):
        return (
            dpdz(self.n2_n, self.n_ion, self.overlap_p, self.sigma_p, 0, self.Pp)
            * self.direction
        )

    def setup_rk45_Pp(self, dz):
        self._rk45_Pp = RK45(
            fun=self._dPp_dz,
            t0=self._z_start,
            y0=self.Pp,
            t_bound=np.inf,
            max_step=dz,
        )

    @property
    def rk45_Pp(self):
        assert self._rk45_Pp is not None, "setup rk45 by calling setup_rk45_Pp(dz)"
        return self._rk45_Pp

    @property
    def Pp(self):
        return self._Pp

    def update_Pp(self):
        while self.rk45_Pp.t < self.z:
            self.rk45_Pp.step()
            self._Pp = self.rk45_Pp.y


class Model_EDF(pynlo.model.Model):
    def __init__(self, pulse, mode):
        super().__init__(pulse, mode)

    def propagate(self, a_v, z, z_stop, dz, local_error, k5_v=None, cont=False):
        """
        Propagate the given pulse spectrum from `z` to `z_stop` using an
        adaptive step size algorithm.

        The step size algorithm utilizes an embedded Runge–Kutta scheme with
        orders 3 and 4 (ERK4(3)-IP) [1]_.

        Parameters
        ----------
        a_v : ndarray of complex
            The root-power spectrum of the pulse.
        z : float
            The starting point.
        z_stop : float
            The stopping point.
        dz : float
            The initial step size.
        local_error : float
            The relative local error of the adaptive step size algorithm.
        k5_v : ndarray of complex, optional
            The action of the nonlinear operator on the solution from the
            preceding step. The default is ``None``.
        cont : bool, optional
            A flag that indicates the current step is continuous with the
            previous, i.e. that it begins where the other ended. The default is
            ``False``.

        Returns
        -------
        a_v : ndarray of complex
            The root-power spectrum of the pulse.
        z : float
            The z position in the mode.
        dz : float
            The step size.
        k5_v : ndarray of complex
            The nonlinear action of the 4th-order result.
        cont : bool
            A flag indicating that the next step may be continuous.

        References
        ----------
        .. [1] S. Balac and F. Mahé, "Embedded Runge–Kutta scheme for
            step-size control in the interaction picture method," Computer
            Physics Communications, Volume 184, Issue 4, 2013, Pages 1211-1219
            https://doi.org/10.1016/j.cpc.2012.12.020

        """
        p_v = abs(a_v) ** 2
        if self._use_fftshift:
            p_v = np.fft.fftshift(p_v)
        self.mode.p_v[:] = p_v[:]
        self.mode.update_Pp()

        while z < z_stop:
            z_next = z + dz
            if z_next >= z_stop:
                final_step = True
                z_next = z_stop
                dz_adaptive = dz  # save value of last step size
                dz = z_next - z  # force smaller step size to hit z_stop
            else:
                final_step = False

            # ---- Integrate by dz
            a_RK4_v, a_RK3_v, k5_v_next = self.step(
                a_v, z, z_next, k5_v=k5_v, cont=cont
            )

            # ---- Estimate Relative Local Error
            est_error = pynlo.model.l2_error(a_RK4_v, a_RK3_v)
            error_ratio = (est_error / local_error) ** 0.25

            # ---- Propagate Solution
            if error_ratio > 2:
                # Reject this step and calculate with a smaller dz
                dz = dz / 2
                cont = False
            else:
                # Update parameters for the next loop
                z = z_next
                a_v = a_RK4_v
                k5_v = k5_v_next
                if (not final_step) or (error_ratio > 1):
                    dz = dz / max(error_ratio, 0.5)
                else:
                    dz = dz_adaptive  # if final step, use adaptive step size
                cont = True

                # update pulse energy for gain calculation
                p_v[:] = abs(a_v) ** 2
                if self._use_fftshift:
                    p_v = np.fft.fftshift(p_v)
                self.mode.p_v[:] = p_v[:]
                self.mode.update_Pp()

        return a_v, z, dz, k5_v, cont

    @package_sim_output
    def simulate(self, z_grid, dz=None, local_error=1e-6, n_records=None, plot=None):
        """
        Simulate propagation of the input pulse through the optical mode.

        Parameters
        ----------
        z_grid : float or array_like of floats
            The total propagation distance over which to simulate, or the z
            positions at which to solve for the pulse spectrum. An adaptive
            step-size algorithm is used to propagate between these points. If
            only the end point is given the starting point is assumed to be the
            origin.
        dz : float, optional
            The initial step size. If ``None``, one will be estimated.
        local_error : float, optional
            The target relative local error for the adaptive step size
            algorithm. The default is 1e-6.
        n_records : None or int, optional
            The number of simulation points to return. If set, the z positions
            will be linearly spaced between the first and last points of
            `z_grid`. If ``None``, the default is to return all points as
            defined in `z_grid`. The record always includes the starting and
            ending points.
        plot : None or string, optional
            A flag that activates real-time visualization of the simulation.
            The options are ``"frq"``, ``"time"``, or ``"wvl"``, corresponding
            to the frequency, time, and wavelength domains. If set, the plot is
            updated each time the simulation reaches one of the z positions
            returned at the output. If ``None``, the default is to run the
            simulation without real-time plotting.

        Returns
        -------
        pulse : :py:class:`~pynlo.light.Pulse`
            The output pulse. This object can be used as the input to another
            simulation.
        z : ndarray of float
            The z positions at which the pulse spectrum (`a_v`) and complex
            envelope (`a_t`) have been returned.
        a_t : ndarray of complex
            The root-power complex envelope of the pulse at each z position.
        a_v : ndarray of complex
            The root-power spectrum of the pulse at each z position.
        """
        # ---- Z Grid
        z_grid = np.asarray(z_grid, dtype=float)
        if z_grid.size == 1:
            # Since only the end point was given, the start point is the origin
            z_grid = np.append(0.0, z_grid)

        if n_records is None:
            n_records = z_grid.size
            z_record = z_grid
        else:
            assert n_records >= 2, "The output must include atleast 2 points."
            z_record = np.linspace(z_grid.min(), z_grid.max(), n_records)
            z_grid = np.unique(np.append(z_grid, z_record))
        z_record = {z: idx for idx, z in enumerate(z_record)}

        if self.mode.z_nonlinear.pol:  # support subclasses with poling
            # always simulate up to the edge of a poled domain
            z_grid = np.unique(np.append(z_grid, list(self.mode.g2_inv)))

        # ---- Setup
        z = z_grid[0]
        pulse_out = self.pulse.copy()

        # Frequency Domain
        a_v_record = np.empty((n_records, pulse_out.n), dtype=complex)
        a_v_record[0, :] = pulse_out.a_v

        # Time Domain
        a_t_record = np.empty((n_records, pulse_out.n), dtype=complex)
        a_t_record[0, :] = pulse_out.a_t

        # Pump power
        Pp = np.empty(n_records, dtype=float)
        Pp[0] = self.mode.Pp

        # inversion
        n2_n = np.empty(n_records, dtype=float)
        n2_n[0] = self.mode.n2_n

        # gain
        g_v = np.empty((n_records, pulse_out.n), dtype=float)
        g_v[0, :] = self.mode.gain(None, None)

        # Step Size
        if dz is None:
            dz = self.estimate_step_size(pulse_out.a_v, z, local_error)
            print("Initial Step Size:\t{:.3g}m".format(dz))

        # Plotting
        if plot is not None:
            assert plot in ["frq", "time", "wvl"], (
                "Plot choice '{:}' is unrecognized"
            ).format(plot)
            # Setup Plots
            self._setup_plots(plot, pulse_out, z)

        # ---- Propagate
        k5_v = None
        cont = False
        for z_stop in z_grid[1:]:
            # Step
            (pulse_out.a_v, z, dz, k5_v, cont) = self.propagate(
                pulse_out.a_v, z, z_stop, dz, local_error, k5_v=k5_v, cont=cont
            )

            # Record
            if z in z_record:
                idx = z_record[z]
                a_t_record[idx, :] = pulse_out.a_t
                a_v_record[idx, :] = pulse_out.a_v
                Pp[idx] = self.mode.Pp
                n2_n[idx] = self.mode.n2_n
                g_v[idx, :] = self.mode.gain(None, None)

                # Plot
                if plot is not None:
                    # Update Plots
                    self._update_plots(plot, pulse_out, z)

                    if z == z_grid[-1]:
                        # End animation with the last step
                        for artist in self._artists:
                            artist.set_animated(False)

        sim_res = SimulationResult(
            pulse=pulse_out,
            z=np.fromiter(z_record.keys(), dtype=float),
            a_t=a_t_record,
            a_v=a_v_record,
            Pp=Pp,
            n2_n=n2_n,
            g_v=g_v,
        )
        return sim_res


class NLSE(pynlo.model.NLSE):
    def __init__(self, pulse, mode):
        super().__init__(pulse, mode)

    def propagate(self, a_v, z, z_stop, dz, local_error, k5_v=None, cont=False):
        # ---- Standard FFT Order
        a_v = np.fft.ifftshift(a_v)
        self._use_fftshift = True

        # ---- Propagate
        a_v, z, dz, k5_v, cont = Model_EDF.propagate(
            self, a_v, z, z_stop, dz, local_error, k5_v=k5_v, cont=cont
        )

        # ---- Monotonic Order
        a_v = np.fft.fftshift(a_v)
        return a_v, z, dz, k5_v, cont

    def simulate(self, z_grid, dz=None, local_error=1e-6, n_records=None, plot=None):
        return Model_EDF.simulate(self, z_grid, dz, local_error, n_records, plot)


class EDFA(pynlo.materials.SilicaFiber):
    def __init__(
        self,
        f_r=100e6,
        overlap_p=1.0,
        overlap_s=1.0,
        n_ion=7e24,
        a_eff=3.14e-12,
        sigma_p=None,
        sigma_a=None,
        sigma_e=None,
        tau=1e-3,
    ):
        super().__init__()

        self.f_r = f_r
        self.overlap_p = overlap_p
        self.overlap_s = overlap_s
        self.n_ion = n_ion
        self.a_eff = a_eff
        self.sigma_p = sigma_p
        self.sigma_a = sigma_a
        self.sigma_e = sigma_e
        self.tau = tau

    def generate_model(
        self, pulse, t_shock="auto", raman_on=True, Pp_fwd=0, direction=1.0
    ):
        """
        generate pynlo.model.UPE or NLSE instance

        Args:
            pulse (object):
                instance of pynlo.light.Pulse
            t_shock (float, optional):
                time for optical shock formation, defaults to 1 / (2 pi pulse.v0)
            raman_on (bool, optional):
                whether to include raman effects, default is True
            alpha (array or callable, optional):
                default is 0, otherwise is a callable alpha(z, e_p) that returns a
                float or array, or fixed alpha.
        Returns:
            model
        """
        assert isinstance(pulse, pynlo.light.Pulse)
        pulse: pynlo.light.Pulse

        if isinstance(t_shock, str):
            assert t_shock.lower() == "auto"
            t_shock = 1 / (2 * np.pi * pulse.v0)
        else:
            assert isinstance(t_shock, float) or t_shock is None

        analytic = True
        n = pulse.n
        dt = pulse.dt

        v_grid = pulse.v_grid
        beta = self.beta(v_grid)
        g3 = self.g3(v_grid, t_shock=t_shock)
        if raman_on:
            rv_grid, raman = self.raman(n, dt, analytic=analytic)
        else:
            rv_grid = raman = None

        mode = Mode(
            v_grid,
            beta,
            g2=None,
            g2_inv=None,
            g3=g3,
            rv_grid=rv_grid,
            r3=raman,
            z=0.0,
            # --------- parameters for EDFA --------
            p_v=pulse.p_v.copy(),
            f_r=self.f_r,
            overlap_p=self.overlap_p,
            overlap_s=self.overlap_s,
            n_ion=self.n_ion,
            a_eff=self.a_eff,
            sigma_p=self.sigma_p,
            sigma_a=self.sigma_a,
            sigma_e=self.sigma_e,
            tau=self.tau,
            Pp_fwd=np.array([Pp_fwd]),
            direction=direction,
        )

        print("USING NLSE")
        model = NLSE(pulse, mode)
        dz = model.estimate_step_size()
        model.mode.setup_rk45_Pp(dz / 10)
        return model, dz

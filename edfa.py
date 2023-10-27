from scipy.constants import c
import pandas as pd
import clipboard
from re_nlse_joint_5level import EDF, eps_p, xi_p
import pynlo
import numpy as np
import collections
from scipy.interpolate import make_interp_spline, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from scipy.integrate import odeint

ns = 1e-9
ps = 1e-12
us = 1e-6
ms = 1e-3
nm = 1e-9
um = 1e-6
km = 1e3
W = 1.0

output = collections.namedtuple("output", ["model", "sim"])


def dPp_dz(
    n1,
    n3,
    P_p,
    sigma_p,
    overlap_p,
):
    return (
        (-sigma_p * n1 + sigma_p * xi_p * n3 - sigma_p * eps_p * n3) * overlap_p * P_p
    )


def propagate_amp(length, edf, pulse, Pp, p_v_prev=None, Pp_prev=None, n_records=None):
    edf: EDF

    model, dz = edf.generate_model(
        pulse,
        t_shock="auto",
        raman_on=True,
        Pp_fwd=Pp,
        Pp_bck=0,
        p_v_prev=p_v_prev,
        Pp_prev=Pp_prev,
    )
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


def amplify(length, edf, p_fwd, p_bck=None, Pp_fwd=0.0, Pp_bck=0.0, n_records=100):
    if p_bck is None:
        if Pp_bck == 0:  # forward seeding + forward pumping
            return propagate_amp(length, edf, p_fwd, Pp_fwd, n_records=n_records)

        propagate_back = False
    else:
        propagate_back = True

    # same discretization as the maximum step size that I allow in the
    # NLSE's rk4
    n_records = int(np.round(length / 1e-3))
    print(f"setting n_records to {n_records}")

    # forward propagation with no backward info
    model_fwd, sim_fwd = propagate_amp(
        length, edf, p_fwd, Pp_fwd, p_v_prev=None, Pp_prev=None, n_records=n_records
    )

    done = False
    loop_count = 0
    tol = 1e-3
    while not done:
        # solve the backward propagation using forward info
        if propagate_back:
            p_v = make_interp_spline(sim_fwd.z, sim_fwd.p_v[::-1])
            Pp = InterpolatedUnivariateSpline(sim_fwd.z, sim_fwd.Pp[::-1])

            model_bck, sim_bck = propagate_amp(
                length,
                edf,
                p_bck,
                Pp_bck,
                p_v_prev=p_v,
                Pp_prev=Pp,
                n_records=n_records,
            )

            p_v = make_interp_spline(sim_bck.z, sim_bck.p_v[::-1])
            Pp = InterpolatedUnivariateSpline(sim_bck.z, sim_bck.Pp[::-1])

        else:
            n1 = InterpolatedUnivariateSpline(sim_fwd.z, sim_fwd.n1_n[::-1] * edf.n_ion)
            n3 = InterpolatedUnivariateSpline(sim_fwd.z, sim_fwd.n3_n[::-1] * edf.n_ion)
            func = lambda P_p, z: dPp_dz(n1(z), n3(z), P_p, edf.sigma_p, edf.overlap_p)
            sol = np.squeeze(odeint(func, np.array([Pp_bck]), sim_fwd.z))
            Pp = InterpolatedUnivariateSpline(sim_fwd.z, sol[::-1])
            p_v = lambda z: 0

        # solve the forward propagation using backward info
        model_fwd, sim_fwd = propagate_amp(
            length, edf, p_fwd, Pp_fwd, p_v_prev=p_v, Pp_prev=Pp, n_records=n_records
        )

        # book keeping
        if loop_count == 0:
            e_p_fwd_old = sim_fwd.pulse_out.e_p

            if propagate_back:
                e_p_bck_old = sim_bck.pulse_out.e_p
                Pp_old = sim_fwd.Pp[-1] + sim_bck.Pp[-1]
            else:
                Pp_old = sim_fwd.Pp[-1] + sol[-1]

            loop_count += 1
            continue

        error = np.zeros(3)

        e_p_fwd_new = sim_fwd.pulse_out.e_p
        error_e_p_fwd = abs(e_p_fwd_new - e_p_fwd_old) / e_p_fwd_new
        e_p_fwd_old = e_p_fwd_new
        error[0] = error_e_p_fwd

        if propagate_back:
            e_p_bck_new = sim_bck.pulse_out.e_p
            error_e_p_bck = abs(e_p_bck_new - e_p_bck_old) / e_p_bck_new
            e_p_bck_old = e_p_bck_new
            error[1] = error_e_p_bck

            Pp_new = sim_fwd.Pp[-1] + sim_bck.Pp[-1]
        else:
            Pp_new = sim_fwd.Pp[-1] + sol[-1]
        error_Pp = abs(Pp_new - Pp_old) / Pp_new
        Pp_old = Pp_new
        error[2] = error_Pp

        if np.all(error < tol):
            done = True

        print(loop_count, error)
        loop_count += 1

    if propagate_back:
        return output(model=model_fwd, sim=sim_fwd), output(
            model=model_bck, sim=sim_bck
        )
    else:
        return output(model=model_fwd, sim=sim_fwd), sol

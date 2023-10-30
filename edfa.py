from re_nlse_joint_5level_wASE import EDF, eps_p, xi_p, Model_EDF
import numpy as np
import collections
from scipy.interpolate import InterpolatedUnivariateSpline

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
    """
    Calculate dP/dz for the pump

    Args:
        n1 (float):
            ground state population
        n3 (float):
            pump excited state population
        P_p (float):
            the current pump power
        sigma_p (float):
            pump absorption coefficient
        overlap_p (float):
            the pump mode and core doping overlap (0 -> 1)
    """
    return (
        (-sigma_p * n1 + sigma_p * xi_p * n3 - sigma_p * eps_p * n3) * overlap_p * P_p
    )


def propagate_amp(
    length,
    edf,
    pulse,
    Pp,
    sum_a_prev=None,
    sum_e_prev=None,
    Pp_prev=None,
    n_records=None,
    record_sim=False,
):
    """
    propagate pump and signal together through erbium doped fiber

    Args:
        length (float):
            length of edf
        edf (EDF instance):
            erbium doped fiber instance
        pulse (Pulse instance):
            pynlo Pulse instance
        Pp (float): input pump power
        sum_a_prev(callable, optional):
            should automatically be obtained from the model instance
            this adds an additional signal spectrum when calculating the
            population inversion, which is applicable for co-propagating signals!
            callable(z) : sum(overlap_s * p_v * dv * f_r * sigma_a/ (h * a_eff * v_grid))
        sum_e_prev(callable, optional):
            this adds an additional signal spectrum when calculating the
            population inversion, which is applicable for co-propagating signals!
            should automatically be obtained from the model instance
            callable(z) : sum(overlap_s * p_v * dv * f_r * sigma_e / (h * a_eff * v_grid))
        Pp_prev (callable, optional):
            callable(z) returns pump power recorded from the previous
            propagation. this will be added at each step to calculate the
            population inversion. This is applicable for counter propagating
            pump and signal.
        n_records (int, optional):
            number of spectra to record along the fiber during propagation

    Returns:
        output: output(model=model, sim=sim)
    """
    edf: EDF

    model, dz = edf.generate_model(
        pulse,
        t_shock="auto",
        raman_on=True,
        Pp_fwd=Pp,
        Pp_bck=0,
        sum_a_prev=sum_a_prev,
        sum_e_prev=sum_e_prev,
        Pp_prev=Pp_prev,
        record_sim=record_sim,
    )
    sim = model.simulate(length, dz=dz, n_records=n_records)
    return output(model=model, sim=sim)


def amplify(length, edf, p_fwd, p_bck=None, Pp_fwd=0.0, Pp_bck=0.0, n_records=100):
    """
    This is the most general function of edfa.py It calculates propagation of
    pumps and signals in erbium doped fiber for all cases: forward and
    backward pumping, and forward and backward seeding

    Args:
        length (float):
            length of erbium doped fiber
        edf (EDF instance):
            instance of EDF
        p_fwd (Pulse instance):
            instance of pynlo Pulse
        p_bck (Pulse, optional):
            instance of pynlo Pulse
        Pp_fwd (float, optional):
            forward input pump power
        Pp_bck (float, optional):
            backward input pump power
        n_records (int, optional):
            number of spectra to record during propagation

    Returns:
        output, output:
            for forward pumping only it returns one outupt collection for
            forward and backward seeding it returns two output collections,
            one for forward, and one for backward.
    """
    if p_bck is None:
        if Pp_bck == 0:  # forward seeding + forward pumping
            return propagate_amp(length, edf, p_fwd, Pp_fwd, n_records=n_records)
        else:
            propagate_back = False
            p_bck = p_fwd.copy()
            p_bck.p_t[:] = 0.0  # set the pulse energy to 0
    else:
        propagate_back = True

    # forward propagation with no backward info
    model_fwd, sim_fwd = propagate_amp(
        length, edf, p_fwd, Pp_fwd, n_records=n_records, record_sim=True
    )

    done = False
    loop_count = 0
    tol = 1e-3
    # z_grid = np.linspace(0, length, model_fwd.z_record.size)
    while not done:
        # get info from forward propagation
        sum_a = InterpolatedUnivariateSpline(
            model_fwd.z_record, model_fwd.sum_a_record[::-1]
        )
        sum_e = InterpolatedUnivariateSpline(
            model_fwd.z_record, model_fwd.sum_e_record[::-1]
        )
        Pp = InterpolatedUnivariateSpline(model_fwd.z_record, model_fwd.Pp_record[::-1])
        if propagate_back:
            # solve the backward propagation using forward info
            model_bck, sim_bck = propagate_amp(
                length,
                edf,
                p_bck,
                Pp_bck,
                sum_a_prev=sum_a,
                sum_e_prev=sum_e,
                Pp_prev=Pp,
                n_records=n_records,
                record_sim=True,
            )

            # save backward info for calculating forward propagation
            sum_a = InterpolatedUnivariateSpline(
                model_bck.z_record, model_bck.sum_a_record[::-1]
            )
            sum_e = InterpolatedUnivariateSpline(
                model_bck.z_record, model_bck.sum_e_record[::-1]
            )
            Pp = InterpolatedUnivariateSpline(
                model_bck.z_record, model_bck.Pp_record[::-1]
            )
        else:
            # solve the backward propagation using forward info
            edf: EDF
            model_bck, dz = edf.generate_model(
                p_bck,
                Pp_fwd=Pp_bck,
                sum_a_prev=sum_a,
                sum_e_prev=sum_e,
                Pp_prev=Pp,
            )
            model_bck: Model_EDF
            rk45 = model_bck.mode.rk45
            z = [rk45.t]
            sol = [rk45.y[0]]
            sum_a = [model_bck.mode._sum_a_no_pre]
            sum_e = [model_bck.mode._sum_e_no_pre]
            p_v_ase = [np.zeros(p_fwd.n)]
            z_record = [0]
            z_step = length / n_records
            idx = 1
            while rk45.t < length:
                rk45.step()
                z.append(rk45.t)
                sol.append(rk45.y[0])
                sum_a.append(model_bck.mode._sum_a_no_pre)
                sum_e.append(model_bck.mode._sum_e_no_pre)
                if rk45.t > z_step * idx:
                    p_v_ase.append(model_bck.mode.P_ASE)
                    z_record.append(rk45.t)
                    idx += 1

            # save backward info for calculating forward propagation
            sum_a = InterpolatedUnivariateSpline(z, sum_a[::-1])
            sum_e = InterpolatedUnivariateSpline(z, sum_e[::-1])
            Pp = InterpolatedUnivariateSpline(z, sol[::-1])

        # solve the forward propagation using backward info
        model_fwd, sim_fwd = propagate_amp(
            length,
            edf,
            p_fwd,
            Pp_fwd,
            sum_a_prev=sum_a,
            sum_e_prev=sum_e,
            Pp_prev=Pp,
            n_records=n_records,
            record_sim=True,
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

        print(error)
        loop_count += 1

    if propagate_back:
        return output(model=model_fwd, sim=sim_fwd), output(
            model=model_bck, sim=sim_bck
        )
    else:  # backward pumping
        sim_fwd.Pp += Pp(sim_fwd.z)
        return output(model=model_fwd, sim=sim_fwd), [z_record, p_v_ase]

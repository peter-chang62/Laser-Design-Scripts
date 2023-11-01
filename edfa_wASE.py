# %% ----- imports
from re_nlse_joint_5level_wASE import EDF
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def propagate_amp(
    pulse,
    edf,
    length,
    Pp,
    n_records=None,
    sum_a_prev=None,
    sum_e_prev=None,
    Pp_prev=None,
):
    edf: EDF
    model = edf.generate_model(
        pulse,
        Pp_fwd=Pp,
        sum_a_prev=sum_a_prev,
        sum_e_prev=sum_e_prev,
        Pp_prev=Pp_prev,
    )
    sim = model.simulate(
        length,
        n_records=n_records,
    )
    return model, sim


def amplify(p_fwd, p_bck, edf, length, Pp_fwd, Pp_bck, n_records=None):
    if p_bck is None:
        if Pp_bck == 0:
            model_fwd, sim_fwd = propagate_amp(
                p_fwd, edf, length, Pp_fwd, n_records=n_records
            )
            return model_fwd, sim_fwd, None, None
        else:
            p_bck = p_fwd.copy()
            p_bck.p_t[:] = 0
            bck_seeded = False
    else:
        bck_seeded = True

    edf: EDF
    done = False
    loop_count = 0
    sum_a_prev = lambda z: 0
    sum_e_prev = lambda z: 0
    Pp_prev = lambda z: 0
    threshold = 1e-3
    while not done:
        model_fwd = edf.generate_model(
            p_fwd,
            Pp_fwd=Pp_fwd,
            sum_a_prev=sum_a_prev,
            sum_e_prev=sum_e_prev,
            Pp_prev=Pp_prev,
        )
        sim_fwd = model_fwd.simulate(length, n_records=n_records)
        e_p_fwd = sim_fwd.pulse_out.e_p

        sum_a_prev = InterpolatedUnivariateSpline(
            model_fwd.z_record, model_fwd.sum_a_record[::-1]
        )
        sum_e_prev = InterpolatedUnivariateSpline(
            model_fwd.z_record, model_fwd.sum_e_record[::-1]
        )
        Pp_prev = InterpolatedUnivariateSpline(
            model_fwd.z_record, model_fwd.Pp_record[::-1]
        )

        model_bck = edf.generate_model(
            p_bck,
            Pp_fwd=Pp_bck,
            sum_a_prev=sum_a_prev,
            sum_e_prev=sum_e_prev,
            Pp_prev=Pp_prev,
        )
        if bck_seeded:
            sim_bck = model_bck.simulate(length, n_records=n_records)
            e_p_bck = sim_bck.pulse_out.e_p

            sum_a_prev = InterpolatedUnivariateSpline(
                model_bck.z_record, model_bck.sum_a_record[::-1]
            )
            sum_e_prev = InterpolatedUnivariateSpline(
                model_bck.z_record, model_bck.sum_e_record[::-1]
            )
            Pp_prev = InterpolatedUnivariateSpline(
                model_bck.z_record, model_bck.Pp_record[::-1]
            )

            # for return results
            Pp_total = sim_fwd.Pp + sim_bck.Pp[::-1]
            sim_fwd.Pp = Pp_total
            sim_bck.Pp = Pp_total
        else:
            rk45 = model_bck.mode.rk45
            t = [rk45.t]
            y = [rk45.y[0]]

            idx = 1
            step = sim_fwd.z[1] - sim_fwd.z[0]
            P_v_ase_record = np.zeros(sim_fwd.P_v_ase.shape)
            P_v_ase_record[0] = 0
            while rk45.t < length:
                model_bck.mode.z = rk45.t  # update z-dependent parameter
                rk45.step()
                t.append(rk45.t)
                y.append(rk45.y[0])

                if rk45.t > idx * step:
                    P_v_ase_record[idx] = rk45.y[1:]
                    idx += 1

            t = np.asarray(t)
            y = np.asarray(y)

            sum_a_prev = lambda z: 0
            sum_e_prev = lambda z: 0
            Pp_prev = InterpolatedUnivariateSpline(t, y[::-1])

            # no error to calculate
            e_p_bck = 1e-20  # avoid divide 0 errors

            # for return results
            sim_fwd.Pp += Pp_prev(sim_fwd.z)
            sim_fwd.P_v_ase += P_v_ase_record[::-1]
            sim_bck = None

        # book keeping
        if loop_count == 0:
            e_p_fwd_old = e_p_fwd
            e_p_bck_old = e_p_bck
            loop_count += 1
            continue

        error_fwd = abs(e_p_fwd - e_p_fwd_old) / e_p_fwd
        error_bck = abs(e_p_bck - e_p_bck_old) / e_p_bck
        e_p_fwd_old = e_p_fwd
        e_p_bck_old = e_p_bck
        if error_fwd < threshold and error_bck < threshold:
            done = True
        print(loop_count, error_fwd, error_bck)
        loop_count += 1

    return model_fwd, sim_fwd, model_bck, sim_bck
